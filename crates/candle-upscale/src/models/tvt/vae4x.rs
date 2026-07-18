//! TVT's VAE-D4 autoencoder (candle port).
//!
//! A diffusers `AutoencoderKL` with a 4× (rather than the usual 8×) spatial
//! compression: `block_out_channels [128, 256, 256]` gives three stages / two
//! down- (encode) and up- (decode) samples. Both halves are needed — the encoder
//! maps the ×4 bicubic-upscaled low-res image to the 4-channel latent the UNet
//! refines, and the decoder maps the refined latent back to the ×4 image. This
//! VAE variant drops the mid-block entirely (the reference `VAED4/vae.py`
//! comments it out), so neither half has a bottleneck attention.

use candle_core::{Module, Result, Tensor};
use candle_nn::ops::silu;
use candle_nn::{conv2d, group_norm, Conv2d, Conv2dConfig, GroupNorm, VarBuilder};

use crate::common::blocks::{conv1x1, conv3x3, ResnetBlock2D, Upsample2D};

const GROUPS: usize = 32;
const EPS: f64 = 1e-6;
/// Encoder-order channel widths; the decoder walks these reversed.
const BLOCK_OUT: [usize; 3] = [128, 256, 256];
/// Resnets per down block (`layers_per_block`); the decoder adds one extra.
const LAYERS_PER_BLOCK: usize = 2;

/// Strided 3×3 conv with the asymmetric `(0,1,0,1)` zero-pad diffusers uses for
/// VAE downsampling (`downsample_padding = 0`).
#[derive(Debug)]
struct Downsample2D {
    conv: Conv2d,
}

impl Downsample2D {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv: conv2d(
                channels,
                channels,
                3,
                Conv2dConfig {
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("conv"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.pad_with_zeros(3, 0, 1)?.pad_with_zeros(2, 0, 1)?;
        self.conv.forward(&x)
    }
}

#[derive(Debug)]
struct DownEncoderBlock {
    resnets: Vec<ResnetBlock2D>,
    downsampler: Option<Downsample2D>,
}

impl DownEncoderBlock {
    fn new(
        in_ch: usize,
        out_ch: usize,
        num_resnets: usize,
        add_downsample: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let rvb = vb.pp("resnets");
        let resnets = (0..num_resnets)
            .map(|i| {
                let ci = if i == 0 { in_ch } else { out_ch };
                ResnetBlock2D::new(ci, out_ch, GROUPS, EPS, rvb.pp(i))
            })
            .collect::<Result<Vec<_>>>()?;
        let downsampler = if add_downsample {
            Some(Downsample2D::new(out_ch, vb.pp("downsamplers").pp(0))?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            downsampler,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for resnet in &self.resnets {
            h = resnet.forward(&h)?;
        }
        match &self.downsampler {
            Some(d) => d.forward(&h),
            None => Ok(h),
        }
    }
}

/// VAE-D4 encoder (no mid-block), ending at `quant_conv`; the caller takes the
/// mean (first half of the 8 moment channels).
#[derive(Debug)]
pub(crate) struct Encoder {
    conv_in: Conv2d,
    down_blocks: Vec<DownEncoderBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
    quant_conv: Conv2d,
}

impl Encoder {
    pub(crate) fn new(vb: VarBuilder) -> Result<Self> {
        let evb = vb.pp("encoder");
        let dvb = evb.pp("down_blocks");
        let mut down_blocks = Vec::with_capacity(BLOCK_OUT.len());
        let mut prev = BLOCK_OUT[0];
        for (i, &out) in BLOCK_OUT.iter().enumerate() {
            down_blocks.push(DownEncoderBlock::new(
                prev,
                out,
                LAYERS_PER_BLOCK,
                i != BLOCK_OUT.len() - 1,
                dvb.pp(i),
            )?);
            prev = out;
        }
        let last = BLOCK_OUT[BLOCK_OUT.len() - 1];
        Ok(Self {
            conv_in: conv3x3(3, BLOCK_OUT[0], evb.pp("conv_in"))?,
            down_blocks,
            conv_norm_out: group_norm(GROUPS, last, EPS, evb.pp("conv_norm_out"))?,
            conv_out: conv3x3(last, 8, evb.pp("conv_out"))?,
            quant_conv: conv1x1(8, 8, vb.pp("quant_conv"))?,
        })
    }

    /// Image `[N, 3, H, W]` in `[-1, 1]` -> latent mean `[N, 4, H/4, W/4]`.
    pub(crate) fn encode_mean(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(x)?;
        for block in &self.down_blocks {
            h = block.forward(&h)?;
        }
        let h = silu(&self.conv_norm_out.forward(&h)?)?;
        let moments = self.quant_conv.forward(&self.conv_out.forward(&h)?)?;
        moments.narrow(1, 0, 4)
    }
}

#[derive(Debug)]
struct UpDecoderBlock {
    resnets: Vec<ResnetBlock2D>,
    upsampler: Option<Upsample2D>,
}

impl UpDecoderBlock {
    fn new(
        in_ch: usize,
        out_ch: usize,
        num_resnets: usize,
        add_upsample: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let rvb = vb.pp("resnets");
        let resnets = (0..num_resnets)
            .map(|i| {
                let ci = if i == 0 { in_ch } else { out_ch };
                ResnetBlock2D::new(ci, out_ch, GROUPS, EPS, rvb.pp(i))
            })
            .collect::<Result<Vec<_>>>()?;
        let upsampler = if add_upsample {
            Some(Upsample2D::new(out_ch, vb.pp("upsamplers").pp(0))?)
        } else {
            None
        };
        Ok(Self { resnets, upsampler })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for resnet in &self.resnets {
            h = resnet.forward(&h)?;
        }
        match &self.upsampler {
            Some(u) => u.forward(&h),
            None => Ok(h),
        }
    }
}

/// VAE-D4 decoder (no mid-block): latent `[N, 4, H, W]` -> image
/// `[N, 3, 4H, 4W]` in ~`[-1, 1]`, preceded by `post_quant_conv`.
#[derive(Debug)]
pub(crate) struct Decoder {
    post_quant_conv: Conv2d,
    conv_in: Conv2d,
    up_blocks: Vec<UpDecoderBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Decoder {
    pub(crate) fn new(vb: VarBuilder) -> Result<Self> {
        let reversed: Vec<usize> = BLOCK_OUT.iter().copied().rev().collect();
        let first = reversed[0];
        let num_resnets = LAYERS_PER_BLOCK + 1;

        let dvb = vb.pp("decoder");
        let uvb = dvb.pp("up_blocks");
        let mut up_blocks = Vec::with_capacity(reversed.len());
        let mut prev = first;
        for (i, &out) in reversed.iter().enumerate() {
            up_blocks.push(UpDecoderBlock::new(
                prev,
                out,
                num_resnets,
                i != reversed.len() - 1,
                uvb.pp(i),
            )?);
            prev = out;
        }
        Ok(Self {
            post_quant_conv: conv1x1(4, 4, vb.pp("post_quant_conv"))?,
            conv_in: conv3x3(4, first, dvb.pp("conv_in"))?,
            up_blocks,
            conv_norm_out: group_norm(GROUPS, BLOCK_OUT[0], EPS, dvb.pp("conv_norm_out"))?,
            conv_out: conv3x3(BLOCK_OUT[0], 3, dvb.pp("conv_out"))?,
        })
    }

    /// Decode a latent `[N, 4, H, W]` to the image `[N, 3, 4H, 4W]`.
    pub(crate) fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(&self.post_quant_conv.forward(z)?)?;
        for up in &self.up_blocks {
            h = up.forward(&h)?;
        }
        let h = silu(&self.conv_norm_out.forward(&h)?)?;
        self.conv_out.forward(&h)
    }
}
