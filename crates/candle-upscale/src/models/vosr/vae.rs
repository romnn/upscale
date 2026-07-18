//! VOSR's two SD2.1 autoencoder halves (candle ports).
//!
//! [`Encoder`] is the standard SD2.1 `AutoencoderKL` encoder — it maps the ×4
//! upscaled conditioning image to the 4-channel latent the DiT denoises. Only
//! the mean of the diagonal Gaussian is used (inference is deterministic).
//!
//! [`LightDecoder`] is VOSR's distilled replacement for the SD2.1 decoder: the
//! same block topology at halved channel widths (`block_out_channels =
//! [128, 128, 256, 256]`, per the shipped checkpoint config), decoding the
//! denoised latent straight to the ×4 image without a `post_quant_conv`.

use candle_core::{Module, Result, Tensor};
use candle_nn::ops::silu;
use candle_nn::{conv2d, group_norm, Conv2d, Conv2dConfig, GroupNorm, VarBuilder};

use crate::common::blocks::{conv1x1, conv3x3, AttnBlock, ResnetBlock2D, Upsample2D};

const GROUPS: usize = 32;
const EPS: f64 = 1e-6;

/// The 2-resnet + 1-attention bottleneck shared by the encoder and decoder.
#[derive(Debug)]
struct MidBlock {
    resnets: [ResnetBlock2D; 2],
    attention: AttnBlock,
}

impl MidBlock {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let rvb = vb.pp("resnets");
        Ok(Self {
            resnets: [
                ResnetBlock2D::new(channels, channels, GROUPS, EPS, rvb.pp(0))?,
                ResnetBlock2D::new(channels, channels, GROUPS, EPS, rvb.pp(1))?,
            ],
            attention: AttnBlock::new(channels, GROUPS, EPS, vb.pp("attentions").pp(0))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.resnets[0].forward(x)?;
        let h = self.attention.forward(&h)?;
        self.resnets[1].forward(&h)
    }
}

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

/// SD2.1 `AutoencoderKL` encoder, ending at `quant_conv`; the caller takes the
/// mean (first half of the 8 moment channels).
#[derive(Debug)]
pub(crate) struct Encoder {
    conv_in: Conv2d,
    down_blocks: Vec<DownEncoderBlock>,
    mid_block: MidBlock,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
    quant_conv: Conv2d,
}

impl Encoder {
    /// SD2.1 encoder geometry: `block_out_channels = [128, 256, 512, 512]`,
    /// two resnets per down block.
    pub(crate) fn new(vb: VarBuilder) -> Result<Self> {
        let block_out = [128usize, 256, 512, 512];
        let evb = vb.pp("encoder");
        let dvb = evb.pp("down_blocks");
        let mut down_blocks = Vec::with_capacity(block_out.len());
        let mut prev = block_out[0];
        for (i, &out) in block_out.iter().enumerate() {
            down_blocks.push(DownEncoderBlock::new(
                prev,
                out,
                2,
                i != block_out.len() - 1,
                dvb.pp(i),
            )?);
            prev = out;
        }
        let last = block_out[block_out.len() - 1];
        Ok(Self {
            conv_in: conv3x3(3, block_out[0], evb.pp("conv_in"))?,
            down_blocks,
            mid_block: MidBlock::new(last, evb.pp("mid_block"))?,
            conv_norm_out: group_norm(GROUPS, last, EPS, evb.pp("conv_norm_out"))?,
            conv_out: conv3x3(last, 8, evb.pp("conv_out"))?,
            quant_conv: conv1x1(8, 8, vb.pp("quant_conv"))?,
        })
    }

    /// Image `[N, 3, H, W]` in `[-1, 1]` -> latent mean `[N, 4, H/8, W/8]`.
    pub(crate) fn encode_mean(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(x)?;
        for block in &self.down_blocks {
            h = block.forward(&h)?;
        }
        let h = self.mid_block.forward(&h)?;
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

/// VOSR's lightweight SD2.1 decoder: latent `[N, 4, H, W]` -> image
/// `[N, 3, 8H, 8W]` in ~`[-1, 1]`.
#[derive(Debug)]
pub(crate) struct LightDecoder {
    conv_in: Conv2d,
    mid_block: MidBlock,
    up_blocks: Vec<UpDecoderBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl LightDecoder {
    /// Build the decoder from `block_out_channels = [128, 128, 256, 256]` with
    /// three resnets per up block (`layers_per_block + 1`).
    pub(crate) fn new(vb: VarBuilder) -> Result<Self> {
        let block_out = [128usize, 128, 256, 256];
        let reversed: Vec<usize> = block_out.iter().copied().rev().collect();
        let first = reversed[0];
        let num_resnets = 3;

        let uvb = vb.pp("up_blocks");
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
            conv_in: conv3x3(4, first, vb.pp("conv_in"))?,
            mid_block: MidBlock::new(first, vb.pp("mid_block"))?,
            up_blocks,
            conv_norm_out: group_norm(GROUPS, block_out[0], EPS, vb.pp("conv_norm_out"))?,
            conv_out: conv3x3(block_out[0], 3, vb.pp("conv_out"))?,
        })
    }

    /// Decode a latent `[N, 4, H, W]` to the image `[N, 3, 8H, 8W]`.
    pub(crate) fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let mut h = self.mid_block.forward(&self.conv_in.forward(z)?)?;
        for up in &self.up_blocks {
            h = up.forward(&h)?;
        }
        let h = silu(&self.conv_norm_out.forward(&h)?)?;
        self.conv_out.forward(&h)
    }
}
