//! `AutoencoderKL` decoder for the SD x4 upscaler (candle port).
//!
//! Only the decode path is ported: at inference the latents come from the
//! diffusion loop, never from the VAE encoder. Mirrors `sd-upscale/src/vae.rs`.

use candle_core::{Module, Result, Tensor};
use candle_nn::ops::silu;
use candle_nn::{group_norm, Conv2d, GroupNorm, VarBuilder};

use crate::common::blocks::{conv1x1, conv3x3, AttnBlock, ResnetBlock2D, Upsample2D};

const GROUPS: usize = 32;
const EPS: f64 = 1e-6;

/// Geometry of the x4-upscaler VAE decoder.
#[derive(Clone, Debug)]
pub(crate) struct VaeConfig {
    /// Channels in the latent the decoder consumes (4 for SD).
    pub latent_channels: usize,
    /// Channels in the decoded image (3 for RGB).
    pub out_channels: usize,
    /// Encoder-order channels; the decoder walks these reversed.
    pub block_out_channels: Vec<usize>,
    /// Resnet layers per up-block; the decoder adds one extra per block.
    pub layers_per_block: usize,
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self {
            latent_channels: 4,
            out_channels: 3,
            block_out_channels: vec![128, 256, 512],
            layers_per_block: 2,
        }
    }
}

#[derive(Debug)]
struct MidBlock {
    resnets: Vec<ResnetBlock2D>,
    attentions: Vec<AttnBlock>,
}

impl MidBlock {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let rvb = vb.pp("resnets");
        Ok(Self {
            resnets: vec![
                ResnetBlock2D::new(channels, channels, GROUPS, EPS, rvb.pp(0))?,
                ResnetBlock2D::new(channels, channels, GROUPS, EPS, rvb.pp(1))?,
            ],
            attentions: vec![AttnBlock::new(
                channels,
                GROUPS,
                EPS,
                vb.pp("attentions").pp(0),
            )?],
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.resnets[0].forward(x)?;
        let h = self.attentions[0].forward(&h)?;
        self.resnets[1].forward(&h)
    }
}

#[derive(Debug)]
struct UpDecoderBlock {
    resnets: Vec<ResnetBlock2D>,
    upsamplers: Vec<Upsample2D>,
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
        let mut resnets = Vec::with_capacity(num_resnets);
        for i in 0..num_resnets {
            let ci = if i == 0 { in_ch } else { out_ch };
            resnets.push(ResnetBlock2D::new(ci, out_ch, GROUPS, EPS, rvb.pp(i))?);
        }
        let upsamplers = if add_upsample {
            vec![Upsample2D::new(out_ch, vb.pp("upsamplers").pp(0))?]
        } else {
            vec![]
        };
        Ok(Self {
            resnets,
            upsamplers,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for resnet in &self.resnets {
            h = resnet.forward(&h)?;
        }
        for up in &self.upsamplers {
            h = up.forward(&h)?;
        }
        Ok(h)
    }
}

#[derive(Debug)]
struct Decoder {
    conv_in: Conv2d,
    mid_block: MidBlock,
    up_blocks: Vec<UpDecoderBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Decoder {
    fn new(config: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let reversed: Vec<usize> = config.block_out_channels.iter().copied().rev().collect();
        let first = reversed[0];
        let last = reversed[reversed.len() - 1];
        let num_resnets = config.layers_per_block + 1;

        let uvb = vb.pp("up_blocks");
        let mut up_blocks = Vec::with_capacity(reversed.len());
        let mut prev = first;
        for (i, &out_ch) in reversed.iter().enumerate() {
            let add_upsample = i != reversed.len() - 1;
            up_blocks.push(UpDecoderBlock::new(
                prev,
                out_ch,
                num_resnets,
                add_upsample,
                uvb.pp(i),
            )?);
            prev = out_ch;
        }

        Ok(Self {
            conv_in: conv3x3(config.latent_channels, first, vb.pp("conv_in"))?,
            mid_block: MidBlock::new(first, vb.pp("mid_block"))?,
            up_blocks,
            conv_norm_out: group_norm(GROUPS, last, EPS, vb.pp("conv_norm_out"))?,
            conv_out: conv3x3(last, config.out_channels, vb.pp("conv_out"))?,
        })
    }

    fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let h = self.conv_in.forward(z)?;
        let mut h = self.mid_block.forward(&h)?;
        for up in &self.up_blocks {
            h = up.forward(&h)?;
        }
        let h = silu(&self.conv_norm_out.forward(&h)?)?;
        self.conv_out.forward(&h)
    }
}

/// `AutoencoderKL.decode`: `post_quant_conv` then the decoder stack.
#[derive(Debug)]
pub(crate) struct VaeDecoder {
    post_quant_conv: Conv2d,
    decoder: Decoder,
}

impl VaeDecoder {
    /// Builds the decoder from `config`, loading every weight from `vb`
    /// (positioned at the diffusers VAE state-dict root).
    pub(crate) fn new(config: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            post_quant_conv: conv1x1(
                config.latent_channels,
                config.latent_channels,
                vb.pp("post_quant_conv"),
            )?,
            decoder: Decoder::new(config, vb.pp("decoder"))?,
        })
    }

    /// Latent `[N, 4, H, W]` -> image `[N, 3, 4H, 4W]` (unclamped, ~`[-1, 1]`).
    pub(crate) fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let h = self.post_quant_conv.forward(z)?;
        self.decoder.forward(&h)
    }
}
