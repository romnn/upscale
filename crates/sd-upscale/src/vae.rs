//! `AutoencoderKL` decoder for the SD x4 upscaler (4-channel latent -> 3-channel
//! image, 4x spatial upsample). Only the decode path is ported: at inference the
//! latents come from the diffusion loop, never from the VAE encoder.

use burn::module::Module;
use burn::nn::conv::Conv2d;
use burn::nn::GroupNorm;
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::blocks::{conv1x1, conv3x3, group_norm, AttnBlock, ResnetBlock2D, Upsample2D};

const GROUPS: usize = 32;
const EPS: f64 = 1e-6;

/// Geometry of the x4-upscaler VAE decoder.
#[derive(Clone, Debug)]
pub struct VaeConfig {
    pub latent_channels: usize,
    pub out_channels: usize,
    /// Encoder-order channels; the decoder walks these reversed.
    pub block_out_channels: Vec<usize>,
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

#[derive(Module, Debug)]
struct MidBlock<B: Backend> {
    resnets: Vec<ResnetBlock2D<B>>,
    attentions: Vec<AttnBlock<B>>,
}

impl<B: Backend> MidBlock<B> {
    fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            resnets: vec![
                ResnetBlock2D::new(channels, channels, GROUPS, EPS, device),
                ResnetBlock2D::new(channels, channels, GROUPS, EPS, device),
            ],
            attentions: vec![AttnBlock::new(channels, GROUPS, EPS, device)],
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = self.resnets[0].forward(x);
        let h = self.attentions[0].forward(h);
        self.resnets[1].forward(h)
    }
}

#[derive(Module, Debug)]
struct UpDecoderBlock<B: Backend> {
    resnets: Vec<ResnetBlock2D<B>>,
    upsamplers: Vec<Upsample2D<B>>,
}

impl<B: Backend> UpDecoderBlock<B> {
    fn new(
        in_ch: usize,
        out_ch: usize,
        num_resnets: usize,
        add_upsample: bool,
        device: &B::Device,
    ) -> Self {
        let mut resnets = Vec::with_capacity(num_resnets);
        for i in 0..num_resnets {
            let ci = if i == 0 { in_ch } else { out_ch };
            resnets.push(ResnetBlock2D::new(ci, out_ch, GROUPS, EPS, device));
        }
        let upsamplers = if add_upsample {
            vec![Upsample2D::new(out_ch, device)]
        } else {
            vec![]
        };
        Self {
            resnets,
            upsamplers,
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut h = x;
        for resnet in &self.resnets {
            h = resnet.forward(h);
        }
        for up in &self.upsamplers {
            h = up.forward(h);
        }
        h
    }
}

#[derive(Module, Debug)]
struct Decoder<B: Backend> {
    conv_in: Conv2d<B>,
    mid_block: MidBlock<B>,
    up_blocks: Vec<UpDecoderBlock<B>>,
    conv_norm_out: GroupNorm<B>,
    conv_out: Conv2d<B>,
}

impl<B: Backend> Decoder<B> {
    fn new(config: &VaeConfig, device: &B::Device) -> Self {
        let reversed: Vec<usize> = config.block_out_channels.iter().copied().rev().collect();
        let first = reversed[0];
        let last = *reversed.last().unwrap();
        let num_resnets = config.layers_per_block + 1;

        let mut up_blocks = Vec::with_capacity(reversed.len());
        let mut prev = first;
        for (i, &out_ch) in reversed.iter().enumerate() {
            let add_upsample = i != reversed.len() - 1;
            up_blocks.push(UpDecoderBlock::new(
                prev,
                out_ch,
                num_resnets,
                add_upsample,
                device,
            ));
            prev = out_ch;
        }

        Self {
            conv_in: conv3x3(config.latent_channels, first, device),
            mid_block: MidBlock::new(first, device),
            up_blocks,
            conv_norm_out: group_norm(GROUPS, last, EPS, device),
            conv_out: conv3x3(last, config.out_channels, device),
        }
    }

    fn forward_trace(&self, z: Tensor<B, 4>) -> VaeTrace<B> {
        let out_conv_in = self.conv_in.forward(z);
        let out_mid = self.mid_block.forward(out_conv_in.clone());

        let mut h = out_mid.clone();
        let mut out_up = Vec::with_capacity(self.up_blocks.len());
        for up in &self.up_blocks {
            h = up.forward(h);
            out_up.push(h.clone());
        }

        let out_norm = self.conv_norm_out.forward(h);
        let output = self.conv_out.forward(silu(out_norm.clone()));
        VaeTrace {
            out_conv_in,
            out_mid,
            out_up,
            out_norm,
            output,
        }
    }
}

/// Per-stage decoder activations, matching the hook points dumped by
/// `python/dump_vae_fixture.py` so a parity test can localise a divergence.
#[derive(Clone, Debug)]
pub struct VaeTrace<B: Backend> {
    pub out_conv_in: Tensor<B, 4>,
    pub out_mid: Tensor<B, 4>,
    pub out_up: Vec<Tensor<B, 4>>,
    pub out_norm: Tensor<B, 4>,
    pub output: Tensor<B, 4>,
}

/// `AutoencoderKL.decode`: `post_quant_conv` then the decoder stack.
#[derive(Module, Debug)]
pub struct VaeDecoder<B: Backend> {
    post_quant_conv: Conv2d<B>,
    decoder: Decoder<B>,
}

impl<B: Backend> VaeDecoder<B> {
    pub fn new(config: &VaeConfig, device: &B::Device) -> Self {
        Self {
            post_quant_conv: conv1x1(config.latent_channels, config.latent_channels, device),
            decoder: Decoder::new(config, device),
        }
    }

    /// Latent `[N, 4, H, W]` -> image `[N, 3, 4H, 4W]` (unclamped, ~`[-1, 1]`).
    pub fn forward(&self, z: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward_trace(z).output
    }

    /// Like [`forward`](Self::forward) but keeps every intermediate stage.
    pub fn forward_trace(&self, z: Tensor<B, 4>) -> VaeTrace<B> {
        let h = self.post_quant_conv.forward(z);
        self.decoder.forward_trace(h)
    }
}
