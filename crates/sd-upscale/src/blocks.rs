//! Building blocks shared across the VAE decoder (and, later, the UNet).
//!
//! Field names mirror the diffusers/PyTorch state-dict layout so the pretrained
//! safetensors load with only the norm `weight`/`bias` -> `gamma`/`beta` remap
//! that every burn norm layer needs.

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{GroupNorm, GroupNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::activation::{silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use burn::tensor::Tensor;

/// GroupNorm eps used throughout the SD VAE / UNet (diffusers default `1e-6`
/// for the VAE resnets and `1e-5` for the UNet — callers pass the right one).
pub fn group_norm<B: Backend>(
    num_groups: usize,
    num_channels: usize,
    eps: f64,
    device: &B::Device,
) -> GroupNorm<B> {
    GroupNormConfig::new(num_groups, num_channels)
        .with_epsilon(eps)
        .init(device)
}

/// Shape-preserving 3x3 conv (`padding=1`), the workhorse conv of the VAE/UNet.
pub fn conv3x3<B: Backend>(in_ch: usize, out_ch: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([in_ch, out_ch], [3, 3])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .init(device)
}

/// 1x1 conv (`padding=0`), a per-pixel channel projection used for shortcuts.
pub fn conv1x1<B: Backend>(in_ch: usize, out_ch: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([in_ch, out_ch], [1, 1])
        .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0))
        .init(device)
}

/// diffusers `ResnetBlock2D` without timestep conditioning (the VAE variant).
#[derive(Module, Debug)]
pub struct ResnetBlock2D<B: Backend> {
    norm1: GroupNorm<B>,
    conv1: Conv2d<B>,
    norm2: GroupNorm<B>,
    conv2: Conv2d<B>,
    conv_shortcut: Option<Conv2d<B>>,
}

impl<B: Backend> ResnetBlock2D<B> {
    /// Builds a block mapping `in_ch` -> `out_ch` channels, with `groups`/`eps`
    /// for both GroupNorms. A 1x1 shortcut conv is added only when the channel
    /// count changes; otherwise the residual is the input itself.
    pub fn new(in_ch: usize, out_ch: usize, groups: usize, eps: f64, device: &B::Device) -> Self {
        let conv_shortcut = (in_ch != out_ch).then(|| conv1x1(in_ch, out_ch, device));
        Self {
            norm1: group_norm(groups, in_ch, eps, device),
            conv1: conv3x3(in_ch, out_ch, device),
            norm2: group_norm(groups, out_ch, eps, device),
            conv2: conv3x3(out_ch, out_ch, device),
            conv_shortcut,
        }
    }

    /// Applies `norm -> SiLU -> conv` twice, then adds the (optionally
    /// projected) residual. Shape-preserving except for the channel remap:
    /// `[N, in_ch, H, W]` -> `[N, out_ch, H, W]`.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = self.conv1.forward(silu(self.norm1.forward(x.clone())));
        let h = self.conv2.forward(silu(self.norm2.forward(h)));
        let residual = match &self.conv_shortcut {
            Some(conv) => conv.forward(x),
            None => x,
        };
        residual + h
    }
}

/// Query-row budget per [`AttnBlock`] score block, summed across the batch.
///
/// The score matrix is materialized `[B, chunk, HW]` with
/// `chunk = ATTN_ROW_BUDGET / B`, so the block stays `~ATTN_ROW_BUDGET × HW`
/// regardless of batch. At a 128px tile `HW = 16384`, that is ~67 MB in bf16,
/// versus the ~1 GB a full `[B, HW, HW]` matrix would need — which the CUDA
/// backend's allocator chokes on.
const ATTN_ROW_BUDGET: usize = 2048;

/// Spatial self-attention used in the VAE mid-block (single head, old diffusers
/// `query`/`key`/`value`/`proj_attn` naming that this checkpoint stores on disk).
#[derive(Module, Debug)]
pub struct AttnBlock<B: Backend> {
    group_norm: GroupNorm<B>,
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    proj_attn: Linear<B>,
    channels: usize,
}

impl<B: Backend> AttnBlock<B> {
    /// Builds a single-head self-attention over `channels`, with `groups`/`eps`
    /// for the pre-norm. Query/key/value/output projections are all
    /// `channels -> channels`.
    pub fn new(channels: usize, groups: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            group_norm: group_norm(groups, channels, eps, device),
            query: LinearConfig::new(channels, channels).init(device),
            key: LinearConfig::new(channels, channels).init(device),
            value: LinearConfig::new(channels, channels).init(device),
            proj_attn: LinearConfig::new(channels, channels).init(device),
            channels,
        }
    }

    /// Flattens the spatial grid to a length-`H·W` sequence, applies scaled
    /// dot-product self-attention, and adds the input residual. Shape-preserving
    /// (`[N, C, H, W]` in and out).
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, c, h, w] = x.dims();
        let residual = x.clone();

        // [B, C, H, W] -> [B, HW, C]
        let hidden = self.group_norm.forward(x);
        let hidden = hidden.reshape([b, c, h * w]).swap_dims(1, 2);

        let q = self.query.forward(hidden.clone());
        let k = self.key.forward(hidden.clone());
        let v = self.value.forward(hidden);

        // Single head, scaled by 1/sqrt(C). Attend in query-row chunks so the
        // score matrix is `[B, chunk, HW]` rather than the full `[B, HW, HW]`
        // (see `ATTN_ROW_BUDGET`). Splitting the query rows is exact: each row's
        // softmax runs over all keys independently.
        let scale = 1.0 / (self.channels as f64).sqrt();
        let n = h * w;
        let chunk = (ATTN_ROW_BUDGET / b).max(1);
        let kt = k.swap_dims(1, 2); // [B, C, HW]
        let mut parts = Vec::with_capacity(n.div_ceil(chunk));
        let mut start = 0;
        while start < n {
            let len = chunk.min(n - start);
            let scores = q
                .clone()
                .narrow(1, start, len)
                .matmul(kt.clone())
                .mul_scalar(scale);
            parts.push(softmax(scores, 2).matmul(v.clone())); // [B, len, C]
            start += len;
        }
        let out = self.proj_attn.forward(Tensor::cat(parts, 1)); // [B, HW, C]

        // [B, HW, C] -> [B, C, H, W]
        let out = out.swap_dims(1, 2).reshape([b, c, h, w]);
        residual + out
    }
}

/// Nearest-neighbour 2x upsample followed by a 3x3 conv (diffusers `Upsample2D`).
#[derive(Module, Debug)]
pub struct Upsample2D<B: Backend> {
    conv: Conv2d<B>,
}

impl<B: Backend> Upsample2D<B> {
    /// Builds a `channels -> channels` upsampler (channel count is unchanged).
    pub fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            conv: conv3x3(channels, channels, device),
        }
    }

    /// Nearest-neighbour 2x upsample then a 3x3 conv:
    /// `[N, C, H, W]` -> `[N, C, 2H, 2W]`.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, _, h, w] = x.dims();
        let x = interpolate(
            x,
            [h * 2, w * 2],
            InterpolateOptions::new(InterpolateMode::Nearest),
        );
        self.conv.forward(x)
    }
}
