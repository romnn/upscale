//! Building blocks shared by the candle VAE decoder and UNet.
//!
//! Ported from the burn reference (`sd-upscale/src/blocks.rs`); the field names
//! and `VarBuilder` prefixes mirror the diffusers state dict so the pretrained
//! safetensors load directly. candle stores `Linear` weight as `[out, in]`
//! (PyTorch layout) and conv weight as `[out, in, kh, kw]`, so — unlike the burn
//! port — no transpose is needed, and GroupNorm/LayerNorm keep the PyTorch
//! `weight`/`bias` names.

#[cfg(any(feature = "sdx4", feature = "vosr"))]
use candle_core::D;
use candle_core::{Module, Result, Tensor};
use candle_nn::ops::silu;
#[cfg(any(feature = "sdx4", feature = "vosr"))]
use candle_nn::ops::softmax;
use candle_nn::{conv2d, group_norm, Conv2d, Conv2dConfig, GroupNorm, VarBuilder};
#[cfg(any(feature = "sdx4", feature = "vosr"))]
use candle_nn::{linear, Linear};

/// Shape-preserving 3x3 conv (`padding=1`), the workhorse conv of the VAE/UNet.
pub(crate) fn conv3x3(in_ch: usize, out_ch: usize, vb: VarBuilder) -> Result<Conv2d> {
    conv2d(
        in_ch,
        out_ch,
        3,
        Conv2dConfig {
            padding: 1,
            ..Default::default()
        },
        vb,
    )
}

/// 1x1 conv (`padding=0`), a per-pixel channel projection used for shortcuts.
pub(crate) fn conv1x1(in_ch: usize, out_ch: usize, vb: VarBuilder) -> Result<Conv2d> {
    conv2d(in_ch, out_ch, 1, Conv2dConfig::default(), vb)
}

/// diffusers `ResnetBlock2D` without timestep conditioning (the VAE variant).
#[derive(Debug)]
pub(crate) struct ResnetBlock2D {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    conv_shortcut: Option<Conv2d>,
}

impl ResnetBlock2D {
    pub(crate) fn new(
        in_ch: usize,
        out_ch: usize,
        groups: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv_shortcut = if in_ch == out_ch {
            None
        } else {
            Some(conv1x1(in_ch, out_ch, vb.pp("conv_shortcut"))?)
        };
        Ok(Self {
            norm1: group_norm(groups, in_ch, eps, vb.pp("norm1"))?,
            conv1: conv3x3(in_ch, out_ch, vb.pp("conv1"))?,
            norm2: group_norm(groups, out_ch, eps, vb.pp("norm2"))?,
            conv2: conv3x3(out_ch, out_ch, vb.pp("conv2"))?,
            conv_shortcut,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(&silu(&self.norm1.forward(x)?)?)?;
        let h = self.conv2.forward(&silu(&self.norm2.forward(&h)?)?)?;
        let residual = match &self.conv_shortcut {
            Some(conv) => conv.forward(x)?,
            None => x.clone(),
        };
        residual + h
    }
}

/// Query-row budget per [`AttnBlock`] score block, summed across the batch.
///
/// The score matrix is materialized `[B, chunk, HW]` with
/// `chunk = ATTN_ROW_BUDGET / B`, keeping it `~ATTN_ROW_BUDGET x HW` regardless
/// of batch, versus the ~1 GB a full `[B, HW, HW]` matrix would need at a 128px
/// tile. Splitting the query rows is exact: each row's softmax runs over all
/// keys independently.
#[cfg(any(feature = "sdx4", feature = "vosr"))]
const ATTN_ROW_BUDGET: usize = 2048;

/// Spatial self-attention used in the VAE mid-block (single head, old diffusers
/// `query`/`key`/`value`/`proj_attn` naming that the fp32 checkpoint stores; the
/// fp16 re-export uses `to_q`/`to_k`/`to_v`/`to_out.0`, both handled here).
#[derive(Debug)]
#[cfg(any(feature = "sdx4", feature = "vosr"))]
pub(crate) struct AttnBlock {
    group_norm: GroupNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    proj_attn: Linear,
    channels: usize,
}

#[cfg(any(feature = "sdx4", feature = "vosr"))]
impl AttnBlock {
    pub(crate) fn new(channels: usize, groups: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        // The fp32 checkpoint names the projections query/key/value/proj_attn;
        // the fp16 re-export renamed them to_q/to_k/to_v/to_out.0.
        let names = if vb.contains_tensor("query.weight") {
            ["query", "key", "value", "proj_attn"]
        } else {
            ["to_q", "to_k", "to_v", "to_out.0"]
        };
        Ok(Self {
            group_norm: group_norm(groups, channels, eps, vb.pp("group_norm"))?,
            query: linear(channels, channels, vb.pp(names[0]))?,
            key: linear(channels, channels, vb.pp(names[1]))?,
            value: linear(channels, channels, vb.pp(names[2]))?,
            proj_attn: linear(channels, channels, vb.pp(names[3]))?,
            channels,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let residual = x;

        // [B, C, H, W] -> [B, HW, C]
        let hidden = self.group_norm.forward(x)?;
        let hidden = hidden
            .reshape((b, c, h * w))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.query.forward(&hidden)?;
        let k = self.key.forward(&hidden)?;
        let v = self.value.forward(&hidden)?;

        // Single head, scaled by 1/sqrt(C). Attend in query-row chunks so the
        // score matrix is `[B, chunk, HW]` rather than the full `[B, HW, HW]`.
        let scale = 1.0 / (self.channels as f64).sqrt();
        let n = h * w;
        let chunk = (ATTN_ROW_BUDGET / b).max(1);
        let kt = k.transpose(1, 2)?.contiguous()?; // [B, C, HW]
        let mut parts = Vec::with_capacity(n.div_ceil(chunk));
        let mut start = 0;
        while start < n {
            let len = chunk.min(n - start);
            let qc = q.narrow(1, start, len)?.contiguous()?;
            let scores = (qc.matmul(&kt)? * scale)?; // [B, len, HW]
            let probs = softmax(&scores, D::Minus1)?;
            parts.push(probs.matmul(&v)?); // [B, len, C]
            start += len;
        }
        let out = Tensor::cat(&parts, 1)?; // [B, HW, C]
        let out = self.proj_attn.forward(&out)?;

        // [B, HW, C] -> [B, C, H, W]
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, c, h, w))?;
        residual + out
    }
}

/// Nearest-neighbour 2x upsample followed by a 3x3 conv (diffusers `Upsample2D`).
#[derive(Debug)]
pub(crate) struct Upsample2D {
    conv: Conv2d,
}

impl Upsample2D {
    pub(crate) fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv: conv3x3(channels, channels, vb.pp("conv"))?,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        let x = x.upsample_nearest2d(h * 2, w * 2)?;
        self.conv.forward(&x)
    }
}
