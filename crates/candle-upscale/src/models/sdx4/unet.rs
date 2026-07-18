//! UNet2DConditionModel for the SD x4 upscaler (candle port).
//!
//! Mirrors `sd-upscale/src/unet.rs`: 7-channel `conv_in`, a noise-level
//! class-embedding added to the timestep embedding,
//! `only_cross_attention = [T,T,T,F]` across the down blocks,
//! `block_out_channels [256,512,512,1024]`, 8 attention heads, v-prediction.
//! Field names and `VarBuilder` prefixes match the diffusers state dict.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::ops::silu;
use candle_nn::{
    conv2d, embedding, group_norm, layer_norm, linear, linear_no_bias, Conv2d, Conv2dConfig,
    Embedding, GroupNorm, LayerNorm, Linear, VarBuilder,
};

use crate::common::blocks::{conv3x3, Upsample2D};

const NORM_EPS: f64 = 1e-5;
const GROUPS: usize = 32;
const HEADS: usize = 8;
const CROSS_DIM: usize = 1024;
const FREQ_DIM: usize = 256;
const TIME_DIM: usize = 1024;

/// Sinusoidal timestep embedding, matching diffusers `get_timestep_embedding`
/// with `flip_sin_to_cos=true`, `downscale_freq_shift=0` (output is
/// `[cos(t·freq), sin(t·freq)]`). Computed on the host since it depends only on
/// the scalar timestep, then moved to `device` at the compute `dtype`.
fn timestep_sinusoid(timestep: f32, dim: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let half = dim / 2;
    let mut v = vec![0f32; dim];
    let log_max_period = 10_000f32.ln();
    for i in 0..half {
        let freq = (-log_max_period * i as f32 / half as f32).exp();
        let arg = timestep * freq;
        v[i] = arg.cos();
        v[half + i] = arg.sin();
    }
    Tensor::from_vec(v, (1, dim), device)?.to_dtype(dtype)
}

/// `time_embedding`: Linear(freq_dim → time_dim), SiLU, Linear(time_dim → time_dim).
#[derive(Debug)]
struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    fn new(freq_dim: usize, time_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_1: linear(freq_dim, time_dim, vb.pp("linear_1"))?,
            linear_2: linear(time_dim, time_dim, vb.pp("linear_2"))?,
        })
    }

    fn forward(&self, timestep: f32, device: &Device, dtype: DType) -> Result<Tensor> {
        let t = timestep_sinusoid(timestep, FREQ_DIM, device, dtype)?;
        self.linear_2.forward(&silu(&self.linear_1.forward(&t)?)?)
    }
}

/// Look up one class label (noise level) → `[1, time_dim]`.
fn class_embed_lookup(embedding: &Embedding, class_label: i64, device: &Device) -> Result<Tensor> {
    let idx = Tensor::from_vec(vec![class_label as u32], 1, device)?;
    embedding.forward(&idx)
}

/// diffusers `ResnetBlock2D` with timestep conditioning (the UNet variant).
#[derive(Debug)]
struct ResnetBlockTemb {
    norm1: GroupNorm,
    conv1: Conv2d,
    time_emb_proj: Linear,
    norm2: GroupNorm,
    conv2: Conv2d,
    conv_shortcut: Option<Conv2d>,
    out_ch: usize,
}

impl ResnetBlockTemb {
    fn new(in_ch: usize, out_ch: usize, time_dim: usize, vb: VarBuilder) -> Result<Self> {
        let conv_shortcut = if in_ch == out_ch {
            None
        } else {
            Some(conv2d(
                in_ch,
                out_ch,
                1,
                Conv2dConfig::default(),
                vb.pp("conv_shortcut"),
            )?)
        };
        Ok(Self {
            norm1: group_norm(GROUPS, in_ch, NORM_EPS, vb.pp("norm1"))?,
            conv1: conv3x3(in_ch, out_ch, vb.pp("conv1"))?,
            time_emb_proj: linear(time_dim, out_ch, vb.pp("time_emb_proj"))?,
            norm2: group_norm(GROUPS, out_ch, NORM_EPS, vb.pp("norm2"))?,
            conv2: conv3x3(out_ch, out_ch, vb.pp("conv2"))?,
            conv_shortcut,
            out_ch,
        })
    }

    fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(&silu(&self.norm1.forward(x)?)?)?;

        let t = self.time_emb_proj.forward(&silu(temb)?)?;
        let t = t.reshape((t.dim(0)?, self.out_ch, 1, 1))?;
        let h = h.broadcast_add(&t)?;

        let h = self.conv2.forward(&silu(&self.norm2.forward(&h)?)?)?;
        let residual = match &self.conv_shortcut {
            Some(conv) => conv.forward(x)?,
            None => x.clone(),
        };
        residual + h
    }
}

/// Multi-head attention (self or cross), diffusers `Attention` with
/// `to_q`/`to_k`/`to_v` (no bias) and `to_out.0` (bias). `heads = 8`.
#[derive(Debug)]
struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    heads: usize,
}

impl Attention {
    fn new(query_dim: usize, context_dim: usize, vb: VarBuilder) -> Result<Self> {
        let inner = query_dim; // inner_dim = heads * head_dim = query_dim here
        Ok(Self {
            to_q: linear_no_bias(query_dim, inner, vb.pp("to_q"))?,
            to_k: linear_no_bias(context_dim, inner, vb.pp("to_k"))?,
            to_v: linear_no_bias(context_dim, inner, vb.pp("to_v"))?,
            to_out: linear(inner, query_dim, vb.pp("to_out").pp("0"))?,
            heads: HEADS,
        })
    }

    fn forward(&self, hidden: &Tensor, context: &Tensor) -> Result<Tensor> {
        let (b, n, inner) = hidden.dims3()?;
        let m = context.dim(1)?;
        let head_dim = inner / self.heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let q = self.to_q.forward(hidden)?;
        let k = self.to_k.forward(context)?;
        let v = self.to_v.forward(context)?;

        // [B, L, inner] -> [B*heads, L, head_dim]
        let split = |t: &Tensor, len: usize| -> Result<Tensor> {
            t.reshape((b, len, self.heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b * self.heads, len, head_dim))
        };
        let q = split(&q, n)?;
        let k = split(&k, m)?;
        let v = split(&v, m)?;

        let scores = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * scale)?; // [B*heads, N, M]
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = probs.matmul(&v)?; // [B*heads, N, head_dim]

        let out = out
            .reshape((b, self.heads, n, head_dim))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, n, inner))?;
        self.to_out.forward(&out)
    }
}

/// GEGLU feed-forward. diffusers keys are `ff.net.0.proj` (dim → 2·inner) and
/// `ff.net.2` (inner → dim), read directly here (no rename).
#[derive(Debug)]
struct FeedForward {
    proj_in: Linear,
    proj_out: Linear,
}

impl FeedForward {
    fn new(dim: usize, mult: usize, vb: VarBuilder) -> Result<Self> {
        let inner = dim * mult;
        Ok(Self {
            proj_in: linear(dim, inner * 2, vb.pp("net").pp("0").pp("proj"))?,
            proj_out: linear(inner, dim, vb.pp("net").pp("2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.proj_in.forward(x)?; // [B, N, 2*inner]
        let inner = h.dim(D::Minus1)? / 2;
        let a = h.narrow(D::Minus1, 0, inner)?.contiguous()?;
        let gate = h.narrow(D::Minus1, inner, inner)?.contiguous()?;
        let h = (a * gate.gelu_erf()?)?;
        self.proj_out.forward(&h)
    }
}

/// diffusers `BasicTransformerBlock`: (self-or-cross attn1) + (cross attn2) +
/// GEGLU FF, each with a pre-LayerNorm and residual.
#[derive(Debug)]
struct BasicTransformerBlock {
    norm1: LayerNorm,
    attn1: Attention,
    norm2: LayerNorm,
    attn2: Attention,
    norm3: LayerNorm,
    ff: FeedForward,
    only_cross_attention: bool,
}

impl BasicTransformerBlock {
    fn new(dim: usize, only_cross_attention: bool, vb: VarBuilder) -> Result<Self> {
        let attn1_context = if only_cross_attention { CROSS_DIM } else { dim };
        Ok(Self {
            norm1: layer_norm(dim, 1e-5, vb.pp("norm1"))?,
            attn1: Attention::new(dim, attn1_context, vb.pp("attn1"))?,
            norm2: layer_norm(dim, 1e-5, vb.pp("norm2"))?,
            attn2: Attention::new(dim, CROSS_DIM, vb.pp("attn2"))?,
            norm3: layer_norm(dim, 1e-5, vb.pp("norm3"))?,
            ff: FeedForward::new(dim, 4, vb.pp("ff"))?,
            only_cross_attention,
        })
    }

    fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        let n1 = self.norm1.forward(x)?;
        let x = if self.only_cross_attention {
            (x + self.attn1.forward(&n1, context)?)?
        } else {
            (x + self.attn1.forward(&n1, &n1)?)?
        };

        let n2 = self.norm2.forward(&x)?;
        let x = (x + self.attn2.forward(&n2, context)?)?;

        let n3 = self.norm3.forward(&x)?;
        x + self.ff.forward(&n3)?
    }
}

/// diffusers `Transformer2DModel` with `use_linear_projection=true`: GroupNorm,
/// linear `proj_in`, transformer blocks, linear `proj_out`, residual.
#[derive(Debug)]
struct Transformer2D {
    norm: GroupNorm,
    proj_in: Linear,
    transformer_blocks: Vec<BasicTransformerBlock>,
    proj_out: Linear,
}

impl Transformer2D {
    fn new(
        channels: usize,
        num_blocks: usize,
        only_cross_attention: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut transformer_blocks = Vec::with_capacity(num_blocks);
        let tvb = vb.pp("transformer_blocks");
        for i in 0..num_blocks {
            transformer_blocks.push(BasicTransformerBlock::new(
                channels,
                only_cross_attention,
                tvb.pp(i),
            )?);
        }
        Ok(Self {
            // Transformer2D GroupNorm uses eps 1e-6 (not the resnet 1e-5).
            norm: group_norm(GROUPS, channels, 1e-6, vb.pp("norm"))?,
            proj_in: linear(channels, channels, vb.pp("proj_in"))?,
            transformer_blocks,
            proj_out: linear(channels, channels, vb.pp("proj_out"))?,
        })
    }

    fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let residual = x;

        let hidden = self.norm.forward(x)?;
        // use_linear_projection: reshape to [B, HW, C] then linear proj_in.
        let hidden = hidden
            .reshape((b, c, h * w))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut hidden = self.proj_in.forward(&hidden)?;
        for block in &self.transformer_blocks {
            hidden = block.forward(&hidden, context)?;
        }
        let hidden = self.proj_out.forward(&hidden)?;

        let hidden = hidden
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, c, h, w))?;
        hidden + residual
    }
}

/// Strided-conv downsample (diffusers `Downsample2D`, `downsample_padding=1`).
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
                    padding: 1,
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("conv"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

/// A down stage: `layers_per_block` resnets, optional per-resnet transformers
/// (empty ⇒ plain `DownBlock2D`), optional downsampler. Returns the new hidden
/// state plus the residuals the up path will consume.
#[derive(Debug)]
struct DownBlock {
    resnets: Vec<ResnetBlockTemb>,
    attentions: Vec<Transformer2D>,
    downsamplers: Vec<Downsample2D>,
}

impl DownBlock {
    fn forward(
        &self,
        mut h: Tensor,
        temb: &Tensor,
        context: &Tensor,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut states = Vec::new();
        for (i, resnet) in self.resnets.iter().enumerate() {
            h = resnet.forward(&h, temb)?;
            if let Some(attn) = self.attentions.get(i) {
                h = attn.forward(&h, context)?;
            }
            states.push(h.clone());
        }
        for d in &self.downsamplers {
            h = d.forward(&h)?;
        }
        if !self.downsamplers.is_empty() {
            states.push(h.clone());
        }
        Ok((h, states))
    }
}

/// The mid stage (`UNetMidBlock2DCrossAttn`): resnet → transformer → resnet.
#[derive(Debug)]
struct MidBlock {
    resnets: Vec<ResnetBlockTemb>,
    attentions: Vec<Transformer2D>,
}

impl MidBlock {
    fn forward(&self, h: &Tensor, temb: &Tensor, context: &Tensor) -> Result<Tensor> {
        let h = self.resnets[0].forward(h, temb)?;
        let h = self.attentions[0].forward(&h, context)?;
        self.resnets[1].forward(&h, temb)
    }
}

/// An up stage: consumes one residual per resnet (concatenated on channels,
/// most-recent first), optional per-resnet transformers, optional upsampler.
#[derive(Debug)]
struct UpBlock {
    resnets: Vec<ResnetBlockTemb>,
    attentions: Vec<Transformer2D>,
    upsamplers: Vec<Upsample2D>,
}

impl UpBlock {
    fn forward(
        &self,
        mut h: Tensor,
        mut skips: Vec<Tensor>,
        temb: &Tensor,
        context: &Tensor,
    ) -> Result<Tensor> {
        for (i, resnet) in self.resnets.iter().enumerate() {
            let skip = skips
                .pop()
                .ok_or_else(|| candle_core::Error::Msg("skip connection underflow".into()))?;
            h = Tensor::cat(&[&h, &skip], 1)?;
            h = resnet.forward(&h, temb)?;
            if let Some(attn) = self.attentions.get(i) {
                h = attn.forward(&h, context)?;
            }
        }
        for u in &self.upsamplers {
            h = u.forward(&h)?;
        }
        Ok(h)
    }
}

fn resnets(vb: &VarBuilder, chans: &[(usize, usize)]) -> Result<Vec<ResnetBlockTemb>> {
    let rvb = vb.pp("resnets");
    chans
        .iter()
        .enumerate()
        .map(|(i, &(ic, oc))| ResnetBlockTemb::new(ic, oc, TIME_DIM, rvb.pp(i)))
        .collect()
}

fn transformers(
    vb: &VarBuilder,
    channels: usize,
    count: usize,
    only_cross: bool,
) -> Result<Vec<Transformer2D>> {
    let avb = vb.pp("attentions");
    (0..count)
        .map(|i| Transformer2D::new(channels, 1, only_cross, avb.pp(i)))
        .collect()
}

/// `UNet2DConditionModel` for the SD x4 upscaler (7-channel input, v-prediction).
#[derive(Debug)]
pub(crate) struct Unet {
    conv_in: Conv2d,
    time_embedding: TimestepEmbedding,
    class_embedding: Embedding,
    down_blocks: Vec<DownBlock>,
    mid_block: MidBlock,
    up_blocks: Vec<UpBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Unet {
    /// Build the x4-upscaler UNet with the verified channel plan, loading every
    /// weight from `vb` (positioned at the diffusers UNet state-dict root).
    pub(crate) fn new(vb: VarBuilder) -> Result<Self> {
        let dvb = vb.pp("down_blocks");
        let down_blocks = vec![
            // DownBlock2D: no attention, downsample.
            DownBlock {
                resnets: resnets(&dvb.pp(0), &[(256, 256), (256, 256)])?,
                attentions: vec![],
                downsamplers: vec![Downsample2D::new(256, dvb.pp(0).pp("downsamplers").pp(0))?],
            },
            // CrossAttnDownBlock2D ×3 (only_cross_attention T,T,F).
            DownBlock {
                resnets: resnets(&dvb.pp(1), &[(256, 512), (512, 512)])?,
                attentions: transformers(&dvb.pp(1), 512, 2, true)?,
                downsamplers: vec![Downsample2D::new(512, dvb.pp(1).pp("downsamplers").pp(0))?],
            },
            DownBlock {
                resnets: resnets(&dvb.pp(2), &[(512, 512), (512, 512)])?,
                attentions: transformers(&dvb.pp(2), 512, 2, true)?,
                downsamplers: vec![Downsample2D::new(512, dvb.pp(2).pp("downsamplers").pp(0))?],
            },
            DownBlock {
                resnets: resnets(&dvb.pp(3), &[(512, 1024), (1024, 1024)])?,
                attentions: transformers(&dvb.pp(3), 1024, 2, false)?,
                downsamplers: vec![],
            },
        ];

        let mvb = vb.pp("mid_block");
        let mid_block = MidBlock {
            resnets: resnets(&mvb, &[(1024, 1024), (1024, 1024)])?,
            attentions: transformers(&mvb, 1024, 1, false)?,
        };

        // Up path (only_cross_attention reversed → F,T,T,-).
        let uvb = vb.pp("up_blocks");
        let up_blocks = vec![
            UpBlock {
                resnets: resnets(&uvb.pp(0), &[(2048, 1024), (2048, 1024), (1536, 1024)])?,
                attentions: transformers(&uvb.pp(0), 1024, 3, false)?,
                upsamplers: vec![Upsample2D::new(1024, uvb.pp(0).pp("upsamplers").pp(0))?],
            },
            UpBlock {
                resnets: resnets(&uvb.pp(1), &[(1536, 512), (1024, 512), (1024, 512)])?,
                attentions: transformers(&uvb.pp(1), 512, 3, true)?,
                upsamplers: vec![Upsample2D::new(512, uvb.pp(1).pp("upsamplers").pp(0))?],
            },
            UpBlock {
                resnets: resnets(&uvb.pp(2), &[(1024, 512), (1024, 512), (768, 512)])?,
                attentions: transformers(&uvb.pp(2), 512, 3, true)?,
                upsamplers: vec![Upsample2D::new(512, uvb.pp(2).pp("upsamplers").pp(0))?],
            },
            // UpBlock2D: no attention, no upsampler.
            UpBlock {
                resnets: resnets(&uvb.pp(3), &[(768, 256), (512, 256), (512, 256)])?,
                attentions: vec![],
                upsamplers: vec![],
            },
        ];

        Ok(Self {
            conv_in: conv3x3(7, 256, vb.pp("conv_in"))?,
            time_embedding: TimestepEmbedding::new(FREQ_DIM, TIME_DIM, vb.pp("time_embedding"))?,
            class_embedding: embedding(1000, TIME_DIM, vb.pp("class_embedding"))?,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out: group_norm(GROUPS, 256, NORM_EPS, vb.pp("conv_norm_out"))?,
            conv_out: conv3x3(256, 4, vb.pp("conv_out"))?,
        })
    }

    /// `sample`: `[N, 7, H, W]` (latent ⊕ low-res). `timestep`: diffusion step.
    /// `context`: `[N, 77, 1024]` text embedding. `class_label`: noise level.
    /// Returns the predicted `v` (`[N, 4, H, W]`).
    pub(crate) fn forward(
        &self,
        sample: &Tensor,
        timestep: f32,
        context: &Tensor,
        class_label: i64,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let time_emb = self.time_embedding.forward(timestep, device, dtype)?;
        let class_emb = class_embed_lookup(&self.class_embedding, class_label, device)?;
        let emb = (time_emb + class_emb)?;

        let mut h = self.conv_in.forward(sample)?;
        let mut res: Vec<Tensor> = vec![h.clone()];

        for db in &self.down_blocks {
            let (nh, states) = db.forward(h, &emb, context)?;
            h = nh;
            res.extend(states);
        }

        h = self.mid_block.forward(&h, &emb, context)?;

        for ub in &self.up_blocks {
            let n = ub.resnets.len();
            let skips = res.split_off(res.len() - n);
            h = ub.forward(h, skips, &emb, context)?;
        }

        let h = silu(&self.conv_norm_out.forward(&h)?)?;
        self.conv_out.forward(&h)
    }
}
