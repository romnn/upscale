//! UNet2DConditionModel for the SD x4 upscaler.
//!
//! Ported bottom-up, each block checked against
//! `tests/fixtures/unet_forward.safetensors` (see `tests/unet_parity.rs`).
//! Field names mirror the diffusers state dict so the pretrained safetensors
//! load via `PyTorchToBurnAdapter` (Linear transpose + norm rename).
//!
//! Verified config: `block_out_channels [256,512,512,1024]`, 2 layers/block,
//! 8 attention heads (head_dim = channels/8), `cross_attention_dim 1024`,
//! `use_linear_projection`, `only_cross_attention [T,T,T,F]`, `norm_eps 1e-5`.

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{
    Embedding, EmbeddingConfig, GroupNorm, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    PaddingConfig2d,
};
use burn::tensor::activation::{gelu, silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use crate::blocks::{conv3x3, group_norm, Upsample2D};

const NORM_EPS: f64 = 1e-5;
const GROUPS: usize = 32;
const HEADS: usize = 8;
const CROSS_DIM: usize = 1024;
const FREQ_DIM: usize = 256;
const TIME_DIM: usize = 1024;

/// Sinusoidal timestep embedding, matching diffusers `get_timestep_embedding`
/// with `flip_sin_to_cos=true`, `downscale_freq_shift=0` (so the output is
/// `[cos(t·freq), sin(t·freq)]`). Computed on the host since it depends only on
/// the scalar timestep.
fn timestep_sinusoid<B: Backend>(timestep: f32, dim: usize, device: &B::Device) -> Tensor<B, 2> {
    let half = dim / 2;
    let mut v = vec![0f32; dim];
    let log_max_period = 10_000f32.ln();
    for i in 0..half {
        let freq = (-log_max_period * i as f32 / half as f32).exp();
        let arg = timestep * freq;
        v[i] = arg.cos();
        v[half + i] = arg.sin();
    }
    Tensor::from_data(TensorData::new(v, [1, dim]), device)
}

/// `time_embedding`: Linear(freq_dim → time_dim), SiLU, Linear(time_dim → time_dim).
#[derive(Module, Debug)]
pub struct TimestepEmbedding<B: Backend> {
    linear_1: Linear<B>,
    linear_2: Linear<B>,
}

impl<B: Backend> TimestepEmbedding<B> {
    pub fn new(freq_dim: usize, time_dim: usize, device: &B::Device) -> Self {
        Self {
            linear_1: LinearConfig::new(freq_dim, time_dim).init(device),
            linear_2: LinearConfig::new(time_dim, time_dim).init(device),
        }
    }

    /// `timestep` (diffusion step) → `[1, time_dim]` embedding.
    pub fn forward(&self, timestep: f32, freq_dim: usize, device: &B::Device) -> Tensor<B, 2> {
        let t = timestep_sinusoid::<B>(timestep, freq_dim, device);
        self.linear_2.forward(silu(self.linear_1.forward(t)))
    }
}

/// diffusers `ResnetBlock2D` with timestep conditioning (the UNet variant).
#[derive(Module, Debug)]
pub struct ResnetBlockTemb<B: Backend> {
    norm1: GroupNorm<B>,
    conv1: Conv2d<B>,
    time_emb_proj: Linear<B>,
    norm2: GroupNorm<B>,
    conv2: Conv2d<B>,
    conv_shortcut: Option<Conv2d<B>>,
    out_ch: usize,
}

impl<B: Backend> ResnetBlockTemb<B> {
    pub fn new(in_ch: usize, out_ch: usize, time_dim: usize, device: &B::Device) -> Self {
        let conv_shortcut = (in_ch != out_ch).then(|| {
            Conv2dConfig::new([in_ch, out_ch], [1, 1])
                .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0))
                .init(device)
        });
        Self {
            norm1: group_norm(GROUPS, in_ch, NORM_EPS, device),
            conv1: conv3x3(in_ch, out_ch, device),
            time_emb_proj: LinearConfig::new(time_dim, out_ch).init(device),
            norm2: group_norm(GROUPS, out_ch, NORM_EPS, device),
            conv2: conv3x3(out_ch, out_ch, device),
            conv_shortcut,
            out_ch,
        }
    }

    /// `x`: `[N, in_ch, H, W]`, `temb`: `[N, time_dim]`.
    pub fn forward(&self, x: Tensor<B, 4>, temb: Tensor<B, 2>) -> Tensor<B, 4> {
        let h = self.conv1.forward(silu(self.norm1.forward(x.clone())));

        let n = temb.dims()[0];
        let t = self
            .time_emb_proj
            .forward(silu(temb))
            .reshape([n, self.out_ch, 1, 1]);
        let h = h + t;

        let h = self.conv2.forward(silu(self.norm2.forward(h)));
        let residual = match &self.conv_shortcut {
            Some(conv) => conv.forward(x),
            None => x,
        };
        residual + h
    }
}

/// Multi-head attention (self or cross), diffusers `Attention` with
/// `to_q/to_k/to_v` (no bias) and `to_out.0` (bias). `heads = 8`. `to_out` is a
/// `Vec` of one `Linear` so its record path is `to_out.0.*` (diffusers stores
/// the output projection in an `nn.Sequential`, index 0).
#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    to_q: Linear<B>,
    to_k: Linear<B>,
    to_v: Linear<B>,
    to_out: Vec<Linear<B>>,
    heads: usize,
}

impl<B: Backend> Attention<B> {
    pub fn new(query_dim: usize, context_dim: usize, device: &B::Device) -> Self {
        let inner = query_dim; // inner_dim = heads * head_dim = query_dim here
        Self {
            to_q: LinearConfig::new(query_dim, inner)
                .with_bias(false)
                .init(device),
            to_k: LinearConfig::new(context_dim, inner)
                .with_bias(false)
                .init(device),
            to_v: LinearConfig::new(context_dim, inner)
                .with_bias(false)
                .init(device),
            to_out: vec![LinearConfig::new(inner, query_dim).init(device)],
            heads: HEADS,
        }
    }

    /// `hidden`: `[B, N, query_dim]`, `context`: `[B, M, context_dim]`.
    pub fn forward(&self, hidden: Tensor<B, 3>, context: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, inner] = hidden.dims();
        let m = context.dims()[1];
        let head_dim = inner / self.heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let q = self.to_q.forward(hidden);
        let k = self.to_k.forward(context.clone());
        let v = self.to_v.forward(context);

        // [B, L, inner] -> [B, heads, L, head_dim]
        let split =
            |t: Tensor<B, 3>, len: usize| t.reshape([b, len, self.heads, head_dim]).swap_dims(1, 2);
        let q = split(q, n);
        let k = split(k, m);
        let v = split(v, m);

        let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(scale); // [B,heads,N,M]
        let probs = softmax(scores, 3);
        let out = probs.matmul(v); // [B,heads,N,head_dim]

        let out = out.swap_dims(1, 2).reshape([b, n, inner]);
        self.to_out[0].forward(out)
    }
}

/// GEGLU feed-forward. diffusers keys are `ff.net.0.proj` (dim → 2·inner) and
/// `ff.net.2` (inner → dim); the loader remaps those to `proj_in`/`proj_out`.
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    proj_in: Linear<B>,
    proj_out: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(dim: usize, mult: usize, device: &B::Device) -> Self {
        let inner = dim * mult;
        Self {
            proj_in: LinearConfig::new(dim, inner * 2).init(device),
            proj_out: LinearConfig::new(inner, dim).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.proj_in.forward(x); // [B,N,2*inner]
        let inner = h.dims()[2] / 2;
        let a = h.clone().narrow(2, 0, inner);
        let gate = h.narrow(2, inner, inner);
        let h = a * gelu(gate);
        self.proj_out.forward(h)
    }
}

/// diffusers `BasicTransformerBlock`: (self-or-cross attn1) + (cross attn2) + GEGLU FF,
/// each with a pre-LayerNorm and residual.
#[derive(Module, Debug)]
pub struct BasicTransformerBlock<B: Backend> {
    norm1: LayerNorm<B>,
    attn1: Attention<B>,
    norm2: LayerNorm<B>,
    attn2: Attention<B>,
    norm3: LayerNorm<B>,
    ff: FeedForward<B>,
    only_cross_attention: bool,
}

impl<B: Backend> BasicTransformerBlock<B> {
    pub fn new(dim: usize, only_cross_attention: bool, device: &B::Device) -> Self {
        let attn1_context = if only_cross_attention { CROSS_DIM } else { dim };
        Self {
            norm1: LayerNormConfig::new(dim).init(device),
            attn1: Attention::new(dim, attn1_context, device),
            norm2: LayerNormConfig::new(dim).init(device),
            attn2: Attention::new(dim, CROSS_DIM, device),
            norm3: LayerNormConfig::new(dim).init(device),
            ff: FeedForward::new(dim, 4, device),
            only_cross_attention,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, context: Tensor<B, 3>) -> Tensor<B, 3> {
        let n1 = self.norm1.forward(x.clone());
        let ctx1 = if self.only_cross_attention {
            context.clone()
        } else {
            n1.clone()
        };
        let x = x + self.attn1.forward(n1, ctx1);

        let n2 = self.norm2.forward(x.clone());
        let x = x + self.attn2.forward(n2, context);

        let n3 = self.norm3.forward(x.clone());
        x + self.ff.forward(n3)
    }
}

/// diffusers `Transformer2DModel` with `use_linear_projection=true`: GroupNorm,
/// linear `proj_in`, transformer blocks, linear `proj_out`, residual.
#[derive(Module, Debug)]
pub struct Transformer2D<B: Backend> {
    norm: GroupNorm<B>,
    proj_in: Linear<B>,
    transformer_blocks: Vec<BasicTransformerBlock<B>>,
    proj_out: Linear<B>,
}

impl<B: Backend> Transformer2D<B> {
    pub fn new(
        channels: usize,
        num_blocks: usize,
        only_cross_attention: bool,
        device: &B::Device,
    ) -> Self {
        let transformer_blocks = (0..num_blocks)
            .map(|_| BasicTransformerBlock::new(channels, only_cross_attention, device))
            .collect();
        Self {
            // Transformer2D GroupNorm uses eps 1e-6 (not the resnet 1e-5).
            norm: group_norm(GROUPS, channels, 1e-6, device),
            proj_in: LinearConfig::new(channels, channels).init(device),
            transformer_blocks,
            proj_out: LinearConfig::new(channels, channels).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, c, h, w] = x.dims();
        let residual = x.clone();

        let hidden = self.norm.forward(x);
        // use_linear_projection: reshape to [B, HW, C] then linear proj_in.
        let hidden = hidden.reshape([b, c, h * w]).swap_dims(1, 2);
        let mut hidden = self.proj_in.forward(hidden);
        for block in &self.transformer_blocks {
            hidden = block.forward(hidden, context.clone());
        }
        let hidden = self.proj_out.forward(hidden);

        let hidden = hidden.swap_dims(1, 2).reshape([b, c, h, w]);
        hidden + residual
    }
}

/// Build the noise-level class-embedding table (`nn.Embedding(num, time_dim)`).
/// Kept as a bare `Embedding` (not a wrapper) so its record path is
/// `class_embedding.weight`, matching diffusers.
pub fn class_embedding<B: Backend>(
    num_classes: usize,
    time_dim: usize,
    device: &B::Device,
) -> Embedding<B> {
    EmbeddingConfig::new(num_classes, time_dim).init(device)
}

/// Look up one class label (noise level) → `[1, time_dim]`.
pub fn class_embed_lookup<B: Backend>(
    embedding: &Embedding<B>,
    class_label: i64,
    device: &B::Device,
) -> Tensor<B, 2> {
    let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![class_label], [1]), device)
        .reshape([1, 1]);
    let e = embedding.forward(idx);
    let [_, _, d] = e.dims();
    e.reshape([1, d])
}

/// Strided-conv downsample (diffusers `Downsample2D`, `downsample_padding=1`).
#[derive(Module, Debug)]
pub struct Downsample2D<B: Backend> {
    conv: Conv2d<B>,
}

impl<B: Backend> Downsample2D<B> {
    pub fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            conv: Conv2dConfig::new([channels, channels], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv.forward(x)
    }
}

/// A down stage: `layers_per_block` resnets, optional per-resnet transformers
/// (empty ⇒ plain `DownBlock2D`), optional downsampler. Returns the new hidden
/// state plus the residuals the up path will consume (one per resnet, plus one
/// after downsampling).
#[derive(Module, Debug)]
pub struct DownBlock<B: Backend> {
    resnets: Vec<ResnetBlockTemb<B>>,
    attentions: Vec<Transformer2D<B>>,
    downsamplers: Vec<Downsample2D<B>>,
}

impl<B: Backend> DownBlock<B> {
    fn forward(
        &self,
        mut h: Tensor<B, 4>,
        temb: Tensor<B, 2>,
        context: Tensor<B, 3>,
    ) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        let mut states = Vec::new();
        for (i, resnet) in self.resnets.iter().enumerate() {
            h = resnet.forward(h, temb.clone());
            if let Some(attn) = self.attentions.get(i) {
                h = attn.forward(h, context.clone());
            }
            states.push(h.clone());
        }
        for d in &self.downsamplers {
            h = d.forward(h);
        }
        if !self.downsamplers.is_empty() {
            states.push(h.clone());
        }
        (h, states)
    }
}

/// The mid stage (`UNetMidBlock2DCrossAttn`): resnet → transformer → resnet.
#[derive(Module, Debug)]
pub struct MidBlock<B: Backend> {
    resnets: Vec<ResnetBlockTemb<B>>,
    attentions: Vec<Transformer2D<B>>,
}

impl<B: Backend> MidBlock<B> {
    fn forward(&self, h: Tensor<B, 4>, temb: Tensor<B, 2>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let h = self.resnets[0].forward(h, temb.clone());
        let h = self.attentions[0].forward(h, context);
        self.resnets[1].forward(h, temb)
    }
}

/// An up stage: consumes one residual per resnet (concatenated on channels,
/// most-recent first), optional per-resnet transformers, optional upsampler.
#[derive(Module, Debug)]
pub struct UpBlock<B: Backend> {
    resnets: Vec<ResnetBlockTemb<B>>,
    attentions: Vec<Transformer2D<B>>,
    upsamplers: Vec<Upsample2D<B>>,
}

impl<B: Backend> UpBlock<B> {
    fn forward(
        &self,
        mut h: Tensor<B, 4>,
        mut skips: Vec<Tensor<B, 4>>,
        temb: Tensor<B, 2>,
        context: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        for (i, resnet) in self.resnets.iter().enumerate() {
            #[expect(
                clippy::expect_used,
                reason = "each up-block resnet consumes exactly one skip by construction"
            )]
            let skip = skips.pop().expect("skip connection underflow");
            h = Tensor::cat(vec![h, skip], 1);
            h = resnet.forward(h, temb.clone());
            if let Some(attn) = self.attentions.get(i) {
                h = attn.forward(h, context.clone());
            }
        }
        for u in &self.upsamplers {
            h = u.forward(h);
        }
        h
    }
}

fn resnets<B: Backend>(chans: &[(usize, usize)], device: &B::Device) -> Vec<ResnetBlockTemb<B>> {
    chans
        .iter()
        .map(|&(i, o)| ResnetBlockTemb::new(i, o, TIME_DIM, device))
        .collect()
}

fn transformers<B: Backend>(
    channels: usize,
    count: usize,
    only_cross: bool,
    device: &B::Device,
) -> Vec<Transformer2D<B>> {
    (0..count)
        .map(|_| Transformer2D::new(channels, 1, only_cross, device))
        .collect()
}

/// `UNet2DConditionModel` for the SD x4 upscaler (7-channel input, v-prediction).
#[derive(Module, Debug)]
pub struct Unet<B: Backend> {
    conv_in: Conv2d<B>,
    time_embedding: TimestepEmbedding<B>,
    class_embedding: Embedding<B>,
    down_blocks: Vec<DownBlock<B>>,
    mid_block: MidBlock<B>,
    up_blocks: Vec<UpBlock<B>>,
    conv_norm_out: GroupNorm<B>,
    conv_out: Conv2d<B>,
}

impl<B: Backend> Unet<B> {
    /// Build the x4-upscaler UNet with the verified channel plan.
    pub fn new(device: &B::Device) -> Self {
        let down_blocks = vec![
            // DownBlock2D: no attention, downsample.
            DownBlock {
                resnets: resnets(&[(256, 256), (256, 256)], device),
                attentions: vec![],
                downsamplers: vec![Downsample2D::new(256, device)],
            },
            // CrossAttnDownBlock2D ×3 (only_cross_attention T,T,F).
            DownBlock {
                resnets: resnets(&[(256, 512), (512, 512)], device),
                attentions: transformers(512, 2, true, device),
                downsamplers: vec![Downsample2D::new(512, device)],
            },
            DownBlock {
                resnets: resnets(&[(512, 512), (512, 512)], device),
                attentions: transformers(512, 2, true, device),
                downsamplers: vec![Downsample2D::new(512, device)],
            },
            DownBlock {
                resnets: resnets(&[(512, 1024), (1024, 1024)], device),
                attentions: transformers(1024, 2, false, device),
                downsamplers: vec![],
            },
        ];

        let mid_block = MidBlock {
            resnets: resnets(&[(1024, 1024), (1024, 1024)], device),
            attentions: transformers(1024, 1, false, device),
        };

        // Up path (only_cross_attention reversed → F,T,T,-).
        let up_blocks = vec![
            UpBlock {
                resnets: resnets(&[(2048, 1024), (2048, 1024), (1536, 1024)], device),
                attentions: transformers(1024, 3, false, device),
                upsamplers: vec![Upsample2D::new(1024, device)],
            },
            UpBlock {
                resnets: resnets(&[(1536, 512), (1024, 512), (1024, 512)], device),
                attentions: transformers(512, 3, true, device),
                upsamplers: vec![Upsample2D::new(512, device)],
            },
            UpBlock {
                resnets: resnets(&[(1024, 512), (1024, 512), (768, 512)], device),
                attentions: transformers(512, 3, true, device),
                upsamplers: vec![Upsample2D::new(512, device)],
            },
            // UpBlock2D: no attention, no upsampler.
            UpBlock {
                resnets: resnets(&[(768, 256), (512, 256), (512, 256)], device),
                attentions: vec![],
                upsamplers: vec![],
            },
        ];

        Self {
            conv_in: conv3x3(7, 256, device),
            time_embedding: TimestepEmbedding::new(FREQ_DIM, TIME_DIM, device),
            class_embedding: class_embedding(1000, TIME_DIM, device),
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out: group_norm(GROUPS, 256, NORM_EPS, device),
            conv_out: conv3x3(256, 4, device),
        }
    }

    /// `sample`: `[N, 7, H, W]` (latent ⊕ low-res). `timestep`: diffusion step.
    /// `context`: `[N, 77, 1024]` text embedding. `class_label`: noise level.
    /// Returns the predicted `v` (`[N, 4, H, W]`).
    pub fn forward(
        &self,
        sample: Tensor<B, 4>,
        timestep: f32,
        context: Tensor<B, 3>,
        class_label: i64,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let emb = self.time_embedding.forward(timestep, FREQ_DIM, device)
            + class_embed_lookup(&self.class_embedding, class_label, device);

        let mut h = self.conv_in.forward(sample);
        let mut res: Vec<Tensor<B, 4>> = vec![h.clone()];

        for db in &self.down_blocks {
            let (nh, states) = db.forward(h, emb.clone(), context.clone());
            h = nh;
            res.extend(states);
        }

        h = self.mid_block.forward(h, emb.clone(), context.clone());

        for ub in &self.up_blocks {
            let n = ub.resnets.len();
            let skips = res.split_off(res.len() - n);
            h = ub.forward(h, skips, emb.clone(), context.clone());
        }

        let h = silu(self.conv_norm_out.forward(h));
        self.conv_out.forward(h)
    }
}
