//! LightningDiT: the PixArt-style adaLN-single diffusion transformer that VOSR
//! runs in latent space (candle port of `models/lightningdit.py`).
//!
//! One forward maps the 8-channel patchified input (noisy latent concatenated
//! with the low-res latent) plus a flow-time and a DINOv2 context to a
//! 4-channel velocity latent. The 28 blocks are self-attn (2D-RoPE, QK-RMSNorm)
//! → cross-attn onto the vision context → SwiGLU MLP, each gated by a shared
//! adaLN-single modulation derived from the timestep.

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::ops::{silu, softmax};
use candle_nn::{conv2d, layer_norm, linear, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};

const DIM: usize = 1024;
const DEPTH: usize = 28;
const HEADS: usize = 16;
const HEAD_DIM: usize = 64;
const PATCH: usize = 2;
const IN_CH: usize = 8;
const OUT_CH: usize = 4;
const FREQ_EMB: usize = 256;
const SWIGLU_HIDDEN: usize = 2730;
const ENC_DIM: usize = 768;
const MLP_CA_HIDDEN: usize = 3072;
const RMS_EPS: f64 = 1e-6;
/// Patch-grid side the RoPE frequencies were trained at (`input_size / patch =
/// 512/8 / 2`). At inference the tile grid can differ, so RoPE is rebuilt per
/// grid size but the frequency spacing stays anchored to this length.
const TRAIN_GRID: f64 = 32.0;

/// RMS normalization over the last dim with a learned per-channel weight,
/// computed in f32 for stability then cast back (matching the reference).
#[derive(Debug)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps: RMS_EPS,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x32 = x.to_dtype(DType::F32)?;
        let ms = x32.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x32.broadcast_div(&(ms + self.eps)?.sqrt()?)?;
        normed.to_dtype(dtype)?.broadcast_mul(&self.weight)
    }
}

/// `x * (1 + scale) + shift` with `scale`/`shift` broadcast over the token axis
/// (adaLN-single modulation; the tensors arrive already shaped `[B, 1, C]`).
fn modulate(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

/// Interleaved rotate-half: pairs `(x0, x1)` become `(-x1, x0)`.
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let mut shape = x.dims().to_vec();
    let last = shape
        .pop()
        .ok_or_else(|| candle_core::Error::Msg("rotate_half on scalar".into()))?;
    let mut pair = shape.clone();
    pair.push(last / 2);
    pair.push(2);
    let x = x.reshape(pair)?;
    let x0 = x.narrow(D::Minus1, 0, 1)?;
    let x1 = x.narrow(D::Minus1, 1, 1)?;
    let rot = Tensor::cat(&[&x1.neg()?, &x0], D::Minus1)?;
    let mut back = shape;
    back.push(last);
    rot.reshape(back)
}

/// Apply 2D RoPE to `t` `[B, heads, N, head_dim]` given `cos`/`sin` `[N, head_dim]`.
fn apply_rope(t: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (n, hd) = cos.dims2()?;
    let cos = cos.reshape((1, 1, n, hd))?;
    let sin = sin.reshape((1, 1, n, hd))?;
    t.broadcast_mul(&cos)? + rotate_half(t)?.broadcast_mul(&sin)?
}

/// Precompute the RoPE `cos`/`sin` tables `[G*G, head_dim]` for a `g×g` patch
/// grid. Mirrors `VisionRotaryEmbeddingFast` / `_get_dynamic_rope`: half the
/// head dim encodes the row position, half the column, each frequency repeated
/// twice so consecutive dims share a rotation angle.
fn rope_tables(g: usize, device: &Device, dtype: DType) -> Result<(Tensor, Tensor)> {
    let n_freq = HEAD_DIM / 4; // 16 frequencies, each covering two of the 64 dims
    let freqs: Vec<f64> = (0..n_freq)
        .map(|j| 10000f64.powf(-((2 * j) as f64) / (HEAD_DIM / 2) as f64))
        .collect();
    let mut cos = vec![0f32; g * g * HEAD_DIM];
    let mut sin = vec![0f32; g * g * HEAD_DIM];
    for ph in 0..g {
        for pw in 0..g {
            let base = (ph * g + pw) * HEAD_DIM;
            let th = ph as f64 / g as f64 * TRAIN_GRID;
            let tw = pw as f64 / g as f64 * TRAIN_GRID;
            for (j, &f) in freqs.iter().enumerate() {
                let (ah, aw) = (th * f, tw * f);
                for (off, ang) in [(2 * j, ah), (HEAD_DIM / 2 + 2 * j, aw)] {
                    cos[base + off] = ang.cos() as f32;
                    cos[base + off + 1] = ang.cos() as f32;
                    sin[base + off] = ang.sin() as f32;
                    sin[base + off + 1] = ang.sin() as f32;
                }
            }
        }
    }
    let cos = Tensor::from_vec(cos, (g * g, HEAD_DIM), device)?.to_dtype(dtype)?;
    let sin = Tensor::from_vec(sin, (g * g, HEAD_DIM), device)?.to_dtype(dtype)?;
    Ok((cos, sin))
}

/// Scaled-dot-product attention with an explicit `1/sqrt(head_dim)` scale.
/// `q,k,v` are `[B, heads, Nq/Nk, head_dim]`; returns `[B, heads, Nq, head_dim]`.
fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    let scores = (q.contiguous()?.matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)? * scale)?;
    let probs = softmax(&scores, D::Minus1)?;
    probs.matmul(&v.contiguous()?)
}

/// Fused-QKV self-attention with QK-RMSNorm and 2D RoPE.
#[derive(Debug)]
struct SelfAttn {
    qkv: Linear,
    proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
}

impl SelfAttn {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            qkv: linear(DIM, 3 * DIM, vb.pp("qkv"))?,
            proj: linear(DIM, DIM, vb.pp("proj"))?,
            q_norm: RmsNorm::new(HEAD_DIM, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(HEAD_DIM, vb.pp("k_norm"))?,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, n, c) = x.dims3()?;
        let qkv = self
            .qkv
            .forward(x)?
            .reshape((b, n, 3, HEADS, HEAD_DIM))?
            .permute((2, 0, 3, 1, 4))?;
        let q = self.q_norm.forward(&qkv.i(0)?.contiguous()?)?;
        let k = self.k_norm.forward(&qkv.i(1)?.contiguous()?)?;
        let v = qkv.i(2)?.contiguous()?;
        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;
        let out = sdpa(&q, &k, &v)?
            .transpose(1, 2)?
            .reshape((b, n, c))?;
        self.proj.forward(&out)
    }
}

/// Cross-attention from the latent tokens onto the DINOv2 vision context.
#[derive(Debug)]
struct CrossAttn {
    q_linear: Linear,
    k_linear: Linear,
    v_linear: Linear,
    proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
}

impl CrossAttn {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q_linear: linear(DIM, DIM, vb.pp("q_linear"))?,
            k_linear: linear(DIM, DIM, vb.pp("k_linear"))?,
            v_linear: linear(DIM, DIM, vb.pp("v_linear"))?,
            proj: linear(DIM, DIM, vb.pp("proj"))?,
            q_norm: RmsNorm::new(HEAD_DIM, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(HEAD_DIM, vb.pp("k_norm"))?,
        })
    }

    fn heads(x: &Tensor) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        x.reshape((b, n, HEADS, HEAD_DIM))?.transpose(1, 2)
    }

    fn forward(&self, x: &Tensor, ctx: &Tensor) -> Result<Tensor> {
        let (b, n, c) = x.dims3()?;
        let q = self.q_norm.forward(&Self::heads(&self.q_linear.forward(x)?)?)?;
        let k = self.k_norm.forward(&Self::heads(&self.k_linear.forward(ctx)?)?)?;
        let v = Self::heads(&self.v_linear.forward(ctx)?)?;
        let out = sdpa(&q, &k, &v)?.transpose(1, 2)?.reshape((b, n, c))?;
        self.proj.forward(&out)
    }
}

/// SwiGLU feed-forward: `w3(silu(a) * b)` with `[a, b] = w12(x)`.
#[derive(Debug)]
struct SwiGlu {
    w12: Linear,
    w3: Linear,
}

impl SwiGlu {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w12: linear(DIM, 2 * SWIGLU_HIDDEN, vb.pp("w12"))?,
            w3: linear(SWIGLU_HIDDEN, DIM, vb.pp("w3"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x12 = self.w12.forward(x)?;
        let a = x12.narrow(D::Minus1, 0, SWIGLU_HIDDEN)?;
        let b = x12.narrow(D::Minus1, SWIGLU_HIDDEN, SWIGLU_HIDDEN)?;
        self.w3.forward(&(silu(&a)? * b)?)
    }
}

/// One LightningDiT block: gated self-attn, then (ungated) cross-attn, then
/// gated SwiGLU, with adaLN-single shift/scale/gate from the timestep vector.
#[derive(Debug)]
struct Block {
    norm1: RmsNorm,
    attn: SelfAttn,
    cross_attn: CrossAttn,
    norm2: RmsNorm,
    mlp: SwiGlu,
    scale_shift_table: Tensor,
}

impl Block {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: RmsNorm::new(DIM, vb.pp("norm1"))?,
            attn: SelfAttn::new(vb.pp("attn"))?,
            cross_attn: CrossAttn::new(vb.pp("cross_attn"))?,
            norm2: RmsNorm::new(DIM, vb.pp("norm2"))?,
            mlp: SwiGlu::new(vb.pp("mlp"))?,
            scale_shift_table: vb.get((6, DIM), "scale_shift_table")?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        c0: &Tensor,
        ctx: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let b = x.dim(0)?;
        let mods = self
            .scale_shift_table
            .unsqueeze(0)?
            .broadcast_add(&c0.reshape((b, 6, DIM))?)?;
        let part = |i: usize| mods.narrow(1, i, 1);
        let (shift_msa, scale_msa, gate_msa) = (part(0)?, part(1)?, part(2)?);
        let (shift_mlp, scale_mlp, gate_mlp) = (part(3)?, part(4)?, part(5)?);

        let h = modulate(&self.norm1.forward(x)?, &scale_msa, &shift_msa)?;
        let x = (x + self.attn.forward(&h, cos, sin)?.broadcast_mul(&gate_msa)?)?;
        let x = (&x + self.cross_attn.forward(&x, ctx)?)?;
        let h = modulate(&self.norm2.forward(&x)?, &scale_mlp, &shift_mlp)?;
        &x + self.mlp.forward(&h)?.broadcast_mul(&gate_mlp)?
    }
}

/// Sinusoidal timestep embedding (cos then sin) followed by a 2-layer SiLU MLP.
///
/// The sinusoid is built in f32 (as the reference does) and cast to the compute
/// dtype only for the MLP, so a bf16 run keeps full precision in the embedding.
#[derive(Debug)]
struct TimestepEmbedder {
    fc1: Linear,
    fc2: Linear,
    freqs: Tensor,
    dtype: DType,
}

impl TimestepEmbedder {
    fn new(vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let half = FREQ_EMB / 2;
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-(10000f64.ln()) * i as f64 / half as f64).exp() as f32)
            .collect();
        Ok(Self {
            fc1: linear(FREQ_EMB, DIM, vb.pp(0))?,
            fc2: linear(DIM, DIM, vb.pp(2))?,
            freqs: Tensor::from_vec(freqs, half, device)?,
            dtype,
        })
    }

    /// `t` is a flow time `[B]` in f32.
    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let args = t.unsqueeze(1)?.broadcast_mul(&self.freqs.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(self.dtype)?;
        self.fc2.forward(&silu(&self.fc1.forward(&emb)?)?)
    }
}

/// Final adaLN layer: modulate, project each token to `patch²·out_channels`.
#[derive(Debug)]
struct FinalLayer {
    norm_final: RmsNorm,
    linear: Linear,
    ada: Linear,
}

impl FinalLayer {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm_final: RmsNorm::new(DIM, vb.pp("norm_final"))?,
            linear: linear(DIM, PATCH * PATCH * OUT_CH, vb.pp("linear"))?,
            ada: linear(DIM, 2 * DIM, vb.pp("adaLN_modulation").pp(1))?,
        })
    }

    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let sc = self.ada.forward(&silu(c)?)?;
        let shift = sc.narrow(D::Minus1, 0, DIM)?.unsqueeze(1)?;
        let scale = sc.narrow(D::Minus1, DIM, DIM)?.unsqueeze(1)?;
        let x = modulate(&self.norm_final.forward(x)?, &scale, &shift)?;
        self.linear.forward(&x)
    }
}

/// The full LightningDiT, plus the DINOv2-context projection (`layer_norm` +
/// `mlp_ca`) that turns raw layer-8 features into the cross-attention context.
#[derive(Debug)]
pub(crate) struct Dit {
    x_proj: Conv2d,
    t_embedder: TimestepEmbedder,
    t_block: Linear,
    blocks: Vec<Block>,
    final_layer: FinalLayer,
    ctx_norm: LayerNorm,
    ctx_fc1: Linear,
    ctx_fc2: Linear,
    device: Device,
    dtype: DType,
}

impl Dit {
    /// Build the DiT from a VarBuilder positioned at the checkpoint root.
    pub(crate) fn new(vb: VarBuilder, device: Device, dtype: DType) -> Result<Self> {
        let x_proj = conv2d(
            IN_CH,
            DIM,
            PATCH,
            Conv2dConfig {
                stride: PATCH,
                ..Default::default()
            },
            vb.pp("x_embedder").pp("proj"),
        )?;
        let bvb = vb.pp("blocks");
        let blocks = (0..DEPTH)
            .map(|i| Block::new(bvb.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            x_proj,
            t_embedder: TimestepEmbedder::new(vb.pp("t_embedder").pp("mlp"), &device, dtype)?,
            t_block: linear(DIM, 6 * DIM, vb.pp("t_block").pp(1))?,
            blocks,
            final_layer: FinalLayer::new(vb.pp("final_layer"))?,
            ctx_norm: layer_norm(ENC_DIM, 1e-5, vb.pp("layer_norm"))?,
            ctx_fc1: linear(ENC_DIM, MLP_CA_HIDDEN, vb.pp("mlp_ca").pp("fc1"))?,
            ctx_fc2: linear(MLP_CA_HIDDEN, DIM, vb.pp("mlp_ca").pp("fc2"))?,
            device,
            dtype,
        })
    }

    /// Project raw DINOv2 layer-8 features `[B, N, 768]` into the cross-attention
    /// context `[B, N, 1024]` (LayerNorm → fc1 → tanh-GELU → fc2).
    fn context(&self, feats: &Tensor) -> Result<Tensor> {
        let z = self.ctx_norm.forward(feats)?;
        let z = self.ctx_fc1.forward(&z)?.gelu()?;
        self.ctx_fc2.forward(&z)
    }

    /// Predict the velocity latent for `x` `[B, 8, H, W]` at flow time `t` `[B]`
    /// conditioned on DINOv2 features `ctx_feats` `[B, N, 768]`.
    pub(crate) fn forward(&self, x: &Tensor, t: &Tensor, ctx_feats: &Tensor) -> Result<Tensor> {
        let tokens = self.x_proj.forward(x)?;
        let (b, dim, gh, gw) = tokens.dims4()?;
        let mut h = tokens.reshape((b, dim, gh * gw))?.transpose(1, 2)?.contiguous()?;

        let c = self.t_embedder.forward(t)?;
        let c0 = self.t_block.forward(&silu(&c)?)?;
        let ctx = self.context(ctx_feats)?;
        let (cos, sin) = rope_tables(gh, &self.device, self.dtype)?;

        for block in &self.blocks {
            h = block.forward(&h, &c0, &ctx, &cos, &sin)?;
        }
        let h = self.final_layer.forward(&h, &c)?;
        unpatchify(&h, gh)
    }
}

/// Reassemble per-patch channels `[B, T, patch²·C]` into the latent image
/// `[B, C, gh·patch, gw·patch]` (`einsum nhwpqc->nchpwq`).
fn unpatchify(x: &Tensor, g: usize) -> Result<Tensor> {
    let b = x.dim(0)?;
    x.reshape((b, g, g, PATCH, PATCH, OUT_CH))?
        .permute((0, 5, 1, 3, 2, 4))?
        .contiguous()?
        .reshape((b, OUT_CH, g * PATCH, g * PATCH))
}
