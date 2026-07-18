//! Minimal DINOv2 ViT-B/14 for VOSR's vision conditioning (candle port).
//!
//! VOSR conditions its cross-attention on the *layer-8* patch tokens of a
//! DINOv2 ViT-B/14 run on a 448² crop, so only the patch embed, positional
//! embedding, and the first nine transformer blocks are needed — the final norm
//! and classification head the pretrained checkpoint lacks are irrelevant.
//! Rather than depend on `candle-transformers` (whose `DinoVisionTransformer`
//! requires a head the backbone checkpoint omits and fixes the input at 518²),
//! this hand-rolls the slice that VOSR actually uses.
//!
//! The 518²-trained positional grid (37×37) is bicubically resampled to the 32×32
//! grid of a 448² input once at load, matching DINOv2's `interpolate_pos_encoding`.

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::ops::softmax;
use candle_nn::{conv2d, layer_norm, linear, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};

use crate::common::resize::bicubic_chw;

/// DINOv2 input resolution VOSR preprocesses crops to (`args.dinov2_size`).
pub(crate) const INPUT: usize = 448;
const PATCH: usize = 14;
const DIM: usize = 768;
const HEADS: usize = 12;
const HEAD_DIM: usize = DIM / HEADS;
const MLP_HIDDEN: usize = DIM * 4;
/// Block whose output tokens feed VOSR's cross-attention (`layer_dinov2b_list`),
/// so blocks `0..=LAYER` are evaluated and the rest skipped.
const LAYER: usize = 8;
/// Side of the pretrained positional grid (518 / 14).
const PRETRAIN_GRID: usize = 37;
const NORM_EPS: f64 = 1e-6;

/// ImageNet normalization the reference applies before the encoder.
pub(crate) const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub(crate) const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug)]
struct Attention {
    qkv: Linear,
    proj: Linear,
}

impl Attention {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            qkv: linear(DIM, 3 * DIM, vb.pp("qkv"))?,
            proj: linear(DIM, DIM, vb.pp("proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, c) = x.dims3()?;
        let qkv = self
            .qkv
            .forward(x)?
            .reshape((b, n, 3, HEADS, HEAD_DIM))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.i(0)?.contiguous()?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? * scale)?;
        let out = softmax(&scores, D::Minus1)?
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, n, c))?;
        self.proj.forward(&out)
    }
}

#[derive(Debug)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(DIM, MLP_HIDDEN, vb.pp("fc1"))?,
            fc2: linear(MLP_HIDDEN, DIM, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // DINOv2's MLP uses exact (erf) GELU.
        self.fc2.forward(&self.fc1.forward(x)?.gelu_erf()?)
    }
}

#[derive(Debug)]
struct Block {
    norm1: LayerNorm,
    attn: Attention,
    ls1: Tensor,
    norm2: LayerNorm,
    mlp: Mlp,
    ls2: Tensor,
}

impl Block {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: layer_norm(DIM, NORM_EPS, vb.pp("norm1"))?,
            attn: Attention::new(vb.pp("attn"))?,
            ls1: vb.get(DIM, "ls1.gamma")?,
            norm2: layer_norm(DIM, NORM_EPS, vb.pp("norm2"))?,
            mlp: Mlp::new(vb.pp("mlp"))?,
            ls2: vb.get(DIM, "ls2.gamma")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self
            .attn
            .forward(&self.norm1.forward(x)?)?
            .broadcast_mul(&self.ls1)?)?;
        let h = self
            .mlp
            .forward(&self.norm2.forward(&x)?)?
            .broadcast_mul(&self.ls2)?;
        x + h
    }
}

/// The truncated DINOv2 ViT-B/14 backbone used for conditioning.
#[derive(Debug)]
pub(crate) struct Dinov2 {
    patch_embed: Conv2d,
    cls_token: Tensor,
    pos_embed: Tensor,
    blocks: Vec<Block>,
}

impl Dinov2 {
    /// Build the backbone, precomputing the 32×32 positional grid from the
    /// checkpoint's 37×37 one.
    pub(crate) fn new(vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let patch_embed = conv2d(
            3,
            DIM,
            PATCH,
            Conv2dConfig {
                stride: PATCH,
                ..Default::default()
            },
            vb.pp("patch_embed").pp("proj"),
        )?;
        let cls_token = vb.get((1, 1, DIM), "cls_token")?;
        let pos_embed = Self::interp_pos_embed(&vb, device, dtype)?;
        let bvb = vb.pp("blocks");
        let blocks = (0..=LAYER)
            .map(|i| Block::new(bvb.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_embed,
            cls_token,
            pos_embed,
            blocks,
        })
    }

    /// Bicubically resample the pretrained 37×37 patch positional embedding to
    /// the 32×32 grid of a 448² input, keeping the class-token embedding.
    fn interp_pos_embed(vb: &VarBuilder, device: &Device, dtype: DType) -> Result<Tensor> {
        let grid = INPUT / PATCH;
        let n_pre = PRETRAIN_GRID * PRETRAIN_GRID;
        let full = vb.get((1, n_pre + 1, DIM), "pos_embed")?.to_dtype(DType::F32)?;
        let cls = full.i((.., 0..1))?;
        // [1, N, C] -> [C, grid, grid] for the separable bicubic pass.
        let patch = full
            .i((.., 1..))?
            .reshape((PRETRAIN_GRID, PRETRAIN_GRID, DIM))?
            .permute((2, 0, 1))?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let resized = bicubic_chw(&patch, DIM, PRETRAIN_GRID, PRETRAIN_GRID, grid, grid);
        let patch = Tensor::from_vec(resized, (DIM, grid, grid), device)?
            .permute((1, 2, 0))?
            .reshape((1, grid * grid, DIM))?;
        Tensor::cat(&[&cls, &patch], 1)?.to_dtype(dtype)
    }

    /// Layer-8 patch tokens `[B, grid², 768]` (class token dropped) for a
    /// normalized `[B, 3, 448, 448]` input.
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let b = x.dim(0)?;
        let tokens = self.patch_embed.forward(x)?;
        let (_, c, gh, gw) = tokens.dims4()?;
        let tokens = tokens.reshape((b, c, gh * gw))?.transpose(1, 2)?;
        let cls = self.cls_token.broadcast_as((b, 1, DIM))?;
        let mut h = Tensor::cat(&[&cls, &tokens], 1)?.broadcast_add(&self.pos_embed)?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        h.i((.., 1..))?.contiguous()
    }
}
