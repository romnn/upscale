//! The VOSR upscale pipeline (candle port of `inference_vosr.py`'s
//! `tiled_latent_inference`).
//!
//! Bicubically upsample the low-res image ×4 to form the conditioning image,
//! encode it once to a latent, and run a CFG flow-matching Euler loop that tiles
//! the DiT forward in latent space — each tile conditioned on DINOv2 layer-8
//! features of the matching pixel region, its velocity predictions blended
//! across overlaps with a Gaussian mask. The denoised latent is decoded once by
//! the LightDecoder and mapped back to RGBA.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;

use super::dino::{Dinov2, IMAGENET_MEAN, IMAGENET_STD, INPUT as DINO_INPUT};
use super::dit::Dit;
use super::vae::{Encoder, LightDecoder};
use crate::common::noise::gaussian;
use crate::common::resize::bicubic_chw;
use crate::model::{UpscaleModel, UpscaleOptions};

/// SD2.1 VAE latent scaling factor (`vae.config.scaling_factor`).
const SCALING_FACTOR: f64 = 0.18215;
/// Classifier-free guidance scale (`vosr_model.cfg_scale`, from `args.json`).
const CFG_SCALE: f64 = 2.0;
/// Weak-conditioning interpolation strength toward zeros. The reference uses the
/// midpoint of `weak_cond_strength_aelq_list = [0.05, 0.25]`, i.e. 0.15.
const WEAK_COND_STRENGTH: f64 = 0.15;
/// Fixed integer upscale factor.
const SCALE: usize = 4;
/// Latent-to-pixel ratio of the SD2.1 autoencoder.
const AE_FACTOR: usize = 8;
/// Patch size the DiT/latent tiling aligns to.
const PATCH: usize = 2;
/// Variance of the Gaussian blend mask (`_gaussian_weights`).
const BLEND_VAR: f64 = 0.01;
/// Latent-space tile size for the VAE encode/decode, so their activations stay
/// bounded on large images instead of processing the full frame at once. `64`
/// latent = a 512×512 pixel tile (the autoencoder's comfortable size); tiles
/// overlap and are Gaussian-blended, so the stitched result is seam-free.
const VAE_LAT_TILE: usize = 64;
/// Latent-space overlap between adjacent VAE tiles.
const VAE_LAT_OVERLAP: usize = 8;

fn io_err(context: &str, e: std::io::Error) -> candle_core::Error {
    candle_core::Error::Msg(format!("{context}: {e}"))
}

/// Loaded VOSR model: the DiT, the SD2.1 encoder, the LightDecoder, and the
/// DINOv2 conditioning backbone, all on one device at one compute dtype.
pub struct Vosr {
    dit: Dit,
    encoder: Encoder,
    decoder: LightDecoder,
    dino: Dinov2,
    device: Device,
    dtype: DType,
}

impl Vosr {
    /// Load every component from its weight file.
    ///
    /// `dit_path` is safetensors (the EMA DiT); `encoder_path` is the SD2.1 VAE
    /// safetensors; `decoder_path` and `dino_path` are PyTorch `.pth` pickles
    /// candle reads directly.
    ///
    /// # Errors
    /// Fails if any weight file cannot be read or does not match the expected
    /// architecture.
    pub fn load(
        dit_path: &Path,
        encoder_path: &Path,
        decoder_path: &Path,
        dino_path: &Path,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let dit_bytes = std::fs::read(dit_path).map_err(|e| io_err("read DiT", e))?;
        let dit_vb = VarBuilder::from_buffered_safetensors(dit_bytes, dtype, &device)?;
        let dit = Dit::new(dit_vb, device.clone(), dtype)?;

        let enc_bytes = std::fs::read(encoder_path).map_err(|e| io_err("read VAE", e))?;
        let enc_vb = VarBuilder::from_buffered_safetensors(enc_bytes, dtype, &device)?;
        let encoder = Encoder::new(enc_vb)?;

        let dec_vb =
            VarBuilder::from_pth_with_state(decoder_path, dtype, "model_state_dict", &device)?;
        let decoder = LightDecoder::new(dec_vb)?;

        let dino_vb = VarBuilder::from_pth(dino_path, dtype, &device)?;
        let dino = Dinov2::new(dino_vb, &device, dtype)?;

        Ok(Self {
            dit,
            encoder,
            decoder,
            dino,
            device,
            dtype,
        })
    }

    /// The device the model lives on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The compute dtype (`F32` or `BF16`).
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// DINOv2 layer-8 features `[1, N, 768]` for an HR pixel crop `[0, 1]`
    /// (planar `[3, ch, cw]`): bicubic to 448², clamp, ImageNet-normalize.
    fn dino_features(&self, crop: &[f32], ch: usize, cw: usize) -> Result<Tensor> {
        let resized = bicubic_chw(crop, 3, ch, cw, DINO_INPUT, DINO_INPUT);
        let plane = DINO_INPUT * DINO_INPUT;
        let mut norm = vec![0f32; 3 * plane];
        for c in 0..3 {
            for i in 0..plane {
                let v = resized[c * plane + i].clamp(0.0, 1.0);
                norm[c * plane + i] = (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
        }
        let x = Tensor::from_vec(norm, (1, 3, DINO_INPUT, DINO_INPUT), &self.device)?
            .to_dtype(self.dtype)?;
        self.dino.forward(&x)
    }

    /// One CFG DiT forward for a tile: batches the conditioned and weak inputs,
    /// runs the DiT, and combines the two velocity predictions.
    fn tile_velocity(
        &self,
        lq_tile: &Tensor,
        lq_weak_tile: &Tensor,
        z_tile: &Tensor,
        feats: &Tensor,
        t_cur: f32,
    ) -> Result<Tensor> {
        let inp_cond = Tensor::cat(&[lq_tile, z_tile], 1)?;
        let inp_weak = Tensor::cat(&[lq_weak_tile, z_tile], 1)?;
        let model_inp = Tensor::cat(&[&inp_cond, &inp_weak], 0)?;
        let z_ctx = Tensor::cat(&[feats, &feats.zeros_like()?], 0)?;
        let t = Tensor::from_vec(vec![t_cur, t_cur], 2, &self.device)?;

        let d_out = self.dit.forward(&model_inp, &t, &z_ctx)?;
        let d_cond = d_out.i(0)?.unsqueeze(0)?;
        let d_weak = d_out.i(1)?.unsqueeze(0)?;
        d_weak.broadcast_add(&(d_cond.broadcast_sub(&d_weak)?.affine(CFG_SCALE, 0.0)?))
    }
}

/// Sorted, deduplicated tile start positions covering `length`
/// (`_make_tile_grid`).
fn tile_grid(length: usize, tile: usize, overlap: usize) -> Vec<usize> {
    if length <= tile {
        return vec![0];
    }
    let stride = tile.saturating_sub(overlap).max(1);
    let mut pos: Vec<usize> = (0..=length - tile).step_by(stride).collect();
    if pos.last().is_some_and(|&p| p + tile < length) {
        pos.push(length - tile);
    }
    pos.dedup();
    pos
}

/// Gaussian blend mask `[1, channels, tile, tile]` peaked at the centre
/// (`_gaussian_weights`). Feathers overlapping tiles: the 4-channel latent in the
/// DiT loop and the VAE encode, the 3-channel image in the VAE decode.
fn gaussian_mask(tile: usize, channels: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let mid = (tile as f64 - 1.0) / 2.0;
    let axis: Vec<f64> = (0..tile)
        .map(|i| (-((i as f64 - mid) / tile as f64).powi(2) / (2.0 * BLEND_VAR)).exp())
        .collect();
    let mut w = vec![0f32; tile * tile];
    for y in 0..tile {
        for x in 0..tile {
            w[y * tile + x] = (axis[y] * axis[x]) as f32;
        }
    }
    Tensor::from_vec(w, (1, 1, tile, tile), device)?
        .broadcast_as((1, channels, tile, tile))?
        .to_dtype(dtype)?
        .contiguous()
}

/// Crop a planar `[c, h, w]` buffer to `[c, y1-y0, x1-x0]`.
fn crop_chw(src: &[f32], c: usize, h: usize, w: usize, y0: usize, y1: usize, x0: usize, x1: usize) -> Vec<f32> {
    let (ch, cw) = (y1 - y0, x1 - x0);
    let mut out = vec![0f32; c * ch * cw];
    for ci in 0..c {
        for y in 0..ch {
            let src_row = ci * h * w + (y0 + y) * w + x0;
            let dst_row = ci * ch * cw + y * cw;
            out[dst_row..dst_row + cw].copy_from_slice(&src[src_row..src_row + cw]);
        }
    }
    out
}

impl Vosr {
    /// VAE-encode the HR conditioning image tile-by-tile so the encoder's
    /// activations stay bounded on large frames, returning the stitched mean
    /// latent `[1, 4, hh/8, ww/8]` (unscaled). Latent-aligned tiles overlap and
    /// are Gaussian-blended, so the result is seam-free.
    fn encode_tiled(&self, hr: &Tensor, hh: usize, ww: usize) -> Result<Tensor> {
        let (lh, lw) = (hh / AE_FACTOR, ww / AE_FACTOR);
        let lt = VAE_LAT_TILE.min(lh).min(lw).max(1);
        let lo = VAE_LAT_OVERLAP.min(lt.saturating_sub(1));
        let g = gaussian_mask(lt, 4, &self.device, self.dtype)?;
        let mut acc = Tensor::zeros((1, 4, lh, lw), self.dtype, &self.device)?;
        let mut wacc = acc.clone();
        for &ly in &tile_grid(lh, lt, lo) {
            for &lx in &tile_grid(lw, lt, lo) {
                let ptile = hr
                    .narrow(2, ly * AE_FACTOR, lt * AE_FACTOR)?
                    .narrow(3, lx * AE_FACTOR, lt * AE_FACTOR)?
                    .contiguous()?;
                let lat = self.encoder.encode_mean(&ptile)?;
                let (pr, pd) = (lh - ly - lt, lw - lx - lt);
                acc = (acc + (lat * &g)?.pad_with_zeros(2, ly, pr)?.pad_with_zeros(3, lx, pd)?)?;
                wacc = (wacc + g.pad_with_zeros(2, ly, pr)?.pad_with_zeros(3, lx, pd)?)?;
            }
        }
        acc.div(&wacc)
    }

    /// VAE-decode the latent tile-by-tile through the LightDecoder so its
    /// activations stay bounded, returning the image `[1, 3, lh*8, lw*8]` in
    /// `[-1, 1]`. Tiles overlap and are Gaussian-blended in pixel space.
    fn decode_tiled(&self, latent: &Tensor, lh: usize, lw: usize) -> Result<Tensor> {
        let (hh, ww) = (lh * AE_FACTOR, lw * AE_FACTOR);
        let lt = VAE_LAT_TILE.min(lh).min(lw).max(1);
        let lo = VAE_LAT_OVERLAP.min(lt.saturating_sub(1));
        let pt = lt * AE_FACTOR;
        let g = gaussian_mask(pt, 3, &self.device, self.dtype)?;
        let mut acc = Tensor::zeros((1, 3, hh, ww), self.dtype, &self.device)?;
        let mut wacc = acc.clone();
        for &ly in &tile_grid(lh, lt, lo) {
            for &lx in &tile_grid(lw, lt, lo) {
                let ltile = latent.narrow(2, ly, lt)?.narrow(3, lx, lt)?.contiguous()?;
                let px = self.decoder.forward(&ltile)?;
                let (py, pxx) = (ly * AE_FACTOR, lx * AE_FACTOR);
                let (pr, pd) = (hh - py - pt, ww - pxx - pt);
                acc = (acc + (px * &g)?.pad_with_zeros(2, py, pr)?.pad_with_zeros(3, pxx, pd)?)?;
                wacc = (wacc + g.pad_with_zeros(2, py, pr)?.pad_with_zeros(3, pxx, pd)?)?;
            }
        }
        acc.div(&wacc)
    }

    /// Run the full pipeline on an HR conditioning image (planar `[3, hh, ww]`
    /// in `[0, 1]`), returning the decoded image tensor `[1, 3, hh, ww]` in
    /// `[0, 1]`.
    fn run(
        &self,
        hr: &[f32],
        hh: usize,
        ww: usize,
        opts: &UpscaleOptions,
        seed: u64,
        on_progress: &mut dyn FnMut(f32),
    ) -> Result<Tensor> {
        let lq: Vec<f32> = hr.iter().map(|v| v * 2.0 - 1.0).collect();
        let lq = Tensor::from_vec(lq, (1, 3, hh, ww), &self.device)?.to_dtype(self.dtype)?;
        let lq_latent = (self.encode_tiled(&lq, hh, ww)? * SCALING_FACTOR)?;
        let (_, _, lh, lw) = lq_latent.dims4()?;

        // `opts.tile`/`opts.overlap` are low-res pixel sizes; the ×4 upscale makes
        // the HR tile `opts.tile * SCALE`, which the reference maps to a latent
        // tile of `(hr_tile / 8 / patch) * patch`. The default (128) yields a
        // 64-latent tile — patch grid 32, the DiT's training resolution — so the
        // model runs at native scale; a much smaller tile drops the grid far
        // below training and degrades quality, so guard the tile at one patch.
        let tile_px = (opts.tile * SCALE).max(PATCH * AE_FACTOR);
        let overlap_px = opts.overlap * SCALE;
        let lt = (((tile_px / AE_FACTOR / PATCH) * PATCH).max(PATCH))
            .min(lh)
            .min(lw);
        let lo = (overlap_px / AE_FACTOR)
            .max(lt / 8)
            .min(lt.saturating_sub(1));
        let h_pos = tile_grid(lh, lt, lo);
        let w_pos = tile_grid(lw, lt, lo);

        // DINOv2 features per tile from the matching HR pixel region.
        let mut feats: HashMap<(usize, usize), Tensor> = HashMap::new();
        for &hi in &h_pos {
            for &wi in &w_pos {
                let (y0, y1) = (hi * AE_FACTOR, ((hi + lt) * AE_FACTOR).min(hh));
                let (x0, x1) = (wi * AE_FACTOR, ((wi + lt) * AE_FACTOR).min(ww));
                let crop = crop_chw(hr, 3, hh, ww, y0, y1, x0, x1);
                feats.insert((hi, wi), self.dino_features(&crop, y1 - y0, x1 - x0)?);
            }
        }

        let lq_weak = (&lq_latent * WEAK_COND_STRENGTH)?;
        let g = gaussian_mask(lt, 4, &self.device, self.dtype)?;

        let noise = gaussian(seed, 4 * lh * lw);
        let mut z = Tensor::from_vec(noise, (1, 4, lh, lw), &self.device)?.to_dtype(self.dtype)?;

        let n_steps = opts.steps.max(1);
        on_progress(0.0);
        for step in 0..n_steps {
            let t_cur = 1.0 - step as f32 / n_steps as f32;
            let t_nxt = 1.0 - (step + 1) as f32 / n_steps as f32;
            let dt = f64::from(t_cur - t_nxt);

            let mut u_acc = Tensor::zeros((1, 4, lh, lw), self.dtype, &self.device)?;
            let mut w_acc = u_acc.clone();
            for &hi in &h_pos {
                for &wi in &w_pos {
                    let tile = |t: &Tensor| -> Result<Tensor> {
                        t.narrow(2, hi, lt)?.narrow(3, wi, lt)?.contiguous()
                    };
                    let u = self.tile_velocity(
                        &tile(&lq_latent)?,
                        &tile(&lq_weak)?,
                        &tile(&z)?,
                        &feats[&(hi, wi)],
                        t_cur,
                    )?;
                    let contrib = (u * &g)?
                        .pad_with_zeros(2, hi, lh - hi - lt)?
                        .pad_with_zeros(3, wi, lw - wi - lt)?;
                    let wpad = g
                        .pad_with_zeros(2, hi, lh - hi - lt)?
                        .pad_with_zeros(3, wi, lw - wi - lt)?;
                    u_acc = (u_acc + contrib)?;
                    w_acc = (w_acc + wpad)?;
                }
            }
            z = (z - (u_acc.div(&w_acc)?.affine(dt, 0.0))?)?;
            on_progress((step + 1) as f32 / n_steps as f32);
        }

        let decoded = self
            .decode_tiled(&(z / SCALING_FACTOR)?, lh, lw)?
            .clamp(-1f32, 1f32)?;
        decoded.affine(0.5, 0.5)
    }
}

/// Upscale by VOSR's fixed ×4 factor.
impl UpscaleModel for Vosr {
    fn native_scale(&self) -> usize {
        SCALE
    }

    fn upscale_rgba(
        &self,
        rgba: &[u8],
        width: usize,
        height: usize,
        opts: &UpscaleOptions,
        seed: u64,
        on_progress: &mut dyn FnMut(f32),
    ) -> Result<(Vec<u8>, usize, usize)> {
        if rgba.len() != width * height * 4 {
            return Err(candle_core::Error::Msg(format!(
                "rgba len {} != {width}x{height}x4",
                rgba.len()
            )));
        }

        // RGBA8 -> planar RGB [0,1], then bicubic ×4 to the HR conditioning image.
        let hw = width * height;
        let mut rgb = vec![0f32; 3 * hw];
        for i in 0..hw {
            for c in 0..3 {
                rgb[c * hw + i] = f32::from(rgba[i * 4 + c]) / 255.0;
            }
        }
        let (ww, hh) = (width * SCALE, height * SCALE);
        let hr = bicubic_chw(&rgb, 3, height, width, hh, ww);

        let img = self.run(&hr, hh, ww, opts, seed, on_progress)?;
        let vals = img.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

        let plane = hh * ww;
        let mut out = vec![0u8; plane * 4];
        for i in 0..plane {
            for c in 0..3 {
                let v = vals[c * plane + i].clamp(0.0, 1.0);
                out[i * 4 + c] = (v * 255.0 + 0.5) as u8;
            }
            out[i * 4 + 3] = 255;
        }
        Ok((out, ww, hh))
    }
}
