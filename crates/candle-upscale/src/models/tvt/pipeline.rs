//! The TVT upscale pipeline (candle port of `TVTModel.py`'s `test_forward` /
//! `tile_forward`).
//!
//! Bicubically upscale the low-res image ×4 to the target resolution, encode it
//! with the VAE-D4 in overlapping tiles, run a single UNet forward (conditioned
//! on a fixed prompt embedding) over the same latent tiles, apply the one-step
//! epsilon → x₀ solve, and decode the refined latent in tiles. There is no
//! multi-step loop and no classifier-free guidance (`--cfg 0`): the model does
//! its super-resolution in one shot, leaning on the detail-preserving 4× VAE
//! plus a light latent refinement.

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::unet::Unet;
use super::vae4x::{Decoder, Encoder};
use crate::common::resize::bicubic_chw;
use crate::common::tiling::tile_origins;
use crate::model::{UpscaleModel, UpscaleOptions};

/// VAE-D4 latent scaling factor (`vae.config.scaling_factor`, inherited from SD).
const SCALING_FACTOR: f64 = 0.18215;
/// Fixed integer upscale factor.
const SCALE: usize = 4;
/// Latent-to-pixel ratio of VAE-D4.
const AE_FACTOR: usize = 4;
/// Fixed UNet conditioning timestep (`--time_step 1` in the reference script).
const TIME_STEP: i64 = 1;
/// SD2.1 DDPM schedule bounds (`scaled_linear`).
const BETA_START: f64 = 0.000_85;
const BETA_END: f64 = 0.012;
const NUM_TRAIN_TIMESTEPS: usize = 1000;
/// Variance of the Gaussian latent-tile blend mask (`_gaussian_weights`).
const BLEND_VAR: f64 = 0.01;

fn io_err(context: &str, e: std::io::Error) -> candle_core::Error {
    candle_core::Error::Msg(format!("{context}: {e}"))
}

/// `alphas_cumprod[t]` for the SD2.1 `scaled_linear` schedule: `betas =
/// linspace(sqrt(beta_start), sqrt(beta_end), N)²`, `acp = cumprod(1 - betas)`.
fn alpha_cumprod_at(t: usize) -> f64 {
    let start = BETA_START.sqrt();
    let end = BETA_END.sqrt();
    let n = NUM_TRAIN_TIMESTEPS;
    let mut acp = 1.0f64;
    for i in 0..=t.min(n - 1) {
        let s = start + (end - start) * (i as f64) / ((n - 1) as f64);
        acp *= 1.0 - s * s;
    }
    acp
}

/// Loaded TVT model: the SD2.1 UNet, the VAE-D4 encoder + decoder, and the fixed
/// prompt embedding, all on one device at one compute dtype.
pub struct Tvt {
    unet: Unet,
    encoder: Encoder,
    decoder: Decoder,
    /// Precomputed prompt CLIP embedding `[1, 77, 1024]`.
    text_embed: Tensor,
    device: Device,
    dtype: DType,
}

impl Tvt {
    /// Load the fused UNet, the VAE-D4, and the prompt embedding at `dtype`.
    ///
    /// `unet_path` is the offline-fused safetensors (base SD2.1 UNet with the TVT
    /// LoRA deltas merged in); `vae_path` is the VAE-D4 safetensors (encoder +
    /// decoder + quant convs). Both keep diffusers layout, so no transpose or
    /// rename is needed.
    ///
    /// # Errors
    /// Fails if a weight file cannot be read or does not match the architecture.
    pub fn load(
        unet_path: &Path,
        vae_path: &Path,
        embed_bytes: &[u8],
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let unet_bytes = std::fs::read(unet_path).map_err(|e| io_err("read UNet", e))?;
        let unet_vb = VarBuilder::from_buffered_safetensors(unet_bytes, dtype, &device)?;
        let unet = Unet::new(unet_vb)?;

        let vae_bytes = std::fs::read(vae_path).map_err(|e| io_err("read VAE", e))?;
        let vae_vb = VarBuilder::from_buffered_safetensors(vae_bytes, dtype, &device)?;
        let encoder = Encoder::new(vae_vb.clone())?;
        let decoder = Decoder::new(vae_vb)?;

        let embed_vb = VarBuilder::from_buffered_safetensors(embed_bytes.to_vec(), dtype, &device)?;
        let text_embed = embed_vb.get((1, 77, 1024), "prompt_embed")?;

        Ok(Self {
            unet,
            encoder,
            decoder,
            text_embed,
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

    /// One-step epsilon prediction over the whole (possibly tiled) latent, then
    /// the DDPM one-step solve `x₀ = (z - sqrt(1-acp[t])·ε) / sqrt(acp[t])`.
    ///
    /// `lq_latent` is the scaled VAE-D4 latent `[1, 4, h, w]`; `opts.tile` /
    /// `opts.overlap` bound the latent tile so a large image runs the UNet in
    /// gaussian-blended tiles (mirroring the reference `tile_forward`).
    fn refine_latent(&self, lq_latent: &Tensor, opts: &UpscaleOptions) -> Result<Tensor> {
        let (_, _, lh, lw) = lq_latent.dims4()?;
        let eps = self.unet_over_tiles(lq_latent, lh, lw, opts)?;

        let t = TIME_STEP.max(0) as usize;
        let acp = alpha_cumprod_at(t);
        let sqrt_acp = acp.sqrt();
        let sqrt_beta = (1.0 - acp).sqrt();
        // pred_original_sample for epsilon prediction; the one-step DDPM at t with
        // num_inference_steps=1 has prev_t<0 → alpha_prod_t_prev=1 → this is the
        // whole update (no added variance).
        let x0 = ((lq_latent - (eps * sqrt_beta)?)? * (1.0 / sqrt_acp))?;
        Ok(x0)
    }

    /// UNet epsilon prediction, single forward when the latent fits one tile,
    /// otherwise gaussian-blended latent tiles.
    fn unet_over_tiles(
        &self,
        lq_latent: &Tensor,
        lh: usize,
        lw: usize,
        opts: &UpscaleOptions,
    ) -> Result<Tensor> {
        let (origins, th, tw, _) = tile_origins(lw, lh, opts.tile, opts.overlap);
        if lh <= th && lw <= tw {
            return self.unet_forward(lq_latent);
        }

        let g = gaussian_mask(th, tw, 4, &self.device, self.dtype)?;
        let mut acc = Tensor::zeros((1, 4, lh, lw), self.dtype, &self.device)?;
        let mut wsum = acc.clone();
        for &(y0, x0) in &origins {
            let tile_in = lq_latent
                .narrow(2, y0, th)?
                .narrow(3, x0, tw)?
                .contiguous()?;
            let pred = (self.unet_forward(&tile_in)? * &g)?;
            acc = (acc
                + pred.pad_with_zeros(2, y0, lh - y0 - th)?.pad_with_zeros(
                    3,
                    x0,
                    lw - x0 - tw,
                )?)?;
            wsum = (wsum
                + g.pad_with_zeros(2, y0, lh - y0 - th)?
                    .pad_with_zeros(3, x0, lw - x0 - tw)?)?;
        }
        acc.div(&wsum)
    }

    /// A single UNet epsilon forward on a latent tile `[1, 4, h, w]`.
    fn unet_forward(&self, latent: &Tensor) -> Result<Tensor> {
        self.unet.forward(
            latent,
            TIME_STEP as f32,
            &self.text_embed,
            &self.device,
            self.dtype,
        )
    }

    /// VAE-encode the HR conditioning image in latent-aligned tiles so its
    /// activations stay bounded, returning the stitched, unscaled mean latent
    /// `[1, 4, hh/4, ww/4]`. The user-facing low-res tile size maps directly to
    /// VAE-D4 latent pixels because both upscale and compression factors are ×4.
    fn encode_tiled(
        &self,
        conditioning: &Tensor,
        hh: usize,
        ww: usize,
        opts: &UpscaleOptions,
    ) -> Result<Tensor> {
        let (lh, lw) = (hh / AE_FACTOR, ww / AE_FACTOR);
        let (origins, th, tw, _) = tile_origins(lw, lh, opts.tile, opts.overlap);
        if lh <= th && lw <= tw {
            return self.encoder.encode_mean(conditioning);
        }

        let g = gaussian_mask(th, tw, 4, &self.device, self.dtype)?;
        let mut acc = Tensor::zeros((1, 4, lh, lw), self.dtype, &self.device)?;
        let mut wsum = acc.clone();
        for &(ly, lx) in &origins {
            let tile = conditioning
                .narrow(2, ly * AE_FACTOR, th * AE_FACTOR)?
                .narrow(3, lx * AE_FACTOR, tw * AE_FACTOR)?
                .contiguous()?;
            let latent = (self.encoder.encode_mean(&tile)? * &g)?;
            let (bottom, right) = (lh - ly - th, lw - lx - tw);
            acc = (acc
                + latent
                    .pad_with_zeros(2, ly, bottom)?
                    .pad_with_zeros(3, lx, right)?)?;
            wsum = (wsum
                + g.pad_with_zeros(2, ly, bottom)?
                    .pad_with_zeros(3, lx, right)?)?;
        }
        acc.div(&wsum)
    }

    /// VAE-decode the latent in overlapping tiles so its activations stay
    /// bounded, returning the stitched image `[1, 3, lh*4, lw*4]` in
    /// approximately `[-1, 1]`.
    fn decode_tiled(
        &self,
        latent: &Tensor,
        lh: usize,
        lw: usize,
        opts: &UpscaleOptions,
    ) -> Result<Tensor> {
        let (origins, th, tw, _) = tile_origins(lw, lh, opts.tile, opts.overlap);
        if lh <= th && lw <= tw {
            return self.decoder.forward(latent);
        }

        let (hh, ww) = (lh * AE_FACTOR, lw * AE_FACTOR);
        let (ph, pw) = (th * AE_FACTOR, tw * AE_FACTOR);
        let g = gaussian_mask(ph, pw, 3, &self.device, self.dtype)?;
        let mut acc = Tensor::zeros((1, 3, hh, ww), self.dtype, &self.device)?;
        let mut wsum = acc.clone();
        for &(ly, lx) in &origins {
            let tile = latent.narrow(2, ly, th)?.narrow(3, lx, tw)?.contiguous()?;
            let decoded = (self.decoder.forward(&tile)? * &g)?;
            let (py, px) = (ly * AE_FACTOR, lx * AE_FACTOR);
            let (bottom, right) = (hh - py - ph, ww - px - pw);
            acc = (acc
                + decoded
                    .pad_with_zeros(2, py, bottom)?
                    .pad_with_zeros(3, px, right)?)?;
            wsum = (wsum
                + g.pad_with_zeros(2, py, bottom)?
                    .pad_with_zeros(3, px, right)?)?;
        }
        acc.div(&wsum)
    }

    /// Run the full pipeline on an HR conditioning image (planar `[3, hh, ww]` in
    /// `[0, 1]`), returning the decoded image tensor `[1, 3, hh, ww]` in `[0, 1]`.
    fn run(&self, hr: &[f32], hh: usize, ww: usize, opts: &UpscaleOptions) -> Result<Tensor> {
        // Reference: c_t = to_tensor(image)*2-1, then vae.encode * scaling_factor.
        let c_t: Vec<f32> = hr.iter().map(|v| v * 2.0 - 1.0).collect();
        let c_t = Tensor::from_vec(c_t, (1, 3, hh, ww), &self.device)?.to_dtype(self.dtype)?;
        let lq_latent = (self.encode_tiled(&c_t, hh, ww, opts)? * SCALING_FACTOR)?;
        let (_, _, lh, lw) = lq_latent.dims4()?;

        let x0 = self.refine_latent(&lq_latent, opts)?;

        let decoded = self
            .decode_tiled(&(x0 * (1.0 / SCALING_FACTOR))?, lh, lw, opts)?
            .clamp(-1f32, 1f32)?;
        decoded.affine(0.5, 0.5)
    }
}

/// Gaussian blend mask `[1, channels, th, tw]` peaked at the centre.
fn gaussian_mask(
    th: usize,
    tw: usize,
    channels: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let axis = |len: usize| -> Vec<f64> {
        let mid = (len as f64 - 1.0) / 2.0;
        (0..len)
            .map(|i| (-((i as f64 - mid) / len as f64).powi(2) / (2.0 * BLEND_VAR)).exp())
            .collect()
    };
    let (ay, ax) = (axis(th), axis(tw));
    let mut w = vec![0f32; th * tw];
    for y in 0..th {
        for x in 0..tw {
            w[y * tw + x] = (ay[y] * ax[x]) as f32;
        }
    }
    Tensor::from_vec(w, (1, 1, th, tw), device)?
        .broadcast_as((1, channels, th, tw))?
        .to_dtype(dtype)?
        .contiguous()
}

/// Upscale by TVT's fixed ×4 factor.
impl UpscaleModel for Tvt {
    fn native_scale(&self) -> usize {
        SCALE
    }

    fn upscale_rgba(
        &self,
        rgba: &[u8],
        width: usize,
        height: usize,
        opts: &UpscaleOptions,
        _seed: u64,
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

        on_progress(0.0);
        let img = self.run(&hr, hh, ww, opts)?;
        on_progress(1.0);

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
