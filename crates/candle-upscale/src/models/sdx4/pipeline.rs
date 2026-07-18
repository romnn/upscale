//! The candle upscale pipeline, mirroring `sd-upscale/src/pipeline.rs`.
//!
//! preprocess low-res to `[-1,1]`, DDPM-noise it once at `noise_level`, then a
//! DDIM loop of `unet(cat[latents, image]) → step`, and finally
//! `vae.decode(latents / scaling_factor)` mapped back to `[0,1]`, with the same
//! seam-blended tiling as the burn reference.

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::unet::Unet;
use super::vae::{VaeConfig, VaeDecoder};
use crate::common::noise::gaussian;
use crate::common::scheduler::{DdimScheduler, LowResNoiser};
use crate::common::tiling::{accumulate, normalize_to_rgba, rgba_to_tensor, tile_origins};
use crate::model::{UpscaleModel, UpscaleOptions};

/// VAE latent scaling factor for the x4 upscaler (`vae.config.scaling_factor`).
const SCALING_FACTOR: f64 = 0.08333;
const SCALE: usize = 4;

fn io_err(context: &str, e: std::io::Error) -> candle_core::Error {
    candle_core::Error::Msg(format!("{context}: {e}"))
}

/// Loaded SD-x4 model + device + compute dtype, ready to upscale images.
pub struct Sdx4 {
    unet: Unet,
    vae: VaeDecoder,
    /// Precomputed empty-prompt CLIP embedding `[1, 77, 1024]`.
    text_embed: Tensor,
    device: Device,
    dtype: DType,
}

impl Sdx4 {
    /// Load the UNet, VAE, and empty-prompt embedding at the compute `dtype`.
    ///
    /// The weights ship as f32 on disk; `VarBuilder` converts them to `dtype` on
    /// load, so a bf16 run reads f32 and casts once (matching burn's
    /// `cast_weights`). candle keeps PyTorch layout, so no transpose or name
    /// remap is needed.
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
        let vae = VaeDecoder::new(&VaeConfig::default(), vae_vb)?;

        let embed_vb = VarBuilder::from_buffered_safetensors(embed_bytes.to_vec(), dtype, &device)?;
        let text_embed = embed_vb.get((1, 77, 1024), "empty_prompt_embed")?;

        Ok(Self {
            unet,
            vae,
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

    /// Build a `[b, c, h, w]` tensor of standard-normal noise at the compute
    /// dtype from a deterministic host RNG.
    pub fn noise(&self, seed: u64, b: usize, c: usize, h: usize, w: usize) -> Result<Tensor> {
        let data = gaussian(seed, b * c * h * w);
        Tensor::from_vec(data, (b, c, h, w), &self.device)?.to_dtype(self.dtype)
    }

    /// Run the DDIM denoising loop for one (batched) tile, returning the final
    /// latents. Deterministic: the caller supplies the initial latents and
    /// low-res noise, so this reproduces the reference pipeline exactly.
    /// `low_res01`: `[b,3,h,w]` in `[0,1]`; returns latents `[b,4,h,w]`.
    pub fn denoise(
        &self,
        low_res01: &Tensor,
        noise_level: i64,
        num_steps: usize,
        init_latents: Tensor,
        low_res_noise: &Tensor,
    ) -> Result<Tensor> {
        let b = init_latents.dim(0)?;
        let context = if b == 1 {
            self.text_embed.clone()
        } else {
            self.text_embed.broadcast_as((b, 77, 1024))?.contiguous()?
        };

        // preprocess to [-1,1], then DDPM-noise the conditioning image once.
        let image = low_res01.affine(2.0, -1.0)?;
        let image = LowResNoiser::new().add_noise(&image, low_res_noise, noise_level)?;

        let mut ddim = DdimScheduler::new();
        ddim.set_timesteps(num_steps);

        let mut latents = init_latents;
        for &t in ddim.timesteps() {
            let model_input = Tensor::cat(&[&latents, &image], 1)?;
            let noise_pred = self.unet.forward(
                &model_input,
                t as f32,
                &context,
                noise_level,
                &self.device,
                self.dtype,
            )?;
            latents = ddim.step(&noise_pred, t, &latents)?;
        }
        Ok(latents)
    }

    /// VAE-decode latents to an image `[b, 3, 4h, 4w]` in `[0, 1]`.
    pub fn decode_latents(&self, latents: &Tensor) -> Result<Tensor> {
        let decoded = self.vae.forward(&(latents * (1.0 / SCALING_FACTOR))?)?;
        decoded.affine(0.5, 0.5)?.clamp(0f32, 1f32)
    }

    /// [`denoise`](Self::denoise) then [`decode_latents`](Self::decode_latents).
    pub fn denoise_decode(
        &self,
        low_res01: &Tensor,
        noise_level: i64,
        num_steps: usize,
        init_latents: Tensor,
        low_res_noise: &Tensor,
    ) -> Result<Tensor> {
        let latents = self.denoise(
            low_res01,
            noise_level,
            num_steps,
            init_latents,
            low_res_noise,
        )?;
        self.decode_latents(&latents)
    }

}

/// Upscale by the SD-x4 model's fixed ×4 factor with seam-blended tiling.
impl UpscaleModel for Sdx4 {
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

        let full = rgba_to_tensor(rgba, width, height, &self.device, self.dtype)?;
        let (ow, oh) = (width * SCALE, height * SCALE);
        let mut out = vec![0f32; ow * oh * 3];
        let mut weight = vec![0f32; ow * oh];

        let (origins, th, tw, total) = tile_origins(width, height, opts.tile, opts.overlap);
        let batch = opts.batch.max(1);
        let mut done = 0;

        for (chunk_idx, chunk) in origins.chunks(batch).enumerate() {
            let tiles: Vec<Tensor> = chunk
                .iter()
                .map(|&(y0, x0)| full.narrow(2, y0, th)?.narrow(3, x0, tw)?.contiguous())
                .collect::<Result<_>>()?;
            let b = tiles.len();
            let tile_refs: Vec<&Tensor> = tiles.iter().collect();
            let batch_t = Tensor::cat(&tile_refs, 0)?;
            // Per-tile noise seeded by the tile's global index, so a tile gets the
            // same noise regardless of `--batch` (distinct init vs low-res streams).
            let noise_batch = |c: usize, twist: u64| -> Result<Tensor> {
                let per: Vec<Tensor> = (0..b)
                    .map(|s| {
                        let gi = (chunk_idx * batch + s) as u64;
                        self.noise(seed ^ gi.wrapping_mul(0x1000_0001) ^ twist, 1, c, th, tw)
                    })
                    .collect::<Result<_>>()?;
                Tensor::cat(&per.iter().collect::<Vec<_>>(), 0)
            };
            let init = noise_batch(4, 0)?;
            let lrn = noise_batch(3, 0xA5A5_A5A5)?;
            let out_tiles =
                self.denoise_decode(&batch_t, opts.noise_level, opts.steps, init, &lrn)?;

            let (_, _, th4, tw4) = out_tiles.dims4()?;
            let vals = out_tiles
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;

            let per_tile = 3 * th4 * tw4;
            for (s, &(y0, x0)) in chunk.iter().enumerate() {
                accumulate(
                    &mut out,
                    &mut weight,
                    &vals[s * per_tile..(s + 1) * per_tile],
                    th4,
                    tw4,
                    ow,
                    x0 * SCALE,
                    y0 * SCALE,
                );
                done += 1;
                on_progress(done as f32 / total as f32);
            }
        }

        Ok((normalize_to_rgba(&out, &weight, ow, oh), ow, oh))
    }
}
