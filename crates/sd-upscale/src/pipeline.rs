//! The upscale pipeline — the public entry point the frontend calls.
//!
//! Replicates `StableDiffusionUpscalePipeline` (guidance_scale=0, so no CFG):
//! preprocess low-res to `[-1,1]`, DDPM-noise it once at `noise_level`, then a
//! DDIM loop of `unet(cat[latents, image]) → step`, and finally
//! `vae.decode(latents / scaling_factor)` mapped back to `[0,1]`.

use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, FloatDType, Tensor, TensorData};

use crate::scheduler::{DdimScheduler, LowResNoiser};
use crate::unet::Unet;
use crate::vae::VaeDecoder;
use crate::weights::{load_embed_bytes, load_unet_bytes, load_vae_decoder_bytes};

/// VAE latent scaling factor for the x4 upscaler (`vae.config.scaling_factor`).
const SCALING_FACTOR: f64 = 0.08333;
const SCALE: usize = 4;

/// Tunables exposed to the UI. Defaults mirror the reference pipeline's fast path.
#[derive(Clone, Debug)]
pub struct UpscaleOptions {
    /// DDIM steps (more = slower, sharper). Reference default 20–40.
    pub steps: usize,
    /// Low-res conditioning noise level `0..=350`. Lower = more faithful.
    pub noise_level: i64,
    /// Low-res tile size in pixels (bounds GPU/VRAM per tile; the model is
    /// trained around 128px tiles).
    pub tile: usize,
    /// Tile overlap in pixels (blended to hide seams).
    pub overlap: usize,
    /// Tiles processed per UNet/VAE forward. A single 128px tile badly
    /// underutilizes a discrete GPU; batching stacks tiles on the batch
    /// dimension to fill it, trading VRAM for throughput. `1` runs one tile per
    /// forward.
    pub batch: usize,
}

impl Default for UpscaleOptions {
    fn default() -> Self {
        Self {
            steps: 20,
            noise_level: 20,
            tile: 128,
            overlap: 16,
            batch: 1,
        }
    }
}

/// Loaded model + device, ready to upscale images.
pub struct Upscaler<B: Backend> {
    vae: VaeDecoder<B>,
    unet: Option<Unet<B>>,
    /// Precomputed empty-prompt CLIP embedding `[1, 77, 1024]`.
    text_embed: Option<Tensor<B, 3>>,
    device: B::Device,
}

impl<B: Backend> Upscaler<B> {
    /// Full pipeline from already-loaded components (used by tests and, once the
    /// browser wires in the UNet + embedding, by [`Self::upscale_rgba`]).
    pub fn new(
        unet: Unet<B>,
        vae: VaeDecoder<B>,
        text_embed: Tensor<B, 3>,
        device: B::Device,
    ) -> Self {
        Self {
            vae,
            unet: Some(unet),
            text_embed: Some(text_embed),
            device,
        }
    }

    /// Full browser entry point: load the UNet, VAE, and precomputed empty-prompt
    /// embedding from fetched safetensors bytes. Enables real diffusion upscaling.
    ///
    /// Set `half = true` when the UNet/VAE bytes are `*.fp16.safetensors` (~half
    /// the download); they're up-converted to f32 on load, so inference is
    /// identical. The embedding is always f32.
    ///
    /// Takes `unet_bytes`/`vae_bytes` by value and consumes them as it loads (the
    /// UNet first, so its ~1.76 GB frees before the VAE loads). This bounds peak
    /// wasm memory — a retained or copied UNet buffer overflows the wasm32 4 GB
    /// address space.
    pub fn load_full(
        unet_bytes: Vec<u8>,
        vae_bytes: Vec<u8>,
        embed_bytes: &[u8],
        half: bool,
        device: B::Device,
    ) -> Result<Self, String> {
        let unet = load_unet_bytes::<B>(unet_bytes, half, &device)?;
        let vae = load_vae_decoder_bytes::<B>(vae_bytes, half, &device)?;
        let embed = load_embed_bytes::<B>(embed_bytes, &device)?;
        Ok(Self::new(unet, vae, embed, device))
    }

    /// Placeholder loader: VAE only. `upscale_rgba` falls back to nearest ×4
    /// until the UNet + embedding are supplied via [`Self::load_full`]/[`Self::new`].
    /// Signature kept stable for the frontend.
    pub fn from_safetensors_bytes(vae_bytes: &[u8], device: B::Device) -> Result<Self, String> {
        let vae = load_vae_decoder_bytes::<B>(vae_bytes.to_vec(), false, &device)?;
        Ok(Self {
            vae,
            unet: None,
            text_embed: None,
            device,
        })
    }

    fn is_full(&self) -> bool {
        self.unet.is_some() && self.text_embed.is_some()
    }

    /// Cast the loaded UNet, VAE, and text-embedding tensors to `dtype`.
    ///
    /// Weights load at their on-disk precision (f32), whereas the pipeline's
    /// activations (`Tensor::random`, `rgba_to_tensor`, the embedding) follow the
    /// backend's default float element. Running the backend in bf16/f16 for
    /// tensor-core throughput therefore needs the weights brought to that same
    /// element, which this does in a single pass.
    ///
    /// A no-op when `dtype` already matches the weights (e.g. the default f32
    /// backend casting to f32).
    #[must_use]
    pub fn cast_weights(mut self, dtype: FloatDType) -> Self {
        let mut caster = WeightCaster { dtype };
        self.unet = self.unet.map(|unet| unet.map(&mut caster));
        self.vae = self.vae.map(&mut caster);
        self.text_embed = self.text_embed.map(|embed| embed.cast(dtype));
        self
    }

    /// Run the DDIM denoising loop for one tile, returning the final latents.
    /// Deterministic: the caller supplies the initial latents and low-res noise,
    /// so this reproduces the reference pipeline exactly. `low_res01`: `[1,3,h,w]`
    /// in `[0,1]`; returns latents `[1,4,h,w]`.
    pub fn denoise(
        &self,
        low_res01: Tensor<B, 4>,
        noise_level: i64,
        num_steps: usize,
        init_latents: Tensor<B, 4>,
        low_res_noise: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        #[expect(
            clippy::expect_used,
            reason = "denoise requires a full pipeline (new/load_full); upscale_rgba guards this via is_full()"
        )]
        let unet = self.unet.as_ref().expect("unet not loaded");
        #[expect(clippy::expect_used, reason = "full pipeline invariant, as above")]
        let context = self.text_embed.clone().expect("text embed not loaded");
        // Cross-attention reshapes the context to the hidden state's batch, so
        // it must match. The empty-prompt embedding is identical for every tile,
        // so repeating it along the batch dim yields the expected shape.
        let context = context.repeat_dim(0, init_latents.dims()[0]);

        // preprocess to [-1,1], then DDPM-noise the conditioning image once.
        let image = low_res01.mul_scalar(2.0).sub_scalar(1.0);
        let image = LowResNoiser::new().add_noise(image, low_res_noise, noise_level);

        let mut ddim = DdimScheduler::new();
        ddim.set_timesteps(num_steps);

        let mut latents = init_latents;
        for t in ddim.timesteps().to_vec() {
            let model_input = Tensor::cat(vec![latents.clone(), image.clone()], 1);
            let noise_pred = unet.forward(
                model_input,
                t as f32,
                context.clone(),
                noise_level,
                &self.device,
            );
            latents = ddim.step(noise_pred, t, latents);
        }
        latents
    }

    /// VAE-decode latents to an image `[1, 3, 4h, 4w]` in `[0, 1]`.
    pub fn decode_latents(&self, latents: Tensor<B, 4>) -> Tensor<B, 4> {
        let decoded = self.vae.forward(latents.div_scalar(SCALING_FACTOR));
        decoded.mul_scalar(0.5).add_scalar(0.5).clamp(0.0, 1.0)
    }

    /// [`denoise`](Self::denoise) then [`decode_latents`](Self::decode_latents).
    pub fn denoise_decode(
        &self,
        low_res01: Tensor<B, 4>,
        noise_level: i64,
        num_steps: usize,
        init_latents: Tensor<B, 4>,
        low_res_noise: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let latents = self.denoise(
            low_res01,
            noise_level,
            num_steps,
            init_latents,
            low_res_noise,
        );
        self.decode_latents(latents)
    }

    /// Upscale an RGBA8 image ×4. `on_progress` receives values in `[0, 1]`.
    ///
    /// Full model: tiled diffusion (each tile drawn with fresh noise). Without a
    /// UNet loaded: nearest-neighbour ×4 placeholder.
    ///
    /// Async because the per-tile GPU→CPU readback must be `into_data_async` —
    /// WASM/WebGPU can't block on the readback future. On native backends the
    /// future resolves immediately; drive it with e.g. `pollster::block_on`.
    pub async fn upscale_rgba(
        &self,
        rgba: &[u8],
        width: usize,
        height: usize,
        opts: &UpscaleOptions,
        on_progress: &mut dyn FnMut(f32),
    ) -> Result<(Vec<u8>, usize, usize), String> {
        if rgba.len() != width * height * 4 {
            return Err(format!("rgba len {} != {width}x{height}x4", rgba.len()));
        }
        if !self.is_full() {
            return Ok(nearest_x4(rgba, width, height, on_progress));
        }

        let full = rgba_to_tensor::<B>(rgba, width, height, &self.device);
        let (ow, oh) = (width * SCALE, height * SCALE);
        let mut out = vec![0f32; ow * oh * 3];
        let mut weight = vec![0f32; ow * oh];

        let tile = opts.tile.clamp(8, width.max(height).max(8));
        let overlap = opts.overlap.min(tile / 2);
        let stride = (tile - overlap).max(1);
        let ys: Vec<usize> = (0..height).step_by(stride).collect();
        let xs: Vec<usize> = (0..width).step_by(stride).collect();
        let total = (ys.len() * xs.len()).max(1);

        // Every tile is exactly `th × tw`: the origin clamps back by a full
        // `tile` at the edges, so the size is uniform (`min(tile, dim)`). That
        // uniformity is what lets us stack tiles on the batch dim and run one
        // forward for a whole group.
        let (th, tw) = (tile.min(height), tile.min(width));
        let origins: Vec<(usize, usize)> = ys
            .iter()
            .flat_map(|&y| {
                let y0 = (y + tile).min(height).saturating_sub(tile);
                xs.iter()
                    .map(move |&x| (y0, (x + tile).min(width).saturating_sub(tile)))
            })
            .collect();
        let batch = opts.batch.max(1);
        let mut done = 0;

        for chunk in origins.chunks(batch) {
            let tiles: Vec<Tensor<B, 4>> = chunk
                .iter()
                .map(|&(y0, x0)| full.clone().narrow(2, y0, th).narrow(3, x0, tw))
                .collect();
            let b = tiles.len();
            let batch_t = Tensor::cat(tiles, 0);
            let init = Tensor::random([b, 4, th, tw], Distribution::Normal(0.0, 1.0), &self.device);
            let lrn = Tensor::random([b, 3, th, tw], Distribution::Normal(0.0, 1.0), &self.device);
            let out_tiles = self.denoise_decode(batch_t, opts.noise_level, opts.steps, init, lrn);

            let [_, _, th4, tw4] = out_tiles.dims();
            let data = out_tiles
                .cast(FloatDType::F32)
                .into_data_async()
                .await
                .map_err(|e| format!("tile readback failed: {e:?}"))?;
            let vals = data
                .as_slice::<f32>()
                .map_err(|e| format!("tile readback: {e:?}"))?;

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

/// [`ModuleMapper`] that casts every float parameter to `dtype`, bringing
/// f32-loaded weights to a half-precision backend's element type.
struct WeightCaster {
    dtype: FloatDType,
}

impl<B: Backend> ModuleMapper<B> for WeightCaster {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        param.map(|tensor| tensor.cast(self.dtype))
    }
}

/// RGBA8 `[h*w*4]` → `[1, 3, h, w]` in `[0, 1]` (drops alpha).
fn rgba_to_tensor<B: Backend>(
    rgba: &[u8],
    width: usize,
    height: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let hw = width * height;
    let mut v = vec![0f32; 3 * hw];
    for i in 0..hw {
        for c in 0..3 {
            v[c * hw + i] = rgba[i * 4 + c] as f32 / 255.0;
        }
    }
    Tensor::from_data(TensorData::new(v, [1, 3, height, width]), device)
}

/// Add a decoded output tile's CHW `vals` (`[3, th, tw]`) into the accumulation
/// buffers at pixel offset `(ox, oy)`, one weight unit per covered pixel.
fn accumulate(
    out: &mut [f32],
    weight: &mut [f32],
    vals: &[f32],
    th: usize,
    tw: usize,
    out_width: usize,
    ox: usize,
    oy: usize,
) {
    let plane = th * tw;
    for ty in 0..th {
        for tx in 0..tw {
            let dst_px = (oy + ty) * out_width + (ox + tx);
            for c in 0..3 {
                out[dst_px * 3 + c] += vals[c * plane + ty * tw + tx];
            }
            weight[dst_px] += 1.0;
        }
    }
}

fn normalize_to_rgba(out: &[f32], weight: &[f32], width: usize, height: usize) -> Vec<u8> {
    let mut rgba = vec![0u8; width * height * 4];
    for px in 0..width * height {
        let w = weight[px].max(1.0);
        for c in 0..3 {
            let v = (out[px * 3 + c] / w).clamp(0.0, 1.0);
            rgba[px * 4 + c] = (v * 255.0 + 0.5) as u8;
        }
        rgba[px * 4 + 3] = 255;
    }
    rgba
}

fn nearest_x4(
    rgba: &[u8],
    width: usize,
    height: usize,
    on_progress: &mut dyn FnMut(f32),
) -> (Vec<u8>, usize, usize) {
    let (ow, oh) = (width * SCALE, height * SCALE);
    let mut out = vec![0u8; ow * oh * 4];
    for y in 0..oh {
        let sy = y / SCALE;
        for x in 0..ow {
            let sx = x / SCALE;
            let si = (sy * width + sx) * 4;
            let di = (y * ow + x) * 4;
            out[di..di + 4].copy_from_slice(&rgba[si..si + 4]);
        }
        on_progress((y + 1) as f32 / oh as f32);
    }
    (out, ow, oh)
}
