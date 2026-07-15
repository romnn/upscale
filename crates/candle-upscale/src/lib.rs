//! CUDA-native candle port of the Stable Diffusion x4 latent-diffusion upscaler.
//!
//! This is a second implementation of the model that lives in
//! `sd-upscale` (the burn reference), built to measure whether candle is
//! substantially faster than burn on CUDA with matching output. The module
//! layout mirrors the reference: [`blocks`]-style building blocks, the UNet, the
//! VAE decoder, the DDIM/DDPM [`scheduler`], and the tiled [`pipeline`].
//!
//! The safetensors are loaded unchanged: candle stores `Linear` weight as
//! `[out, in]` (PyTorch layout) and conv weight as `[out, in, kh, kw]`, so —
//! unlike the burn port — no weight transpose is needed, and GroupNorm/LayerNorm
//! keep their PyTorch `weight`/`bias` names.
//!
//! Build with the default `cuda` feature for GPU execution; without it candle
//! falls back to its CPU backend, which keeps the crate compiling toolkit-free
//! for lint/CI (the model runs, just slowly).

mod blocks;
pub mod noise;
mod scheduler;
mod unet;
mod vae;

pub mod pipeline;

pub use candle_core::{DType, Device};
pub use pipeline::{UpscaleOptions, Upscaler};

/// Open CUDA device 0. Only available with the `cuda` feature; the candle engine
/// is CUDA-native by design.
///
/// # Errors
/// Fails if no CUDA device is present or the driver cannot be initialized.
#[cfg(feature = "cuda")]
pub fn cuda_device() -> candle_core::Result<Device> {
    Device::new_cuda(0)
}
