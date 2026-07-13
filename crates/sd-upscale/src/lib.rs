//! Stable Diffusion x4 latent-diffusion upscaler, ported to `burn`.
//!
//! The model is written once against `burn::tensor::backend::Backend` and runs
//! on any backend: `NdArray` (CPU) and CUDA for local development and
//! numerical-parity tests against the reference diffusers pipeline, then `wgpu`
//! (WebGPU) for in-browser, on-device inference.
//!
//! See `ROADMAP.md` for architecture and status.

pub mod blocks;
pub mod pipeline;
pub mod scheduler;
pub mod unet;
pub mod vae;
pub mod weights;
