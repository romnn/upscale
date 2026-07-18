//! A multi-model candle image upscaler.
//!
//! Exposes a small, model-agnostic surface — the [`UpscaleModel`] trait, the
//! [`UpscaleOptions`] tunables, and a [`load_model`] registry keyed by
//! [`ModelId`] / [`Preset`] — so a caller (the CLI, the browser) picks a model by
//! intent and drives it uniformly. Each concrete model lives under [`models`] and
//! is feature-gated; shared neural-net blocks, the diffusion scheduler,
//! deterministic noise, and tiled I/O live in a private `common` module.
//!
//! The first model is the Stable Diffusion x4 latent-diffusion upscaler
//! ([`Sdx4`], feature `sdx4`), a candle port of the burn reference in
//! `sd-upscale`. The safetensors are loaded unchanged: candle stores `Linear`
//! weight as `[out, in]` (PyTorch layout) and conv weight as `[out, in, kh, kw]`,
//! so — unlike the burn port — no weight transpose is needed, and
//! GroupNorm/LayerNorm keep their PyTorch `weight`/`bias` names.
//!
//! Build with `cuda` (or `metal`) for GPU execution; without a GPU feature candle
//! falls back to its CPU backend, which keeps the crate compiling toolkit-free
//! for lint/CI (the models run, just slowly).

mod common;
pub mod device;
pub mod model;
pub mod models;

pub use candle_core::{DType, Device};
pub use device::select_device;
pub use model::{load_model, LoadConfig, ModelId, Preset, UpscaleModel, UpscaleOptions};

#[cfg(feature = "cuda")]
pub use device::cuda_device;
#[cfg(feature = "sdx4")]
pub use models::sdx4::Sdx4;
#[cfg(feature = "tvt")]
pub use models::tvt::Tvt;
#[cfg(feature = "vosr")]
pub use models::vosr::Vosr;
