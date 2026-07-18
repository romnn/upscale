//! Building blocks shared across the candle upscaler models: neural-net layers,
//! the DDIM/DDPM scheduler, deterministic host-side noise, and tiled image I/O.

pub(crate) mod blocks;
pub(crate) mod noise;
pub(crate) mod resize;
pub(crate) mod scheduler;
pub(crate) mod tiling;
