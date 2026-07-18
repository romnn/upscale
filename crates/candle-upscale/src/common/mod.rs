//! Building blocks shared across the candle upscaler models: neural-net layers,
//! the DDIM/DDPM scheduler, deterministic host-side noise, and tiled image I/O.

pub(crate) mod blocks;
#[cfg(any(feature = "sdx4", feature = "vosr"))]
pub(crate) mod noise;
#[cfg(any(feature = "tvt", feature = "vosr"))]
pub(crate) mod resize;
#[cfg(feature = "sdx4")]
pub(crate) mod scheduler;
#[cfg(any(feature = "sdx4", feature = "tvt"))]
pub(crate) mod tiling;
