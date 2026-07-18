//! The model-agnostic interface every candle upscaler implements, plus the
//! registry that maps a [`ModelId`] / [`Preset`] to a loaded model.

use std::path::PathBuf;

use candle_core::{DType, Device, Result};

/// Tunables shared by every model's tiled run. Individual models read only the
/// fields they use (e.g. a one-step model ignores `steps`).
#[derive(Clone, Debug)]
pub struct UpscaleOptions {
    /// Sampler steps (more = slower; model-dependent quality trade-off).
    pub steps: usize,
    /// Low-res conditioning noise level `0..=350`. Lower = more faithful.
    pub noise_level: i64,
    /// Low-res tile size in pixels.
    pub tile: usize,
    /// Tile overlap in pixels (blended to hide seams).
    pub overlap: usize,
    /// Tiles processed per model forward (stacked on the batch dim).
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

/// A loaded upscaling model. `native_scale` is the fixed integer factor the model
/// produces before the CLI's pre/post shrink.
pub trait UpscaleModel {
    /// The fixed integer upscale factor this model produces (e.g. `4` for SD-x4).
    fn native_scale(&self) -> usize;

    /// Upscale an RGBA8 image by [`native_scale`](Self::native_scale).
    ///
    /// `rgba` is `width`×`height`×4 bytes; `on_progress` receives a `[0, 1]`
    /// completion fraction; `seed` makes the per-tile noise reproducible. Returns
    /// the upscaled RGBA8 buffer with its new width and height.
    ///
    /// # Errors
    /// Fails if the input length does not match `width`×`height`×4 or if a tensor
    /// operation fails.
    fn upscale_rgba(
        &self,
        rgba: &[u8],
        width: usize,
        height: usize,
        opts: &UpscaleOptions,
        seed: u64,
        on_progress: &mut dyn FnMut(f32),
    ) -> Result<(Vec<u8>, usize, usize)>;
}

/// Which model to load. Stable regardless of enabled features (loading a model
/// whose feature is off returns an error, so callers / CLI parsing never change).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelId {
    /// The Stable Diffusion x4 latent-diffusion upscaler.
    Sdx4,
    /// The VOSR flow-matching latent upscaler.
    Vosr,
    /// The TVT one-step latent upscaler.
    Tvt,
}

impl ModelId {
    /// The CLI's sampler-step default for this model.
    pub fn default_steps(self) -> usize {
        match self {
            Self::Sdx4 => 20,
            Self::Vosr => 4,
            Self::Tvt => 1,
        }
    }

    /// The CLI's model-forward tile-batch default for this model.
    pub fn default_batch(self) -> usize {
        match self {
            Self::Vosr => 8,
            Self::Sdx4 | Self::Tvt => 1,
        }
    }
}

/// A user intent that maps to a concrete model. The mapping is PROVISIONAL, to be
/// finalized after A/B testing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    /// Optimize for documents (text, line art, screenshots).
    Document,
    /// Optimize for photographic / natural images.
    Image,
}

impl Preset {
    /// The concrete model this preset selects.
    // TODO(after testing): both presets map to `Sdx4` for now; revisit once VOSR
    // and TVT land and the A/B comparison picks a per-preset winner.
    pub fn model(self) -> ModelId {
        match self {
            Preset::Document | Preset::Image => ModelId::Sdx4,
        }
    }
}

/// Everything a model needs to load its weights.
pub struct LoadConfig {
    /// Device the model's tensors live on.
    pub device: Device,
    /// Compute dtype the weights are cast to on load (`F32` or `BF16`).
    pub dtype: DType,
    /// Directory holding the model's default weight files.
    pub weights_root: PathBuf,
    /// Explicit UNet weights path (overrides `weights_root`), if any.
    pub unet: Option<PathBuf>,
    /// Explicit VAE weights path (overrides `weights_root`), if any.
    pub vae: Option<PathBuf>,
}

/// Load `id` into a boxed trait object.
///
/// # Errors
/// Returns an error if the model's feature is disabled in this build, if the
/// model is not yet implemented, or if loading its weights fails.
pub fn load_model(id: ModelId, cfg: &LoadConfig) -> Result<Box<dyn UpscaleModel>> {
    match id {
        ModelId::Sdx4 => load_sdx4(cfg),
        ModelId::Vosr => load_vosr(cfg),
        ModelId::Tvt => load_tvt(cfg),
    }
}

#[cfg(feature = "sdx4")]
fn load_sdx4(cfg: &LoadConfig) -> Result<Box<dyn UpscaleModel>> {
    Ok(Box::new(crate::models::sdx4::load(cfg)?))
}

#[cfg(not(feature = "sdx4"))]
fn load_sdx4(_cfg: &LoadConfig) -> Result<Box<dyn UpscaleModel>> {
    Err(candle_core::Error::Msg(
        "built without the sdx4 model (rebuild with --features model-sdx4)".into(),
    ))
}

#[cfg(feature = "vosr")]
fn load_vosr(cfg: &LoadConfig) -> Result<Box<dyn UpscaleModel>> {
    Ok(Box::new(crate::models::vosr::load(cfg)?))
}

#[cfg(not(feature = "vosr"))]
fn load_vosr(_cfg: &LoadConfig) -> Result<Box<dyn UpscaleModel>> {
    Err(candle_core::Error::Msg(
        "built without the vosr model (rebuild with --features model-vosr)".into(),
    ))
}

#[cfg(feature = "tvt")]
fn load_tvt(cfg: &LoadConfig) -> Result<Box<dyn UpscaleModel>> {
    Ok(Box::new(crate::models::tvt::load(cfg)?))
}

#[cfg(not(feature = "tvt"))]
fn load_tvt(_cfg: &LoadConfig) -> Result<Box<dyn UpscaleModel>> {
    Err(candle_core::Error::Msg(
        "built without the tvt model (rebuild with --features model-tvt)".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::ModelId;

    #[test]
    fn vosr_defaults_to_its_converged_step_count_and_batched_tiles() {
        assert_eq!(ModelId::Vosr.default_steps(), 4);
        assert_eq!(ModelId::Vosr.default_batch(), 8);
    }

    #[test]
    fn sdx4_retains_its_existing_defaults() {
        assert_eq!(ModelId::Sdx4.default_steps(), 20);
        assert_eq!(ModelId::Sdx4.default_batch(), 1);
    }
}
