//! The TVT super-resolution model (candle port).
//!
//! A one-step latent upscaler: the low-res image is bicubically upscaled ×4,
//! encoded by a 4× VAE (VAE-D4), refined by a single SD2.1 UNet forward, and
//! decoded back to the ×4 image. The trained LoRA deltas are fused into the base
//! UNet offline, so this module loads plain diffusers weights. See [`pipeline`]
//! for the end-to-end flow.

mod pipeline;
mod unet;
mod vae4x;

pub use pipeline::Tvt;

/// Fused UNet weights (base SD2.1 UNet + merged TVT LoRA), relative to
/// `weights_root`. Produced by the offline preprocessing step.
const UNET_REL: &str = "ckp/fused_unet.safetensors";
/// VAE-D4 weights (encoder + decoder + quant convs), relative to `weights_root`.
const VAE_REL: &str = "ckp/vae.safetensors";

/// The precomputed prompt CLIP embedding `[1, 77, 1024]` this model conditions
/// on. Fixed for the model release, so it is baked into the binary rather than
/// resolved at runtime.
const PROMPT_EMBED: &[u8] = include_bytes!("../../../assets/tvt_prompt_embed.safetensors");

/// Load the TVT model, resolving weight paths from `cfg`.
///
/// The UNet and VAE paths come from `cfg.unet` / `cfg.vae` when set, otherwise
/// from the fixed sub-paths under `cfg.weights_root`.
///
/// # Errors
/// Fails if a weight file cannot be read or does not match the expected model.
pub fn load(cfg: &crate::model::LoadConfig) -> candle_core::Result<Tvt> {
    let root = &cfg.weights_root;
    let unet = cfg.unet.clone().unwrap_or_else(|| root.join(UNET_REL));
    let vae = cfg.vae.clone().unwrap_or_else(|| root.join(VAE_REL));
    Tvt::load(&unet, &vae, PROMPT_EMBED, cfg.device.clone(), cfg.dtype)
}
