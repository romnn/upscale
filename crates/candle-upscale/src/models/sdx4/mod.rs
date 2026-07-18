//! The Stable Diffusion x4 latent-diffusion upscaler (candle port).

pub mod pipeline;
mod unet;
mod vae;

pub use pipeline::Sdx4;

/// The precomputed empty-prompt CLIP embedding `[1, 77, 1024]` this model
/// conditions on. Fixed for the model release, so it is baked into the binary
/// rather than resolved at runtime.
const EMPTY_PROMPT_EMBED: &[u8] = include_bytes!("../../../assets/empty_prompt_embed.safetensors");

/// Load the SD-x4 model, resolving weight paths from `cfg`.
///
/// The UNet and VAE paths come from `cfg.unet` / `cfg.vae` when set, otherwise
/// from `unet.safetensors` / `vae.safetensors` under `cfg.weights_root`.
///
/// # Errors
/// Fails if a weight file cannot be read or does not match the expected model.
pub fn load(cfg: &crate::model::LoadConfig) -> candle_core::Result<Sdx4> {
    let unet = cfg
        .unet
        .clone()
        .unwrap_or_else(|| cfg.weights_root.join("unet.safetensors"));
    let vae = cfg
        .vae
        .clone()
        .unwrap_or_else(|| cfg.weights_root.join("vae.safetensors"));
    Sdx4::load(
        &unet,
        &vae,
        EMPTY_PROMPT_EMBED,
        cfg.device.clone(),
        cfg.dtype,
    )
}
