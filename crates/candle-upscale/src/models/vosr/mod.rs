//! The VOSR super-resolution model (candle port).
//!
//! A LightningDiT flow-matching upscaler: a bicubic ×4 low-res image is encoded
//! by the SD2.1 VAE, a CFG flow-matching loop denoises the latent conditioned on
//! DINOv2 layer-8 features, and a distilled LightDecoder decodes the result. See
//! [`pipeline`] for the end-to-end flow.

mod attention;
mod dino;
mod dit;
pub mod pipeline;
mod profile;
mod vae;

pub use pipeline::Vosr;

/// DiT (EMA) weights, relative to `weights_root`.
const DIT_REL: &str = "VOSR_0.5B_ms/checkpoints/ema_model.safetensors";
/// SD2.1 VAE (encoder) weights, relative to `weights_root`.
const VAE_REL: &str = "stable-diffusion-2-1-base/vae/diffusion_pytorch_model.safetensors";
/// LightDecoder weights, relative to `weights_root`.
const DECODER_REL: &str = "sd21_lwdecoder.pth";
/// DINOv2 ViT-B/14 backbone weights, relative to `weights_root`.
const DINO_REL: &str = "torch_cache/checkpoints/dinov2_vitb14_pretrain.pth";

/// Load the VOSR model, resolving weight paths from `cfg`.
///
/// The DiT and VAE paths come from `cfg.unet` / `cfg.vae` when set, otherwise
/// from the fixed sub-paths under `cfg.weights_root`; the LightDecoder and
/// DINOv2 backbone are always resolved under `cfg.weights_root`.
///
/// # Errors
/// Fails if a weight file cannot be read or does not match the expected model.
pub fn load(cfg: &crate::model::LoadConfig) -> candle_core::Result<Vosr> {
    let root = &cfg.weights_root;
    let dit = cfg.unet.clone().unwrap_or_else(|| root.join(DIT_REL));
    let vae = cfg.vae.clone().unwrap_or_else(|| root.join(VAE_REL));
    Vosr::load(
        &dit,
        &vae,
        &root.join(DECODER_REL),
        &root.join(DINO_REL),
        cfg.device.clone(),
        cfg.dtype,
    )
}
