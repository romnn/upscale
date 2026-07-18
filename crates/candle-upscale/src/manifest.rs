//! Static description of each model's downloadable weight files.
//!
//! HTTP-free by design: this module only NAMES the upstream URL and the
//! cache-relative destination for each file. A caller (the CLI) owns the actual
//! downloading, so the library keeps no network or progress-bar dependency.
//! [`ModelId::manifest`] returns `None` for models whose weights are derived
//! locally and cannot be fetched.

use crate::model::ModelId;

/// One downloadable weight file: where to fetch it and where it lands under the
/// model's `weights_root`.
#[derive(Clone, Copy, Debug)]
pub struct WeightFile {
    /// The `https://` URL the file is fetched from — a Hugging Face
    /// `resolve/main` link that redirects to the LFS/CDN blob.
    pub url: &'static str,
    /// Destination path relative to the model's `weights_root`. Equals the exact
    /// sub-path the model's `load()` resolves, so a fetched cache is a drop-in
    /// `--models-dir`.
    pub dest: &'static str,
}

/// SD-x4 upscaler weights, from the `stabilityai/stable-diffusion-x4-upscaler`
/// diffusers repo. These are the fp32 files the dev default (`crates/web/models`)
/// symlinks to, so a fetched cache reproduces its output byte-for-byte. The
/// dests match [`sdx4::load`](crate::models::sdx4)'s `unet.safetensors` /
/// `vae.safetensors`.
const SDX4: &[WeightFile] = &[
    WeightFile {
        url: "https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/resolve/main/unet/diffusion_pytorch_model.safetensors",
        dest: "unet.safetensors",
    },
    WeightFile {
        url: "https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/resolve/main/vae/diffusion_pytorch_model.safetensors",
        dest: "vae.safetensors",
    },
];

/// VOSR weights, from the `CSWRY/VOSR` release repo. Its tree mirrors the
/// sub-paths [`vosr::load`](crate::models::vosr) resolves under `weights_root`,
/// so each `dest` equals the upstream repo path (keep these in sync with vosr's
/// `DIT_REL` / `VAE_REL` / `DECODER_REL` / `DINO_REL`). `args.json` is not read
/// at runtime — its values are baked into the port — but is fetched so the cache
/// is a faithful mirror of the published checkpoint.
const VOSR: &[WeightFile] = &[
    WeightFile {
        url: "https://huggingface.co/CSWRY/VOSR/resolve/main/VOSR_0.5B_ms/checkpoints/ema_model.safetensors",
        dest: "VOSR_0.5B_ms/checkpoints/ema_model.safetensors",
    },
    WeightFile {
        url: "https://huggingface.co/CSWRY/VOSR/resolve/main/stable-diffusion-2-1-base/vae/diffusion_pytorch_model.safetensors",
        dest: "stable-diffusion-2-1-base/vae/diffusion_pytorch_model.safetensors",
    },
    WeightFile {
        url: "https://huggingface.co/CSWRY/VOSR/resolve/main/sd21_lwdecoder.pth",
        dest: "sd21_lwdecoder.pth",
    },
    WeightFile {
        url: "https://huggingface.co/CSWRY/VOSR/resolve/main/torch_cache/checkpoints/dinov2_vitb14_pretrain.pth",
        dest: "torch_cache/checkpoints/dinov2_vitb14_pretrain.pth",
    },
    WeightFile {
        url: "https://huggingface.co/CSWRY/VOSR/resolve/main/VOSR_0.5B_ms/args.json",
        dest: "VOSR_0.5B_ms/args.json",
    },
];

/// TVT's runtime weights are pre-fused/converted (the TVT LoRA merged into the
/// SD2.1 UNet, plus the VAE-D4 as a plain safetensors) and hosted ready-to-load;
/// the dests match [`tvt::load`](crate::models::tvt)'s `ckp/…` sub-paths.
const TVT: &[WeightFile] = &[
    WeightFile {
        url: "https://huggingface.co/romnnn/tvtsr-fused/resolve/main/fused_unet.safetensors",
        dest: "ckp/fused_unet.safetensors",
    },
    WeightFile {
        url: "https://huggingface.co/romnnn/tvtsr-fused/resolve/main/vae.safetensors",
        dest: "ckp/vae.safetensors",
    },
];

impl ModelId {
    /// The set of weight files this model downloads on first run, or `None` for a
    /// model whose weights are derived locally and not published (the CLI then
    /// asks for `--models-dir`).
    ///
    /// The returned dests are relative to the model's `weights_root`, so writing
    /// each file to `<cache>/<dest>` yields a directory usable as `--models-dir`.
    pub fn manifest(self) -> Option<&'static [WeightFile]> {
        match self {
            ModelId::Sdx4 => Some(SDX4),
            ModelId::Vosr => Some(VOSR),
            ModelId::Tvt => Some(TVT),
        }
    }
}
