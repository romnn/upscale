//! Native CLI for the SD x4 upscaler.
//!
//! Drives the exact same [`sd_upscale::pipeline::Upscaler`] the browser uses, on
//! a real GPU, so it doubles as a benchmark: wgpu by default (`--backend wgpu`),
//! or CUDA when built with `--features cuda` (`--backend cuda`). The pipeline is
//! generic over [`burn::tensor::backend::Backend`], so [`run`] is shared across
//! backends and only the device construction differs.
//!
//! ```text
//! cargo run --release -p cli -- -i in.png -o out.png --steps 20
//! cargo run --release -p cli --features cuda -- -i in.png --backend cuda
//! ```

use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context};
use burn::tensor::backend::Backend;
use burn::tensor::FloatDType;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use sd_upscale::pipeline::{UpscaleOptions, Upscaler};

/// The precomputed empty-prompt CLIP embedding `[1, 77, 1024]`. Fixed for this
/// model release, so it's baked in rather than passed on the command line.
const EMPTY_PROMPT_EMBED: &[u8] =
    include_bytes!("../../sd-upscale/assets/empty_prompt_embed.safetensors");

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum BackendChoice {
    /// Use CUDA when this binary was built with `--features cuda` and a device is
    /// present, otherwise wgpu.
    Auto,
    /// WebGPU / wgpu (portable; the browser build's backend). Always available.
    Wgpu,
    /// NVIDIA CUDA. Requires a build with `--features cuda`.
    Cuda,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum DtypeChoice {
    /// Full fp32 compute — matches the browser/reference output exactly.
    F32,
    /// bf16 compute — uses tensor cores for a large speedup, at some precision
    /// loss. bf16 (not fp16) keeps fp32's exponent range, which the x4-upscaler
    /// VAE needs to avoid overflow. Backend support varies (works on CUDA; wgpu
    /// may lack it).
    Bf16,
}

/// Output upscale factor.
///
/// The model is a native ×4 upscaler, so a ×2 result is produced by
/// pre-downsampling the input rather than by a different model.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum ScaleChoice {
    /// Downsample the input ×2, then run the ×4 model — a ×2 result on a quarter
    /// of the pixels (≈4× faster), at some loss of fine input detail.
    #[value(name = "2")]
    Two,
    /// The model's native ×4 upscale (full detail, full cost).
    #[value(name = "4")]
    Four,
}

#[derive(Parser)]
#[command(about = "SD x4 latent-diffusion image upscaler (native).")]
struct Args {
    /// Input image (PNG or JPEG).
    #[arg(short, long)]
    input: PathBuf,
    /// Output PNG path.
    #[arg(short, long, default_value = "upscaled.png")]
    output: PathBuf,
    /// Compute backend. `auto` (default) prefers CUDA when available, else wgpu;
    /// `cuda` requires a build with `--features cuda`.
    #[arg(long, value_enum, default_value = "auto")]
    backend: BackendChoice,
    /// Compute precision. Defaults to `bf16` on CUDA (tensor cores, much faster,
    /// tiny accuracy cost) and `f32` elsewhere (exact); `bf16` may be unsupported
    /// on wgpu.
    #[arg(long, value_enum)]
    dtype: Option<DtypeChoice>,
    /// Output upscale factor. `2` (default) downsamples the input first for a ≈4×
    /// speedup at some detail loss; `4` is the model's native full-detail scale.
    #[arg(long, value_enum, default_value = "2")]
    scale: ScaleChoice,
    /// Directory holding `unet.safetensors` / `vae.safetensors` (and `.fp16`
    /// variants). Ignored for a part when its explicit `--unet` / `--vae` is set.
    #[arg(long, default_value = "crates/web/models")]
    models_dir: PathBuf,
    /// Explicit UNet safetensors path (overrides `--models-dir`).
    #[arg(long)]
    unet: Option<PathBuf>,
    /// Explicit VAE safetensors path (overrides `--models-dir`).
    #[arg(long)]
    vae: Option<PathBuf>,
    /// Use the `*.fp16.safetensors` weights (up-converted to f32 on load, so the
    /// output is identical; halves the bytes read).
    #[arg(long)]
    fp16: bool,
    /// DDIM denoising steps (more = slower, sharper).
    #[arg(long, default_value_t = 20)]
    steps: usize,
    /// Low-res conditioning noise level (`0..=350`; lower = more faithful).
    #[arg(long, default_value_t = 20)]
    noise_level: i64,
    /// Tile size in pixels (the model is trained around 128).
    #[arg(long, default_value_t = 128)]
    tile: usize,
    /// Tile overlap in pixels (blended to hide seams).
    #[arg(long, default_value_t = 16)]
    overlap: usize,
    /// Tiles per UNet/VAE forward. Higher fills the GPU for a large speedup but
    /// needs proportionally more free VRAM. If the CUDA backend can't grow its
    /// memory pool it prints `Memory page` panics on stderr and the output may
    /// be corrupt — lower `--batch` (or free GPU memory) if you see them.
    #[arg(long, default_value_t = 1)]
    batch: usize,
    /// Center-crop the input to this many pixels before upscaling (handy for a
    /// quick single-tile benchmark).
    #[arg(long)]
    crop: Option<u32>,
    /// Suppress the progress bar and all informational logging, for programmatic
    /// use. Errors are still reported (and set a non-zero exit code).
    #[arg(short, long)]
    quiet: bool,
}

impl Args {
    /// Resolve the UNet and VAE safetensors paths from the explicit overrides or,
    /// failing those, `--models-dir` plus the fp16/f32 suffix.
    fn model_paths(&self) -> (PathBuf, PathBuf) {
        let suffix = if self.fp16 {
            "fp16.safetensors"
        } else {
            "safetensors"
        };
        let unet = self
            .unet
            .clone()
            .unwrap_or_else(|| self.models_dir.join(format!("unet.{suffix}")));
        let vae = self
            .vae
            .clone()
            .unwrap_or_else(|| self.models_dir.join(format!("vae.{suffix}")));
        (unet, vae)
    }
}

fn center_crop(img: image::RgbaImage, crop: u32) -> image::RgbaImage {
    let (iw, ih) = (img.width(), img.height());
    let (cw, ch) = (crop.min(iw), crop.min(ih));
    let (cx, cy) = ((iw - cw) / 2, (ih - ch) / 2);
    image::imageops::crop_imm(&img, cx, cy, cw, ch).to_image()
}

/// Resolution of the tile-progress bar. The pipeline reports a `[0, 1]` fraction
/// (it owns the tile count), so we map that onto a fixed length for indicatif to
/// render the bar and derive an ETA from.
const PROGRESS_LEN: u64 = 1000;

/// Build the tile-progress bar, or `None` in `--quiet` mode.
///
/// Drawn to stderr, so stdout stays free for programmatic use. The steady tick
/// keeps the elapsed timer and bar refreshing during the slow first tile (kernel
/// compile / autotune), which otherwise looks hung.
fn progress_bar(quiet: bool) -> anyhow::Result<Option<ProgressBar>> {
    if quiet {
        return Ok(None);
    }
    let pb = ProgressBar::new(PROGRESS_LEN);
    pb.set_style(
        ProgressStyle::with_template(
            "  {bar:40.cyan/blue} {percent:>3}%  [{elapsed_precise}] eta {eta}",
        )
        .context("build progress-bar template")?
        .progress_chars("=>-"),
    );
    pb.enable_steady_tick(Duration::from_millis(120));
    Ok(Some(pb))
}

/// Backend-generic pipeline run: load the models, upscale the input, write the
/// output.
///
/// Shared by every backend; only `device` and `weight_dtype` differ.
/// `weight_dtype` must match `B`'s float element (e.g. `bf16` for `Cuda<bf16>`):
/// weights load as f32, so they are cast to the backend's element and the whole
/// graph runs at one precision.
fn run<B: Backend>(device: B::Device, args: &Args, weight_dtype: FloatDType) -> anyhow::Result<()> {
    let quiet = args.quiet;
    let (unet_path, vae_path) = args.model_paths();
    if !quiet {
        eprintln!(
            "loading models ({}): {} + {}",
            if args.fp16 { "fp16→f32" } else { "f32" },
            unet_path.display(),
            vae_path.display(),
        );
    }
    let t = Instant::now();
    let unet_bytes =
        fs::read(&unet_path).with_context(|| format!("read UNet {}", unet_path.display()))?;
    let vae_bytes =
        fs::read(&vae_path).with_context(|| format!("read VAE {}", vae_path.display()))?;
    let up = Upscaler::<B>::load_full(unet_bytes, vae_bytes, EMPTY_PROMPT_EMBED, args.fp16, device)
        .map_err(|e| anyhow!("load models: {e}"))?
        .cast_weights(weight_dtype);
    if !quiet {
        eprintln!("  loaded in {:.1}s", t.elapsed().as_secs_f32());
    }

    let img = image::open(&args.input)
        .with_context(|| format!("open input {}", args.input.display()))?
        .to_rgba8();
    let img = match args.crop {
        Some(c) => center_crop(img, c),
        None => img,
    };
    // For ×2 output, downsample the input ×2 up front so the native ×4 pass lands
    // at ×2 of the original while processing a quarter of the pixels.
    let img = match args.scale {
        ScaleChoice::Four => img,
        ScaleChoice::Two => {
            let (iw, ih) = (img.width(), img.height());
            let (dw, dh) = ((iw / 2).max(1), (ih / 2).max(1));
            if !quiet {
                eprintln!("×2 mode: downsampling input {iw}×{ih} → {dw}×{dh} before the ×4 pass");
            }
            image::imageops::resize(&img, dw, dh, image::imageops::FilterType::Lanczos3)
        }
    };
    let (w, h) = (img.width() as usize, img.height() as usize);
    let rgba = img.into_raw();

    let opts = UpscaleOptions {
        steps: args.steps,
        noise_level: args.noise_level,
        tile: args.tile,
        overlap: args.overlap,
        batch: args.batch,
    };
    let prec = if matches!(weight_dtype, FloatDType::BF16) {
        "bf16"
    } else {
        "f32"
    };
    if !quiet {
        eprintln!(
            "upscaling {w}x{h} → {}x{} ({} steps, noise {}, {prec})…",
            w * 4,
            h * 4,
            args.steps,
            args.noise_level,
        );
    }

    let pb = progress_bar(quiet)?;
    let t = Instant::now();
    let (out, ow, oh) = pollster::block_on(up.upscale_rgba(&rgba, w, h, &opts, &mut |p| {
        if let Some(pb) = &pb {
            pb.set_position((p * PROGRESS_LEN as f32).round() as u64);
        }
    }))
    .map_err(|e| anyhow!("upscale: {e}"))?;
    if let Some(pb) = pb {
        pb.finish_and_clear();
    }
    if !quiet {
        eprintln!("  done in {:.1}s", t.elapsed().as_secs_f32());
    }

    image::RgbaImage::from_raw(ow as u32, oh as u32, out)
        .context("assemble output image")?
        .save(&args.output)
        .with_context(|| format!("write output {}", args.output.display()))?;
    if !quiet {
        eprintln!("wrote {} ({ow}×{oh})", args.output.display());
    }
    Ok(())
}

fn run_wgpu(args: &Args, dtype: DtypeChoice) -> anyhow::Result<()> {
    let device = burn::backend::wgpu::WgpuDevice::default();
    match dtype {
        DtypeChoice::F32 => run::<burn::backend::Wgpu>(device, args, FloatDType::F32),
        DtypeChoice::Bf16 => {
            run::<burn::backend::Wgpu<burn::tensor::bf16>>(device, args, FloatDType::BF16)
        }
    }
}

/// Highest-versioned `MAJOR.MINOR` toolkit subdirectory of `base` (for installs
/// that keep several toolkits under one prefix, e.g. `/usr/local/cuda/13.2`).
#[cfg(feature = "cuda")]
fn newest_versioned_toolkit(base: &std::path::Path) -> Option<PathBuf> {
    let Ok(entries) = std::fs::read_dir(base) else {
        return None;
    };
    let mut best: Option<((u32, u32), PathBuf)> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let mut parts = name.splitn(2, '.');
        let (Some(maj), Some(min)) = (parts.next(), parts.next()) else {
            continue;
        };
        let (Ok(maj), Ok(min)) = (maj.parse::<u32>(), min.parse::<u32>()) else {
            continue;
        };
        if best.as_ref().is_none_or(|(v, _)| (maj, min) > *v) {
            best = Some(((maj, min), path));
        }
    }
    best.map(|(_, p)| p)
}

/// Ensure `CUDA_PATH` points at a CUDA toolkit whose `include/cuda_runtime.h`
/// exists.
///
/// `CUDA_PATH` is the only variable cubecl-cuda reads to give NVRTC the headers
/// its generated kernels `#include`. When it is unset, cubecl falls back to a
/// bare `/usr/local/cuda`; on installs where that path is a directory of
/// versioned toolkits rather than the usual symlink, the fallback's include
/// directory is missing and every kernel fails to compile — which surfaces not
/// as a startup error but as a mid-run hang.
///
/// Deriving a valid root here (from the other conventional CUDA env vars, then a
/// probe) turns that trap into a working run, or a clear warning if no toolkit
/// is found.
#[cfg(feature = "cuda")]
fn ensure_cuda_path(quiet: bool) {
    use std::path::Path;

    fn has_headers(root: &Path) -> bool {
        root.join("include").join("cuda_runtime.h").is_file()
    }

    if let Some(cur) = std::env::var_os("CUDA_PATH") {
        if !cur.is_empty() && has_headers(Path::new(&cur)) {
            return;
        }
    }

    let mut candidates: Vec<PathBuf> = ["CUDA_HOME", "CUDA_INSTALL_PATH", "CUDA_ROOT"]
        .into_iter()
        .filter_map(|v| std::env::var_os(v).map(PathBuf::from))
        .collect();
    for base in ["/usr/local/cuda", "/opt/cuda"] {
        candidates.push(PathBuf::from(base));
        candidates.extend(newest_versioned_toolkit(Path::new(base)));
    }

    if let Some(root) = candidates.into_iter().find(|c| has_headers(c)) {
        std::env::set_var("CUDA_PATH", &root);
        if !quiet {
            eprintln!(
                "note: CUDA_PATH was unset/invalid; using {} so NVRTC can find cuda_runtime.h",
                root.display(),
            );
        }
    } else if !quiet {
        eprintln!(
            "warning: no CUDA toolkit with include/cuda_runtime.h found; set CUDA_PATH to your \
             toolkit root or the CUDA backend will hang on the first kernel compile"
        );
    }
}

#[cfg(feature = "cuda")]
fn run_cuda(args: &Args, dtype: DtypeChoice) -> anyhow::Result<()> {
    ensure_cuda_path(args.quiet);
    let device = burn::backend::cuda::CudaDevice::default();
    match dtype {
        DtypeChoice::F32 => run::<burn::backend::Cuda>(device, args, FloatDType::F32),
        DtypeChoice::Bf16 => {
            run::<burn::backend::Cuda<burn::tensor::bf16>>(device, args, FloatDType::BF16)
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn run_cuda(_args: &Args, _dtype: DtypeChoice) -> anyhow::Result<()> {
    anyhow::bail!("this binary was built without CUDA support; rebuild with `--features cuda`")
}

/// Whether a usable CUDA device is present. Always `false` without the `cuda`
/// feature, so `--backend auto` falls back to wgpu on non-CUDA builds.
#[cfg(feature = "cuda")]
fn cuda_available() -> bool {
    matches!(cudarc::driver::CudaContext::device_count(), Ok(n) if n > 0)
}

#[cfg(not(feature = "cuda"))]
fn cuda_available() -> bool {
    false
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    // Resolve `auto`: prefer CUDA when this build can use it and a device exists.
    let use_cuda = match args.backend {
        BackendChoice::Cuda => true,
        BackendChoice::Wgpu => false,
        BackendChoice::Auto => cuda_available(),
    };
    // Precision defaults per backend: bf16 on CUDA (tensor cores), f32 on wgpu.
    let dtype = args.dtype.unwrap_or(if use_cuda {
        DtypeChoice::Bf16
    } else {
        DtypeChoice::F32
    });
    // WebGPU has no bf16 type, so cubecl-wgpu's WGSL compiler panics on it.
    // Reject the combination up front instead of crashing mid-run.
    if !use_cuda && matches!(dtype, DtypeChoice::Bf16) {
        anyhow::bail!(
            "bf16 is unsupported on the wgpu backend (WebGPU has no bf16 type); \
             use --dtype f32, or --backend cuda"
        );
    }
    if !args.quiet {
        eprintln!("backend: {}", if use_cuda { "cuda" } else { "wgpu" });
    }
    if use_cuda {
        run_cuda(&args, dtype)
    } else {
        run_wgpu(&args, dtype)
    }
}
