//! Native, candle-only multi-model image upscaler.
//!
//! Loads a model by `--model` (or a `--preset` intent) and runs it on the
//! fastest available device (CUDA, else Metal, else CPU), upscaling a PNG/JPEG
//! with seam-blended tiling.
//!
//! ```text
//! cargo run --release -p cli -- -i in.png -o out.png --steps 20
//! cargo run --release -p cli --features cuda -- -i in.png --model sdx4
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

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

impl DtypeChoice {
    /// Short label for logs (`f32` / `bf16`).
    fn as_str(self) -> &'static str {
        match self {
            DtypeChoice::F32 => "f32",
            DtypeChoice::Bf16 => "bf16",
        }
    }
}

/// Which model runs the upscale. Maps to [`candle_upscale::ModelId`].
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum ModelChoice {
    /// Stable Diffusion x4 latent-diffusion upscaler.
    Sdx4,
    /// VOSR flow-matching latent upscaler.
    Vosr,
    /// TVT one-step latent upscaler.
    Tvt,
}

impl ModelChoice {
    /// The library model id this choice selects.
    fn model_id(self) -> candle_upscale::ModelId {
        match self {
            ModelChoice::Sdx4 => candle_upscale::ModelId::Sdx4,
            ModelChoice::Vosr => candle_upscale::ModelId::Vosr,
            ModelChoice::Tvt => candle_upscale::ModelId::Tvt,
        }
    }
}

/// A user-intent preset that maps to a model. Maps to [`candle_upscale::Preset`].
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum PresetChoice {
    /// Optimize for documents (text, line art, screenshots).
    Document,
    /// Optimize for photographic / natural images.
    Image,
}

impl PresetChoice {
    /// The library preset this choice selects.
    fn preset(self) -> candle_upscale::Preset {
        match self {
            PresetChoice::Document => candle_upscale::Preset::Document,
            PresetChoice::Image => candle_upscale::Preset::Image,
        }
    }
}

/// The model's fixed upscale factor. Baked into the VAE decoder's upsampling
/// stack (`[b,4,h,w]` latent → `[b,3,4h,4w]` image), so it is not a tunable — a
/// smaller net `--scale` is only reachable by shrinking around this pass.
const NATIVE_SCALE: f32 = 4.0;

/// Fixed seed for the per-tile noise, so a run is reproducible.
const SEED: u64 = 0x5D5C_A1E0;

#[derive(Parser)]
#[command(about = "Multi-model image upscaler (candle).")]
struct Args {
    /// Input image (PNG or JPEG).
    #[arg(short, long)]
    input: PathBuf,
    /// Output path. PNG or JPEG, chosen by the file extension.
    #[arg(short, long, default_value = "upscaled.png")]
    output: PathBuf,
    /// Model to run (mutually exclusive with `--preset`). With neither set, the
    /// `document` preset's model is used.
    #[arg(long, value_enum)]
    model: Option<ModelChoice>,
    /// Preset intent that selects a model (mutually exclusive with `--model`).
    #[arg(long, value_enum)]
    preset: Option<PresetChoice>,
    /// Compute precision. Defaults to `bf16` (tensor cores, much faster, tiny
    /// accuracy cost); `f32` is exact but slower. `bf16` may be unsupported on
    /// some backends.
    #[arg(long, value_enum)]
    dtype: Option<DtypeChoice>,
    /// Net output upscale factor, any value in `1 < scale ≤ 4`. The model runs a
    /// fixed ×4 pass, so a smaller factor is reached by shrinking by `4 / scale`:
    /// by default the *output* is shrunk after a full ×4 pass (full input detail,
    /// supersampled — so `--scale 2` costs the same as `--scale 4`); with `--fast`
    /// the *input* is shrunk first instead (fewer pixels, faster, some detail
    /// loss). At `--scale 4` there is no shrink either way.
    #[arg(long, default_value_t = 2.0, value_name = "FACTOR")]
    scale: f32,
    /// Reach a `--scale` below 4 by pre-shrinking the *input* before the ×4 pass
    /// — roughly `(4/scale)²`× fewer pixels through the model (e.g. ~4× faster at
    /// `--scale 2`), trading fine detail for speed — instead of the default of
    /// supersampling (full ×4 pass, then shrink the output). No effect at
    /// `--scale 4`; irrelevant for tiny inputs, where pre-shrinking only loses
    /// detail.
    #[arg(long)]
    fast: bool,
    /// Directory holding `unet.safetensors` / `vae.safetensors`. Ignored for a
    /// part when its explicit `--unet` / `--vae` is set.
    #[arg(long, default_value = "crates/web/models")]
    models_dir: PathBuf,
    /// Explicit UNet safetensors path (overrides `--models-dir`).
    #[arg(long)]
    unet: Option<PathBuf>,
    /// Explicit VAE safetensors path (overrides `--models-dir`).
    #[arg(long)]
    vae: Option<PathBuf>,
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
    /// Tiles per model forward. Higher fills the GPU for a large speedup but needs
    /// proportionally more free VRAM. Lower `--batch` (or free GPU memory) if a
    /// batch runs out of memory.
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
    /// Total linear shrink applied around the fixed ×4 pass to hit the requested
    /// net [`scale`](Self::scale). Always ≥ 1 (the model only upscales); `1.0` at
    /// `--scale 4`, `2.0` at `--scale 2`.
    fn shrink(&self) -> f32 {
        NATIVE_SCALE / self.scale
    }

    /// Shrink applied to the *input* before the ×4 pass — only under `--fast`.
    fn pre_shrink(&self) -> f32 {
        if self.fast { self.shrink() } else { 1.0 }
    }

    /// Shrink applied to the *output* after the ×4 pass — the default path.
    fn post_shrink(&self) -> f32 {
        if self.fast { 1.0 } else { self.shrink() }
    }
}

/// Resolve which model to load from `--model` / `--preset`.
///
/// The two flags are mutually exclusive; with neither set, the `Document`
/// preset's model is used.
fn resolve_model(args: &Args) -> anyhow::Result<candle_upscale::ModelId> {
    match (args.model, args.preset) {
        (Some(_), Some(_)) => {
            bail!("--model and --preset are mutually exclusive; pass at most one")
        }
        (Some(m), None) => Ok(m.model_id()),
        (None, Some(p)) => Ok(p.preset().model()),
        (None, None) => Ok(candle_upscale::Preset::Document.model()),
    }
}

/// Resize `dim` down by `shrink` (≥ 1), rounding to the nearest pixel and never
/// below 1. `shrink == 1.0` is returned unchanged.
fn shrink_dim(dim: u32, shrink: f32) -> u32 {
    if shrink <= 1.0 {
        return dim;
    }
    ((dim as f32 / shrink).round() as u32).max(1)
}

/// `"WxH → …"` description of the scale transform for model-input dims `w`×`h`,
/// naming the fixed ×4 pass and, on the supersample path, the post-shrink to the
/// requested net scale (e.g. `64x64 → 256x256 (×4), supersampled → 128x128`).
fn scale_desc(w: usize, h: usize, args: &Args) -> String {
    let native = NATIVE_SCALE as usize;
    let (mw, mh) = (w * native, h * native);
    let shrink = args.post_shrink();
    if shrink > 1.0 {
        let (fw, fh) = (
            shrink_dim(mw as u32, shrink) as usize,
            shrink_dim(mh as u32, shrink) as usize,
        );
        format!("{w}x{h} → {mw}x{mh} (×{native}), supersampled → {fw}x{fh}")
    } else {
        format!("{w}x{h} → {mw}x{mh}")
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

/// Load the input image and apply the model-agnostic preprocessing: optional
/// center-crop, then — only under `--fast` — an up-front `4/scale`× downsample so
/// the native ×4 pass lands at the requested net scale. Returns RGBA8 bytes plus
/// width/height.
fn prepare_input(args: &Args) -> anyhow::Result<(Vec<u8>, usize, usize)> {
    let img = image::open(&args.input)
        .with_context(|| format!("open input {}", args.input.display()))?
        .to_rgba8();
    let img = match args.crop {
        Some(c) => center_crop(img, c),
        None => img,
    };
    let shrink = args.pre_shrink();
    let img = if shrink > 1.0 {
        let (iw, ih) = (img.width(), img.height());
        let (dw, dh) = (shrink_dim(iw, shrink), shrink_dim(ih, shrink));
        if !args.quiet {
            eprintln!("--fast: pre-downsampling input {iw}×{ih} → {dw}×{dh} before the ×4 pass");
        }
        image::imageops::resize(&img, dw, dh, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };
    let (w, h) = (img.width() as usize, img.height() as usize);
    Ok((img.into_raw(), w, h))
}

/// Write the upscaled RGBA8 buffer to `--output` (PNG or JPEG, by extension).
///
/// On the default (non-`--fast`) path the model's full ×4 output is downsampled
/// by `4/scale` here so the saved image lands at the requested net `--scale` —
/// supersampling, which also anti-aliases. JPEG can't encode an alpha channel, so
/// the alpha is dropped for JPEG output — the upscaled image is fully opaque, so
/// nothing is lost.
fn write_output(args: &Args, out: Vec<u8>, ow: usize, oh: usize) -> anyhow::Result<()> {
    let img =
        image::RgbaImage::from_raw(ow as u32, oh as u32, out).context("assemble output image")?;
    let shrink = args.post_shrink();
    let img = if shrink > 1.0 {
        let (dw, dh) = (shrink_dim(ow as u32, shrink), shrink_dim(oh as u32, shrink));
        if !args.quiet {
            eprintln!("supersample: downsampling output {ow}×{oh} → {dw}×{dh}");
        }
        image::imageops::resize(&img, dw, dh, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };
    let (ow, oh) = (img.width() as usize, img.height() as usize);
    match image::ImageFormat::from_path(&args.output) {
        Ok(image::ImageFormat::Jpeg) => image::DynamicImage::ImageRgba8(img)
            .into_rgb8()
            .save(&args.output),
        _ => img.save(&args.output),
    }
    .with_context(|| format!("write output {}", args.output.display()))?;
    if !args.quiet {
        eprintln!("wrote {} ({ow}×{oh})", args.output.display());
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    // The scheduler indexes `alphas_cumprod` (len 1000) by step-derived timesteps
    // and by `noise_level`. Reject inputs that would index out of bounds or skip
    // denoising entirely and emit pure noise.
    if !(1..1000).contains(&args.steps) {
        bail!("--steps must be in 1..=999 (got {})", args.steps);
    }
    if !(0..=350).contains(&args.noise_level) {
        bail!("--noise-level must be in 0..=350 (got {})", args.noise_level);
    }
    // The model only ever upscales ×4; a net factor at or below 1 is not an
    // upscale, and above 4 the model cannot reach. (`!(_ > 1.0 && …)` also rejects
    // a NaN `--scale`.)
    if !(args.scale > 1.0 && args.scale <= NATIVE_SCALE) {
        bail!(
            "--scale must be in 1 < scale ≤ {NATIVE_SCALE} (got {})",
            args.scale
        );
    }

    let quiet = args.quiet;
    let id = resolve_model(&args)?;
    let dtype_choice = args.dtype.unwrap_or(DtypeChoice::Bf16);
    let dtype = match dtype_choice {
        DtypeChoice::F32 => candle_upscale::DType::F32,
        DtypeChoice::Bf16 => candle_upscale::DType::BF16,
    };
    let device = candle_upscale::select_device()?;

    let cfg = candle_upscale::LoadConfig {
        device,
        dtype,
        weights_root: args.models_dir.clone(),
        unet: args.unet.clone(),
        vae: args.vae.clone(),
    };
    if !quiet {
        eprintln!("loading {id:?} model ({})…", dtype_choice.as_str());
    }
    let t = Instant::now();
    let model = candle_upscale::load_model(id, &cfg).map_err(|e| anyhow!("load model: {e}"))?;
    if !quiet {
        eprintln!("  loaded in {:.1}s", t.elapsed().as_secs_f32());
    }

    let (rgba, w, h) = prepare_input(&args)?;
    let opts = candle_upscale::UpscaleOptions {
        steps: args.steps,
        noise_level: args.noise_level,
        tile: args.tile,
        overlap: args.overlap,
        batch: args.batch,
    };
    if !quiet {
        eprintln!(
            "upscaling {} ({} steps, noise {}, {})…",
            scale_desc(w, h, &args),
            args.steps,
            args.noise_level,
            dtype_choice.as_str(),
        );
    }

    let pb = progress_bar(quiet)?;
    let t = Instant::now();
    let (out, ow, oh) = model
        .upscale_rgba(&rgba, w, h, &opts, SEED, &mut |p| {
            if let Some(pb) = &pb {
                pb.set_position((p * PROGRESS_LEN as f32).round() as u64);
            }
        })
        .map_err(|e| anyhow!("upscale: {e}"))?;
    if let Some(pb) = pb {
        pb.finish_and_clear();
    }
    if !quiet {
        eprintln!("  done in {:.1}s", t.elapsed().as_secs_f32());
    }
    write_output(&args, out, ow, oh)
}
