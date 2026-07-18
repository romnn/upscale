//! Native, candle-only multi-model image upscaler.
//!
//! Loads a model by `--model` (or a `--preset` intent) and runs it on the
//! fastest available device (CUDA, else Metal, else CPU), upscaling a PNG/JPEG
//! with seam-blended tiling.
//!
//! ```text
//! cargo run --release -p cli -- -i in.png -o out.png --steps 20
//! cargo run --release -p cli -- -i in.png --model sdx4
//! ```

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context};
use clap::Parser;
use indicatif::{HumanBytes, ProgressBar, ProgressStyle};

mod download;

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

/// Multi-model image upscaler (candle).
///
/// With no subcommand this upscales `--input`; the `cache` subcommand manages the
/// downloaded-weights cache instead.
#[derive(Parser)]
#[command(
    about = "Multi-model image upscaler (candle).",
    args_conflicts_with_subcommands = true,
    subcommand_negates_reqs = true
)]
struct Cli {
    /// Optional subcommand. Omit to run the upscaler (the default action).
    #[command(subcommand)]
    command: Option<Command>,
    /// The upscale arguments, used when no subcommand is given.
    #[command(flatten)]
    upscale: Args,
}

/// Top-level subcommands. Absent for the default upscale action.
#[derive(clap::Subcommand)]
enum Command {
    /// Inspect or clear the downloaded-weights cache.
    Cache {
        /// The cache action to perform.
        #[command(subcommand)]
        action: CacheAction,
    },
}

/// Actions for the `cache` subcommand.
#[derive(clap::Subcommand)]
enum CacheAction {
    /// Delete cached weights and report the bytes freed. With `--model`, only
    /// that model's weights are removed; otherwise the whole cache directory.
    Clean {
        /// Restrict the cleanup to a single model's weights.
        #[arg(long, value_enum)]
        model: Option<ModelChoice>,
    },
    /// Print the cache directory path, for manual inspection or removal.
    Dir,
}

/// Arguments for the default upscale action.
#[derive(clap::Args)]
struct Args {
    /// Input image (PNG or JPEG). Required for the upscale action — optional at
    /// the parse layer only so a bare `cache` subcommand does not demand it
    /// (clap's `subcommand_negates_reqs` does not reach `#[command(flatten)]`
    /// args), then validated by [`Args::input_path`].
    #[arg(short, long)]
    input: Option<PathBuf>,
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
    /// Directory holding the model's weight files. When omitted, the weights are
    /// downloaded from Hugging Face to a per-model cache dir on first run (and
    /// reused after), except for `--model tvt`, whose weights are prepared
    /// offline. Ignored for a part when its explicit `--unet` / `--vae` is set.
    #[arg(long)]
    models_dir: Option<PathBuf>,
    /// Explicit UNet safetensors path (overrides `--models-dir`).
    #[arg(long)]
    unet: Option<PathBuf>,
    /// Explicit VAE safetensors path (overrides `--models-dir`).
    #[arg(long)]
    vae: Option<PathBuf>,
    /// Sampler steps. Defaults to 4 for VOSR, 20 for SD-x4, and 1 for TVT.
    #[arg(long)]
    steps: Option<usize>,
    /// Low-res conditioning noise level (`0..=350`; lower = more faithful).
    #[arg(long, default_value_t = 20)]
    noise_level: i64,
    /// Tile size in pixels (the model is trained around 128). Lower values use
    /// less VRAM but produce more tiles; TVT's upstream low-memory profile uses
    /// `--tile 96 --overlap 32`.
    #[arg(long, default_value_t = 128)]
    tile: usize,
    /// Tile overlap in pixels (blended to hide seams).
    #[arg(long, default_value_t = 16)]
    overlap: usize,
    /// Tiles per model forward. Higher fills the GPU for a large speedup but needs
    /// proportionally more free VRAM. Defaults to 8 for VOSR and 1 for the other
    /// models. Lower `--batch` (or free GPU memory) if a batch runs out of memory.
    #[arg(long)]
    batch: Option<usize>,
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

    /// The validated input path. `--input` is parsed as optional (see the field
    /// doc) so the `cache` subcommand need not supply it; the upscale action
    /// requires it, so a missing value is a clear error rather than a panic.
    fn input_path(&self) -> anyhow::Result<&Path> {
        self.input
            .as_deref()
            .ok_or_else(|| anyhow!("--input <INPUT> is required"))
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

/// The cache sub-directory name for a model's downloaded weights (also the label
/// in the download note). Stable across builds so the cache is reused.
fn model_dir_name(id: candle_upscale::ModelId) -> &'static str {
    match id {
        candle_upscale::ModelId::Sdx4 => "sdx4",
        candle_upscale::ModelId::Vosr => "vosr",
        candle_upscale::ModelId::Tvt => "tvt",
    }
}

/// Resolve the directory the model loads its weights from.
///
/// With `--models-dir` set, that directory is used as-is (the dev workflow, no
/// download). Without it, the model's [`manifest`](candle_upscale::ModelId::manifest)
/// weights are downloaded to (and reused from) a per-model cache dir; a model
/// with no manifest (tvt) has no auto-download path and errors with guidance.
fn resolve_weights_root(
    id: candle_upscale::ModelId,
    args: &Args,
) -> anyhow::Result<PathBuf> {
    if let Some(dir) = &args.models_dir {
        return Ok(dir.clone());
    }
    match id.manifest() {
        Some(files) => download::ensure_weights(model_dir_name(id), files, args.quiet),
        None => bail!(
            "--model tvt needs locally-prepared weights; pass --models-dir <dir> containing \
             fused_unet.safetensors and vae.safetensors (auto-download not yet supported for tvt)"
        ),
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
    let input = args.input_path()?;
    let img = image::open(input)
        .with_context(|| format!("open input {}", input.display()))?
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

fn create_output_parent(output: &Path) -> anyhow::Result<()> {
    let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return Ok(());
    };
    std::fs::create_dir_all(parent)
        .with_context(|| format!("create output directory {}", parent.display()))
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
    create_output_parent(&args.output)?;
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
    let cli = Cli::parse();
    match cli.command {
        None => run_upscale(cli.upscale),
        Some(Command::Cache { action }) => run_cache(action),
    }
}

/// Run the `cache` subcommand: print the cache path or clear cached weights.
fn run_cache(action: CacheAction) -> anyhow::Result<()> {
    match action {
        CacheAction::Dir => {
            println!("{}", download::cache_dir().display());
            Ok(())
        }
        CacheAction::Clean { model } => run_cache_clean(model),
    }
}

/// Remove cached weights (a single model's with `--model`, else the whole cache),
/// reporting the freed size. A not-yet-cached target is a no-op, not an error.
fn run_cache_clean(model: Option<ModelChoice>) -> anyhow::Result<()> {
    let target = match model {
        Some(m) => download::model_cache_dir(model_dir_name(m.model_id())),
        None => download::cache_dir(),
    };
    if !target.exists() {
        println!("nothing to clean: {} does not exist", target.display());
        return Ok(());
    }
    let freed = download::dir_size(&target);
    std::fs::remove_dir_all(&target)
        .with_context(|| format!("remove {}", target.display()))?;
    println!("removed {} (freed {})", target.display(), HumanBytes(freed));
    Ok(())
}

/// Run the default upscale action for `args`.
fn run_upscale(args: Args) -> anyhow::Result<()> {
    // Validate `--input` up front so a missing path fails immediately rather than
    // after a first-run weight download.
    args.input_path()?;
    let id = resolve_model(&args)?;
    let steps = args.steps.unwrap_or_else(|| id.default_steps());
    let batch = args.batch.unwrap_or_else(|| id.default_batch());

    // The shared upper bound keeps the value valid for SD-x4's 1000-entry
    // scheduler; every iterative model requires at least one step.
    if !(1..1000).contains(&steps) {
        bail!("--steps must be in 1..=999 (got {steps})");
    }
    if batch == 0 {
        bail!("--batch must be at least 1");
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
    let dtype_choice = args.dtype.unwrap_or(DtypeChoice::Bf16);
    let dtype = match dtype_choice {
        DtypeChoice::F32 => candle_upscale::DType::F32,
        DtypeChoice::Bf16 => candle_upscale::DType::BF16,
    };
    let device = candle_upscale::select_device()?;

    let weights_root = resolve_weights_root(id, &args)?;
    let cfg = candle_upscale::LoadConfig {
        device,
        dtype,
        weights_root,
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
        steps,
        noise_level: args.noise_level,
        tile: args.tile,
        overlap: args.overlap,
        batch,
    };
    if !quiet {
        eprintln!(
            "upscaling {} ({} steps, noise {}, {})…",
            scale_desc(w, h, &args),
            steps,
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

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::create_output_parent;

    #[test]
    fn accepts_output_filename_without_parent() -> anyhow::Result<()> {
        create_output_parent(Path::new("output.png"))
    }
}
