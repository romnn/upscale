//! First-run download of a model's weights from Hugging Face into a local cache.
//!
//! The library ([`candle_upscale::ModelId::manifest`]) only NAMES the upstream
//! URLs and cache-relative destinations; the blocking HTTP GET, the
//! `.part`-temp-then-rename, and the per-file progress bar live here so the
//! library stays network- and UI-free.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use candle_upscale::WeightFile;
use indicatif::{ProgressBar, ProgressStyle};

/// Read/write buffer for the streaming copy. Large enough that the per-chunk
/// syscall and progress-bar update overhead is negligible against multi-GB
/// downloads.
const CHUNK: usize = 256 * 1024;

/// The OS-standard cache base — `~/.cache` on Linux, `~/Library/Caches` on
/// macOS, `%LOCALAPPDATA%` on Windows. Falls back to `~/.cache`, then a relative
/// `.cache`, when both the OS cache and home dirs are unknown, so a headless host
/// never panics resolving it.
fn os_cache() -> PathBuf {
    dirs::cache_dir()
        .or_else(|| dirs::home_dir().map(|home| home.join(".cache")))
        .unwrap_or_else(|| PathBuf::from(".cache"))
}

/// This app's cache root (`<os-cache>/upscale`): the whole tree `cache clean`
/// (with no model) removes and `cache dir` prints.
pub fn cache_dir() -> PathBuf {
    os_cache().join("upscale")
}

/// Directory holding every model's downloaded weights (`<cache_dir>/models`).
fn models_root() -> PathBuf {
    cache_dir().join("models")
}

/// The cache sub-directory for a single model's weights (`<models_root>/<name>`),
/// the download destination root and the target of `cache clean --model`.
pub fn model_cache_dir(model_name: &str) -> PathBuf {
    models_root().join(model_name)
}

/// Recursively sum the byte size of every regular file under `path`, or `0` if it
/// does not exist or cannot be read. Used to report how much `cache clean` frees
/// before it deletes. Entry metadata is not symlink-followed, but the cache only
/// ever holds files this tool wrote, so there are no symlinks to chase.
pub fn dir_size(path: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| match entry.metadata() {
            Ok(meta) if meta.is_dir() => dir_size(&entry.path()),
            Ok(meta) => meta.len(),
            Err(_) => 0,
        })
        .sum()
}

/// Ensure every file in `files` exists under the model's cache dir, downloading
/// any that are missing, and return that cache dir for use as `weights_root`.
///
/// Each file lands at `<cache_root>/<model_name>/<file.dest>`. An existing,
/// non-empty file is reused (so a second run does not re-download); a missing
/// one is streamed to a sibling `.part` temp and atomically renamed on success,
/// so an interrupted download never leaves a truncated file that a later run
/// would mistake for complete. With `quiet`, no progress bar or notes are drawn.
///
/// # Errors
/// Fails if the cache dir cannot be created, the HTTP GET fails or returns a
/// non-200 status, or the streamed bytes cannot be written.
pub fn ensure_weights(model_name: &str, files: &[WeightFile], quiet: bool) -> Result<PathBuf> {
    let root = model_cache_dir(model_name);
    let missing: Vec<&WeightFile> = files
        .iter()
        .filter(|file| !is_present(&root.join(file.dest)))
        .collect();
    if !missing.is_empty() && !quiet {
        eprintln!(
            "downloading {model_name} weights to {} (first run)…",
            root.display()
        );
    }
    for file in missing {
        let dest = root.join(file.dest);
        download_file(file.url, &dest, quiet).with_context(|| format!("download {}", file.url))?;
    }
    Ok(root)
}

/// Whether `path` is an existing, non-empty regular file — the cache-hit test.
/// A zero-byte file (e.g. a leftover from a failed create) counts as absent so
/// it is re-fetched rather than loaded as a truncated weight.
fn is_present(path: &Path) -> bool {
    fs::metadata(path).is_ok_and(|meta| meta.is_file() && meta.len() > 0)
}

/// Stream `url` to `dest`, via a sibling `<dest>.part` temp renamed on success.
///
/// The rename is atomic on the same filesystem, so `dest` only ever exists once
/// fully written. HF `resolve` links redirect to a CDN blob; ureq follows those
/// automatically. A non-200 final status is reported rather than silently
/// writing an error page to disk.
fn download_file(url: &str, dest: &Path, quiet: bool) -> Result<()> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create cache dir {}", parent.display()))?;
    }

    let resp = match ureq::get(url).call() {
        Ok(resp) => resp,
        Err(ureq::Error::Status(code, resp)) => {
            bail!("server returned HTTP {code} {}", resp.status_text());
        }
        Err(err) => return Err(anyhow!(err)).context("request failed"),
    };
    if resp.status() != 200 {
        bail!("server returned HTTP {} (expected 200)", resp.status());
    }
    let total: Option<u64> = resp
        .header("Content-Length")
        .and_then(|len| len.parse().ok());

    let name = dest.file_name().map_or_else(
        || dest.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    let bar = if quiet {
        None
    } else {
        Some(download_bar(&name, total)?)
    };

    let tmp = part_path(dest);
    let mut reader = resp.into_reader();
    let mut file =
        fs::File::create(&tmp).with_context(|| format!("create temp {}", tmp.display()))?;
    let mut buf = vec![0u8; CHUNK];
    loop {
        let read = reader.read(&mut buf).context("read response body")?;
        if read == 0 {
            break;
        }
        file.write_all(&buf[..read])
            .with_context(|| format!("write {}", tmp.display()))?;
        if let Some(bar) = &bar {
            bar.inc(read as u64);
        }
    }
    file.flush()
        .with_context(|| format!("flush {}", tmp.display()))?;
    drop(file);
    fs::rename(&tmp, dest)
        .with_context(|| format!("rename {} -> {}", tmp.display(), dest.display()))?;
    if let Some(bar) = bar {
        bar.finish();
    }
    Ok(())
}

/// The `<dest>.part` scratch path, formed by appending to the full destination
/// (not replacing its extension) so it stays unique per file and beside `dest`
/// on the same filesystem — a prerequisite for the atomic rename.
fn part_path(dest: &Path) -> PathBuf {
    let mut name = dest.as_os_str().to_owned();
    name.push(".part");
    PathBuf::from(name)
}

/// Per-file download bar: a byte gauge with rate and ETA when the server sent a
/// `Content-Length`, else a byte-counting spinner. Drawn to stderr (like the
/// upscale bar) with the same cyan/blue style, the filename in the message.
fn download_bar(name: &str, total: Option<u64>) -> Result<ProgressBar> {
    let (bar, template) = match total {
        Some(total) => (
            ProgressBar::new(total),
            "  {msg}  {bar:32.cyan/blue} {bytes:>10}/{total_bytes:<10} {bytes_per_sec:>11} eta {eta}",
        ),
        None => (
            ProgressBar::new_spinner(),
            "  {msg}  {bytes:>10} {bytes_per_sec:>11} [{elapsed_precise}]",
        ),
    };
    bar.set_style(
        ProgressStyle::with_template(template)
            .context("build download progress-bar template")?
            .progress_chars("=>-"),
    );
    bar.set_message(name.to_owned());
    // Keep the rate/elapsed refreshing during the slow TLS handshake and the
    // first chunk of a large file, which otherwise looks stalled.
    bar.enable_steady_tick(Duration::from_millis(120));
    Ok(bar)
}
