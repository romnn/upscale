//! Candle side of the burn-vs-candle parity check.
//!
//! Produces a single-tile candle output PLUS the exact host-generated inputs
//! (`lowres.bin`, `init.bin`, `lrn.bin`, `meta.txt`) so `parity_burn` in the
//! `sd-upscale` crate can run burn on byte-identical low-res pixels, initial
//! latents, and low-res noise. Feeding both frameworks the same noise is what
//! makes the pixel comparison fair (each framework's own `randn` differs).
//!
//!     cargo run --release -p candle-upscale --example parity_candle -- \
//!         <input.png> <out_dir> <crop> <scale 2|4> <steps> <noise> <seed>

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "example/parity harness fails loudly by design"
)]

use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Tensor};
use candle_upscale::model::LoadConfig;

/// `n` standard-normal `f32` samples, deterministic in `seed`.
///
/// A local copy of the crate-internal host RNG: the burn parity harness must be
/// fed byte-identical noise, so the example generates it the same way the
/// pipeline does (splitmix64 → Box–Muller).
fn gaussian(seed: u64, n: usize) -> Vec<f32> {
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn u01(bits: u64) -> f64 {
        (bits >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = u01(splitmix64(&mut state)).max(f64::MIN_POSITIVE);
        let u2 = u01(splitmix64(&mut state));
        let radius = (-2.0 * u1.ln()).sqrt();
        out.push((radius * (2.0 * PI * u2).cos()) as f32);
        if out.len() < n {
            out.push((radius * (2.0 * PI * u2).sin()) as f32);
        }
    }
    out
}

fn write_f32(path: &Path, data: &[f32]) {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).expect("write bin");
}

fn main() {
    let mut args = std::env::args().skip(1);
    let home = std::env::var("HOME").unwrap();
    let input = args
        .next()
        .unwrap_or_else(|| format!("{home}/dev/upscale-experiments/input/upscale_me.png"));
    let out_dir = args.next().unwrap_or_else(|| ".".to_string());
    let crop: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let scale: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(2);
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(6);
    let noise_level: i64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(42);
    let out = Path::new(&out_dir);
    std::fs::create_dir_all(out).expect("mkdir out");

    let models = format!("{home}/dev/upscale/crates/web/models");
    let device = candle_upscale::cuda_device().expect("cuda");
    let dtype = DType::F32;

    eprintln!("loading models (f32)…");
    let t = Instant::now();
    let cfg = LoadConfig {
        device: device.clone(),
        dtype,
        weights_root: PathBuf::from(&models),
        unet: None,
        vae: None,
    };
    let up = candle_upscale::models::sdx4::load(&cfg).expect("load");
    eprintln!("  loaded in {:.1}s", t.elapsed().as_secs_f32());

    // Center-crop, then (for scale 2) downsample ×2 so the native ×4 pass lands
    // at ×2 — exactly what the CLI does. One tile: crop small enough.
    let img = image::open(&input).expect("open").to_rgba8();
    let (iw, ih) = (img.width(), img.height());
    let (cw, ch) = (crop.min(iw), crop.min(ih));
    let (cx, cy) = ((iw - cw) / 2, (ih - ch) / 2);
    let cropped = image::imageops::crop_imm(&img, cx, cy, cw, ch).to_image();
    let tile = if scale == 2 {
        let (dw, dh) = ((cw / 2).max(1), (ch / 2).max(1));
        image::imageops::resize(&cropped, dw, dh, image::imageops::FilterType::Lanczos3)
    } else {
        cropped
    };
    let (w, h) = (tile.width() as usize, tile.height() as usize);
    let rgba = tile.into_raw();

    // low_res01: [3, h, w] in [0,1] (drop alpha), CHW.
    let hw = w * h;
    let mut low = vec![0f32; 3 * hw];
    for i in 0..hw {
        for c in 0..3 {
            low[c * hw + i] = f32::from(rgba[i * 4 + c]) / 255.0;
        }
    }
    let init = gaussian(seed, 4 * hw);
    let lrn = gaussian(seed ^ 0xA5A5_A5A5, 3 * hw);

    write_f32(&out.join("lowres.bin"), &low);
    write_f32(&out.join("init.bin"), &init);
    write_f32(&out.join("lrn.bin"), &lrn);
    std::fs::write(
        out.join("meta.txt"),
        format!("{h} {w} {steps} {noise_level}\n"),
    )
    .expect("meta");

    let low_t = Tensor::from_vec(low, (1, 3, h, w), &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let init_t = Tensor::from_vec(init, (1, 4, h, w), &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let lrn_t = Tensor::from_vec(lrn, (1, 3, h, w), &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    eprintln!(
        "candle denoise+decode {w}x{h} → {}x{} ({steps} steps)…",
        w * 4,
        h * 4
    );
    let t = Instant::now();
    let decoded = up
        .denoise_decode(&low_t, noise_level, steps, init_t, &lrn_t)
        .expect("denoise_decode");
    eprintln!("  candle done in {:.2}s", t.elapsed().as_secs_f32());

    save_png(&decoded, &out.join("candle.png"));
    eprintln!("wrote {}", out.join("candle.png").display());
}

/// Save a `[1, 3, H, W]` f32 tensor in `[0,1]` as an RGBA PNG.
fn save_png(decoded: &Tensor, path: &Path) {
    let (_, _, oh, ow) = decoded.dims4().unwrap();
    let vals = decoded
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let plane = oh * ow;
    let mut rgba = vec![0u8; plane * 4];
    for px in 0..plane {
        for c in 0..3 {
            let v = vals[c * plane + px].clamp(0.0, 1.0);
            rgba[px * 4 + c] = (v * 255.0 + 0.5) as u8;
        }
        rgba[px * 4 + 3] = 255;
    }
    image::RgbaImage::from_raw(ow as u32, oh as u32, rgba)
        .unwrap()
        .save(path)
        .unwrap();
}
