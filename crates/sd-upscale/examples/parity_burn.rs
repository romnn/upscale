//! Burn side of the burn-vs-candle parity check (CUDA, f32).
//!
//! Reads the exact host-generated inputs written by the `candle-upscale`
//! `parity_candle` example (`lowres.bin`, `init.bin`, `lrn.bin`, `meta.txt`) and
//! runs the burn pipeline on byte-identical low-res pixels, initial latents, and
//! low-res noise, so the two output PNGs differ only in kernel/precision, not in
//! the random draw. Writes `burn.png` next to the shared inputs.
//!
//!     cargo run --release -p sd-upscale --features cuda --example parity_burn -- <out_dir>

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "example/parity harness fails loudly by design"
)]

use std::path::Path;
use std::time::Instant;

use burn::backend::cuda::CudaDevice;
use burn::backend::Cuda;
use burn::tensor::{Tensor, TensorData};
use sd_upscale::pipeline::Upscaler;

type B = Cuda<f32>;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read bin");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let home = std::env::var("HOME").unwrap();
    let out_dir = args.next().unwrap_or_else(|| ".".to_string());
    let out = Path::new(&out_dir);

    let meta = std::fs::read_to_string(out.join("meta.txt")).expect("meta");
    let nums: Vec<i64> = meta
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let (h, w, steps, noise_level) = (
        nums[0] as usize,
        nums[1] as usize,
        nums[2] as usize,
        nums[3],
    );

    let low = read_f32(&out.join("lowres.bin"));
    let init = read_f32(&out.join("init.bin"));
    let lrn = read_f32(&out.join("lrn.bin"));

    let device = CudaDevice::default();
    let models = format!("{home}/dev/upscale/crates/web/models");
    eprintln!("loading models (f32)…");
    let t = Instant::now();
    let unet_bytes = std::fs::read(format!("{models}/unet.safetensors")).expect("unet");
    let vae_bytes = std::fs::read(format!("{models}/vae.safetensors")).expect("vae");
    let embed = include_bytes!("../assets/empty_prompt_embed.safetensors");
    let up = Upscaler::<B>::load_full(unet_bytes, vae_bytes, embed, false, device.clone())
        .expect("load_full");
    eprintln!("  loaded in {:.1}s", t.elapsed().as_secs_f32());

    let low_t = Tensor::<B, 4>::from_data(TensorData::new(low, [1, 3, h, w]), &device);
    let init_t = Tensor::<B, 4>::from_data(TensorData::new(init, [1, 4, h, w]), &device);
    let lrn_t = Tensor::<B, 4>::from_data(TensorData::new(lrn, [1, 3, h, w]), &device);

    eprintln!(
        "burn denoise+decode {w}x{h} → {}x{} ({steps} steps)…",
        w * 4,
        h * 4
    );
    let t = Instant::now();
    let decoded = up.denoise_decode(low_t, noise_level, steps, init_t, lrn_t);
    let data = pollster::block_on(decoded.into_data_async()).expect("readback");
    eprintln!("  burn done in {:.2}s", t.elapsed().as_secs_f32());

    let vals = data.as_slice::<f32>().unwrap();
    // decoded is [1, 3, 4h, 4w] in [0,1].
    let (oh, ow) = (h * 4, w * 4);
    let plane = oh * ow;
    let mut rgba = vec![0u8; plane * 4];
    for px in 0..plane {
        for c in 0..3 {
            let v = vals[c * plane + px].clamp(0.0, 1.0);
            rgba[px * 4 + c] = (v * 255.0 + 0.5) as u8;
        }
        rgba[px * 4 + 3] = 255;
    }
    image::RgbaImage::from_raw(ow as u32, oh as u32, rgba.clone())
        .unwrap()
        .save(out.join("burn.png"))
        .unwrap();
    eprintln!("wrote {}", out.join("burn.png").display());

    // If the candle output is present, report the pixel-wise parity directly.
    let candle_path = out.join("candle.png");
    if candle_path.exists() {
        let candle = image::open(&candle_path).unwrap().to_rgba8();
        if candle.width() as usize == ow && candle.height() as usize == oh {
            let cbuf = candle.into_raw();
            let (mut max_abs, mut sum_abs, mut n) = (0u32, 0u64, 0u64);
            for px in 0..plane {
                for c in 0..3 {
                    let a = i32::from(rgba[px * 4 + c]);
                    let b = i32::from(cbuf[px * 4 + c]);
                    let d = (a - b).unsigned_abs();
                    max_abs = max_abs.max(d);
                    sum_abs += u64::from(d);
                    n += 1;
                }
            }
            let mean = sum_abs as f64 / n as f64;
            eprintln!("PARITY burn vs candle (0-255): max_abs={max_abs}  mean_abs={mean:.4}");
        } else {
            eprintln!("PARITY: size mismatch, skipping ({ow}x{oh} vs candle)");
        }
    }
}
