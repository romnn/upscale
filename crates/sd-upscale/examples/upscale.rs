//! Native end-to-end smoke test on the wgpu backend: load the real UNet + VAE +
//! prompt embedding, upscale a small real image crop ×4, and write a PNG.
//!
//! This drives the exact `Upscaler::upscale_rgba` path the browser uses, on the
//! actual GPU, so a plausible (non-garbage) output confirms the whole pipeline
//! works on real image bytes — not just on parity fixtures.
//!
//!     cargo run --release -p sd-upscale --features wgpu --example upscale -- \
//!         [input.png] [out.png] [crop] [steps] [noise_level]

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "test/example code fails loudly by design"
)]
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use sd_upscale::pipeline::{UpscaleOptions, Upscaler};
use std::time::Instant;

const SNAP: &str = "dev/upscale-experiments/cache/hf/models--stabilityai--\
stable-diffusion-x4-upscaler/snapshots/572c99286543a273bfd17fac263db5a77be12c4c";

fn main() {
    let mut args = std::env::args().skip(1);
    let home = std::env::var("HOME").unwrap();
    let input = args
        .next()
        .unwrap_or_else(|| format!("{home}/dev/upscale-experiments/input/upscale_me.png"));
    let out_path = args.next().unwrap_or_else(|| "upscaled.png".to_string());
    let crop: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(15);
    let noise_level: i64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);

    let device = WgpuDevice::default();

    // Opt in to the smaller fp16 weights with `SD_FP16=1` (up-converted to f32).
    let half = std::env::var("SD_FP16").is_ok_and(|v| v != "0");
    let suffix = if half {
        "fp16.safetensors"
    } else {
        "safetensors"
    };
    eprintln!(
        "loading models ({} UNet + VAE)…",
        if half {
            "fp16→f32, ~1GB"
        } else {
            "f32, ~2GB"
        }
    );
    let t = Instant::now();
    let unet_bytes = std::fs::read(format!(
        "{home}/{SNAP}/unet/diffusion_pytorch_model.{suffix}"
    ))
    .expect("read unet");
    let vae_bytes = std::fs::read(format!(
        "{home}/{SNAP}/vae/diffusion_pytorch_model.{suffix}"
    ))
    .expect("read vae");
    let embed_bytes = include_bytes!("../assets/empty_prompt_embed.safetensors");
    let up = Upscaler::<Wgpu>::load_full(&unet_bytes, &vae_bytes, embed_bytes, half, device)
        .expect("load_full");
    eprintln!("  loaded in {:.1}s", t.elapsed().as_secs_f32());

    // Take a small centre crop so a single tile runs quickly.
    let img = image::open(&input).expect("open input").to_rgba8();
    let (iw, ih) = (img.width(), img.height());
    let (cx, cy) = (iw.saturating_sub(crop) / 2, ih.saturating_sub(crop) / 2);
    let tile = image::imageops::crop_imm(&img, cx, cy, crop.min(iw), crop.min(ih)).to_image();
    let (w, h) = (tile.width() as usize, tile.height() as usize);
    let rgba = tile.into_raw();
    eprintln!(
        "upscaling {w}x{h} → {}x{} ({steps} steps, noise {noise_level})…",
        w * 4,
        h * 4
    );

    let opts = UpscaleOptions {
        steps,
        noise_level,
        tile: 128, // the model's native tile size
        overlap: 16,
    };
    let t = Instant::now();
    let (out, ow, oh) = pollster::block_on(up.upscale_rgba(&rgba, w, h, &opts, &mut |p| {
        eprint!("\r  {:.0}%   ", p * 100.0)
    }))
    .expect("upscale");
    eprintln!("\n  done in {:.1}s", t.elapsed().as_secs_f32());

    image::RgbaImage::from_raw(ow as u32, oh as u32, out)
        .expect("build output image")
        .save(&out_path)
        .expect("save output");
    eprintln!("wrote {out_path} ({ow}x{oh})");
}
