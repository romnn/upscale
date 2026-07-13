//! End-to-end pipeline parity: the burn upscaler's denoise loop + VAE decode
//! must reproduce the reference `StableDiffusionUpscalePipeline`, fed the
//! identical (dumped) noise and initial latents.
//!
//!     cargo test -p sd-upscale --features wgpu --test pipeline_parity -- --nocapture
//!
//! Requires the pretrained UNet + VAE and `python/dump_pipeline_fixture.py`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::clone_on_copy,
    reason = "test fails loudly by design; device.clone() is required for the non-Copy wgpu device"
)]
mod common;

use common::{rel_l2, tensor_from, test_device, TestBackend as B};
use safetensors::SafeTensors;
use sd_upscale::pipeline::Upscaler;
use sd_upscale::weights::{load_unet, load_vae_decoder};

const NOISE_LEVEL: i64 = 20;
const NUM_STEPS: usize = 3;

fn model_path(subfolder: &str, env: &str) -> String {
    std::env::var(env).unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap();
        format!(
            "{home}/dev/upscale-experiments/cache/hf/models--stabilityai--\
             stable-diffusion-x4-upscaler/snapshots/\
             572c99286543a273bfd17fac263db5a77be12c4c/{subfolder}/diffusion_pytorch_model.safetensors"
        )
    })
}

#[ignore = "requires local model weights/fixtures + GPU; run with: cargo test --features wgpu -- --include-ignored"]
#[test]
fn pipeline_matches_diffusers() {
    let device = test_device();

    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/pipeline.safetensors"
    ))
    .expect("run python/dump_pipeline_fixture.py first");
    let st = SafeTensors::deserialize(&bytes).unwrap();

    let unet =
        load_unet::<B>(&model_path("unet", "SD_X4_UNET"), false, &device).expect("load unet");
    let vae =
        load_vae_decoder::<B>(&model_path("vae", "SD_X4_VAE"), false, &device).expect("load vae");
    let embed = tensor_from::<B, 3>(&st, "prompt_embeds", &device);
    let up = Upscaler::new(unet, vae, embed, device.clone());

    let low_res = tensor_from::<B, 4>(&st, "low_res", &device);
    let init_latents = tensor_from::<B, 4>(&st, "init_latents", &device);
    let low_res_noise = tensor_from::<B, 4>(&st, "low_res_noise", &device);

    // Stage 1: the denoise loop (unet + scheduler).
    let latents = up.denoise(
        low_res.clone(),
        NOISE_LEVEL,
        NUM_STEPS,
        init_latents,
        low_res_noise,
    );
    let latents_err = rel_l2::<B, 4>(
        latents.clone(),
        tensor_from::<B, 4>(&st, "final_latents", &device),
    );
    println!("final_latents rel_l2 = {latents_err:.3e}");
    assert!(
        latents_err < 2e-2,
        "denoise loop diverged: {latents_err:.3e}"
    );

    // Stage 2: VAE decode + postprocess.
    let out = up.decode_latents(latents);
    let out_err = rel_l2::<B, 4>(out, tensor_from::<B, 4>(&st, "output", &device));
    println!("output         rel_l2 = {out_err:.3e}");
    assert!(out_err < 2e-2, "decoded output diverged: {out_err:.3e}");
}
