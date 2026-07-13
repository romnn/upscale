//! Numerical-parity test: the burn VAE decoder must reproduce the reference
//! diffusers decoder stage by stage.
//!
//!     cargo test -p sd-upscale --features wgpu --test vae_parity -- --nocapture
//!
//! (drop `--features wgpu` for CPU, or use `--features cuda`). Requires the
//! pretrained VAE (`SD_X4_VAE`, or the default HF cache path) and the fixtures
//! from `python/dump_vae_fixture.py`.

mod common;

use burn::tensor::Tensor;
use common::{rel_l2, tensor_from, test_device, TestBackend as B};
use safetensors::SafeTensors;
use sd_upscale::weights::load_vae_decoder;

fn vae_path() -> String {
    std::env::var("SD_X4_VAE").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap();
        format!(
            "{home}/dev/upscale-experiments/cache/hf/models--stabilityai--\
             stable-diffusion-x4-upscaler/snapshots/\
             572c99286543a273bfd17fac263db5a77be12c4c/vae/diffusion_pytorch_model.safetensors"
        )
    })
}

#[test]
fn vae_decoder_matches_diffusers() {
    let device = test_device();

    let fixture_bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/vae_decode.safetensors"
    ))
    .expect("run python/dump_vae_fixture.py first");
    let st = SafeTensors::deserialize(&fixture_bytes).unwrap();

    let path = vae_path();
    assert!(
        std::path::Path::new(&path).exists(),
        "VAE weights not found at {path}; set SD_X4_VAE"
    );
    let vae = load_vae_decoder::<B>(&path, false, &device).expect("load vae");

    let z = tensor_from::<B, 4>(&st, "decode_input", &device);
    let trace = vae.forward_trace(z);

    let stages: Vec<(&str, Tensor<B, 4>)> = vec![
        ("out_conv_in", trace.out_conv_in),
        ("out_mid", trace.out_mid),
        ("out_up0", trace.out_up[0].clone()),
        ("out_up1", trace.out_up[1].clone()),
        ("out_up2", trace.out_up[2].clone()),
        ("out_norm", trace.out_norm),
        ("decode_output", trace.output),
    ];

    let mut worst = 0.0f32;
    for (name, actual) in stages {
        let expected = tensor_from::<B, 4>(&st, name, &device);
        let err = rel_l2::<B, 4>(actual, expected);
        worst = worst.max(err);
        println!("stage {name:14} rel_l2 = {err:.3e}");
        assert!(err < 2e-2, "stage {name} diverged: rel_l2 = {err:.3e}");
    }
    println!("worst stage rel_l2 = {worst:.3e}");
}
