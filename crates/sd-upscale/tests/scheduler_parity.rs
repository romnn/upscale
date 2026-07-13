//! Numerical-parity test: the burn DDIM (v-prediction) denoising scheduler and
//! DDPM low-res noiser must reproduce the reference diffusers schedulers.
//!
//!     cargo test -p sd-upscale --test scheduler_parity -- --nocapture
//!
//! (add `--features wgpu` or `--features cuda` to run on GPU backends).
//! Requires the fixture from `python/dump_scheduler_fixture.py`.

mod common;

use common::{rel_l2, tensor_from, test_device, TestBackend as B};
use safetensors::SafeTensors;
use sd_upscale::scheduler::{DdimScheduler, LowResNoiser};

#[test]
fn ddim_step_matches_diffusers() {
    let device = test_device();

    let fixture_bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/scheduler.safetensors"
    ))
    .expect("run python/dump_scheduler_fixture.py first");
    let st = SafeTensors::deserialize(&fixture_bytes).unwrap();

    let mut ddim = DdimScheduler::new();
    ddim.set_timesteps(8);

    // Compare our computed inference timesteps against diffusers'.
    let expected_timesteps: Vec<i64> = st
        .tensor("ddim_timesteps")
        .unwrap()
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]).round() as i64)
        .collect();
    assert_eq!(
        ddim.timesteps(),
        expected_timesteps.as_slice(),
        "inference timesteps diverge from diffusers"
    );

    let t0 = ddim.timesteps()[0];

    let sample = tensor_from::<B, 4>(&st, "ddim_sample", &device);
    let model_output = tensor_from::<B, 4>(&st, "ddim_model_output", &device);
    let expected = tensor_from::<B, 4>(&st, "ddim_step0_out", &device);

    let actual = ddim.step(model_output, t0, sample);
    let err = rel_l2::<B, 4>(actual, expected);
    println!("ddim step0 (t={t0}) rel_l2 = {err:.3e}");
    assert!(err < 1e-4, "ddim step diverged: rel_l2 = {err:.3e}");
}

#[test]
fn ddpm_add_noise_matches_diffusers() {
    let device = test_device();

    let fixture_bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/scheduler.safetensors"
    ))
    .expect("run python/dump_scheduler_fixture.py first");
    let st = SafeTensors::deserialize(&fixture_bytes).unwrap();

    let noiser = LowResNoiser::new();

    let original = tensor_from::<B, 4>(&st, "ddpm_original", &device);
    let noise = tensor_from::<B, 4>(&st, "ddpm_noise", &device);
    let expected = tensor_from::<B, 4>(&st, "ddpm_addnoise_out", &device);
    let t = 20i64;

    let actual = noiser.add_noise(original, noise, t);
    let err = rel_l2::<B, 4>(actual, expected);
    println!("ddpm add_noise (t={t}) rel_l2 = {err:.3e}");
    assert!(err < 1e-4, "ddpm add_noise diverged: rel_l2 = {err:.3e}");
}
