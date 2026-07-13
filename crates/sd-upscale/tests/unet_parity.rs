//! Numerical-parity tests for the UNet, block by block, against
//! `tests/fixtures/unet_forward.safetensors` (from `python/dump_unet_fixture.py`).
//!
//!     cargo test -p sd-upscale --features wgpu --test unet_parity -- --nocapture
//!
//! Requires the pretrained UNet (`SD_X4_UNET`, or the default HF cache path).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "test/example code fails loudly by design"
)]
mod common;

use common::{rel_l2, tensor_from, test_device, TestBackend as B};
use safetensors::SafeTensors;
use sd_upscale::unet::{
    class_embed_lookup, class_embedding, ResnetBlockTemb, TimestepEmbedding, Transformer2D, Unet,
};
use sd_upscale::weights::{load_submodule, load_unet};

fn unet_path() -> String {
    std::env::var("SD_X4_UNET").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap();
        format!(
            "{home}/dev/upscale-experiments/cache/hf/models--stabilityai--\
             stable-diffusion-x4-upscaler/snapshots/\
             572c99286543a273bfd17fac263db5a77be12c4c/unet/diffusion_pytorch_model.safetensors"
        )
    })
}

fn fixtures() -> Vec<u8> {
    std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/unet_forward.safetensors"
    ))
    .expect("run python/dump_unet_fixture.py first")
}

fn check(name: &str, err: f32) {
    println!("{name:16} rel_l2 = {err:.3e}");
    assert!(err < 2e-2, "{name} diverged: rel_l2 = {err:.3e}");
}

#[ignore = "requires local model weights/fixtures + GPU; run with: cargo test --features wgpu -- --include-ignored"]
#[test]
fn time_embedding_matches() {
    let device = test_device();
    let bytes = fixtures();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let path = unet_path();

    let te = TimestepEmbedding::<B>::new(256, 1024, &device);
    let te = load_submodule(te, &path, "time_embedding.").expect("load time_embedding");
    // timestep = 500 (see dump_unet_fixture.py)
    let out = te.forward(500.0, 256, &device);
    let expected = tensor_from::<B, 2>(&st, "out_time_emb", &device);
    check("time_embedding", rel_l2::<B, 2>(out, expected));
}

#[ignore = "requires local model weights/fixtures + GPU; run with: cargo test --features wgpu -- --include-ignored"]
#[test]
fn class_embedding_matches() {
    let device = test_device();
    let bytes = fixtures();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let path = unet_path();

    let ce = class_embedding::<B>(1000, 1024, &device);
    let ce = load_submodule(ce, &path, "class_embedding.").expect("load class_embedding");
    let out = class_embed_lookup(&ce, 20, &device); // class_labels = 20
    let expected = tensor_from::<B, 2>(&st, "out_class_emb", &device);
    check("class_embedding", rel_l2::<B, 2>(out, expected));
}

#[ignore = "requires local model weights/fixtures + GPU; run with: cargo test --features wgpu -- --include-ignored"]
#[test]
fn resnet0_matches() {
    let device = test_device();
    let bytes = fixtures();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let path = unet_path();

    let resnet = ResnetBlockTemb::<B>::new(256, 256, 1024, &device);
    let resnet = load_submodule(resnet, &path, "down_blocks.0.resnets.0.").expect("load resnet0");

    let x = tensor_from::<B, 4>(&st, "out_conv_in", &device);
    // temb = time_emb + class_emb (added in the UNet before the blocks)
    let temb = tensor_from::<B, 2>(&st, "out_time_emb", &device)
        + tensor_from::<B, 2>(&st, "out_class_emb", &device);
    let out = resnet.forward(x, temb);
    let expected = tensor_from::<B, 4>(&st, "out_resnet0", &device);
    check("resnet0", rel_l2::<B, 4>(out, expected));
}

#[ignore = "requires local model weights/fixtures + GPU; run with: cargo test --features wgpu -- --include-ignored"]
#[test]
fn transformer2d_matches() {
    let device = test_device();
    let bytes = fixtures();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let path = unet_path();

    // down_blocks.1 operates at 512 channels; only_cross_attention = true.
    let tf = Transformer2D::<B>::new(512, 1, true, &device);
    let tf = load_submodule(tf, &path, "down_blocks.1.attentions.0.").expect("load transformer");

    let x = tensor_from::<B, 4>(&st, "tf_in", &device);
    let context = tensor_from::<B, 3>(&st, "tf_context", &device);
    let out = tf.forward(x, context);
    let expected = tensor_from::<B, 4>(&st, "tf_out", &device);
    check("transformer2d", rel_l2::<B, 4>(out, expected));
}

#[ignore = "requires local model weights/fixtures + GPU; run with: cargo test --features wgpu -- --include-ignored"]
#[test]
fn full_unet_matches() {
    let device = test_device();
    let bytes = fixtures();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let path = unet_path();

    let unet: Unet<B> = load_unet(&path, false, &device).expect("load unet");

    let sample = tensor_from::<B, 4>(&st, "sample", &device);
    let context = tensor_from::<B, 3>(&st, "encoder_hidden_states", &device);
    // timestep = 500, class_labels (noise level) = 20 (see dump_unet_fixture.py)
    let out = unet.forward(sample, 500.0, context, 20, &device);
    let expected = tensor_from::<B, 4>(&st, "output", &device);
    check("full_unet", rel_l2::<B, 4>(out, expected));
}
