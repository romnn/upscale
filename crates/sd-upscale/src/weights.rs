//! Loading pretrained diffusers safetensors into the burn modules.
//!
//! The weights ship in PyTorch layout, so [`PyTorchToBurnAdapter`] does the two
//! conversions burn needs: transpose 2-D `Linear` weights and rename norm
//! `weight`/`bias` to `gamma`/`beta`. `allow_partial` lets the VAE decoder load
//! from the full VAE file while ignoring the (unused at inference) encoder.

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_store::{
    ChainAdapter, HalfPrecisionAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};
use safetensors::SafeTensors;

/// Attach the PyTorch→burn adapter, plus (opt-in) an fp16→fp32 cast so the
/// smaller `*.fp16.safetensors` files load into the same f32 modules. Weights
/// are always f16 *storage* / f32 *compute* — the download halves, accuracy and
/// speed are unchanged. `PyTorchToBurnAdapter` runs first (transpose Linear,
/// rename norm weight/bias→gamma/beta), then the dtype cast.
fn from_adapter(store: SafetensorsStore, half: bool) -> SafetensorsStore {
    if half {
        store.with_from_adapter(ChainAdapter::new(
            PyTorchToBurnAdapter,
            HalfPrecisionAdapter::new(),
        ))
    } else {
        store.with_from_adapter(PyTorchToBurnAdapter)
    }
}

use crate::unet::Unet;
use crate::vae::{VaeConfig, VaeDecoder};

/// Load the precomputed empty-prompt embedding `[1, 77, 1024]` from safetensors
/// bytes (key `empty_prompt_embed`), as produced by `dump_pipeline_fixture.py`.
pub fn load_embed_bytes<B: Backend>(
    bytes: &[u8],
    device: &B::Device,
) -> Result<Tensor<B, 3>, String> {
    let st = SafeTensors::deserialize(bytes).map_err(|e| format!("bad embed file: {e}"))?;
    let view = st
        .tensor("empty_prompt_embed")
        .map_err(|e| format!("embed tensor missing: {e}"))?;
    let shape: [usize; 3] = view
        .shape()
        .try_into()
        .map_err(|_| "embed must be rank 3".to_string())?;
    let data: Vec<f32> = view
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(Tensor::from_data(TensorData::new(data, shape), device))
}

/// diffusers → burn structural key remaps needed by the UNet: the GEGLU
/// feed-forward's `net.0.proj` / `net.2` become our `ff.proj_in` / `ff.proj_out`.
fn unet_remaps(store: SafetensorsStore) -> SafetensorsStore {
    store
        .with_key_remapping(r"\.ff\.net\.0\.proj\.", ".ff.proj_in.")
        .with_key_remapping(r"\.ff\.net\.2\.", ".ff.proj_out.")
}

/// Load the full UNet2DConditionModel from a diffusers `unet/*.safetensors` file.
/// Set `half` when pointing at a `*.fp16.safetensors` file.
pub fn load_unet<B: Backend>(
    path: &str,
    half: bool,
    device: &B::Device,
) -> Result<Unet<B>, String> {
    let base = from_adapter(SafetensorsStore::from_file(path).allow_partial(true), half);
    finish_unet(unet_remaps(base), device)
}

/// Load the UNet from in-memory safetensors bytes (browser path).
///
/// Takes the bytes by value and moves them into the store rather than copying:
/// the fp32 UNet is ~1.76 GB, and a second copy would blow the wasm32 4 GB
/// address space during load. The caller should not retain the bytes afterward.
pub fn load_unet_bytes<B: Backend>(
    bytes: Vec<u8>,
    half: bool,
    device: &B::Device,
) -> Result<Unet<B>, String> {
    let base = from_adapter(
        SafetensorsStore::from_bytes(Some(bytes)).allow_partial(true),
        half,
    );
    finish_unet(unet_remaps(base), device)
}

fn finish_unet<B: Backend>(
    mut store: SafetensorsStore,
    device: &B::Device,
) -> Result<Unet<B>, String> {
    let mut model = Unet::new(device);
    let result = model
        .load_from(&mut store)
        .map_err(|e| format!("failed to load unet: {e:?}"))?;
    if !result.errors.is_empty() {
        return Err(format!("unet load errors: {:?}", result.errors));
    }
    if !result.missing.is_empty() {
        return Err(format!(
            "unet missing {} params, e.g. {:?}",
            result.missing.len(),
            result.missing.iter().take(8).collect::<Vec<_>>()
        ));
    }
    Ok(model)
}

/// Load a single sub-module by stripping a state-dict `prefix` (e.g.
/// `"down_blocks.0.resnets.0."`) so its inner keys line up with a bare module.
/// Used by the UNet parity tests to check one block at a time against the real
/// pretrained weights. `allow_partial` ignores every non-matching key.
pub fn load_submodule<B: Backend, M: Module<B>>(
    mut module: M,
    path: &str,
    prefix: &str,
) -> Result<M, String> {
    let escaped = prefix.replace('.', "\\.");
    let mut store = SafetensorsStore::from_file(path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .with_key_remapping(format!("^{escaped}"), "")
        .with_key_remapping(r"\.ff\.net\.0\.proj\.", ".ff.proj_in.")
        .with_key_remapping(r"\.ff\.net\.2\.", ".ff.proj_out.")
        .allow_partial(true);
    let result = module
        .load_from(&mut store)
        .map_err(|e| format!("failed to load {prefix}: {e:?}"))?;
    if !result.errors.is_empty() {
        return Err(format!("{prefix} load errors: {:?}", result.errors));
    }
    if !result.missing.is_empty() {
        return Err(format!(
            "{prefix} missing {} params, e.g. {:?}",
            result.missing.len(),
            result.missing.iter().take(5).collect::<Vec<_>>()
        ));
    }
    Ok(module)
}

/// Normalize the VAE mid-block attention key names. The fp32 checkpoint uses the
/// old diffusers `query/key/value/proj_attn`; the fp16 checkpoint was re-exported
/// with the newer `to_q/to_k/to_v/to_out.0`. Both map to our `AttnBlock` fields;
/// these rules only fire on the new-style names, so they're no-ops for fp32.
fn vae_remaps(store: SafetensorsStore) -> SafetensorsStore {
    store
        .with_key_remapping(r"\.to_q\.", ".query.")
        .with_key_remapping(r"\.to_k\.", ".key.")
        .with_key_remapping(r"\.to_v\.", ".value.")
        .with_key_remapping(r"\.to_out\.0\.", ".proj_attn.")
}

/// Load the x4-upscaler VAE decoder from a diffusers `vae/*.safetensors` file.
/// Set `half` when pointing at a `*.fp16.safetensors` file.
pub fn load_vae_decoder<B: Backend>(
    path: &str,
    half: bool,
    device: &B::Device,
) -> Result<VaeDecoder<B>, String> {
    let store = vae_remaps(from_adapter(
        SafetensorsStore::from_file(path).allow_partial(true),
        half,
    ));
    finish_vae(store, device)
}

/// Load the VAE decoder from in-memory safetensors bytes (browser path: fetched
/// then cached, never a filesystem).
///
/// Takes the bytes by value and moves them into the store rather than copying,
/// to keep peak wasm memory down (see [`load_unet_bytes`]).
pub fn load_vae_decoder_bytes<B: Backend>(
    bytes: Vec<u8>,
    half: bool,
    device: &B::Device,
) -> Result<VaeDecoder<B>, String> {
    let store = vae_remaps(from_adapter(
        SafetensorsStore::from_bytes(Some(bytes)).allow_partial(true),
        half,
    ));
    finish_vae(store, device)
}

fn finish_vae<B: Backend>(
    mut store: SafetensorsStore,
    device: &B::Device,
) -> Result<VaeDecoder<B>, String> {
    let mut model = VaeDecoder::new(&VaeConfig::default(), device);
    let result = model
        .load_from(&mut store)
        .map_err(|e| format!("failed to load vae: {e:?}"))?;

    if !result.errors.is_empty() {
        return Err(format!("vae load errors: {:?}", result.errors));
    }
    if !result.missing.is_empty() {
        return Err(format!(
            "vae is missing {} params, e.g. {:?}",
            result.missing.len(),
            result.missing.iter().take(5).collect::<Vec<_>>()
        ));
    }
    Ok(model)
}
