//! WebGPU availability check + one-time async device init.
//!
//! `sd_upscale::pipeline::Upscaler::from_safetensors_bytes` creates tensors on
//! a `WgpuDevice` the moment it's called. On native targets burn will lazily
//! and *synchronously* set up the wgpu adapter/device the first time a device
//! is used. On wasm that synchronous path is unavailable — requesting a
//! WebGPU adapter/device from the browser is inherently async — so
//! `cubecl`/`burn` panics if a wgpu op runs on a device that hasn't already
//! been registered via [`burn::backend::wgpu::init_setup_async`]. We call
//! that once, awaited, before the first model load; every subsequent
//! `WgpuDevice::default()` elsewhere in the app (e.g. inside
//! `Upscaler::from_safetensors_bytes`) resolves to the same device id and
//! reuses the already-initialized client.

use std::cell::Cell;

use burn::backend::wgpu::{
    graphics::AutoGraphicsApi, init_setup_async, RuntimeOptions, WgpuDevice,
};

thread_local! {
    static GPU_READY: Cell<bool> = const { Cell::new(false) };
}

/// Whether `navigator.gpu` exists, i.e. the browser implements the WebGPU API
/// at all. Doesn't guarantee an adapter/device can actually be obtained (that
/// can still fail, e.g. no compatible GPU, WebGPU disabled by flag), but it's
/// the same coarse check every "does my browser support WebGPU" guide uses.
pub fn navigator_has_webgpu() -> bool {
    let Some(window) = web_sys::window() else {
        return false;
    };
    let navigator = window.navigator();
    // Avoid pulling in the full (large, still-unstable-in-places) `web_sys::Gpu`
    // API surface just to check presence; a plain property lookup is enough.
    js_sys::Reflect::get(&navigator, &wasm_bindgen::JsValue::from_str("gpu"))
        .map(|v| !v.is_undefined())
        .unwrap_or(false)
}

/// Idempotent: primes the default `WgpuDevice` exactly once per page load.
/// Cheap to call again after the first time (returns immediately).
pub async fn ensure_gpu_ready() -> Result<(), String> {
    if GPU_READY.with(Cell::get) {
        return Ok(());
    }
    let device = WgpuDevice::default();
    init_setup_async::<AutoGraphicsApi>(&device, RuntimeOptions::default()).await;
    GPU_READY.with(|ready| ready.set(true));
    Ok(())
}
