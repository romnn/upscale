use std::rc::Rc;

use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use leptos::prelude::*;
use sd_upscale::pipeline::{UpscaleOptions, Upscaler};
use wasm_bindgen::JsCast;
use web_sys::{DragEvent, Event, HtmlInputElement};

use crate::gpu;
use crate::image_io::{self, ImageBuf};
use crate::model_cache::{self, ModelStatus};

const DEFAULT_MODEL_BASE_URL: &str = "/models";

/// Precomputed empty-prompt CLIP embedding (`[1, 77, 1024]`, ~315 KB) is small
/// and fixed for this model release, so unlike the multi-hundred-MB UNet/VAE
/// weights it's simplest to just bake it into the wasm binary rather than
/// give it its own fetch-and-cache round trip.
const EMPTY_PROMPT_EMBED: &[u8] = include_bytes!("../assets/empty_prompt_embed.safetensors");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelPart {
    Unet,
    Vae,
}

#[derive(Clone, Debug, Default)]
enum Status {
    #[default]
    Idle,
    CheckingGpu,
    LoadingModel(ModelPart, ModelStatus),
    BuildingPipeline,
    Running,
    Done,
    Error(String),
}

fn file_from_input(input: &HtmlInputElement) -> Option<web_sys::File> {
    input.files()?.get(0)
}

fn file_from_drop(ev: &DragEvent) -> Option<web_sys::File> {
    ev.data_transfer()?.files()?.get(0)
}

async fn fetch_part(part: ModelPart, url: &str, status: RwSignal<Status>) -> Result<Rc<Vec<u8>>, String> {
    let bytes = model_cache::load_model_bytes(url, |s| {
        status.set(Status::LoadingModel(part, s));
    })
    .await?;
    Ok(Rc::new(bytes))
}

#[component]
pub fn App() -> impl IntoView {
    let gpu_ok = gpu::navigator_has_webgpu();

    let model_base_url = RwSignal::new(DEFAULT_MODEL_BASE_URL.to_string());
    let steps = RwSignal::new(20u32);
    let noise_level = RwSignal::new(20i64);
    // Opt-in fp16 weights: ~1 GB download instead of ~2 GB, up-converted to f32
    // on load so inference is identical. Off by default.
    let fp16 = RwSignal::new(false);

    let original: RwSignal<Option<ImageBuf>> = RwSignal::new(None);
    let result: RwSignal<Option<ImageBuf>> = RwSignal::new(None);
    let status: RwSignal<Status> = RwSignal::new(Status::Idle);
    let progress = RwSignal::new(0.0f32);
    let busy = RwSignal::new(false);
    let dragover = RwSignal::new(false);

    // Non-reactive caches, kept across repeated "Upscale" clicks so the
    // ~2 GB of downloads + safetensors parsing only happens once per page
    // load. `StoredValue<_, LocalStorage>` (rather than a signal) because
    // neither holding the bytes nor the constructed `Upscaler` should
    // trigger a re-render, and `Upscaler<Wgpu>` isn't `Send`/`Sync` (wasm is
    // single-threaded, so that's fine — `LocalStorage` is exactly the
    // opt-out for that).
    let unet_bytes: StoredValue<Option<Rc<Vec<u8>>>, LocalStorage> = StoredValue::new_local(None);
    let vae_bytes: StoredValue<Option<Rc<Vec<u8>>>, LocalStorage> = StoredValue::new_local(None);
    // `Rc` so we can clone a handle out of the scoped `with_value` borrow and
    // hold it across the `.await` of the (async) upscale.
    let upscaler: StoredValue<Option<Rc<Upscaler<Wgpu>>>, LocalStorage> =
        StoredValue::new_local(None);
    // Which precision the cached bytes/pipeline above were built for, so
    // toggling fp16 mid-session reloads the right weights instead of reusing
    // the wrong ones.
    let loaded_half: StoredValue<Option<bool>, LocalStorage> = StoredValue::new_local(None);

    let load_file = move |file: web_sys::File| {
        status.set(Status::Idle);
        leptos::task::spawn_local(async move {
            match image_io::decode_blob_to_rgba(&file).await {
                Ok(buf) => {
                    original.set(Some(buf));
                    result.set(None);
                }
                Err(e) => status.set(Status::Error(format!("failed to decode image: {e}"))),
            }
        });
    };

    let on_file_input = move |ev: Event| {
        let Some(input) = ev.target().and_then(|t| t.dyn_into::<HtmlInputElement>().ok()) else {
            return;
        };
        if let Some(file) = file_from_input(&input) {
            load_file(file);
        }
    };

    let on_drop = move |ev: DragEvent| {
        ev.prevent_default();
        dragover.set(false);
        if let Some(file) = file_from_drop(&ev) {
            load_file(file);
        }
    };

    let on_dragover = move |ev: DragEvent| {
        ev.prevent_default();
        dragover.set(true);
    };

    let on_dragleave = move |_: DragEvent| {
        dragover.set(false);
    };

    let on_upscale = move |_| {
        let Some(src) = original.get_untracked() else {
            return;
        };
        if busy.get_untracked() {
            return;
        }
        let base = model_base_url.get_untracked();
        let half = fp16.get_untracked();
        let suffix = if half { "fp16.safetensors" } else { "safetensors" };
        let unet_url = format!("{base}/unet.{suffix}");
        let vae_url = format!("{base}/vae.{suffix}");
        let opts = UpscaleOptions {
            steps: steps.get_untracked() as usize,
            noise_level: noise_level.get_untracked(),
            ..UpscaleOptions::default()
        };

        // Switching precision means the cached bytes/pipeline are for the wrong
        // weights — drop them so they re-fetch (the browser cache still keys the
        // two URLs separately, so each precision downloads at most once).
        if loaded_half.with_value(|h| *h != Some(half)) {
            unet_bytes.set_value(None);
            vae_bytes.set_value(None);
            upscaler.set_value(None);
            loaded_half.set_value(Some(half));
        }

        busy.set(true);
        progress.set(0.0);
        result.set(None);
        status.set(Status::CheckingGpu);

        leptos::task::spawn_local(async move {
            if let Err(e) = gpu::ensure_gpu_ready().await {
                status.set(Status::Error(format!("WebGPU init failed: {e}")));
                busy.set(false);
                return;
            }

            if unet_bytes.with_value(Option::is_none) {
                match fetch_part(ModelPart::Unet, &unet_url, status).await {
                    Ok(bytes) => unet_bytes.set_value(Some(bytes)),
                    Err(e) => {
                        status.set(Status::Error(format!("UNet download failed: {e}")));
                        busy.set(false);
                        return;
                    }
                }
            }
            if vae_bytes.with_value(Option::is_none) {
                match fetch_part(ModelPart::Vae, &vae_url, status).await {
                    Ok(bytes) => vae_bytes.set_value(Some(bytes)),
                    Err(e) => {
                        status.set(Status::Error(format!("VAE download failed: {e}")));
                        busy.set(false);
                        return;
                    }
                }
            }

            let need_build = upscaler.with_value(Option::is_none);
            if need_build {
                let unet = unet_bytes.with_value(Clone::clone).expect("just populated above");
                let vae = vae_bytes.with_value(Clone::clone).expect("just populated above");
                status.set(Status::BuildingPipeline);
                let built = Upscaler::<Wgpu>::load_full(&unet, &vae, EMPTY_PROMPT_EMBED, half, WgpuDevice::default());
                match built {
                    Ok(up) => upscaler.set_value(Some(Rc::new(up))),
                    Err(e) => {
                        status.set(Status::Error(format!("failed to build pipeline: {e}")));
                        busy.set(false);
                        return;
                    }
                }
            }

            status.set(Status::Running);
            let up = upscaler.with_value(|u| u.as_ref().expect("built above").clone());
            let out = up
                .upscale_rgba(&src.rgba, src.width as usize, src.height as usize, &opts, &mut |p| {
                    progress.set(p);
                })
                .await;

            match out {
                Ok((rgba, ow, oh)) => match image_io::rgba_to_data_url(&rgba, ow as u32, oh as u32) {
                    Ok(data_url) => {
                        result.set(Some(ImageBuf {
                            rgba,
                            width: ow as u32,
                            height: oh as u32,
                            data_url,
                        }));
                        status.set(Status::Done);
                    }
                    Err(e) => status.set(Status::Error(format!("failed to encode result: {e}"))),
                },
                Err(e) => status.set(Status::Error(format!("upscale failed: {e}"))),
            }
            busy.set(false);
        });
    };

    let status_line = move || match status.get() {
        Status::Idle => String::new(),
        Status::CheckingGpu => "Initializing WebGPU device...".to_string(),
        Status::LoadingModel(part, model_status) => {
            let label = match part {
                ModelPart::Unet => "UNet",
                ModelPart::Vae => "VAE",
            };
            match model_status {
                ModelStatus::CheckingCache => format!("Checking local cache for {label}..."),
                ModelStatus::Downloading { received, total } => match total {
                    Some(total) if total > 0 => format!(
                        "Downloading {label}... {:.0} / {:.0} MB ({:.0}%)",
                        received as f64 / 1e6,
                        total as f64 / 1e6,
                        received as f64 / total as f64 * 100.0
                    ),
                    _ => format!("Downloading {label}... {:.0} MB", received as f64 / 1e6),
                },
                ModelStatus::Saving => format!("Saving {label} to browser cache..."),
                ModelStatus::Ready => format!("{label} loaded from cache."),
            }
        }
        Status::BuildingPipeline => "Building pipeline from downloaded weights...".to_string(),
        Status::Running => format!("Upscaling... {:.0}%", progress.get() * 100.0),
        Status::Done => "Done.".to_string(),
        Status::Error(e) => format!("Error: {e}"),
    };

    let is_error = move || matches!(status.get(), Status::Error(_));

    view! {
        <main>
            <header>
                <h1>"SD x4 Upscaler"</h1>
                <p>"On-device 4x image upscaling powered by Stable Diffusion, running entirely in your browser via WebGPU. Nothing is uploaded anywhere."</p>
            </header>

            // `gpu_ok` is decided once at mount (`navigator.gpu` doesn't
            // change mid-session) so this branch is plain, non-reactive Rust
            // control flow rather than a `move ||` view closure — it only
            // needs to run once, and (unlike a reactive closure, which must
            // be callable more than once) is allowed to move owned captures
            // like `on_upscale` into the DOM it builds.
            {if !gpu_ok {
                    view! {
                        <div class="notice error">
                            "This browser doesn't support WebGPU. Please use a recent Chrome or Edge (113+) with WebGPU enabled."
                        </div>
                    }.into_any()
                } else {
                    view! {
                        <div class="panel model-url">
                            <div class="field">
                                <label for="model-url">"Model base URL (expects " <code>"unet.safetensors"</code> " and " <code>"vae.safetensors"</code> " under it — plus " <code>".fp16.safetensors"</code> " variants if fp16 is checked)"</label>
                                <input
                                    id="model-url"
                                    type="text"
                                    prop:value=move || model_base_url.get()
                                    on:input=move |ev| model_base_url.set(event_target_value(&ev))
                                />
                            </div>
                            <div class="field checkbox-field">
                                <label>
                                    <input
                                        type="checkbox"
                                        prop:checked=move || fp16.get()
                                        on:change=move |ev| fp16.set(event_target_checked(&ev))
                                    />
                                    " Use fp16 weights (~1 GB download instead of ~2 GB; up-converted to f32 on load, identical output)"
                                </label>
                            </div>
                        </div>

                        <label
                            class=move || if dragover.get() { "dropzone dragover" } else { "dropzone" }
                            on:dragover=on_dragover
                            on:dragleave=on_dragleave
                            on:drop=on_drop
                        >
                            <input type="file" accept="image/*" on:change=on_file_input />
                            {move || match original.get() {
                                Some(_) => "Drop a new image here or click to replace it.",
                                None => "Drop an image here, or click to choose a file.",
                            }}
                        </label>

                        <div class="panel">
                            <div class="controls">
                                <div class="field">
                                    <label for="steps">"Steps: " {move || steps.get()}</label>
                                    <input
                                        id="steps"
                                        type="range"
                                        min="1"
                                        max="50"
                                        prop:value=move || steps.get().to_string()
                                        on:input=move |ev| {
                                            if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                                steps.set(v);
                                            }
                                        }
                                    />
                                </div>
                                <div class="field">
                                    <label for="noise">"Noise level: " {move || noise_level.get()}</label>
                                    <input
                                        id="noise"
                                        type="range"
                                        min="0"
                                        max="350"
                                        prop:value=move || noise_level.get().to_string()
                                        on:input=move |ev| {
                                            if let Ok(v) = event_target_value(&ev).parse::<i64>() {
                                                noise_level.set(v);
                                            }
                                        }
                                    />
                                </div>
                            </div>

                            <button
                                disabled=move || original.get().is_none() || busy.get()
                                on:click=on_upscale
                            >
                                {move || if busy.get() { "Upscaling..." } else { "Upscale 4x" }}
                            </button>

                            <div class=move || if is_error() { "status-line notice error" } else { "status-line" }>
                                {status_line}
                            </div>

                            {move || busy.get().then(|| view! {
                                <div class="progress-track">
                                    <div class="progress-fill" style:width=move || format!("{}%", progress.get() * 100.0)></div>
                                </div>
                            })}
                        </div>

                        <div class="previews">
                            {move || original.get().map(|img| view! {
                                <div class="preview">
                                    <h3>"Original"</h3>
                                    <img src=img.data_url.clone() alt="original" />
                                    <div class="meta">{format!("{} x {}", img.width, img.height)}</div>
                                </div>
                            })}
                            {move || result.get().map(|img| view! {
                                <div class="preview">
                                    <h3>"Upscaled (4x)"</h3>
                                    <img src=img.data_url.clone() alt="upscaled result" />
                                    <div class="meta">{format!("{} x {}", img.width, img.height)}</div>
                                    <br/>
                                    <a
                                        class="download-btn"
                                        href=img.data_url.clone()
                                        download="upscaled-4x.png"
                                    >
                                        "Download PNG"
                                    </a>
                                </div>
                            })}
                        </div>
                    }.into_any()
                }}

            <footer>
                "First run downloads the model weights (UNet + VAE, ~2 GB combined) and caches them in your browser via the Cache Storage API. Subsequent runs load from cache with no re-download."
            </footer>
        </main>
    }
}
