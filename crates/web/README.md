# web — SD x4 upscaler, in the browser

Leptos 0.8 (CSR) + WASM frontend for the `sd-upscale` crate. Upload an image,
pick a few options, and it's upscaled 4x **entirely on-device** via WebGPU —
nothing is ever sent to a server. The model weights are downloaded once and
cached in the browser (Cache Storage API), so subsequent visits skip the
download.

This crate is its own Cargo workspace (`[workspace]` in `Cargo.toml`), kept
deliberately separate from the parent `../../Cargo.toml` workspace so the
wasm/leptos/web-sys dependency graph never disturbs the native CPU/CUDA test
harness in `crates/sd-upscale`. It depends on `sd-upscale` only via a `path`
dependency with the `wgpu` feature.

## Requirements

- Browser: **Chrome or Edge 113+** (or another browser with WebGPU enabled).
  The app checks `navigator.gpu` on load and shows a clear message if WebGPU
  isn't available — everything else in the UI is gated behind that check.
- Rust target `wasm32-unknown-unknown` (`rustup target add wasm32-unknown-unknown`).
- [`trunk`](https://trunkrs.dev/) (tested with 0.21.x): `cargo install trunk`.

## Build

```bash
cd crates/web
trunk build            # dev profile, output in dist/
trunk build --release  # optimized + wasm-opt'd
```

Both succeed as of this writing. `trunk build` alone is enough to sanity-check
the crate; `--release` additionally runs `wasm-opt`, which is slow (a minute
or so) given burn/wgpu's size.

## Dev model serving

The frontend loads the real diffusion pipeline via `Upscaler::load_full`,
which needs two large diffusers safetensors files fetched from a configurable
**model base URL** (default `/models`, editable in the UI):

- `{base}/unet.safetensors` — `unet/diffusion_pytorch_model.safetensors`, fp32, ~1.76 GB
- `{base}/vae.safetensors` — `vae/diffusion_pytorch_model.safetensors`, fp32, ~212 MB

(fp32, not fp16, to match what `crates/sd-upscale`'s parity tests are
verified against.) The third input `load_full` needs — the precomputed
empty-prompt embedding, ~315 KB — is small and fixed for this model release,
so it's just `include_bytes!`'d into the wasm binary from
`assets/empty_prompt_embed.safetensors` (copied from
`crates/sd-upscale/assets/`) rather than fetched separately.

Neither of the two big files is bundled into the wasm binary or copied into
`dist/` by trunk's asset pipeline — at ~2 GB combined that would make every
`trunk build`/`trunk serve` rebuild copy/hash gigabytes of data, which is
both slow and pointless for a dev loop.

Instead, `crates/web/models/{unet,vae}.safetensors` are **symlinks** to the
cached HF checkout, served by a plain static file server that `Trunk.toml`'s
`[[proxy]]` section forwards `/models/*` requests to. This keeps the app
itself only ever talking to one origin (`trunk serve`'s), so there's no CORS
to worry about.

If you tick **"Use fp16 weights"** in the UI (~1 GB download instead of ~2 GB),
the app fetches `/models/{unet,vae}.fp16.safetensors` instead, so add those
symlinks too:

```bash
snap=~/dev/upscale-experiments/cache/hf/models--stabilityai--stable-diffusion-x4-upscaler/snapshots/*/
ln -sf "$snap"/unet/diffusion_pytorch_model.fp16.safetensors crates/web/models/unet.fp16.safetensors
ln -sf "$snap"/vae/diffusion_pytorch_model.fp16.safetensors  crates/web/models/vae.fp16.safetensors
```

1. In one terminal, serve `crates/web/` (the parent of `models/`) as static
   files on port 8787, so `http://127.0.0.1:8787/models/unet.safetensors` and
   `.../models/vae.safetensors` resolve:

   ```bash
   cd crates/web
   python3 -m http.server 8787
   ```

2. In another terminal, run trunk as usual:

   ```bash
   cd crates/web
   trunk serve --open
   ```

   `Trunk.toml` proxies any request to `/models/*` on trunk's dev server
   (default `http://127.0.0.1:8080`) through to the file server from step 1.

If you'd rather not run a second process, an alternative is to add
`<link data-trunk rel="copy-dir" href="models" data-target-path="models">`
to `index.html`, which makes trunk copy `models/` into `dist/` on every
build — simple, but it re-copies/re-hashes ~2 GB each time, so it's not the
default here.

To point at a different host entirely (e.g. a CDN for a "production" build),
just change the "Model base URL" field in the UI — it's a plain text input,
not hardcoded — as long as it serves both files at the `unet.safetensors` /
`vae.safetensors` filenames underneath it.

## Model caching

On first run, the app streams each of the two downloads (reporting progress
as it goes) and stores the responses in the Cache Storage API under the
cache name `sd-x4-v1` (`src/model_cache.rs`). On every subsequent run —
including a fresh page load — both are served straight from that cache with
**no network request at all**. Bump `CACHE_NAME` in `src/model_cache.rs` if
you change the served model in an incompatible way and want old cached bytes
to stop being picked up.

## Layout

```
crates/web/
├── Cargo.toml       # separate [workspace]; deps on sd-upscale (path, wgpu feature)
├── Trunk.toml        # dist dir, dev-server proxy for /models
├── index.html         # leptos CSR mount point + stylesheet link
├── assets/
│   ├── style.css
│   └── empty_prompt_embed.safetensors  # copied from crates/sd-upscale/assets/, include_bytes!'d
├── models/
│   ├── unet.safetensors  # symlink to the local HF cache checkout (gitignored)
│   └── vae.safetensors   # symlink to the local HF cache checkout (gitignored)
└── src/
    ├── main.rs         # mounts <App/> to <body>
    ├── app.rs           # the whole UI: upload, controls, progress, previews
    ├── gpu.rs            # navigator.gpu check + one-time async wgpu device init
    ├── image_io.rs        # File/Blob -> RGBA8 via canvas; RGBA8 -> PNG data URL
    └── model_cache.rs      # fetch-with-progress + Cache Storage API
```

## A note on `gpu.rs` (why it exists)

`burn`'s wgpu backend can set up its adapter/device either synchronously
(native) or asynchronously (`init_setup_async`, required on wasm — requesting
a WebGPU adapter from the browser is an inherently async operation, and wasm
can't block on it). If you skip the async init and just call
`Upscaler::load_full(..., WgpuDevice::default())` directly on a fresh page
load, it panics deep in `cubecl` trying to synchronously resolve a future
that can't resolve synchronously on wasm.

`gpu::ensure_gpu_ready()` calls `init_setup_async` once (idempotent — cheap
to call again), before the first model load. Every later
`WgpuDevice::default()` elsewhere in the app (including the one inside
`Upscaler::load_full`) resolves to the same device id and reuses the
already-registered client, so it works synchronously from then on. Once
that's warmed up, the actual `upscale_rgba` call runs synchronously (it tiles
the image and runs the DDIM loop per tile) and can block the main thread for
a while on a large image — acceptable for this MVP per the project roadmap;
the progress bar reflects whatever `on_progress` calls happen during that
call (one update per tile) rather than animating smoothly frame-by-frame or
step-by-step within a tile.

## Status vs. the pipeline

The UI targets the stable `Upscaler`/`UpscaleOptions` API in
`crates/sd-upscale/src/pipeline.rs`: `Upscaler::load_full(unet_bytes,
vae_bytes, embed_bytes, device)` builds the real tiled-diffusion pipeline,
and `upscale_rgba`'s signature hasn't changed since before the UNet/scheduler
landed — see the parent `ROADMAP.md`. `Upscaler::from_safetensors_bytes`
(VAE-only, nearest-neighbour-x4 fallback) still exists upstream but this
frontend doesn't use it now that the full pipeline is available.
