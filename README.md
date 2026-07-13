# upscale — Stable Diffusion x4 upscaler, in your browser, on WebGPU

An on-device image upscaler: upload an image, get a 4× version, **all inference
in the browser on WebGPU** — no server, no upload of your image anywhere. The
model is [`stabilityai/stable-diffusion-x4-upscaler`](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)
(the real latent-diffusion upscaler), ported from scratch to the
[`burn`](https://burn.dev) deep-learning framework and run on its `wgpu` backend.

## Why this exists / how it's built

`burn` is backend-agnostic, so the model is written **once** against its `Backend`
trait and runs on CPU (`NdArray`), CUDA, or `wgpu` (WebGPU). We developed and
debugged natively on the GPU with **per-block numerical-parity tests against the
reference diffusers pipeline**, then shipped the identical code to the browser on
`wgpu`. Every component matches diffusers to ~`1e-6` relative error:

| component | test | parity (rel L2) |
| --- | --- | --- |
| VAE decoder | `vae_parity` | 5.5e-6 |
| DDIM (v-pred) scheduler + DDPM noising | `scheduler_parity` | 9e-8 / 1.6e-8 |
| UNet2DConditionModel (full forward) | `unet_parity` | 4.2e-6 |
| end-to-end pipeline (denoise + decode) | `pipeline_parity` | 3.7e-6 / 2.2e-6 |

See `ROADMAP.md` for the full architecture and status.

## Layout

```
crates/sd-upscale/   the model: VAE, UNet, schedulers, pipeline (backend-generic)
  src/               blocks.rs vae.rs unet.rs scheduler.rs pipeline.rs weights.rs
  tests/             *_parity.rs — numerical parity vs diffusers (CPU/CUDA/wgpu)
  python/            dump_*_fixture.py — golden tensors from the reference model
  examples/upscale.rs  native GPU smoke test → writes a real upscaled PNG
crates/web/          Leptos + WASM frontend, wgpu inference, Cache-API model caching
```

## Run the parity tests (on the GPU)

```bash
cargo test -p sd-upscale --features wgpu          # ~150× faster than CPU
# or --features cuda, or omit for CPU (slow)
```
Needs the model cached locally (the tests default to the HF cache path; override
with `SD_X4_UNET` / `SD_X4_VAE`) and the fixtures (`python/dump_*_fixture.py`).

## Native end-to-end (real image → PNG, on wgpu)

```bash
cargo run --release -p sd-upscale --features wgpu --example upscale -- \
    input.png out.png 64 15 20     # crop=64px, steps=15, noise_level=20
```

## The browser app

```bash
cd crates/web
python3 -m http.server 8787        # terminal 1: serve the model weights (see README)
trunk serve --open                 # terminal 2
```
Requires a WebGPU browser (Chrome/Edge 113+). First load downloads and caches the
model; everything then runs on-device. See `crates/web/README.md` for details.
