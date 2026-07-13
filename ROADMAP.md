# SD x4 upscaler → burn/wgpu, in the browser

Goal: run **`stabilityai/stable-diffusion-x4-upscaler`** (the real latent-diffusion
upscaler, not a GAN) fully **on-device in the browser** — Rust + Leptos + WASM,
inference on **wgpu (WebGPU)**, model cached after first load.

## Why this shape

`burn` is the one Rust ML framework with a solid WebGPU-in-the-browser story, and
it is **backend-agnostic**: the model is written once against `Backend` and runs on
`NdArray` (CPU), `cuda`, or `wgpu`. So we **develop and debug natively** (CPU/CUDA,
fast, `cargo test`) with **numerical-parity tests against the reference diffusers
pipeline**, block by block — then the *same code* runs on wgpu. Parity-on-CPU ⇒
runs-on-wgpu; the only wgpu-specific risks are a missing/slow op or wasm memory,
which surface as clear failures, not silent wrongness.

## Model facts (from the cached checkpoint)

- **Latent** is at the **low-res** spatial size (e.g. 128²); VAE decoder upsamples ×4.
- **UNet2DConditionModel** (`in=7` [4 latent ⊕ 3 low-res], `out=4`):
  `block_out_channels [256,512,512,1024]`, 2 layers/block,
  down `[Down, CrossAttnDown×3]`, up `[CrossAttnUp×3, Up]`,
  `only_cross_attention [T,T,T,F]`, 8 heads, `cross_attention_dim 1024`,
  **`use_linear_projection`**, **`num_class_embeds 1000`** (the noise-level class
  embedding), `norm_groups 32`, `eps 1e-5`, **v_prediction**.
- **VAE AutoencoderKL** decoder: `block_out_channels [128,256,512]`,
  `scaling_factor 0.08333`, `eps 1e-6`.
- **Schedulers**: DDIM (v-pred, scaled_linear β 1e-4→2e-2, 1000 steps, steps_offset 1)
  for denoising; DDPM (epsilon) to noise the low-res conditioning at `noise_level`.
- **Text encoder is skipped in-browser**: with `guidance_scale=0` and an empty
  prompt the only text embedding needed is the constant empty-prompt embedding,
  which we **precompute offline** and ship as a small tensor.

## Weights

diffusers safetensors, PyTorch layout. Loaded with `burn-store`
`SafetensorsStore::from_file(..).with_from_adapter(PyTorchToBurnAdapter).allow_partial(true)`
— the adapter transposes Linear weights and renames norm `weight/bias`→`gamma/beta`.
Ship fp16 (UNet ~904 MB + VAE ~106 MB ≈ 1 GB) or fp32 (~2 GB) — user accepts the size.

## Status

Every ML component is ported and **numerically verified against diffusers on the
wgpu backend** (parity tests run GPU-side; `--features wgpu`, ~150× faster than CPU):

- [x] Workspace + backend plumbing (`ndarray`/`cuda`/`wgpu`); shared test harness
      (`tests/common/mod.rs`) selects the backend by feature.
- [x] **VAE decoder** (`src/vae.rs`, `src/blocks.rs`) — `tests/vae_parity.rs`, worst 5.5e-6.
- [x] **DDIM (v-pred) scheduler + DDPM low-res noising** (`src/scheduler.rs`) —
      `tests/scheduler_parity.rs`, step 9e-8 / add_noise 1.6e-8.
- [x] **UNet2DConditionModel** (`src/unet.rs`) — `tests/unet_parity.rs`: per-block
      (time/class embed, resnet-temb, Transformer2D) **and the full forward, 4.2e-6**.
- [x] **End-to-end pipeline** (`src/pipeline.rs`) — `tests/pipeline_parity.rs`:
      denoise loop 3.7e-6, decoded output 2.2e-6. Reproduces `StableDiffusionUpscalePipeline`.
- [x] Precomputed empty-prompt embedding shipped (`assets/empty_prompt_embed.safetensors`);
      browser entry point `Upscaler::load_full(unet_bytes, vae_bytes, embed_bytes, device)`;
      tiled `upscale_rgba`.
- [x] `crates/web` Leptos/wgpu frontend + Cache-API model caching — WebGPU-gated UI,
      drag/drop upload, streamed model download cached under `sd-x4-v1`, wired to
      `Upscaler::load_full`; `trunk build`/`--release` + wasm `cargo check` green, UI
      renders in a real WebGPU browser (needs `init_setup_async` before first op — see
      `crates/web/src/gpu.rs`).
- [x] End-to-end verified on real image data: `examples/upscale.rs` on wgpu upscales a
      document crop 64→256 (single tile, ~17 s) and 192→768 (2×2 tiles + overlap blend,
      no seams) — sharp, legible text. Same `upscale_rgba` path the browser calls.
- [x] **fp16 opt-in** (default stays fp32): `load_*`/`load_full` take a `half` flag that
      chains `HalfPrecisionAdapter` (f16→f32 on load) so the ~half-size `*.fp16.safetensors`
      load into the same f32 modules. Native via `SD_FP16=1`; browser via a checkbox.
      Note: the fp16 VAE uses newer `to_q/to_k/…` attention keys — normalized in `vae_remaps`.
- [x] **Live in-browser end-to-end verified** (Playwright, real NVIDIA Blackwell WebGPU):
      upload → fp16 ~1 GB download → f32 pipeline build → tiled diffusion → 48→192 result +
      PNG download, 0 console errors. Required making `upscale_rgba` **async**
      (`into_data_async` — WASM can't block on GPU readback) and `Cache.put` failures
      non-fatal (the API rejects single ~1 GB entries; download still succeeds).
- [ ] Optimizations: chunked/IndexedDB caching so ~1 GB weights actually persist; true f16
      compute (WebGPU `shader-f16`) to also cut VRAM/time; wasm memory headroom.

**Iteration-speed note:** CPU (ndarray) conv is slow (~7 min for a 24² VAE decode); the GPU
backends are the way — a full-UNet parity run is ~8 s on wgpu. `--features cuda` also works.
- [ ] Precompute empty-prompt text embedding (`python/dump_prompt_embed.py`).
- [ ] **UNet2DConditionModel** (the big one): timestep + class embedding, ResnetBlock2D
      *with* temb, Transformer2D (linear proj, cross-attn), Down/Up sampling,
      CrossAttn down/up blocks + mid block. Parity-tested per block.
- [ ] DDIM scheduler (v-prediction) + DDPM low-res noising.
- [ ] Pipeline: tile → noise low-res → denoise loop → VAE decode → stitch.
- [ ] `crates/web`: Leptos CSR frontend, wgpu backend, upload/preview/download.
- [ ] Model caching in the browser (Cache Storage API), progress UI.
- [ ] Run on native wgpu, then wasm/WebGPU.

## UNet port blueprint (verified from the checkpoint)

Top level: `conv_in` (7→256, 3×3) · `time_embedding` (Linear 256→1024, silu, Linear
1024→1024) · `class_embedding` (Embedding 1000→1024; `emb = time_emb + class_emb`) ·
`down_blocks[4]` · `mid_block` · `up_blocks[4]` · `conv_norm_out` (GroupNorm 32, 256) ·
`conv_out` (256→4). `sample_size 128`, `norm_eps 1e-5`.

- **Timestep embedding**: sinusoidal(dim=256, flip_sin_to_cos=true, freq_shift=0) →
  `time_embedding`. `class_labels` = the noise level (int) → `class_embedding`.
- **ResnetBlock2D (temb)**: `h=conv1(silu(norm1(x)))`; `h += time_emb_proj(silu(temb))[:,:,None,None]`;
  `h=conv2(silu(norm2(h)))`; `+ (conv_shortcut(x) if in≠out else x)`. GroupNorm(32, eps 1e-5).
- **Downsample2D**: conv 3×3 **stride 2**, pad 1. **Upsample2D**: nearest ×2 then conv 3×3.
- **Transformer2DModel** (`use_linear_projection=true`): GroupNorm(32, eps 1e-6) →
  **Linear** `proj_in` → N× `BasicTransformerBlock` → **Linear** `proj_out`, residual add.
  - `BasicTransformerBlock`: `x += attn1(LN norm1(x), ctx1)`; `x += attn2(LN norm2(x), context)`;
    `x += ff(LN norm3(x))`. `only_cross_attention[i]` picks `ctx1` = context (T) or x (F).
  - `attn1/attn2` (diffusers `Attention`): `to_q/to_k/to_v` **no bias**, `to_out.0` has bias.
    k/v project from 1024-dim context (`to_k/to_v` are 512×1024). **Heads: `attention_head_dim=8`
    is ambiguous (num-heads vs dim-per-head) — the `out_attn0` fixture decides it.** scale = head_dim^-0.5.
  - `ff`: GEGLU — `net.0.proj` (512→4096) split into (a,b), `a * gelu(b)` (→2048), `net.2` (2048→512).
- **down_blocks**: `[DownBlock2D, CrossAttnDownBlock2D×3]`, 2 resnets each (+2 transformers for
  cross-attn blocks), downsampler on blocks 0-2. **mid_block**: resnet → transformer → resnet.
  **up_blocks**: `[CrossAttnUpBlock2D×3, UpBlock2D]`, **3** resnets/transformers each (consume the
  down-block skip connections via channel concat), upsampler on blocks 0-2.
- Channels per stage: `[256, 512, 512, 1024]`. Fixtures: `tests/fixtures/unet_forward.safetensors`
  (`python/dump_unet_fixture.py`).

## Dev commands

```bash
# regenerate VAE golden fixtures (in the diffusers venv)
HF_HOME=~/dev/upscale-experiments/cache/hf \
  ~/dev/upscale-experiments/05-comfyui-diffusion/.venv/bin/python \
  crates/sd-upscale/python/dump_vae_fixture.py

# CPU parity test
cargo test -p sd-upscale --test vae_parity -- --nocapture
```
