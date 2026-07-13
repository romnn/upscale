"""Dump a deterministic end-to-end pipeline fixture from the reference models.

Replicates `StableDiffusionUpscalePipeline.__call__` (guidance_scale=0, so no CFG)
step by step with FIXED noise/latents that are saved, so the burn pipeline can be
fed the identical tensors and checked against `final_latents` and `output`. Also
emits the empty-prompt CLIP embedding the browser pipeline ships (no text encoder
in-browser).

    HF_HOME=~/dev/upscale-experiments/cache/hf \
    ~/dev/upscale-experiments/05-comfyui-diffusion/.venv/bin/python \
    crates/sd-upscale/python/dump_pipeline_fixture.py
"""

from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from safetensors.torch import save_file
from transformers import CLIPTextModel, CLIPTokenizer

MODEL = "stabilityai/stable-diffusion-x4-upscaler"
FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "pipeline.safetensors"
EMBED = Path(__file__).resolve().parents[1] / "assets" / "empty_prompt_embed.safetensors"

NOISE_LEVEL = 20
NUM_STEPS = 3
SIZE = 32  # low-res; decoded output is 4x = 128


def main() -> None:
    torch.manual_seed(0)
    unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=torch.float32).eval()
    vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32).eval()
    text_encoder = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=torch.float32).eval()
    tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    ddim = DDIMScheduler.from_pretrained(MODEL, subfolder="scheduler")
    ddpm = DDPMScheduler.from_pretrained(MODEL, subfolder="low_res_scheduler")

    # Empty-prompt embedding (guidance_scale=0 ⇒ this is the only context needed).
    tok = tokenizer("", padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    with torch.no_grad():
        prompt_embeds = text_encoder(tok.input_ids)[0]  # [1,77,1024]

    low_res = torch.rand(1, 3, SIZE, SIZE)  # [0,1]
    image = 2.0 * low_res - 1.0  # preprocess -> [-1,1]

    low_res_noise = torch.randn(image.shape)
    image = ddpm.add_noise(image, low_res_noise, torch.tensor([NOISE_LEVEL]))

    ddim.set_timesteps(NUM_STEPS)
    init_latents = torch.randn(1, 4, SIZE, SIZE) * ddim.init_noise_sigma
    latents = init_latents.clone()

    noise_level_t = torch.tensor([NOISE_LEVEL], dtype=torch.long)
    with torch.no_grad():
        for t in ddim.timesteps:
            lmi = torch.cat([latents, image], dim=1)  # [1,7,H,W]
            lmi = ddim.scale_model_input(lmi, t)  # identity for DDIM
            noise_pred = unet(lmi, t, encoder_hidden_states=prompt_embeds, class_labels=noise_level_t).sample
            latents = ddim.step(noise_pred, t, latents).prev_sample
        final_latents = latents.clone()
        decoded = vae.decode(latents / vae.config.scaling_factor).sample
    output = (decoded / 2 + 0.5).clamp(0, 1)

    tensors = {
        "low_res": low_res,
        "prompt_embeds": prompt_embeds,
        "low_res_noise": low_res_noise,
        "init_latents": init_latents,
        "final_latents": final_latents,
        "output": output,
    }
    tensors = {k: v.contiguous().float() for k, v in tensors.items()}

    FIX.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(FIX))
    EMBED.parent.mkdir(parents=True, exist_ok=True)
    save_file({"empty_prompt_embed": prompt_embeds.contiguous().float()}, str(EMBED))

    print(f"wrote {FIX}\nwrote {EMBED}")
    print(f"NOISE_LEVEL={NOISE_LEVEL} NUM_STEPS={NUM_STEPS} timesteps={list(map(int, ddim.timesteps))}")
    for k, v in tensors.items():
        print(f"  {k:14s} {tuple(v.shape)}  mean={v.mean():+.4f} std={v.std():.4f}")


if __name__ == "__main__":
    main()
