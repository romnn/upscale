"""Dump golden DDIM/DDPM scheduler outputs from the reference diffusers
schedulers, loaded from the actual x4-upscaler checkpoint so the config
(beta schedule, prediction type, steps_offset, ...) is guaranteed correct.

Produces a single safetensors file of named f32 tensors that the burn parity
test (`tests/scheduler_parity.rs`) loads and compares against. Run inside the
diffusers venv:

    HF_HOME=~/dev/upscale-experiments/cache/hf \
    HUGGINGFACE_HUB_CACHE=~/dev/upscale-experiments/cache/hf \
    ~/dev/upscale-experiments/05-comfyui-diffusion/.venv/bin/python \
    crates/sd-upscale/python/dump_scheduler_fixture.py
"""

from pathlib import Path

import torch
from diffusers import DDIMScheduler, DDPMScheduler
from safetensors.torch import save_file

MODEL = "stabilityai/stable-diffusion-x4-upscaler"
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "scheduler.safetensors"


def main() -> None:
    tensors: dict[str, torch.Tensor] = {}

    ddim = DDIMScheduler.from_pretrained(MODEL, subfolder="scheduler")
    ddim.set_timesteps(8)

    torch.manual_seed(0)
    sample = torch.randn(1, 4, 8, 8, dtype=torch.float32)
    model_output = torch.randn(1, 4, 8, 8, dtype=torch.float32)

    t0 = int(ddim.timesteps[0])
    ddim_out = ddim.step(model_output, t0, sample).prev_sample

    tensors["ddim_timesteps"] = ddim.timesteps.clone().float()
    tensors["ddim_t0"] = torch.tensor(float(t0))
    tensors["ddim_sample"] = sample
    tensors["ddim_model_output"] = model_output
    tensors["ddim_step0_out"] = ddim_out

    ddpm = DDPMScheduler.from_pretrained(MODEL, subfolder="low_res_scheduler")

    torch.manual_seed(0)
    original = torch.randn(1, 3, 8, 8, dtype=torch.float32)
    noise = torch.randn(1, 3, 8, 8, dtype=torch.float32)
    t = 20
    ddpm_out = ddpm.add_noise(original, noise, torch.tensor([t]))

    tensors["ddpm_original"] = original
    tensors["ddpm_noise"] = noise
    tensors["ddpm_t"] = torch.tensor(float(t))
    tensors["ddpm_addnoise_out"] = ddpm_out

    tensors = {k: v.contiguous().float() for k, v in tensors.items()}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT))
    print(f"wrote {OUT}")
    for k, v in tensors.items():
        if v.numel() > 1:
            print(f"  {k:20s} {tuple(v.shape)}  mean={v.mean():+.4f} std={v.std():.4f}")
        else:
            print(f"  {k:20s} {tuple(v.shape)}  value={v.item():+.4f}")


if __name__ == "__main__":
    main()
