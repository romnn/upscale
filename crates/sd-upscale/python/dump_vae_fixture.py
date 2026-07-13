"""Dump golden VAE-decoder activations from the reference diffusers pipeline.

Produces a single safetensors file of named f32 tensors that the burn parity
tests load and compare against, stage by stage (post_quant->conv_in->mid->
up0->up1->up2->norm_out->output). Run inside the diffusers venv:

    HF_HOME=~/dev/upscale-experiments/cache/hf \
    ~/dev/upscale-experiments/05-comfyui-diffusion/.venv/bin/python \
    crates/sd-upscale/python/dump_vae_fixture.py
"""

import os
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from safetensors.torch import save_file

MODEL = "stabilityai/stable-diffusion-x4-upscaler"
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "vae_decode.safetensors"


def main() -> None:
    torch.manual_seed(0)
    vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    vae.eval()

    captured: dict[str, torch.Tensor] = {}

    def grab(name):
        def hook(_module, _inp, out):
            captured[name] = (out[0] if isinstance(out, tuple) else out).detach().clone()
        return hook

    dec = vae.decoder
    dec.conv_in.register_forward_hook(grab("out_conv_in"))
    dec.mid_block.register_forward_hook(grab("out_mid"))
    for i, up in enumerate(dec.up_blocks):
        up.register_forward_hook(grab(f"out_up{i}"))
    dec.conv_norm_out.register_forward_hook(grab("out_norm"))

    # Small latent keeps the CPU parity test fast: [1,4,24,24] -> image [1,3,96,96].
    z = torch.randn(1, vae.config.latent_channels, 24, 24, dtype=torch.float32)

    with torch.no_grad():
        image = vae.decode(z).sample

    tensors = {"decode_input": z, "decode_output": image}
    tensors.update(captured)
    tensors = {k: v.contiguous().float() for k, v in tensors.items()}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT))
    print(f"wrote {OUT}")
    for k, v in tensors.items():
        print(f"  {k:14s} {tuple(v.shape)}  mean={v.mean():+.4f} std={v.std():.4f}")


if __name__ == "__main__":
    main()
