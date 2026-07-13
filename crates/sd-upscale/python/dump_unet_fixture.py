"""Dump golden UNet activations from the reference diffusers pipeline.

Fixed small inputs + per-stage hooks give the burn UNet port a parity ladder
(time/class embedding -> first resnet-with-temb -> first Transformer2D -> mid ->
final). Resolves the `attention_head_dim` heads-vs-dim ambiguity by construction:
whichever burn reshape reproduces `out_attn0` is the correct one.

    HF_HOME=~/dev/upscale-experiments/cache/hf \
    ~/dev/upscale-experiments/05-comfyui-diffusion/.venv/bin/python \
    crates/sd-upscale/python/dump_unet_fixture.py
"""

from pathlib import Path

import torch
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file

MODEL = "stabilityai/stable-diffusion-x4-upscaler"
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "unet_forward.safetensors"


def main() -> None:
    torch.manual_seed(0)
    unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=torch.float32)
    unet.eval()

    captured: dict[str, torch.Tensor] = {}

    def grab(name):
        def hook(_m, _i, out):
            if hasattr(out, "sample"):  # diffusers *Output dataclass
                t = out.sample
            elif isinstance(out, tuple):
                t = out[0]
            else:
                t = out
            captured[name] = t.detach().clone()
        return hook

    unet.time_embedding.register_forward_hook(grab("out_time_emb"))
    unet.class_embedding.register_forward_hook(grab("out_class_emb"))
    unet.conv_in.register_forward_hook(grab("out_conv_in"))
    unet.down_blocks[0].resnets[0].register_forward_hook(grab("out_resnet0"))
    unet.down_blocks[1].attentions[0].register_forward_hook(grab("out_attn0"))
    unet.mid_block.register_forward_hook(grab("out_mid"))

    # Small spatial size keeps the CPU parity test cheap: latent 16x16.
    sample = torch.randn(1, 7, 16, 16, dtype=torch.float32)
    encoder_hidden_states = torch.randn(1, 77, 1024, dtype=torch.float32)
    timestep = torch.tensor([500], dtype=torch.long)
    class_labels = torch.tensor([20], dtype=torch.long)  # noise_level

    with torch.no_grad():
        out = unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
        ).sample

    # Isolated Transformer2D unit fixture (the trickiest block): feed a known
    # hidden state + context straight into down_blocks[1].attentions[0].
    tf = unet.down_blocks[1].attentions[0]
    tf_in = torch.randn(1, 512, 8, 8, dtype=torch.float32)
    tf_context = torch.randn(1, 77, 1024, dtype=torch.float32)
    with torch.no_grad():
        tf_out = tf(tf_in, encoder_hidden_states=tf_context).sample

    # Resolve the attention_head_dim ambiguity explicitly.
    b0 = tf.transformer_blocks[0]
    print(f"  attn heads: attn1={b0.attn1.heads} attn2={b0.attn2.heads} "
          f"inner_dim={b0.attn1.to_q.out_features} "
          f"only_cross_attn={getattr(b0, 'only_cross_attention', '?')}")

    tensors = {
        "sample": sample,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep.float(),
        "class_labels": class_labels.float(),
        "output": out,
        "tf_in": tf_in,
        "tf_context": tf_context,
        "tf_out": tf_out,
    }
    tensors.update(captured)
    tensors = {k: v.contiguous().float() for k, v in tensors.items()}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT))
    print(f"wrote {OUT}")
    for k, v in tensors.items():
        print(f"  {k:22s} {tuple(v.shape)}  mean={v.float().mean():+.4f} std={v.float().std():.4f}")


if __name__ == "__main__":
    main()
