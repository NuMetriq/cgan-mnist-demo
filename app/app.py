from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from src.models import Generator
from src.sample import generate_samples, interpolate
from src.utils import load_checkpoint, resolve_device
from torchvision.utils import make_grid


def tensor_to_pil_grid(x: torch.Tensor, nrow: int) -> Image.Image:
    # x in [-1,1], shape Bx1x28x28
    grid = make_grid(x, nrow=nrow, normalize=True, value_range=(-1, 1), padding=2)
    img = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img)


def load_generator(ckpt_path: str, device: torch.device) -> tuple[Generator, int]:
    ck = load_checkpoint(ckpt_path, map_location=device)
    cfg = ck.get("cfg", {})
    latent_dim = int(cfg.get("model", {}).get("latent_dim", 64))
    emb_dim = int(cfg.get("model", {}).get("emb_dim", 32))
    g_ch = int(cfg.get("model", {}).get("g_channels", 64))

    G = Generator(latent_dim=latent_dim, emb_dim=emb_dim, base_ch=g_ch).to(device)
    G.load_state_dict(ck["g_state"])
    G.eval()
    return G, latent_dim


def ui_generate(ckpt_path, digit, n, seed, device_choice):
    device = resolve_device(device_choice)
    if not Path(ckpt_path).exists():
        raise gr.Error(f"Checkpoint not found: {ckpt_path}")

    G, latent_dim = load_generator(ckpt_path, device)
    x = generate_samples(G, device, latent_dim, int(digit), int(n), seed=int(seed))
    nrow = min(int(n), 8)
    return tensor_to_pil_grid(x, nrow=nrow)


def ui_interpolate(ckpt_path, digit, steps, seed, device_choice):
    device = resolve_device(device_choice)
    if not Path(ckpt_path).exists():
        raise gr.Error(f"Checkpoint not found: {ckpt_path}")

    G, latent_dim = load_generator(ckpt_path, device)
    x = interpolate(G, device, latent_dim, int(digit), int(steps), seed=int(seed))
    return tensor_to_pil_grid(x, nrow=int(steps))


def build_app():
    with gr.Blocks(title="cGAN MNIST (DCGAN-ish) Demo") as demo:
        gr.Markdown(
            "# Conditional GAN (MNIST)\n"
            "Pick a digit and generate samples from a trained conditional DCGAN-ish model.\n"
            "Train first with `python -m src.train --config configs/default.yaml`."
        )

        with gr.Row():
            ckpt_path = gr.Textbox(
                label="Checkpoint path",
                value="outputs/checkpoints/cgan_mnist_dcgan.pt",
            )
            device_choice = gr.Dropdown(
                label="Device",
                choices=["auto", "cpu", "cuda"],
                value="auto",
            )

        with gr.Row():
            digit = gr.Slider(0, 9, value=7, step=1, label="Digit")
            seed = gr.Number(value=42, precision=0, label="Seed")

        with gr.Tab("Generate"):
            n = gr.Slider(1, 64, value=16, step=1, label="Num samples")
            btn = gr.Button("Generate")
            out = gr.Image(label="Samples", type="pil")
            btn.click(
                ui_generate,
                inputs=[ckpt_path, digit, n, seed, device_choice],
                outputs=out,
            )

        with gr.Tab("Interpolate"):
            steps = gr.Slider(3, 12, value=8, step=1, label="Interpolation steps")
            btn2 = gr.Button("Interpolate")
            out2 = gr.Image(label="Interpolation", type="pil")
            btn2.click(
                ui_interpolate,
                inputs=[ckpt_path, digit, steps, seed, device_choice],
                outputs=out2,
            )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
