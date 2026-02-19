from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid


@torch.no_grad()
def save_class_grid(
    G,
    device: torch.device,
    latent_dim: int,
    rows_per_class: int,
    out_path: str | Path,
    seed: Optional[int] = None,
) -> None:
    if seed is not None:
        torch.manual_seed(seed)

    G.eval()
    # Make labels 0..9 repeated rows_per_class times
    labels = torch.arange(10, device=device).repeat_interleave(rows_per_class)
    z = torch.randn(labels.size(0), latent_dim, device=device)
    x = G(z, labels)  # [-1,1]
    # grid: 10 columns (digits), rows_per_class rows
    grid = make_grid(x, nrow=10, normalize=True, value_range=(-1, 1), padding=2)
    img = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(str(out_path))


@torch.no_grad()
def generate_samples(
    G,
    device: torch.device,
    latent_dim: int,
    digit: int,
    n: int,
    seed: Optional[int] = None,
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    G.eval()
    y = torch.full((n,), int(digit), device=device, dtype=torch.long)
    z = torch.randn(n, latent_dim, device=device)
    x = G(z, y)
    return x  # [-1,1]


@torch.no_grad()
def interpolate(
    G,
    device: torch.device,
    latent_dim: int,
    digit: int,
    steps: int,
    seed: Optional[int] = None,
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)

    G.eval()
    y = torch.full((steps,), int(digit), device=device, dtype=torch.long)
    z1 = torch.randn(1, latent_dim, device=device)
    z2 = torch.randn(1, latent_dim, device=device)
    alphas = torch.linspace(0, 1, steps, device=device).unsqueeze(1)
    z = (1 - alphas) * z1 + alphas * z2
    x = G(z, y)
    return x  # [-1,1]
