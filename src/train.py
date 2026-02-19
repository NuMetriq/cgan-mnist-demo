from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from .data import get_dataloader
from .models import Discriminator, Generator
from .sample import save_class_grid
from .utils import (
    Checkpoint,
    ensure_dir,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    set_seed,
    weights_init_dcgan,
)


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def bce_logits():
    return nn.BCEWithLogitsLoss()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument(
        "--resume", action="store_true", help="resume from checkpoint if exists"
    )
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = resolve_device(cfg.get("device", "auto"))
    print(f"[device] {device}")

    out_dir = Path(cfg["paths"]["out_dir"])
    ckpt_dir = ensure_dir(cfg["paths"]["ckpt_dir"])
    sample_dir = ensure_dir(cfg["paths"]["sample_dir"])
    ckpt_path = ckpt_dir / cfg["paths"]["ckpt_name"]

    dl = get_dataloader(
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
    )

    latent_dim = int(cfg["model"]["latent_dim"])
    emb_dim = int(cfg["model"]["emb_dim"])
    g_ch = int(cfg["model"]["g_channels"])
    d_ch = int(cfg["model"]["d_channels"])

    G = Generator(latent_dim=latent_dim, emb_dim=emb_dim, base_ch=g_ch).to(device)
    D = Discriminator(emb_dim=emb_dim, base_ch=d_ch).to(device)

    G.apply(weights_init_dcgan)
    D.apply(weights_init_dcgan)

    opt_g = torch.optim.Adam(
        G.parameters(),
        lr=float(cfg["train"]["lr_g"]),
        betas=tuple(cfg["train"]["betas"]),
    )
    opt_d = torch.optim.Adam(
        D.parameters(),
        lr=float(cfg["train"]["lr_d"]),
        betas=tuple(cfg["train"]["betas"]),
    )
    loss_fn = bce_logits()

    start_epoch = 1
    if args.resume and ckpt_path.exists():
        ck = load_checkpoint(ckpt_path, map_location=device)
        G.load_state_dict(ck["g_state"])
        D.load_state_dict(ck["d_state"])
        opt_g.load_state_dict(ck["opt_g_state"])
        opt_d.load_state_dict(ck["opt_d_state"])
        start_epoch = int(ck["epoch"]) + 1
        print(f"[resume] from {ckpt_path} at epoch {ck['epoch']}")

    epochs = int(cfg["train"]["epochs"])
    d_steps = int(cfg["train"].get("d_steps", 1))
    smooth_real = float(cfg["train"].get("label_smooth_real", 1.0))
    sample_every = int(cfg["train"]["sample_every"])
    ckpt_every = int(cfg["train"]["ckpt_every"])
    rows_per_class = int(cfg["sampling"]["grid_rows_per_class"])

    for epoch in range(start_epoch, epochs + 1):
        G.train()
        D.train()

        pbar = tqdm(dl, desc=f"epoch {epoch}/{epochs}")
        g_loss_ema = None
        d_loss_ema = None

        for x_real, y_real in pbar:
            x_real = x_real.to(device)
            y_real = y_real.to(device)
            b = x_real.size(0)

            # -------------------------
            # Train Discriminator
            # -------------------------
            for _ in range(d_steps):
                opt_d.zero_grad(set_to_none=True)

                # real
                logits_real = D(x_real, y_real)
                real_targets = torch.full((b, 1), smooth_real, device=device)
                loss_real = loss_fn(logits_real, real_targets)

                # fake
                z = torch.randn(b, latent_dim, device=device)
                y_fake = torch.randint(0, 10, (b,), device=device)
                x_fake = G(z, y_fake).detach()
                logits_fake = D(x_fake, y_fake)
                fake_targets = torch.zeros((b, 1), device=device)
                loss_fake = loss_fn(logits_fake, fake_targets)

                d_loss = loss_real + loss_fake
                d_loss.backward()
                opt_d.step()

            # -------------------------
            # Train Generator
            # -------------------------
            opt_g.zero_grad(set_to_none=True)
            z = torch.randn(b, latent_dim, device=device)
            y = torch.randint(0, 10, (b,), device=device)
            x_gen = G(z, y)
            logits = D(x_gen, y)
            # want D to say "real" for fakes
            g_targets = torch.ones((b, 1), device=device)
            g_loss = loss_fn(logits, g_targets)
            g_loss.backward()
            opt_g.step()

            # EMA for nicer display
            d_val = float(d_loss.item())
            g_val = float(g_loss.item())
            d_loss_ema = (
                d_val if d_loss_ema is None else 0.95 * d_loss_ema + 0.05 * d_val
            )
            g_loss_ema = (
                g_val if g_loss_ema is None else 0.95 * g_loss_ema + 0.05 * g_val
            )

            pbar.set_postfix(d_loss=f"{d_loss_ema:.3f}", g_loss=f"{g_loss_ema:.3f}")

        # sampling
        if epoch % sample_every == 0:
            out_path = sample_dir / f"epoch_{epoch:03d}.png"
            save_class_grid(G, device, latent_dim, rows_per_class, out_path, seed=seed)
            print(f"[sample] saved {out_path}")

        # checkpoint
        if epoch % ckpt_every == 0 or epoch == epochs:
            save_checkpoint(
                ckpt_path,
                Checkpoint(
                    epoch=epoch,
                    g_state=G.state_dict(),
                    d_state=D.state_dict(),
                    opt_g_state=opt_g.state_dict(),
                    opt_d_state=opt_d.state_dict(),
                    cfg=cfg,
                ),
            )
            print(f"[ckpt] saved {ckpt_path}")


if __name__ == "__main__":
    main()
