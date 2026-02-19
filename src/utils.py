from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_device(device: str) -> torch.device:
    device = device.lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device in ("cuda", "gpu"):
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class Checkpoint:
    epoch: int
    g_state: Dict[str, Any]
    d_state: Dict[str, Any]
    opt_g_state: Dict[str, Any]
    opt_d_state: Dict[str, Any]
    cfg: Dict[str, Any]


def save_checkpoint(path: str | Path, ckpt: Checkpoint) -> None:
    ensure_dir(Path(path).parent)
    torch.save(
        {
            "epoch": ckpt.epoch,
            "g_state": ckpt.g_state,
            "d_state": ckpt.d_state,
            "opt_g_state": ckpt.opt_g_state,
            "opt_d_state": ckpt.opt_d_state,
            "cfg": ckpt.cfg,
        },
        str(path),
    )


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> Dict[str, Any]:
    return torch.load(str(path), map_location=map_location)


def weights_init_dcgan(m: torch.nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
