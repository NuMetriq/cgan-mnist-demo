from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(batch_size: int, num_workers: int) -> DataLoader:
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # -> [-1, 1]
        ]
    )
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
