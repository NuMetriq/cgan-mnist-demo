from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    DCGAN-ish generator for 28x28 MNIST.
    Conditioning: concat z with label embedding, then project to feature map.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        emb_dim: int = 32,
        base_ch: int = 64,
        num_classes: int = 10,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.emb = nn.Embedding(num_classes, emb_dim)

        in_dim = latent_dim + emb_dim

        # Project to (base_ch*4, 7, 7)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, (base_ch * 4) * 7 * 7),
            nn.BatchNorm1d((base_ch * 4) * 7 * 7),
            nn.ReLU(True),
        )

        # Upsample 7->14->28
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                base_ch * 4, base_ch * 2, 4, 2, 1, bias=False
            ),  # 7 -> 14
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1, bias=False),  # 14 -> 28
            nn.BatchNorm2d(base_ch),
            nn.ReLU(True),
            nn.Conv2d(base_ch, 1, 3, 1, 1),  # keep 28
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_emb = self.emb(y)
        x = torch.cat([z, y_emb], dim=1)
        x = self.proj(x)
        x = x.view(x.size(0), -1, 7, 7)
        return self.up(x)


class Discriminator(nn.Module):
    """
    DCGAN-ish discriminator for 28x28.
    Conditioning: label embedding -> (1,28,28) "label map" concatenated as an extra channel.
    """

    def __init__(self, emb_dim: int = 32, base_ch: int = 64, num_classes: int = 10):
        super().__init__()
        self.emb = nn.Embedding(num_classes, emb_dim)
        self.to_label_map = nn.Linear(emb_dim, 28 * 28)

        # input channels: image(1) + labelmap(1) = 2
        self.net = nn.Sequential(
            nn.Conv2d(2, base_ch, 4, 2, 1),  # 28 -> 14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1, bias=False),  # 14 -> 7
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, 1, 1, bias=False),  # 7 -> 7
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear((base_ch * 4) * 7 * 7, 1),  # logits
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_emb = self.emb(y)
        y_map = self.to_label_map(y_emb).view(x.size(0), 1, 28, 28)
        x_in = torch.cat([x, y_map], dim=1)
        h = self.net(x_in)
        return self.head(h)
