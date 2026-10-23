"""A time-conditioned residual U-Net: [current RGB, H&E RGB] -> RGB velocity."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


def group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(8, channels), channels)


class TimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        half = dimension // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half) / max(half - 1, 1))
        self.register_buffer("frequencies", frequencies)
        self.mlp = nn.Sequential(
            nn.Linear(2 * half, dimension), nn.SiLU(), nn.Linear(dimension, dimension)
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        angles = time.float()[:, None] * self.frequencies[None] * 1000
        return self.mlp(torch.cat((angles.sin(), angles.cos()), dim=-1))


class ResidualBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int, time_dim: int, dropout: float):
        super().__init__()
        self.norm1 = group_norm(incoming)
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 2 * outgoing))
        self.norm2 = group_norm(outgoing)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1)
        self.skip = nn.Conv2d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(self, image: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(image)))
        scale, shift = self.time(time).chunk(2, dim=1)
        hidden = self.norm2(hidden) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(hidden)))
        return self.skip(image) + hidden


class BlockStack(nn.Module):
    def __init__(self, incoming: int, outgoing: int, count: int, time_dim: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(incoming if i == 0 else outgoing, outgoing, time_dim, dropout)
                for i in range(count)
            ]
        )

    def forward(self, image: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            image = block(image, time)
        return image


class ConditionalUNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        channels = config.channels
        time_dim = channels[0] * 4
        self.time = TimeEmbedding(time_dim)
        self.input = nn.Conv2d(6, channels[0], 3, padding=1)
        self.encoder = nn.ModuleList(
            [BlockStack(c, c, config.blocks_per_level, time_dim, config.dropout) for c in channels]
        )
        self.down = nn.ModuleList(
            [
                nn.Conv2d(a, b, 3, stride=2, padding=1)
                for a, b in zip(channels[:-1], channels[1:], strict=True)
            ]
        )
        self.middle = BlockStack(channels[-1], channels[-1], 2, time_dim, config.dropout)
        reversed_channels = tuple(reversed(channels))
        self.decoder = nn.ModuleList(
            [
                BlockStack(
                    (reversed_channels[i - 1] if i else c) + c,
                    c,
                    config.blocks_per_level,
                    time_dim,
                    config.dropout,
                )
                for i, c in enumerate(reversed_channels)
            ]
        )
        self.output = nn.Sequential(
            group_norm(channels[0]), nn.SiLU(), nn.Conv2d(channels[0], 3, 3, padding=1)
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self, state: torch.Tensor, time: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if state.shape != condition.shape or state.ndim != 4 or state.shape[1] != 3:
            raise ValueError("State and condition must have matching [B, 3, H, W] shapes")
        embedding = self.time(time.reshape(-1).expand(len(state)))
        hidden = self.input(torch.cat((state, condition), dim=1))
        skips = []
        for level, encoder in enumerate(self.encoder):
            hidden = encoder(hidden, embedding)
            skips.append(hidden)
            if level < len(self.down):
                hidden = self.down[level](hidden)
        hidden = self.middle(hidden, embedding)
        for decoder, skip in zip(self.decoder, reversed(skips), strict=True):
            if hidden.shape[-2:] != skip.shape[-2:]:
                hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
            hidden = decoder(torch.cat((hidden, skip), dim=1), embedding)
        return self.output(hidden)
