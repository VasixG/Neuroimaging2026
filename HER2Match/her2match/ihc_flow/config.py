from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ModelConfig:
    channels: tuple[int, ...] = (64, 128, 256, 512)
    blocks_per_level: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        self.channels = tuple(self.channels)
        if len(self.channels) < 2 or any(c < 8 for c in self.channels):
            raise ValueError("Use at least two levels, with at least 8 channels each")
        if self.blocks_per_level < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid residual block configuration")
