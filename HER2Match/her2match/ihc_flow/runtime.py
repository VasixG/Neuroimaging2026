from __future__ import annotations
import json,hashlib,os
from pathlib import Path
import torch
from PIL import Image

def stable_seed(seed: int, identity: str) -> int:
    value = hashlib.sha256(f"{seed}:{identity}".encode()).digest()
    return int.from_bytes(value[:8], "little") % (2**63 - 1)


def initial_noise(condition: torch.Tensor, identities: list[str], seed: int) -> torch.Tensor:
    """Per-image seeds make noise independent of batching, ordering and ODE solver."""
    if len(identities) != len(condition):
        raise ValueError("One identity is required per image")
    noise = [
        torch.randn(
            condition.shape[1:],
            generator=torch.Generator().manual_seed(stable_seed(seed, identity)),
        )
        for identity in identities
    ]
    return torch.stack(noise).to(device=condition.device, dtype=condition.dtype)


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (tensor.detach().float().cpu().clamp(-1, 1) + 1) * 127.5
    array = array.round().to(torch.uint8).permute(1, 2, 0).numpy()
    Image.fromarray(array).save(path)
