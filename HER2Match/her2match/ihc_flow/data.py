from __future__ import annotations
import json,random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,Sampler
from .runtime import stable_seed

def read_records(manifest: Path, split: str) -> list[dict]:
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    for record in records:
        for field in ("id", "he", "ihc"):
            path = Path(record[field])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe {field} path in manifest: {record[field]}")
        if record["split"] not in ("train", "val", "test"):
            raise ValueError(f"Unknown split in manifest: {record['split']}")
    identities = [r["id"] for r in records]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate pair IDs in manifest")
    selected = [r for r in records if r["split"] == split]
    if not selected:
        raise ValueError(f"Empty {split} split in {manifest}")
    return selected


class PairedDataset(Dataset):
    def __init__(
        self,
        root: Path,
        manifest: Path,
        split: str,
        crop_size: int | None = None,
        augment: bool = False,
        limit: int | None = None,
    ):
        self.root = root.resolve()
        self.records = read_records(manifest, split)
        if limit is not None:
            if limit < 1:
                raise ValueError("Pair limit must be positive")
            self.records = self.records[:limit]
        self.crop_size = crop_size
        self.augment = augment
        for record in self.records:
            for field in ("he", "ihc"):
                path = (self.root / record[field]).resolve()
                if not path.is_relative_to(self.root) or not path.is_file():
                    raise ValueError(f"Invalid image path in manifest: {record[field]}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int | tuple[int, int]) -> dict:
        index, seed = index if isinstance(index, tuple) else (index, 0)
        record = self.records[index]
        images = []
        for field in ("he", "ihc"):
            with Image.open(self.root / record[field]) as image:
                images.append(np.array(image.convert("RGB"), copy=True))
        if images[0].shape != images[1].shape:
            raise ValueError(f"Pair shape mismatch: {record['id']}")
        generator = random.Random(seed)
        height, width = images[0].shape[:2]
        if self.crop_size is not None:
            size = self.crop_size
            if size > min(height, width):
                raise ValueError(f"Crop {size} exceeds image dimensions for {record['id']}")
            top = generator.randrange(height - size + 1) if self.augment else (height - size) // 2
            left = generator.randrange(width - size + 1) if self.augment else (width - size) // 2
            images = [im[top : top + size, left : left + size] for im in images]
        if self.augment:
            flip_x, flip_y, rotations = (
                generator.random() < 0.5,
                generator.random() < 0.5,
                generator.randrange(4),
            )
            images = [
                np.rot90(im[:: (-1 if flip_y else 1), :: (-1 if flip_x else 1)], rotations)
                for im in images
            ]
        tensors = [
            torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 127.5 - 1
            for im in images
        ]
        return {"he": tensors[0], "ihc": tensors[1], "id": record["id"], "group": record["group"]}


class TrainingBatches(Sampler):
    """Deterministic infinite shuffled epochs; resume by the number of consumed batches.

    Augmentation seeds travel with indices, so worker prefetch does not affect resume.
    The final batch of an epoch can be smaller, but no training image is dropped.
    """

    def __init__(self, size: int, batch_size: int, seed: int, start_batch: int = 0):
        if size < 1 or batch_size < 1:
            raise ValueError("Dataset and batch size must be positive")
        self.size, self.batch_size, self.seed, self.start_batch = (
            size,
            batch_size,
            seed,
            start_batch,
        )

    def __iter__(self):
        batches_per_epoch = (self.size + self.batch_size - 1) // self.batch_size
        epoch, offset = divmod(self.start_batch, batches_per_epoch)
        while True:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            indices = torch.randperm(self.size, generator=generator).tolist()
            for batch in range(offset, batches_per_epoch):
                selected = indices[batch * self.batch_size : (batch + 1) * self.batch_size]
                yield [(i, stable_seed(self.seed, f"{epoch}:{i}")) for i in selected]
            epoch, offset = epoch + 1, 0
