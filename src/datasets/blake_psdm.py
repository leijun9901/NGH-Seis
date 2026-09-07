"""PyTorch loader for Blake Ridge PSDM-to-saturation benchmark samples."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


SATURATION_SCALES = np.asarray([0.25, 0.10], dtype=np.float32)


def stratified_records(
    root: str | Path,
    *,
    seed: int = 20260817,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[dict[str, Any]]]:
    """Return deterministic train/val/test records stratified by geology style."""
    root = Path(root)
    records = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to one")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["style"])].append(record)
    rng = np.random.default_rng(seed)
    result: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for style in sorted(groups):
        group = sorted(groups[style], key=lambda row: int(row["sample_index"]))
        order = rng.permutation(len(group))
        group = [group[int(i)] for i in order]
        n_train = int(round(fractions[0] * len(group)))
        n_val = int(round(fractions[1] * len(group)))
        n_train = min(n_train, len(group) - 2)
        n_val = min(max(n_val, 1), len(group) - n_train - 1)
        result["train"].extend(group[:n_train])
        result["val"].extend(group[n_train:n_train + n_val])
        result["test"].extend(group[n_train + n_val:])
    for split in result:
        result[split].sort(key=lambda row: int(row["sample_index"]))
    return result


class BlakePSDMDataset(Dataset):
    """Load one PSDM input and two co-registered saturation targets."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        split_seed: int = 20260817,
        augment: bool = False,
        cache: bool = True,
        height: int = 512,
        records: list[dict[str, Any]] | None = None,
    ) -> None:
        self.root = Path(root)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.split = split
        self.records = records if records is not None else stratified_records(
            self.root, seed=split_seed
        )[split]
        self.augment = bool(augment)
        self.cache_enabled = bool(cache)
        self.height = int(height)
        if self.height <= 0 or self.height % 16:
            raise ValueError("height must be a positive multiple of 16")
        self._cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if not self.records:
            raise ValueError(f"Empty split: {split}")

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if index in self._cache:
            return self._cache[index]
        record = self.records[index]
        path = self.root / f"{record['stem']}.npz"
        with np.load(path, allow_pickle=False) as data:
            image = np.asarray(data["input_psdm"], dtype=np.float32)[None]
            physical = np.stack(
                [data["target_Sh"], data["target_Sg"]], axis=0
            ).astype(np.float32)
            mask = np.asarray(data["valid_mask"], dtype=np.float32)[None]
        if image.shape != (1, 512, 96) or physical.shape != (2, 512, 96):
            raise ValueError(f"Unexpected arrays in {path.name}: {image.shape}, {physical.shape}")
        target = np.clip(physical / SATURATION_SCALES[:, None, None], 0.0, 1.0)
        if self.height != 512:
            image_t = F.interpolate(
                torch.from_numpy(image)[None], size=(self.height, 96), mode="bilinear",
                align_corners=False, antialias=True,
            )[0]
            # Area averaging preserves mean pore saturation during downsampling.
            target_t = F.interpolate(
                torch.from_numpy(target)[None], size=(self.height, 96), mode="area"
            )[0]
            mask_t = F.interpolate(
                torch.from_numpy(mask)[None], size=(self.height, 96), mode="nearest"
            )[0]
            image = image_t.numpy()
            target = target_t.numpy()
            mask = mask_t.numpy()
        arrays = (image, target.astype(np.float32), mask)
        if self.cache_enabled:
            self._cache[index] = arrays
        return arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, target, mask = self._load(index)
        image_t = torch.from_numpy(image)
        target_t = torch.from_numpy(target)
        mask_t = torch.from_numpy(mask)
        if self.augment and bool(torch.rand(()) < 0.5):
            image_t = image_t.flip(-1)
            target_t = target_t.flip(-1)
            mask_t = mask_t.flip(-1)
        record = self.records[index]
        return {
            "input": image_t,
            "target": target_t,
            "mask": mask_t,
            "sample_id": str(record["stem"]),
            "style": str(record["style"]),
        }


def make_blake_loader(
    root: str | Path,
    split: str,
    *,
    batch_size: int = 8,
    split_seed: int = 20260817,
    augment: bool | None = None,
    shuffle: bool | None = None,
    num_workers: int = 0,
    height: int = 512,
    records: list[dict[str, Any]] | None = None,
) -> DataLoader:
    dataset = BlakePSDMDataset(
        root,
        split,
        split_seed=split_seed,
        augment=(split == "train") if augment is None else augment,
        cache=num_workers == 0,
        height=height,
        records=records,
    )
    generator = torch.Generator().manual_seed(split_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train") if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )
