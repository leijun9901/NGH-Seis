"""Lazy PyTorch loader for released NGH-Seis v1.0 pairs.

The original pair files remain immutable.  Optional full-domain Vp labels and
sediment masks live in a parallel directory and are joined by sample id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class HFGSatDataset(Dataset):
    """Load 8x320x128 gathers, 2x200x200 saturation, and optional Vp."""

    def __init__(
        self,
        pair_root: str | Path,
        split: str,
        *,
        vp_root: str | Path | None = None,
        require_vp: bool = False,
        return_metadata: bool = False,
        amplitude_clip: float = 8.0,
        vp_range_mps: tuple[float, float] = (1100.0, 3200.0),
        limit: int | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.split = split
        self.files = sorted((Path(pair_root) / split).glob("sample_*.npz"))
        if limit is not None:
            self.files = self.files[:limit]
        if not self.files:
            raise FileNotFoundError(f"No pair files found for split={split}")
        self.vp_root = Path(vp_root) if vp_root is not None else None
        self.require_vp = require_vp
        self.return_metadata = return_metadata
        self.amplitude_clip = float(amplitude_clip)
        self.vp_min, self.vp_max = map(float, vp_range_mps)
        if self.amplitude_clip <= 0 or self.vp_max <= self.vp_min:
            raise ValueError("Invalid normalization range")

    def __len__(self) -> int:
        return len(self.files)

    def _vp_path(self, pair_path: Path) -> Path:
        if self.vp_root is None:
            return Path()
        return self.vp_root / self.split / pair_path.name

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as pair:
            gather = np.asarray(pair["input"], dtype=np.float32)
            saturation = np.asarray(pair["target"], dtype=np.float32)
            metadata_text = str(pair["metadata_json"])
        if gather.shape != (8, 320, 128) or saturation.shape != (2, 200, 200):
            raise ValueError(f"Unexpected arrays in {path.name}: {gather.shape}, {saturation.shape}")

        # One scale per survey preserves relative amplitudes across shots and traces.
        rms = float(np.sqrt(np.mean(gather.astype(np.float64) ** 2)))
        gather = np.clip(gather / max(rms, 1e-8), -self.amplitude_clip, self.amplitude_clip)
        gather = gather / self.amplitude_clip
        item: dict[str, Any] = {
            "input": torch.from_numpy(np.ascontiguousarray(gather)),
            "saturation": torch.from_numpy(np.ascontiguousarray(saturation)),
            "sample_id": path.stem,
            "amplitude_rms": torch.tensor(rms, dtype=torch.float32),
        }

        vp_path = self._vp_path(path)
        if self.vp_root is not None and vp_path.exists():
            with np.load(vp_path, allow_pickle=False) as auxiliary:
                vp = np.asarray(auxiliary["vp_mps"], dtype=np.float32)
                sediment = np.asarray(auxiliary["sediment_fraction"], dtype=np.float32)
            if vp.shape != (200, 200) or sediment.shape != (200, 200):
                raise ValueError(f"Unexpected Vp arrays in {vp_path.name}")
            vp_norm = np.clip((vp - self.vp_min) / (self.vp_max - self.vp_min), 0.0, 1.0)
            item["vp_norm"] = torch.from_numpy(vp_norm[None].copy())
            item["vp_mps"] = torch.from_numpy(vp[None].copy())
            item["sediment_fraction"] = torch.from_numpy(sediment[None].copy())
        elif self.require_vp:
            raise FileNotFoundError(f"Missing auxiliary Vp label: {vp_path}")

        if self.return_metadata:
            item["metadata"] = json.loads(metadata_text)
        return item


def make_loader(
    pair_root: str | Path,
    split: str,
    *,
    vp_root: str | Path | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool | None = None,
    seed: int = 20260813,
    limit: int | None = None,
) -> DataLoader:
    dataset = HFGSatDataset(
        pair_root, split, vp_root=vp_root, require_vp=vp_root is not None, limit=limit
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train") if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )
