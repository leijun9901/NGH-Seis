"""Accepted-only loader for the NGH-Seis v1.0 RTM task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SATURATION_SCALES = np.asarray([0.22, 0.065], dtype=np.float32)


class NGHSeisRTMDataset(Dataset):
    """Load RTM -> (Sh, Sg) pairs selected by the frozen release manifests.

    The loss mask is returned separately and is never concatenated to the
    seismic image.  Vp is optional auxiliary truth, not a model input.
    """

    def __init__(
        self,
        release_root: str | Path,
        split: str,
        *,
        protocol: str = "iid",
        ood_fold: str | None = None,
        augment: bool = False,
        cache: bool = False,
        include_auxiliary_vp: bool = False,
        include_migration_vp: bool = False,
    ) -> None:
        self.release_root = Path(release_root).resolve()
        audit = json.loads(
            (self.release_root / "dataset_audit_summary.json").read_text(encoding="utf-8")
        )
        recorded_production_root = Path(audit["production_root"])
        if recorded_production_root.exists():
            self.production_root = recorded_production_root
        else:
            # Release metadata and production payloads are sibling folders in
            # the portable server layout.  The audit retains the original
            # absolute path for provenance, but it must not prevent migration
            # from Windows to Linux or to a different mount point.
            portable_root = self.release_root.parent / recorded_production_root.name
            if not portable_root.exists():
                raise FileNotFoundError(
                    "Production data not found at its recorded path or portable "
                    f"sibling path: {recorded_production_root} / {portable_root}"
                )
            self.production_root = portable_root
        records = json.loads(
            (self.release_root / "accepted_manifest.json").read_text(encoding="utf-8")
        )
        lookup = {int(row["candidate_id"]): row for row in records}
        split_key = "validation" if split in {"val", "validation"} else split
        if split_key not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split {split!r}")
        if protocol == "iid":
            split_data = json.loads(
                (self.release_root / "iid_splits_80_10_10.json").read_text(encoding="utf-8")
            )
            calibration = json.loads(
                (self.release_root / "rtm_amplitude_calibration.json").read_text(encoding="utf-8")
            )
        elif protocol == "structural-ood":
            if not ood_fold:
                raise ValueError("ood_fold is required for structural-ood")
            protocols = json.loads(
                (self.release_root / "structural_ood_leave_one_style_out.json").read_text(
                    encoding="utf-8"
                )
            )["folds"]
            if ood_fold not in protocols:
                raise ValueError(f"Unknown structural OOD fold {ood_fold!r}")
            split_data = protocols[ood_fold]
            calibrations = json.loads(
                (self.release_root / "structural_ood_amplitude_calibration.json").read_text(
                    encoding="utf-8"
                )
            )["folds"]
            calibration = calibrations[ood_fold]
        else:
            raise ValueError(f"Unknown evaluation protocol {protocol!r}")
        self.records = [lookup[int(candidate_id)] for candidate_id in split_data[split_key]]
        self.input_bound = float(calibration["scale"]) * float(calibration["clip_multiple"])
        if not np.isfinite(self.input_bound) or self.input_bound <= 0.0:
            raise ValueError("Invalid frozen RTM amplitude calibration")
        self.augment = bool(augment)
        self.cache_enabled = bool(cache)
        self.include_auxiliary_vp = bool(include_auxiliary_vp)
        self.include_migration_vp = bool(include_migration_vp)
        if self.include_migration_vp:
            if protocol != "iid":
                raise ValueError(
                    "Two-channel structural OOD requires fold-specific migration-Vp "
                    "calibration, which has not been frozen yet"
                )
            acoustic = json.loads(
                (self.release_root / "task1_acoustic_calibration.json").read_text(
                    encoding="utf-8"
                )
            )
            migration = acoustic["input"]["migration_vp"]
            self.migration_mean = float(migration["mean"])
            self.migration_std = float(migration["std"])
            self.migration_clip = float(acoustic["input"]["standard_deviation_clip"])
        self.protocol = protocol
        self.ood_fold = ood_fold
        self._cache: dict[int, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _read(self, index: int) -> dict[str, np.ndarray]:
        if index in self._cache:
            return self._cache[index]
        record = self.records[index]
        path = self.production_root / record["pair"]
        with np.load(path, allow_pickle=False) as payload:
            image = np.asarray(payload["input_rtm_conditioned_unscaled"], dtype=np.float32)
            sh = np.asarray(payload["target_Sh"], dtype=np.float32)
            sg = np.asarray(payload["target_Sg"], dtype=np.float32)
            mask = np.asarray(payload["loss_mask"], dtype=np.float32)
            valid = np.asarray(payload["valid_mask"], dtype=np.float32)
            vp = (
                np.asarray(payload["auxiliary_Vp"], dtype=np.float32)
                if self.include_auxiliary_vp else None
            )
            migration_vp = (
                np.asarray(payload["migration_Vp"], dtype=np.float32)
                if self.include_migration_vp else None
            )
        image = np.clip(image, -self.input_bound, self.input_bound) / self.input_bound
        target_physical = np.stack([sh, sg], axis=0).astype(np.float32)
        target = np.clip(
            target_physical / SATURATION_SCALES[:, None, None], 0.0, 1.0
        ).astype(np.float32)
        input_channels = [image]
        if migration_vp is not None:
            migration_input = (migration_vp - self.migration_mean) / self.migration_std
            migration_input = np.clip(
                migration_input, -self.migration_clip, self.migration_clip
            ) / self.migration_clip
            input_channels.append((migration_input * valid).astype(np.float32))
        arrays: dict[str, np.ndarray] = {
            "input": np.stack(input_channels, axis=0).astype(np.float32),
            "target": target,
            "target_physical": target_physical,
            "mask": mask[None].astype(np.float32),
        }
        if vp is not None:
            arrays["auxiliary_vp"] = vp[None].astype(np.float32)
        if self.cache_enabled:
            self._cache[index] = arrays
        return arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        arrays = self._read(index)
        item = {key: torch.from_numpy(value.copy()) for key, value in arrays.items()}
        if self.augment and bool(torch.rand(()) < 0.5):
            item = {key: value.flip(-1) for key, value in item.items()}
        record = self.records[index]
        item.update(
            candidate_id=int(record["candidate_id"]),
            style=str(record["style"]),
        )
        return item


def make_ngh_seis_loader(
    release_root: str | Path,
    split: str,
    *,
    batch_size: int,
    seed: int = 20260824,
    num_workers: int = 0,
    cache: bool = False,
    protocol: str = "iid",
    ood_fold: str | None = None,
    include_migration_vp: bool = False,
) -> DataLoader:
    dataset = NGHSeisRTMDataset(
        release_root,
        split,
        protocol=protocol,
        ood_fold=ood_fold,
        augment=split == "train",
        cache=cache and num_workers == 0,
        include_migration_vp=include_migration_vp,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )
