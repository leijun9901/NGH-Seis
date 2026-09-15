"""Task 1 loader: (RTM, migration Vp) -> (true Vp, true AI)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _production_root(release_root: Path) -> Path:
    root = release_root.parent
    if not (root / "data").is_dir():
        raise FileNotFoundError(f"Expected the downloaded dataset beside {release_root}")
    return root


class NGHSeisTask1Dataset(Dataset):
    """Load the frozen acoustic-property benchmark without mask leakage.

    Channel 0 is the frozen RTM amplitude normalization. Channel 1 is the
    known smooth migration Vp. The loss mask is returned separately and is
    never passed to the network.
    """

    def __init__(
        self,
        release_root: str | Path,
        split: str,
        *,
        augment: bool = False,
        cache: bool = False,
        protocol: str = "iid",
        ood_fold: str | None = None,
    ) -> None:
        self.release_root = Path(release_root).resolve()
        self.project_root = self.release_root.parent.parent
        self.production_root = _production_root(self.release_root)
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
            rtm_calibration = json.loads(
                (self.release_root / "rtm_amplitude_calibration.json").read_text(
                    encoding="utf-8"
                )
            )
            acoustic_calibration = json.loads(
                (self.release_root / "task1_acoustic_calibration.json").read_text(
                    encoding="utf-8"
                )
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
            rtm_calibration = json.loads(
                (self.release_root / "structural_ood_amplitude_calibration.json").read_text(
                    encoding="utf-8"
                )
            )["folds"][ood_fold]
            acoustic_calibration = json.loads(
                (self.release_root / "structural_ood_acoustic_calibration.json").read_text(
                    encoding="utf-8"
                )
            )["folds"][ood_fold]
        else:
            raise ValueError(f"Unknown evaluation protocol {protocol!r}")
        self.records = [lookup[int(candidate_id)] for candidate_id in split_data[split_key]]
        self.rtm_bound = float(rtm_calibration["scale"]) * float(
            rtm_calibration["clip_multiple"]
        )
        self.calibration = acoustic_calibration
        self.protocol = protocol
        self.ood_fold = ood_fold
        migration = self.calibration["input"]["migration_vp"]
        targets = self.calibration["target"]
        self.migration_mean = float(migration["mean"])
        self.migration_std = float(migration["std"])
        self.input_clip = float(self.calibration["input"]["standard_deviation_clip"])
        self.delta_mean = float(targets["delta_vp"]["mean"])
        self.delta_std = float(targets["delta_vp"]["std"])
        self.log_ai_residual_mean = float(targets["log_ai_residual"]["mean"])
        self.log_ai_residual_std = float(targets["log_ai_residual"]["std"])
        self.log_ai_intercept = float(targets["log_ai_baseline"]["intercept"])
        self.log_ai_slope = float(targets["log_ai_baseline"]["slope"])
        values = (
            self.rtm_bound,
            self.migration_std,
            self.input_clip,
            self.delta_std,
            self.log_ai_residual_std,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Invalid Task 1 calibration")
        self.augment = bool(augment)
        self.cache_enabled = bool(cache)
        self._cache: dict[int, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _read(self, index: int) -> dict[str, np.ndarray]:
        if index in self._cache:
            return self._cache[index]
        record = self.records[index]
        pair_path = (
            self.project_root / record["source_pair"]
            if "source_pair" in record
            else self.production_root / record["pair"]
        )
        geology_path = (
            self.project_root / record["source_geology"]
            if "source_geology" in record
            else self.production_root / record["geology"]
        )
        with np.load(pair_path, allow_pickle=False) as pair:
            rtm = np.asarray(pair["input_rtm_conditioned_unscaled"], dtype=np.float32)
            vp = np.asarray(pair["auxiliary_Vp"], dtype=np.float32)
            migration_vp = np.asarray(pair["migration_Vp"], dtype=np.float32)
            mask = np.asarray(pair["loss_mask"], dtype=np.float32)
            valid = np.asarray(pair["valid_mask"], dtype=np.float32)
            sh = np.asarray(pair["target_Sh"], dtype=np.float32)
            sg = np.asarray(pair["target_Sg"], dtype=np.float32)
        with np.load(geology_path, allow_pickle=False) as geology:
            ai = np.asarray(geology["AI_view"], dtype=np.float32)

        rtm_input = np.clip(rtm, -self.rtm_bound, self.rtm_bound) / self.rtm_bound
        migration_input = (migration_vp - self.migration_mean) / self.migration_std
        migration_input = np.clip(migration_input, -self.input_clip, self.input_clip)
        migration_input = migration_input / self.input_clip
        # The single padded column is not a physical input. Setting it to zero
        # prevents arbitrary standardized values from revealing the pad mask.
        rtm_input = rtm_input * valid
        migration_input = migration_input * valid

        ai_baseline = np.exp(
            self.log_ai_intercept
            + self.log_ai_slope * np.log(np.maximum(migration_vp, 1.0))
        ).astype(np.float32)
        delta_vp = vp - migration_vp
        log_ai_residual = np.log(np.maximum(ai, 1.0) / np.maximum(ai_baseline, 1.0))
        target = np.stack(
            [
                (delta_vp - self.delta_mean) / self.delta_std,
                (log_ai_residual - self.log_ai_residual_mean)
                / self.log_ai_residual_std,
            ],
            axis=0,
        ).astype(np.float32)
        target *= mask[None]
        reservoir = ((sh >= 0.01) | (sg >= 0.003)).astype(np.float32) * mask
        arrays = {
            "input": np.stack([rtm_input, migration_input], axis=0).astype(np.float32),
            "target": target,
            "target_physical": np.stack([vp, ai], axis=0).astype(np.float32),
            "vp_baseline": migration_vp[None].astype(np.float32),
            "ai_baseline": ai_baseline[None].astype(np.float32),
            "mask": mask[None].astype(np.float32),
            "reservoir_mask": reservoir[None].astype(np.float32),
        }
        if self.cache_enabled:
            self._cache[index] = arrays
        return arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        arrays = self._read(index)
        item = {key: torch.from_numpy(value.copy()) for key, value in arrays.items()}
        if self.augment and bool(torch.rand(()) < 0.5):
            item = {key: value.flip(-1) for key, value in item.items()}
        record = self.records[index]
        item.update(candidate_id=int(record["candidate_id"]), style=str(record["style"]))
        return item


def make_ngh_seis_task1_loader(
    release_root: str | Path,
    split: str,
    *,
    batch_size: int,
    seed: int = 20260824,
    num_workers: int = 0,
    cache: bool = False,
    protocol: str = "iid",
    ood_fold: str | None = None,
) -> DataLoader:
    dataset = NGHSeisTask1Dataset(
        release_root,
        split,
        augment=split == "train",
        cache=cache and num_workers == 0,
        protocol=protocol,
        ood_fold=ood_fold,
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
