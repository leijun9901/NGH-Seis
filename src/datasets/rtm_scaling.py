"""Label-free, survey-level RTM amplitude calibration.

Per-sample percentile normalization destroys between-sample amplitude ratios.
The formal workflow therefore stores an unscaled, deterministically
conditioned image and fits one robust scale on the training split only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def condition_rtm_unscaled(
    image: np.ndarray,
    valid_mask: np.ndarray,
    *,
    background_sigma: tuple[float, float] = (7.0, 11.0),
) -> np.ndarray:
    """Remove a fixed long-wavelength component without amplitude scaling."""

    raw = np.asarray(image, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    if raw.ndim != 2 or raw.shape != mask.shape:
        raise ValueError("image and valid_mask must be identically shaped 2-D arrays")
    if not np.all(np.isfinite(raw)):
        raise ValueError("RTM image contains non-finite values")
    conditioned = raw - gaussian_filter(raw, sigma=background_sigma, mode="nearest")
    conditioned = conditioned.astype(np.float32)
    conditioned[~mask] = 0.0
    return conditioned


def display_normalize_rtm(
    conditioned_unscaled: np.ndarray,
    valid_mask: np.ndarray,
    *,
    percentile: float = 99.5,
) -> tuple[np.ndarray, float]:
    """Create a diagnostic display image; never use it as quantitative input."""

    image = np.asarray(conditioned_unscaled, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    values = np.abs(image[mask])
    scale = max(float(np.percentile(values, percentile)), 1e-20) if values.size else 1.0
    display = np.clip(image / scale, -1.0, 1.0).astype(np.float32)
    display[~mask] = 0.0
    return display, scale


@dataclass(frozen=True)
class RTMAmplitudeCalibration:
    scale: float
    percentile: float
    clip_multiple: float
    fitted_split: str
    field_name: str
    sample_count: int
    value_count: int
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


def fit_training_scale(
    files: list[Path],
    *,
    field_name: str = "rtm_conditioned_unscaled",
    mask_name: str = "valid_mask",
    percentile: float = 99.5,
    clip_multiple: float = 4.0,
    max_values_per_file: int = 65536,
    seed: int = 20260818,
) -> RTMAmplitudeCalibration:
    """Fit one robust scale using input amplitudes from training files only."""

    if not files:
        raise ValueError("no training files were supplied")
    if not 50.0 < percentile < 100.0:
        raise ValueError("percentile must be in (50, 100)")
    if clip_multiple <= 0.0 or max_values_per_file <= 0:
        raise ValueError("clip_multiple and max_values_per_file must be positive")
    rng = np.random.default_rng(seed)
    sampled: list[np.ndarray] = []
    for path in sorted(map(Path, files)):
        with np.load(path, allow_pickle=False) as payload:
            if field_name not in payload or mask_name not in payload:
                raise KeyError(f"{path.name} lacks {field_name!r} or {mask_name!r}")
            image = np.asarray(payload[field_name], dtype=np.float32)
            mask = np.asarray(payload[mask_name], dtype=bool)
        values = np.abs(image[mask & np.isfinite(image)]).astype(np.float32)
        if values.size > max_values_per_file:
            indices = rng.choice(values.size, size=max_values_per_file, replace=False)
            values = values[indices]
        if values.size:
            sampled.append(values)
    if not sampled:
        raise ValueError("training files contain no valid finite RTM amplitudes")
    joined = np.concatenate(sampled)
    scale = max(float(np.percentile(joined, percentile)), 1e-20)
    return RTMAmplitudeCalibration(
        scale=scale,
        percentile=float(percentile),
        clip_multiple=float(clip_multiple),
        fitted_split="train",
        field_name=field_name,
        sample_count=len(files),
        value_count=int(joined.size),
        seed=int(seed),
    )


def apply_training_scale(
    conditioned_unscaled: np.ndarray,
    valid_mask: np.ndarray,
    calibration: RTMAmplitudeCalibration,
) -> np.ndarray:
    """Apply the frozen training scale while retaining relative amplitudes."""

    image = np.asarray(conditioned_unscaled, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    if image.shape != mask.shape:
        raise ValueError("image and valid_mask must have identical shapes")
    bound = calibration.scale * calibration.clip_multiple
    output = np.clip(image, -bound, bound) / bound
    output = output.astype(np.float32)
    output[~mask] = 0.0
    return output
