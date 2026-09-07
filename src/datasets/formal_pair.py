"""Build the released network pair from formal physical arrays."""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly


def select_uniform_shots(candidate_count: int, selected_count: int = 8) -> np.ndarray:
    if selected_count < 1 or selected_count > candidate_count:
        raise ValueError("selected_count must be within candidate_count")
    return np.rint(np.linspace(0, candidate_count - 1, selected_count)).astype(np.int32)


def subset_geometry(geometry: dict[str, np.ndarray], shot_indices: np.ndarray) -> dict[str, np.ndarray]:
    """Subset arrays whose first dimension is the candidate-shot dimension."""
    candidate_count = int(geometry["shot_x_m"].shape[0])
    output = {}
    for key, value in geometry.items():
        array = np.asarray(value)
        output[key] = array[shot_indices] if array.ndim and array.shape[0] == candidate_count else array
    return output


def resample_gathers_to_network(gathers: np.ndarray) -> np.ndarray:
    """Anti-alias resample 1.5 ms/2200 samples to 10 ms/330."""
    if gathers.ndim != 3 or gathers.shape[1] != 2200:
        raise ValueError("Expected gathers with shape [shot, 2200, receiver]")
    output = resample_poly(gathers, up=3, down=20, axis=1, window=("kaiser", 8.0))
    return np.ascontiguousarray(output, dtype=np.float32)


def crop_and_pool_targets(sh: np.ndarray, sg: np.ndarray) -> np.ndarray:
    """Crop 4 km x 2 km and average 10 m x samples to 20 m pixels."""
    if sh.shape != sg.shape or sh.shape[0] < 200 or sh.shape[1] < 520:
        raise ValueError("Formal saturation fields must cover at least [200, 520]")
    stacked = np.stack((sh[:200, 120:520], sg[:200, 120:520])).astype(np.float32)
    return stacked.reshape(2, 200, 200, 2).mean(axis=3, dtype=np.float32)


def crop_and_pool_continuous(field: np.ndarray) -> np.ndarray:
    """Crop one formal field and average only the horizontal sample pairs."""
    if field.ndim != 2 or field.shape[0] < 200 or field.shape[1] < 520:
        raise ValueError("Formal field must cover at least [200, 520]")
    cropped = np.asarray(field[:200, 120:520], dtype=np.float32)
    return cropped.reshape(200, 200, 2).mean(axis=2, dtype=np.float32)
