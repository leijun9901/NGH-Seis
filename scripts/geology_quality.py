from __future__ import annotations

import argparse

import hashlib

import json

from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]

DEFAULT_FOLDER = PROJECT / "outputs/ngh_seis_v1_geology_source/geology"

DEFAULT_CONFIG = PROJECT / "configs/ngh_seis_geology.json"

VIEW_ARRAY_KEYS = (
    "Vp_view",
    "rho_view",
    "phi_view",
    "Sh_view",
    "Sg_view",
    "AI_view",
    "lithology_view",
)

def _rowwise_lateral_correlation(field: np.ndarray, rows: slice, lag: int) -> float:
    array = np.asarray(field[rows, :95], dtype=np.float64)
    array -= array.mean(axis=1, keepdims=True)
    denominator = float(np.sum(array * array))
    if denominator <= 1.0e-20 or lag >= array.shape[1]:
        return 0.0
    return float(np.sum(array[:, :-lag] * array[:, lag:]) / denominator)

def _decorrelation_distance_m(
    field: np.ndarray,
    rows: slice,
    *,
    dx_m: float = 37.5,
    threshold: float = 0.20,
) -> float:
    for lag in range(1, 25):
        if _rowwise_lateral_correlation(field, rows, lag) <= threshold:
            return lag * dx_m
    return 24 * dx_m

def _sample_audit(
    sample_index: int,
    arrays: dict[str, np.ndarray],
    metadata: dict,
    config: dict,
) -> dict:
    saturation_sum_max = float(np.max(arrays["Sh_full"] + arrays["Sg_full"]))
    sediment = arrays["phi_full"] < 0.99
    minimum_sediment_vp = float(np.min(arrays["Vp_full"][sediment]))
    gas_bearing = arrays["Sg_full"] > 0.0
    if not np.any(gas_bearing):
        raise ValueError(f"sample {sample_index} has no gas-bearing cells")
    gas_bearing_vp_p01 = float(np.quantile(arrays["Vp_full"][gas_bearing], 0.01))
    effective_max_hz = float(
        config["forward_sampling"]["effective_max_frequency_hz"]
    )
    points_per_wavelength = minimum_sediment_vp / (
        effective_max_hz * float(config["grid"]["dx_m"])
    )
    finite = bool(all(np.all(np.isfinite(arrays[key])) for key in VIEW_ARRAY_KEYS))
    padding_zero = bool(all(np.all(arrays[key][:, 95] == 0.0) for key in VIEW_ARRAY_KEYS))
    return {
        "sample_index": sample_index,
        "style": metadata["style"],
        "water_depth_mean_m": metadata["water_depth_mean_m"],
        "bsr_depth_mbsf_mean": metadata["bsr_depth_mbsf_mean"],
        "hydrate_mean_in_zone": metadata["hydrate_mean_in_zone"],
        "hydrate_max": metadata["hydrate_max"],
        "gas_mean_in_zone": metadata["gas_mean_in_zone"],
        "gas_max": metadata["gas_max"],
        "hydrate_q95_positive": metadata["hydrate_q95_positive"],
        "gas_q95_positive": metadata["gas_q95_positive"],
        "hydrate_fraction_within_one_percent_of_max": metadata[
            "hydrate_fraction_within_one_percent_of_max"
        ],
        "gas_fraction_within_one_percent_of_max": metadata[
            "gas_fraction_within_one_percent_of_max"
        ],
        "gas_zone_thickness_m": metadata["gas_zone_thickness_m"],
        "coexistence_enabled": metadata["coexistence_enabled"],
        "coexistence_thickness_m": metadata["coexistence_thickness_m"],
        "coexistence_pixel_fraction": metadata["coexistence_pixel_fraction"],
        "saturation_sum_max": saturation_sum_max,
        "minimum_sediment_vp_m_s": minimum_sediment_vp,
        "gas_bearing_vp_p01_m_s": gas_bearing_vp_p01,
        "minimum_points_per_effective_wavelength": points_per_wavelength,
        "full_shape": list(arrays["Vp_full"].shape),
        "view_shape": list(arrays["Vp_view"].shape),
        "finite": finite,
        "padding_zero": padding_zero,
        "loss_mask_fraction": float(arrays["loss_mask"].mean()),
        "seafloor_p95_absolute_slope": float(metadata["seafloor_p95_absolute_slope"]),
        "Sh_lag150m_correlation": _rowwise_lateral_correlation(
            arrays["Sh_view"], slice(124, 244), 4
        ),
        "Sg_lag150m_correlation": _rowwise_lateral_correlation(
            arrays["Sg_view"], slice(214, 334), 4
        ),
        "Sh_decorrelation_distance_m": _decorrelation_distance_m(
            arrays["Sh_view"], slice(124, 244)
        ),
        "Sg_decorrelation_distance_m": _decorrelation_distance_m(
            arrays["Sg_view"], slice(214, 334)
        ),
    }
