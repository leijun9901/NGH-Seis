"""Physically sample full models into an EW0008-aligned network view."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def extract_seafloor_relative_view(
    field: np.ndarray,
    seafloor_m: np.ndarray,
    *,
    source_dx_m: float,
    source_dz_m: float,
    x_start_m: float = 1310.0,
    trace_count: int = 95,
    padded_trace_count: int = 96,
    output_dx_m: float = 37.5,
    output_dz_m: float = 3.0,
    output_nz: int = 512,
    water_samples: int = 64,
    interpolation_order: int = 1,
    antialias_lateral: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return a 512x96 seafloor-relative view and its valid-data mask.

    The 95 physical traces are sampled at EW0008 inline spacing. One zero
    column is appended for neural-network divisibility and is always invalid.
    Continuous fields are low-pass filtered before lateral decimation.
    """

    array = np.asarray(field, dtype=np.float32)
    seafloor = np.asarray(seafloor_m, dtype=np.float64)
    if array.ndim != 2 or seafloor.shape != (array.shape[1],):
        raise ValueError("field must be [z,x] and seafloor_m must match x")
    if padded_trace_count < trace_count:
        raise ValueError("padded_trace_count cannot be smaller than trace_count")
    x_out_m = x_start_m + np.arange(trace_count, dtype=np.float64) * output_dx_m
    if x_out_m[0] < 0.0 or x_out_m[-1] > (array.shape[1] - 1) * source_dx_m:
        raise ValueError("requested EW0008-aligned view exceeds the source model width")
    source_x = np.arange(array.shape[1], dtype=np.float64) * source_dx_m
    seafloor_out = np.interp(x_out_m, source_x, seafloor)
    z_relative_m = (
        np.arange(output_nz, dtype=np.float64) - float(water_samples)
    ) * output_dz_m
    z_query_m = seafloor_out[None, :] + z_relative_m[:, None]
    if float(z_query_m.min()) < 0.0 or float(z_query_m.max()) > (array.shape[0] - 1) * source_dz_m:
        raise ValueError("requested seafloor-relative view exceeds model depth")

    work = array
    decimation_ratio = output_dx_m / source_dx_m
    if antialias_lateral and decimation_ratio > 1.0:
        # A fixed half-decimation Gaussian is conservative and label-free.
        work = gaussian_filter(
            work, sigma=(0.0, 0.5 * decimation_ratio), mode="nearest"
        ).astype(np.float32)
    rows = z_query_m / source_dz_m
    cols = np.broadcast_to(
        (x_out_m / source_dx_m)[None, :], rows.shape
    )
    sampled = map_coordinates(
        work, [rows, cols], order=interpolation_order, mode="nearest"
    ).astype(np.float32)
    pad = padded_trace_count - trace_count
    output = np.pad(sampled, ((0, 0), (0, pad)), mode="constant")
    valid = np.zeros_like(output, dtype=np.uint8)
    valid[:, :trace_count] = 1
    metadata = {
        "shape": [output_nz, padded_trace_count],
        "physical_trace_count": trace_count,
        "padding_trace_count": pad,
        "output_spacing_m": [output_dz_m, output_dx_m],
        "x_start_m": x_start_m,
        "x_end_m": float(x_out_m[-1]),
        "vertical_relative_range_m": [float(z_relative_m[0]), float(z_relative_m[-1])],
        "seafloor_row": water_samples,
        "lateral_antialias_sigma_source_cells": (
            0.5 * decimation_ratio if antialias_lateral and decimation_ratio > 1.0 else 0.0
        ),
    }
    return output, valid, metadata
