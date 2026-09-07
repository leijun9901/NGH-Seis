"""High-resolution marine stratigraphy derived from log-constrained models."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

from src.rock_physics.shenhu import shenhu_log_calibrated_properties


def _resample_field(
    field: np.ndarray,
    *,
    original_dx_m: float,
    original_dz_m: float,
    x_start_m: float,
    nx: int,
    nz: int,
    dx_m: float,
    dz_m: float,
) -> np.ndarray:
    z_index = np.arange(nz, dtype=np.float64) * dz_m / original_dz_m
    x_index = (x_start_m + np.arange(nx, dtype=np.float64) * dx_m) / original_dx_m
    zz, xx = np.meshgrid(z_index, x_index, indexing="ij")
    return map_coordinates(
        np.asarray(field, dtype=np.float32), [zz, xx], order=1, mode="nearest"
    ).astype(np.float32)


def _unit_smooth_noise(rng: np.random.Generator, size: int, sigma: float) -> np.ndarray:
    values = gaussian_filter1d(rng.normal(size=size), sigma=sigma, mode="reflect")
    values -= values.mean()
    values /= max(values.std(), 1e-8)
    return values


def add_log_constrained_thin_beds(
    phi: np.ndarray,
    seafloor_m: np.ndarray,
    Sh: np.ndarray,
    Sg: np.ndarray,
    *,
    stratigraphic_shift_m: np.ndarray | None = None,
    dz_m: float,
    seed: int,
) -> np.ndarray:
    """Add physical porosity laminae before rock-physics conversion.

    The perturbation is stratigraphically coordinated relative to the local
    seafloor, with 8--80 m vertical scales and slowly varying lateral phase.
    Zone means are restored after perturbation so borehole-scale porosity
    constraints remain unchanged.
    """
    base = np.asarray(phi, dtype=np.float32)
    rng = np.random.default_rng(seed)
    nz, nx = base.shape
    relative_depth_m = (
        np.arange(nz, dtype=np.float32)[:, None] * np.float32(dz_m)
        - np.asarray(seafloor_m, dtype=np.float32)[None, :]
    )
    if stratigraphic_shift_m is None:
        stratigraphic_shift = np.zeros(nx, dtype=np.float32)
    else:
        stratigraphic_shift = np.asarray(stratigraphic_shift_m, dtype=np.float32)
        if stratigraphic_shift.shape != (nx,) or np.any(~np.isfinite(stratigraphic_shift)):
            raise ValueError("stratigraphic_shift_m must be a finite [nx] array")
    # Constant stratigraphic coordinates follow regional dip, boundary relief,
    # and fault throw.  A positive displacement moves a bed deeper.
    stratigraphic_depth_m = relative_depth_m - stratigraphic_shift[None, :]
    max_relative_cells = int(np.ceil(max(1000.0, float(relative_depth_m.max()) + 200.0) / dz_m))
    # A band-limited random vertical log produces real impedance contrasts,
    # not texture pasted onto the final seismic image.
    log = rng.normal(size=max_relative_cells + 300)
    fine = gaussian_filter1d(log, sigma=0.65, mode="reflect")
    fine -= gaussian_filter1d(fine, sigma=5.0, mode="reflect")
    medium = _unit_smooth_noise(rng, log.size, 2.0)
    profile = fine / max(fine.std(), 1e-8) + 0.45 * medium
    profile /= max(profile.std(), 1e-8)

    lateral_warp_m = (
        10.0 * _unit_smooth_noise(rng, nx, 45.0)
        + 4.0 * _unit_smooth_noise(rng, nx, 14.0)
    )
    coordinate = (
        stratigraphic_depth_m / dz_m
        + lateral_warp_m[None, :] / dz_m
        + 120.0
    )
    stratigraphy = np.interp(
        coordinate, np.arange(profile.size, dtype=np.float32), profile
    ).astype(np.float32)
    amplitude = 0.010 + 0.006 * (
        0.5 + 0.5 * _unit_smooth_noise(rng, nx, 60.0) / 3.0
    )
    amplitude = np.clip(amplitude, 0.007, 0.017)
    sediment = relative_depth_m >= 0.0
    result = base + sediment * stratigraphy * amplitude[None, :]

    hydrate = (Sh > 0.01) & (Sg <= 0.005)
    mixed = (Sh > 0.01) & (Sg > 0.005)
    gas = (Sh <= 0.01) & (Sg > 0.005)
    for mask in (hydrate, mixed, gas):
        if np.any(mask):
            result[mask] += float(base[mask].mean() - result[mask].mean())
    result = np.clip(result, 0.28, 0.55).astype(np.float32)
    result[~sediment] = 1.0
    return result


def build_marine_highres_model(
    physical_10m: dict[str, np.ndarray],
    generation_metadata: dict,
    *,
    x_start_original_m: float = 1150.0,
    nx: int = 1024,
    nz: int = 512,
    dx_m: float = 4.0,
    dz_m: float = 4.0,
    seed: int = 2026081501,
) -> dict[str, np.ndarray]:
    """Return a 4 m, 4.096 km by 2.048 km marine acoustic model."""
    kwargs = {
        "original_dx_m": 10.0,
        "original_dz_m": 10.0,
        "x_start_m": x_start_original_m,
        "nx": nx,
        "nz": nz,
        "dx_m": dx_m,
        "dz_m": dz_m,
    }
    phi = _resample_field(physical_10m["phi"], **kwargs)
    Sh = np.clip(_resample_field(physical_10m["Sh"], **kwargs), 0.0, 0.92)
    Sg = np.clip(_resample_field(physical_10m["Sg"], **kwargs), 0.0, 0.92)
    total = Sh + Sg
    excessive = total > 0.92
    Sh[excessive] *= 0.92 / total[excessive]
    Sg[excessive] *= 0.92 / total[excessive]
    x_original = x_start_original_m + np.arange(nx, dtype=np.float32) * dx_m
    seafloor = np.interp(
        x_original,
        np.arange(physical_10m["seafloor_m"].size, dtype=np.float32) * 10.0,
        physical_10m["seafloor_m"],
    ).astype(np.float32)
    source_shift = np.asarray(
        physical_10m.get(
            "stratigraphic_shift_m",
            np.zeros(physical_10m["seafloor_m"].shape, dtype=np.float32),
        ),
        dtype=np.float32,
    )
    stratigraphic_shift = np.interp(
        x_original,
        np.arange(source_shift.size, dtype=np.float32) * 10.0,
        source_shift,
    ).astype(np.float32)
    source_fault_displacement = np.asarray(
        physical_10m.get(
            "fault_displacement_m",
            np.zeros(physical_10m["seafloor_m"].shape, dtype=np.float32),
        ),
        dtype=np.float32,
    )
    fault_displacement = np.interp(
        x_original,
        np.arange(source_fault_displacement.size, dtype=np.float32) * 10.0,
        source_fault_displacement,
    ).astype(np.float32)
    water = np.arange(nz, dtype=np.float32)[:, None] * dz_m < seafloor[None, :]
    Sh[water] = 0.0
    Sg[water] = 0.0
    phi = add_log_constrained_thin_beds(
        phi,
        seafloor,
        Sh,
        Sg,
        stratigraphic_shift_m=stratigraphic_shift,
        dz_m=dz_m,
        seed=seed,
    )
    zeros = np.zeros_like(Sh)
    calibration = {
        "hydrate_response_scale": generation_metadata["hydrate_response_scale"],
        "gas_response_scale": generation_metadata["gas_response_scale"],
    }
    vp_brine, rho_brine = shenhu_log_calibrated_properties(
        phi, zeros, zeros, water, **calibration
    )
    vp_true, rho_true = shenhu_log_calibrated_properties(
        phi, Sh, Sg, water, **calibration
    )
    return {
        "x_m": np.arange(nx, dtype=np.float32) * dx_m,
        "x_original_m": x_original.astype(np.float32),
        "z_m": np.arange(nz, dtype=np.float32) * dz_m,
        "seafloor_m": seafloor,
        "stratigraphic_shift_m": stratigraphic_shift,
        "fault_displacement_m": fault_displacement,
        "phi": phi,
        "Sh": Sh.astype(np.float32),
        "Sg": Sg.astype(np.float32),
        "vp_brine": vp_brine,
        "rho_brine": rho_brine,
        "vp_true": vp_true,
        "rho_true": rho_true,
        "water_mask": water.astype(np.uint8),
    }
