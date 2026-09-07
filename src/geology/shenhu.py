"""GMGS6-SH02 log-constrained 2-D scenario generator."""

from __future__ import annotations

import numpy as np


def _smooth_curve(rng: np.random.Generator, nx: int, correlation_cells: int = 31) -> np.ndarray:
    raw = rng.normal(size=nx)
    kernel = np.hanning(correlation_cells)
    kernel /= kernel.sum()
    curve = np.convolve(raw, kernel, mode="same")
    curve -= curve.mean()
    curve /= max(curve.std(), 1.0e-8)
    return curve


def _bounded_field(
    rng: np.random.Generator,
    zz: np.ndarray,
    x_count: int,
    top: np.ndarray,
    bottom: np.ndarray,
    mean: float,
    maximum: float,
    phase: float,
) -> np.ndarray:
    center = 0.5 * (top + bottom)
    half = np.maximum(0.5 * (bottom - top), 1.0)
    vertical = 0.72 + 0.42 * np.cos(np.pi * (zz - center) / half)
    lateral = 1.0 + 0.22 * np.sin(np.linspace(phase, phase + 4.0 * np.pi, x_count))[None, :]
    lateral += 0.10 * _smooth_curve(rng, x_count)[None, :]
    mask = (zz >= top) & (zz < bottom)
    field = np.maximum(vertical * lateral, 0.05) * mask
    if np.any(mask):
        field *= mean / field[mask].mean()
    return np.clip(field, 0.0, maximum).astype(np.float32)


def generate_shenhu_gmgs6_sh02(
    nz: int = 401,
    nx: int = 321,
    dz: float = 5.0,
    dx: float = 10.0,
    seed: int = 20260812,
) -> dict[str, np.ndarray]:
    """Generate a laterally heterogeneous realization tied to GMGS6-SH02.

    Reference vertical intervals below seafloor:
      207.8-253.4 m: hydrate, mean phi=0.373, mean Sh=0.31
      253.4-278.0 m: hydrate+gas, phi=0.346, Sh=0.10, Sg=0.13
      278.0-297.0 m: free gas, phi=0.347, Sg=0.073
    """
    rng = np.random.default_rng(seed)
    x = np.arange(nx, dtype=np.float64) * dx
    z = np.arange(nz, dtype=np.float64) * dz
    zz = z[:, None]

    # The reference well water depth is 1225.2 m. Gentle bathymetry makes a
    # 2-D survey line without changing the well-tied vertical zonation.
    seafloor = 1225.2 + 8.0 * np.sin(2.0 * np.pi * x / x[-1] + 0.3)
    seafloor += 2.0 * _smooth_curve(rng, nx)
    sf = seafloor[None, :]
    mbsf = zz - sf
    water_mask = mbsf < 0.0
    vertical_log_variation = _smooth_curve(rng, nz, correlation_cells=9)[:, None]

    # Regional compaction trend, followed by log-tied zone means.
    phi = 0.405 - 0.00020 * np.maximum(mbsf, 0.0)
    phi += 0.008 * _smooth_curve(rng, nx)[None, :]
    phi += 0.004 * vertical_log_variation
    phi = np.clip(phi, 0.30, 0.46)

    relief = 3.0 * _smooth_curve(rng, nx)
    h_top = sf + 207.8 + relief[None, :]
    h_bottom = sf + 253.4 + relief[None, :]
    mix_bottom = sf + 278.0 + relief[None, :]
    gas_bottom = sf + 297.0 + relief[None, :]

    zone_h = (zz >= h_top) & (zz < h_bottom)
    zone_m = (zz >= h_bottom) & (zz < mix_bottom)
    zone_g = (zz >= mix_bottom) & (zz < gas_bottom)

    # Set porosity means from NMR/sonic log interpretation and retain small
    # vertical/lateral variability within each interval.
    for mask, target in ((zone_h, 0.373), (zone_m, 0.346), (zone_g, 0.347)):
        local = target + 0.008 * _smooth_curve(rng, nx)[None, :]
        local = local + 0.004 * vertical_log_variation
        local = local + (target - float(local[mask].mean()))
        phi = np.where(mask, np.clip(local, target - 0.025, target + 0.025), phi)
    phi[water_mask] = 1.0

    Sh_h = _bounded_field(rng, zz, nx, h_top, h_bottom, 0.31, 0.545, 0.2)
    Sh_m = _bounded_field(rng, zz, nx, h_bottom, mix_bottom, 0.10, 0.22, 1.1)
    Sg_m = _bounded_field(rng, zz, nx, h_bottom, mix_bottom, 0.13, 0.32, 2.0)
    Sg_g = _bounded_field(rng, zz, nx, mix_bottom, gas_bottom, 0.073, 0.18, 2.8)
    Sh = Sh_h + Sh_m
    Sg = Sg_m + Sg_g

    return {
        "x_m": x.astype(np.float32),
        "z_m": z.astype(np.float32),
        "seafloor_m": seafloor.astype(np.float32),
        "hydrate_top_m": h_top[0].astype(np.float32),
        "mixed_top_m": h_bottom[0].astype(np.float32),
        "free_gas_top_m": mix_bottom[0].astype(np.float32),
        "free_gas_bottom_m": gas_bottom[0].astype(np.float32),
        "phi": phi.astype(np.float32),
        "Sh": Sh.astype(np.float32),
        "Sg": Sg.astype(np.float32),
        "hydrate_mask": (Sh > 0).astype(np.uint8),
        "gas_mask": (Sg > 0).astype(np.uint8),
        "zone_hydrate": zone_h.astype(np.uint8),
        "zone_mixed": zone_m.astype(np.uint8),
        "zone_free_gas": zone_g.astype(np.uint8),
    }
