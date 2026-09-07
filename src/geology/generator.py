"""Generate a compact 2-D marine hydrate/free-gas model."""

from __future__ import annotations

import numpy as np


def _smooth_random_curve(rng: np.random.Generator, nx: int, scale: float) -> np.ndarray:
    knots_x = np.linspace(0, nx - 1, 12)
    knots_y = rng.normal(0.0, 1.0, knots_x.size)
    curve = np.interp(np.arange(nx), knots_x, knots_y)
    kernel = np.ones(21, dtype=np.float64) / 21.0
    return scale * np.convolve(curve, kernel, mode="same")


def generate_geology(
    nz: int = 201,
    nx: int = 321,
    dz: float = 10.0,
    dx: float = 10.0,
    seed: int = 20260812,
) -> dict[str, np.ndarray]:
    """Return phi, Sh, Sg and masks on a depth-by-distance grid.

    Sh and Sg are fractions of total pore volume.  They are spatially
    separated around a BSR-like boundary in the controlled model.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(nx, dtype=np.float64) * dx
    z = np.arange(nz, dtype=np.float64) * dz

    seafloor = 390.0 + 35.0 * np.sin(2.0 * np.pi * x / x[-1])
    seafloor += _smooth_random_curve(rng, nx, 10.0)
    bsr = seafloor + 500.0 + 35.0 * np.sin(2.0 * np.pi * x / 1450.0 + 0.4)
    bsr += _smooth_random_curve(rng, nx, 12.0)

    zz = z[:, None]
    sf = seafloor[None, :]
    bsr2d = bsr[None, :]
    below_seafloor = zz >= sf

    burial = np.maximum(zz - sf, 0.0)
    phi = 0.43 * np.exp(-burial / 2600.0) + 0.035
    phi += 0.008 * np.sin(2.0 * np.pi * x[None, :] / 900.0)
    phi = np.clip(phi, 0.24, 0.48)
    phi[~below_seafloor] = 1.0

    # Hydrate occupies a laterally variable interval above the BSR.
    hydrate_top = bsr2d - (235.0 + 30.0 * np.sin(2.0 * np.pi * x[None, :] / 1100.0))
    h_center = 0.5 * (hydrate_top + bsr2d)
    h_sigma = np.maximum((bsr2d - hydrate_top) / 2.5, 20.0)
    h_lateral = 0.62 + 0.25 * np.sin(2.0 * np.pi * x[None, :] / 1250.0 + 0.7)
    h_lateral += 0.12 * np.cos(2.0 * np.pi * x[None, :] / 530.0)
    Sh = 0.50 * np.exp(-0.5 * ((zz - h_center) / h_sigma) ** 2) * h_lateral
    hydrate_mask = (zz >= hydrate_top) & (zz < bsr2d) & below_seafloor
    Sh = np.clip(Sh, 0.0, 0.58) * hydrate_mask

    # Free gas is patchier and thinner below the BSR.
    gas_bottom = bsr2d + 135.0 + 20.0 * np.cos(2.0 * np.pi * x[None, :] / 1000.0)
    g_center = bsr2d + 48.0
    g_sigma = 42.0
    lenses = (
        np.exp(-0.5 * ((x[None, :] - 950.0) / 360.0) ** 2)
        + 0.85 * np.exp(-0.5 * ((x[None, :] - 2300.0) / 430.0) ** 2)
    )
    Sg = 0.105 * np.exp(-0.5 * ((zz - g_center) / g_sigma) ** 2) * lenses
    gas_mask = (zz >= bsr2d) & (zz <= gas_bottom) & (lenses > 0.18) & below_seafloor
    Sg = np.clip(Sg, 0.0, 0.12) * gas_mask

    # Preserve a brine fraction in every pore cell.
    excess = np.maximum(Sh + Sg - 0.90, 0.0)
    Sh = Sh - excess * Sh / np.maximum(Sh + Sg, 1.0e-12)
    Sg = Sg - excess * Sg / np.maximum(Sh + Sg, 1.0e-12)

    return {
        "x_m": x.astype(np.float32),
        "z_m": z.astype(np.float32),
        "seafloor_m": seafloor.astype(np.float32),
        "bsr_m": bsr.astype(np.float32),
        "phi": phi.astype(np.float32),
        "Sh": Sh.astype(np.float32),
        "Sg": Sg.astype(np.float32),
        "hydrate_mask": hydrate_mask.astype(np.uint8),
        "gas_mask": gas_mask.astype(np.uint8),
    }
