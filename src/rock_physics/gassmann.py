"""NGH-Seis hydrate/free-gas rock-physics mapping.

This is a transparent controlled model for pipeline validation.  Its constants
must be calibrated and justified before producing the released benchmark.
"""

from __future__ import annotations

import numpy as np


def _vrh(values: tuple[float, float], fractions: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    f0, f1 = fractions
    v0, v1 = values
    voigt = f0 * v0 + f1 * v1
    reuss = 1.0 / (f0 / v0 + f1 / v1)
    return 0.5 * (voigt + reuss)


def saturation_to_acoustic_properties(
    phi: np.ndarray,
    Sh: np.ndarray,
    Sg: np.ndarray,
    water_mask: np.ndarray,
    *,
    mineral_bulk_pa: float = 32.0e9,
    mineral_shear_pa: float = 28.0e9,
    mineral_density: float = 2650.0,
    hydrate_bulk_pa: float = 7.9e9,
    hydrate_shear_pa: float = 3.3e9,
    hydrate_density: float = 920.0,
    brine_bulk_pa: float = 2.35e9,
    brine_density: float = 1030.0,
    gas_bulk_pa: float = 0.08e9,
    gas_density: float = 120.0,
    critical_phi: float = 0.52,
    frame_exponent: float = 3.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Map porosity and pore saturations to Vp and bulk density.

    Hydrate is treated as a pore-filling solid. Remaining pore fluid is a
    Wood mixture of brine and gas. Gassmann substitution saturates a soft
    critical-porosity dry-frame proxy.
    """
    phi = np.asarray(phi, dtype=np.float64)
    Sh = np.asarray(Sh, dtype=np.float64)
    Sg = np.asarray(Sg, dtype=np.float64)
    water_mask = np.asarray(water_mask, dtype=bool)

    # SI units.
    K_m, G_m, rho_m = mineral_bulk_pa, mineral_shear_pa, mineral_density
    K_h, G_h, rho_h = hydrate_bulk_pa, hydrate_shear_pa, hydrate_density
    K_b, rho_b = brine_bulk_pa, brine_density
    K_g, rho_g = gas_bulk_pa, gas_density

    mineral_volume = np.maximum(1.0 - phi, 1.0e-8)
    hydrate_volume = phi * Sh
    composite_solid = mineral_volume + hydrate_volume
    f_m = mineral_volume / composite_solid
    f_h = hydrate_volume / composite_solid
    K_s = _vrh((K_m, K_h), (f_m, f_h))
    G_s = _vrh((G_m, G_h), (f_m, f_h))

    effective_phi = np.clip(phi * (1.0 - Sh), 1.0e-5, critical_phi * 0.995)
    frame_factor = np.maximum(1.0 - effective_phi / critical_phi, 0.005) ** frame_exponent
    K_dry = K_s * frame_factor
    G_dry = G_s * frame_factor

    remaining = np.maximum(1.0 - Sh, 1.0e-8)
    gas_fraction = np.clip(Sg / remaining, 0.0, 0.95)
    brine_fraction = 1.0 - gas_fraction
    K_fl = 1.0 / (brine_fraction / K_b + gas_fraction / K_g)
    rho_fl = brine_fraction * rho_b + gas_fraction * rho_g

    denom = effective_phi / K_fl + (1.0 - effective_phi) / K_s - K_dry / (K_s * K_s)
    K_sat = K_dry + (1.0 - K_dry / K_s) ** 2 / np.maximum(denom, 1.0e-20)
    rho = mineral_volume * rho_m + hydrate_volume * rho_h + effective_phi * rho_fl
    vp = np.sqrt(np.maximum(K_sat + 4.0 * G_dry / 3.0, 1.0) / rho)

    vp[water_mask] = 1500.0
    rho[water_mask] = 1025.0
    return vp.astype(np.float32), rho.astype(np.float32)
