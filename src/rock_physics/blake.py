"""Literature-bounded acoustic prior for the NGH-Seis geological models.

This module deliberately avoids the term "well calibrated": the downloaded
EW0008 volume is aligned to ODP Leg 164 regional knowledge, but no well tie is
used here.  The soft-frame parameters reproduce the first-order velocity and
density envelope of high-porosity deep-marine muds and keep every constant in
the released metadata.
"""

from __future__ import annotations

import numpy as np

from .gassmann import saturation_to_acoustic_properties


BLAKE_SOFT_MUD_CONSTANTS = {
    "mineral_bulk_pa": 36.0e9,
    "mineral_shear_pa": 30.0e9,
    "mineral_density": 2700.0,
    "hydrate_bulk_pa": 7.9e9,
    "hydrate_shear_pa": 3.3e9,
    "hydrate_density": 920.0,
    "brine_bulk_pa": 2.35e9,
    "brine_density": 1030.0,
    "gas_bulk_pa": 0.08e9,
    "gas_density": 120.0,
    "critical_phi": 0.75,
    "frame_exponent": 3.2,
}


def blake_prior_acoustic_properties(
    phi: np.ndarray,
    Sh: np.ndarray,
    Sg: np.ndarray,
    water_mask: np.ndarray,
    *,
    hydrate_response_scale: float = 1.0,
    gas_response_scale: float = 0.85,
    gas_patchiness: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert porosity and pore saturation to Vp and bulk density.

    ``gas_patchiness=None`` uses the default empirical anomaly scaling.
    Otherwise, the gas response is interpolated in P-wave modulus between the
    Gassmann-Wood (uniform pore-fluid pressure) and Gassmann-Hill (separate
    brine- and gas-saturated patches) end members.  The latter is an explicit
    rock-physics nuisance parameter rather than an image augmentation.
    """
    zeros = np.zeros_like(Sh, dtype=np.float32)
    vp_brine, _ = saturation_to_acoustic_properties(
        phi, zeros, zeros, water_mask, **BLAKE_SOFT_MUD_CONSTANTS
    )
    vp_hydrate, rho_hydrate = saturation_to_acoustic_properties(
        phi, Sh, zeros, water_mask, **BLAKE_SOFT_MUD_CONSTANTS
    )
    vp_raw, rho = saturation_to_acoustic_properties(
        phi, Sh, Sg, water_mask, **BLAKE_SOFT_MUD_CONSTANTS
    )
    if gas_patchiness is None:
        vp = (
            vp_brine
            + hydrate_response_scale * (vp_hydrate - vp_brine)
            + gas_response_scale * (vp_raw - vp_hydrate)
        )
    else:
        patchiness = float(np.clip(gas_patchiness, 0.0, 1.0))
        remaining = np.maximum(1.0 - np.asarray(Sh, dtype=np.float64), 1.0e-8)
        gas_fraction = np.clip(np.asarray(Sg, dtype=np.float64) / remaining, 0.0, 0.95)

        # A fully gas-saturated patch shares the hydrate-bearing frame with
        # the brine patch.  Hill's isotropic mixture is harmonic in
        # K_sat + 4G/3 when the patch shear moduli are equal.
        sg_gas_patch = (0.95 * remaining).astype(np.float32)
        vp_gas_patch, rho_gas_patch = saturation_to_acoustic_properties(
            phi, Sh, sg_gas_patch, water_mask, **BLAKE_SOFT_MUD_CONSTANTS
        )
        pmod_brine = np.asarray(rho_hydrate, dtype=np.float64) * np.asarray(vp_hydrate, dtype=np.float64) ** 2
        pmod_gas = np.asarray(rho_gas_patch, dtype=np.float64) * np.asarray(vp_gas_patch, dtype=np.float64) ** 2
        pmod_hill = 1.0 / (
            (1.0 - gas_fraction) / np.maximum(pmod_brine, 1.0)
            + gas_fraction / np.maximum(pmod_gas, 1.0)
        )
        pmod_wood = np.asarray(rho, dtype=np.float64) * np.asarray(vp_raw, dtype=np.float64) ** 2
        pmod_mixed = (1.0 - patchiness) * pmod_wood + patchiness * pmod_hill
        vp_gas_mixed = np.sqrt(np.maximum(pmod_mixed, 1.0) / np.maximum(rho, 1.0))
        vp = (
            np.asarray(vp_brine, dtype=np.float64)
            + hydrate_response_scale
            * (np.asarray(vp_hydrate, dtype=np.float64) - np.asarray(vp_brine, dtype=np.float64))
            + (vp_gas_mixed - np.asarray(vp_hydrate, dtype=np.float64))
        )
    vp[water_mask] = 1500.0
    rho[water_mask] = 1025.0
    return vp.astype(np.float32), rho.astype(np.float32)
