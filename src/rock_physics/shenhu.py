"""Regional log-constrained acoustic mapping for the Shenhu scenario."""

from __future__ import annotations

import numpy as np

from .gassmann import saturation_to_acoustic_properties


# Mineral aggregate measured/reported for Shenhu Site SH2.
SHENHU_CONSTANTS = {
    "mineral_bulk_pa": 45.36e9,
    "mineral_shear_pa": 25.24e9,
    "mineral_density": 2660.0,
    "hydrate_bulk_pa": 8.41e9,
    "hydrate_shear_pa": 3.54e9,
    "hydrate_density": 922.0,
    "brine_bulk_pa": 2.50e9,
    "brine_density": 1032.0,
    # In-situ methane at pressure is much stiffer than atmospheric gas.
    "gas_bulk_pa": 0.08e9,
    "gas_density": 120.0,
    "critical_phi": 0.52,
    "frame_exponent": 3.2,
}


def shenhu_log_calibrated_properties(
    phi: np.ndarray,
    Sh: np.ndarray,
    Sg: np.ndarray,
    water_mask: np.ndarray,
    hydrate_response_scale: float = 0.30,
    gas_response_scale: float = 0.58,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Vp and density with regional log-response calibration.

    The underlying model is a soft-frame Gassmann proxy.  Hydrate and gas
    velocity anomalies are scaled separately because Shenhu logs indicate
    pore-filling hydrate and patchily distributed free gas; a uniform Wood
    mixture otherwise over-predicts the gas-related velocity collapse.

    The default factors were selected against two independent regional log
    constraints: an approximately 2070 m/s hydrate-reservoir mean at SH2 and
    an approximately 1725 m/s hydrate/free-gas coexistence mean reported at
    SHSC-4J1.  These are scenario calibration targets, not universal laws.
    """
    zeros = np.zeros_like(Sh, dtype=np.float32)
    vp_brine, _ = saturation_to_acoustic_properties(
        phi, zeros, zeros, water_mask, **SHENHU_CONSTANTS
    )
    vp_hydrate_raw, _ = saturation_to_acoustic_properties(
        phi, Sh, zeros, water_mask, **SHENHU_CONSTANTS
    )
    vp_gas_raw, rho = saturation_to_acoustic_properties(
        phi, Sh, Sg, water_mask, **SHENHU_CONSTANTS
    )

    hydrate_delta = vp_hydrate_raw - vp_brine
    gas_delta = vp_gas_raw - vp_hydrate_raw
    vp = vp_brine + hydrate_response_scale * hydrate_delta + gas_response_scale * gas_delta
    vp[water_mask] = 1500.0
    rho[water_mask] = 1025.0
    return vp.astype(np.float32), rho.astype(np.float32)
