"""Transparent 2-D staggered-grid SLS solver for numerical QA.

This CPU/Numba implementation is a validation reference, not the production
dataset engine. It uses the same pressure--velocity--memory-variable equations
as :mod:`src.forward.sls_1d` and a second-order staggered spatial stencil.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .sls_1d import relaxation_times, ricker_wavelet


@dataclass(frozen=True)
class SLS2DResult:
    traces: np.ndarray
    time_s: np.ndarray
    receiver_locations_zi_xi: np.ndarray


def _sponge_2d(nz: int, nx: int, width: int, strength: float = 0.025) -> np.ndarray:
    if width <= 0:
        return np.ones((nz, nx), dtype=np.float32)
    if 2 * width >= min(nz, nx):
        raise ValueError("sponge is too wide")
    mz = np.ones(nz, dtype=np.float32)
    mx = np.ones(nx, dtype=np.float32)
    ramp = np.arange(width, 0, -1, dtype=np.float32) / np.float32(width)
    edge = np.exp(-np.float32(strength) * ramp**2).astype(np.float32)
    mz[:width], mz[-width:] = edge, edge[::-1]
    mx[:width], mx[-width:] = edge, edge[::-1]
    return mz[:, None] * mx[None, :]


@njit(cache=True)
def _time_loop(
    pressure: np.ndarray,
    vx: np.ndarray,
    vz: np.ndarray,
    memory: np.ndarray,
    kappa: np.ndarray,
    tau: np.ndarray,
    decay: np.ndarray,
    inv_rho_x: np.ndarray,
    inv_rho_z: np.ndarray,
    mask_p: np.ndarray,
    mask_vx: np.ndarray,
    mask_vz: np.ndarray,
    wavelet: np.ndarray,
    source_z: int,
    source_x: int,
    receivers: np.ndarray,
    dx: float,
    dz: float,
    dt: float,
) -> np.ndarray:
    nz, nx = pressure.shape
    traces = np.zeros((receivers.shape[0], wavelet.size), dtype=np.float32)
    divergence = np.zeros_like(pressure)
    for it in range(wavelet.size):
        for iz in range(nz):
            for ix in range(nx - 1):
                vx[iz, ix] -= (
                    dt * inv_rho_x[iz, ix]
                    * (pressure[iz, ix + 1] - pressure[iz, ix]) / dx
                )
                vx[iz, ix] *= mask_vx[iz, ix]
        for iz in range(nz - 1):
            for ix in range(nx):
                vz[iz, ix] -= (
                    dt * inv_rho_z[iz, ix]
                    * (pressure[iz + 1, ix] - pressure[iz, ix]) / dz
                )
                vz[iz, ix] *= mask_vz[iz, ix]

        divergence.fill(0.0)
        for iz in range(1, nz - 1):
            for ix in range(1, nx - 1):
                divergence[iz, ix] = (
                    (vx[iz, ix] - vx[iz, ix - 1]) / dx
                    + (vz[iz, ix] - vz[iz - 1, ix]) / dz
                )
        for iz in range(1, nz - 1):
            for ix in range(1, nx - 1):
                memory[iz, ix] = (
                    decay[iz, ix] * memory[iz, ix]
                    - tau[iz, ix] * kappa[iz, ix]
                    * (1.0 - decay[iz, ix]) * divergence[iz, ix]
                )
                pressure[iz, ix] -= dt * (
                    kappa[iz, ix] * (tau[iz, ix] + 1.0)
                    * divergence[iz, ix] + memory[iz, ix]
                )
        pressure[source_z, source_x] += wavelet[it] * dt / (dx * dz)
        for iz in range(nz):
            for ix in range(nx):
                pressure[iz, ix] *= mask_p[iz, ix]
                memory[iz, ix] *= mask_p[iz, ix]
        for ir in range(receivers.shape[0]):
            traces[ir, it] = pressure[receivers[ir, 0], receivers[ir, 1]]
    return traces


def simulate_sls_2d(
    vp_m_s: np.ndarray,
    rho_kg_m3: np.ndarray,
    q: float | np.ndarray,
    *,
    dx_m: float,
    dz_m: float,
    dt_s: float,
    nt: int,
    f_ref_hz: float,
    source_location_zi_xi: tuple[int, int],
    receiver_locations_zi_xi: np.ndarray,
    source_delay_s: float = 0.12,
    sponge_width_cells: int = 24,
) -> SLS2DResult:
    """Run a 2-D SLS pressure simulation in a rectangular Cartesian grid."""

    vp = np.ascontiguousarray(vp_m_s, dtype=np.float32)
    rho = np.ascontiguousarray(rho_kg_m3, dtype=np.float32)
    if vp.shape != rho.shape or vp.ndim != 2:
        raise ValueError("vp and rho must be same-shaped 2-D arrays")
    cfl = float(np.max(vp)) * dt_s * np.sqrt(dx_m**-2 + dz_m**-2)
    if cfl >= 1.0:
        raise ValueError(f"2-D CFL condition failed: {cfl:.3f}")
    q_field = np.broadcast_to(np.asarray(q, dtype=np.float64), vp.shape)
    tau_sigma, tau_epsilon = relaxation_times(q_field, f_ref_hz)
    tau = np.ascontiguousarray(tau_epsilon / tau_sigma - 1.0, dtype=np.float32)
    decay = np.ascontiguousarray(np.exp(-dt_s / tau_sigma), dtype=np.float32)
    kappa = np.ascontiguousarray(rho * vp**2, dtype=np.float32)
    inv_rho_x = np.ascontiguousarray(
        2.0 / (rho[:, :-1] + rho[:, 1:]), dtype=np.float32
    )
    inv_rho_z = np.ascontiguousarray(
        2.0 / (rho[:-1, :] + rho[1:, :]), dtype=np.float32
    )
    mask_p = np.ascontiguousarray(
        _sponge_2d(*vp.shape, sponge_width_cells), dtype=np.float32
    )
    mask_vx = np.ascontiguousarray(
        0.5 * (mask_p[:, :-1] + mask_p[:, 1:]), dtype=np.float32
    )
    mask_vz = np.ascontiguousarray(
        0.5 * (mask_p[:-1, :] + mask_p[1:, :]), dtype=np.float32
    )
    receivers = np.ascontiguousarray(receiver_locations_zi_xi, dtype=np.int64)
    if receivers.ndim != 2 or receivers.shape[1] != 2:
        raise ValueError("receiver locations must have shape [receiver, 2]")
    source_z, source_x = map(int, source_location_zi_xi)
    nz, nx = vp.shape
    if not (0 <= source_z < nz and 0 <= source_x < nx):
        raise ValueError("source is outside the model")
    if np.any(receivers[:, 0] < 0) or np.any(receivers[:, 0] >= nz):
        raise ValueError("receiver z index is outside the model")
    if np.any(receivers[:, 1] < 0) or np.any(receivers[:, 1] >= nx):
        raise ValueError("receiver x index is outside the model")

    traces = _time_loop(
        np.zeros_like(vp),
        np.zeros((nz, nx - 1), dtype=np.float32),
        np.zeros((nz - 1, nx), dtype=np.float32),
        np.zeros_like(vp),
        kappa,
        tau,
        decay,
        inv_rho_x,
        inv_rho_z,
        mask_p,
        mask_vx,
        mask_vz,
        np.ascontiguousarray(
            ricker_wavelet(f_ref_hz, dt_s, nt, source_delay_s), dtype=np.float32
        ),
        source_z,
        source_x,
        receivers,
        float(dx_m),
        float(dz_m),
        float(dt_s),
    )
    return SLS2DResult(
        traces=traces,
        time_s=np.arange(nt, dtype=np.float64) * dt_s,
        receiver_locations_zi_xi=receivers,
    )
