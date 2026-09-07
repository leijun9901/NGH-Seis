"""Constant-density Born modeling for SIGMA-style scattered-data RTM."""

from __future__ import annotations

import numpy as np
from numba import njit

from .acoustic import damping_mask, ricker_wavelet


@njit(cache=True)
def _step_constant_density(
    old: np.ndarray,
    current: np.ndarray,
    new: np.ndarray,
    velocity_squared: np.ndarray,
    dt2: float,
    idx12: float,
    idz12: float,
) -> None:
    nz, nx = current.shape
    for iz in range(2, nz - 2):
        for ix in range(2, nx - 2):
            center = current[iz, ix]
            lap_x = (
                -current[iz, ix + 2] + 16.0 * current[iz, ix + 1]
                - 30.0 * center + 16.0 * current[iz, ix - 1]
                - current[iz, ix - 2]
            ) * idx12
            lap_z = (
                -current[iz + 2, ix] + 16.0 * current[iz + 1, ix]
                - 30.0 * center + 16.0 * current[iz - 1, ix]
                - current[iz - 2, ix]
            ) * idz12
            new[iz, ix] = (
                2.0 * center - old[iz, ix]
                + velocity_squared[iz, ix] * dt2 * (lap_x + lap_z)
            )


@njit(cache=True)
def _simulate_one_born_shot(
    velocity_squared: np.ndarray,
    slowness_squared_perturbation: np.ndarray,
    nt: int,
    dt: float,
    dx: float,
    dz: float,
    source: np.ndarray,
    sx: int,
    sz: int,
    receiver_x: np.ndarray,
    receiver_z: np.ndarray,
    damp: np.ndarray,
) -> np.ndarray:
    """Propagate incident and first-order scattered wavefields together."""
    nz, nx = velocity_squared.shape
    source_old = np.zeros((nz, nx), dtype=np.float32)
    source_now = np.zeros((nz, nx), dtype=np.float32)
    source_new = np.zeros((nz, nx), dtype=np.float32)
    scatter_old = np.zeros((nz, nx), dtype=np.float32)
    scatter_now = np.zeros((nz, nx), dtype=np.float32)
    scatter_new = np.zeros((nz, nx), dtype=np.float32)
    gathers = np.zeros((nt, receiver_x.size), dtype=np.float32)
    dt2 = dt * dt
    idx12 = 1.0 / (12.0 * dx * dx)
    idz12 = 1.0 / (12.0 * dz * dz)

    for it in range(nt):
        _step_constant_density(
            source_old, source_now, source_new, velocity_squared,
            dt2, idx12, idz12,
        )
        source_new[sz, sx] += source[it]

        _step_constant_density(
            scatter_old, scatter_now, scatter_new, velocity_squared,
            dt2, idx12, idz12,
        )
        # From m0 d_tt(delta p) - Laplacian(delta p)
        # = -delta_m d_tt(p_s).  The dt^2 factors cancel in the update.
        for iz in range(2, nz - 2):
            for ix in range(2, nx - 2):
                source_second_difference = (
                    source_new[iz, ix] - 2.0 * source_now[iz, ix]
                    + source_old[iz, ix]
                )
                scatter_new[iz, ix] -= (
                    velocity_squared[iz, ix]
                    * slowness_squared_perturbation[iz, ix]
                    * source_second_difference
                )

        source_new *= damp
        scatter_new *= damp
        for ir in range(receiver_x.size):
            gathers[it, ir] = scatter_new[receiver_z[ir], receiver_x[ir]]

        temporary = source_old
        source_old = source_now
        source_now = source_new
        source_new = temporary
        temporary = scatter_old
        scatter_old = scatter_now
        scatter_now = scatter_new
        scatter_new = temporary
    return gathers


def velocity_to_slowness_squared_perturbation(
    velocity: np.ndarray,
    background_velocity: np.ndarray,
) -> np.ndarray:
    """Return delta m = 1/v^2 - 1/v0^2 used by the Born equation."""
    velocity = np.asarray(velocity, dtype=np.float32)
    background = np.asarray(background_velocity, dtype=np.float32)
    if velocity.shape != background.shape or np.any(velocity <= 0.0) or np.any(background <= 0.0):
        raise ValueError("velocity and background_velocity must be positive and identically shaped")
    return (1.0 / velocity**2 - 1.0 / background**2).astype(np.float32)


def simulate_born_shot_gathers(
    background_velocity: np.ndarray,
    slowness_squared_perturbation: np.ndarray,
    geometry: dict[str, np.ndarray],
    *,
    nt: int,
    dt: float,
    dx: float,
    dz: float,
    f0: float = 15.0,
    t0: float = 0.10,
    boundary_width: int = 30,
) -> np.ndarray:
    """Generate scattered-only shot gathers under the first Born approximation."""
    vp0 = np.ascontiguousarray(background_velocity, dtype=np.float32)
    delta_m = np.ascontiguousarray(slowness_squared_perturbation, dtype=np.float32)
    if vp0.shape != delta_m.shape:
        raise ValueError("background_velocity and perturbation must have identical shapes")
    cfl = float(vp0.max()) * dt * np.sqrt(1.0 / dx**2 + 1.0 / dz**2)
    if cfl >= 1.0:
        raise ValueError(f"Unstable Born-modeling CFL number: {cfl:.3f}")
    wavelet = ricker_wavelet(nt, dt, f0, t0)
    damp = damping_mask(*vp0.shape, width=boundary_width, damp_top=True)
    velocity_squared = np.ascontiguousarray(vp0 * vp0, dtype=np.float32)
    receiver_x = np.asarray(geometry["receiver_x_index"], dtype=np.int32)
    receiver_z = np.asarray(geometry["receiver_z_index"], dtype=np.int32)
    nshots = int(np.asarray(geometry["shot_x_index"]).size)
    if receiver_x.shape[0] != nshots or receiver_z.shape != receiver_x.shape:
        raise ValueError("Geometry arrays have incompatible shot dimensions")
    output = np.zeros((nshots, nt, receiver_x.shape[1]), dtype=np.float32)
    for ishot in range(nshots):
        output[ishot] = _simulate_one_born_shot(
            velocity_squared,
            delta_m,
            nt, dt, dx, dz, wavelet,
            int(geometry["shot_x_index"][ishot]),
            int(geometry["shot_z_index"][ishot]),
            np.ascontiguousarray(receiver_x[ishot]),
            np.ascontiguousarray(receiver_z[ishot]),
            damp,
        )
    return output
