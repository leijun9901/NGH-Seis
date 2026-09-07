"""Fourth-order-space variable-density acoustic FD with streamer geometry."""

from __future__ import annotations

import numpy as np
from numba import njit


def ricker_wavelet(nt: int, dt: float, f0: float, t0: float) -> np.ndarray:
    time = np.arange(nt, dtype=np.float64) * dt
    arg = np.pi * f0 * (time - t0)
    return ((1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def damping_mask(
    nz: int,
    nx: int,
    width: int = 30,
    strength: float = 0.015,
    *,
    damp_top: bool = False,
) -> np.ndarray:
    """Cerjan-like damping on the sides/bottom and optionally the top.

    ``damp_top=False`` is used for a pressure-release free surface.  A
    primary-only/deghosted experiment uses ``damp_top=True`` together with a
    water padding above the physical source and receivers, so neither is
    placed inside the absorbing strip.
    """
    mask = np.ones((nz, nx), dtype=np.float32)
    for i in range(width):
        value = np.exp(-strength * ((width - i) / width) ** 2)
        mask[:, i] *= value
        mask[:, nx - 1 - i] *= value
        mask[nz - 1 - i, :] *= value
        if damp_top:
            mask[i, :] *= value
    return mask


def build_marine_streamer_geometry(
    nx: int,
    dx: float,
    nshots: int = 16,
    nreceivers: int = 64,
    shot_start_m: float = 850.0,
    shot_spacing_m: float = 100.0,
    near_offset_m: float = 50.0,
    receiver_spacing_m: float = 10.0,
    source_depth_m: float = 20.0,
    receiver_depth_m: float = 20.0,
    vessel_to_source_m: float = 50.0,
    sailing_direction: int = 1,
    dz: float = 10.0,
) -> dict[str, np.ndarray]:
    """Build a 2-D end-on marine streamer geometry.

    ``sailing_direction=1`` means the vessel sails toward increasing x.  The
    air-gun source is towed behind the vessel and every receiver is farther
    astern than the source. Channel 1 is the near-offset receiver.
    """
    if sailing_direction not in (-1, 1):
        raise ValueError("sailing_direction must be +1 or -1")
    if min(near_offset_m, receiver_spacing_m, vessel_to_source_m) < 0.0:
        raise ValueError("Marine offsets and tow distance must be non-negative")
    shot_x_m = shot_start_m + np.arange(nshots) * shot_spacing_m
    receiver_offsets_m = near_offset_m + np.arange(nreceivers) * receiver_spacing_m
    vessel_x_m = shot_x_m + sailing_direction * vessel_to_source_m
    receiver_x_m = shot_x_m[:, None] - sailing_direction * receiver_offsets_m[None, :]
    model_max_x = (nx - 1) * dx
    if min(receiver_x_m.min(), shot_x_m.min(), vessel_x_m.min()) < 0.0 or max(
        receiver_x_m.max(), shot_x_m.max(), vessel_x_m.max()
    ) > model_max_x:
        raise ValueError("Streamer geometry extends outside the model.")
    return {
        "vessel_x_m": vessel_x_m.astype(np.float32),
        "shot_x_m": shot_x_m.astype(np.float32),
        "shot_x_index": np.rint(shot_x_m / dx).astype(np.int32),
        "shot_z_index": np.full(nshots, round(source_depth_m / dz), dtype=np.int32),
        "receiver_x_m": receiver_x_m.astype(np.float32),
        "receiver_x_index": np.rint(receiver_x_m / dx).astype(np.int32),
        "receiver_z_index": np.full((nshots, nreceivers), round(receiver_depth_m / dz), dtype=np.int32),
        "offset_m": np.broadcast_to(receiver_offsets_m, (nshots, nreceivers)).astype(np.float32),
        "receiver_channel": np.broadcast_to(
            np.arange(1, nreceivers + 1), (nshots, nreceivers)
        ).astype(np.int32),
        "sailing_direction": np.asarray(sailing_direction, dtype=np.int8),
        "vessel_to_source_m": np.asarray(vessel_to_source_m, dtype=np.float32),
        "channel_order_near_to_far": np.asarray(1, dtype=np.int8),
    }


def build_fixed_spread_geometry(
    nx: int,
    dx: float,
    *,
    nshots: int = 25,
    nreceivers: int = 100,
    source_start_m: float = 0.0,
    source_end_m: float | None = None,
    receiver_start_m: float = 0.0,
    receiver_end_m: float | None = None,
    source_depth_m: float = 20.0,
    receiver_depth_m: float = 20.0,
    dz: float = 10.0,
) -> dict[str, np.ndarray]:
    """Build the fixed full-spread surface geometry used by SIGMA-style RTM."""
    model_end_m = (nx - 1) * dx
    source_end_m = model_end_m if source_end_m is None else source_end_m
    receiver_end_m = model_end_m if receiver_end_m is None else receiver_end_m
    shot_x_m = np.linspace(source_start_m, source_end_m, nshots, dtype=np.float32)
    one_receiver_line = np.linspace(
        receiver_start_m, receiver_end_m, nreceivers, dtype=np.float32
    )
    if min(shot_x_m.min(), one_receiver_line.min()) < 0.0 or max(
        shot_x_m.max(), one_receiver_line.max()
    ) > model_end_m:
        raise ValueError("Fixed-spread geometry extends outside the model")
    receiver_x_m = np.broadcast_to(one_receiver_line, (nshots, nreceivers)).copy()
    return {
        "shot_x_m": shot_x_m,
        "shot_x_index": np.rint(shot_x_m / dx).astype(np.int32),
        "shot_z_index": np.full(nshots, round(source_depth_m / dz), dtype=np.int32),
        "receiver_x_m": receiver_x_m,
        "receiver_x_index": np.rint(receiver_x_m / dx).astype(np.int32),
        "receiver_z_index": np.full(
            (nshots, nreceivers), round(receiver_depth_m / dz), dtype=np.int32
        ),
    }


@njit(cache=True)
def _simulate_one_shot(
    velocity_squared: np.ndarray,
    grad_logrho_x: np.ndarray,
    grad_logrho_z: np.ndarray,
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
    free_surface: bool,
) -> np.ndarray:
    nz, nx = velocity_squared.shape
    pold = np.zeros((nz, nx), dtype=np.float32)
    p = np.zeros((nz, nx), dtype=np.float32)
    pnew = np.zeros((nz, nx), dtype=np.float32)
    gathers = np.zeros((nt, receiver_x.size), dtype=np.float32)
    dt2 = dt * dt
    idx12 = 1.0 / (12.0 * dx * dx)
    idz12 = 1.0 / (12.0 * dz * dz)
    inv12dx = 1.0 / (12.0 * dx)
    inv12dz = 1.0 / (12.0 * dz)

    for it in range(nt):
        for iz in range(2, nz - 2):
            for ix in range(2, nx - 2):
                center = p[iz, ix]
                xp1 = p[iz, ix + 1]
                xm1 = p[iz, ix - 1]
                xp2 = p[iz, ix + 2]
                xm2 = p[iz, ix - 2]
                zp1 = p[iz + 1, ix]
                zm1 = p[iz - 1, ix]
                zp2 = p[iz + 2, ix]
                zm2 = p[iz - 2, ix]
                lap_x = (
                    -xp2 + 16.0 * xp1 - 30.0 * center + 16.0 * xm1 - xm2
                ) * idx12
                lap_z = (
                    -zp2 + 16.0 * zp1 - 30.0 * center + 16.0 * zm1 - zm2
                ) * idz12
                grad_p_x = (-xp2 + 8.0 * xp1 - 8.0 * xm1 + xm2) * inv12dx
                grad_p_z = (-zp2 + 8.0 * zp1 - 8.0 * zm1 + zm2) * inv12dz
                operator = (
                    lap_x + lap_z
                    - grad_logrho_x[iz, ix] * grad_p_x
                    - grad_logrho_z[iz, ix] * grad_p_z
                )
                pnew[iz, ix] = (
                    2.0 * p[iz, ix] - pold[iz, ix]
                    + velocity_squared[iz, ix] * dt2 * operator
                )

        # The marine source and streamer are commonly one grid cell below
        # the pressure-release surface. A centered fourth-order z stencil is
        # unavailable there, so use second order vertically and retain fourth
        # order horizontally. Without this closure, z=1 receivers remain
        # numerically frozen and record only noise.
        iz = 1
        for ix in range(2, nx - 2):
            center = p[iz, ix]
            xp1 = p[iz, ix + 1]
            xm1 = p[iz, ix - 1]
            xp2 = p[iz, ix + 2]
            xm2 = p[iz, ix - 2]
            lap_x = (-xp2 + 16.0 * xp1 - 30.0 * center + 16.0 * xm1 - xm2) * idx12
            lap_z = (p[2, ix] - 2.0 * center + p[0, ix]) * (12.0 * idz12)
            grad_p_x = (-xp2 + 8.0 * xp1 - 8.0 * xm1 + xm2) * inv12dx
            grad_p_z = (p[2, ix] - p[0, ix]) * (6.0 * inv12dz)
            operator = (
                lap_x + lap_z
                - grad_logrho_x[iz, ix] * grad_p_x
                - grad_logrho_z[iz, ix] * grad_p_z
            )
            pnew[iz, ix] = (
                2.0 * center - pold[iz, ix]
                + velocity_squared[iz, ix] * dt2 * operator
            )

        pnew[sz, sx] += source[it]
        if free_surface:
            pnew[0, :] = 0.0

        for iz in range(nz):
            for ix in range(nx):
                pnew[iz, ix] *= damp[iz, ix]

        for ir in range(receiver_x.size):
            gathers[it, ir] = pnew[receiver_z[ir], receiver_x[ir]]

        tmp = pold
        pold = p
        p = pnew
        pnew = tmp
    return gathers


def simulate_shot_gathers(
    vp: np.ndarray,
    geometry: dict[str, np.ndarray],
    rho: np.ndarray | None = None,
    nt: int = 2500,
    dt: float = 0.001,
    dx: float = 10.0,
    dz: float = 10.0,
    f0: float = 12.0,
    t0: float = 0.10,
    boundary_width: int = 30,
    free_surface: bool = True,
) -> np.ndarray:
    vp = np.ascontiguousarray(vp, dtype=np.float32)
    if rho is None:
        rho = np.ones_like(vp, dtype=np.float32)
    rho = np.ascontiguousarray(rho, dtype=np.float32)
    if rho.shape != vp.shape or np.any(rho <= 0.0):
        raise ValueError("rho must be positive and have the same shape as vp.")
    cfl = float(vp.max()) * dt * np.sqrt(1.0 / dx**2 + 1.0 / dz**2)
    if cfl >= 1.0:
        raise ValueError(f"Unstable CFL number: {cfl:.3f}")
    source = ricker_wavelet(nt, dt, f0, t0)
    damp = damping_mask(
        *vp.shape, width=boundary_width, damp_top=not free_surface
    )
    velocity_squared = np.ascontiguousarray(vp * vp, dtype=np.float32)
    log_rho = np.log(rho)
    grad_logrho_x = np.zeros_like(log_rho, dtype=np.float32)
    grad_logrho_z = np.zeros_like(log_rho, dtype=np.float32)
    grad_logrho_x[:, 2:-2] = (
        -log_rho[:, 4:] + 8.0 * log_rho[:, 3:-1]
        - 8.0 * log_rho[:, 1:-3] + log_rho[:, :-4]
    ) / (12.0 * dx)
    grad_logrho_z[2:-2, :] = (
        -log_rho[4:, :] + 8.0 * log_rho[3:-1, :]
        - 8.0 * log_rho[1:-3, :] + log_rho[:-4, :]
    ) / (12.0 * dz)
    grad_logrho_x = np.ascontiguousarray(grad_logrho_x)
    grad_logrho_z = np.ascontiguousarray(grad_logrho_z)
    nshots, nreceivers = geometry["receiver_x_index"].shape
    result = np.zeros((nshots, nt, nreceivers), dtype=np.float32)
    for ishot in range(nshots):
        result[ishot] = _simulate_one_shot(
            velocity_squared,
            grad_logrho_x,
            grad_logrho_z,
            nt,
            dt,
            dx,
            dz,
            source,
            int(geometry["shot_x_index"][ishot]),
            int(geometry["shot_z_index"][ishot]),
            np.ascontiguousarray(geometry["receiver_x_index"][ishot]),
            np.ascontiguousarray(geometry["receiver_z_index"][ishot]),
            damp,
            free_surface,
        )
    return result
