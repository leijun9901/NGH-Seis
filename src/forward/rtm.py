"""CPU reverse-time migration matched to the project's acoustic FD stencil."""

from __future__ import annotations

import numpy as np
from numba import njit

from .acoustic import damping_mask, ricker_wavelet


@njit(cache=True)
def _step_variable_density(
    pold: np.ndarray,
    p: np.ndarray,
    pnew: np.ndarray,
    velocity_squared: np.ndarray,
    grad_logrho_x: np.ndarray,
    grad_logrho_z: np.ndarray,
    dt2: float,
    idx12: float,
    idz12: float,
    inv12dx: float,
    inv12dz: float,
) -> None:
    nz, nx = p.shape
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

    # Free-surface closure for a source/receiver at z=1: second order in z,
    # fourth order in x. This is the adjoint-consistent counterpart of the
    # forward closure in acoustic._simulate_one_shot.
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


@njit(cache=True)
def _rtm_one_shot(
    velocity_squared: np.ndarray,
    grad_logrho_x: np.ndarray,
    grad_logrho_z: np.ndarray,
    gather: np.ndarray,
    source: np.ndarray,
    sx: int,
    sz: int,
    receiver_x: np.ndarray,
    receiver_z: np.ndarray,
    damp: np.ndarray,
    dt: float,
    dx: float,
    dz: float,
    snapshot_stride: int,
    z0: int,
    z1: int,
    x0: int,
    x1: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cross-correlation, source illumination, and receiver illumination."""
    nt = gather.shape[0]
    nz, nx = velocity_squared.shape
    nsnap = (nt - 1) // snapshot_stride + 1
    snapshots = np.zeros((nsnap, z1 - z0, x1 - x0), dtype=np.float32)
    image = np.zeros((z1 - z0, x1 - x0), dtype=np.float32)
    illumination = np.zeros_like(image)
    receiver_illumination = np.zeros_like(image)
    dt2 = dt * dt
    idx12 = 1.0 / (12.0 * dx * dx)
    idz12 = 1.0 / (12.0 * dz * dz)
    inv12dx = 1.0 / (12.0 * dx)
    inv12dz = 1.0 / (12.0 * dz)

    pold = np.zeros((nz, nx), dtype=np.float32)
    p = np.zeros((nz, nx), dtype=np.float32)
    pnew = np.zeros((nz, nx), dtype=np.float32)
    for it in range(nt):
        _step_variable_density(
            pold, p, pnew, velocity_squared, grad_logrho_x, grad_logrho_z,
            dt2, idx12, idz12, inv12dx, inv12dz,
        )
        pnew[sz, sx] += source[it]
        pnew[0, :] = 0.0
        pnew *= damp
        if it % snapshot_stride == 0:
            snapshot = pnew[z0:z1, x0:x1]
            snapshots[it // snapshot_stride] = snapshot
            illumination += snapshot * snapshot
        tmp = pold
        pold = p
        p = pnew
        pnew = tmp

    rold = np.zeros((nz, nx), dtype=np.float32)
    r = np.zeros((nz, nx), dtype=np.float32)
    rnew = np.zeros((nz, nx), dtype=np.float32)
    for reverse_it in range(nt):
        physical_it = nt - 1 - reverse_it
        _step_variable_density(
            rold, r, rnew, velocity_squared, grad_logrho_x, grad_logrho_z,
            dt2, idx12, idz12, inv12dx, inv12dz,
        )
        for ir in range(receiver_x.size):
            rnew[receiver_z[ir], receiver_x[ir]] += gather[physical_it, ir]
        rnew[0, :] = 0.0
        rnew *= damp
        if physical_it % snapshot_stride == 0:
            receiver_snapshot = rnew[z0:z1, x0:x1]
            image += snapshots[physical_it // snapshot_stride] * receiver_snapshot
            receiver_illumination += receiver_snapshot * receiver_snapshot
        tmp = rold
        rold = r
        r = rnew
        rnew = tmp
    return image, illumination, receiver_illumination


def mute_direct_water_wave(
    gather: np.ndarray,
    offset_m: np.ndarray,
    *,
    dt: float,
    t0: float,
    water_velocity: float = 1500.0,
    extra_mute_s: float = 0.08,
    taper_s: float = 0.06,
) -> np.ndarray:
    """Mute direct-water arrivals trace by trace before migration."""
    output = np.asarray(gather, dtype=np.float32).copy()
    nt = output.shape[0]
    taper_samples = max(1, round(taper_s / dt))
    for receiver, offset in enumerate(np.asarray(offset_m)):
        stop = min(nt, round((t0 + float(offset) / water_velocity + extra_mute_s) / dt))
        output[:stop, receiver] = 0.0
        end = min(nt, stop + taper_samples)
        if end > stop:
            phase = np.linspace(0.0, np.pi / 2.0, end - stop, dtype=np.float32)
            output[stop:end, receiver] *= np.sin(phase) ** 2
    return output


def cosine_taper_gather(
    gather: np.ndarray,
    *,
    receiver_fraction: float = 0.08,
    end_time_fraction: float = 0.04,
) -> np.ndarray:
    """Taper aperture ends and the recording tail to reduce truncation smiles."""
    output = np.asarray(gather, dtype=np.float32).copy()
    nt, nr = output.shape
    receiver_count = max(1, round(nr * receiver_fraction))
    ramp = np.sin(np.linspace(0.0, np.pi / 2.0, receiver_count, dtype=np.float32)) ** 2
    output[:, :receiver_count] *= ramp[None, :]
    output[:, -receiver_count:] *= ramp[::-1][None, :]
    time_count = max(1, round(nt * end_time_fraction))
    time_ramp = np.cos(np.linspace(0.0, np.pi / 2.0, time_count, dtype=np.float32)) ** 2
    output[-time_count:] *= time_ramp[:, None]
    return output


def migrate_shot(
    vp_migration: np.ndarray,
    rho_migration: np.ndarray,
    gather: np.ndarray,
    *,
    sx: int,
    sz: int,
    receiver_x: np.ndarray,
    receiver_z: np.ndarray,
    nt: int,
    dt: float,
    dx: float,
    dz: float,
    f0: float,
    t0: float,
    snapshot_stride: int = 8,
    crop: tuple[int, int, int, int] = (0, 400, 120, 520),
    free_surface: bool = True,
    boundary_width: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Public checked wrapper for a single-shot RTM contribution."""
    vp = np.ascontiguousarray(vp_migration, dtype=np.float32)
    rho = np.ascontiguousarray(rho_migration, dtype=np.float32)
    data = np.ascontiguousarray(gather, dtype=np.float32)
    if data.shape[0] != nt or rho.shape != vp.shape:
        raise ValueError("Incompatible migration arrays")
    cfl = float(vp.max()) * dt * np.sqrt(1.0 / dx**2 + 1.0 / dz**2)
    if cfl >= 1.0:
        raise ValueError(f"Unstable migration CFL number: {cfl:.3f}")
    z0, z1, x0, x1 = crop
    if not (0 <= z0 < z1 <= vp.shape[0] and 0 <= x0 < x1 <= vp.shape[1]):
        raise ValueError("Invalid imaging crop")
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
    return _rtm_one_shot(
        velocity_squared,
        np.ascontiguousarray(grad_logrho_x),
        np.ascontiguousarray(grad_logrho_z),
        data, ricker_wavelet(nt, dt, f0, t0), int(sx), int(sz),
        np.ascontiguousarray(receiver_x, dtype=np.int32),
        np.ascontiguousarray(receiver_z, dtype=np.int32),
        damping_mask(
            *vp.shape, width=boundary_width, damp_top=not free_surface
        ),
        dt, dx, dz, snapshot_stride, z0, z1, x0, x1,
    )
