"""One-dimensional standard-linear-solid (SLS) numerical QA utilities.

This module is intentionally small and independent of the production 2-D
solver.  It implements the pressure--particle-velocity--memory-variable form
used in the Devito viscoacoustic tutorial.  Its purpose is to verify the
attenuation/dispersion parameterisation against the analytic complex modulus
before any Q model is admitted to the dataset generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def relaxation_times(q: float | np.ndarray, f_ref_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Return stress and strain relaxation times for an SLS mechanism.

    The formula makes the material quality factor equal to ``q`` at the
    reference frequency.  It follows the parameterisation in the official
    Devito viscoacoustic example.
    """

    q_arr = np.asarray(q, dtype=np.float64)
    if np.any(q_arr <= 0.0):
        raise ValueError("q must be positive")
    if f_ref_hz <= 0.0:
        raise ValueError("f_ref_hz must be positive")
    root = np.sqrt(q_arr * q_arr + 1.0)
    denom = 2.0 * np.pi * float(f_ref_hz) * q_arr
    tau_sigma = (root - 1.0) / denom
    tau_epsilon = (root + 1.0) / denom
    return tau_sigma, tau_epsilon


def complex_modulus(
    frequency_hz: float | np.ndarray,
    relaxed_modulus_pa: float,
    q: float,
    f_ref_hz: float,
) -> np.ndarray:
    """Complex SLS modulus under an ``exp(+i omega t)`` convention."""

    tau_sigma, tau_epsilon = relaxation_times(q, f_ref_hz)
    omega = 2.0 * np.pi * np.asarray(frequency_hz, dtype=np.float64)
    return float(relaxed_modulus_pa) * (
        (1.0 + 1j * omega * tau_epsilon)
        / (1.0 + 1j * omega * tau_sigma)
    )


def analytic_wavenumber(
    frequency_hz: float | np.ndarray,
    velocity_m_s: float,
    density_kg_m3: float,
    q: float,
    f_ref_hz: float,
) -> np.ndarray:
    """Return the analytic complex wavenumber of the homogeneous SLS medium."""

    modulus = complex_modulus(
        frequency_hz,
        density_kg_m3 * velocity_m_s**2,
        q,
        f_ref_hz,
    )
    omega = 2.0 * np.pi * np.asarray(frequency_hz, dtype=np.float64)
    return omega * np.sqrt(float(density_kg_m3) / modulus)


def ricker_wavelet(f_peak_hz: float, dt_s: float, nt: int, delay_s: float) -> np.ndarray:
    """Return a causal, delayed Ricker wavelet."""

    t = np.arange(nt, dtype=np.float64) * dt_s - delay_s
    arg = np.pi * float(f_peak_hz) * t
    return (1.0 - 2.0 * arg**2) * np.exp(-(arg**2))


@dataclass(frozen=True)
class SLS1DResult:
    traces: np.ndarray
    receiver_x_m: np.ndarray
    time_s: np.ndarray


def _sponge(n: int, width: int, strength: float = 0.018) -> np.ndarray:
    mask = np.ones(n, dtype=np.float64)
    if width <= 0:
        return mask
    if 2 * width >= n:
        raise ValueError("sponge width is too large for the grid")
    ramp = np.arange(width, 0, -1, dtype=np.float64) / float(width)
    edge = np.exp(-strength * ramp**2)
    mask[:width] = edge
    mask[-width:] = edge[::-1]
    return mask


def simulate_homogeneous_sls_1d(
    *,
    velocity_m_s: float = 1500.0,
    density_kg_m3: float = 1025.0,
    q: float = 50.0,
    f_ref_hz: float = 15.0,
    dx_m: float = 2.0,
    dt_s: float = 0.0004,
    nt: int = 5000,
    nx: int = 1601,
    source_x_m: float = 300.0,
    receiver_x_m: tuple[float, ...] = (800.0, 1800.0),
    source_delay_s: float = 0.12,
    sponge_width_cells: int = 100,
) -> SLS1DResult:
    """Simulate homogeneous SLS propagation on a staggered 1-D grid.

    Velocity is leapfrogged at cell faces.  The memory variable is advanced
    with its exact exponential solution while divergence is held constant over
    one time step, avoiding an avoidable Euler error in the relaxation term.
    """

    if velocity_m_s * dt_s / dx_m >= 1.0:
        raise ValueError("1-D CFL condition is not satisfied")
    if nx < 16 or nt < 2:
        raise ValueError("grid is too small")

    source_index = int(round(source_x_m / dx_m))
    receiver_indices = np.rint(np.asarray(receiver_x_m) / dx_m).astype(int)
    if source_index < 0 or source_index >= nx:
        raise ValueError("source is outside the grid")
    if np.any(receiver_indices < 0) or np.any(receiver_indices >= nx):
        raise ValueError("a receiver is outside the grid")

    pressure = np.zeros(nx, dtype=np.float64)
    velocity = np.zeros(nx - 1, dtype=np.float64)
    memory = np.zeros(nx, dtype=np.float64)
    divergence = np.zeros(nx, dtype=np.float64)
    traces = np.zeros((len(receiver_indices), nt), dtype=np.float64)

    relaxed_modulus = density_kg_m3 * velocity_m_s**2
    tau_sigma, tau_epsilon = relaxation_times(q, f_ref_hz)
    tau = float(tau_epsilon / tau_sigma - 1.0)
    memory_decay = float(np.exp(-dt_s / tau_sigma))
    wavelet = ricker_wavelet(f_ref_hz, dt_s, nt, source_delay_s)

    pressure_mask = _sponge(nx, sponge_width_cells)
    velocity_mask = 0.5 * (pressure_mask[:-1] + pressure_mask[1:])

    for it in range(nt):
        velocity -= (
            dt_s / density_kg_m3 * np.diff(pressure) / dx_m
        )
        velocity *= velocity_mask

        divergence.fill(0.0)
        divergence[1:-1] = np.diff(velocity) / dx_m
        memory = (
            memory_decay * memory
            - tau * relaxed_modulus * (1.0 - memory_decay) * divergence
        )
        pressure -= dt_s * (
            relaxed_modulus * (tau + 1.0) * divergence + memory
        )
        pressure[source_index] += wavelet[it] * dt_s / dx_m
        pressure *= pressure_mask
        memory *= pressure_mask
        traces[:, it] = pressure[receiver_indices]

    return SLS1DResult(
        traces=traces,
        receiver_x_m=np.asarray(receiver_x_m, dtype=np.float64),
        time_s=np.arange(nt, dtype=np.float64) * dt_s,
    )


def estimate_transfer_properties(
    result: SLS1DResult,
    *,
    dt_s: float,
    q: float,
    f_ref_hz: float,
    velocity_m_s: float,
    density_kg_m3: float,
    frequency_band_hz: tuple[float, float] = (7.0, 25.0),
) -> dict[str, np.ndarray]:
    """Estimate attenuation and phase velocity from a two-receiver transfer."""

    if result.traces.shape[0] != 2:
        raise ValueError("exactly two receivers are required")
    nfft = int(2 ** np.ceil(np.log2(result.traces.shape[1] * 2)))
    spectra = np.fft.rfft(result.traces, n=nfft, axis=1)
    frequencies = np.fft.rfftfreq(nfft, dt_s)
    band = (
        (frequencies >= frequency_band_hz[0])
        & (frequencies <= frequency_band_hz[1])
    )
    f = frequencies[band]
    transfer = spectra[1, band] / (spectra[0, band] + np.finfo(float).tiny)
    distance = float(result.receiver_x_m[1] - result.receiver_x_m[0])

    alpha_numeric = -np.log(np.maximum(np.abs(transfer), 1e-30)) / distance
    phase = np.unwrap(np.angle(transfer))

    k_analytic = analytic_wavenumber(
        f, velocity_m_s, density_kg_m3, q, f_ref_hz
    )
    phase_expected = -np.real(k_analytic) * distance
    ref_index = int(np.argmin(np.abs(f - f_ref_hz)))
    cycles = np.rint((phase_expected[ref_index] - phase[ref_index]) / (2.0 * np.pi))
    phase_aligned = phase + cycles * 2.0 * np.pi
    k_numeric = -phase_aligned / distance
    phase_velocity_numeric = 2.0 * np.pi * f / k_numeric

    return {
        "frequency_hz": f,
        "alpha_numeric_np_m": alpha_numeric,
        "alpha_analytic_np_m": np.abs(np.imag(k_analytic)),
        "phase_velocity_numeric_m_s": phase_velocity_numeric,
        "phase_velocity_analytic_m_s": 2.0 * np.pi * f / np.real(k_analytic),
        "transfer": transfer,
    }
