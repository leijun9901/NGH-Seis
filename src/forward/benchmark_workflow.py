"""Field-compatible preprocessing and migration utilities for NGH-Seis v1.0.

The functions in this module deliberately cannot access saturation, porosity,
or the true velocity model.  This keeps the migration side of the benchmark
independent from its labels.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, sosfiltfilt

from .rtm import cosine_taper_gather, mute_direct_water_wave


def add_observation_noise(
    gather: np.ndarray,
    *,
    seed: int,
    snr_db: float,
    dt: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Add reproducible label-free ambient and weak coherent marine noise.

    SNR is defined using the RMS of the recorded gather after the source
    transient.  The mixture contains uncorrelated hydrophone noise and a weak
    spatially coherent, low-frequency component.  Both are added before the
    common preprocessing and migration steps.
    """
    data = np.asarray(gather, dtype=np.float32)
    if data.ndim != 2 or not np.isfinite(snr_db):
        raise ValueError("gather must be 2-D and snr_db finite")
    rng = np.random.default_rng(seed)
    start = min(data.shape[0] - 1, max(0, round(0.15 / dt)))
    signal_rms = max(float(np.sqrt(np.mean(data[start:] ** 2, dtype=np.float64))), 1e-12)
    target_noise_rms = signal_rms / (10.0 ** (float(snr_db) / 20.0))

    independent = rng.normal(size=data.shape).astype(np.float32)
    common = rng.normal(size=data.shape[0]).astype(np.float32)
    common = gaussian_filter(common, sigma=max(1.0, 0.012 / dt), mode="reflect")
    receiver_weights = gaussian_filter(
        rng.normal(size=data.shape[1]).astype(np.float32), sigma=8.0, mode="reflect"
    )
    coherent = common[:, None] * receiver_weights[None, :]
    mixture = independent + np.float32(0.35) * coherent
    mixture_rms = max(float(np.sqrt(np.mean(mixture ** 2, dtype=np.float64))), 1e-12)
    noise = mixture * np.float32(target_noise_rms / mixture_rms)
    return (data + noise).astype(np.float32), {
        "seed": int(seed),
        "snr_db": float(snr_db),
        "signal_rms": signal_rms,
        "added_noise_rms": target_noise_rms,
    }


def build_independent_migration_model(
    seafloor_m: np.ndarray,
    *,
    nz: int,
    nx: int,
    dz: float,
    seed: int,
    water_velocity_mps: float = 1500.0,
    water_bottom_transition_m: float = 60.0,
    constant_density: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Construct a low-wavenumber regional migration prior.

    Only picked bathymetry and an independent seed are accepted.  In
    particular, this routine has no access to Sh, Sg, porosity, density, or
    the true Vp.  The ranges represent a deliberately broad unconsolidated
    marine-sediment compaction prior and must be reported as such.
    """
    seafloor = np.asarray(seafloor_m, dtype=np.float32)
    if seafloor.shape != (nx,):
        raise ValueError(f"Expected seafloor_m shape {(nx,)}, got {seafloor.shape}")
    if np.any(~np.isfinite(seafloor)) or np.any(seafloor <= 0.0):
        raise ValueError("seafloor_m must be finite and positive")

    rng = np.random.default_rng(seed)
    # Regional sonic constraints in the same Shenhu scenario put the shallow
    # sediment trend well above water velocity. Keep the prior independent
    # of every generated physical field, but avoid a broad global range that
    # creates a systematic 200+ m/s migration bias. These low-wavenumber
    # ranges bracket the three well-family background trends.
    sediment_top_velocity = float(rng.uniform(1660.0, 1720.0))
    compaction_gradient = float(rng.uniform(0.55, 0.75))
    curvature = float(rng.uniform(2.0e-4, 4.0e-4))
    lateral_fraction = float(rng.uniform(0.005, 0.015))

    z_m = np.arange(nz, dtype=np.float32)[:, None] * np.float32(dz)
    depth_bsf = np.maximum(z_m - seafloor[None, :], 0.0)
    background = (
        sediment_top_velocity
        + compaction_gradient * depth_bsf
        + curvature * depth_bsf * depth_bsf
    )

    # Long-wavelength estimation variability.  It is generated independently
    # of the true model and contains no reservoir-scale boundaries.
    noise = rng.normal(size=(nz, nx)).astype(np.float32)
    noise = gaussian_filter(noise, sigma=(55.0, 85.0), mode="reflect")
    noise -= float(noise.mean())
    noise /= max(float(noise.std()), 1e-6)
    background *= 1.0 + lateral_fraction * noise
    background = gaussian_filter(background, sigma=(8.0, 18.0), mode="nearest")

    water = z_m < seafloor[None, :]
    vp_migration = np.where(water, water_velocity_mps, background).astype(np.float32)
    # RTM propagation must use a genuinely low-wavenumber background.  A
    # hard water-bottom jump in the migration operator creates a reflected
    # source wavefield and strong conventional cross-correlation
    # back-scattering.  Retain the picked bathymetry for masking, while using
    # a short, label-free transition in the propagation model.
    transition_cells = max(float(water_bottom_transition_m) / float(dz), 0.0)
    if transition_cells:
        vp_migration = gaussian_filter(
            vp_migration, sigma=(0.5 * transition_cells, 1.0), mode="nearest"
        )
    far_water = z_m <= seafloor[None, :] - np.float32(water_bottom_transition_m)
    vp_migration[far_water] = np.float32(water_velocity_mps)
    vp_migration = np.clip(vp_migration, 1480.0, 2650.0).astype(np.float32)

    # Constant-density acoustic RTM is intentional here.  The observed data
    # remain variable-density, but copying a sharp water/sediment density
    # contrast into the migration operator introduces a second strong source
    # reflector and avoidable low-wavenumber artifacts.
    if constant_density:
        rho_migration = np.full((nz, nx), 1800.0, dtype=np.float32)
    else:
        rho_migration = (1640.0 + 0.34 * depth_bsf).astype(np.float32)
        rho_migration = gaussian_filter(
            rho_migration, sigma=(8.0, 18.0), mode="nearest"
        )
        rho_migration = np.clip(rho_migration, 1025.0, 2200.0).astype(np.float32)

    metadata = {
        "seed": int(seed),
        "sediment_top_velocity_mps": sediment_top_velocity,
        "compaction_gradient_s_inv": compaction_gradient,
        "curvature_inv_m_s": curvature,
        "lateral_fraction": lateral_fraction,
        "water_velocity_mps": float(water_velocity_mps),
        "water_bottom_transition_m": float(water_bottom_transition_m),
        "constant_density_migration": bool(constant_density),
    }
    return vp_migration, rho_migration, metadata


def preprocess_observed_gather(
    gather: np.ndarray,
    offset_m: np.ndarray,
    *,
    dt: float,
    t0: float,
    low_hz: float = 3.0,
    high_hz: float = 32.0,
    gain_power: float = 0.65,
    direct_mute_extra_s: float = 0.04,
    coherent_noise_lowpass_hz: float = 8.0,
) -> np.ndarray:
    """Apply one fixed, label-free marine preprocessing sequence.

    The operation preserves relative amplitudes between traces: there is no
    per-trace AGC or trace-wise normalization.  Noise, when enabled by a
    caller, must be added before this function so it migrates into the image.
    """
    data = np.asarray(gather, dtype=np.float32)
    offsets = np.asarray(offset_m, dtype=np.float32)
    if data.ndim != 2 or offsets.shape != (data.shape[1],):
        raise ValueError("gather must be [time, receiver] and match offset_m")
    nyquist = 0.5 / dt
    if not (0.0 < low_hz < high_hz < nyquist):
        raise ValueError("Invalid band-pass frequencies")

    output = data - data.mean(axis=0, keepdims=True, dtype=np.float64)
    sos = butter(4, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    output = sosfiltfilt(sos, output, axis=0).astype(np.float32)
    # Suppress only the low-frequency, receiver-coherent component (swell and
    # streamer/common-mode noise proxy).  Restricting the subtraction to the
    # low band avoids removing the 15 Hz reflection signal and its moveout.
    if coherent_noise_lowpass_hz > 0.0:
        if coherent_noise_lowpass_hz >= nyquist:
            raise ValueError("coherent_noise_lowpass_hz must be below Nyquist")
        common = np.mean(output, axis=1, dtype=np.float64)
        low_sos = butter(
            4, coherent_noise_lowpass_hz / nyquist, btype="lowpass", output="sos"
        )
        common = sosfiltfilt(low_sos, common).astype(np.float32)
        output -= common[:, None]
    output = mute_direct_water_wave(
        output,
        offsets,
        dt=dt,
        t0=t0,
        extra_mute_s=direct_mute_extra_s,
        taper_s=0.06,
    )

    # A survey-wide deterministic spherical-divergence proxy.  It depends
    # only on recording time, not on labels or individual trace amplitudes.
    time_s = np.arange(output.shape[0], dtype=np.float32) * np.float32(dt)
    gain = np.maximum(time_s - np.float32(t0) + 0.05, 0.05) ** np.float32(gain_power)
    gain /= max(float(gain.max()), 1e-12)
    output *= gain[:, None]
    return cosine_taper_gather(output, receiver_fraction=0.08, end_time_fraction=0.04)


def build_valid_imaging_mask(
    seafloor_m: np.ndarray,
    joint_illumination: np.ndarray,
    *,
    x_start_index: int,
    dz: float,
    water_bottom_buffer_m: float = 25.0,
    model_bottom_buffer_m: float = 150.0,
    lateral_buffer_cells: int = 10,
    propagation_bottom_m: float | None = None,
) -> np.ndarray:
    """Return a geometry/illumination-derived mask without using labels."""
    illumination = np.asarray(joint_illumination, dtype=np.float32)
    nz, nx = illumination.shape
    local_seafloor = np.asarray(
        seafloor_m[x_start_index:x_start_index + nx], dtype=np.float32
    )
    if local_seafloor.shape != (nx,):
        raise ValueError("seafloor_m does not cover the imaging crop")
    z_m = np.arange(nz, dtype=np.float32)[:, None] * np.float32(dz)
    geometric = z_m >= local_seafloor[None, :] + np.float32(water_bottom_buffer_m)
    # The imaging crop may be shallower than the propagation grid.  Use the
    # latter's true bottom when provided, so a computational absorber located
    # below the released target does not erase the target's deepest samples.
    actual_bottom_m = (nz - 1) * dz if propagation_bottom_m is None else propagation_bottom_m
    geometric &= z_m <= np.float32(actual_bottom_m - model_bottom_buffer_m)
    if lateral_buffer_cells:
        geometric[:, :lateral_buffer_cells] = False
        geometric[:, -lateral_buffer_cells:] = False

    positive = illumination[illumination > 0.0]
    if positive.size:
        illumination_floor = float(np.percentile(positive, 2.0))
        geometric &= illumination >= illumination_floor
    return geometric.astype(np.uint8)


def standardize_rtm(
    image: np.ndarray,
    source_illumination: np.ndarray,
    receiver_illumination: np.ndarray,
    valid_mask: np.ndarray,
    *,
    dx: float = 10.0,
    dz: float = 10.0,
    maximum_dip_deg: float = 70.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply stabilized source normalization and soft artifact suppression."""
    raw = np.asarray(image, dtype=np.float32)
    source = np.asarray(source_illumination, dtype=np.float32)
    receiver = np.asarray(receiver_illumination, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    if raw.shape != source.shape or raw.shape != receiver.shape or raw.shape != mask.shape:
        raise ValueError("RTM arrays and valid_mask must have identical shapes")

    joint = np.sqrt(np.maximum(source * receiver, 0.0)).astype(np.float32)
    positive_source = source[mask & (source > 0.0)]
    floor = (
        max(float(np.percentile(positive_source, 85.0)) * 0.04, 1e-20)
        if positive_source.size else 1.0
    )
    # Source-deconvolution normalization is less aggressive than division by
    # sqrt(source*receiver), which amplified receiver-side illumination
    # streaks in low-coverage areas.
    normalized = raw / (source + floor)
    processed = normalized - gaussian_filter(
        normalized, sigma=(5.0, 10.0), mode="nearest"
    )

    # A soft f-k taper removes only near-vertical energy.  The shallow Shenhu
    # sediment reflectors and fault terminations are retained below 70 deg;
    # the transition is deliberately broad to avoid ringing.
    if maximum_dip_deg < 90.0:
        nz, nx = processed.shape
        kz = np.fft.fftfreq(nz, d=dz)[:, None]
        kx = np.fft.fftfreq(nx, d=dx)[None, :]
        angle = np.degrees(np.arctan2(np.abs(kx), np.maximum(np.abs(kz), 1e-12)))
        transition = 12.0
        weight = np.ones_like(angle)
        weight[angle >= maximum_dip_deg + transition] = 0.0
        middle = (angle > maximum_dip_deg) & (angle < maximum_dip_deg + transition)
        weight[middle] = 0.5 * (
            1.0 + np.cos(
                np.pi * (angle[middle] - maximum_dip_deg) / transition
            )
        )
        weight[0, 0] = 0.0
        processed = np.fft.ifft2(np.fft.fft2(processed) * weight).real.astype(np.float32)

    values = np.abs(processed[mask])
    scale = max(float(np.percentile(values, 99.4)), 1e-12) if values.size else 1.0
    processed = np.clip(processed / scale, -1.0, 1.0).astype(np.float32)
    processed[~mask] = 0.0
    return processed, joint


def measure_rtm_directional_artifacts(
    image: np.ndarray,
    valid_mask: np.ndarray,
    *,
    dx: float = 10.0,
    dz: float = 10.0,
    near_vertical_dip_deg: float = 70.0,
) -> dict[str, float]:
    """Return label-free directional QC metrics for a released RTM image.

    The metrics are audit signals rather than training labels.  A high
    horizontal-gradient/vertical-gradient ratio or a high near-vertical f-k
    fraction is characteristic of acquisition-footprint columns.  They are
    stored per sample so a formal batch can be stopped when the distribution
    drifts, without inspecting Sh, Sg, or the true acoustic model.
    """
    array = np.asarray(image, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    if array.shape != mask.shape or array.ndim != 2:
        raise ValueError("image and valid_mask must be identically shaped 2-D arrays")
    if not np.any(mask):
        return {
            "horizontal_to_vertical_gradient_energy": 0.0,
            "near_vertical_fk_energy_fraction": 0.0,
        }

    work = np.where(mask, array, 0.0).astype(np.float64)
    work[mask] -= float(np.mean(work[mask]))
    gradient_x = np.gradient(work, dx, axis=1)
    gradient_z = np.gradient(work, dz, axis=0)
    horizontal_energy = float(np.mean(gradient_x[mask] ** 2))
    vertical_energy = float(np.mean(gradient_z[mask] ** 2))

    spectrum_energy = np.abs(np.fft.fft2(work)) ** 2
    kz = np.fft.fftfreq(array.shape[0], d=dz)[:, None]
    kx = np.fft.fftfreq(array.shape[1], d=dx)[None, :]
    dip = np.degrees(
        np.arctan2(np.abs(kx), np.maximum(np.abs(kz), 1e-12))
    )
    total_spectrum_energy = float(np.sum(spectrum_energy))
    near_vertical_energy = float(
        np.sum(spectrum_energy[dip > near_vertical_dip_deg])
    )
    return {
        "horizontal_to_vertical_gradient_energy": horizontal_energy / max(
            vertical_energy, 1e-20
        ),
        "near_vertical_fk_energy_fraction": near_vertical_energy / max(
            total_spectrum_energy, 1e-20
        ),
    }
