"""EW0008-scale deep-water hydrate/free-gas geology for NGH-Seis v1.0."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import fftconvolve

from src.rock_physics import blake_prior_acoustic_properties


def _smooth_unit(rng: np.random.Generator, shape, sigma) -> np.ndarray:
    a = gaussian_filter(rng.normal(size=shape), sigma=sigma, mode="reflect")
    a -= a.mean()
    a /= max(float(a.std()), 1.0e-8)
    return a


def _scale_to_cap(field: np.ndarray, mask: np.ndarray, mean: float, cap: float) -> np.ndarray:
    out = np.maximum(field, 0.0) * mask
    if not mask.any() or mean <= 0:
        return np.zeros_like(out, dtype=np.float32)
    for _ in range(6):
        current = float(out[mask].mean())
        if current <= 1.0e-10:
            break
        out = np.minimum(out * mean / current, cap) * mask
    return out.astype(np.float32)


def _scale_to_soft_cap(
    field: np.ndarray,
    mask: np.ndarray,
    mean: float,
    cap: float,
) -> np.ndarray:
    """Match a masked mean with a smooth asymptote instead of hard clipping.

    ``cap * tanh(scale * field / cap)`` is monotonic and never creates the
    constant-valued plateaus produced by ``minimum(..., cap)``.  A scalar
    bisection preserves the requested regional mean while keeping every value
    strictly below the documented local upper bound.
    """
    active = np.asarray(mask, dtype=bool)
    out = np.zeros_like(field, dtype=np.float64)
    if not active.any() or mean <= 0.0 or cap <= mean:
        return out.astype(np.float32)
    weights = np.maximum(np.asarray(field, dtype=np.float64)[active], 1.0e-8)
    weights /= max(float(weights.mean()), 1.0e-12)

    def mapped(scale: float) -> np.ndarray:
        return cap * np.tanh(scale * weights / cap)

    low, high = 0.0, max(mean, 1.0e-6)
    while float(mapped(high).mean()) < mean and high < cap * 1.0e6:
        high *= 2.0
    for _ in range(64):
        middle = 0.5 * (low + high)
        if float(mapped(middle).mean()) < mean:
            low = middle
        else:
            high = middle
    out[active] = mapped(0.5 * (low + high))
    return out.astype(np.float32)


def _seafloor_relative_crop(
    field: np.ndarray,
    seafloor_m: np.ndarray,
    dz_m: float,
    n_out: int = 512,
    samples_above: int = 64,
    order: int = 1,
) -> np.ndarray:
    rows = (seafloor_m[None, :] / dz_m - samples_above) + np.arange(n_out)[:, None]
    cols = np.broadcast_to(np.arange(field.shape[1], dtype=np.float64), rows.shape)
    cropped = map_coordinates(field, [rows, cols], order=order, mode="nearest")
    return np.pad(cropped, ((0, 0), (0, 1)), mode="constant").astype(np.float32)


def _stochastic_stratigraphy(
    rng: np.random.Generator,
    depositional_m: np.ndarray,
    x_m: np.ndarray,
    *,
    dx_m: float = 5.0,
    continuity: dict | None = None,
) -> np.ndarray:
    """Create lithologic variability in depositional coordinates.

    The 1-D component represents a correlated vertical sediment log. Warping
    it with the depositional coordinate gives coherent beds; slow 2-D facies
    modulation and localized marker beds prevent perfectly repeated layers.
    """
    s_axis = np.arange(-300.0, 2500.0 + 3.0, 3.0)
    fine = gaussian_filter(rng.normal(size=s_axis.size), rng.uniform(0.7, 1.4), mode="reflect")
    medium = gaussian_filter(rng.normal(size=s_axis.size), rng.uniform(3.0, 7.0), mode="reflect")
    log = 0.62 * fine / max(float(fine.std()), 1e-8) + 0.38 * medium / max(float(medium.std()), 1e-8)
    # Sparse high-contrast beds emulate occasional impedance markers without
    # forcing every reflector to span the entire section.
    impulses = np.zeros_like(s_axis)
    locations = rng.choice(np.arange(50, s_axis.size - 50), size=18, replace=False)
    impulses[locations] = rng.normal(0.0, 2.0, size=locations.size)
    impulses = gaussian_filter(impulses, rng.uniform(0.7, 1.4), mode="constant")
    log += impulses
    log -= log.mean()
    log /= max(float(log.std()), 1e-8)

    sampled = np.interp(depositional_m, s_axis, log, left=log[0], right=log[-1])
    nz, nx = depositional_m.shape
    if continuity is None:
        lateral_sigma = rng.uniform(7.0, 18.0)
        facies_sigma_x = rng.uniform(7.0, 17.0)
        lateral_amplitude = 0.28
        facies_amplitude = 0.28
        lens_width_m = (170.0, 520.0)
    else:
        lateral_sigma = rng.uniform(*continuity["stratigraphy_lateral_sigma_m"]) / dx_m
        facies_sigma_x = rng.uniform(*continuity["facies_lateral_sigma_m"]) / dx_m
        lateral_amplitude = rng.uniform(*continuity["stratigraphy_lateral_amplitude"])
        facies_amplitude = rng.uniform(*continuity["facies_amplitude"])
        lens_width_m = continuity["lithology_lens_halfwidth_m"]
    lateral = 1.0 + lateral_amplitude * _smooth_unit(rng, nx, lateral_sigma)[None, :]
    facies = _smooth_unit(
        rng, (nz, nx), (rng.uniform(10.0, 24.0), facies_sigma_x)
    )
    sampled = sampled * lateral + facies_amplitude * facies

    # A few lenses/pinch-outs modify the properties themselves, so subsequent
    # seismic texture remains a consequence of acoustic impedance.
    for _ in range(4):
        center_x = rng.uniform(0.05, 0.95) * x_m[-1]
        center_s = rng.uniform(100.0, 1200.0)
        lens = np.exp(
            -0.5 * ((x_m[None, :] - center_x) / rng.uniform(*lens_width_m)) ** 2
        )
        lens = lens * np.exp(-0.5 * ((depositional_m - center_s) / rng.uniform(10.0, 35.0)) ** 2)
        sampled += rng.uniform(-1.0, 1.0) * lens
    sampled -= sampled.mean()
    sampled /= max(float(sampled.std()), 1e-8)
    return sampled


def _limit_curve_slope(
    curve_m: np.ndarray,
    *,
    dx_m: float,
    maximum_p95_absolute_slope: float,
) -> tuple[np.ndarray, float]:
    """Scale relief about its mean until its p95 slope meets a physical gate."""
    curve = np.asarray(curve_m, dtype=np.float64)
    centered = curve - curve.mean()
    p95 = float(np.percentile(np.abs(np.diff(centered) / dx_m), 95.0))
    if p95 > maximum_p95_absolute_slope:
        centered *= maximum_p95_absolute_slope / p95
    output = centered + curve.mean()
    final_p95 = float(np.percentile(np.abs(np.diff(output) / dx_m), 95.0))
    return output, final_p95


def _migrated_impedance_image(
    impedance: np.ndarray,
    dz_m: float,
    water_mask: np.ndarray,
    rng: np.random.Generator,
    k_cycles_per_km: float,
    lateral_sigma: float,
    snr_db: float,
    attenuation_length_m: float,
    illumination_variability: float,
) -> tuple[np.ndarray, np.ndarray]:
    logz = np.log(np.maximum(impedance.astype(np.float64), 1.0))
    reflectivity = np.zeros_like(logz)
    reflectivity[1:] = 0.5 * np.diff(logz, axis=0)

    half = 15
    depth_lag = np.arange(-half, half + 1) * dz_m
    k_per_m = k_cycles_per_km / 1000.0
    a = np.pi * k_per_m * depth_lag
    wavelet = (1.0 - 2.0 * a * a) * np.exp(-(a * a))
    wavelet -= wavelet.mean()
    wavelet /= max(float(np.sqrt(np.sum(wavelet * wavelet))), 1.0e-8)
    image = fftconvolve(reflectivity, wavelet[:, None], mode="same")
    image = gaussian_filter(image, sigma=(0.18, lateral_sigma), mode="reflect")

    # Smooth illumination and weak coherent acquisition/migration residuals.
    nz, nx = image.shape
    zq = np.linspace(0.0, 1.0, nz)[:, None]
    lateral_gain = 1.0 + 0.10 * _smooth_unit(rng, nx, 12.0)[None, :]
    # PSDM amplitudes in EW0008 decay below the main sediment package.  The
    # attenuation uses depth below seafloor, not saturation labels.
    sediment_depth = np.maximum(np.cumsum(~water_mask, axis=0) * dz_m, 0.0)
    attenuation = np.exp(-sediment_depth / attenuation_length_m)
    illumination_raw = _smooth_unit(rng, image.shape, (28.0, 5.0))
    illumination = 1.0 / (1.0 + np.exp(-1.25 * illumination_raw))
    illumination = (1.0 - illumination_variability) + 2.0 * illumination_variability * illumination
    image *= (0.92 + 0.12 * zq) * lateral_gain * attenuation * illumination
    signal_rms = max(float(np.sqrt(np.mean(image[~water_mask] ** 2))), 1.0e-9)
    noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    random_noise = _smooth_unit(rng, image.shape, (0.65, 0.85))
    coherent = _smooth_unit(rng, (nz, 1), (7.0, 0.0)) * _smooth_unit(rng, (1, nx), (0.0, 16.0))
    observed = image + noise_rms * (0.82 * random_noise + 0.18 * coherent)
    return reflectivity.astype(np.float32), observed.astype(np.float32)


def generate_blake_ew0008(
    config_path: str | Path,
    sample_index: int,
    seed: int,
    *,
    style_override: str | None = None,
    include_proxy_image: bool = True,
) -> tuple[dict[str, np.ndarray], dict]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    g = config["grid"]
    p = config["geological_priors"]
    im = config["image_model"]
    nx, nz = int(g["nx_physical"]), int(g["nz"])
    dx, dz = float(g["dx_m"]), float(g["dz_m"])
    rng = np.random.default_rng(seed)
    styles = config["scenario_styles"]
    style = style_override or styles[(sample_index - 1) % len(styles)]
    continuity = config.get("lateral_continuity")
    architecture_level = int(config.get("architecture_level", 1))
    occurrence_index = (sample_index - 1) // max(len(styles), 1)
    architecture_variant = "regional_diffuse_system"
    if architecture_level >= 2 and style == "diffuse_continuous_bsr":
        architecture_variant = "regional_diffuse_system"
    elif architecture_level >= 2 and style == "localized_enriched_lenses":
        architecture_variant = "facies_conditioned_lenses"
    elif architecture_level >= 2 and style == "discontinuous_bsr":
        architecture_variant = (
            "gas_patchy",
            "hydrate_patchy",
            "coincident_fairways",
        )[occurrence_index % 3]
    elif architecture_level >= 2 and style == "fault_gas_chimney":
        architecture_variant = (
            "thermal_peripheral_hydrate",
            "fracture_fill_hydrate",
            "leaky_mixed_chimney",
        )[occurrence_index % 3]
    elif architecture_level >= 2 and style == "anticline_structural_trap":
        architecture_variant = "folded_facies_trap"

    x = np.arange(nx, dtype=np.float64) * dx
    z = np.arange(nz, dtype=np.float64) * dz
    zz = z[:, None]
    xx = x[None, :]
    water_depth = rng.uniform(*p["water_depth_m"])
    relief = rng.uniform(*p["seafloor_relief_m"])
    if continuity is None:
        broad = _smooth_unit(rng, nx, rng.uniform(13.0, 28.0))
        wave = np.sin(2.0 * np.pi * (x / rng.uniform(1300.0, 2600.0) + rng.random()))
        seafloor = water_depth + relief * (0.72 * broad + 0.28 * wave)
        seafloor_p95_slope = float(np.percentile(np.abs(np.diff(seafloor) / dx), 95.0))
    else:
        broad_sigma = rng.uniform(*continuity["seafloor_gaussian_sigma_m"]) / dx
        broad = _smooth_unit(rng, nx, broad_sigma)
        wavelength = rng.uniform(*continuity["seafloor_wavelength_m"])
        wave = np.sin(2.0 * np.pi * (x / wavelength + rng.random()))
        shape = 0.68 * broad + 0.32 * wave
        shape -= shape.mean()
        shape /= max(float(shape.std()), 1.0e-8)
        seafloor = water_depth + relief * shape
        slope_limit = rng.uniform(*continuity["seafloor_p95_slope"])
        seafloor, seafloor_p95_slope = _limit_curve_slope(
            seafloor, dx_m=dx, maximum_p95_absolute_slope=slope_limit
        )

    # Beds use a depositional coordinate; the BSR is generated independently
    # so it can physically cross stratigraphy instead of being a copied layer.
    # Resolve faulting as a finite-width damage zone.  A near-step offset
    # becomes an artificial vertical impedance edge and can dominate RTM.
    structural_prior = p.get("structural_fault_prior", {})
    fault_x = rng.uniform(*structural_prior.get("x_model_fraction", [0.35, 0.68])) * x[-1]
    fault_throw = rng.choice([-1.0, 1.0]) * rng.uniform(
        *structural_prior.get("throw_m", [18.0, 55.0])
    )
    fault_transition = rng.uniform(
        *structural_prior.get("damage_zone_halfwidth_m", [18.0, 45.0])
    )
    fault_step = 0.5 * (1.0 + np.tanh((x - fault_x) / fault_transition))
    structure_shift = np.zeros(nx)
    style_geometry: dict[str, float | int | list[float]] = {}
    if style == "fault_gas_chimney":
        structure_shift += fault_throw * fault_step
        style_geometry.update(
            {
                "fault_x_m": float(fault_x),
                "fault_throw_m": float(fault_throw),
                "fault_damage_zone_halfwidth_m": float(fault_transition),
            }
        )
    if style == "sediment_wave_crosscut":
        wave_prior = continuity["styles"][style] if continuity is not None else {}
        wavelength_range = wave_prior.get("structure_wavelength_m", (800.0, 1500.0))
        amplitude_range = wave_prior.get("structure_amplitude_m", (35.0, 80.0))
        structure_shift += rng.uniform(*amplitude_range) * np.sin(
            2 * np.pi * x / rng.uniform(*wavelength_range)
        )
    if style == "anticline_structural_trap":
        anticline_prior = continuity["styles"][style] if continuity is not None else {}
        anticline_center = rng.uniform(
            *anticline_prior.get("anticline_center_fraction", [0.38, 0.62])
        ) * x[-1]
        anticline_sigma = rng.uniform(
            *anticline_prior.get("anticline_halfwidth_m", [650.0, 1100.0])
        )
        anticline_relief = rng.uniform(
            *anticline_prior.get("anticline_relief_m", [55.0, 125.0])
        )
        if architecture_level >= 3:
            asymmetry_range = anticline_prior.get(
                "anticline_wing_width_factor", [0.78, 1.28]
            )
            left_width_factor = rng.uniform(*asymmetry_range)
            right_width_factor = rng.uniform(*asymmetry_range)
            local_sigma = np.where(
                x < anticline_center,
                anticline_sigma * left_width_factor,
                anticline_sigma * right_width_factor,
            )
        else:
            left_width_factor = right_width_factor = 1.0
            local_sigma = np.full_like(x, anticline_sigma)
        anticline_shape = np.exp(-0.5 * ((x - anticline_center) / local_sigma) ** 2)
        structure_shift += anticline_relief * anticline_shape
        style_geometry.update(
            {
                "anticline_center_m": float(anticline_center),
                "anticline_halfwidth_m": float(anticline_sigma),
                "anticline_relief_m": float(anticline_relief),
                "anticline_left_width_factor": float(left_width_factor),
                "anticline_right_width_factor": float(right_width_factor),
            }
        )
    mbsf = zz - seafloor[None, :]
    depositional_tilt = rng.uniform(-0.012, 0.012)
    depositional = mbsf + structure_shift[None, :] + depositional_tilt * (xx - x.mean())

    phi0 = rng.uniform(*p["background_porosity_at_seafloor"])
    compaction = rng.uniform(*p["porosity_compaction_length_m"])
    phi = 0.29 + (phi0 - 0.29) * np.exp(-np.maximum(mbsf, 0.0) / compaction)
    lithology = _stochastic_stratigraphy(
        rng, depositional, x, dx_m=dx, continuity=continuity
    )
    phi += rng.uniform(0.0045, 0.0120) * lithology
    phi = np.clip(phi, 0.38, 0.70)

    bsr_mbsf = rng.uniform(*p["bsr_depth_mbsf"])
    bsr_sigma = (
        rng.uniform(*continuity["bsr_gaussian_sigma_m"]) / dx
        if continuity is not None else rng.uniform(18.0, 35.0)
    )
    bsr_relief = rng.uniform(10.0, 38.0) * _smooth_unit(rng, nx, bsr_sigma)
    bsr_dip = rng.uniform(-0.010, 0.010) * (x - x.mean())
    bsr = seafloor + bsr_mbsf + bsr_relief + bsr_dip
    if style == "fault_gas_chimney":
        bsr += 0.35 * fault_throw * fault_step
    distance_to_bsr = zz - bsr[None, :]

    saturation_model = config.get("saturation_model")
    if saturation_model is None:
        hydrate_top = rng.uniform(185.0, min(310.0, bsr_mbsf - 80.0))
    else:
        hydrate_top = rng.uniform(*saturation_model["hydrate_top_mbsf"])
    hydrate_zone = (mbsf >= hydrate_top) & (distance_to_bsr < 0.0)
    if saturation_model is None:
        gas_thickness = rng.uniform(*p["free_gas_zone_thickness_m"])
    else:
        gas_thickness = rng.uniform(
            *saturation_model["styles"][style]["free_gas_zone_thickness_m"]
        )
    gas_zone = (distance_to_bsr >= 0.0) & (distance_to_bsr <= gas_thickness)

    if saturation_model is None:
        vertical_pref = np.exp(
            -0.5
            * (
                (distance_to_bsr + rng.uniform(45.0, 125.0))
                / rng.uniform(75.0, 150.0)
            )
            ** 2
        )
        upper_center_mbsf = None
        near_bsr_offset_m = None
    else:
        vertical = saturation_model["hydrate_vertical_distribution"]
        upper_center_mbsf = rng.uniform(*vertical["upper_enrichment_center_mbsf"])
        upper_sigma = rng.uniform(*vertical["upper_enrichment_sigma_m"])
        near_bsr_offset_m = rng.uniform(*vertical["near_bsr_center_offset_m"])
        near_bsr_sigma = rng.uniform(*vertical["near_bsr_sigma_m"])
        upper = np.exp(-0.5 * ((mbsf - upper_center_mbsf) / upper_sigma) ** 2)
        lower = np.exp(
            -0.5 * ((distance_to_bsr + near_bsr_offset_m) / near_bsr_sigma) ** 2
        )
        vertical_pref = (
            float(vertical["diffuse_weight"])
            + rng.uniform(*vertical["upper_enrichment_weight"]) * upper
            + rng.uniform(*vertical["near_bsr_weight"]) * lower
        )
        vertical_pref /= max(float(vertical_pref[hydrate_zone].max()), 1.0e-8)
    if continuity is None:
        style_continuity = None
        h_sigma_x, g_sigma_x = 10.0, 8.0
        h_strength = rng.uniform(0.28, 0.58)
        g_strength = rng.uniform(0.35, 0.75)
    else:
        style_continuity = continuity["styles"][style]
        h_sigma_x = rng.uniform(*style_continuity["hydrate_texture_sigma_m"]) / dx
        g_sigma_x = rng.uniform(*style_continuity["gas_texture_sigma_m"]) / dx
        h_strength = rng.uniform(*style_continuity["hydrate_texture_strength"])
        g_strength = rng.uniform(*style_continuity["gas_texture_strength"])
    h_texture = np.exp(
        h_strength * _smooth_unit(rng, (nz, nx), (3.0, h_sigma_x))
    )
    g_texture = np.exp(
        g_strength * _smooth_unit(rng, (nz, nx), (2.0, g_sigma_x))
    )
    # Two-dimensional gates allow truly stratigraphic lenses rather than
    # vertical columns produced by a one-dimensional lateral multiplier.
    h_gate = np.ones((nz, nx), dtype=np.float64)
    g_gate = np.ones((nz, nx), dtype=np.float64)

    if style == "localized_enriched_lenses":
        if style_continuity is not None and "lens_vertical_sigma_m" in style_continuity:
            lens_count = int(
                rng.integers(
                    int(style_continuity["lens_count"][0]),
                    int(style_continuity["lens_count"][1]) + 1,
                )
            )
            release_x0, release_x1 = continuity["release_window_m"]
            fractions = (
                np.arange(lens_count) + rng.uniform(0.22, 0.78, size=lens_count)
            ) / lens_count
            centers = release_x0 + fractions * (release_x1 - release_x0)
            lens_field = np.zeros((nz, nx), dtype=np.float64)
            reservoir_field = np.zeros((nz, nx), dtype=np.float64)
            lens_centers: list[float] = []
            for center in centers:
                width = rng.uniform(*style_continuity["lens_halfwidth_m"])
                vertical_sigma = rng.uniform(*style_continuity["lens_vertical_sigma_m"])
                bsr_offset = rng.uniform(*style_continuity["lens_bsr_offset_m"])
                if architecture_level >= 2:
                    center_index = int(np.clip(np.rint(center / dx), 0, nx - 1))
                    reservoir_mbsf = (
                        bsr[center_index] - seafloor[center_index] - bsr_offset
                    )
                    reservoir_depositional = (
                        reservoir_mbsf
                        + structure_shift[center_index]
                        + depositional_tilt * (x[center_index] - x.mean())
                    )
                    if architecture_level >= 3:
                        width_factors = style_continuity.get(
                            "lens_width_asymmetry_factor", [0.72, 1.35]
                        )
                        left_width = width * rng.uniform(*width_factors)
                        right_width = width * rng.uniform(*width_factors)
                        local_width = np.where(xx < center, left_width, right_width)
                        taper_strength = rng.uniform(
                            *style_continuity.get(
                                "lens_thickness_taper_strength", [0.12, 0.30]
                            )
                        )
                        normalized_x = np.clip(
                            (xx - center) / max(left_width, right_width), -1.5, 1.5
                        )
                        taper_direction = rng.choice([-1.0, 1.0])
                        local_vertical_sigma = vertical_sigma * np.clip(
                            1.0 + taper_direction * taper_strength * normalized_x,
                            0.58,
                            1.42,
                        )
                        relief_fraction = rng.uniform(
                            *style_continuity.get(
                                "lens_internal_relief_fraction", [0.08, 0.22]
                            )
                        )
                        relief_wavelength = rng.uniform(2.2, 3.8) * width
                        relief_phase = rng.uniform(0.0, 2.0 * np.pi)
                        internal_relief = (
                            relief_fraction
                            * vertical_sigma
                            * np.sin(
                                2.0 * np.pi * (xx - center) / relief_wavelength
                                + relief_phase
                            )
                        )
                        lens = np.exp(
                            -0.5 * ((xx - center) / local_width) ** 2
                            -0.5
                            * (
                                (depositional - reservoir_depositional - internal_relief)
                                / local_vertical_sigma
                            )
                            ** 2
                        )
                    else:
                        left_width = right_width = width
                        taper_strength = relief_fraction = 0.0
                        lens = np.exp(
                            -0.5 * ((xx - center) / width) ** 2
                            -0.5
                            * ((depositional - reservoir_depositional) / vertical_sigma) ** 2
                        )
                else:
                    left_width = right_width = width
                    taper_strength = relief_fraction = 0.0
                    lens = np.exp(
                        -0.5 * ((xx - center) / width) ** 2
                        -0.5 * ((distance_to_bsr + bsr_offset) / vertical_sigma) ** 2
                    )
                lens_field = np.maximum(lens_field, lens)
                lens_centers.append(float(center))
                if architecture_level >= 3:
                    style_geometry.setdefault("lens_left_halfwidths_m", []).append(
                        float(left_width)
                    )
                    style_geometry.setdefault("lens_right_halfwidths_m", []).append(
                        float(right_width)
                    )
                    style_geometry.setdefault("lens_taper_strengths", []).append(
                        float(taper_strength)
                    )
                    style_geometry.setdefault("lens_internal_relief_fractions", []).append(
                        float(relief_fraction)
                    )
            if architecture_level >= 2:
                facies_quantile = float(style_continuity.get("facies_quantile", 0.55))
                facies_slope = float(style_continuity.get("facies_logistic_slope", 2.2))
                reservoir_floor = float(style_continuity.get("facies_reservoir_floor", 0.30))
                lithology_threshold = float(
                    np.quantile(lithology[hydrate_zone], facies_quantile)
                )
                facies_suitability = 1.0 / (
                    1.0 + np.exp(-facies_slope * (lithology - lithology_threshold))
                )
                reservoir_field = lens_field * (
                    reservoir_floor + (1.0 - reservoir_floor) * facies_suitability
                )
                reservoir_field /= max(float(reservoir_field.max()), 1.0e-8)
                phi += rng.uniform(0.005, 0.014) * reservoir_field
            else:
                reservoir_field = lens_field
            background = float(style_continuity.get("hydrate_background_gate", 0.015))
            h_gate *= background + (1.0 - background) * reservoir_field
            # Gas remains broadly connected in this stratigraphic-lens family;
            # this separates it from the fairway family where both phases gap.
            lens_projection = np.max(reservoir_field, axis=0, keepdims=True)
            g_gate *= 0.72 + 0.28 * lens_projection
            style_geometry.update(
                {
                    "lens_count": lens_count,
                    "lens_centers_m": lens_centers,
                    "reservoir_control": (
                        "depositional_coordinate_plus_lithology"
                        if architecture_level >= 2 else "BSR_relative_geometry"
                    ),
                }
            )
        else:
            h_gate *= 0.22
            for _ in range(3):
                c = rng.uniform(0.1, 0.9) * x[-1]
                width = (
                    rng.uniform(*style_continuity["lens_halfwidth_m"])
                    if style_continuity is not None else rng.uniform(180.0, 420.0)
                )
                h_gate += rng.uniform(0.6, 1.15) * np.exp(
                    -0.5 * ((x[None, :] - c) / width) ** 2
                )
    elif style == "discontinuous_bsr":
        if style_continuity is None:
            gate_curve = _smooth_unit(rng, nx, rng.uniform(7.0, 15.0))
            gate = gaussian_filter(
                (gate_curve > np.quantile(gate_curve, 0.35)).astype(float), 2.0
            )
        else:
            # A discontinuous BSR should contain a few geologically broad
            # fairways, not dozens of grid-scale random fragments.
            segment_count = int(
                rng.integers(
                    int(style_continuity["segment_count"][0]),
                    int(style_continuity["segment_count"][1]) + 1,
                )
            )
            release_x0, release_x1 = continuity["release_window_m"]
            release_width = release_x1 - release_x0
            fractions = 0.20 + 0.60 * (
                np.arange(segment_count) + rng.uniform(0.20, 0.80, size=segment_count)
            ) / segment_count
            centers = release_x0 + fractions * release_width
            gate = np.zeros(nx, dtype=np.float64)
            for center in centers:
                halfwidth = rng.uniform(*style_continuity["segment_halfwidth_m"])
                distance = np.abs(x - center) / halfwidth
                compact = np.where(
                    distance < 1.0, 0.5 * (1.0 + np.cos(np.pi * distance)), 0.0
                )
                gate = np.maximum(gate, compact)
            edge_sigma = rng.uniform(*style_continuity["gate_edge_sigma_m"]) / dx
            gate = gaussian_filter(gate, edge_sigma, mode="nearest")
            gate /= max(float(gate.max()), 1.0e-8)
        h_background = (
            float(style_continuity["hydrate_background_gate"])
            if style_continuity is not None else 0.18
        )
        g_background = (
            float(style_continuity["gas_background_gate"])
            if style_continuity is not None else 0.04
        )
        if architecture_level >= 2 and architecture_variant == "gas_patchy":
            # Hydrate remains diffuse while the free-gas support is patchy.
            h_gate *= 0.82 + 0.18 * gate[None, :]
            g_gate *= g_background + (1.0 - g_background) * gate[None, :]
        elif architecture_level >= 2 and architecture_variant == "hydrate_patchy":
            # Stratigraphically localized hydrate above a broadly connected
            # free-gas band is a second mechanism for an irregular BSR.
            h_gate *= h_background + (1.0 - h_background) * gate[None, :]
            g_gate *= 0.78 + 0.22 * gate[None, :]
        else:
            h_gate *= h_background + (1.0 - h_background) * gate[None, :]
            g_gate *= g_background + (1.0 - g_background) * gate[None, :]
        if style_continuity is not None and "hydrate_fairway_bsr_offset_m" in style_continuity:
            fairway_offset = rng.uniform(
                *style_continuity["hydrate_fairway_bsr_offset_m"]
            )
            fairway_sigma = rng.uniform(
                *style_continuity["hydrate_fairway_sigma_m"]
            )
            fairway_layer = np.exp(
                -0.5 * ((distance_to_bsr + fairway_offset) / fairway_sigma) ** 2
            )
            if not (
                architecture_level >= 2
                and architecture_variant == "gas_patchy"
            ):
                h_gate *= 0.01 + 0.99 * fairway_layer
            style_geometry.update(
                {
                    "fairway_segment_count": segment_count,
                    "fairway_centers_m": list(map(float, centers)),
                    "patchiness_mechanism": architecture_variant,
                    "hydrate_fairway_bsr_offset_m": float(fairway_offset),
                    "hydrate_fairway_sigma_m": float(fairway_sigma),
                }
            )
    elif style == "fault_gas_chimney":
        chimney_width = (
            rng.uniform(*style_continuity["chimney_halfwidth_m"])
            if style_continuity is not None else rng.uniform(100.0, 210.0)
        )
        if architecture_level >= 2:
            bend_period = rng.uniform(650.0, 1100.0)
            bend_phase = rng.uniform(0.0, 2.0 * np.pi)
            centerline = fault_x + 0.18 * chimney_width * np.sin(
                2.0 * np.pi * zz / bend_period + bend_phase
            )
            vertical_envelope = (
                1.0 / (1.0 + np.exp(-(mbsf - 65.0) / 35.0))
            )
            chimney = (
                np.exp(-0.5 * ((xx - centerline) / chimney_width) ** 2)
                * vertical_envelope
            )
            flank_offset = 1.20 * chimney_width
            flank_sigma = 0.58 * chimney_width
            flanks = vertical_envelope * (
                np.exp(-0.5 * ((xx - (centerline - flank_offset)) / flank_sigma) ** 2)
                + np.exp(-0.5 * ((xx - (centerline + flank_offset)) / flank_sigma) ** 2)
            )
            if architecture_variant == "thermal_peripheral_hydrate":
                h_gate *= 0.22 + 0.95 * flanks + 0.12 * (1.0 - chimney)
                g_gate *= 0.10 + 2.25 * chimney
            elif architecture_variant == "fracture_fill_hydrate":
                h_gate *= 0.20 + 1.55 * chimney + 0.25 * flanks
                g_gate *= 0.16 + 1.65 * chimney
            else:
                internal_texture = 1.0 / (
                    1.0
                    + np.exp(
                        -1.5
                        * _smooth_unit(
                            rng, (nz, nx), (14.0, max(chimney_width / dx, 4.0))
                        )
                    )
                )
                h_gate *= 0.18 + chimney * (0.45 + 0.85 * internal_texture) + 0.30 * flanks
                g_gate *= 0.10 + chimney * (1.50 + 0.70 * (1.0 - internal_texture))
        else:
            chimney = np.exp(-0.5 * ((x - fault_x) / chimney_width) ** 2)[None, :]
            flank_offset = 1.25 * chimney_width
            flank_sigma = 0.55 * chimney_width
            flanks = (
                np.exp(-0.5 * ((x - (fault_x - flank_offset)) / flank_sigma) ** 2)
                + np.exp(-0.5 * ((x - (fault_x + flank_offset)) / flank_sigma) ** 2)
            )[None, :]
            h_gate *= 0.18 + 0.92 * flanks + 0.20 * (1.0 - chimney)
            g_gate *= 0.12 + 2.20 * chimney
        disruption = chimney * np.exp(-0.5 * ((distance_to_bsr + 40.0) / 260.0) ** 2)
        phi += rng.uniform(0.008, 0.020) * disruption
        style_geometry.update(
            {
                "chimney_halfwidth_m": float(chimney_width),
                "chimney_mechanism": architecture_variant,
            }
        )
    elif style == "anticline_structural_trap":
        anticline_center = float(style_geometry["anticline_center_m"])
        anticline_sigma = float(style_geometry["anticline_halfwidth_m"])
        crest_width = rng.uniform(
            *style_continuity.get("crest_saturation_halfwidth_m", [450.0, 850.0])
        )
        crest = np.exp(-0.5 * ((x - anticline_center) / crest_width) ** 2)[None, :]
        reservoir_offset = rng.uniform(
            *style_continuity.get("hydrate_reservoir_bsr_offset_m", [35.0, 85.0])
        )
        reservoir_sigma = rng.uniform(
            *style_continuity.get("hydrate_reservoir_sigma_m", [24.0, 55.0])
        )
        if architecture_level >= 2:
            center_index = int(np.clip(np.rint(anticline_center / dx), 0, nx - 1))
            reservoir_mbsf = (
                bsr[center_index] - seafloor[center_index] - reservoir_offset
            )
            reservoir_depositional = (
                reservoir_mbsf
                + structure_shift[center_index]
                + depositional_tilt * (x[center_index] - x.mean())
            )
            reservoir_layer = np.exp(
                -0.5
                * ((depositional - reservoir_depositional) / reservoir_sigma) ** 2
            )
            facies_quantile = float(style_continuity.get("facies_quantile", 0.45))
            facies_slope = float(style_continuity.get("facies_logistic_slope", 1.8))
            lithology_threshold = float(
                np.quantile(lithology[hydrate_zone], facies_quantile)
            )
            reservoir_quality = 0.45 + 0.55 / (
                1.0 + np.exp(-facies_slope * (lithology - lithology_threshold))
            )
            reservoir_layer *= reservoir_quality
            if architecture_level >= 3:
                reservoir_variability = _smooth_unit(
                    rng,
                    nx,
                    rng.uniform(
                        *style_continuity.get(
                            "reservoir_completeness_sigma_m", [420.0, 780.0]
                        )
                    )
                    / dx,
                )
                completeness_strength = rng.uniform(
                    *style_continuity.get(
                        "reservoir_completeness_strength", [0.12, 0.28]
                    )
                )
                reservoir_completeness = np.clip(
                    1.0 + completeness_strength * reservoir_variability[None, :],
                    0.58,
                    1.30,
                )
                reservoir_layer *= reservoir_completeness
                style_geometry["reservoir_completeness_strength"] = float(
                    completeness_strength
                )
        else:
            reservoir_layer = np.exp(
                -0.5 * ((distance_to_bsr + reservoir_offset) / reservoir_sigma) ** 2
            )
        background = float(style_continuity.get("hydrate_background_gate", 0.025))
        h_gate *= background + (0.35 + 1.25 * crest) * reservoir_layer
        g_gate *= 0.08 + 2.00 * crest
        # Weak porosity enhancement represents a coarse-grained reservoir bed
        # at the crest without using saturation labels to manufacture an edge.
        phi += rng.uniform(0.004, 0.010) * crest * reservoir_layer
        style_geometry.update(
            {
                "crest_saturation_halfwidth_m": float(crest_width),
                "hydrate_reservoir_bsr_offset_m": float(reservoir_offset),
                "hydrate_reservoir_sigma_m": float(reservoir_sigma),
                "reservoir_control": (
                    "folded_depositional_layer_plus_lithology"
                    if architecture_level >= 2 else "BSR_relative_geometry"
                ),
            }
        )
    elif style == "sediment_wave_crosscut":
        gate_wavelength = (
            rng.uniform(*style_continuity["saturation_wavelength_m"])
            if style_continuity is not None else rng.uniform(700.0, 1200.0)
        )
        h_gate *= 0.65 + 0.35 * (
            1.0 + np.sin(2 * np.pi * x[None, :] / gate_wavelength)
        ) / 2

    transition_mask = np.zeros_like(hydrate_zone, dtype=bool)
    coexistence_enabled = False
    coexistence_thickness_m = 0.0
    if saturation_model is None:
        h_mean = rng.uniform(*p["hydrate_pore_saturation_background"])
        h_cap = rng.uniform(*p["hydrate_pore_saturation_local_max"])
        if style == "localized_enriched_lenses":
            h_mean = rng.uniform(0.055, 0.105)
            h_cap = rng.uniform(0.18, 0.22)
        g_mean = rng.uniform(*p["free_gas_pore_saturation_background"])
        g_cap = rng.uniform(*p["free_gas_pore_saturation_local_max"])
        Sh = _scale_to_cap(
            h_texture * h_gate * (0.40 + 0.60 * vertical_pref),
            hydrate_zone,
            h_mean,
            h_cap,
        )
        Sg = _scale_to_cap(
            g_texture
            * g_gate
            * np.exp(-np.maximum(distance_to_bsr, 0.0) / 180.0),
            gas_zone,
            g_mean,
            g_cap,
        )
    else:
        style_saturation = saturation_model["styles"][style]
        h_mean = rng.uniform(*style_saturation["hydrate_zone_mean"])
        h_cap = rng.uniform(*style_saturation["hydrate_local_soft_cap"])
        g_mean = rng.uniform(*style_saturation["gas_zone_mean"])
        g_cap = rng.uniform(*style_saturation["gas_local_soft_cap"])
        Sh = _scale_to_soft_cap(
            h_texture * h_gate * vertical_pref,
            hydrate_zone,
            h_mean,
            h_cap,
        )
        gas_decay = rng.uniform(*saturation_model["gas_decay_length_m"])
        Sg = _scale_to_soft_cap(
            g_texture
            * g_gate
            * np.exp(-np.maximum(distance_to_bsr, 0.0) / gas_decay),
            gas_zone,
            g_mean,
            g_cap,
        )
        if (
            architecture_level >= 2
            and style == "fault_gas_chimney"
            and architecture_variant == "leaky_mixed_chimney"
        ):
            # A small transient gas tail inside the GHSZ represents vigorous
            # leakage without turning the whole chimney into a gas reservoir.
            leak_height = rng.uniform(35.0, 90.0)
            leak_mask = (
                hydrate_zone
                & (distance_to_bsr >= -leak_height)
                & (distance_to_bsr < 0.0)
            )
            leak_mean = rng.uniform(0.00025, 0.0010)
            leak_cap = rng.uniform(0.0015, 0.0040)
            Sg += _scale_to_soft_cap(
                chimney
                * np.exp(distance_to_bsr / max(leak_height, dz)),
                leak_mask,
                leak_mean,
                leak_cap,
            )
            style_geometry.update(
                {
                    "transient_gas_tail_height_m": float(leak_height),
                    "transient_gas_tail_mean": float(leak_mean),
                }
            )
        transition = saturation_model["controlled_coexistence"]
        coexistence_enabled = bool(rng.random() < transition["probability"])
        if coexistence_enabled:
            coexistence_thickness_m = rng.uniform(*transition["thickness_m"])
            transition_mask = (
                (distance_to_bsr >= 0.0)
                & (distance_to_bsr <= coexistence_thickness_m)
                & (mbsf >= hydrate_top)
            )
            transition_mean = rng.uniform(*transition["hydrate_mean"])
            transition_cap = rng.uniform(*transition["hydrate_local_soft_cap"])
            Sh += _scale_to_soft_cap(
                h_texture
                * h_gate
                * np.exp(-distance_to_bsr / max(coexistence_thickness_m, dz)),
                transition_mask,
                transition_mean,
                transition_cap,
            )
        pore_cap = float(saturation_model["maximum_Sh_plus_Sg"])
        total = Sh + Sg
        excessive = total > pore_cap
        Sh[excessive] *= pore_cap / total[excessive]
        Sg[excessive] *= pore_cap / total[excessive]
    phi = np.clip(phi, 0.38, 0.70)
    water = mbsf < 0.0
    Sh[water] = 0.0
    Sg[water] = 0.0
    phi[water] = 1.0

    rock_physics = config.get("rock_physics", {})
    hydrate_scale = rng.uniform(
        *rock_physics.get("hydrate_response_scale", [0.82, 1.12])
    )
    gas_patchiness_range = rock_physics.get("gas_patchiness")
    if gas_patchiness_range is None:
        gas_scale = rng.uniform(
            *rock_physics.get("gas_response_scale", [0.65, 0.95])
        )
        gas_patchiness = None
    else:
        gas_patchiness = rng.uniform(*gas_patchiness_range)
        gas_scale = None
    vp, rho = blake_prior_acoustic_properties(
        phi,
        Sh,
        Sg,
        water,
        hydrate_response_scale=hydrate_scale,
        gas_response_scale=(0.85 if gas_scale is None else gas_scale),
        gas_patchiness=gas_patchiness,
    )
    impedance = vp * rho
    physical_arrays = {
        "x_m": x.astype(np.float32),
        "z_m": z.astype(np.float32),
        "seafloor_m": seafloor.astype(np.float32),
        "bsr_m": bsr.astype(np.float32),
        "lithology_proxy_full": lithology.astype(np.float32),
        "phi_full": phi.astype(np.float32),
        "Sh_full": Sh,
        "Sg_full": Sg,
        "Vp_full": vp,
        "rho_full": rho,
        "AI_full": impedance.astype(np.float32),
    }
    physical_metadata = {
        "version": config["version"],
        "sample_index": sample_index,
        "seed": seed,
        "style": style,
        "style_geometry": style_geometry,
        "water_depth_mean_m": float(seafloor.mean()),
        "water_depth_range_m": [float(seafloor.min()), float(seafloor.max())],
        "bsr_depth_mbsf_mean": float(np.mean(bsr - seafloor)),
        "hydrate_mean_in_zone": float(Sh[hydrate_zone].mean()),
        "hydrate_max": float(Sh.max()),
        "gas_mean_in_zone": float(Sg[gas_zone].mean()),
        "gas_max": float(Sg.max()),
        "hydrate_q95_positive": float(np.quantile(Sh[Sh > 0.0], 0.95)),
        "gas_q95_positive": float(np.quantile(Sg[Sg > 0.0], 0.95)),
        "hydrate_fraction_within_one_percent_of_max": float(
            np.mean(Sh[Sh > 0.0] >= 0.99 * float(Sh.max()))
        ),
        "gas_fraction_within_one_percent_of_max": float(
            np.mean(Sg[Sg > 0.0] >= 0.99 * float(Sg.max()))
        ),
        "gas_zone_thickness_m": float(gas_thickness),
        "coexistence_enabled": coexistence_enabled,
        "coexistence_thickness_m": float(coexistence_thickness_m),
        "coexistence_pixel_fraction": float(np.mean((Sh > 0.0) & (Sg > 0.0))),
        "upper_enrichment_center_mbsf": upper_center_mbsf,
        "near_bsr_enrichment_offset_m": near_bsr_offset_m,
        "hydrate_response_scale": hydrate_scale,
        "gas_response_scale": gas_scale,
        "gas_patchiness": gas_patchiness,
        "architecture_level": architecture_level,
        "architecture_variant": architecture_variant,
        "seafloor_p95_absolute_slope": seafloor_p95_slope,
        "lateral_continuity_parameters": style_continuity,
        "physical_grid_shape": [nz, nx],
        "physical_grid_spacing_m": [dz, dx],
        "provenance": config["provenance"],
    }
    if not include_proxy_image:
        return physical_arrays, physical_metadata

    k = rng.uniform(*im["vertical_wavenumber_cycles_per_km"])
    lateral_sigma = rng.uniform(*im["horizontal_psf_sigma_traces"])
    snr = rng.uniform(*im["snr_db"])
    attenuation_length = rng.uniform(*im["sediment_amplitude_decay_length_m"])
    illumination_variability = rng.uniform(*im["illumination_variability"])
    reflectivity, psdm = _migrated_impedance_image(
        impedance, dz, water, rng, k, lateral_sigma, snr, attenuation_length, illumination_variability
    )

    crop = lambda a, order=1: _seafloor_relative_crop(a, seafloor, dz, order=order)
    psdm_crop = crop(psdm)
    scale = max(float(np.percentile(np.abs(psdm_crop[:, :nx]), 99.5)), 1.0e-9)
    input_psdm = np.clip(psdm_crop / scale, -1.0, 1.0).astype(np.float32)
    valid_mask = np.zeros_like(input_psdm, dtype=np.uint8)
    valid_mask[:, :nx] = 1

    arrays = {
        **physical_arrays,
        "reflectivity_full": reflectivity, "psdm_full": psdm,
        "input_psdm": input_psdm, "target_Sh": crop(Sh), "target_Sg": crop(Sg),
        "target_phi": crop(phi), "target_Vp": crop(vp),
        "valid_mask": valid_mask,
    }
    metadata = {
        **physical_metadata,
        "structural_fault_prior": config["geological_priors"].get("structural_fault_prior", {}),
        "vertical_wavenumber_cycles_per_km": k,
        "horizontal_psf_sigma_traces": lateral_sigma, "snr_db": snr,
        "sediment_amplitude_decay_length_m": attenuation_length,
        "illumination_variability": illumination_variability,
        "input_shape": list(input_psdm.shape), "target_shape": [2, *input_psdm.shape],
        "coordinate_system": "seafloor-relative crop: row 64 is the picked seafloor; dx=37.5 m, dz=3 m",
        "provenance": config["provenance"],
    }
    return arrays, metadata
