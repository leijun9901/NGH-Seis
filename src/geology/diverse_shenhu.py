"""Literature-bounded diverse Shenhu hydrate/free-gas scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def _normalized_smooth_noise(
    rng: np.random.Generator, shape: tuple[int, int], sigma: tuple[float, float]
) -> np.ndarray:
    noise = gaussian_filter(rng.normal(size=shape), sigma=sigma, mode="reflect")
    noise -= noise.mean()
    noise /= max(noise.std(), 1.0e-8)
    return noise


def _smooth_1d(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    curve = gaussian_filter(rng.normal(size=n), sigma=sigma, mode="reflect")
    curve -= curve.mean()
    curve /= max(curve.std(), 1.0e-8)
    return curve


def _scale_masked_mean(field: np.ndarray, mask: np.ndarray, target: float, cap: float) -> np.ndarray:
    out = np.maximum(field, 0.0) * mask
    if not np.any(mask) or target <= 0.0:
        return np.zeros_like(out, dtype=np.float32)
    for _ in range(5):
        mean = float(out[mask].mean())
        if mean <= 1.0e-10:
            break
        out = np.minimum(out * (target / mean), cap) * mask
    return out.astype(np.float32)


def _choose_family(config: dict, sample_index: int) -> str:
    names = list(config["families"])
    return names[(sample_index - 1) % len(names)]


def generate_diverse_shenhu(
    config_path: str | Path,
    sample_index: int,
    seed: int,
    *,
    family_name_override: str | None = None,
    style_override: str | None = None,
    geometry_seed: int | None = None,
    property_seed: int | None = None,
    parent_parameters: dict[str, float] | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    grid = config["grid"]
    nx, nz = grid["nx"], grid["nz"]
    dx, dz = grid["dx_m"], grid["dz_m"]
    # Geometry and property randomness are deliberately independent.  Several
    # conditional property realizations may therefore share one parent geology
    # without accidentally changing its interfaces or structural style.
    rng_geo = np.random.default_rng(seed if geometry_seed is None else geometry_seed)
    rng_prop = np.random.default_rng(seed if property_seed is None else property_seed)
    parent_parameters = parent_parameters or {}
    family_name = family_name_override or _choose_family(config, sample_index)
    # The hierarchical configuration uses the full borehole identifier while
    # The source metadata shortens SHSC-4J1 to SHSC-4.
    if family_name not in config["families"] and family_name == "SHSC-4J1":
        family_name = "SHSC-4"
    family = config["families"][family_name]
    styles = config["structural_styles"]
    style = style_override or styles[((sample_index - 1) // len(config["families"])) % len(styles)]

    def quantile_or_uniform(pair: list[float], key: str, rng: np.random.Generator) -> float:
        q = parent_parameters.get(key)
        if q is None:
            return float(rng.uniform(pair[0], pair[1]))
        return float(pair[0] + np.clip(q, 0.0, 1.0) * (pair[1] - pair[0]))

    water_depth = quantile_or_uniform(family["water_depth_range_m"], "water_depth_q", rng_geo)
    hydrate_top_mbsf = quantile_or_uniform(family["hydrate_top_range_mbsf"], "hydrate_top_q", rng_geo)
    thickness = {
        key: quantile_or_uniform(value, "thickness_q", rng_geo)
        for key, value in family["thickness_ranges_m"].items()
    }
    porosity_targets = {
        key: quantile_or_uniform(value, f"phi_{key}_q", rng_prop)
        for key, value in family["porosity_ranges"].items()
    }
    saturation_targets = {
        key: quantile_or_uniform(value, f"{key}_q", rng_prop)
        for key, value in family["mean_saturation_ranges"].items()
    }
    hydrate_response_scale = quantile_or_uniform(
        config["rock_physics_ranges"]["hydrate_response_scale"], "hydrate_response_q", rng_prop
    )
    gas_response_scale = quantile_or_uniform(
        config["rock_physics_ranges"]["gas_response_scale"], "gas_response_q", rng_prop
    )

    x = np.arange(nx, dtype=np.float64) * dx
    z = np.arange(nz, dtype=np.float64) * dz
    zz = z[:, None]
    lateral_q = parent_parameters.get("lateral_scale_q", rng_geo.random())
    long_relief = _smooth_1d(rng_geo, nx, 20.0 + 35.0 * lateral_q)
    short_relief = _smooth_1d(rng_geo, nx, 5.0 + 10.0 * lateral_q)
    relief_amp = 4.0 + 14.0 * parent_parameters.get("relief_amplitude_q", rng_geo.random())
    seafloor = water_depth + relief_amp * long_relief + rng_geo.uniform(0.5, 3) * short_relief
    sf = seafloor[None, :]
    mbsf = zz - sf

    boundary_relief = rng_geo.uniform(3, 14) * _smooth_1d(rng_geo, nx, 12.0 + 28.0 * lateral_q)
    dip_q = parent_parameters.get("regional_dip_q", rng_geo.random())
    regional_dip = (-0.018 + 0.036 * dip_q) * (x - x.mean())
    # A shared stratigraphic displacement field controls every geological
    # interface.  Downstream high-resolution laminae also read this field, so
    # a fault cannot offset only the hydrate/gas labels while leaving the
    # sedimentary background continuous.
    stratigraphic_shift = boundary_relief + regional_dip
    fault_displacement = np.zeros(nx, dtype=np.float64)
    fault_x = float("nan")
    fault_throw = 0.0

    thickness_factor = np.ones(nx, dtype=np.float64)
    style_q1 = parent_parameters.get("style_parameter_1_q", rng_geo.random())
    style_q2 = parent_parameters.get("style_parameter_2_q", rng_geo.random())
    if style == "lenticular":
        lens_count = 2 + min(int(style_q1 * 3), 2)
        centers = rng_geo.uniform(0, x[-1], size=lens_count)
        width_center = 250.0 + 550.0 * style_q2
        widths = np.clip(rng_geo.normal(width_center, 80.0, size=centers.size), 250.0, 800.0)
        thickness_factor = 0.35 + sum(
            rng_geo.uniform(0.35, 0.9) * np.exp(-0.5 * ((x - c) / w) ** 2)
            for c, w in zip(centers, widths)
        )
        thickness_factor = np.clip(thickness_factor, 0.25, 1.45)
    elif style == "pinchout":
        pinch_center = (0.15 + 0.70 * style_q1) * x[-1]
        side = rng_geo.choice([-1.0, 1.0])
        transition_width = 120.0 + 180.0 * style_q2
        transition = 1.0 / (1.0 + np.exp(side * (x - pinch_center) / transition_width))
        thickness_factor = 0.12 + 1.25 * transition
    elif style == "faulted":
        fault_fraction_range = config.get("structural_priors", {}).get(
            "fault_x_model_fraction_range", [0.25, 0.75]
        )
        fault_x = (
            float(fault_fraction_range[0])
            + (float(fault_fraction_range[1]) - float(fault_fraction_range[0]))
            * style_q1
        ) * x[-1]
        fault_throw = (8.0 + 27.0 * style_q2) * rng_geo.choice([-1.0, 1.0])
        fault_displacement = (x > fault_x).astype(np.float64) * fault_throw
        stratigraphic_shift = stratigraphic_shift + fault_displacement

    top = sf + hydrate_top_mbsf + stratigraphic_shift[None, :]

    local_factor = np.clip(
        thickness_factor * (1.0 + rng_geo.uniform(0.05, 0.20) * _smooth_1d(rng_geo, nx, 8.0 + 16.0 * lateral_q)),
        0.08,
        1.55,
    )
    h_bottom = top + thickness["hydrate"] * local_factor[None, :]
    m_bottom = h_bottom + thickness["mixed"] * np.clip(0.75 + 0.35 * local_factor, 0.3, 1.4)[None, :]
    g_bottom = m_bottom + thickness["free_gas"] * np.clip(0.70 + 0.40 * local_factor, 0.25, 1.45)[None, :]

    zone_h = (zz >= top) & (zz < h_bottom)
    zone_m = (zz >= h_bottom) & (zz < m_bottom)
    zone_g = (zz >= m_bottom) & (zz < g_bottom)
    water = mbsf < 0.0

    background_noise = _normalized_smooth_noise(rng_prop, (nz, nx), (rng_prop.uniform(2, 6), rng_prop.uniform(12, 35)))
    phi = 0.43 - rng_prop.uniform(0.00016, 0.00028) * np.maximum(mbsf, 0.0)
    phi += rng_prop.uniform(0.008, 0.025) * background_noise
    # Keep the shallow background inside the log-supported unconsolidated
    # sediment envelope. Zone-specific values below can still reach 0.28.
    phi = np.clip(phi, 0.31, 0.55)
    for mask, key in ((zone_h, "hydrate"), (zone_m, "mixed"), (zone_g, "free_gas")):
        local = porosity_targets[key] + rng_prop.uniform(0.012, 0.035) * _normalized_smooth_noise(
            rng_prop, (nz, nx), (rng_prop.uniform(1.5, 4), rng_prop.uniform(10, 30))
        )
        local += porosity_targets[key] - float(local[mask].mean())
        lower = max(0.28, porosity_targets[key] - 0.05)
        upper = min(0.55, porosity_targets[key] + 0.05)
        phi = np.where(mask, np.clip(local, lower, upper), phi)
    phi[water] = 1.0

    texture_h = np.exp(rng_prop.uniform(0.30, 0.70) * _normalized_smooth_noise(rng_prop, (nz, nx), (2.5, 22)))
    texture_mh = np.exp(rng_prop.uniform(0.35, 0.85) * _normalized_smooth_noise(rng_prop, (nz, nx), (1.8, 15)))
    texture_mg = np.exp(rng_prop.uniform(0.45, 1.0) * _normalized_smooth_noise(rng_prop, (nz, nx), (1.5, 12)))
    texture_g = np.exp(rng_prop.uniform(0.45, 1.0) * _normalized_smooth_noise(rng_prop, (nz, nx), (1.5, 18)))

    # Some realizations contain true lateral gaps, as reported for discontinuous
    # BSRs and fault/gas-chimney controlled accumulations in Shenhu.
    if style in {"lenticular", "pinchout", "faulted"}:
        lateral_gate = gaussian_filter(rng_prop.normal(size=nx), sigma=rng_prop.uniform(8, 25))
        threshold = np.quantile(lateral_gate, rng_prop.uniform(0.15, 0.38))
        gate = (lateral_gate > threshold)[None, :]
        texture_mg *= 0.08 + 0.92 * gate
        texture_g *= 0.05 + 0.95 * gate

    Sh_h = _scale_masked_mean(texture_h, zone_h, saturation_targets["Sh_hydrate"], cap=0.76)
    Sh_m = _scale_masked_mean(texture_mh, zone_m, saturation_targets["Sh_mixed"], cap=0.55)
    Sg_m = _scale_masked_mean(texture_mg, zone_m, saturation_targets["Sg_mixed"], cap=0.36)
    Sg_g = _scale_masked_mean(texture_g, zone_g, saturation_targets["Sg_free_gas"], cap=0.30)
    Sh = Sh_h + Sh_m
    Sg = Sg_m + Sg_g
    total = Sh + Sg
    over = total > 0.92
    Sh[over] *= 0.92 / total[over]
    Sg[over] *= 0.92 / total[over]

    arrays = {
        "x_m": x.astype(np.float32),
        "z_m": z.astype(np.float32),
        "seafloor_m": seafloor.astype(np.float32),
        "stratigraphic_shift_m": stratigraphic_shift.astype(np.float32),
        "fault_displacement_m": fault_displacement.astype(np.float32),
        "hydrate_top_m": top[0].astype(np.float32),
        "mixed_top_m": h_bottom[0].astype(np.float32),
        "free_gas_top_m": m_bottom[0].astype(np.float32),
        "free_gas_bottom_m": g_bottom[0].astype(np.float32),
        "phi": phi.astype(np.float32),
        "Sh": Sh.astype(np.float32),
        "Sg": Sg.astype(np.float32),
        "hydrate_mask": (Sh > 0.01).astype(np.uint8),
        "gas_mask": (Sg > 0.005).astype(np.uint8),
        "zone_hydrate": zone_h.astype(np.uint8),
        "zone_mixed": zone_m.astype(np.uint8),
        "zone_free_gas": zone_g.astype(np.uint8),
    }
    metadata = {
        "sample_index": sample_index,
        "seed": seed,
        "geometry_seed": seed if geometry_seed is None else geometry_seed,
        "property_seed": seed if property_seed is None else property_seed,
        "family": family_name,
        "structural_style": style,
        "fault_x_m": None if not np.isfinite(fault_x) else float(fault_x),
        "fault_throw_m": float(fault_throw),
        "fault_x_model_fraction_range": (
            list(config.get("structural_priors", {}).get(
                "fault_x_model_fraction_range", [0.25, 0.75]
            ))
        ),
        "water_depth_m": water_depth,
        "hydrate_top_mbsf": hydrate_top_mbsf,
        "thickness_m": thickness,
        "porosity_targets": porosity_targets,
        "saturation_targets": saturation_targets,
        "hydrate_response_scale": hydrate_response_scale,
        "gas_response_scale": gas_response_scale,
    }
    return arrays, metadata
