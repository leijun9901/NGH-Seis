"""Generate deterministic NGH-Seis v1.0 geological models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.field_aligned_view import extract_seafloor_relative_view  # noqa: E402
from src.geology.blake_ew0008 import generate_blake_ew0008  # noqa: E402


CONFIG = ROOT / "configs/ngh_seis_geology.json"
OUTPUT = ROOT / "outputs/NGH-Seis-v1.0-generation/geology_candidates"


def _rowwise_lateral_correlation(field: np.ndarray, rows: slice, lag: int) -> float:
    array = np.asarray(field[rows, :95], dtype=np.float64)
    array -= array.mean(axis=1, keepdims=True)
    denominator = float(np.sum(array * array))
    if denominator <= 1.0e-20 or lag >= array.shape[1]:
        return 0.0
    return float(np.sum(array[:, :-lag] * array[:, lag:]) / denominator)


def _decorrelation_distance_m(
    field: np.ndarray,
    rows: slice,
    *,
    dx_m: float = 37.5,
    threshold: float = 0.20,
) -> float:
    for lag in range(1, 25):
        if _rowwise_lateral_correlation(field, rows, lag) <= threshold:
            return lag * dx_m
    return 24 * dx_m


def _extract(field: np.ndarray, seafloor: np.ndarray, config: dict):
    grid = config["grid"]
    crop = config["network_crop"]
    return extract_seafloor_relative_view(
        field,
        seafloor,
        source_dx_m=float(grid["dx_m"]),
        source_dz_m=float(grid["dz_m"]),
        x_start_m=float(crop["x_start_m"]),
        trace_count=int(crop["physical_trace_count"]),
        padded_trace_count=int(crop["padded_trace_count"]),
        output_dx_m=float(crop["output_dx_m"]),
        output_dz_m=float(crop["output_dz_m"]),
        output_nz=int(crop["nz"]),
        water_samples=int(crop["samples_above_seafloor"]),
        interpolation_order=1,
        antialias_lateral=True,
    )


def _render(path: Path, full: dict, view: dict, metadata: dict, config: dict) -> None:
    grid = config["grid"]
    full_extent = [
        0.0,
        (int(grid["nx_physical"]) - 1) * float(grid["dx_m"]) / 1000.0,
        (int(grid["nz"]) - 1) * float(grid["dz_m"]) / 1000.0,
        0.0,
    ]
    crop = config["network_crop"]
    view_extent = [
        0.0,
        float(crop["physical_trace_count"] - 1) * float(crop["output_dx_m"]) / 1000.0,
        (int(crop["samples_below_seafloor"]) - 1) * float(crop["output_dz_m"]) / 1000.0,
        -int(crop["samples_above_seafloor"]) * float(crop["output_dz_m"]) / 1000.0,
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5), constrained_layout=True)
    panels = [
        (full["Vp_full"], "Full Vp model", "turbo", 1450, 2800, full_extent),
        (view["Vp"], "EW0008-aligned Vp", "turbo", 1450, 2800, view_extent),
        (view["rho"], "EW0008-aligned density", "viridis", 1000, 2100, view_extent),
        (view["phi"], "Porosity", "viridis", 0.35, 0.70, view_extent),
        (view["Sh"], "Hydrate saturation Sh", "magma", 0.0, 0.22, view_extent),
        (view["Sg"], "Free-gas saturation Sg", "cividis", 0.0, 0.08, view_extent),
    ]
    for ax, (array, title, cmap, vmin, vmax, extent) in zip(axes.flat, panels):
        shown = array[:, :95] if array.shape[1] == 96 else array
        image = ax.imshow(
            shown, cmap=cmap, vmin=vmin, vmax=vmax,
            aspect="auto", extent=extent, interpolation="nearest",
        )
        ax.set(title=title, xlabel="x (km)", ylabel="Depth relative to sea surface/seafloor (km)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(
        f"NGH-Seis geology {metadata['sample_index']:02d} | "
        f"{metadata['style']} | geology/rock physics only",
        fontsize=14,
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main(
    config_path: Path = CONFIG,
    output: Path = OUTPUT,
    *,
    start_index: int = 1,
    count: int = 5,
) -> None:
    """Generate deterministic geology candidates for NGH-Seis v1.0."""
    if start_index < 1 or count < 1:
        raise ValueError("start_index and count must be positive")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    audit_samples = []
    for sample_index in range(start_index, start_index + count):
        seed = 20_260_818 + sample_index * 104_729
        full, metadata = generate_blake_ew0008(
            config_path,
            sample_index,
            seed,
            include_proxy_image=False,
        )
        view = {}
        valid = None
        view_metadata = None
        for output_name, source_name in (
            ("Vp", "Vp_full"),
            ("rho", "rho_full"),
            ("phi", "phi_full"),
            ("Sh", "Sh_full"),
            ("Sg", "Sg_full"),
            ("AI", "AI_full"),
            ("lithology", "lithology_proxy_full"),
        ):
            sampled, current_valid, current_metadata = _extract(
                full[source_name], full["seafloor_m"], config
            )
            view[output_name] = sampled
            valid = current_valid if valid is None else valid
            view_metadata = current_metadata if view_metadata is None else view_metadata
        loss_mask = valid.copy()
        loss_mask[: int(config["network_crop"]["samples_above_seafloor"]), :] = 0
        saturation_sum_max = float(np.max(full["Sh_full"] + full["Sg_full"]))
        sediment = full["phi_full"] < 0.99
        minimum_sediment_vp = float(np.min(full["Vp_full"][sediment]))
        gas_bearing = full["Sg_full"] > 0.0
        gas_bearing_vp_p01 = float(
            np.quantile(full["Vp_full"][gas_bearing], 0.01)
        )
        effective_max_hz = float(
            config["forward_sampling_candidate"]["effective_max_frequency_hz"]
        )
        points_per_wavelength = minimum_sediment_vp / (
            effective_max_hz * float(config["grid"]["dx_m"])
        )
        sample_audit = {
            "sample_index": sample_index,
            "style": metadata["style"],
            "water_depth_mean_m": metadata["water_depth_mean_m"],
            "bsr_depth_mbsf_mean": metadata["bsr_depth_mbsf_mean"],
            "hydrate_mean_in_zone": metadata["hydrate_mean_in_zone"],
            "hydrate_max": metadata["hydrate_max"],
            "gas_mean_in_zone": metadata["gas_mean_in_zone"],
            "gas_max": metadata["gas_max"],
            "hydrate_q95_positive": metadata["hydrate_q95_positive"],
            "gas_q95_positive": metadata["gas_q95_positive"],
            "hydrate_fraction_within_one_percent_of_max": metadata[
                "hydrate_fraction_within_one_percent_of_max"
            ],
            "gas_fraction_within_one_percent_of_max": metadata[
                "gas_fraction_within_one_percent_of_max"
            ],
            "gas_zone_thickness_m": metadata["gas_zone_thickness_m"],
            "coexistence_enabled": metadata["coexistence_enabled"],
            "coexistence_thickness_m": metadata["coexistence_thickness_m"],
            "coexistence_pixel_fraction": metadata["coexistence_pixel_fraction"],
            "saturation_sum_max": saturation_sum_max,
            "minimum_sediment_vp_m_s": minimum_sediment_vp,
            "gas_bearing_vp_p01_m_s": gas_bearing_vp_p01,
            "minimum_points_per_effective_wavelength": points_per_wavelength,
            "full_shape": list(full["Vp_full"].shape),
            "view_shape": list(view["Vp"].shape),
            "finite": bool(all(np.all(np.isfinite(value)) for value in view.values())),
            "padding_zero": bool(all(np.all(value[:, 95] == 0.0) for value in view.values())),
            "loss_mask_fraction": float(loss_mask.mean()),
            "seafloor_p95_absolute_slope": float(
                metadata["seafloor_p95_absolute_slope"]
            ),
            "Sh_lag150m_correlation": _rowwise_lateral_correlation(
                view["Sh"], slice(124, 244), 4
            ),
            "Sg_lag150m_correlation": _rowwise_lateral_correlation(
                view["Sg"], slice(214, 334), 4
            ),
            "Sh_decorrelation_distance_m": _decorrelation_distance_m(
                view["Sh"], slice(124, 244)
            ),
            "Sg_decorrelation_distance_m": _decorrelation_distance_m(
                view["Sg"], slice(214, 334)
            ),
        }
        metadata.update(
            {
                "role": "geology and rock-physics model; no seismic proxy released as network input",
                "field_aligned_view": view_metadata,
                "well_validation_status": "regional ODP prior only; no strict EW0008 through-well trace",
                "sample_audit": sample_audit,
            }
        )
        destination = output / f"sample_{sample_index:04d}_blake_geology.npz"
        np.savez_compressed(
            destination,
            x_m=full["x_m"],
            z_m=full["z_m"],
            seafloor_m=full["seafloor_m"],
            bsr_m=full["bsr_m"],
            phi_full=full["phi_full"],
            Sh_full=full["Sh_full"],
            Sg_full=full["Sg_full"],
            Vp_full=full["Vp_full"],
            rho_full=full["rho_full"],
            Vp_view=view["Vp"],
            rho_view=view["rho"],
            phi_view=view["phi"],
            Sh_view=view["Sh"],
            Sg_view=view["Sg"],
            AI_view=view["AI"],
            lithology_proxy_full=full["lithology_proxy_full"],
            lithology_view=view["lithology"],
            valid_mask=valid,
            loss_mask=loss_mask,
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
        (output / f"sample_{sample_index:04d}_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _render(
            output / f"sample_{sample_index:04d}_geology.png",
            full,
            view,
            metadata,
            config,
        )
        audit_samples.append(sample_audit)
        print(f"generated {destination.name}", flush=True)

    priors = config["geological_priors"]
    gates = {
        "all_configured_styles_represented_when_full_cycle": (
            set(config["scenario_styles"]).issubset({item["style"] for item in audit_samples})
            if count >= len(config["scenario_styles"])
            else True
        ),
        "all_full_shapes_correct": all(item["full_shape"] == [1001, 1229] for item in audit_samples),
        "all_views_512x96": all(item["view_shape"] == [512, 96] for item in audit_samples),
        "all_finite": all(item["finite"] for item in audit_samples),
        "all_padding_zero": all(item["padding_zero"] for item in audit_samples),
        "all_saturation_sums_le_one": all(item["saturation_sum_max"] <= 1.0 for item in audit_samples),
        "all_sampling_candidates_ge_6_points": all(
            item["minimum_points_per_effective_wavelength"]
            >= config["forward_sampling_candidate"]["minimum_points_per_effective_wavelength"]
            for item in audit_samples
        ),
        "all_bsr_within_prior": all(
            priors["bsr_depth_mbsf"][0] <= item["bsr_depth_mbsf_mean"] <= priors["bsr_depth_mbsf"][1]
            for item in audit_samples
        ),
    }
    continuity_acceptance = config.get("continuity_acceptance")
    if continuity_acceptance is not None:
        continuous_styles = {
            "diffuse_continuous_bsr",
            "localized_enriched_lenses",
            "sediment_wave_crosscut",
        }
        continuous = [
            item for item in audit_samples if item["style"] in continuous_styles
        ]
        gates.update(
            {
                "all_seafloor_slopes_within_field_gate": all(
                    item["seafloor_p95_absolute_slope"]
                    <= continuity_acceptance["seafloor_p95_absolute_slope_max"]
                    for item in audit_samples
                ),
                "continuous_styles_Sh_lag150m_gate": all(
                    item["Sh_lag150m_correlation"]
                    >= continuity_acceptance[
                        "continuous_style_Sh_lag150m_correlation_min"
                    ]
                    for item in continuous
                ),
                "continuous_styles_Sg_lag150m_gate": all(
                    item["Sg_lag150m_correlation"]
                    >= continuity_acceptance[
                        "continuous_style_Sg_lag150m_correlation_min"
                    ]
                    for item in continuous
                ),
                "continuous_styles_Sh_decorrelation_distance_gate": all(
                    item["Sh_decorrelation_distance_m"]
                    >= continuity_acceptance[
                        "continuous_style_Sh_decorrelation_distance_m_min"
                    ]
                    for item in continuous
                ),
                "continuous_styles_Sg_decorrelation_distance_gate": all(
                    item["Sg_decorrelation_distance_m"]
                    >= continuity_acceptance[
                        "continuous_style_Sg_decorrelation_distance_m_min"
                    ]
                    for item in continuous
                ),
            }
        )
    saturation_acceptance = config.get("saturation_acceptance")
    if saturation_acceptance is not None:
        gas_mean_min, gas_mean_max = saturation_acceptance["gas_zone_mean_range"]
        gas_thickness_min, gas_thickness_max = saturation_acceptance[
            "gas_zone_thickness_range_m"
        ]
        gas_vp_min, gas_vp_max = saturation_acceptance[
            "gas_bearing_vp_p01_range_m_s"
        ]
        gates.update(
            {
                "all_hydrate_distributions_without_cap_plateau": all(
                    item["hydrate_fraction_within_one_percent_of_max"]
                    <= saturation_acceptance[
                        "hydrate_positive_near_max_fraction_max"
                    ]
                    for item in audit_samples
                ),
                "all_gas_distributions_without_cap_plateau": all(
                    item["gas_fraction_within_one_percent_of_max"]
                    <= saturation_acceptance["gas_positive_near_max_fraction_max"]
                    for item in audit_samples
                ),
                "all_hydrate_maxima_within_ODP_prior": all(
                    item["hydrate_max"]
                    <= saturation_acceptance["hydrate_global_max"]
                    for item in audit_samples
                ),
                "all_gas_maxima_within_ODP_prior": all(
                    item["gas_max"] <= saturation_acceptance["gas_global_max"]
                    for item in audit_samples
                ),
                "all_gas_zone_means_within_ODP_prior": all(
                    gas_mean_min <= item["gas_mean_in_zone"] <= gas_mean_max
                    for item in audit_samples
                ),
                "all_gas_zone_thicknesses_within_ODP_prior": all(
                    gas_thickness_min
                    <= item["gas_zone_thickness_m"]
                    <= gas_thickness_max
                    for item in audit_samples
                ),
                "all_gas_bearing_vp_p01_within_regional_envelope": all(
                    gas_vp_min
                    <= item["gas_bearing_vp_p01_m_s"]
                    <= gas_vp_max
                    for item in audit_samples
                ),
            }
        )
    report = {
        "config": str(config_path.resolve().relative_to(ROOT)),
        "start_index": start_index,
        "count": count,
        "samples": audit_samples,
        "acceptance": gates,
        "all_gates_pass": all(gates.values()),
        "explicit_exclusions": [
            "no proxy PSDM is admitted as a formal input",
            "no full-wave or RTM result has yet been generated",
            "ODP wells are regional priors, not strict through-well validation",
        ],
    }
    (output / "geology_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--count", type=int, default=5)
    arguments = parser.parse_args()
    main(
        arguments.config,
        arguments.output,
        start_index=arguments.start_index,
        count=arguments.count,
    )
