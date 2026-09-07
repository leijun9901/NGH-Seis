"""Run resumable 25-shot forward modelling and RTM for NGH-Seis v1.0.

The independent smooth-velocity RTM is the candidate network input.  A
true-model migration is saved only as a diagnostic that separates migration
velocity error from acquisition/adjoint artifacts; it is never a benchmark
input.  Sh/Sg are not accessed when building the smooth migration model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import deepwave
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.run_blake_rtm_shot_density_gate import (  # noqa: E402
    _bsr_contrast,
    _extract_view,
    _generate_observations,
    _image_metrics,
    build_blake_migration_model,
    build_geometry,
)
from src.datasets.rtm_scaling import (  # noqa: E402
    condition_rtm_unscaled,
    display_normalize_rtm,
)


ACQUISITION_CONFIG = PROJECT / "configs/marine_streamer.json"
GEOLOGY_CONFIG = PROJECT / "configs/ngh_seis_geology.json"
GEOLOGY_FOLDER = PROJECT / "outputs/NGH-Seis-v1.0-generation/geology_candidates"
OUTPUT = PROJECT / "outputs/NGH-Seis-v1.0-generation/rtm_candidates"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_geology(sample_id: int, geology_folder: Path) -> dict[str, np.ndarray]:
    path = geology_folder / f"sample_{sample_id:04d}_blake_geology.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files if key != "metadata_json"}


def _select_25_shots(geometry49: dict, config: dict | None = None) -> dict:
    """Select the frozen production subset from the 49-shot reference grid."""
    if config is None:
        indices = np.arange(0, 49, 2, dtype=np.int64)
    else:
        indices = np.asarray(
            config["release_acquisition"]["parent_49_zero_based_indices"],
            dtype=np.int64,
        )
    if indices.size != 25 or not np.array_equal(indices, np.arange(0, 49, 2)):
        raise ValueError("NGH-Seis v1.0 requires the 25 even-indexed reference shots")
    return {
        "source_locations": geometry49["source_locations"][indices],
        "receiver_locations": geometry49["receiver_locations"][indices],
        "shot_x_m": geometry49["shot_x_m"][indices],
        "receiver_x_m": geometry49["receiver_x_m"][indices],
        "offset_m": geometry49["offset_m"],
        "original_49_indices": indices.astype(np.int32),
    }


def _batch_records(nshot: int, batch_size: int):
    for batch_id, start in enumerate(range(0, nshot, batch_size)):
        yield batch_id, np.arange(start, min(start + batch_size, nshot), dtype=np.int64)


def _migrate_batches(
    processed_gathers: np.ndarray,
    migration_vp: np.ndarray,
    migration_rho: np.ndarray,
    config: dict,
    geometry: dict,
    device: torch.device,
    folder: Path,
    *,
    model_role: str,
    batch_size: int,
    stop_after_batches: int | None,
) -> dict:
    recording = config["recording_candidate"]
    nt, dt = int(recording["nt"]), float(recording["dt_s"])
    f0 = float(recording["source_peak_frequency_hz"])
    t0 = float(recording["source_delay_s"])
    base_source = deepwave.wavelets.ricker(f0, nt, dt, t0).to(device).reshape(1, 1, nt)
    rho = torch.as_tensor(migration_rho, device=device)
    records = list(_batch_records(processed_gathers.shape[0], batch_size))
    folder.mkdir(parents=True, exist_ok=True)
    new_count = 0
    runtimes = []
    for batch_id, indices in records:
        stem = f"batch_{batch_id:02d}_shots_{int(indices[0]):02d}_{int(indices[-1]):02d}"
        gradient_path = folder / f"{stem}_vp_gradient.npy"
        runtime_path = folder / f"{stem}_runtime.json"
        if gradient_path.exists() and runtime_path.exists():
            continue
        if stop_after_batches is not None and new_count >= stop_after_batches:
            break
        vp = torch.as_tensor(migration_vp, device=device).requires_grad_(True)
        source = base_source.expand(indices.size, -1, -1).contiguous()
        started = time.perf_counter()
        predicted = deepwave.acoustic(
            vp,
            rho,
            float(config["propagation_grid"]["dx_m"]),
            dt,
            source_amplitudes_p=source,
            source_locations_p=geometry["source_locations"][indices],
            receiver_locations_p=geometry["receiver_locations"][indices],
            accuracy=int(recording["fd_accuracy_order"]),
            pml_width=int(recording["pml_width_cells"]),
            pml_freq=f0,
            max_vel=max(3000.0, float(migration_vp.max()) + 50.0),
            model_gradient_sampling_interval=4,
            storage_mode="cpu",
        )[-3]
        observed = torch.as_tensor(
            processed_gathers[indices].transpose(0, 2, 1), device=device
        )
        objective = torch.sum(predicted * observed)
        gradient = torch.autograd.grad(objective, vp)[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        np.save(gradient_path, gradient.detach().cpu().numpy().astype(np.float32))
        runtime_path.write_text(
            json.dumps(
                {
                    "runtime_s": elapsed,
                    "objective": float(objective.detach().cpu()),
                    "local_shot_indices": list(map(int, indices)),
                    "original_49_shot_indices": list(
                        map(int, geometry["original_49_indices"][indices])
                    ),
                    "migration_model_role": model_role,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        new_count += 1
        runtimes.append(elapsed)
        print(
            f"{model_role} adjoint {batch_id + 1}/{len(records)} | "
            f"shots {int(indices[0])}-{int(indices[-1])} | {elapsed:.1f} s",
            flush=True,
        )
        del vp, source, predicted, observed, objective, gradient
        torch.cuda.empty_cache()
    complete = 0
    for batch_id, indices in records:
        stem = f"batch_{batch_id:02d}_shots_{int(indices[0]):02d}_{int(indices[-1]):02d}"
        if (folder / f"{stem}_vp_gradient.npy").exists() and (
            folder / f"{stem}_runtime.json"
        ).exists():
            complete += 1
    return {
        "complete": complete == len(records),
        "completed_batches": complete,
        "total_batches": len(records),
        "new_runtime_s": float(sum(runtimes)),
    }


def _sum_gradients(folder: Path, nshot: int, batch_size: int) -> tuple[np.ndarray, float]:
    total = None
    runtime = 0.0
    for batch_id, indices in _batch_records(nshot, batch_size):
        stem = f"batch_{batch_id:02d}_shots_{int(indices[0]):02d}_{int(indices[-1]):02d}"
        gradient = np.load(folder / f"{stem}_vp_gradient.npy", mmap_mode="r")
        total = np.asarray(gradient, dtype=np.float64) if total is None else total + gradient
        runtime += float(_read_json(folder / f"{stem}_runtime.json")["runtime_s"])
    return (total / float(nshot)).astype(np.float32), runtime


def _render(
    path: Path,
    raw_gather: np.ndarray,
    processed_gather: np.ndarray,
    smooth_rtm: np.ndarray,
    true_rtm: np.ndarray,
    sh: np.ndarray,
    sg: np.ndarray,
    dt: float,
    sample_id: int,
    dataset_name: str = "NGH-Seis",
) -> None:
    raw_scale = max(float(np.percentile(np.abs(raw_gather), 99.5)), 1.0e-12)
    processed_scale = max(
        float(np.percentile(np.abs(processed_gather), 99.5)), 1.0e-12
    )
    depth_extent = [0.0, 3.525, 1.341, -0.192]
    time_extent = [1, raw_gather.shape[1], raw_gather.shape[0] * dt, 0.0]
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.4), constrained_layout=True)
    panels = [
        (raw_gather / raw_scale, "Raw marine streamer shot gather", "gray", -1.0, 1.0, time_extent),
        (processed_gather / processed_scale, "Fixed processed shot gather", "gray", -1.0, 1.0, time_extent),
        (smooth_rtm, "NETWORK INPUT: smooth-Vp 25-shot RTM", "gray", -1.0, 1.0, depth_extent),
        (true_rtm, "DIAGNOSTIC ONLY: true-model RTM", "gray", -1.0, 1.0, depth_extent),
        (sh, "TARGET 1: hydrate saturation Sh", "magma", 0.0, 0.22, depth_extent),
        (sg, "TARGET 2: free-gas saturation Sg", "cividis", 0.0, 0.065, depth_extent),
    ]
    for ax, (array, title, cmap, vmin, vmax, extent) in zip(axes.flat, panels):
        shown = array[:, :95] if array.shape == (512, 96) else array
        artist = ax.imshow(
            shown, cmap=cmap, vmin=vmin, vmax=vmax,
            extent=extent, aspect="auto", interpolation="nearest",
        )
        ax.set_title(title)
        if array.shape == (512, 96):
            ax.set(xlabel="x (km)", ylabel="depth relative to seafloor (km)")
        else:
            ax.set(xlabel="receiver channel", ylabel="time (s)")
        fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(
        f"{dataset_name} sample {sample_id:04d} | forward, RTM input and Sh/Sg targets",
        fontsize=14,
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _rtm_quality_gate(
    metrics: dict[str, float], correlation: float | None, rules: dict
) -> dict[str, bool]:
    """Reject gross directional artifacts without using Sh/Sg or true Vp."""
    checks = {
        "horizontal_to_vertical_gradient_energy": bool(
            metrics["horizontal_to_vertical_gradient_energy"]
            <= rules["smooth_horizontal_to_vertical_gradient_energy_max"]
        ),
        "near_vertical_fk_energy_fraction": bool(
            metrics["near_vertical_fk_energy_fraction"]
            <= rules["smooth_near_vertical_fk_energy_fraction_max"]
        ),
        "adjacent_trace_correlation_median": bool(
            metrics["adjacent_trace_correlation_median"]
            >= rules["smooth_adjacent_trace_correlation_median_min"]
        ),
        "normalized_lateral_roughness": bool(
            metrics["normalized_lateral_roughness"]
            <= rules["smooth_normalized_lateral_roughness_max"]
        ),
    }
    if "smooth_structure_aware_adjacent_trace_correlation_median_min" in rules:
        checks["structure_aware_adjacent_trace_correlation_median"] = bool(
            metrics["structure_aware_adjacent_trace_correlation_median"]
            >= rules["smooth_structure_aware_adjacent_trace_correlation_median_min"]
        )
        checks["structure_aware_normalized_lateral_roughness"] = bool(
            metrics["structure_aware_normalized_lateral_roughness"]
            <= rules["smooth_structure_aware_normalized_lateral_roughness_max"]
        )
    # `correlation` is retained in the call signature for report compatibility,
    # but deliberately excluded from admission.  It requires a true-Vp
    # migration, is computed only for a diagnostic subset, and would otherwise
    # apply a label-aware gate to only every Nth candidate.
    return checks


def run_sample(
    sample_id: int,
    *,
    batch_size: int,
    stop_after_batches: int | None,
    geology_config_path: Path = GEOLOGY_CONFIG,
    geology_folder: Path = GEOLOGY_FOLDER,
    output: Path = OUTPUT,
    include_true_model_diagnostic: bool = True,
    write_visualization: bool = True,
) -> Path | None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output.mkdir(parents=True, exist_ok=True)
    config = _read_json(ACQUISITION_CONFIG)
    geology_config = _read_json(geology_config_path)
    geology = _load_geology(sample_id, geology_folder)
    device = torch.device("cuda")
    geometry = _select_25_shots(build_geometry(config, device), config)

    observations_path = output / f"sample_{sample_id:04d}_observations25.npz"
    raw, processed, forward_runtime, peak_memory = _generate_observations(
        geology, config, geometry, device, observations_path
    )
    smooth_vp, smooth_rho, smooth_metadata = build_blake_migration_model(
        geology["seafloor_m"],
        nz=int(config["propagation_grid"]["nz"]),
        nx=int(config["propagation_grid"]["nx"]),
        dz_m=float(config["propagation_grid"]["dz_m"]),
        seed=84_000_019 + sample_id * 65_537,
    )
    smooth_folder = output / f"sample_{sample_id:04d}_smooth_gradients_b{batch_size}"
    smooth_progress = _migrate_batches(
        processed, smooth_vp, smooth_rho, config, geometry, device, smooth_folder,
        model_role="independent_smooth_model",
        batch_size=batch_size,
        stop_after_batches=stop_after_batches,
    )
    if not smooth_progress["complete"]:
        return None
    smooth_full, smooth_runtime = _sum_gradients(smooth_folder, 25, batch_size)
    true_full = None
    true_runtime = None
    if include_true_model_diagnostic:
        true_folder = output / f"sample_{sample_id:04d}_true_gradients_b{batch_size}"
        true_progress = _migrate_batches(
            processed, geology["Vp_full"], geology["rho_full"], config, geometry,
            device, true_folder, model_role="true_model_diagnostic_only",
            batch_size=batch_size, stop_after_batches=stop_after_batches,
        )
        if not true_progress["complete"]:
            return None
        true_full, true_runtime = _sum_gradients(true_folder, 25, batch_size)
    smooth_view, support, view_metadata = _extract_view(
        smooth_full, geology["seafloor_m"], geology_config
    )
    migration_vp_view, _, _ = _extract_view(
        smooth_vp, geology["seafloor_m"], geology_config
    )
    loss_mask = support.copy()
    loss_mask[: int(geology_config["network_crop"]["samples_above_seafloor"]), :] = 0
    mask = loss_mask.astype(bool)
    smooth_conditioned = condition_rtm_unscaled(smooth_view, loss_mask)
    smooth_display, smooth_display_scale = display_normalize_rtm(
        smooth_conditioned, loss_mask
    )
    true_conditioned = None
    true_display = None
    true_display_scale = None
    correlation = None
    if true_full is not None:
        true_view, _, _ = _extract_view(true_full, geology["seafloor_m"], geology_config)
        true_conditioned = condition_rtm_unscaled(true_view, loss_mask)
        true_display, true_display_scale = display_normalize_rtm(true_conditioned, loss_mask)
        correlation = float(np.corrcoef(smooth_conditioned[mask], true_conditioned[mask])[0, 1])
    smooth_metrics = _image_metrics(smooth_display, mask)
    true_metrics = _image_metrics(true_display, mask) if true_display is not None else None
    quality_rules = geology_config["rtm_quality_acceptance"]
    quality_checks = _rtm_quality_gate(smooth_metrics, correlation, quality_rules)
    report = {
        "schema_version": "1.0",
        "sample_id": sample_id,
        "geology_version": geology_config["version"],
        "role": (
            f"{geology_config.get('public_release_name', geology_config['version'])} "
            "NGH-Seis v1.0 25-shot marine-streamer forward and RTM workflow"
        ),
        "device": torch.cuda.get_device_name(0),
        "shot_count": 25,
        "original_49_shot_indices": list(map(int, geometry["original_49_indices"])),
        "forward_runtime_s": forward_runtime,
        "forward_peak_cuda_tensor_memory_gb": peak_memory,
        "smooth_migration_runtime_s": smooth_runtime,
        "true_model_diagnostic_runtime_s": true_runtime,
        "input_shape": list(smooth_conditioned.shape),
        "target_shape": [2, *geology["Sh_view"].shape],
        "smooth_migration_model": smooth_metadata,
        "view_metadata": view_metadata,
        "smooth_rtm_metrics": smooth_metrics,
        "true_model_rtm_metrics": true_metrics,
        "smooth_vs_true_conditioned_correlation": correlation,
        "rtm_quality_acceptance": quality_rules,
        "rtm_quality_checks": quality_checks,
        "label_aware_diagnostic_not_a_gate": (
            {
                "smooth_vs_true_conditioned_correlation": correlation,
                "smooth_vs_true_conditioned_correlation_reference_min": quality_rules.get(
                    "smooth_true_conditioned_correlation_min"
                ),
                "smooth_vs_true_conditioned_correlation_meets_reference": bool(
                    correlation >= quality_rules["smooth_true_conditioned_correlation_min"]
                ),
                "smooth_bsr_rms_contrast": _bsr_contrast(smooth_display, geology["bsr_m"], geology["seafloor_m"], mask),
                "true_bsr_rms_contrast": _bsr_contrast(true_display, geology["bsr_m"], geology["seafloor_m"], mask),
            } if true_display is not None else None
        ),
        "admission": {
            "policy": "ngh_seis_rtm_quality_gate",
            "finite_input": bool(np.isfinite(smooth_conditioned).all()),
            "nonzero_input": bool(np.max(np.abs(smooth_conditioned)) > 0.0),
            "shape_512x96": smooth_conditioned.shape == (512, 96),
            "training_input_not_per_sample_normalized": True,
            "true_model_migration_excluded_from_training": True,
            "true_model_diagnostic_computed": include_true_model_diagnostic,
            "rtm_quality_gate_pass": bool(all(quality_checks.values())),
        },
        "status": "admitted" if all(quality_checks.values()) else "rejected_by_rtm_quality_gate",
    }
    payload = dict(
        input_rtm_conditioned_unscaled=smooth_conditioned,
        input_rtm_display_only=smooth_display,
        input_display_scale=np.asarray(smooth_display_scale),
        target_Sh=geology["Sh_view"],
        target_Sg=geology["Sg_view"],
        auxiliary_Vp=geology["Vp_view"],
        migration_Vp=migration_vp_view,
        valid_mask=support,
        loss_mask=loss_mask,
        raw_representative_gather=raw[12],
        processed_representative_gather=processed[12],
        shot_x_m=geometry["shot_x_m"],
        receiver_x_m=geometry["receiver_x_m"],
    )
    if true_conditioned is not None:
        payload.update(
            diagnostic_true_model_rtm_conditioned_unscaled=true_conditioned,
            diagnostic_true_model_rtm_display=true_display,
            diagnostic_display_scale=np.asarray(true_display_scale),
        )
    np.savez_compressed(output / f"sample_{sample_id:04d}_training_pair25.npz", **payload)
    report_path = output / f"sample_{sample_id:04d}_rtm_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if write_visualization and true_display is not None:
        _render(
            output / f"sample_{sample_id:04d}_training_pair25.png",
            raw[12], processed[12], smooth_display, true_display,
            geology["Sh_view"], geology["Sg_view"],
            float(config["recording_candidate"]["dt_s"]), sample_id,
            geology_config.get("public_release_name", geology_config["version"]),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-ids", type=int, nargs="+", default=[1, 3, 4])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--geology-config", type=Path, default=GEOLOGY_CONFIG)
    parser.add_argument("--geology-folder", type=Path, default=GEOLOGY_FOLDER)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--stop-after-batches", type=int, default=None,
        help="Resume aid: run at most this many new batches per migration model",
    )
    args = parser.parse_args()
    for sample_id in args.sample_ids:
        result = run_sample(
            sample_id,
            batch_size=args.batch_size,
            stop_after_batches=args.stop_after_batches,
            geology_config_path=args.geology_config,
            geology_folder=args.geology_folder,
            output=args.output,
        )
        print(result if result is not None else f"sample {sample_id} incomplete; rerun to resume")


if __name__ == "__main__":
    main()
