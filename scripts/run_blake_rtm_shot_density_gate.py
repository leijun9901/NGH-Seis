"""Compare 25-shot and 49-shot Blake Ridge RTM on one fixed model.

The 25-shot set is the even-index subset of the 49-shot survey. Observed
gathers are generated once and cached. Each migration-shot gradient is saved
individually, so the run can resume without repeating completed adjoints.
No Sh/Sg field is accessed while building the migration model or deciding the
label-free shot-density gate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import deepwave
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, shift


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.datasets.field_aligned_view import extract_seafloor_relative_view  # noqa: E402
from src.datasets.rtm_scaling import condition_rtm_unscaled, display_normalize_rtm  # noqa: E402
from src.forward.benchmark_workflow import (  # noqa: E402
    measure_rtm_directional_artifacts,
    preprocess_observed_gather,
)


CONFIG_PATH = PROJECT / "configs" / "marine_streamer.json"
GEOLOGY_CONFIG_PATH = PROJECT / "configs" / "ngh_seis_geology.json"
GEOLOGY_FOLDER = PROJECT / "outputs" / "NGH-Seis-v1.0-generation" / "geology_candidates"
OUTPUT = PROJECT / "outputs" / "NGH-Seis-v1.0-quality-gate"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _exact_index(value_m: float, spacing_m: float, name: str) -> int:
    index = int(round(value_m / spacing_m))
    if not np.isclose(index * spacing_m, value_m, atol=1.0e-9):
        raise ValueError(f"{name}={value_m} m is not exact on a {spacing_m} m grid")
    return index


def build_geometry(config: dict, device: torch.device) -> dict:
    grid = config["propagation_grid"]
    acquisition = config["marine_acquisition_candidate"]
    dx, dz = float(grid["dx_m"]), float(grid["dz_m"])
    nshot = int(acquisition["shot_count"])
    shot_x_m = float(acquisition["shot_start_m"]) + np.arange(nshot) * float(
        acquisition["shot_spacing_m"]
    )
    offsets_m = float(acquisition["near_offset_m"]) + np.arange(
        int(acquisition["receiver_count"])
    ) * float(acquisition["receiver_spacing_m"])
    receiver_x_m = shot_x_m[:, None] - offsets_m[None, :]

    source_x = np.asarray([_exact_index(x, dx, "shot_x") for x in shot_x_m])
    receiver_x = np.asarray(
        [[_exact_index(x, dx, "receiver_x") for x in row] for row in receiver_x_m],
        dtype=np.int64,
    )
    source_z = _exact_index(float(acquisition["source_depth_m"]), dz, "source_depth")
    receiver_z = _exact_index(float(acquisition["receiver_depth_m"]), dz, "receiver_depth")
    pml = int(config["recording_candidate"]["pml_width_cells"])
    safety = pml + 2
    if receiver_x.min() < safety or source_x.max() >= int(grid["nx"]) - safety:
        raise ValueError("survey geometry enters the PML safety margin")

    source_locations = torch.zeros((nshot, 1, 2), dtype=torch.long, device=device)
    receiver_locations = torch.zeros(
        (nshot, offsets_m.size, 2), dtype=torch.long, device=device
    )
    source_locations[..., 0] = source_z
    source_locations[:, 0, 1] = torch.as_tensor(source_x, device=device)
    receiver_locations[..., 0] = receiver_z
    receiver_locations[..., 1] = torch.as_tensor(receiver_x, device=device)
    return {
        "source_locations": source_locations,
        "receiver_locations": receiver_locations,
        "shot_x_m": shot_x_m.astype(np.float32),
        "receiver_x_m": receiver_x_m.astype(np.float32),
        "offset_m": offsets_m.astype(np.float32),
        "subset25_indices": np.arange(0, nshot, 2, dtype=np.int32),
    }


def build_blake_migration_model(
    seafloor_m: np.ndarray,
    *,
    nz: int,
    nx: int,
    dz_m: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a Blake regional migration model using bathymetry only.

    The 1600 to roughly 1800 m/s increase above the BSR follows the broad
    Site-997 log/VSP trend. The BSR and gas velocity decrease are deliberately
    excluded, keeping the model independent of the synthetic reservoir labels.
    """
    seafloor = np.asarray(seafloor_m, dtype=np.float32)
    if seafloor.shape != (nx,):
        raise ValueError("seafloor shape does not match migration grid")
    rng = np.random.default_rng(seed)
    z_m = np.arange(nz, dtype=np.float32)[:, None] * np.float32(dz_m)
    mbsf = np.maximum(z_m - seafloor[None, :], 0.0)
    top_velocity = float(rng.uniform(1580.0, 1620.0))
    gradient = float(rng.uniform(0.38, 0.48))
    curvature = float(rng.uniform(2.0e-5, 6.0e-5))
    sediment = top_velocity + gradient * mbsf + curvature * mbsf * mbsf

    # Only long-wavelength, independent estimation error is admitted.
    perturbation = gaussian_filter(
        rng.normal(size=(nz, nx)).astype(np.float32), sigma=(75.0, 110.0), mode="reflect"
    )
    perturbation -= float(perturbation.mean())
    perturbation /= max(float(perturbation.std()), 1.0e-6)
    lateral_fraction = float(rng.uniform(0.004, 0.010))
    sediment *= 1.0 + lateral_fraction * perturbation
    water = z_m < seafloor[None, :]
    vp = np.where(water, 1500.0, sediment).astype(np.float32)
    transition_m = 70.0
    vp = gaussian_filter(vp, sigma=(transition_m / (2.0 * dz_m), 2.0), mode="nearest")
    far_water = z_m <= seafloor[None, :] - transition_m
    vp[far_water] = 1500.0
    vp = np.clip(vp, 1480.0, 2350.0).astype(np.float32)

    # Constant density avoids copying a sharp true water-bottom density jump
    # into the migration operator. This choice is fixed across all samples.
    rho = np.full((nz, nx), 1800.0, dtype=np.float32)
    metadata = {
        "label_access": "picked synthetic bathymetry only; no Vp_true, rho_true, Sh, Sg, phi or BSR access",
        "regional_basis": "ODP Site 997 broad log/VSP trend: about 1600 m/s shallow and about 1800-2000 m/s near 450 mbsf",
        "top_velocity_m_s": top_velocity,
        "gradient_s_inv": gradient,
        "curvature_inv_m_s": curvature,
        "lateral_fraction": lateral_fraction,
        "water_bottom_transition_m": transition_m,
        "density_kg_m3": 1800.0,
        "reservoir_scale_anomalies_included": False,
        "seed": seed,
    }
    return vp, rho, metadata


def _extract_view(field: np.ndarray, seafloor_m: np.ndarray, geology_config: dict):
    grid = geology_config["grid"]
    crop = geology_config["network_crop"]
    return extract_seafloor_relative_view(
        field,
        seafloor_m,
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


def _adjacent_trace_correlation(image: np.ndarray, mask: np.ndarray) -> float:
    values = []
    for ix in range(image.shape[1] - 1):
        common = mask[:, ix] & mask[:, ix + 1]
        if np.count_nonzero(common) < 32:
            continue
        left, right = image[common, ix], image[common, ix + 1]
        if np.std(left) > 1.0e-12 and np.std(right) > 1.0e-12:
            values.append(float(np.corrcoef(left, right)[0, 1]))
    return float(np.median(values)) if values else 0.0


def _structure_aware_trace_metrics(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    maximum_shift_samples: float = 2.0,
    shift_step_samples: float = 0.25,
) -> tuple[float, float]:
    """Measure coherence after a small local vertical slope correction.

    The search is label-free and uses only adjacent RTM traces.  It prevents
    coherent anticlines and dipping beds from being penalized as lateral
    roughness while retaining sensitivity to columns and incoherent artifacts.
    """
    correlations: list[float] = []
    roughness: list[float] = []
    image_energy = max(float(np.mean(image[mask] ** 2)), 1.0e-20)
    trial_shifts = np.arange(
        -maximum_shift_samples,
        maximum_shift_samples + 0.5 * shift_step_samples,
        shift_step_samples,
    )
    for ix in range(image.shape[1] - 1):
        best: tuple[float, float] | None = None
        left = image[:, ix]
        left_mask = mask[:, ix]
        for trial in trial_shifts:
            right = shift(
                image[:, ix + 1], trial, order=1, mode="nearest", prefilter=False
            )
            right_mask = shift(
                mask[:, ix + 1].astype(np.float32),
                trial,
                order=0,
                mode="nearest",
                prefilter=False,
            ) > 0.5
            common = left_mask & right_mask
            if np.count_nonzero(common) < 32:
                continue
            a, b = left[common], right[common]
            if np.std(a) <= 1.0e-12 or np.std(b) <= 1.0e-12:
                continue
            corr = float(np.corrcoef(a, b)[0, 1])
            residual = float(np.mean((a - b) ** 2) / image_energy)
            if best is None or corr > best[0]:
                best = (corr, residual)
        if best is not None:
            correlations.append(best[0])
            roughness.append(best[1])
    return (
        float(np.median(correlations)) if correlations else 0.0,
        float(np.median(roughness)) if roughness else float("inf"),
    )


def _image_metrics(image: np.ndarray, mask: np.ndarray) -> dict:
    qc = measure_rtm_directional_artifacts(image, mask, dx=37.5, dz=3.0)
    common = mask[:, 1:] & mask[:, :-1]
    lateral_energy = float(np.mean(np.diff(image, axis=1)[common] ** 2))
    image_energy = float(np.mean(image[mask] ** 2))
    structure_corr, structure_roughness = _structure_aware_trace_metrics(image, mask)
    return {
        **qc,
        "adjacent_trace_correlation_median": _adjacent_trace_correlation(image, mask),
        "normalized_lateral_roughness": lateral_energy / max(image_energy, 1.0e-20),
        "structure_aware_adjacent_trace_correlation_median": structure_corr,
        "structure_aware_normalized_lateral_roughness": structure_roughness,
    }


def _bsr_contrast(image: np.ndarray, bsr_m: np.ndarray, seafloor_m: np.ndarray, mask: np.ndarray) -> float:
    x_out = 1310.0 + np.arange(95) * 37.5
    source_x = np.arange(bsr_m.size) * 5.0
    relative = np.interp(x_out, source_x, bsr_m - seafloor_m)
    rows = 64 + np.rint(relative / 3.0).astype(int)
    band = np.zeros_like(mask, dtype=bool)
    for ix, row in enumerate(rows):
        band[max(0, row - 5):min(image.shape[0], row + 6), ix] = True
    band &= mask
    background = mask & ~band
    return float(
        np.sqrt(np.mean(image[band] ** 2))
        / max(float(np.sqrt(np.mean(image[background] ** 2))), 1.0e-20)
    )


def _render(
    path: Path,
    rtm25: np.ndarray,
    rtm49: np.ndarray,
    migration_vp: np.ndarray,
    sh: np.ndarray,
    sg: np.ndarray,
    mask: np.ndarray,
    report: dict,
) -> None:
    difference = np.where(mask, rtm49 - rtm25, 0.0)
    diff_scale = max(float(np.percentile(np.abs(difference[mask]), 99.5)), 1.0e-12)
    extent = [0.0, 3.525, 1.341, -0.192]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), constrained_layout=True)
    panels = [
        (rtm25, "25-shot RTM", "gray", -1.0, 1.0),
        (rtm49, "49-shot RTM", "gray", -1.0, 1.0),
        (difference / diff_scale, "49 minus 25 (display-scaled)", "seismic", -1.0, 1.0),
        (migration_vp, "Independent Blake migration Vp", "turbo", 1450.0, 2350.0),
        (sh, "Sh reference (not used by density gate)", "magma", 0.0, 0.22),
        (sg, "Sg reference (not used by density gate)", "cividis", 0.0, 0.08),
    ]
    for ax, (array, title, cmap, vmin, vmax) in zip(axes.flat, panels):
        shown = array[:, :95]
        artist = ax.imshow(
            shown, cmap=cmap, vmin=vmin, vmax=vmax,
            extent=extent, aspect="auto", interpolation="nearest",
        )
        ax.axhline(0.0, color="cyan", lw=0.6, alpha=0.7)
        ax.set(title=title, xlabel="x (km)", ylabel="depth relative to seafloor (km)")
        fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(
        f"Blake sample 0001 | 25/49-shot density gate: {report['status'].upper()} | "
        f"RTM corr={report['comparison']['rtm_correlation_25_vs_49']:.3f}",
        fontsize=14,
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _load_geology(sample_id: int) -> dict[str, np.ndarray]:
    path = GEOLOGY_FOLDER / f"sample_{sample_id:04d}_blake_geology.npz"
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files if name != "metadata_json"}


def _generate_observations(
    geology: dict[str, np.ndarray],
    config: dict,
    geometry: dict,
    device: torch.device,
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as payload:
            return (
                payload["raw_gathers"].copy(),
                payload["processed_gathers"].copy(),
                float(payload["runtime_s"]),
                float(payload["peak_cuda_memory_gb"]),
            )
    recording = config["recording_candidate"]
    nt, dt = int(recording["nt"]), float(recording["dt_s"])
    f0, t0 = float(recording["source_peak_frequency_hz"]), float(recording["source_delay_s"])
    source = deepwave.wavelets.ricker(f0, nt, dt, t0).to(device).reshape(1, 1, nt)
    vp = torch.as_tensor(geology["Vp_full"], device=device)
    rho = torch.as_tensor(geology["rho_full"], device=device)
    raw_gathers, processed_gathers = [], []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for ishot in range(geometry["source_locations"].shape[0]):
        with torch.no_grad():
            receiver = deepwave.acoustic(
                vp, rho, float(config["propagation_grid"]["dx_m"]), dt,
                source_amplitudes_p=source,
                source_locations_p=geometry["source_locations"][ishot:ishot + 1],
                receiver_locations_p=geometry["receiver_locations"][ishot:ishot + 1],
                accuracy=int(recording["fd_accuracy_order"]),
                pml_width=int(recording["pml_width_cells"]),
                pml_freq=f0,
                max_vel=max(3000.0, float(geology["Vp_full"].max()) + 50.0),
            )[-3]
        raw = receiver[0].detach().cpu().numpy().T.astype(np.float32)
        processed = preprocess_observed_gather(
            raw, geometry["offset_m"], dt=dt, t0=t0,
            low_hz=3.0, high_hz=28.0, gain_power=0.65,
            direct_mute_extra_s=0.08, coherent_noise_lowpass_hz=0.0,
        )
        raw_gathers.append(raw)
        processed_gathers.append(processed)
        print(
            f"forward {ishot + 1}/{geometry['source_locations'].shape[0]}",
            flush=True,
        )
    torch.cuda.synchronize()
    runtime_s = time.perf_counter() - started
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024.0**3)
    raw_stack = np.stack(raw_gathers)
    processed_stack = np.stack(processed_gathers)
    np.savez_compressed(
        cache_path,
        raw_gathers=raw_stack,
        processed_gathers=processed_stack,
        runtime_s=np.asarray(runtime_s),
        peak_cuda_memory_gb=np.asarray(peak_memory_gb),
    )
    return raw_stack, processed_stack, runtime_s, peak_memory_gb


def _migrate_missing_shots(
    processed_gathers: np.ndarray,
    migration_vp: np.ndarray,
    migration_rho: np.ndarray,
    config: dict,
    geometry: dict,
    device: torch.device,
    gradient_folder: Path,
    stop_after: int | None,
) -> dict:
    recording = config["recording_candidate"]
    nt, dt = int(recording["nt"]), float(recording["dt_s"])
    f0, t0 = float(recording["source_peak_frequency_hz"]), float(recording["source_delay_s"])
    source = deepwave.wavelets.ricker(f0, nt, dt, t0).to(device).reshape(1, 1, nt)
    rho = torch.as_tensor(migration_rho, device=device)
    gradient_folder.mkdir(parents=True, exist_ok=True)
    completed = 0
    runtimes = []
    for ishot in range(processed_gathers.shape[0]):
        destination = gradient_folder / f"shot_{ishot:02d}_vp_gradient.npy"
        runtime_path = gradient_folder / f"shot_{ishot:02d}_runtime.json"
        if destination.exists() and runtime_path.exists():
            completed += 1
            continue
        if stop_after is not None and completed >= stop_after:
            break
        vp = torch.as_tensor(migration_vp, device=device).requires_grad_(True)
        started = time.perf_counter()
        predicted = deepwave.acoustic(
            vp, rho, float(config["propagation_grid"]["dx_m"]), dt,
            source_amplitudes_p=source,
            source_locations_p=geometry["source_locations"][ishot:ishot + 1],
            receiver_locations_p=geometry["receiver_locations"][ishot:ishot + 1],
            accuracy=int(recording["fd_accuracy_order"]),
            pml_width=int(recording["pml_width_cells"]),
            pml_freq=f0,
            max_vel=3000.0,
            model_gradient_sampling_interval=4,
            storage_mode="cpu",
        )[-3]
        observed = torch.as_tensor(processed_gathers[ishot].T[None, :, :], device=device)
        objective = torch.sum(predicted * observed)
        gradient = torch.autograd.grad(objective, vp)[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        np.save(destination, gradient.detach().cpu().numpy().astype(np.float32))
        runtime_path.write_text(
            json.dumps({"runtime_s": elapsed, "objective": float(objective.detach().cpu())}, indent=2),
            encoding="utf-8",
        )
        runtimes.append(elapsed)
        completed += 1
        print(f"adjoint {ishot + 1}/49 | {elapsed:.1f} s", flush=True)
        del vp, predicted, observed, objective, gradient
        torch.cuda.empty_cache()
    return {"completed_shots": completed, "new_runtime_s": float(sum(runtimes))}


def _migration_batches(batch_size: int) -> list[tuple[str, int, np.ndarray]]:
    """Return parity-preserving batches so 25 and 49 shots remain separable."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batches: list[tuple[str, int, np.ndarray]] = []
    for parity_name, indices in (
        ("even25", np.arange(0, 49, 2, dtype=np.int32)),
        ("odd24", np.arange(1, 49, 2, dtype=np.int32)),
    ):
        for ibatch, start in enumerate(range(0, indices.size, batch_size)):
            batches.append((parity_name, ibatch, indices[start:start + batch_size]))
    return batches


def _migrate_missing_batches(
    processed_gathers: np.ndarray,
    migration_vp: np.ndarray,
    migration_rho: np.ndarray,
    config: dict,
    geometry: dict,
    device: torch.device,
    gradient_folder: Path,
    *,
    batch_size: int,
    stop_after_batches: int | None,
) -> dict:
    """Save summed gradients for small parity-preserving shot batches."""
    recording = config["recording_candidate"]
    nt, dt = int(recording["nt"]), float(recording["dt_s"])
    f0, t0 = float(recording["source_peak_frequency_hz"]), float(recording["source_delay_s"])
    base_source = deepwave.wavelets.ricker(f0, nt, dt, t0).to(device).reshape(1, 1, nt)
    rho = torch.as_tensor(migration_rho, device=device)
    gradient_folder.mkdir(parents=True, exist_ok=True)
    batches = _migration_batches(batch_size)
    completed = 0
    new_batches = 0
    runtimes = []
    for parity_name, ibatch, indices in batches:
        index_tag = "_".join(f"{int(i):02d}" for i in indices)
        stem = f"{parity_name}_batch_{ibatch:02d}_shots_{index_tag}"
        destination = gradient_folder / f"{stem}_vp_gradient.npy"
        runtime_path = gradient_folder / f"{stem}_runtime.json"
        if destination.exists() and runtime_path.exists():
            completed += 1
            continue
        if stop_after_batches is not None and new_batches >= stop_after_batches:
            break
        vp = torch.as_tensor(migration_vp, device=device).requires_grad_(True)
        source = base_source.expand(indices.size, -1, -1).contiguous()
        started = time.perf_counter()
        predicted = deepwave.acoustic(
            vp, rho, float(config["propagation_grid"]["dx_m"]), dt,
            source_amplitudes_p=source,
            source_locations_p=geometry["source_locations"][indices],
            receiver_locations_p=geometry["receiver_locations"][indices],
            accuracy=int(recording["fd_accuracy_order"]),
            pml_width=int(recording["pml_width_cells"]),
            pml_freq=f0,
            max_vel=3000.0,
            model_gradient_sampling_interval=4,
            storage_mode="cpu",
        )[-3]
        observed_np = processed_gathers[indices].transpose(0, 2, 1)
        observed = torch.as_tensor(observed_np, device=device)
        objective = torch.sum(predicted * observed)
        gradient = torch.autograd.grad(objective, vp)[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        np.save(destination, gradient.detach().cpu().numpy().astype(np.float32))
        runtime_path.write_text(
            json.dumps(
                {
                    "runtime_s": elapsed,
                    "objective": float(objective.detach().cpu()),
                    "shot_indices_zero_based": list(map(int, indices)),
                    "model_gradient_sampling_interval": 4,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        runtimes.append(elapsed)
        completed += 1
        new_batches += 1
        print(
            f"adjoint batch {completed}/{len(batches)} | {parity_name} "
            f"shots {list(map(int, indices))} | {elapsed:.1f} s",
            flush=True,
        )
        del vp, source, predicted, observed, objective, gradient
        torch.cuda.empty_cache()
    return {
        "completed_batches": completed,
        "total_batches": len(batches),
        "new_batches": new_batches,
        "new_runtime_s": float(sum(runtimes)),
        "batch_size": batch_size,
    }


def _assemble(
    sample_id: int,
    geology: dict[str, np.ndarray],
    geology_config: dict,
    migration_vp: np.ndarray,
    migration_metadata: dict,
    geometry: dict,
    gradient_folder: Path,
    forward_runtime_s: float,
    forward_peak_memory_gb: float,
    batch_size: int,
) -> Path | None:
    batches = _migration_batches(batch_size)
    records = []
    for parity_name, ibatch, indices in batches:
        index_tag = "_".join(f"{int(i):02d}" for i in indices)
        stem = f"{parity_name}_batch_{ibatch:02d}_shots_{index_tag}"
        records.append(
            (
                parity_name,
                indices,
                gradient_folder / f"{stem}_vp_gradient.npy",
                gradient_folder / f"{stem}_runtime.json",
            )
        )
    if not all(path.exists() and runtime.exists() for _, _, path, runtime in records):
        completed = sum(path.exists() and runtime.exists() for _, _, path, runtime in records)
        print(f"RTM assembly deferred: {completed}/{len(records)} gradient batches exist")
        return None
    indices25 = set(map(int, geometry["subset25_indices"]))
    raw49 = np.zeros_like(migration_vp, dtype=np.float64)
    raw25 = np.zeros_like(migration_vp, dtype=np.float64)
    migration_runtime_s = 0.0
    for parity_name, indices, path, runtime_path in records:
        gradient = np.load(path, mmap_mode="r")
        raw49 += gradient
        if parity_name == "even25":
            raw25 += gradient
        migration_runtime_s += float(_read_json(runtime_path)["runtime_s"])
    raw49 = (raw49 / 49.0).astype(np.float32)
    raw25 = (raw25 / 25.0).astype(np.float32)

    view25, physical_support, view_metadata = _extract_view(raw25, geology["seafloor_m"], geology_config)
    view49, _, _ = _extract_view(raw49, geology["seafloor_m"], geology_config)
    migration_view, _, _ = _extract_view(migration_vp, geology["seafloor_m"], geology_config)
    loss_mask = physical_support.copy()
    loss_mask[: int(geology_config["network_crop"]["samples_above_seafloor"]), :] = 0
    mask = loss_mask.astype(bool)
    conditioned25 = condition_rtm_unscaled(view25, loss_mask)
    conditioned49 = condition_rtm_unscaled(view49, loss_mask)
    display25, scale25 = display_normalize_rtm(conditioned25, loss_mask)
    display49, scale49 = display_normalize_rtm(conditioned49, loss_mask)
    metrics25 = _image_metrics(display25, mask)
    metrics49 = _image_metrics(display49, mask)
    common25 = conditioned25[mask].astype(np.float64)
    common49 = conditioned49[mask].astype(np.float64)
    correlation = float(np.corrcoef(common25, common49)[0, 1])
    comparison = {
        "rtm_correlation_25_vs_49": correlation,
        "near_vertical_fk_change_percent": 100.0 * (
            metrics49["near_vertical_fk_energy_fraction"]
            - metrics25["near_vertical_fk_energy_fraction"]
        ) / max(metrics25["near_vertical_fk_energy_fraction"], 1.0e-20),
        "lateral_roughness_change_percent": 100.0 * (
            metrics49["normalized_lateral_roughness"]
            - metrics25["normalized_lateral_roughness"]
        ) / max(metrics25["normalized_lateral_roughness"], 1.0e-20),
        "adjacent_trace_correlation_change": (
            metrics49["adjacent_trace_correlation_median"]
            - metrics25["adjacent_trace_correlation_median"]
        ),
    }
    no_material_worsening = bool(
        comparison["near_vertical_fk_change_percent"] <= 10.0
        and comparison["lateral_roughness_change_percent"] <= 10.0
        and comparison["adjacent_trace_correlation_change"] >= -0.02
    )
    stable_structure = correlation >= 0.95
    report = {
        "schema_version": "1.0",
        "status": "pass" if no_material_worsening and stable_structure else "fail",
        "decision_rule": "49 shots must correlate >=0.95 with 25 shots, must not worsen either artifact metric by >10%, and must not lower adjacent-trace correlation by >0.02",
        "sample_id": sample_id,
        "shot25_indices_zero_based": sorted(indices25),
        "shot49_indices_zero_based": list(range(49)),
        "forward_runtime_s": forward_runtime_s,
        "forward_peak_cuda_tensor_memory_gb": forward_peak_memory_gb,
        "migration_runtime_s": migration_runtime_s,
        "adjoint_batch_size": batch_size,
        "adjoint_batch_count": len(records),
        "migration_model": migration_metadata,
        "view_metadata": view_metadata,
        "metrics25": metrics25,
        "metrics49": metrics49,
        "comparison": comparison,
        "label_free_gate": {
            "no_material_artifact_worsening": no_material_worsening,
            "stable_structure": stable_structure,
        },
        "label_aware_diagnostic_not_used_for_gate": {
            "bsr_rms_contrast25": _bsr_contrast(display25, geology["bsr_m"], geology["seafloor_m"], mask),
            "bsr_rms_contrast49": _bsr_contrast(display49, geology["bsr_m"], geology["seafloor_m"], mask),
        },
        "important_limitations": [
            "noiseless absorbing-surface observations isolate acquisition and migration behavior",
            "this gate does not admit free-surface data, Q attenuation, or formal production",
            "ODP wells provide regional priors and are not strict EW0008 through-well traces",
        ],
    }
    destination = OUTPUT / f"sample_{sample_id:04d}_25_vs_49_rtm.npz"
    np.savez_compressed(
        destination,
        rtm25_raw_view=view25,
        rtm49_raw_view=view49,
        rtm25_conditioned_unscaled=conditioned25,
        rtm49_conditioned_unscaled=conditioned49,
        rtm25_display=display25,
        rtm49_display=display49,
        display_scale25=np.asarray(scale25),
        display_scale49=np.asarray(scale49),
        migration_vp_view=migration_view,
        Sh_view=geology["Sh_view"],
        Sg_view=geology["Sg_view"],
        valid_mask=physical_support,
        loss_mask=loss_mask,
        shot_x_m=geometry["shot_x_m"],
        receiver_x_m=geometry["receiver_x_m"],
    )
    report_path = OUTPUT / f"sample_{sample_id:04d}_25_vs_49_gate.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _render(
        OUTPUT / f"sample_{sample_id:04d}_25_vs_49_rtm.png",
        display25, display49, migration_view, geology["Sh_view"], geology["Sg_view"],
        mask, report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report_path


def run(
    sample_id: int,
    migration_stop_after_batches: int | None,
    batch_size: int,
) -> Path | None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = _read_json(CONFIG_PATH)
    geology_config = _read_json(GEOLOGY_CONFIG_PATH)
    geology = _load_geology(sample_id)
    device = torch.device("cuda")
    geometry = build_geometry(config, device)
    migration_vp, migration_rho, migration_metadata = build_blake_migration_model(
        geology["seafloor_m"],
        nz=int(config["propagation_grid"]["nz"]),
        nx=int(config["propagation_grid"]["nx"]),
        dz_m=float(config["propagation_grid"]["dz_m"]),
        seed=81_000_011 + sample_id * 65_537,
    )
    raw, processed, forward_runtime, forward_peak_memory = _generate_observations(
        geology, config, geometry, device,
        OUTPUT / f"sample_{sample_id:04d}_observations49.npz",
    )
    del raw
    batch_gradient_folder = OUTPUT / f"sample_{sample_id:04d}_batch_gradients_b{batch_size}"
    progress = _migrate_missing_batches(
        processed, migration_vp, migration_rho, config, geometry, device,
        batch_gradient_folder,
        batch_size=batch_size,
        stop_after_batches=migration_stop_after_batches,
    )
    progress_path = OUTPUT / f"sample_{sample_id:04d}_progress.json"
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    return _assemble(
        sample_id, geology, geology_config, migration_vp, migration_metadata,
        geometry, batch_gradient_folder,
        forward_runtime, forward_peak_memory, batch_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", type=int, default=1)
    parser.add_argument(
        "--migration-stop-after-batches", type=int, default=None,
        help="Run at most this many new adjoint batches; omit for all remaining batches",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    result = run(args.sample_id, args.migration_stop_after_batches, args.batch_size)
    print(result if result is not None else "incomplete; rerun to resume")


if __name__ == "__main__":
    main()
