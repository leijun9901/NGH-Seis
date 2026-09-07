"""Compare the 9-shot screening and 41-shot marine acquisition candidates."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SCREENING = PROJECT / "outputs" / "marine_fullwave_science_stratified10_9shot"
CANDIDATE = PROJECT / "outputs" / "marine_fullwave_science_41shot_gate"


def _load(folder: Path, sample_id: int) -> tuple[dict[str, np.ndarray], dict]:
    path = folder / f"sample_{sample_id:04d}_marine_fullwave.npz"
    with np.load(path) as payload:
        arrays = {key: payload[key].copy() for key in payload.files if key != "metadata_json"}
        metadata = json.loads(str(payload["metadata_json"]))
    return arrays, metadata


def _adjacent_trace_correlation(image: np.ndarray, mask: np.ndarray) -> float:
    values = []
    for ix in range(image.shape[1] - 1):
        common = mask[:, ix] & mask[:, ix + 1]
        if np.count_nonzero(common) < 32:
            continue
        left = image[common, ix].astype(np.float64)
        right = image[common, ix + 1].astype(np.float64)
        if np.std(left) > 1e-8 and np.std(right) > 1e-8:
            values.append(float(np.corrcoef(left, right)[0, 1]))
    return float(np.median(values)) if values else 0.0


def _metrics(arrays: dict[str, np.ndarray], metadata: dict) -> dict[str, float]:
    image = arrays["rtm"].astype(np.float64)
    mask = arrays["valid_mask"].astype(bool)
    dx_energy = float(np.mean(np.diff(image, axis=1)[mask[:, 1:] & mask[:, :-1]] ** 2))
    image_energy = float(np.mean(image[mask] ** 2))
    qc = metadata["directional_artifact_qc"]
    return {
        "runtime_s": float(metadata["runtime_s"]),
        "near_vertical_fk_energy_fraction": float(qc["near_vertical_fk_energy_fraction"]),
        "horizontal_to_vertical_gradient_energy": float(
            qc["horizontal_to_vertical_gradient_energy"]
        ),
        "adjacent_trace_correlation_median": _adjacent_trace_correlation(image, mask),
        "normalized_lateral_roughness": dx_energy / max(image_energy, 1e-20),
    }


def _render(
    path: Path,
    sample_id: int,
    nine: dict[str, np.ndarray],
    forty_one: dict[str, np.ndarray],
    comparison: dict,
) -> None:
    mask = nine["valid_mask"].astype(bool) & forty_one["valid_mask"].astype(bool)
    difference = np.where(mask, forty_one["rtm"] - nine["rtm"], 0.0)
    scale = max(float(np.percentile(np.abs(difference[mask]), 99.5)), 1e-12)
    extent = [0.0, 2.048, 2.048, 0.0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panels = [
        (nine["rtm"], "9-shot screening RTM", "gray", -1, 1),
        (forty_one["rtm"], "41-shot candidate RTM", "gray", -1, 1),
        (difference / scale, "41-shot minus 9-shot", "seismic", -1, 1),
        (forty_one["migration_vp_mps"], "Independent migration Vp", "turbo", 1450, 2600),
        (forty_one["Sh"], "Sh reference (not used by gate)", "magma", 0, 0.76),
        (forty_one["Sg"], "Sg reference (not used by gate)", "cividis", 0, 0.36),
    ]
    for ax, (array, title, cmap, vmin, vmax) in zip(axes.flat, panels):
        shown = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect="auto")
        ax.set(title=title, xlabel="target x (km)", ylabel="depth below sea surface (km)")
        fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(
        f"sample_{sample_id:04d} acquisition-density gate | "
        f"vertical-artifact reduction {comparison['near_vertical_artifact_reduction_percent']:.1f}%",
        fontsize=14,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    output = CANDIDATE
    records = []
    for sample_id in (135, 300):
        nine, metadata9 = _load(SCREENING, sample_id)
        forty_one, metadata41 = _load(CANDIDATE, sample_id)
        metrics9 = _metrics(nine, metadata9)
        metrics41 = _metrics(forty_one, metadata41)
        common = nine["valid_mask"].astype(bool) & forty_one["valid_mask"].astype(bool)
        rtm9 = nine["rtm"][common].astype(np.float64)
        rtm41 = forty_one["rtm"][common].astype(np.float64)
        comparison = {
            "sample_id": f"sample_{sample_id:04d}",
            "well_family": metadata41["well_family"],
            "structural_style": metadata41["structural_style"],
            "shot_9": metrics9,
            "shot_41": metrics41,
            "near_vertical_artifact_reduction_percent": 100.0
            * (
                metrics9["near_vertical_fk_energy_fraction"]
                - metrics41["near_vertical_fk_energy_fraction"]
            )
            / max(metrics9["near_vertical_fk_energy_fraction"], 1e-20),
            "adjacent_trace_correlation_change": (
                metrics41["adjacent_trace_correlation_median"]
                - metrics9["adjacent_trace_correlation_median"]
            ),
            "lateral_roughness_reduction_percent": 100.0
            * (
                metrics9["normalized_lateral_roughness"]
                - metrics41["normalized_lateral_roughness"]
            )
            / max(metrics9["normalized_lateral_roughness"], 1e-20),
            "rtm_correlation_9_vs_41": float(np.corrcoef(rtm9, rtm41)[0, 1]),
            "normalized_difference_rms": float(
                np.sqrt(np.mean((rtm41 - rtm9) ** 2))
                / max(np.sqrt(np.mean(rtm41**2)), 1e-20)
            ),
            "runtime_ratio_41_to_9": metrics41["runtime_s"] / metrics9["runtime_s"],
        }
        comparison["quality_gate_pass"] = bool(
            comparison["near_vertical_artifact_reduction_percent"] > 0.0
            and comparison["adjacent_trace_correlation_change"] >= -0.02
            and comparison["lateral_roughness_reduction_percent"] > 0.0
        )
        records.append(comparison)
        _render(
            output / f"sample_{sample_id:04d}_9_vs_41_shots.png",
            sample_id,
            nine,
            forty_one,
            comparison,
        )

    report = {
        "status": "pass" if all(record["quality_gate_pass"] for record in records) else "fail",
        "decision_rule": (
            "Both samples must reduce near-vertical f-k energy and lateral roughness, "
            "without lowering median adjacent-trace correlation by more than 0.02."
        ),
        "recommended_formal_candidate": (
            "41 shots at 48 m spacing" if all(r["quality_gate_pass"] for r in records)
            else "not decided; test an intermediate shot count or revise migration"
        ),
        "records": records,
    }
    destination = output / "shot_density_gate_9_vs_41.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
