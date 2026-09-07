"""Build a machine-readable array schema with observed release-wide ranges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]

UNITS = {
    "seafloor_m": "m below sea surface",
    "bsr_m": "m below sea surface",
    "phi_full": "dimensionless fraction",
    "Sh_full": "fraction of total pore volume",
    "Sg_full": "fraction of total pore volume",
    "Vp_full": "m/s",
    "rho_full": "kg/m^3",
    "phi_view": "dimensionless fraction",
    "Sh_view": "fraction of total pore volume",
    "Sg_view": "fraction of total pore volume",
    "Vp_view": "m/s",
    "rho_view": "kg/m^3",
    "AI_view": "kg/(m^2 s)",
    "valid_mask": "0/1",
    "loss_mask": "0/1",
    "raw_gathers": "solver pressure amplitude",
    "processed_gathers": "relative amplitude",
    "runtime_s": "s",
    "peak_cuda_memory_gb": "GiB",
    "input_rtm_conditioned_unscaled": "dJ/dVp-type adjoint-gradient amplitude",
    "input_rtm_display_only": "dimensionless display scale",
    "input_display_scale": "adjoint-gradient amplitude",
    "target_Sh": "fraction of total pore volume",
    "target_Sg": "fraction of total pore volume",
    "auxiliary_Vp": "m/s",
    "migration_Vp": "m/s",
    "raw_representative_gather": "solver pressure amplitude",
    "processed_representative_gather": "relative amplitude",
    "shot_x_m": "m",
    "receiver_x_m": "m",
    "diagnostic_true_model_rtm_conditioned_unscaled": "dJ/dVp-type adjoint-gradient amplitude",
    "diagnostic_true_model_rtm_display": "dimensionless display scale",
    "diagnostic_display_scale": "adjoint-gradient amplitude",
    "metadata_json": None,
}

ROLES = {
    "input_rtm_conditioned_unscaled": "primary input for Tasks 1 and 2",
    "input_rtm_display_only": "display only; not quantitative input",
    "target_Sh": "Task 2 target",
    "target_Sg": "Task 2 target",
    "auxiliary_Vp": "Task 1 Vp target / auxiliary truth",
    "migration_Vp": "Task 1 second input and physical baseline",
    "AI_view": "Task 1 AI target",
    "loss_mask": "loss/evaluation mask; never a network input",
    "valid_mask": "support metadata; never a network input",
    "diagnostic_true_model_rtm_conditioned_unscaled": "technical validation only",
    "diagnostic_true_model_rtm_display": "display-only technical validation",
}


def update(stats: dict, role: str, name: str, value: np.ndarray) -> None:
    key = f"{role}.{name}"
    item = stats.setdefault(
        key,
        {
            "file_role": role,
            "field": name,
            "shapes": set(),
            "dtypes": set(),
            "present_in_samples": 0,
            "observed_global_min": None,
            "observed_global_max": None,
        },
    )
    item["shapes"].add(tuple(int(x) for x in value.shape))
    dtype_label = "unicode" if np.issubdtype(value.dtype, np.str_) else str(value.dtype)
    item["dtypes"].add(dtype_label)
    item["present_in_samples"] += 1
    if np.issubdtype(value.dtype, np.number):
        local_min = float(np.min(value))
        local_max = float(np.max(value))
        item["observed_global_min"] = local_min if item["observed_global_min"] is None else min(item["observed_global_min"], local_min)
        item["observed_global_max"] = local_max if item["observed_global_max"] is None else max(item["observed_global_max"], local_max)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, default=PROJECT / "outputs" / "NGH-Seis-v1.0-data")
    parser.add_argument("--release", type=Path, default=PROJECT / "outputs" / "NGH-Seis-v1.0-metadata")
    parser.add_argument("--output", type=Path, default=PROJECT / "docs" / "NGH_SEIS_ARRAY_SCHEMA.json")
    parser.add_argument("--skip-observations", action="store_true", help="Skip the 30.5-GiB observation archives for a fast development scan.")
    args = parser.parse_args()
    source_manifest_path = args.release / "accepted_manifest.json"
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    stats: dict[str, dict] = {}
    for index, record in enumerate(manifest, start=1):
        sources = [("geology", "geology"), ("training_pair", "pair")]
        if not args.skip_observations:
            sources.append(("observations", "observations"))
        for role, key in sources:
            source_key = f"source_{key}"
            path = PROJECT / record[source_key] if source_key in record else args.production / record[key]
            with np.load(path, allow_pickle=False) as archive:
                for name in archive.files:
                    update(stats, role, name, archive[name])
        if index % 50 == 0 or index == len(manifest):
            print(f"scanned {index}/{len(manifest)} accepted samples", flush=True)

    fields = []
    for key in sorted(stats):
        item = stats[key]
        item["shapes"] = [list(shape) for shape in sorted(item["shapes"])]
        item["dtypes"] = sorted(item["dtypes"])
        item["unit"] = UNITS.get(item["field"])
        item["benchmark_role"] = ROLES.get(item["field"], "released physical/diagnostic field; see DATA_DICTIONARY.md")
        item["range_definition"] = "observed minimum and maximum across all 1,000 accepted archives"
        fields.append(item)
    payload = {
        "schema_version": "1.0",
        "dataset_name": "NGH-Seis",
        "dataset_version": "1.0",
        "accepted_samples_scanned": len(manifest),
        "observation_archives_scanned": not args.skip_observations,
        "array_order": "vertical/time first; lateral/receiver second",
        "saturation_definition": "fraction of total pore volume",
        "important": [
            "Observed extrema describe the frozen synthetic release; they are not recommended normalization constants.",
            "Use training-split calibration JSON files for quantitative normalization.",
            "loss_mask and valid_mask are metadata/masks, not network input channels.",
            "input_rtm_display_only is display-only; use input_rtm_conditioned_unscaled for quantitative benchmarks.",
        ],
        "fields": fields,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
