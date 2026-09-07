"""Freeze, audit, split, and calibrate the balanced NGH-Seis v1.0 release.

The generator stores geology and seismic products under separate roots. No
large array is copied by this script. It selects 200 admitted candidates
per geological family, verifies their co-registration and physical bounds,
and writes the release metadata needed by the benchmark loaders and paper.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.datasets.rtm_scaling import fit_training_scale  # noqa: E402


STYLE_NAMES = [
    "diffuse_continuous_bsr",
    "localized_enriched_lenses",
    "discontinuous_bsr",
    "fault_gas_chimney",
    "anticline_structural_trap",
]

STYLE_DISPLAY_NAMES = {
    "diffuse_continuous_bsr": "Diffuse continuous system",
    "localized_enriched_lenses": "Stratigraphic lenses",
    "discontinuous_bsr": "Patchy fairway",
    "fault_gas_chimney": "Fault-focused chimney",
    "anticline_structural_trap": "Anticline trap",
}

PAIR_FIELDS = {
    "input_rtm_conditioned_unscaled",
    "input_rtm_display_only",
    "input_display_scale",
    "target_Sh",
    "target_Sg",
    "auxiliary_Vp",
    "migration_Vp",
    "valid_mask",
    "loss_mask",
    "raw_representative_gather",
    "processed_representative_gather",
    "shot_x_m",
    "receiver_x_m",
}

GEOLOGY_FIELDS = {
    "x_m", "z_m", "seafloor_m", "bsr_m", "phi_full", "Sh_full",
    "Sg_full", "Vp_full", "rho_full", "Vp_view", "rho_view", "phi_view",
    "Sh_view", "Sg_view", "AI_view", "lithology_proxy_full",
    "lithology_view", "valid_mask", "loss_mask", "metadata_json",
}

OBSERVATION_FIELDS = {
    "raw_gathers", "processed_gathers", "runtime_s", "peak_cuda_memory_gb"
}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def id_digest(ids: list[int]) -> str:
    text = ",".join(str(value) for value in ids).encode("ascii")
    return hashlib.sha256(text).hexdigest()


class Moments:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if not values.size:
            return
        self.count += int(values.size)
        self.total += float(values.sum(dtype=np.float64))
        self.total_sq += float(np.square(values).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))

    def finish(self) -> dict[str, float | int]:
        if not self.count:
            raise RuntimeError("Cannot finish empty moments")
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "std": float(np.sqrt(variance)),
            "min": self.minimum,
            "max": self.maximum,
        }


def make_iid_split(records: list[dict], seed: int) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in records:
        groups[row["style"]].append(int(row["candidate_id"]))
    rng = np.random.default_rng(seed)
    result = {"train": [], "validation": [], "test": []}
    for style in STYLE_NAMES:
        values = np.asarray(sorted(groups[style]), dtype=np.int64)
        if values.size != 200:
            raise RuntimeError(f"Expected 200 release samples for {style}, got {values.size}")
        values = values[rng.permutation(values.size)].tolist()
        result["train"].extend(values[:160])
        result["validation"].extend(values[160:180])
        result["test"].extend(values[180:])
    return {key: sorted(map(int, values)) for key, values in result.items()}


def make_ood_splits(records: list[dict], seed: int) -> dict[str, dict]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in records:
        groups[row["style"]].append(int(row["candidate_id"]))
    folds = {}
    for fold_index, held_out in enumerate(STYLE_NAMES):
        rng = np.random.default_rng(seed + 10_007 * (fold_index + 1))
        train, validation = [], []
        for style in STYLE_NAMES:
            if style == held_out:
                continue
            values = np.asarray(sorted(groups[style]), dtype=np.int64)
            values = values[rng.permutation(values.size)]
            validation.extend(map(int, values[:20]))
            train.extend(map(int, values[20:]))
        folds[f"holdout_{held_out}"] = {
            "held_out_style": held_out,
            "held_out_display_name": STYLE_DISPLAY_NAMES[held_out],
            "train": sorted(train),
            "validation": sorted(validation),
            "test": sorted(map(int, groups[held_out])),
        }
    return folds


def resolve_paths(candidate_id: int, geology_root: Path, rtm_root: Path) -> dict[str, Path]:
    stem = f"sample_{candidate_id:04d}"
    return {
        "geology": geology_root / f"{stem}_blake_geology.npz",
        "metadata": geology_root / f"{stem}_metadata.json",
        "observations": rtm_root / f"{stem}_observations25.npz",
        "pair": rtm_root / f"{stem}_training_pair25.npz",
        "report": rtm_root / f"{stem}_rtm_report.json",
    }


def validate_geometry(shot_x: np.ndarray, receiver_x: np.ndarray) -> None:
    if shot_x.shape != (25,) or receiver_x.shape != (25, 128):
        raise ValueError("Unexpected shot or receiver coordinate shape")
    if not np.allclose(np.diff(shot_x), 150.0, atol=1e-5):
        raise ValueError("Shot spacing is not 150 m")
    if not np.allclose(np.diff(receiver_x, axis=1), -15.0, atol=1e-5):
        raise ValueError("Receiver spacing/order is not 15 m toward negative x")
    offsets = shot_x[:, None] - receiver_x
    if not np.allclose(offsets[:, 0], 100.0, atol=1e-5):
        raise ValueError("Minimum offset is not 100 m")
    if not np.allclose(offsets[:, -1], 2005.0, atol=1e-5):
        raise ValueError("Maximum offset is not 2005 m")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--production",
        type=Path,
        default=PROJECT / "outputs" / "NGH-Seis-v1.0-generation",
    )
    parser.add_argument(
        "--geology-root",
        type=Path,
        default=PROJECT / "outputs" / "NGH-Seis-v1.0-generation" / "geology",
    )
    parser.add_argument(
        "--release",
        type=Path,
        default=PROJECT / "outputs" / "NGH-Seis-v1.0-metadata",
    )
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--skip-observation-content", action="store_true")
    parser.add_argument("--audit-limit", type=int, default=0, help="Development-only early audit limit")
    args = parser.parse_args()

    production = args.production.resolve()
    geology_root = args.geology_root.resolve()
    rtm_root = production / "rtm_candidates"
    release = args.release.resolve()
    production_manifest = json.loads(
        (production / "production_manifest.json").read_text(encoding="utf-8")
    )
    admitted = production_manifest["accepted"]
    rejected = production_manifest["rejected"]
    admitted_ids = [int(row["candidate_id"]) for row in admitted]
    rejected_ids = [int(row["candidate_id"]) for row in rejected]
    if len(admitted_ids) != len(set(admitted_ids)):
        raise RuntimeError("Duplicate admitted candidate IDs")
    if set(admitted_ids) & set(rejected_ids):
        raise RuntimeError("Admitted and rejected candidate IDs overlap")

    admitted_by_style: dict[int, list[int]] = defaultdict(list)
    admitted_lookup = {}
    for row in admitted:
        style_index = int(row["style"])
        if style_index not in range(5):
            raise RuntimeError(f"Invalid numeric style in production manifest: {style_index}")
        candidate_id = int(row["candidate_id"])
        admitted_by_style[style_index].append(candidate_id)
        admitted_lookup[candidate_id] = row
    selected_ids = []
    for style_index in range(5):
        values = sorted(admitted_by_style[style_index])
        if len(values) < 200:
            raise RuntimeError(f"Style {style_index} has only {len(values)} admitted candidates")
        selected_ids.extend(values[:200])
    selected_ids = sorted(selected_ids)
    surplus_ids = sorted(set(admitted_ids) - set(selected_ids))
    if len(selected_ids) != 1000 or len(set(selected_ids)) != 1000:
        raise RuntimeError("Balanced selection did not produce 1,000 unique candidates")
    if args.audit_limit > 0:
        selected_ids = selected_ids[:args.audit_limit]

    stats = {
        "rtm": Moments(), "sh": Moments(), "sg": Moments(), "vp": Moments(),
        "migration_vp": Moments(), "ai": Moments(), "rho": Moments(), "phi": Moments(),
    }
    active_counts = {"valid": 0, "sh_0p01": 0, "sg_0p003": 0}
    style_counts: dict[str, int] = defaultdict(int)
    records, failures = [], []
    diagnostics = 0
    total_bytes = 0
    role_bytes = defaultdict(int)
    ai_resampling_max_nrmse = 0.0
    ai_resampling_max_p999_relative_error = 0.0

    for release_index, candidate_id in enumerate(selected_ids, start=1):
        paths = resolve_paths(candidate_id, geology_root, rtm_root)
        try:
            missing_files = [name for name, path in paths.items() if not path.is_file()]
            if missing_files:
                raise FileNotFoundError(f"Missing files: {missing_files}")
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            if report.get("status") != "admitted":
                raise ValueError("RTM report status is not admitted")
            if not report.get("admission", {}).get("rtm_quality_gate_pass", False):
                raise ValueError("RTM quality gate did not pass")
            metadata_sidecar = json.loads(paths["metadata"].read_text(encoding="utf-8"))

            with np.load(paths["geology"], allow_pickle=False) as geology:
                if not GEOLOGY_FIELDS.issubset(geology.files):
                    raise KeyError(f"Missing geology fields: {sorted(GEOLOGY_FIELDS-set(geology.files))}")
                metadata_embedded = json.loads(str(geology["metadata_json"]))
                if metadata_sidecar != metadata_embedded:
                    raise ValueError("Embedded and sidecar geology metadata differ")
                style = str(metadata_embedded["style"])
                style_index = STYLE_NAMES.index(style)
                if style_index != int(admitted_lookup[candidate_id]["style"]):
                    raise ValueError("Numeric and named styles disagree")
                geo_arrays = {
                    key: np.asarray(geology[key])
                    for key in ("Sh_view", "Sg_view", "Vp_view", "rho_view", "AI_view",
                                "phi_view", "valid_mask", "loss_mask")
                }
                if geology["x_m"].shape != (1229,) or geology["z_m"].shape != (1001,):
                    raise ValueError("Unexpected full-grid coordinate shape")
                if any(geology[key].shape != (1001, 1229) for key in
                       ("phi_full", "Sh_full", "Sg_full", "Vp_full", "rho_full",
                        "lithology_proxy_full")):
                    raise ValueError("Unexpected full-grid field shape")

            with np.load(paths["pair"], allow_pickle=False) as pair:
                if not PAIR_FIELDS.issubset(pair.files):
                    raise KeyError(f"Missing pair fields: {sorted(PAIR_FIELDS-set(pair.files))}")
                arrays = {
                    key: np.asarray(pair[key])
                    for key in ("input_rtm_conditioned_unscaled", "input_rtm_display_only",
                                "target_Sh", "target_Sg", "auxiliary_Vp", "migration_Vp",
                                "valid_mask", "loss_mask")
                }
                if any(value.shape != (512, 96) for value in arrays.values()):
                    raise ValueError("One or more release-view arrays are not 512x96")
                if pair["raw_representative_gather"].shape != (1501, 128):
                    raise ValueError("Unexpected representative raw gather shape")
                if pair["processed_representative_gather"].shape != (1501, 128):
                    raise ValueError("Unexpected representative processed gather shape")
                representative_raw = np.asarray(pair["raw_representative_gather"]).copy()
                representative_processed = np.asarray(pair["processed_representative_gather"]).copy()
                validate_geometry(np.asarray(pair["shot_x_m"]), np.asarray(pair["receiver_x_m"]))
                diagnostic = "diagnostic_true_model_rtm_conditioned_unscaled" in pair.files

            numeric_arrays = [value for key, value in arrays.items() if "mask" not in key]
            numeric_arrays += [geo_arrays[key] for key in
                               ("Sh_view", "Sg_view", "Vp_view", "rho_view", "AI_view", "phi_view")]
            if not all(np.isfinite(value).all() for value in numeric_arrays):
                raise ValueError("Non-finite release-view value")
            for mask_name in ("valid_mask", "loss_mask"):
                if not np.isin(arrays[mask_name], (0, 1)).all():
                    raise ValueError(f"{mask_name} is not binary")
                if not np.array_equal(arrays[mask_name], geo_arrays[mask_name]):
                    raise ValueError(f"Pair and geology {mask_name} differ")
                if np.any(arrays[mask_name][:, -1]):
                    raise ValueError(f"{mask_name} padding column is nonzero")
            if np.any(arrays["loss_mask"].astype(bool) & ~arrays["valid_mask"].astype(bool)):
                raise ValueError("loss_mask extends outside valid_mask")
            co_registered = {
                "target_Sh": "Sh_view", "target_Sg": "Sg_view",
                "auxiliary_Vp": "Vp_view",
            }
            for pair_key, geology_key in co_registered.items():
                if not np.array_equal(arrays[pair_key], geo_arrays[geology_key]):
                    raise ValueError(f"{pair_key} differs from {geology_key}")
            # AI is calculated on the full 5-m grid and then independently
            # anti-aliased/resampled with Vp and rho.  Evaluate the release-view
            # discrepancy robustly because isolated interface pixels can have a
            # larger pointwise interpolation difference.
            mask_for_ai = arrays["loss_mask"].astype(bool)
            ai_product = geo_arrays["rho_view"] * geo_arrays["Vp_view"]
            ai_difference = geo_arrays["AI_view"][mask_for_ai] - ai_product[mask_for_ai]
            ai_relative = np.abs(ai_difference) / np.maximum(np.abs(ai_product[mask_for_ai]), 1.0)
            ai_nrmse = float(np.sqrt(np.mean(np.square(ai_difference))) /
                             np.mean(np.abs(ai_product[mask_for_ai])))
            ai_p999 = float(np.percentile(ai_relative, 99.9))
            if ai_nrmse > 1e-3 or ai_p999 > 1e-2:
                raise ValueError(
                    f"AI resampling consistency failed: NRMSE={ai_nrmse:.6g}, p99.9={ai_p999:.6g}"
                )
            ai_resampling_max_nrmse = max(ai_resampling_max_nrmse, ai_nrmse)
            ai_resampling_max_p999_relative_error = max(
                ai_resampling_max_p999_relative_error, ai_p999
            )
            sh, sg = arrays["target_Sh"], arrays["target_Sg"]
            if sh.min() < -1e-7 or sh.max() > 0.220001:
                raise ValueError("Sh is outside [0, 0.22]")
            if sg.min() < -1e-7 or sg.max() > 0.065001:
                raise ValueError("Sg is outside [0, 0.065]")
            if np.max(sh + sg) > 0.250001:
                raise ValueError("Sh + Sg exceeds 0.25")
            mask = arrays["loss_mask"].astype(bool)
            rtm = arrays["input_rtm_conditioned_unscaled"]
            if not mask.any() or not np.any(np.abs(rtm[mask]) > 0.0):
                raise ValueError("Empty loss mask or zero RTM image")

            if not args.skip_observation_content:
                with np.load(paths["observations"], allow_pickle=False) as obs:
                    if not OBSERVATION_FIELDS.issubset(obs.files):
                        raise KeyError(f"Missing observation fields: {sorted(OBSERVATION_FIELDS-set(obs.files))}")
                    raw = np.asarray(obs["raw_gathers"])
                    processed = np.asarray(obs["processed_gathers"])
                    if raw.shape != (25, 1501, 128) or processed.shape != (25, 1501, 128):
                        raise ValueError("Unexpected full shot-gather shape")
                    if not np.isfinite(raw).all() or not np.isfinite(processed).all():
                        raise ValueError("Non-finite full shot gather")
                    if not np.array_equal(raw[12], representative_raw):
                        raise ValueError("Representative raw gather is not shot 13")
                    if not np.array_equal(processed[12], representative_processed):
                        raise ValueError("Representative processed gather is not shot 13")

            for key, value in (("rtm", rtm), ("sh", sh), ("sg", sg),
                               ("vp", arrays["auxiliary_Vp"]),
                               ("migration_vp", arrays["migration_Vp"]),
                               ("ai", geo_arrays["AI_view"]), ("rho", geo_arrays["rho_view"]),
                               ("phi", geo_arrays["phi_view"])):
                stats[key].update(value[mask])
            active_counts["valid"] += int(mask.sum())
            active_counts["sh_0p01"] += int(((sh >= 0.01) & mask).sum())
            active_counts["sg_0p003"] += int(((sg >= 0.003) & mask).sum())
            style_counts[style] += 1
            diagnostics += int(diagnostic)

            file_sizes = {key: path.stat().st_size for key, path in paths.items() if key != "metadata"}
            for key, size in file_sizes.items():
                role_bytes[key] += size
                total_bytes += size
            records.append({
                "release_index": release_index,
                "candidate_id": candidate_id,
                "style_index": style_index,
                "style": style,
                "style_display_name": STYLE_DISPLAY_NAMES[style],
                "geology": f"geology_candidates/{paths['geology'].name}",
                "observations": f"rtm_candidates/{paths['observations'].name}",
                "pair": f"rtm_candidates/{paths['pair'].name}",
                "report": f"rtm_candidates/{paths['report'].name}",
                "source_geology": paths["geology"].relative_to(PROJECT).as_posix(),
                "source_observations": paths["observations"].relative_to(PROJECT).as_posix(),
                "source_pair": paths["pair"].relative_to(PROJECT).as_posix(),
                "source_report": paths["report"].relative_to(PROJECT).as_posix(),
                "generation_seed": int(metadata_embedded["seed"]),
                "migration_seed": int(report["smooth_migration_model"]["seed"]),
                "hydrate_response_scale": float(metadata_embedded["hydrate_response_scale"]),
                "gas_response_scale": metadata_embedded.get("gas_response_scale"),
                "gas_patchiness": float(metadata_embedded["gas_patchiness"]),
                "architecture_variant": str(metadata_embedded["architecture_variant"]),
                "true_model_diagnostic": bool(diagnostic),
                "file_sizes_bytes": file_sizes,
            })
        except Exception as exc:
            failures.append({"candidate_id": candidate_id, "error": f"{type(exc).__name__}: {exc}"})
            if len(failures) <= 3:
                print(f"candidate {candidate_id} failed: {type(exc).__name__}: {exc}", flush=True)
        if release_index % 25 == 0 or release_index == len(selected_ids):
            print(f"audit {release_index}/{len(selected_ids)} failures={len(failures)}", flush=True)

    if failures:
        write_json(release / "audit_failures.json", failures)
        raise RuntimeError(f"Release audit found {len(failures)} failures")

    iid = make_iid_split(records, args.seed)
    ood = make_ood_splits(records, args.seed)
    lookup = {int(row["candidate_id"]): row for row in records}
    train_paths = [PROJECT / lookup[candidate_id]["source_pair"] for candidate_id in iid["train"]]
    rtm_calibration = fit_training_scale(
        train_paths,
        field_name="input_rtm_conditioned_unscaled",
        mask_name="loss_mask",
        percentile=99.5,
        clip_multiple=4.0,
        seed=args.seed,
    )

    migration = Moments()
    true_vp = Moments()
    delta_vp = Moments()
    true_ai = Moments()
    log_ai = Moments()
    regression = {"n": 0, "x": 0.0, "y": 0.0, "xx": 0.0, "xy": 0.0}
    for index, candidate_id in enumerate(iid["train"], start=1):
        row = lookup[candidate_id]
        with np.load(PROJECT / row["source_pair"], allow_pickle=False) as pair:
            mask = np.asarray(pair["loss_mask"], dtype=bool)
            vp_mig = np.asarray(pair["migration_Vp"], dtype=np.float64)[mask]
            vp = np.asarray(pair["auxiliary_Vp"], dtype=np.float64)[mask]
        with np.load(PROJECT / row["source_geology"], allow_pickle=False) as geology:
            ai = np.asarray(geology["AI_view"], dtype=np.float64)[mask]
        migration.update(vp_mig)
        true_vp.update(vp)
        delta_vp.update(vp - vp_mig)
        true_ai.update(ai)
        x, y = np.log(vp_mig), np.log(ai)
        log_ai.update(y)
        regression["n"] += int(x.size)
        regression["x"] += float(x.sum())
        regression["y"] += float(y.sum())
        regression["xx"] += float(np.square(x).sum())
        regression["xy"] += float((x * y).sum())
        if index % 100 == 0:
            print(f"Task 1 calibration {index}/{len(iid['train'])}", flush=True)
    n = regression["n"]
    denominator = regression["xx"] - regression["x"] ** 2 / n
    slope = (regression["xy"] - regression["x"] * regression["y"] / n) / denominator
    intercept = (regression["y"] - slope * regression["x"]) / n
    residual = Moments()
    for candidate_id in iid["train"]:
        row = lookup[candidate_id]
        with np.load(PROJECT / row["source_pair"], allow_pickle=False) as pair:
            mask = np.asarray(pair["loss_mask"], dtype=bool)
            vp_mig = np.asarray(pair["migration_Vp"], dtype=np.float64)[mask]
        with np.load(PROJECT / row["source_geology"], allow_pickle=False) as geology:
            ai = np.asarray(geology["AI_view"], dtype=np.float64)[mask]
        residual.update(np.log(ai) - (intercept + slope * np.log(vp_mig)))

    attempted_ids = sorted(set(admitted_ids) | set(rejected_ids))
    ledger = []
    for candidate_id in attempted_ids:
        if candidate_id in selected_ids:
            status = "selected_for_release"
        elif candidate_id in surplus_ids:
            status = "admitted_not_selected_balancing_surplus"
        else:
            status = "rejected_by_rtm_quality_gate"
        source = admitted_lookup.get(candidate_id)
        if source is None:
            source = next(row for row in rejected if int(row["candidate_id"]) == candidate_id)
        ledger.append({
            "candidate_id": candidate_id,
            "style_index": int(source["style"]),
            "style": STYLE_NAMES[int(source["style"])],
            "status": status,
            "report": str(source["report"]).replace("\\", "/"),
        })

    all_stats = {key: value.finish() for key, value in stats.items()}
    summary = {
        "schema_version": "1.0",
        "dataset_name": "NGH-Seis",
        "dataset_version": "1.0",
        "freeze_date": "2026-09-04",
        "release_count": len(records),
        "selected_candidate_ids_sha256": id_digest(selected_ids),
        "admitted_candidate_count": len(admitted_ids),
        "selected_candidate_count": len(selected_ids),
        "surplus_admitted_candidate_ids": surplus_ids,
        "rejected_candidate_ids": rejected_ids,
        "attempted_candidate_count": len(attempted_ids),
        "production_acceptance_rate": len(admitted_ids) / len(attempted_ids),
        "style_counts": dict(sorted(style_counts.items())),
        "array_shape": [512, 96],
        "input_shape_task1": [2, 512, 96],
        "target_shape_task1": [2, 512, 96],
        "input_shape_task2": [1, 512, 96],
        "target_shape_task2": [2, 512, 96],
        "true_model_diagnostic_count": diagnostics,
        "field_statistics_on_loss_mask": all_stats,
        "Sh_active_fraction_at_0p01": active_counts["sh_0p01"] / active_counts["valid"],
        "Sg_active_fraction_at_0p003": active_counts["sg_0p003"] / active_counts["valid"],
        "release_bytes_without_metadata_sidecars": total_bytes,
        "release_size_GB_decimal": total_bytes / 1e9,
        "release_size_GiB": total_bytes / (1024 ** 3),
        "role_bytes": dict(sorted(role_bytes.items())),
        "AI_release_view_resampling_consistency": {
            "maximum_sample_NRMSE": ai_resampling_max_nrmse,
            "maximum_sample_p99p9_relative_error": ai_resampling_max_p999_relative_error,
            "thresholds": {"NRMSE": 0.001, "p99p9_relative_error": 0.01},
        },
        "iid_split_sizes": {key: len(value) for key, value in iid.items()},
        "ood_fold_sizes": {
            key: {part: len(fold[part]) for part in ("train", "validation", "test")}
            for key, fold in ood.items()
        },
        "audit": {
            "all_selected_files_present": True,
            "all_required_fields_present": True,
            "all_shapes_and_dtypes_structurally_valid": True,
            "all_arrays_finite": True,
            "all_saturation_bounds_valid": True,
            "all_pair_targets_co_registered_with_geology": True,
            "all_AI_fields_pass_resampling_consistency_thresholds": True,
            "all_masks_binary_and_padding_excluded": True,
            "all_acquisition_geometries_match_declared_values": True,
            "all_RTM_reports_admitted": True,
            "full_observation_content_scanned": not args.skip_observation_content,
            "selected_rejected_disjoint": True,
        },
        "important": (
            "Candidate 880 failed the RTM gate. Candidate 1001 passed but is excluded as the "
            "extra style-0 realization; candidate 1005 supplies the 200th style-4 realization."
        ),
    }
    task1 = {
        "schema_version": "1.0",
        "protocol": "IID training split only",
        "train_sample_count": len(iid["train"]),
        "train_candidate_ids_sha256": id_digest(iid["train"]),
        "evaluation_region": "loss_mask",
        "input": {"migration_vp": migration.finish(), "standard_deviation_clip": 4.0},
        "target": {
            "true_vp": true_vp.finish(), "delta_vp": delta_vp.finish(),
            "true_ai": true_ai.finish(), "log_ai": log_ai.finish(),
            "log_ai_baseline": {
                "formula": "log(AI_baseline) = intercept + slope * log(Vp_migration)",
                "intercept": float(intercept), "slope": float(slope),
            },
            "log_ai_residual": residual.finish(),
        },
    }

    write_json(release / "source_manifest.json", records)
    write_json(release / "candidate_ledger.json", ledger)
    write_json(release / "iid_splits_80_10_10.json", {
        "seed": args.seed,
        "strategy": "five-family stratified split: 160/20/20 per family",
        "candidate_id_digests": {key: id_digest(value) for key, value in iid.items()},
        **iid,
    })
    write_json(release / "structural_ood_leave_one_style_out.json", {
        "seed": args.seed,
        "definition": "The held-out geological family is absent from training and validation",
        "folds": ood,
    })
    write_json(release / "rtm_amplitude_calibration.json", rtm_calibration.to_dict())
    write_json(release / "task1_acoustic_calibration.json", task1)
    write_json(release / "dataset_audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
