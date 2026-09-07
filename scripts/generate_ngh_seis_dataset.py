"""Resumable NGH-Seis v1.0 generator with RTM quality admission.

This script deliberately runs one GPU worker: Deepwave RTM is GPU-bound and
parallel workers would contend for the same device.  Candidate IDs are never
overwritten, so rejected realisations remain auditable and accepted IDs are
recorded in a manifest for later split construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.generate_ngh_seis_geology import _extract  # noqa: E402
from scripts.release_quality_checks import _sample_audit  # noqa: E402
from scripts.run_ngh_seis_forward_rtm import run_sample  # noqa: E402
from src.geology.blake_ew0008 import generate_blake_ew0008  # noqa: E402


CONFIG = PROJECT / "configs" / "ngh_seis_geology.json"


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _materialize_geology(candidate_id: int, config_path: Path, geology_dir: Path) -> Path:
    """Create a complete, deterministic release-schema geology sample."""
    destination = geology_dir / f"sample_{candidate_id:04d}_blake_geology.npz"
    if destination.exists():
        return destination
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = 20_260_818 + candidate_id * 104_729
    full, metadata = generate_blake_ew0008(
        config_path, candidate_id, seed, include_proxy_image=False
    )
    view: dict[str, np.ndarray] = {}
    valid = None
    view_metadata = None
    for output_name, source_name in (
        ("Vp", "Vp_full"), ("rho", "rho_full"), ("phi", "phi_full"),
        ("Sh", "Sh_full"), ("Sg", "Sg_full"), ("AI", "AI_full"),
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
    if not np.isfinite(full["Vp_full"]).all() or np.max(full["Sh_full"] + full["Sg_full"]) > 1.0:
        raise RuntimeError(f"candidate {candidate_id} failed geology finite/saturation check")

    audit_arrays = {
        "Vp_full": full["Vp_full"],
        "rho_full": full["rho_full"],
        "phi_full": full["phi_full"],
        "Sh_full": full["Sh_full"],
        "Sg_full": full["Sg_full"],
        "Vp_view": view["Vp"],
        "rho_view": view["rho"],
        "phi_view": view["phi"],
        "Sh_view": view["Sh"],
        "Sg_view": view["Sg"],
        "AI_view": view["AI"],
        "lithology_view": view["lithology"],
        "loss_mask": loss_mask,
    }
    metadata.update(
        {
            "role": "geology and rock-physics model; no seismic proxy released as network input",
            "field_aligned_view": view_metadata,
            "well_validation_status": "regional ODP prior only; no strict EW0008 through-well trace",
            "sample_audit": _sample_audit(
                candidate_id, audit_arrays, metadata, config
            ),
        }
    )
    geology_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "x_m": full["x_m"], "z_m": full["z_m"],
        "seafloor_m": full["seafloor_m"], "bsr_m": full["bsr_m"],
        "phi_full": full["phi_full"], "Sh_full": full["Sh_full"],
        "Sg_full": full["Sg_full"], "Vp_full": full["Vp_full"],
        "rho_full": full["rho_full"], "Vp_view": view["Vp"],
        "rho_view": view["rho"], "phi_view": view["phi"],
        "Sh_view": view["Sh"], "Sg_view": view["Sg"],
        "AI_view": view["AI"],
        "lithology_proxy_full": full["lithology_proxy_full"],
        "lithology_view": view["lithology"], "valid_mask": valid,
        "loss_mask": loss_mask,
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
    }
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        with np.load(temporary, allow_pickle=False) as check:
            required = set(payload)
            if set(check.files) != required:
                raise RuntimeError(f"candidate {candidate_id} failed geology schema check")
            if not all(np.all(np.isfinite(check[key])) for key in required - {"metadata_json"}):
                raise RuntimeError(f"candidate {candidate_id} contains non-finite arrays")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    _write_json(
        geology_dir / f"sample_{candidate_id:04d}_metadata.json", metadata
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-target", type=int, default=20)
    parser.add_argument("--candidate-start", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--diagnostic-every", type=int, default=20,
        help="Run the true-Vp diagnostic RTM for every Nth candidate; 0 disables it.",
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--geology-folder", type=Path, default=None,
        help="Optional existing deterministic geology folder; avoids regenerating geology arrays.",
    )
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/NGH-Seis-v1.0-generation")
    args = parser.parse_args()
    if args.accepted_target < 1 or args.max_candidates < 1:
        raise ValueError("accepted-target and max-candidates must both be positive")
    output = args.output
    geology_dir = args.geology_folder or (output / "geology_candidates")
    rtm_dir = output / "rtm_candidates"
    manifest_path = output / "production_manifest.json"
    manifest = {"schema_version": "1.0", "config": str(args.config), "accepted": [], "rejected": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    accepted_ids = {int(x["candidate_id"]) for x in manifest["accepted"]}
    rejected_ids = {int(x["candidate_id"]) for x in manifest["rejected"]}
    accepted_needed = max(args.accepted_target - len(accepted_ids), 0)
    if accepted_needed > 0 and manifest.get("complete"):
        manifest["complete"] = False
        _write_json(manifest_path, manifest)
    if args.max_candidates < accepted_needed:
        raise ValueError(
            "max-candidates must cover at least the remaining accepted target: "
            f"need {accepted_needed}, got {args.max_candidates}"
        )
    for candidate_id in range(args.candidate_start, args.candidate_start + args.max_candidates):
        if len(accepted_ids) >= args.accepted_target:
            break
        if candidate_id in accepted_ids or candidate_id in rejected_ids:
            continue
        _materialize_geology(candidate_id, args.config, geology_dir)
        diagnostic = args.diagnostic_every > 0 and candidate_id % args.diagnostic_every == 0
        report_path = run_sample(
            candidate_id, batch_size=args.batch_size, stop_after_batches=None,
            geology_config_path=args.config, geology_folder=geology_dir, output=rtm_dir,
            include_true_model_diagnostic=diagnostic, write_visualization=diagnostic,
        )
        if report_path is None:
            raise RuntimeError(f"candidate {candidate_id} did not complete; rerun resumes it")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        entry = {"candidate_id": candidate_id, "style": (candidate_id - 1) % 5,
                 "true_model_diagnostic": diagnostic,
                 "report": str(report_path.relative_to(output)), "status": report["status"]}
        if report["admission"]["rtm_quality_gate_pass"]:
            manifest["accepted"].append(entry)
            accepted_ids.add(candidate_id)
        else:
            manifest["rejected"].append(entry)
            rejected_ids.add(candidate_id)
        _write_json(manifest_path, manifest)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    manifest["complete"] = len(accepted_ids) >= args.accepted_target
    _write_json(manifest_path, manifest)
    print(json.dumps({"accepted": len(accepted_ids), "rejected": len(rejected_ids), "complete": manifest["complete"]}))


if __name__ == "__main__":
    main()
