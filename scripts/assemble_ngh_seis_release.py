"""Assemble the frozen NGH-Seis v1.0 directory without duplicating data.

On the same NTFS volume, the default mode creates hard links to the audited
source archives.  ``--copy`` creates independent files for export to another
disk or repository staging area.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


PROJECT = Path(__file__).resolve().parents[1]

ROLE_SOURCE_KEYS = {
    "geology": "source_geology",
    "observations": "source_observations",
    "pair": "source_pair",
    "report": "source_report",
}

METADATA_FILES = [
    "accepted_manifest.json",
    "candidate_ledger.json",
    "checksums.sha256",
    "dataset_audit_summary.json",
    "iid_splits_80_10_10.json",
    "near_duplicate_and_split_leakage_audit.json",
    "NEAR_DUPLICATE_AUDIT.md",
    "NGH_SEIS_ARRAY_SCHEMA.json",
    "rtm_amplitude_calibration.json",
    "scientific_data_manifest.jsonl",
    "scientific_data_release_summary.json",
    "structural_ood_leave_one_style_out.json",
    "task1_acoustic_calibration.json",
    "DATA_DICTIONARY.md",
    "RELEASE_FREEZE_REPORT.md",
    "MANUSCRIPT_NUMBERS.md",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-source",
        type=Path,
        default=PROJECT / "outputs" / "NGH-Seis-v1.0-metadata",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=PROJECT / "outputs" / "NGH-Seis-v1.0",
    )
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()
    metadata_source = args.metadata_source.resolve()
    destination = args.destination.resolve()

    internal = json.loads((metadata_source / "source_manifest.json").read_text(encoding="utf-8"))
    public = json.loads((metadata_source / "accepted_manifest.json").read_text(encoding="utf-8"))
    public_lookup = {int(row["candidate_id"]): row for row in public}
    if len(internal) != 1000 or len(public_lookup) != 1000:
        raise RuntimeError("Expected 1,000 internal and public manifest rows")

    linked = copied = reused = 0
    bytes_total = 0
    for index, row in enumerate(internal, start=1):
        candidate_id = int(row["candidate_id"])
        public_row = public_lookup[candidate_id]
        for role, source_key in ROLE_SOURCE_KEYS.items():
            source = (PROJECT / row[source_key]).resolve()
            target = destination / public_row["files"][role]["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            expected_size = int(public_row["files"][role]["size_bytes"])
            if source.stat().st_size != expected_size:
                raise RuntimeError(f"Source size changed for candidate {candidate_id}, role {role}")
            if target.exists():
                if target.stat().st_size != expected_size:
                    raise RuntimeError(f"Existing target has wrong size: {target}")
                reused += 1
            elif args.copy:
                shutil.copy2(source, target)
                copied += 1
            else:
                os.link(source, target)
                linked += 1
            bytes_total += expected_size
        if index % 100 == 0:
            print(f"assembled {index}/1000", flush=True)

    metadata_destination = destination / "metadata"
    metadata_destination.mkdir(parents=True, exist_ok=True)
    copied_metadata = []
    for name in METADATA_FILES:
        source = metadata_source / name
        if source.is_file():
            shutil.copy2(source, metadata_destination / name)
            copied_metadata.append(name)
    config_destination = destination / "configs"
    config_destination.mkdir(parents=True, exist_ok=True)
    for name in ("marine_streamer.json", "ngh_seis_geology.json"):
        shutil.copy2(PROJECT / "configs" / name, config_destination / name)

    report = {
        "schema_version": "1.0",
        "dataset": "NGH-Seis v1.0",
        "mode": "copy" if args.copy else "NTFS hard link",
        "data_files": 4000,
        "linked": linked,
        "copied": copied,
        "reused": reused,
        "logical_data_bytes": bytes_total,
        "metadata_files": copied_metadata,
        "destination": str(destination),
    }
    (metadata_destination / "assembly_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
