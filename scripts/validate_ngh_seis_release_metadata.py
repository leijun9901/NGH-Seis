"""Validate the internal consistency of frozen NGH-Seis release metadata."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    args = parser.parse_args()
    root = args.metadata.resolve()
    manifest = json.loads((root / "accepted_manifest.json").read_text(encoding="utf-8"))
    ledger = json.loads((root / "candidate_ledger.json").read_text(encoding="utf-8"))
    iid = json.loads((root / "iid_splits_80_10_10.json").read_text(encoding="utf-8"))
    ood = json.loads((root / "structural_ood_leave_one_style_out.json").read_text(encoding="utf-8"))["folds"]
    summary = json.loads((root / "dataset_audit_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "NGH_SEIS_ARRAY_SCHEMA.json").read_text(encoding="utf-8"))

    ids = [int(row["candidate_id"]) for row in manifest]
    id_set = set(ids)
    assert len(manifest) == len(id_set) == 1000
    assert [int(row["release_index"]) for row in manifest] == list(range(1, 1001))
    styles = {int(row["candidate_id"]): row["style"] for row in manifest}
    assert set(Counter(styles.values()).values()) == {200}
    assert summary["release_count"] == schema["accepted_samples_scanned"] == 1000

    paths = []
    total_bytes = 0
    for row in manifest:
        assert set(row["files"]) == {"geology", "observations", "pair", "report"}
        for item in row["files"].values():
            paths.append(item["path"])
            total_bytes += int(item["size_bytes"])
            assert len(item["sha256"]) == 64
    assert len(paths) == len(set(paths)) == 4000
    assert total_bytes == int(summary["release_bytes_without_metadata_sidecars"])

    split_sets = {key: set(map(int, iid[key])) for key in ("train", "validation", "test")}
    assert {key: len(value) for key, value in split_sets.items()} == {
        "train": 800, "validation": 100, "test": 100
    }
    assert not (split_sets["train"] & split_sets["validation"])
    assert not (split_sets["train"] & split_sets["test"])
    assert not (split_sets["validation"] & split_sets["test"])
    assert set.union(*split_sets.values()) == id_set
    expected_per_style = {"train": 160, "validation": 20, "test": 20}
    for split, values in split_sets.items():
        assert set(Counter(styles[value] for value in values).values()) == {expected_per_style[split]}

    for fold in ood.values():
        fold_sets = {key: set(map(int, fold[key])) for key in ("train", "validation", "test")}
        assert {key: len(value) for key, value in fold_sets.items()} == {
            "train": 720, "validation": 80, "test": 200
        }
        assert not (fold_sets["train"] & fold_sets["validation"])
        assert not (fold_sets["train"] & fold_sets["test"])
        assert not (fold_sets["validation"] & fold_sets["test"])
        assert set.union(*fold_sets.values()) == id_set
        held_out = fold["held_out_style"]
        assert {styles[value] for value in fold_sets["test"]} == {held_out}
        assert held_out not in {styles[value] for value in fold_sets["train"] | fold_sets["validation"]}

    checksum_rows = [line.split("  ", 1) for line in
                     (root / "checksums.sha256").read_text(encoding="utf-8").splitlines() if line]
    assert len(checksum_rows) == 4000
    assert {path for _, path in checksum_rows} == set(paths)
    assert len({digest for digest, _ in checksum_rows}) == 4000

    ledger_status = Counter(row["status"] for row in ledger)
    assert ledger_status == {
        "selected_for_release": 1000,
        "admitted_not_selected_balancing_surplus": 1,
        "rejected_by_rtm_quality_gate": 1,
    }
    result = {
        "status": "pass",
        "manifest_rows": 1000,
        "unique_candidate_ids": 1000,
        "continuous_release_indices": True,
        "family_counts": dict(sorted(Counter(styles.values()).items())),
        "unique_data_paths": 4000,
        "unique_file_sha256_digests": 4000,
        "iid_partition_and_stratification_valid": True,
        "five_structural_ood_partitions_valid": True,
        "candidate_ledger_valid": True,
        "logical_data_bytes": total_bytes,
    }
    (root / "final_release_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
