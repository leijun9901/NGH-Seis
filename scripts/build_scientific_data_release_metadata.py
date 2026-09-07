"""Build repository-ready metadata and checksums for NGH-Seis v1.0.

The script does not copy or modify the immutable production arrays. It binds
every accepted sample to its geology, full 25-shot observations, training pair,
and RTM quality report, then writes portable POSIX paths and SHA-256 checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_json(path: Path, payload: object) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--production",
        type=Path,
        default=PROJECT / "outputs" / "NGH-Seis-v1.0-data",
    )
    parser.add_argument(
        "--release",
        type=Path,
        default=PROJECT / "outputs" / "NGH-Seis-v1.0-metadata",
    )
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Build the inventory without reading all file contents.",
    )
    args = parser.parse_args()
    production = args.production.resolve()
    release = args.release.resolve()
    source_manifest = json.loads(
        (release / "source_manifest.json").read_text(encoding="utf-8")
    )
    if len(source_manifest) != 1000:
        raise RuntimeError("The NGH-Seis v1.0 release must contain exactly 1000 accepted rows")

    rows: list[dict] = []
    checksum_lines: list[str] = []
    total_bytes = 0
    role_totals: dict[str, int] = {key: 0 for key in ("geology", "observations", "pair", "report")}
    seen_ids: set[int] = set()
    for index, source in enumerate(source_manifest, start=1):
        candidate_id = int(source["candidate_id"])
        if candidate_id in seen_ids:
            raise RuntimeError(f"Duplicate accepted candidate ID {candidate_id}")
        seen_ids.add(candidate_id)
        stem = f"sample_{candidate_id:04d}"
        relative = {
            "geology": Path("data/geology") / Path(source["geology"]).name,
            "observations": Path("data/observations") / Path(
                source.get("observations", f"rtm_candidates/{stem}_observations25.npz")
            ).name,
            "pair": Path("data/training_pairs") / Path(source["pair"]).name,
            "report": Path("quality/rtm_reports") / Path(source["report"]).name,
        }
        files = {}
        for role, relpath in relative.items():
            source_key = f"source_{role}"
            # The source manifest uses source_pair/source_report/etc. and the
            # public release uses the cleaner role-specific directory above.
            absolute = (
                PROJECT / source[source_key]
                if source_key in source
                else production / relpath
            )
            if not absolute.is_file():
                raise FileNotFoundError(f"Missing {role} file for candidate {candidate_id}: {absolute}")
            size = absolute.stat().st_size
            digest = None if args.skip_hashes else _sha256(absolute)
            portable = relpath.as_posix()
            files[role] = {
                "path": portable,
                "size_bytes": size,
                "sha256": digest,
            }
            total_bytes += size
            role_totals[role] += size
            if digest is not None:
                checksum_lines.append(f"{digest}  {portable}")
        row = {key: value for key, value in source.items() if not key.startswith("source_")}
        row["release_index"] = index
        row["observations"] = relative["observations"].as_posix()
        row["files"] = files
        rows.append(row)
        if index % 50 == 0:
            print(f"release inventory {index}/1000", flush=True)

    config_paths = [
        PROJECT / "configs" / "marine_streamer.json",
        PROJECT / "configs" / "ngh_seis_geology.json",
    ]
    config_records = []
    for path in config_paths:
        digest = None if args.skip_hashes else _sha256(path)
        config_records.append(
            {
                "path": path.relative_to(PROJECT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )

    jsonl = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    _write_text_atomic(release / "scientific_data_manifest.jsonl", jsonl)
    _write_json(release / "accepted_manifest.json", rows)
    if not args.skip_hashes:
        _write_text_atomic(release / "checksums.sha256", "\n".join(checksum_lines) + "\n")
    summary = {
        "schema_version": "1.0",
        "dataset_name": "NGH-Seis",
        "dataset_version": "1.0",
        "accepted_samples": len(rows),
        "files_per_sample": 4,
        "data_file_count": len(rows) * 4,
        "total_size_bytes": total_bytes,
        "total_size_gib": total_bytes / (1024**3),
        "role_size_bytes": role_totals,
        "hash_algorithm": None if args.skip_hashes else "SHA-256",
        "manifest": "scientific_data_manifest.jsonl",
        "checksum_file": None if args.skip_hashes else "checksums.sha256",
        "frozen_configs": config_records,
        "path_rule": "all published data paths are relative to the assembled dataset root and use POSIX separators",
        "license": "CC BY 4.0",
        "rights_holder": "Central South University",
    }
    _write_json(release / "scientific_data_release_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
