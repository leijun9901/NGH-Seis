"""Verify repository paths, sizes and optional SHA-256 after download."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--mode",
        choices=("existence", "size", "sha256"),
        default="sha256",
        help="sha256 is the publication-grade check; size/existence are faster smoke tests",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.max_samples is not None:
        records = records[: args.max_samples]
    failures = []
    checked = 0
    for index, record in enumerate(records, start=1):
        for role, metadata in record["files"].items():
            path = dataset_root / metadata["path"]
            error = None
            if not path.is_file():
                error = "missing"
            elif args.mode in ("size", "sha256") and path.stat().st_size != int(metadata["size_bytes"]):
                error = f"size mismatch: {path.stat().st_size} != {metadata['size_bytes']}"
            elif args.mode == "sha256" and _sha256(path) != metadata["sha256"]:
                error = "SHA-256 mismatch"
            if error is not None:
                failures.append(
                    {
                        "candidate_id": int(record["candidate_id"]),
                        "role": role,
                        "path": metadata["path"],
                        "error": error,
                    }
                )
            checked += 1
        if index % 50 == 0:
            print(f"verify {index}/{len(records)} failures={len(failures)}", flush=True)
    result = {
        "status": "pass" if not failures else "fail",
        "mode": args.mode,
        "samples_checked": len(records),
        "files_checked": checked,
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
