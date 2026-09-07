"""Create two lightweight, non-training preview archives for repository browsing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples" / "preview",
    )
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    destination = args.output_dir.resolve()
    manifest = json.loads(
        (dataset_root / "metadata" / "accepted_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    by_id = {int(row["candidate_id"]): row for row in manifest}
    destination.mkdir(parents=True, exist_ok=True)
    for candidate_id in (1, 4):
        record = by_id[candidate_id]
        pair_path = dataset_root / record["files"]["pair"]["path"]
        with np.load(pair_path, allow_pickle=False) as pair:
            payload = {
                "input_rtm_display_only": pair["input_rtm_display_only"],
                "target_Sh": pair["target_Sh"],
                "target_Sg": pair["target_Sg"],
                "auxiliary_Vp": pair["auxiliary_Vp"],
                "migration_Vp": pair["migration_Vp"],
                "loss_mask": pair["loss_mask"],
                "processed_representative_gather_decimated": pair["processed_representative_gather"][::4, ::2],
                "metadata_json": np.asarray(json.dumps({
                    "dataset": "NGH-Seis v1.0",
                    "candidate_id": candidate_id,
                    "style": record["style"],
                    "preview_only": True,
                    "quantitative_input_warning": "input_rtm_display_only is display-only; use the full release for benchmark training",
                })),
            }
        output = destination / f"candidate_{candidate_id:04d}_preview.npz"
        np.savez_compressed(output, **payload)
        print(output.resolve())


if __name__ == "__main__":
    main()
