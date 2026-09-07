"""Fit a single RTM amplitude scale from a frozen training split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.rtm_scaling import fit_training_scale  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument("--clip-multiple", type=float, default=4.0)
    args = parser.parse_args()
    files = sorted(args.train_dir.glob("sample_*.npz"))
    calibration = fit_training_scale(
        files, percentile=args.percentile, clip_multiple=args.clip_multiple
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(args.output)
    print(json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
