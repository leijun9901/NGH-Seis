"""Load and visualize one NGH-Seis v1.0 sample.

Example:
    python examples/inspect_ngh_seis_sample.py --candidate-id 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]


def load_record(candidate_id: int, release_root: Path) -> dict:
    manifest_path = release_root / "accepted_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [row for row in manifest if int(row["candidate_id"]) == candidate_id]
    if len(matches) != 1:
        raise KeyError(f"candidate_id={candidate_id} is not uniquely present in the release")
    return matches[0]


def resolve_roots(production_root: Path, release_root: Path) -> tuple[Path, Path]:
    """Accept both the development tree and final public repository layout."""
    if (release_root / "accepted_manifest.json").is_file():
        return production_root, release_root
    if (production_root / "metadata" / "accepted_manifest.json").is_file():
        return production_root, production_root / "metadata"
    raise FileNotFoundError("accepted_manifest.json was not found in the supplied release layout")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", type=int, default=1)
    parser.add_argument(
        "--production-root",
        type=Path,
        default=PROJECT / "outputs" / "NGH-Seis_v1.0",
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        default=PROJECT / "outputs" / "NGH-Seis_v1.0" / "metadata",
    )
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()
    production_root, release_root = resolve_roots(args.production_root, args.release_root)
    record = load_record(args.candidate_id, release_root)
    pair_path = production_root / record["pair"]
    observations_path = production_root / record["observations"]

    with np.load(pair_path, allow_pickle=False) as pair:
        rtm = pair["input_rtm_display_only"].copy()
        sh = pair["target_Sh"].copy()
        sg = pair["target_Sg"].copy()
        vp = pair["auxiliary_Vp"].copy()
        migration_vp = pair["migration_Vp"].copy()
        mask = pair["loss_mask"].astype(bool)
    with np.load(observations_path, allow_pickle=False) as observations:
        raw = observations["raw_gathers"][12].copy()
        processed = observations["processed_gathers"][12].copy()

    extent = [0.0, 3.525, 1.341, -0.192]
    panels = [
        (raw, "Representative raw gather", "gray", None),
        (processed, "Representative processed gather", "gray", None),
        (migration_vp, "Migration Vp", "turbo", "m/s"),
        (rtm, "Adjoint migration input", "gray", "display scale"),
        (vp, "True Vp", "turbo", "m/s"),
        (sh, "Hydrate saturation Sh", "viridis", "fraction"),
        (sg, "Free-gas saturation Sg", "magma", "fraction"),
        (mask.astype(float), "Loss mask", "gray", "0/1"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for ax, (array, title, cmap, unit) in zip(axes.ravel(), panels):
        if array.shape == (512, 96):
            image = ax.imshow(array, extent=extent, aspect="auto", cmap=cmap)
            ax.set(xlabel="distance (km)", ylabel="depth relative to seafloor (km)")
        else:
            scale = max(float(np.percentile(np.abs(array), 99.5)), 1e-12)
            image = ax.imshow(
                array,
                aspect="auto",
                cmap=cmap,
                vmin=-scale,
                vmax=scale,
                extent=[1, array.shape[1], array.shape[0] * 0.004, 0],
            )
            ax.set(xlabel="receiver channel", ylabel="time (s)")
        ax.set_title(title)
        if unit is not None:
            fig.colorbar(image, ax=ax, shrink=0.75, label=unit)
    fig.suptitle(
        f"NGH-Seis sample {args.candidate_id:04d} | {record['style']}",
        fontsize=14,
    )
    if args.save is None:
        plt.show()
    else:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=180)
        print(args.save.resolve())


if __name__ == "__main__":
    main()
