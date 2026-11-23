"""Evaluate train-only, per-position mean/median baselines on NGH-Seis v1.0.

Reads the frozen public release without modifying it. Outputs JSON separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np


FIELDS = ("Vp", "AI", "Sh", "Sg")
THRESHOLDS = {"Sh": 0.01, "Sg": 0.003}


def load_data(release: Path, metadata: Path):
    manifest = json.loads((metadata / "accepted_manifest.json").read_text(encoding="utf-8"))
    arrays = {name: np.empty((len(manifest), 512, 96), np.float32) for name in FIELDS}
    masks = np.empty((len(manifest), 512, 96), np.bool_)
    index = {}
    for i, record in enumerate(manifest):
        cid = int(record["candidate_id"])
        index[cid] = i
        with np.load(release / record["pair"], allow_pickle=False) as pair:
            arrays["Vp"][i] = pair["auxiliary_Vp"]
            arrays["Sh"][i] = pair["target_Sh"]
            arrays["Sg"][i] = pair["target_Sg"]
            masks[i] = pair["loss_mask"].astype(bool)
        with np.load(release / record["geology"], allow_pickle=False) as geology:
            arrays["AI"][i] = geology["AI_view"]
        if (i + 1) % 100 == 0:
            print(f"loaded {i + 1}/{len(manifest)}", flush=True)
    return arrays, masks, index


def fit_templates(values: np.ndarray, masks: np.ndarray, train: np.ndarray,
                  methods=("mean", "median")):
    train_values = values[train]
    train_mask = masks[train]
    counts = train_mask.sum(axis=0, dtype=np.int32)
    sums = np.sum(train_values * train_mask, axis=0, dtype=np.float64)
    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0).astype(np.float32)
    if "median" not in methods:
        return {"mean": mean}, int(np.count_nonzero(counts == 0))
    median = np.zeros(values.shape[1:], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for start in range(0, values.shape[1], 16):
            stop = min(start + 16, values.shape[1])
            slab = np.where(train_mask[:, start:stop], train_values[:, start:stop], np.nan)
            median[start:stop] = np.nan_to_num(np.nanmedian(slab, axis=0), nan=0.0)
    return {"mean": mean, "median": median}, int(np.count_nonzero(counts == 0))


def metrics(name: str, template: np.ndarray, truth: np.ndarray, mask: np.ndarray,
            sh: np.ndarray, sg: np.ndarray):
    if name in {"Vp", "AI"}:
        scale = 1.0e-6 if name == "AI" else 1.0
        result = {}
        for region, region_mask in (
            ("all", mask),
            ("reservoir", mask & ((sh >= 0.01) | (sg >= 0.003))),
        ):
            error = (truth - template[None])[region_mask].astype(np.float64) * scale
            result[f"{region}_MAE"] = float(np.mean(np.abs(error)))
            result[f"{region}_RMSE"] = float(np.sqrt(np.mean(error * error)))
        return result
    error = (truth - template[None])[mask].astype(np.float64)
    active = mask & (truth >= THRESHOLDS[name])
    pred_pos = mask & (template[None] >= THRESHOLDS[name])
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error * error))),
        "active_MAE": float(np.mean(np.abs((truth - template[None])[active]))),
        "Dice": float(2 * np.count_nonzero(pred_pos & active) / max(
            np.count_nonzero(pred_pos) + np.count_nonzero(active), 1)),
        "predicted_active_fraction": float(np.count_nonzero(pred_pos) / np.count_nonzero(mask)),
        "true_active_fraction": float(np.count_nonzero(active) / np.count_nonzero(mask)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True,
                        help="Metadata directory of the extracted public dataset")
    parser.add_argument("--methods", nargs="+", choices=("mean", "median"),
                        default=["mean"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = args.release.resolve()
    release = metadata.parent
    arrays, masks, index = load_data(release, metadata)
    iid = json.loads((metadata / "iid_splits_80_10_10.json").read_text(encoding="utf-8"))
    ood = json.loads((metadata / "structural_ood_leave_one_style_out.json").read_text(encoding="utf-8"))["folds"]
    protocols = {"iid": iid} | ood
    output = {
        "dataset": "NGH-Seis v1.0",
        "method": "train-only pixelwise mean/median, masked by loss_mask",
        "units": {"Vp": "m/s", "AI": "MRayl", "Sh": "fraction", "Sg": "fraction"},
        "protocols": {},
    }
    for protocol, split in protocols.items():
        train = np.asarray([index[int(cid)] for cid in split["train"]], dtype=np.int32)
        test = np.asarray([index[int(cid)] for cid in split["test"]], dtype=np.int32)
        train_coverage = np.any(masks[train], axis=0)
        report = {
            "train_count": int(train.size),
            "test_count": int(test.size),
            "test_valid_pixels_without_train_coverage": int(
                np.count_nonzero(masks[test] & ~train_coverage[None])
            ),
            "fields": {},
        }
        for name in FIELDS:
            templates, unobserved = fit_templates(arrays[name], masks, train, args.methods)
            report["fields"][name] = {
                method: metrics(name, template, arrays[name][test], masks[test],
                                arrays["Sh"][test], arrays["Sg"][test])
                for method, template in templates.items()
            }
            report["fields"][name]["unobserved_train_pixels"] = unobserved
            print(f"{protocol}: {name} done", flush=True)
        output["protocols"][protocol] = report
    ood_names = list(ood)
    output["ood_macro"] = {}
    for name in FIELDS:
        output["ood_macro"][name] = {}
        for method in args.methods:
            keys = output["protocols"][ood_names[0]]["fields"][name][method]
            output["ood_macro"][name][method] = {
                key: {"mean": float(np.mean([output["protocols"][fold]["fields"][name][method][key] for fold in ood_names])),
                      "std": float(np.std([output["protocols"][fold]["fields"][name][method][key] for fold in ood_names]))}
                for key in keys
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
