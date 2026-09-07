"""Train the standard U-Net baseline for Task 1: Vp and acoustic impedance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.datasets.ngh_seis_acoustic import make_ngh_seis_acoustic_loader  # noqa: E402
from src.models.benchmark_models import MODEL_NAMES, build_benchmark_model  # noqa: E402


def _masked_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, name: str
) -> torch.Tensor:
    valid = mask.expand_as(target)
    if name == "mae":
        point = torch.abs(prediction - target)
    elif name == "mse":
        point = torch.square(prediction - target)
    elif name == "smoothl1":
        point = F.smooth_l1_loss(prediction, target, reduction="none", beta=0.1)
    else:
        raise ValueError(name)
    return (point * valid).sum() / valid.sum().clamp_min(1.0)


def _decode(
    prediction: torch.Tensor,
    vp_baseline: torch.Tensor,
    ai_baseline: torch.Tensor,
    calibration: dict,
) -> torch.Tensor:
    target = calibration["target"]
    delta = prediction[:, 0:1] * float(target["delta_vp"]["std"]) + float(
        target["delta_vp"]["mean"]
    )
    log_ai_residual = prediction[:, 1:2] * float(
        target["log_ai_residual"]["std"]
    ) + float(target["log_ai_residual"]["mean"])
    # The clamp only protects metric decoding during the first random epochs.
    # Its wide range is far outside the frozen training residual distribution.
    log_ai_residual = torch.clamp(log_ai_residual, -2.0, 2.0)
    return torch.cat(
        [vp_baseline + delta, ai_baseline * torch.exp(log_ai_residual)], dim=1
    )


def _baseline_prediction(batch: dict, calibration: dict, device: torch.device) -> torch.Tensor:
    target = calibration["target"]
    shape = batch["target"].shape
    prediction = torch.empty(shape, device=device, dtype=torch.float32)
    prediction[:, 0] = -float(target["delta_vp"]["mean"]) / float(
        target["delta_vp"]["std"]
    )
    prediction[:, 1] = -float(target["log_ai_residual"]["mean"]) / float(
        target["log_ai_residual"]["std"]
    )
    return prediction


def _metric_state() -> dict:
    return {
        region: {
            "abs": np.zeros(2),
            "sq": np.zeros(2),
            "true_abs": np.zeros(2),
            "true_sum": np.zeros(2),
            "true_sq_sum": np.zeros(2),
            "count": np.zeros(2),
        }
        for region in ("all", "reservoir")
    } | {"loss_sum": 0.0, "batches": 0.0}


def _update_metrics(
    state: dict,
    prediction_physical: torch.Tensor,
    target_physical: torch.Tensor,
    mask: torch.Tensor,
    reservoir_mask: torch.Tensor,
    loss: float,
) -> None:
    for region, region_mask in (("all", mask), ("reservoir", reservoir_mask)):
        valid = region_mask.expand_as(target_physical).bool()
        for channel in range(2):
            channel_valid = valid[:, channel]
            predicted = prediction_physical[:, channel][channel_valid].double()
            target = target_physical[:, channel][channel_valid].double()
            error = predicted - target
            state[region]["abs"][channel] += float(torch.abs(error).sum().cpu())
            state[region]["sq"][channel] += float(torch.square(error).sum().cpu())
            state[region]["true_abs"][channel] += float(torch.abs(target).sum().cpu())
            state[region]["true_sum"][channel] += float(target.sum().cpu())
            state[region]["true_sq_sum"][channel] += float(torch.square(target).sum().cpu())
            state[region]["count"][channel] += int(target.numel())
    state["loss_sum"] += float(loss)
    state["batches"] += 1.0


def _finish_metrics(state: dict) -> dict[str, float]:
    result = {"loss": state["loss_sum"] / max(state["batches"], 1.0)}
    for region in ("all", "reservoir"):
        for channel, name in enumerate(("Vp", "AI")):
            count = max(state[region]["count"][channel], 1.0)
            mae = state[region]["abs"][channel] / count
            rmse = np.sqrt(state[region]["sq"][channel] / count)
            relative = state[region]["abs"][channel] / max(
                state[region]["true_abs"][channel], 1.0
            )
            total_variance = state[region]["true_sq_sum"][channel] - (
                state[region]["true_sum"][channel] ** 2 / count
            )
            r2 = 1.0 - state[region]["sq"][channel] / max(total_variance, 1.0)
            scale = 1.0e-6 if name == "AI" else 1.0
            unit_name = "AI_MRayl" if name == "AI" else "Vp_mps"
            result[f"{region}_{unit_name}_MAE"] = float(mae * scale)
            result[f"{region}_{unit_name}_RMSE"] = float(rmse * scale)
            result[f"{region}_{name}_relative_MAE"] = float(relative)
            result[f"{region}_{name}_R2"] = float(r2)
    return result


def run_epoch(
    model,
    loader,
    device,
    calibration,
    loss_name: str,
    optimizer=None,
    *,
    amp: bool = False,
    physical_baseline: bool = False,
):
    training = optimizer is not None and not physical_baseline
    model.train(training)
    state = _metric_state()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and training)
    for batch in loader:
        image = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        target_physical = batch["target_physical"].to(device, non_blocking=True)
        vp_baseline = batch["vp_baseline"].to(device, non_blocking=True)
        ai_baseline = batch["ai_baseline"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        reservoir_mask = batch["reservoir_mask"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=amp):
                prediction = (
                    _baseline_prediction(batch, calibration, device)
                    if physical_baseline
                    else model(image)
                )
                loss = _masked_loss(prediction, target, mask, loss_name)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
        prediction_physical = _decode(
            prediction.detach(), vp_baseline, ai_baseline, calibration
        )
        _update_metrics(
            state,
            prediction_physical,
            target_physical,
            mask,
            reservoir_mask,
            float(loss.detach()),
        )
    return _finish_metrics(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release", type=Path, default=PROJECT / "outputs/NGH-Seis-v1.0-metadata"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "outputs/NGH-Seis-v1.0-acoustic-unet"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--model", choices=MODEL_NAMES, default="unet")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--loss", choices=("mae", "mse", "smoothl1"), default="smoothl1")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--protocol", choices=("iid", "structural-ood"), default="iid")
    parser.add_argument("--ood-fold", type=str)
    args = parser.parse_args()

    if args.protocol == "structural-ood" and not args.ood_fold:
        parser.error("--ood-fold is required when --protocol structural-ood")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.protocol == "iid":
        calibration = json.loads(
            (args.release / "task1_acoustic_calibration.json").read_text(encoding="utf-8")
        )
    else:
        calibration = json.loads(
            (args.release / "structural_ood_acoustic_calibration.json").read_text(
                encoding="utf-8"
            )
        )["folds"][args.ood_fold]
    loader_options = dict(
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        cache=args.cache,
        protocol=args.protocol,
        ood_fold=args.ood_fold,
    )
    train_loader = make_ngh_seis_acoustic_loader(args.release, "train", **loader_options)
    val_loader = make_ngh_seis_acoustic_loader(args.release, "validation", **loader_options)
    test_loader = make_ngh_seis_acoustic_loader(args.release, "test", **loader_options)
    model = build_benchmark_model(
        args.model,
        in_channels=2,
        out_channels=2,
        base_channels=args.base_channels,
        output_activation="identity",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=6
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    history, best, best_epoch = [], float("inf"), 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, calibration, args.loss, optimizer, amp=args.amp
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, device, calibration, args.loss, amp=args.amp
            )
        scheduler.step(val_metrics["loss"])
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(row)
        if val_metrics["loss"] < best:
            best, best_epoch = val_metrics["loss"], epoch
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                args.output / "best.pt",
            )
        (args.output / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(
            f"epoch={epoch:03d} train={train_metrics['loss']:.6f} "
            f"val={val_metrics['loss']:.6f} "
            f"VpMAE={val_metrics['all_Vp_mps_MAE']:.2f}m/s "
            f"AIMAE={val_metrics['all_AI_MRayl_MAE']:.4f}MRayl",
            flush=True,
        )

    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        test_metrics = run_epoch(model, test_loader, device, calibration, args.loss, amp=args.amp)
        baseline_metrics = run_epoch(
            model,
            test_loader,
            device,
            calibration,
            args.loss,
            amp=args.amp,
            physical_baseline=True,
        )
    result = {
        "task": "RTM-guided acoustic-property reconstruction",
        "model": args.model,
        "parameter_count": parameter_count,
        "input_channels": ["conditioned_RTM", "migration_Vp"],
        "network_targets": ["standardized_delta_Vp", "standardized_log_AI_residual"],
        "reported_outputs": ["Vp_true", "AI_true"],
        "mask_is_input": False,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "base_channels": args.base_channels,
        "loss": args.loss,
        "amp": args.amp,
        "seed": args.seed,
        "protocol": args.protocol,
        "ood_fold": args.ood_fold,
        "elapsed_s": time.time() - started,
        "test": test_metrics,
        "physical_baseline_test": {
            "Vp": "migration Vp (zero update)",
            "AI": "training-only log-linear AI(Vp_mig) fit",
            "metrics": baseline_metrics,
        },
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
