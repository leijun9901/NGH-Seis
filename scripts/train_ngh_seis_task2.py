"""Train an NGH-Seis v1.0 baseline for Task 2 saturation estimation."""

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

from src.datasets.ngh_seis_task2 import SATURATION_SCALES, make_ngh_seis_task2_loader  # noqa: E402
from src.models.benchmark_models import MODEL_NAMES, build_benchmark_model  # noqa: E402


def _masked_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, name: str) -> torch.Tensor:
    valid = mask.expand_as(target)
    if name == "mae":
        point = torch.abs(prediction - target)
    elif name == "mse":
        point = torch.square(prediction - target)
    elif name == "smoothl1":
        point = F.smooth_l1_loss(prediction, target, reduction="none", beta=0.03)
    else:
        raise ValueError(name)
    return (point * valid).sum() / valid.sum().clamp_min(1.0)


def _metric_state() -> dict[str, np.ndarray | float]:
    return {
        "abs": np.zeros(2), "sq": np.zeros(2), "count": np.zeros(2),
        "active_abs": np.zeros(2), "active_count": np.zeros(2),
        "intersection": np.zeros(2), "pred_positive": np.zeros(2),
        "true_positive": np.zeros(2), "loss_sum": 0.0, "batches": 0.0,
    }


def _update_metrics(state: dict, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss: float) -> None:
    scales = pred.new_tensor(SATURATION_SCALES)[None, :, None, None]
    pred_physical, target_physical = pred * scales, target * scales
    valid = mask.expand_as(target).bool()
    thresholds = pred.new_tensor([0.01, 0.003])[None, :, None, None]
    for channel in range(2):
        channel_valid = valid[:, channel]
        error = pred_physical[:, channel] - target_physical[:, channel]
        state["abs"][channel] += float(torch.abs(error[channel_valid]).sum().cpu())
        state["sq"][channel] += float(torch.square(error[channel_valid]).sum().cpu())
        state["count"][channel] += int(channel_valid.sum().cpu())
        active = channel_valid & (target_physical[:, channel] >= thresholds[0, channel])
        state["active_abs"][channel] += float(torch.abs(error[active]).sum().cpu())
        state["active_count"][channel] += int(active.sum().cpu())
        pred_pos = channel_valid & (pred_physical[:, channel] >= thresholds[0, channel])
        true_pos = active
        state["intersection"][channel] += int((pred_pos & true_pos).sum().cpu())
        state["pred_positive"][channel] += int(pred_pos.sum().cpu())
        state["true_positive"][channel] += int(true_pos.sum().cpu())
    state["loss_sum"] += float(loss)
    state["batches"] += 1.0


def _finish_metrics(state: dict) -> dict[str, float]:
    names = ["Sh", "Sg"]
    result = {"loss": state["loss_sum"] / max(state["batches"], 1.0)}
    for i, name in enumerate(names):
        result[f"{name}_MAE"] = state["abs"][i] / max(state["count"][i], 1.0)
        result[f"{name}_RMSE"] = np.sqrt(state["sq"][i] / max(state["count"][i], 1.0))
        result[f"{name}_active_MAE"] = state["active_abs"][i] / max(state["active_count"][i], 1.0)
        result[f"{name}_Dice"] = (
            2.0 * state["intersection"][i]
            / max(state["pred_positive"][i] + state["true_positive"][i], 1.0)
        )
    return {key: float(value) for key, value in result.items()}


def run_epoch(model, loader, device, loss_name: str, optimizer=None, *, amp: bool = False, zero=False):
    training = optimizer is not None and not zero
    model.train(training)
    state = _metric_state()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and training)
    for batch in loader:
        image = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=amp):
                prediction = torch.zeros_like(target) if zero else model(image)
                loss = _masked_loss(prediction, target, mask, loss_name)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
        _update_metrics(state, prediction.detach(), target, mask, float(loss.detach()))
    return _finish_metrics(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=PROJECT / "outputs/NGH-Seis_v1.0/metadata")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/task2_iid_unet")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--model", choices=MODEL_NAMES, default="unet")
    parser.add_argument(
        "--include-migration-vp",
        action="store_true",
        help="Use the unified two-channel [RTM, migration Vp] benchmark input",
    )
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
    if args.protocol == "structural-ood" and args.include_migration_vp:
        parser.error("two-channel OOD needs fold-specific Vp calibration")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    loader_options = dict(batch_size=args.batch_size, seed=args.seed,
                          num_workers=args.num_workers, cache=args.cache,
                          protocol=args.protocol, ood_fold=args.ood_fold,
                          include_migration_vp=args.include_migration_vp)
    train_loader = make_ngh_seis_task2_loader(args.release, "train", **loader_options)
    val_loader = make_ngh_seis_task2_loader(args.release, "validation", **loader_options)
    test_loader = make_ngh_seis_task2_loader(args.release, "test", **loader_options)
    input_channels = 2 if args.include_migration_vp else 1
    model = build_benchmark_model(
        args.model,
        in_channels=input_channels,
        out_channels=2,
        base_channels=args.base_channels,
        output_activation="sigmoid",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=6)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    history, best, best_epoch = [], float("inf"), 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, args.loss, optimizer, amp=args.amp)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, args.loss, amp=args.amp)
        scheduler.step(val_metrics["loss"])
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": train_metrics, "validation": val_metrics}
        history.append(row)
        if val_metrics["loss"] < best:
            best, best_epoch = val_metrics["loss"], epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, args.output / "best.pt")
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch={epoch:03d} train={train_metrics['loss']:.6f} val={val_metrics['loss']:.6f} ShMAE={val_metrics['Sh_MAE']:.5f} SgMAE={val_metrics['Sg_MAE']:.5f}", flush=True)
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        test_metrics = run_epoch(model, test_loader, device, args.loss, amp=args.amp)
        zero_metrics = run_epoch(model, test_loader, device, args.loss, amp=args.amp, zero=True)
    result = {
        "dataset": "NGH-Seis v1.0", "task": "Task 2", "model": args.model, "parameter_count": parameter_count,
        "input_channels": input_channels, "output_channels": 2, "mask_is_input": False,
        "input_variables": ["conditioned_RTM", "migration_Vp"] if args.include_migration_vp else ["conditioned_RTM"],
        "epochs": args.epochs, "best_epoch": best_epoch,
        "batch_size": args.batch_size, "base_channels": args.base_channels,
        "loss": args.loss, "amp": args.amp, "seed": args.seed,
        "protocol": args.protocol, "ood_fold": args.ood_fold,
        "elapsed_s": time.time() - started,
        "test": test_metrics, "zero_predictor_test": zero_metrics,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
