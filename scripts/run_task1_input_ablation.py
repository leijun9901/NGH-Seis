"""Fair IID Task 1 input ablation with absolute train-only target scaling."""

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
from torch.utils.data import DataLoader, Dataset


WORKSPACE = Path(__file__).resolve().parents[1]
CODE = WORKSPACE
sys.path.insert(0, str(CODE))

from src.datasets.ngh_seis_task1 import NGHSeisTask1Dataset  # noqa: E402
from src.models.benchmark_models import build_benchmark_model  # noqa: E402


class AblationDataset(Dataset):
    def __init__(self, release: Path, split: str, mode: str, calibration: dict):
        self.base = NGHSeisTask1Dataset(release, split, augment=split == "train", cache=False)
        self.mode = mode
        target = calibration["target"]
        self.vp_mean = float(target["true_vp"]["mean"])
        self.vp_std = float(target["true_vp"]["std"])
        self.log_ai_mean = float(target["log_ai"]["mean"])
        self.log_ai_std = float(target["log_ai"]["std"])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        image = item["input"].clone()
        if self.mode == "rtm_only":
            image[1].zero_()
        elif self.mode == "smooth_v_only":
            image[0].zero_()
        elif self.mode != "rtm_plus_smooth_v":
            raise ValueError(self.mode)
        physical = item["target_physical"]
        target = torch.stack(
            [
                (physical[0] - self.vp_mean) / self.vp_std,
                (torch.log(physical[1].clamp_min(1.0)) - self.log_ai_mean) / self.log_ai_std,
            ]
        )
        return {"input": image, "target": target, "physical": physical, "mask": item["mask"]}


def loader(release, split, mode, calibration, batch_size, seed, workers):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        AblationDataset(release, split, mode, calibration),
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=generator,
    )


def run_epoch(model, data, device, calibration, optimizer=None, amp=False):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    batches = 0
    abs_sum = np.zeros(2, dtype=np.float64)
    count = np.zeros(2, dtype=np.int64)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and training)
    target_cal = calibration["target"]
    for batch in data:
        image = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        physical = batch["physical"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=amp):
                prediction = model(image)
                valid = mask.expand_as(target)
                point = F.smooth_l1_loss(prediction, target, reduction="none", beta=0.1)
                loss = (point * valid).sum() / valid.sum().clamp_min(1.0)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
        prediction_f32 = prediction.float()
        decoded = torch.stack(
            [
                prediction_f32[:, 0] * float(target_cal["true_vp"]["std"]) + float(target_cal["true_vp"]["mean"]),
                torch.exp(
                    (prediction_f32[:, 1] * float(target_cal["log_ai"]["std"]) + float(target_cal["log_ai"]["mean"])).clamp(12.0, 17.0)
                ),
            ],
            dim=1,
        )
        valid = mask.expand_as(decoded).bool()
        for channel in range(2):
            selected = valid[:, channel]
            abs_sum[channel] += float(torch.abs(decoded[:, channel][selected] - physical[:, channel][selected]).sum().cpu())
            count[channel] += int(selected.sum().cpu())
        loss_sum += float(loss.detach().cpu())
        batches += 1
    return {
        "loss": loss_sum / batches,
        "Vp_MAE_m_s": float(abs_sum[0] / count[0]),
        "AI_MAE_MRayl": float(abs_sum[1] / count[1] / 1e6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True,
                        help="Metadata directory of the extracted public dataset")
    parser.add_argument("--mode", choices=("smooth_v_only", "rtm_only", "rtm_plus_smooth_v"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calibration = json.loads((args.release / "task1_acoustic_calibration.json").read_text(encoding="utf-8"))
    loaders = {
        split: loader(args.release, split, args.mode, calibration, args.batch_size, args.seed, args.num_workers)
        for split in ("train", "validation", "test")
    }
    model = build_benchmark_model(
        "unet", in_channels=2, out_channels=2, base_channels=args.base_channels, output_activation="identity"
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=6)
    args.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    best_epoch = 0
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, loaders["train"], device, calibration, optimizer, args.amp)
        with torch.no_grad():
            validation = run_epoch(model, loaders["validation"], device, calibration, amp=args.amp)
        scheduler.step(validation["loss"])
        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": train, "validation": validation})
        if validation["loss"] < best:
            best = validation["loss"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch}, args.output / "best.pt")
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if epoch == 1 or epoch % 5 == 0:
            print(f"{args.mode} epoch={epoch:03d} val={validation['loss']:.6f} Vp={validation['Vp_MAE_m_s']:.2f} AI={validation['AI_MAE_MRayl']:.4f}", flush=True)

    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        test = run_epoch(model, loaders["test"], device, calibration, amp=args.amp)
    result = {
        "dataset": "NGH-Seis v1.0",
        "task": "Task 1 IID input ablation",
        "method": "U-Net",
        "input_mode": args.mode,
        "target_parameterization": "absolute Vp and log(AI), standardized using IID training partition only",
        "loss": "Smooth L1, beta=0.1",
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "base_channels": args.base_channels,
        "learning_rate": args.learning_rate,
        "optimizer": "AdamW, weight_decay=1e-4",
        "scheduler": "ReduceLROnPlateau, factor=0.5, patience=6",
        "seed": args.seed,
        "augmentation": "random horizontal flip during training",
        "mask": "loss_mask used for loss and metrics; not an input",
        "amp": args.amp,
        "device": str(device),
        "elapsed_s": time.time() - started,
        "test": test,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
