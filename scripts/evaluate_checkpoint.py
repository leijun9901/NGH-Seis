"""Evaluate an archived reference checkpoint with the frozen test protocol."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.models.benchmark_models import build_benchmark_model
from src.datasets.ngh_seis_task1 import make_ngh_seis_task1_loader
from src.datasets.ngh_seis_task2 import make_ngh_seis_task2_loader
from scripts.train_ngh_seis_task1 import run_epoch as evaluate_task1
from scripts.train_ngh_seis_task2 import run_epoch as evaluate_task2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", choices=("1", "2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = saved["args"]
    protocol = config.get("protocol", "iid")
    fold = config.get("ood_fold")
    first_task = args.task == "1"
    use_amp = bool(config.get("amp", False)) and device.type == "cuda"
    channels = 2 if first_task else 1
    model = build_benchmark_model(config["model"], in_channels=channels,
        out_channels=2, base_channels=config["base_channels"],
        output_activation="identity" if first_task else "sigmoid").to(device)
    model.load_state_dict(saved["model"])
    maker = make_ngh_seis_task1_loader if first_task else make_ngh_seis_task2_loader
    loader = maker(args.release, "test", batch_size=config["batch_size"],
        num_workers=0, seed=config["seed"], cache=False, protocol=protocol, ood_fold=fold)
    with torch.no_grad():
        if first_task:
            metrics = evaluate_task1(model, loader, device, loader.dataset.calibration,
                                    config["loss"], amp=use_amp)
        else:
            metrics = evaluate_task2(model, loader, device, config["loss"],
                                    amp=use_amp)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    result = dict(task=int(args.task), model=config["model"], protocol=protocol, ood_fold=fold,
        test_sample_count=len(loader.dataset), checkpoint_sha256=checkpoint_hash,
        checkpoint_epoch=saved["epoch"], archived_initial_learning_rate=config["learning_rate"],
        torch_version=torch.__version__, device=str(device),
        evaluation_amp=use_amp, archived_amp=bool(config.get("amp", False)),
        gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
        verification_scope="checkpoint test evaluation; training was not repeated",
        metrics=metrics)
    if args.expected:
        original = json.loads(args.expected.read_text(encoding="utf-8"))["test"]
        result["differences_from_archived_test"] = {k: metrics[k] - v for k, v in original.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
