"""Run a minimal end-to-end check for both NGH-Seis benchmark tasks."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.datasets.ngh_seis_task1 import NGHSeisTask1Dataset  # noqa: E402
from src.datasets.ngh_seis_task2 import NGHSeisTask2Dataset  # noqa: E402
from src.models.benchmark_models import MODEL_NAMES, build_benchmark_model  # noqa: E402


def _metadata_path(dataset: Path) -> Path:
    dataset = dataset.resolve()
    return dataset if dataset.name == "metadata" else dataset / "metadata"


def _one_update(model: torch.nn.Module, sample: dict[str, torch.Tensor]) -> tuple[int, ...]:
    image = sample["input"].unsqueeze(0)
    target = sample["target"].unsqueeze(0)
    mask = sample["mask"].unsqueeze(0).expand_as(target)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    prediction = model(image)
    loss = (F.smooth_l1_loss(prediction, target, reduction="none") * mask).sum()
    loss = loss / mask.sum().clamp_min(1.0)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite smoke-test loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return tuple(prediction.shape)


def _synthetic_samples() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Small deterministic tensors for installation and CI checks only."""
    generator = torch.Generator().manual_seed(20260824)
    height, width = 64, 32
    mask = torch.ones(1, height, width)
    task1 = {
        "input": torch.randn(2, height, width, generator=generator),
        "target": torch.randn(2, height, width, generator=generator),
        "mask": mask,
        "candidate_id": -1,
    }
    task2 = {
        "input": torch.randn(1, height, width, generator=generator),
        "target": torch.rand(2, height, width, generator=generator),
        "mask": mask,
        "candidate_id": -1,
    }
    return task1, task2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        help="Extracted NGH-Seis_v1.0 directory (or its metadata directory)",
    )
    parser.add_argument("--model", choices=(*MODEL_NAMES, "all"), default="unet")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Check model execution without released data (installation/CI only)",
    )
    args = parser.parse_args()

    if args.synthetic:
        task1, task2 = _synthetic_samples()
    else:
        if args.dataset is None:
            parser.error("dataset is required unless --synthetic is used")
        metadata = _metadata_path(args.dataset)
        task1 = NGHSeisTask1Dataset(metadata, "test")[0]
        task2 = NGHSeisTask2Dataset(metadata, "test")[0]

    model_names = MODEL_NAMES if args.model == "all" else (args.model,)
    results = {}
    for model_name in model_names:
        task1_shape = _one_update(
            build_benchmark_model(
                model_name,
                in_channels=2,
                out_channels=2,
                base_channels=8,
                output_activation="identity",
            ),
            task1,
        )
        task2_shape = _one_update(
            build_benchmark_model(
                model_name,
                in_channels=1,
                out_channels=2,
                base_channels=8,
                output_activation="sigmoid",
            ),
            task2,
        )
        results[model_name] = {
            "task1_output": task1_shape,
            "task2_output": task2_shape,
        }
        gc.collect()
    print(
        json.dumps(
            {
                "status": "PASS",
                "mode": "synthetic" if args.synthetic else "released-data",
                "sample_id": int(task1["candidate_id"]),
                "models": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
