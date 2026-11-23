"""Controlled migration checks for reviewer 1, stored outside the release."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))

from scripts.run_marine_rtm import _migrate_batches, _sum_gradients, _select_25_shots
from scripts.marine_observations import build_geometry, build_blake_migration_model, _extract_view
from src.datasets.rtm_scaling import condition_rtm_unscaled
from src.forward.benchmark_workflow import preprocess_observed_gather

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True,help='Complete extracted frozen release root')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    ROOT=args.release.resolve();OUT=args.output.resolve()
    if OUT.is_relative_to(ROOT):parser.error('Output must be outside the frozen release')
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    config = json.loads((ROOT / "configs/marine_streamer.json").read_text(encoding="utf-8"))
    geology_config = json.loads((ROOT / "configs/ngh_seis_v1_generator.json").read_text(encoding="utf-8"))
    manifest = {r["candidate_id"]: r for r in map(json.loads,
        (ROOT / "metadata/scientific_data_manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    device = torch.device("cuda")
    geometry = _select_25_shots(build_geometry(config, device), config)
    jobs = [(1, "smooth_replay"), *[(i, "true_vp_constant_density") for i in range(1, 6)],
            (20, "smooth_replay"), (20, "extended_direct_mute")]
    results = []
    for cid, role in jobs:
        target = OUT / f"sample_{cid:04d}_{role}.npz"
        if target.exists():
            results.append(json.loads(target.with_suffix(".json").read_text(encoding="utf-8")))
            continue
        record = manifest[cid]
        for field in ("geology", "observations", "pair"):
            assert sha256(ROOT / record[field]) == record["files"][field]["sha256"]
        with np.load(ROOT / record["geology"], allow_pickle=False) as f:
            vp_true, seafloor = f["Vp_full"], f["seafloor_m"]
        with np.load(ROOT / record["observations"], allow_pickle=False) as f:
            processed = f["processed_gathers"]
            raw = f["raw_gathers"] if role == "extended_direct_mute" else None
        with np.load(ROOT / record["pair"], allow_pickle=False) as f:
            released = f["input_rtm_conditioned_unscaled"]
            mask = f["loss_mask"]
            migration_view = f["migration_Vp"]
        smooth_vp, density, _ = build_blake_migration_model(seafloor,
            nz=vp_true.shape[0], nx=vp_true.shape[1], dz_m=5.0, seed=record["migration_seed"])
        replay_velocity, _, _ = _extract_view(smooth_vp, seafloor, geology_config)
        assert np.array_equal(replay_velocity, migration_view)
        velocity = vp_true if role == "true_vp_constant_density" else smooth_vp
        if role == "extended_direct_mute":
            rec = config["recording"]
            processed = np.stack([preprocess_observed_gather(gather, geometry["offset_m"],
                dt=rec["dt_s"], t0=rec["source_delay_s"], low_hz=3.0, high_hz=28.0,
                gain_power=.65, direct_mute_extra_s=.25, coherent_noise_lowpass_hz=0.)
                for gather in raw])
        folder = OUT / f"gradients_{cid:04d}_{role}"
        print(f"START candidate={cid} role={role}", flush=True)
        state = _migrate_batches(processed, velocity, density, config, geometry, device,
            folder, model_role=role, batch_size=4, stop_after_batches=None)
        assert state["complete"]
        full, runtime = _sum_gradients(folder, 25, 4)
        view, support, _ = _extract_view(full, seafloor, geology_config)
        conditioned = condition_rtm_unscaled(view, mask)
        relative = float(np.linalg.norm((conditioned - released)[mask.astype(bool)]) /
                         np.linalg.norm(released[mask.astype(bool)]))
        summary = dict(candidate_id=cid, family=record["style"], role=role,
            migration_runtime_s=runtime, density_kg_m3=1800.,
            gpu=torch.cuda.get_device_name(), relative_difference_to_released=relative,
            observed_source="frozen processed gathers" if raw is None else "raw gathers reprocessed with 0.25 s extra mute",
            input_sha256={field: record["files"][field]["sha256"] for field in ("geology", "observations", "pair")})
        if role == "smooth_replay" and relative > .01:
            raise RuntimeError(f"Replay differs from frozen release by {relative:.4g}")
        np.savez_compressed(target, gradient_view=view, conditioned=conditioned,
                            valid_mask=support, loss_mask=mask)
        target.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        results.append(summary)
        print("DONE " + json.dumps(summary), flush=True)
    (OUT / "run_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
