import json
from pathlib import Path

import numpy as np
import torch

from scripts.run_blake_rtm_shot_density_gate import (
    _migration_batches,
    build_blake_migration_model,
    build_geometry,
)


PROJECT = Path(__file__).resolve().parents[1]


def test_blake_marine_geometry_is_grid_exact_and_trailing():
    config = json.loads(
        (PROJECT / "configs/marine_streamer.json").read_text(
            encoding="utf-8"
        )
    )
    geometry = build_geometry(config, torch.device("cpu"))
    assert geometry["source_locations"].shape == (49, 1, 2)
    assert geometry["receiver_locations"].shape == (49, 128, 2)
    assert np.array_equal(geometry["subset25_indices"], np.arange(0, 49, 2))
    assert np.all(geometry["receiver_x_m"] < geometry["shot_x_m"][:, None])
    assert np.isclose(geometry["offset_m"][0], 100.0)
    assert np.isclose(geometry["offset_m"][-1], 2005.0)
    assert np.allclose(geometry["receiver_x_m"] % 5.0, 0.0)
    formal = config["release_acquisition"]
    assert formal["shot_count"] == 25
    assert formal["effective_shot_spacing_m"] == 150.0
    assert formal["parent_49_zero_based_indices"] == list(range(0, 49, 2))


def test_blake_migration_model_uses_only_bathymetry_and_is_smooth():
    nx, nz, dz = 1229, 1001, 5.0
    seafloor = 2780.0 + 20.0 * np.sin(np.linspace(0.0, 3.0, nx))
    vp1, rho1, metadata1 = build_blake_migration_model(
        seafloor, nz=nz, nx=nx, dz_m=dz, seed=91
    )
    vp2, rho2, metadata2 = build_blake_migration_model(
        seafloor, nz=nz, nx=nx, dz_m=dz, seed=91
    )
    assert np.array_equal(vp1, vp2)
    assert np.array_equal(rho1, rho2)
    assert metadata1 == metadata2
    assert vp1.shape == (nz, nx)
    assert np.isfinite(vp1).all()
    assert 1480.0 <= float(vp1.min()) <= 1520.0
    assert float(vp1.max()) <= 2350.0
    assert np.all(rho1 == 1800.0)
    assert metadata1["reservoir_scale_anomalies_included"] is False


def test_parity_preserving_batches_recover_25_and_49_shot_sets():
    batches = _migration_batches(4)
    even = np.concatenate([indices for name, _, indices in batches if name == "even25"])
    odd = np.concatenate([indices for name, _, indices in batches if name == "odd24"])
    assert np.array_equal(even, np.arange(0, 49, 2))
    assert np.array_equal(odd, np.arange(1, 49, 2))
    assert np.array_equal(np.sort(np.concatenate([even, odd])), np.arange(49))
