import numpy as np

from src.forward.benchmark_workflow import (
    add_observation_noise,
    build_independent_migration_model,
    build_valid_imaging_mask,
    measure_rtm_directional_artifacts,
    preprocess_observed_gather,
    standardize_rtm,
)


def test_migration_prior_is_deterministic_and_has_no_reservoir_input():
    seafloor = np.full(80, 120.0, dtype=np.float32)
    first_vp, first_rho, first_meta = build_independent_migration_model(
        seafloor, nz=60, nx=80, dz=5.0, seed=17,
    )
    second_vp, second_rho, second_meta = build_independent_migration_model(
        seafloor, nz=60, nx=80, dz=5.0, seed=17,
    )
    np.testing.assert_array_equal(first_vp, second_vp)
    np.testing.assert_array_equal(first_rho, second_rho)
    assert first_meta == second_meta
    assert 1660.0 <= first_meta["sediment_top_velocity_mps"] <= 1720.0
    assert 0.55 <= first_meta["compaction_gradient_s_inv"] <= 0.75
    assert 2.0e-4 <= first_meta["curvature_inv_m_s"] <= 4.0e-4
    # The far-water column remains exactly water velocity, while the final
    # 60 m above the seabed is deliberately blended into the sediment trend
    # to avoid making the migration background itself a hard reflector.
    assert np.all(first_vp[:12] == 1500.0)
    assert float(np.mean(first_vp[23])) > 1500.0
    assert np.all(first_rho == 1800.0)
    assert first_meta["water_bottom_transition_m"] == 60.0
    assert first_meta["constant_density_migration"] is True
    assert np.all(np.isfinite(first_vp)) and np.all(first_vp > 0.0)


def test_noise_is_reproducible_and_matches_requested_rms():
    gather = np.ones((500, 12), dtype=np.float32)
    first, meta = add_observation_noise(gather, seed=4, snr_db=20.0, dt=0.001)
    second, _ = add_observation_noise(gather, seed=4, snr_db=20.0, dt=0.001)
    np.testing.assert_array_equal(first, second)
    measured = float(np.sqrt(np.mean((first - gather) ** 2)))
    np.testing.assert_allclose(measured, meta["added_noise_rms"], rtol=2e-5)


def test_preprocessing_and_standardization_are_finite_and_masked():
    rng = np.random.default_rng(8)
    gather = rng.normal(size=(1000, 20)).astype(np.float32)
    processed = preprocess_observed_gather(
        gather, np.linspace(125.0, 600.0, 20), dt=0.001, t0=0.1,
    )
    assert processed.shape == gather.shape
    assert np.all(np.isfinite(processed))
    assert np.allclose(processed[:180, 0], 0.0)

    image = rng.normal(size=(40, 50)).astype(np.float32)
    source = np.ones_like(image)
    receiver = np.ones_like(image)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[10:30, 5:45] = 1
    standardized, joint = standardize_rtm(image, source, receiver, mask)
    assert standardized.shape == image.shape and joint.shape == image.shape
    assert np.all(standardized[mask == 0] == 0.0)
    assert np.max(np.abs(standardized)) <= 1.0


def test_valid_mask_excludes_water_bottom_and_lateral_edges():
    illumination = np.ones((100, 80), dtype=np.float32)
    seafloor = np.full(140, 200.0, dtype=np.float32)
    mask = build_valid_imaging_mask(
        seafloor, illumination, x_start_index=20, dz=5.0,
        water_bottom_buffer_m=25.0, model_bottom_buffer_m=50.0,
        lateral_buffer_cells=5,
    )
    assert np.all(mask[:45] == 0)
    assert np.all(mask[90:] == 0)
    assert np.all(mask[:, :5] == 0) and np.all(mask[:, -5:] == 0)
    assert np.all(mask[50:80, 10:70] == 1)


def test_deeper_propagation_grid_keeps_released_image_bottom_valid():
    illumination = np.ones((400, 80), dtype=np.float32)
    seafloor = np.full(140, 1200.0, dtype=np.float32)
    mask = build_valid_imaging_mask(
        seafloor, illumination, x_start_index=20, dz=5.0,
        model_bottom_buffer_m=150.0, propagation_bottom_m=2300.0,
        lateral_buffer_cells=5,
    )
    assert np.all(mask[-1, 5:-5] == 1)


def test_directional_artifact_metric_separates_columns_from_reflectors():
    z = np.arange(80, dtype=np.float32)[:, None]
    x = np.arange(100, dtype=np.float32)[None, :]
    mask = np.ones((80, 100), dtype=np.uint8)
    horizontal_reflectors = np.broadcast_to(np.sin(z / 3.0), mask.shape)
    vertical_columns = np.broadcast_to(np.sin(x / 3.0), mask.shape)
    reflector_qc = measure_rtm_directional_artifacts(horizontal_reflectors, mask)
    column_qc = measure_rtm_directional_artifacts(vertical_columns, mask)
    assert column_qc["near_vertical_fk_energy_fraction"] > 0.9
    assert reflector_qc["near_vertical_fk_energy_fraction"] < 0.1
    assert column_qc["horizontal_to_vertical_gradient_energy"] > 10.0
    assert reflector_qc["horizontal_to_vertical_gradient_energy"] < 0.1
