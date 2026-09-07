import numpy as np

from src.datasets.formal_pair import (
    crop_and_pool_continuous, crop_and_pool_targets,
    resample_gathers_to_network, select_uniform_shots,
)


def test_uniform_shot_selection_spans_candidate_line():
    selected = select_uniform_shots(66, 8)
    assert selected.tolist() == [0, 9, 19, 28, 37, 46, 56, 65]


def test_time_resampling_shape_and_finiteness():
    rng = np.random.default_rng(3)
    gathers = rng.normal(size=(8, 2200, 4)).astype(np.float32)
    output = resample_gathers_to_network(gathers)
    assert output.shape == (8, 330, 4)
    assert np.isfinite(output).all()


def test_target_crop_and_area_average():
    sh = np.zeros((201, 641), dtype=np.float32)
    sg = np.zeros_like(sh)
    sh[:200, 120:520] = 0.4
    sg[:200, 120:520] = 0.1
    target = crop_and_pool_targets(sh, sg)
    assert target.shape == (2, 200, 200)
    np.testing.assert_allclose(target[0], 0.4)
    np.testing.assert_allclose(target[1], 0.1)


def test_continuous_field_crop_and_area_average():
    field = np.zeros((201, 641), dtype=np.float32)
    field[:200, 120:520] = 1800.0
    output = crop_and_pool_continuous(field)
    assert output.shape == (200, 200)
    np.testing.assert_allclose(output, 1800.0)
