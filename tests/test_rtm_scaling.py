import numpy as np

from src.datasets.rtm_scaling import (
    RTMAmplitudeCalibration,
    apply_training_scale,
    condition_rtm_unscaled,
    display_normalize_rtm,
)


def test_unscaled_conditioning_preserves_between_sample_amplitude_ratio():
    rng = np.random.default_rng(42)
    image = rng.normal(size=(64, 48)).astype(np.float32)
    mask = np.ones_like(image, dtype=bool)
    a = condition_rtm_unscaled(image, mask)
    b = condition_rtm_unscaled(3.0 * image, mask)
    assert np.allclose(b, 3.0 * a, rtol=2e-5, atol=2e-5)


def test_per_sample_display_is_explicitly_not_quantitative():
    rng = np.random.default_rng(7)
    image = rng.normal(size=(40, 32)).astype(np.float32)
    mask = np.ones_like(image, dtype=bool)
    a, scale_a = display_normalize_rtm(image, mask)
    b, scale_b = display_normalize_rtm(3.0 * image, mask)
    assert np.allclose(a, b, rtol=1e-6, atol=1e-6)
    assert np.isclose(scale_b / scale_a, 3.0, rtol=1e-6)


def test_frozen_training_scale_preserves_unclipped_ratio_and_mask():
    image = np.asarray([[0.0, 1.0], [-2.0, 0.25]], dtype=np.float32)
    mask = np.asarray([[False, True], [True, True]])
    calibration = RTMAmplitudeCalibration(
        scale=2.0, percentile=99.5, clip_multiple=4.0,
        fitted_split="train", field_name="rtm_conditioned_unscaled",
        sample_count=10, value_count=100, seed=1,
    )
    a = apply_training_scale(image, mask, calibration)
    b = apply_training_scale(3.0 * image, mask, calibration)
    assert a[0, 0] == 0.0 and b[0, 0] == 0.0
    assert np.allclose(b[mask], 3.0 * a[mask])
