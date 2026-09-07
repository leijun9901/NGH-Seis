from scripts.run_ngh_seis_forward_rtm import _rtm_quality_gate


def test_true_model_correlation_is_diagnostic_not_admission_gate():
    metrics = {
        "horizontal_to_vertical_gradient_energy": 0.002,
        "near_vertical_fk_energy_fraction": 0.0002,
        "adjacent_trace_correlation_median": 0.98,
        "normalized_lateral_roughness": 0.05,
        "structure_aware_adjacent_trace_correlation_median": 0.985,
        "structure_aware_normalized_lateral_roughness": 0.04,
    }
    rules = {
        "smooth_horizontal_to_vertical_gradient_energy_max": 0.0045,
        "smooth_near_vertical_fk_energy_fraction_max": 0.001,
        "smooth_adjacent_trace_correlation_median_min": 0.96,
        "smooth_normalized_lateral_roughness_max": 0.12,
        "smooth_structure_aware_adjacent_trace_correlation_median_min": 0.97,
        "smooth_structure_aware_normalized_lateral_roughness_max": 0.08,
        "smooth_true_conditioned_correlation_min": 0.4,
    }

    without_true_model = _rtm_quality_gate(metrics, None, rules)
    with_low_true_model_correlation = _rtm_quality_gate(metrics, 0.01, rules)

    assert without_true_model == with_low_true_model_correlation
    assert "smooth_true_conditioned_correlation" not in with_low_true_model_correlation
    assert all(with_low_true_model_correlation.values())
