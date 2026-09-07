from pathlib import Path

import numpy as np

from scripts.generate_ngh_seis_geology import (
    _decorrelation_distance_m,
    _rowwise_lateral_correlation,
)
from src.datasets.field_aligned_view import extract_seafloor_relative_view
from src.geology.blake_ew0008 import generate_blake_ew0008


PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs/ngh_seis_geology.json"


def _view(field, seafloor):
    return extract_seafloor_relative_view(
        field, seafloor, source_dx_m=5.0, source_dz_m=5.0
    )[0]


def test_continuous_style_has_physical_lateral_scale():
    arrays, metadata = generate_blake_ew0008(
        CONFIG, 1, 20365547, include_proxy_image=False
    )
    sh = _view(arrays["Sh_full"], arrays["seafloor_m"])
    sg = _view(arrays["Sg_full"], arrays["seafloor_m"])
    assert metadata["seafloor_p95_absolute_slope"] <= 0.085
    assert _rowwise_lateral_correlation(sh, slice(124, 244), 4) >= 0.30
    assert _rowwise_lateral_correlation(sg, slice(214, 334), 4) >= 0.25
    assert _decorrelation_distance_m(sh, slice(124, 244)) >= 150.0
    assert _decorrelation_distance_m(sg, slice(214, 334)) >= 150.0


def test_discontinuous_style_retains_a_real_gap():
    arrays, metadata = generate_blake_ew0008(
        CONFIG, 3, 20575005, include_proxy_image=False
    )
    sh = _view(arrays["Sh_full"], arrays["seafloor_m"])
    lateral_peak = np.max(sh[124:244, :95], axis=0)
    active = lateral_peak >= 0.30 * float(lateral_peak.max())
    transitions = np.diff(np.pad(active.astype(np.int8), (1, 1)))
    segment_count = int(np.count_nonzero(transitions == 1))
    assert metadata["style"] == "discontinuous_bsr"
    assert 2 <= segment_count <= 3
    assert np.count_nonzero(~active) >= 8
