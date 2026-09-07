from pathlib import Path

import numpy as np

from src.geology.blake_ew0008 import generate_blake_ew0008


PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "ngh_seis_geology.json"


def test_blake_generator_physical_shapes_and_ranges():
    arrays, meta = generate_blake_ew0008(CONFIG, 1, 81231)
    assert arrays["input_psdm"].shape == (512, 96)
    assert arrays["target_Sh"].shape == (512, 96)
    assert arrays["Vp_full"].shape == (1668, 95)
    assert np.isfinite(arrays["psdm_full"]).all()
    assert np.all(arrays["Sh_full"] >= 0) and arrays["Sh_full"].max() <= 0.22 + 1e-6
    assert np.all(arrays["Sg_full"] >= 0) and arrays["Sg_full"].max() <= 0.08 + 1e-6
    assert np.all(arrays["Sh_full"] + arrays["Sg_full"] <= 1.0)
    # A few percent of free gas can lower Vp below water velocity; reject only
    # nonphysical/numerically dangerous collapse, not the diagnostic anomaly.
    assert 1100 <= arrays["Vp_full"].min() < 1600
    assert arrays["Vp_full"].max() < 3200
    assert 380 <= meta["bsr_depth_mbsf_mean"] <= 520
    assert np.all(arrays["valid_mask"][:, :95] == 1)
    assert np.all(arrays["valid_mask"][:, 95] == 0)


def test_blake_generator_is_deterministic_and_styles_differ():
    a, ma = generate_blake_ew0008(CONFIG, 2, 991)
    b, mb = generate_blake_ew0008(CONFIG, 2, 991)
    c, mc = generate_blake_ew0008(CONFIG, 3, 992)
    assert ma == mb
    assert np.array_equal(a["input_psdm"], b["input_psdm"])
    assert ma["style"] != mc["style"]
    assert not np.array_equal(a["target_Sh"], c["target_Sh"])


def test_sh_and_sg_perturb_impedance_in_expected_directions():
    arrays, _ = generate_blake_ew0008(CONFIG, 4, 4422)
    h = arrays["Sh_full"] > 0.03
    g = arrays["Sg_full"] > 0.008
    assert h.any() and g.any()
    # Gas-bearing cells must not collapse below a numerically safe marine value.
    assert float(arrays["Vp_full"][g].min()) > 1100.0
    assert float(arrays["AI_full"].min()) > 1.0e6
