from pathlib import Path

import numpy as np

from src.geology.blake_ew0008 import (
    _scale_to_soft_cap,
    generate_blake_ew0008,
)


PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs/ngh_seis_geology.json"


def test_soft_cap_matches_mean_without_a_constant_plateau():
    y, x = np.mgrid[-1.0:1.0:121j, -1.0:1.0:161j]
    field = np.exp(0.7 * x - 0.4 * y + 0.25 * np.sin(5.0 * x))
    mask = (x * x + y * y) < 0.92
    result = _scale_to_soft_cap(field, mask, mean=0.05, cap=0.20)
    positive = result[mask]
    assert np.isclose(float(positive.mean()), 0.05, atol=1.0e-6)
    assert float(positive.max()) < 0.20
    assert float(np.mean(positive >= 0.99 * positive.max())) < 0.08
    assert np.unique(positive).size > 1000


def test_saturation_and_phase_position_gates():
    coexistence_count = 0
    for sample_index in range(1, 6):
        seed = 20_260_818 + sample_index * 104_729
        arrays, metadata = generate_blake_ew0008(
            CONFIG, sample_index, seed, include_proxy_image=False
        )
        sh = arrays["Sh_full"]
        sg = arrays["Sg_full"]
        z = arrays["z_m"][:, None]
        mbsf = z - arrays["seafloor_m"][None, :]
        distance_to_bsr = z - arrays["bsr_m"][None, :]
        gas = sg > 0.0
        overlap = (sh > 0.0) & gas

        assert np.any(sh[(mbsf >= 185.0) & (mbsf <= 260.0)] > 0.0)
        assert not np.any(sg[distance_to_bsr < 0.0] > 0.0)
        assert float(sh.max()) <= 0.22
        assert float(sg.max()) <= 0.065
        assert float(np.max(sh + sg)) <= 0.25 + 1.0e-7
        assert 30.0 <= metadata["gas_zone_thickness_m"] <= 150.0
        assert metadata["hydrate_fraction_within_one_percent_of_max"] <= 0.08
        assert metadata["gas_fraction_within_one_percent_of_max"] <= 0.08
        gas_vp_p01 = float(np.quantile(arrays["Vp_full"][gas], 0.01))
        assert 1250.0 <= gas_vp_p01 <= 1750.0

        if metadata["coexistence_enabled"]:
            coexistence_count += 1
            assert np.any(overlap)
            assert np.all(distance_to_bsr[overlap] >= 0.0)
            assert np.all(
                distance_to_bsr[overlap]
                <= metadata["coexistence_thickness_m"] + 1.0e-3
            )
        else:
            assert not np.any(overlap)

    assert coexistence_count >= 1
