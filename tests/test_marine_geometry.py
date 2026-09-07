import numpy as np
import pytest

from src.forward import build_marine_streamer_geometry


def test_vessel_source_streamer_order_when_sailing_right():
    geometry = build_marine_streamer_geometry(
        nx=641, dx=10.0, nshots=66, nreceivers=128,
        shot_start_m=2100.0, shot_spacing_m=50.0,
        near_offset_m=125.0, receiver_spacing_m=12.5,
        source_depth_m=10.0, receiver_depth_m=10.0,
        vessel_to_source_m=50.0, sailing_direction=1, dz=5.0,
    )
    # Increasing x points along the vessel's sailing direction.
    assert np.all(geometry["vessel_x_m"] > geometry["shot_x_m"])
    assert np.all(geometry["shot_x_m"][:, None] > geometry["receiver_x_m"])
    # Channel 1 is closest to the gun and channel 128 is farthest astern.
    assert np.all(np.diff(geometry["receiver_x_m"], axis=1) < 0)
    assert np.all(np.diff(geometry["offset_m"], axis=1) > 0)
    # Keep all active source/receiver positions outside the 300 m side sponge.
    assert geometry["receiver_x_m"].min() >= 300
    assert geometry["shot_x_m"].max() <= 6100
    assert geometry["vessel_x_m"].max() <= 6400

    # Approximate CMP aperture covers the declared 1.2-4.4 km target window.
    cmp_x = 0.5 * (geometry["shot_x_m"][:, None] + geometry["receiver_x_m"])
    assert cmp_x.min() <= 1250
    assert cmp_x.max() >= 5250


def test_invalid_formal_start_would_put_streamer_outside_model():
    with pytest.raises(ValueError, match="outside"):
        build_marine_streamer_geometry(
            nx=481, dx=10.0, nshots=32, nreceivers=128,
            shot_start_m=650.0, shot_spacing_m=75.0,
            near_offset_m=125.0, receiver_spacing_m=12.5,
            vessel_to_source_m=50.0, sailing_direction=1, dz=5.0,
        )


def test_direct_water_wave_increases_from_near_to_far_channel():
    geometry = build_marine_streamer_geometry(
        nx=481, dx=10.0, nshots=1, nreceivers=128,
        shot_start_m=2100.0, near_offset_m=125.0,
        receiver_spacing_m=12.5, source_depth_m=10.0,
        receiver_depth_m=10.0, vessel_to_source_m=50.0,
        sailing_direction=1, dz=5.0,
    )
    direct_time = 0.10 + geometry["offset_m"][0] / 1500.0
    assert np.all(np.diff(direct_time) > 0)
    assert direct_time[0] == pytest.approx(0.1833333, abs=1e-6)
    assert direct_time[-1] == pytest.approx(1.2416667, abs=1e-6)


def test_padded_formal_rtm_geometry_balances_target_midpoint_coverage():
    # The original target is x=1.2--5.2 km.  A temporary 300 m propagation
    # pad shifts it to x=1.5--5.5 km inside the FD model.
    geometry = build_marine_streamer_geometry(
        nx=701, dx=10.0, nshots=66, nreceivers=128,
        shot_start_m=2020.0, shot_spacing_m=60.0,
        near_offset_m=125.0, receiver_spacing_m=12.5,
        source_depth_m=10.0, receiver_depth_m=10.0,
        vessel_to_source_m=50.0, sailing_direction=1, dz=5.0,
    )
    midpoint = 0.5 * (geometry["shot_x_m"][:, None] + geometry["receiver_x_m"])
    fold, _ = np.histogram(midpoint, bins=np.linspace(1500.0, 5500.0, 201))
    assert geometry["receiver_x_m"].min() >= 300.0
    assert geometry["vessel_x_m"].max() <= 7000.0
    assert fold.min() >= 19
    assert fold[:10].mean() >= 0.5 * fold.max()
    assert fold[-10:].mean() >= 0.5 * fold.max()
