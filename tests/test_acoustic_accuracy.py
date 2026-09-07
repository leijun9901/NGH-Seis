import numpy as np
from scipy.signal import hilbert

from src.forward.acoustic import build_marine_streamer_geometry, simulate_shot_gathers


def _homogeneous_direct_arrival(dx: float) -> float:
    nx = round(1000.0 / dx) + 1
    nz = 161
    vp = np.full((nz, nx), 1500.0, dtype=np.float32)
    rho = np.full_like(vp, 1025.0)
    geometry = {
        "shot_x_index": np.array([round(200.0 / dx)], dtype=np.int32),
        "shot_z_index": np.array([60], dtype=np.int32),
        "receiver_x_index": np.array([[round(800.0 / dx)]], dtype=np.int32),
        "receiver_z_index": np.array([[60]], dtype=np.int32),
    }
    dt = 0.0005
    gather = simulate_shot_gathers(
        vp, geometry, rho=rho,
        nt=1600, dt=dt, dx=dx, dz=5.0, f0=15.0, t0=0.05,
        boundary_width=20, free_surface=True,
    )[0, :, 0]
    envelope = np.abs(hilbert(gather))
    # The analytical peak is t0 + 600/1500 = 0.45 s.  Restricting the
    # search to its causal neighborhood avoids later free-surface energy.
    lo, hi = round(0.40 / dt), round(0.50 / dt)
    peak = int(np.argmax(envelope[lo:hi])) + lo
    return peak * dt


def test_fourth_order_direct_arrival_and_grid_convergence():
    analytical = 0.45
    coarse = _homogeneous_direct_arrival(10.0)
    fine = _homogeneous_direct_arrival(5.0)
    assert abs(coarse - analytical) <= 0.006
    assert abs(fine - analytical) <= 0.002
    assert abs(coarse - fine) <= 0.006


def test_formal_grid_time_step_direct_arrival():
    """The released 10 m/1.5 ms grid must preserve marine travel time."""
    vp = np.full((161, 301), 2000.0, dtype=np.float32)
    geometry = {
        "shot_x_index": np.array([150], dtype=np.int32),
        "shot_z_index": np.array([30], dtype=np.int32),
        "receiver_x_index": np.array([[90]], dtype=np.int32),
        "receiver_z_index": np.array([[30]], dtype=np.int32),
    }
    dt = 0.0015
    gather = simulate_shot_gathers(
        vp, geometry, nt=500, dt=dt, dx=10.0, dz=10.0,
        f0=15.0, t0=0.10, boundary_width=25, free_surface=False,
    )[0, :, 0]
    envelope = np.abs(hilbert(gather))
    lo, hi = round(0.35 / dt), round(0.45 / dt)
    peak_s = (int(np.argmax(envelope[lo:hi])) + lo) * dt
    analytical_s = 0.10 + 600.0 / 2000.0
    # The source and receiver sit below the 25-cell absorbing strip.
    assert abs(peak_s - analytical_s) <= 0.008


def test_one_cell_deep_marine_streamer_records_propagated_energy():
    """Regression test for a 10 m tow depth on the formal 10 m grid."""
    vp = np.full((121, 241), 1500.0, dtype=np.float32)
    geometry = {
        "shot_x_index": np.array([120], dtype=np.int32),
        "shot_z_index": np.array([1], dtype=np.int32),
        "receiver_x_index": np.array([[80]], dtype=np.int32),
        "receiver_z_index": np.array([[1]], dtype=np.int32),
    }
    dt = 0.0015
    gather = simulate_shot_gathers(
        vp, geometry, nt=450, dt=dt, dx=10.0, dz=10.0,
        f0=15.0, t0=0.10, boundary_width=25, free_surface=True,
    )[0, :, 0]
    assert float(np.max(np.abs(gather))) > 1e-8
    envelope = np.abs(hilbert(gather))
    peak_s = int(np.argmax(envelope)) * dt
    analytical_s = 0.10 + 400.0 / 1500.0
    assert abs(peak_s - analytical_s) <= 0.012


def test_absorbing_top_mode_is_not_identical_to_free_surface():
    vp = np.full((101, 201), 1500.0, dtype=np.float32)
    rho = np.full_like(vp, 1025.0)
    geometry = build_marine_streamer_geometry(
        nx=201, dx=10.0, dz=10.0, nshots=1, nreceivers=16,
        shot_start_m=1000.0, shot_spacing_m=50.0,
        near_offset_m=50.0, receiver_spacing_m=10.0,
        source_depth_m=310.0, receiver_depth_m=310.0,
    )
    free = simulate_shot_gathers(
        vp, geometry, rho=rho, nt=700, dt=0.001, dx=10.0, dz=10.0,
        f0=15.0, t0=0.1, free_surface=True,
    )
    absorbing = simulate_shot_gathers(
        vp, geometry, rho=rho, nt=700, dt=0.001, dx=10.0, dz=10.0,
        f0=15.0, t0=0.1, free_surface=False,
    )
    assert not np.array_equal(free, absorbing)
