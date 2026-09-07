import numpy as np

from src.datasets.field_aligned_view import extract_seafloor_relative_view


def test_field_aligned_view_shape_padding_and_coordinates():
    dz = dx = 5.0
    nz, nx = 1001, 1229
    z = np.arange(nz, dtype=np.float32)[:, None] * dz
    x = np.arange(nx, dtype=np.float32)[None, :] * dx
    field = z + 0.01 * x
    seafloor = np.full(nx, 2800.0, dtype=np.float32)
    view, valid, metadata = extract_seafloor_relative_view(
        field, seafloor, source_dx_m=dx, source_dz_m=dz,
        antialias_lateral=True,
    )
    assert view.shape == (512, 96)
    assert valid.shape == view.shape
    assert np.all(valid[:, :95] == 1) and np.all(valid[:, 95] == 0)
    assert np.all(view[:, 95] == 0.0)
    assert metadata["seafloor_row"] == 64
    expected = 2800.0 + 0.01 * 1310.0
    assert np.isclose(view[64, 0], expected, atol=0.2)


def test_field_aligned_view_rejects_insufficient_depth():
    field = np.zeros((600, 1229), dtype=np.float32)
    seafloor = np.full(1229, 2800.0, dtype=np.float32)
    try:
        extract_seafloor_relative_view(
            field, seafloor, source_dx_m=5.0, source_dz_m=5.0
        )
    except ValueError as error:
        assert "depth" in str(error)
    else:
        raise AssertionError("insufficient model depth was not rejected")
