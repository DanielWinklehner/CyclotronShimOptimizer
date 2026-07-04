"""Tests for visualization.field_maps (median-plane field views).

The matplotlib path runs headless (Agg). The PyVista plane construction is
tested for point/scalar alignment; the full 3D viewer (needs radia + a GL
context) is exercised off-screen only if the environment supports it.
"""

import _testenv  # noqa: F401

import numpy as np

from PyPATools.field import Field
from cyclotron_optimizer.visualization.field_maps import (
    _field_plane_mm,
    build_field_plane,
    plot_median_plane_field,
)


def _make_field(nx=9, ny=7):
    """2D median-plane Field with an asymmetric, position-encoded Bz."""
    x = np.linspace(-0.4, 0.4, nx)   # meters
    y = np.linspace(-0.3, 0.3, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    bz = 1.0 + 2.0 * xx - 3.0 * yy + 4.0 * xx * yy
    values = {"x": np.zeros_like(bz), "y": np.zeros_like(bz), "z": bz}
    return Field.from_arrays({"x": x, "y": y}, values), x, y, bz


def test_field_plane_mm_units_and_shape():
    field, x, y, bz = _make_field()
    x_mm, y_mm, bz_out = _field_plane_mm(field)
    assert np.allclose(x_mm, x * 1e3)
    assert np.allclose(y_mm, y * 1e3)
    assert bz_out.shape == (len(x), len(y))
    assert np.allclose(bz_out, bz)


def test_plot_median_plane_field_matplotlib():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    field, x, y, bz = _make_field()
    fig, ax = plot_median_plane_field(field, show=False)
    assert fig is not None
    assert ax.get_xlabel() == "x (mm)"
    # filled contours were drawn
    assert len(ax.collections) > 0
    plt.close(fig)


def test_build_field_plane_scalar_alignment():
    try:
        import pyvista  # noqa: F401
    except ImportError:
        print("        [skipped: pyvista not installed]")
        return

    field, x, y, bz = _make_field()
    plane = build_field_plane(field)

    assert plane.n_points == len(x) * len(y)
    points = np.asarray(plane.points)
    scalars = np.asarray(plane["Bz (T)"])
    assert np.allclose(points[:, 2], 0.0)

    # every rendered point must carry the Bz of ITS OWN (x, y) location
    x_mm, y_mm = x * 1e3, y * 1e3
    ix = np.rint((points[:, 0] - x_mm[0]) / (x_mm[1] - x_mm[0])).astype(int)
    iy = np.rint((points[:, 1] - y_mm[0]) / (y_mm[1] - y_mm[0])).astype(int)
    assert np.allclose(scalars, bz[ix, iy]), "plane scalars misaligned with points"


def test_build_field_plane_subsampling():
    try:
        import pyvista  # noqa: F401
    except ImportError:
        print("        [skipped: pyvista not installed]")
        return

    field, x, y, bz = _make_field(nx=41, ny=33)
    plane = build_field_plane(field, max_points_per_axis=10)
    # 41 -> step 5 -> 9 points, 33 -> step 4 -> 9 points
    assert plane.n_points <= 10 * 10

    # subsampled scalars still aligned with their own coordinates
    points = np.asarray(plane.points)
    scalars = np.asarray(plane["Bz (T)"])
    x_mm, y_mm = x * 1e3, y * 1e3
    ix = np.rint((points[:, 0] - x_mm[0]) / (x_mm[1] - x_mm[0])).astype(int)
    iy = np.rint((points[:, 1] - y_mm[0]) / (y_mm[1] - y_mm[0])).astype(int)
    assert np.allclose(scalars, bz[ix, iy]), "subsampled scalars misaligned"
