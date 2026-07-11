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
    _make_toggle_callbacks,
    _nice_ticks,
    _polydata_bounds,
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


# ---------------------------------------------------------------------------
# Interactive toggle key callbacks (pure logic, tested with fakes -- no GL)
# ---------------------------------------------------------------------------

class _FakeActor:
    """Field actor: supports the pyvista.Actor `.visibility` property."""
    def __init__(self):
        self.visibility = True


class _FakeAxesActor:
    """Ticked-bounds actor: like VTK's CubeAxesActor, `.visibility` cannot be
    assigned, so the toggle must fall back to SetVisibility()."""
    __slots__ = ("_vis",)

    def __init__(self):
        self._vis = 1

    def SetVisibility(self, v):
        self._vis = int(bool(v))

    def GetVisibility(self):
        return self._vis


class _FakeScalarBar:
    def __init__(self):
        self.visible = 1

    def SetVisibility(self, v):
        self.visible = v


class _FakePlotter:
    def __init__(self):
        self.scalar_bars = {"Bz (T)": _FakeScalarBar()}
        self.renders = 0

    def render(self):
        self.renders += 1


def test_toggle_field_hides_and_shows_actors_and_scalar_bar():
    p = _FakePlotter()
    actors = [_FakeActor(), _FakeActor()]
    toggle_field, _toggle_axes, state = _make_toggle_callbacks(p, actors, [], "Bz (T)")

    assert state["field"] is True
    toggle_field()                                   # first press -> hide
    assert all(a.visibility is False for a in actors)
    assert p.scalar_bars["Bz (T)"].visible is False
    assert state["field"] is False
    assert p.renders == 1

    toggle_field()                                   # second press -> show
    assert all(a.visibility is True for a in actors)
    assert p.scalar_bars["Bz (T)"].visible is True
    assert state["field"] is True
    assert p.renders == 2


def test_toggle_axes_flips_bounds_actor_via_setvisibility():
    p = _FakePlotter()
    field = [_FakeActor()]
    axes = [_FakeAxesActor()]                         # CubeAxesActor-like (SetVisibility)
    toggle_field, toggle_axes, state = _make_toggle_callbacks(p, field, axes, "Bz (T)")

    toggle_axes()                                    # hide the ticked axes only
    assert axes[0].GetVisibility() == 0
    assert field[0].visibility is True               # field untouched (independent)
    assert state["field"] is True
    toggle_axes()                                    # show again
    assert axes[0].GetVisibility() == 1
    assert p.renders == 2


def test_toggle_field_survives_missing_scalar_bar():
    # a fallback render may not register the scalar bar under that title
    p = _FakePlotter()
    p.scalar_bars = {}                               # lookup will KeyError -> swallowed
    toggle_field, _toggle_axes, _state = _make_toggle_callbacks(
        p, [_FakeActor()], [_FakeAxesActor()], "Bz (T)")
    toggle_field()                                   # must not raise
    assert p.renders == 1


# ---------------------------------------------------------------------------
# Nice round tick bounds (_nice_ticks) + model bounds (_polydata_bounds)
# ---------------------------------------------------------------------------

def _ticks(lo, hi, n):
    return np.linspace(lo, hi, n)


def test_nice_ticks_symmetric_hundreds():
    lo, hi, n = _nice_ticks(-400.0, 400.0)
    assert (lo, hi) == (-400.0, 400.0)
    t = _ticks(lo, hi, n)
    assert np.allclose(t, [-400, -200, 0, 200, 400])       # multiples of 200
    assert 0.0 in t                                        # origin is a tick


def test_nice_ticks_rounds_outward_to_round_step():
    lo, hi, n = _nice_ticks(-285.0, 55.0)                  # machine z-extent
    t = _ticks(lo, hi, n)
    assert lo <= -285.0 and hi >= 55.0                     # brackets the data
    step = t[1] - t[0]
    assert np.isclose(step % 50, 0) or np.isclose(step % 100, 0)  # round step
    assert np.allclose(t, np.round(t / step) * step)       # every tick on the step
    assert 0.0 in t


def test_nice_ticks_small_scale_uses_fives_or_tens():
    lo, hi, n = _nice_ticks(0.0, 12.0)
    t = _ticks(lo, hi, n)
    assert lo == 0.0 and hi >= 12.0
    step = t[1] - t[0]
    assert step in (2.0, 5.0)                              # tens-scale -> 2 or 5
    assert np.allclose(t, np.round(t))


def test_nice_ticks_degenerate_axis_is_safe():
    lo, hi, n = _nice_ticks(5.0, 5.0)                      # zero-width
    assert n >= 2 and lo == 5.0 and hi == 5.0
    lo2, hi2, n2 = _nice_ticks(10.0, 3.0)                  # inverted
    assert n2 >= 2


def test_polydata_bounds_and_none():
    pgn = {"vertices": list(np.array(
        [[400, 400, -285], [-400, -400, 55], [100, 0, 0]], dtype=float).ravel())}
    b = _polydata_bounds(pgn)
    assert b == [-400.0, 400.0, -400.0, 400.0, -285.0, 55.0]
    assert _polydata_bounds({}) is None
    assert _polydata_bounds({"vertices": []}) is None
