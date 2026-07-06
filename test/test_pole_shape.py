"""PoleShape shim-loading regression (issue #1: top and side shim loading).

PoleShape.from_shim_configs must handle the side and top offset arrays
INDEPENDENTLY. The old callers branched only on side_offsets_deg being None,
which (a) crashed with np.array(None) when only the top array was omitted and
(b) silently dropped the top array when only the side array was omitted.
"""

import _testenv  # noqa: F401

from types import SimpleNamespace

import numpy as np

from cyclotron_optimizer.geometry.pole_shape import PoleShape

N = 3
SIDE = [5.0, 6.0, 7.0, 8.0]      # N+1 values
TOP = [10.0, 11.0, 12.0, 13.0]   # N+1 values
DEF_DEG = 5.0
DEF_MM = 2.0


def _side(offsets):
    return SimpleNamespace(side_offsets_deg=offsets, default_offset_deg=DEF_DEG)


def _top(offsets):
    return SimpleNamespace(top_offsets_mm=offsets, default_offset_mm=DEF_MM)


def test_both_offsets_none_uses_defaults():
    ps = PoleShape.from_shim_configs(N, _side(None), _top(None))
    assert np.allclose(ps.get_side_offsets_deg(), DEF_DEG)
    assert np.allclose(ps.get_top_offsets_mm(), DEF_MM)


def test_side_given_top_none_does_not_crash():
    # was: np.array(None) -> "top offsets must have shape (4,)" crash
    ps = PoleShape.from_shim_configs(N, _side(SIDE), _top(None))
    assert np.allclose(ps.get_side_offsets_deg(), SIDE)
    assert np.allclose(ps.get_top_offsets_mm(), DEF_MM)


def test_top_given_side_none_keeps_top():
    # was: top silently dropped (defaults used) because branch keyed on side
    ps = PoleShape.from_shim_configs(N, _side(None), _top(TOP))
    assert np.allclose(ps.get_side_offsets_deg(), DEF_DEG)
    assert np.allclose(ps.get_top_offsets_mm(), TOP)


def test_both_given_uses_both():
    ps = PoleShape.from_shim_configs(N, _side(SIDE), _top(TOP))
    assert np.allclose(ps.get_side_offsets_deg(), SIDE)
    assert np.allclose(ps.get_top_offsets_mm(), TOP)


def test_offsets_have_n_plus_1_entries():
    ps = PoleShape.from_shim_configs(N, _side(SIDE), _top(TOP))
    assert ps.get_side_offsets_deg().shape == (N + 1,)
    assert ps.get_top_offsets_mm().shape == (N + 1,)


def test_zero_offsets_allowed():
    # single-piece pole -> no minimum offset; zeros must be accepted
    ps = PoleShape(N, side_offsets=np.zeros(N + 1), top_offsets=np.zeros(N + 1))
    assert np.allclose(ps.get_side_offsets_deg(), 0.0)
    assert np.allclose(ps.get_top_offsets_mm(), 0.0)
    # default_offset defaults to 0 now (no shim)
    ps2 = PoleShape(N)
    assert np.allclose(ps2.get_side_offsets_deg(), 0.0)
    assert np.allclose(ps2.get_top_offsets_mm(), 0.0)


def test_negative_offsets_rejected():
    try:
        PoleShape(N, side_offsets=-np.ones(N + 1))
    except ValueError:
        return
    raise AssertionError("negative side offsets should raise")
