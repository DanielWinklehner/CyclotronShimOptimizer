"""End-to-end tests for simulation.field_calculator with a stubbed radia.

Builds a fake cyclotron component tree (symmetric iron + declared-symmetric
coils + asymmetric extraction channel), registers analytic fields on the radia
stub, and checks that the symmetry-folded evaluation reproduces the full
evaluation exactly.
"""

import _testenv  # noqa: F401

import numpy as np

import geometry.components as components
import simulation.field_calculator as fc
from _radia_stub import RadiaStub
from geometry.components import BaseRadiaComponent
from geometry.symmetry import symmetry_group

SYMS = [
    ("perp", [0, 0, 0], [1, -1, 0]),
    ("perp", [0, 0, 0], [1, 0, 0]),
    ("perp", [0, 0, 0], [0, 1, 0]),
    ("para", [0, 0, 0], [0, 0, 1]),
]

_REAL_FC_RAD = fc.rad
_REAL_COMP_RAD = components.rad


def _symmetric_field(seed_scale=1.0):
    ops = symmetry_group(SYMS)

    def base(pts):
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        return seed_scale * np.stack([
            0.3 * np.sin(0.011 * x) + 2.1e-4 * y * z,
            0.2 * np.cos(0.013 * y) + 1.7e-4 * x * z,
            0.5 + 0.02 * np.cos(0.008 * z) + 3.0e-5 * x * y,
        ], axis=1)

    def field(pts):
        pts = np.asarray(pts, dtype=float)
        total = np.zeros((len(pts), 3))
        for r_mat, f_mat in ops:
            total += base(pts @ r_mat.T) @ f_mat
        return total / len(ops)

    return field


def _channel_field(pts):
    pts = np.asarray(pts, dtype=float)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return np.stack([
        0.05 * np.sin(0.017 * x + 0.5) + 1e-3 * y,
        0.03 * np.cos(0.019 * y + 1.1) + 2e-3 * z,
        0.01 * np.sin(0.023 * z + 0.3) + 3e-3 * x + 0.005,
    ], axis=1)


def _make_model(with_channel=True):
    """(stub, top_component). Iron/coils get one shared symmetric field each;
    the optional channel gets an asymmetric field."""
    stub = RadiaStub()
    fc.rad = stub
    components.rad = stub

    iron = BaseRadiaComponent(101, symmetries=SYMS)
    coils = BaseRadiaComponent(102, symmetries=SYMS)
    stub.register_field(101, _symmetric_field(1.0))
    stub.register_field(102, _symmetric_field(0.4))

    members = [iron, coils]
    if with_channel:
        channel = BaseRadiaComponent(103)
        stub.register_field(103, _channel_field)
        members.append(channel)

    top = BaseRadiaComponent.containerize(members)
    return stub, top


def _restore():
    fc.rad = _REAL_FC_RAD
    components.rad = _REAL_COMP_RAD


def _axis(limit, step):
    n = int(round(limit / step))
    half = np.arange(n + 1) * float(step)
    return np.concatenate([-half[:0:-1], half])


# ---------------------------------------------------------------------------
# Source grouping
# ---------------------------------------------------------------------------
def test_field_source_groups():
    stub, top = _make_model(with_channel=True)
    try:
        groups = fc._field_source_groups(top, use_symmetry=True)
        assert len(groups) == 2
        by_temp = {g.temp: g for g in groups}
        # iron + coils share the symmetry set -> combined into a temp container
        assert set(stub.containers[by_temp[True].radia_id]) == {101, 102}
        assert len(by_temp[True].symmetries) == 4
        # the channel stands alone with no symmetries
        assert by_temp[False].radia_id == 103
        assert by_temp[False].symmetries == []

        # symmetry off -> single group, whole model, no folding
        groups_off = fc._field_source_groups(top, use_symmetry=False)
        assert len(groups_off) == 1 and groups_off[0].radia_id == top.id
        assert groups_off[0].symmetries == []

        # bare radia id -> single group
        groups_id = fc._field_source_groups(top.id, use_symmetry=True)
        assert len(groups_id) == 1 and groups_id[0].symmetries == []
    finally:
        _restore()


# ---------------------------------------------------------------------------
# Folded vs full evaluation
# ---------------------------------------------------------------------------
def test_get_field_2d_symmetry_matches_full():
    stub, top = _make_model(with_channel=True)
    try:
        axis = _axis(40.0, 5.0)
        f_sym = fc.get_field_2d(top, axis, axis, 0.0, use_symmetry=True,
                                verbosity=0, use_gpu=False)
        f_full = fc.get_field_2d(top, axis, axis, 0.0, use_symmetry=False,
                                 verbosity=0, use_gpu=False)
        for comp in ("x", "y", "z"):
            assert np.allclose(f_sym.grid_values[comp], f_full.grid_values[comp],
                               atol=1e-12), f"component {comp} differs"
        # grid in meters
        assert np.allclose(f_sym.grid["x"], axis * 1e-3)
        # temp group containers were cleaned up
        created = [c[2] for c in stub.calls if c[0] == "ObjCnt" and set(c[1]) == {101, 102}]
        assert created and all(cid in stub.deleted for cid in created)
    finally:
        _restore()


def test_get_field_3d_symmetry_matches_full():
    stub, top = _make_model(with_channel=True)
    try:
        xy = _axis(30.0, 10.0)
        z = np.arange(-20.0, 10.0 + 1e-9, 10.0)  # asymmetric z range
        f_sym = fc.get_field_3d(top, xy, xy, z, use_symmetry=True,
                                verbosity=0, use_gpu=False)
        f_full = fc.get_field_3d(top, xy, xy, z, use_symmetry=False,
                                 verbosity=0, use_gpu=False)
        for comp in ("x", "y", "z"):
            assert np.allclose(f_sym.grid_values[comp], f_full.grid_values[comp],
                               atol=1e-12), f"component {comp} differs"
        assert f_sym.dim == 3

        # Field evaluation at a grid node reproduces the direct sum of sources
        pt_mm = np.array([[10.0, -20.0, 0.0]])
        expected = (_symmetric_field(1.0)(pt_mm) + _symmetric_field(0.4)(pt_mm)
                    + _channel_field(pt_mm))
        got = f_sym(pt_mm * 1e-3)
        assert np.allclose(got, expected, atol=1e-9)
    finally:
        _restore()


def test_symmetric_model_uses_fewer_evaluations():
    stub, top = _make_model(with_channel=False)
    try:
        axis = _axis(40.0, 5.0)
        fc.get_field_2d(top, axis, axis, 0.0, use_symmetry=True,
                        verbosity=0, use_gpu=False)
        n_folded = sum(c[3] for c in stub.calls if c[0] == "Fld")

        stub.calls.clear()
        fc.get_field_2d(top, axis, axis, 0.0, use_symmetry=False,
                        verbosity=0, use_gpu=False)
        n_full = sum(c[3] for c in stub.calls if c[0] == "Fld")

        assert n_full == len(axis) ** 2
        assert n_folded < 0.2 * n_full  # ~1/8 of the plane (octant + boundary)
    finally:
        _restore()


# ---------------------------------------------------------------------------
# r-z (isochronism) path
# ---------------------------------------------------------------------------
def test_get_field_rz_sector_and_values():
    stub, top = _make_model(with_channel=False)
    try:
        radii = [50.0, 100.0, 150.0]
        rz = fc.get_field_rz(top, radii, num_angles=1000, use_symmetry=True,
                             verbosity=0, use_gpu=False)
        # 8-fold: pi/4 sector, 1000/8 = 125 angles
        assert rz.bz.shape == (3, 125)
        assert len(rz.angles) == 125
        assert np.isclose(rz.angles[-1] + rz.angles[1], np.pi / 4.0)  # endpoint=False

        # values match a direct evaluation of the summed sources
        pts = np.zeros((len(radii), len(rz.angles), 3))
        pts[:, :, 0] = np.asarray(radii)[:, None] * np.cos(rz.angles)[None, :]
        pts[:, :, 1] = np.asarray(radii)[:, None] * np.sin(rz.angles)[None, :]
        expected = (_symmetric_field(1.0)(pts.reshape(-1, 3))
                    + _symmetric_field(0.4)(pts.reshape(-1, 3)))[:, 2]
        assert np.allclose(rz.bz.ravel(), expected, atol=1e-12)
    finally:
        _restore()


def test_get_field_rz_falls_back_to_full_circle_with_channel():
    stub, top = _make_model(with_channel=True)
    try:
        rz = fc.get_field_rz(top, [100.0], num_angles=1000, use_symmetry=True,
                             verbosity=0, use_gpu=False)
        # intersection of symmetries is empty -> full circle, no folding
        assert rz.bz.shape == (1, 1000)
        assert np.isclose(rz.angles[-1], 2.0 * np.pi * 999 / 1000)
    finally:
        _restore()


def test_gpu_precision_plumbing():
    """gpu_precision='single' must reach rad.Fld; the default stays 'double'."""
    stub, top = _make_model(with_channel=False)
    try:
        axis = _axis(20.0, 10.0)
        fc.get_field_2d(top, axis, axis, 0.0, use_symmetry=True,
                        verbosity=0, use_gpu=False, gpu_precision="single")
        precisions = {c[4] for c in stub.calls if c[0] == "Fld"}
        assert precisions == {"single"}

        stub.calls.clear()
        fc.get_field_2d(top, axis, axis, 0.0, use_symmetry=True,
                        verbosity=0, use_gpu=False)
        precisions = {c[4] for c in stub.calls if c[0] == "Fld"}
        assert precisions == {"double"}
    finally:
        _restore()


def test_rz_grid_angles_reach_isochronicity():
    """core.isochronicity must use the angles attached to the RZFieldGrid."""
    from types import SimpleNamespace

    from core.isochronicity import _grid_and_angles

    cfg = SimpleNamespace(field_evaluation=SimpleNamespace(
        num_points_circle=1000, use_symmetry=True))

    bz = np.ones((2, 7))
    angles = np.linspace(0.0, np.pi, 7, endpoint=False)
    rz = fc.RZFieldGrid(bz=bz, angles=angles, radii_mm=np.array([10.0, 20.0]))

    grid, used_angles = _grid_and_angles(rz, cfg)
    assert grid.shape == (2, 7)
    assert np.allclose(used_angles, angles)

    # plain array falls back to the config-derived octant angles
    grid2, fallback = _grid_and_angles(np.ones((2, 125)), cfg)
    assert len(fallback) == 125
    assert np.isclose(fallback[1], (np.pi / 4.0) / 125)
