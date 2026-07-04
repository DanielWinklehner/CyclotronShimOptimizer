"""Tests for geometry.symmetry (pure NumPy, no radia required)."""

import _testenv  # noqa: F401  (sys.path / env setup)

import numpy as np

from cyclotron_optimizer.geometry.symmetry import (
    azimuthal_fold,
    azimuthal_sector,
    canonical_symmetry,
    canonical_symmetry_set,
    collect_field_symmetries,
    reduce_grid,
    symmetry_group,
)

# The standard 8-fold cyclotron symmetry + midplane mirror (matches
# geometry.geometry.CYCLOTRON_SYMMETRIES, duplicated here so the test does not
# import radia via geometry.geometry).
CYCLOTRON_SYMMETRIES = [
    ("perp", [0, 0, 0], [1, -1, 0]),
    ("perp", [0, 0, 0], [1, 0, 0]),
    ("perp", [0, 0, 0], [0, 1, 0]),
    ("para", [0, 0, 0], [0, 0, 1]),
]


def _symmetric_field(ops):
    """A smooth, non-trivial vector field satisfying B(R x) = F B(x) for every
    (R, F) in ops, built by group-symmetrizing an arbitrary base function."""

    def base(pts):
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        return np.stack([
            0.3 * np.sin(0.011 * x) + 2.1e-4 * y * z + 0.05 * np.cos(0.007 * y),
            0.2 * np.cos(0.013 * y) + 1.7e-4 * x * z + 0.04 * np.sin(0.009 * x),
            0.5 + 0.02 * np.cos(0.008 * z) + 3.0e-5 * x * y + 0.01 * np.sin(0.012 * x),
        ], axis=1)

    def field(pts):
        pts = np.asarray(pts, dtype=float)
        total = np.zeros((len(pts), 3))
        for r_mat, f_mat in ops:
            # B(p) = sum_g F_g^T base(R_g p); row form: base(p @ R.T) @ F
            total += base(pts @ r_mat.T) @ f_mat
        return total / len(ops)

    return field


def _asymmetric_field(pts):
    """Deliberately symmetry-free field (models the extraction channel)."""
    pts = np.asarray(pts, dtype=float)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return np.stack([
        0.1 * np.sin(0.017 * x + 0.5) + 1e-3 * y,
        0.07 * np.cos(0.019 * y + 1.1) + 2e-3 * z,
        0.02 * np.sin(0.023 * z + 0.3) + 3e-3 * x + 0.01,
    ], axis=1)


def _sym_axis(limit, step):
    n = int(round(limit / step))
    half = np.arange(n + 1) * float(step)
    return np.concatenate([-half[:0:-1], half])


# ---------------------------------------------------------------------------
# Group construction
# ---------------------------------------------------------------------------
def test_group_sizes():
    assert len(symmetry_group(None)) == 1
    assert len(symmetry_group([])) == 1
    assert len(symmetry_group([("perp", [0, 0, 0], [1, 0, 0])])) == 2
    # x=0 and y=0 mirrors generate {id, mx, my, rot180}
    assert len(symmetry_group(CYCLOTRON_SYMMETRIES[1:3])) == 4
    # full cyclotron set: D4 (8 in-plane) x {id, z-mirror} = 16
    assert len(symmetry_group(CYCLOTRON_SYMMETRIES)) == 16


def test_group_field_transform_signs():
    group = symmetry_group(CYCLOTRON_SYMMETRIES)
    eye = np.eye(3)
    for r_mat, f_mat in group:
        # orthogonal spatial transforms
        assert np.allclose(r_mat @ r_mat.T, eye)
        # Bz never mixes with Bx/By for this group, and Bz is invariant
        assert abs(f_mat[2, 0]) < 1e-12 and abs(f_mat[2, 1]) < 1e-12
        assert abs(f_mat[2, 2] - 1.0) < 1e-12


def test_inconsistent_symmetries_raise():
    bad = [("perp", [0, 0, 0], [0, 0, 1]), ("para", [0, 0, 0], [0, 0, 1])]
    try:
        symmetry_group(bad)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for perp+para on the same plane")


def test_offset_plane_not_implemented():
    try:
        symmetry_group([("perp", [1.0, 0, 0], [1, 0, 0])])
    except NotImplementedError:
        return
    raise AssertionError("Expected NotImplementedError for an offset mirror plane")


def test_canonical_symmetry():
    a = canonical_symmetry(("perp", [0, 0, 0], [2, 0, 0]))
    b = canonical_symmetry(("PERP", (0.0, 0.0, 0.0), [-1, 0, 0]))
    assert a == b
    s1 = canonical_symmetry_set(CYCLOTRON_SYMMETRIES)
    s2 = canonical_symmetry_set(list(reversed(CYCLOTRON_SYMMETRIES)))
    assert s1 == s2
    assert canonical_symmetry_set(None) == frozenset()


# ---------------------------------------------------------------------------
# Grid reduction / scatter
# ---------------------------------------------------------------------------
def test_reduce_scatter_full_3d():
    group = symmetry_group(CYCLOTRON_SYMMETRIES)
    field = _symmetric_field(group)

    axes = (_sym_axis(40.0, 10.0), _sym_axis(40.0, 10.0), _sym_axis(20.0, 10.0))
    reduction = reduce_grid(axes, group)

    n_total = np.prod([len(a) for a in axes])
    assert reduction.n_total == n_total
    assert reduction.n_ops == 16
    # substantial reduction (bounded by orbit sizes; boundary points fold less)
    assert len(reduction.eval_points) < 0.25 * n_total

    b_folded = reduction.scatter_vector(field(reduction.eval_points))

    mesh = np.meshgrid(*axes, indexing="ij")
    all_points = np.column_stack([m.ravel() for m in mesh])
    b_direct = field(all_points)

    assert np.allclose(b_folded, b_direct, atol=1e-12)


def test_reduce_asymmetric_z_range_drops_z_mirror():
    group = symmetry_group(CYCLOTRON_SYMMETRIES)
    field = _symmetric_field(group)

    z_axis = np.arange(-30.0, 10.0 + 1e-9, 10.0)  # NOT symmetric about z=0
    axes = (_sym_axis(40.0, 10.0), _sym_axis(40.0, 10.0), z_axis)
    reduction = reduce_grid(axes, group)

    # z-mirror (and its compositions) unusable -> in-plane D4 only
    assert reduction.n_ops == 8

    b_folded = reduction.scatter_vector(field(reduction.eval_points))
    mesh = np.meshgrid(*axes, indexing="ij")
    b_direct = field(np.column_stack([m.ravel() for m in mesh]))
    assert np.allclose(b_folded, b_direct, atol=1e-12)


def test_reduce_midplane_slice():
    group = symmetry_group(CYCLOTRON_SYMMETRIES)
    field = _symmetric_field(group)

    axes = (_sym_axis(40.0, 5.0), _sym_axis(40.0, 5.0), np.array([0.0]))
    reduction = reduce_grid(axes, group)
    assert reduction.n_ops == 16  # z-mirror maps z=0 onto itself

    b_folded = reduction.scatter_vector(field(reduction.eval_points))
    mesh = np.meshgrid(*axes, indexing="ij")
    b_direct = field(np.column_stack([m.ravel() for m in mesh]))
    assert np.allclose(b_folded, b_direct, atol=1e-12)

    # roughly 1/8 of the plane (octant) is evaluated
    assert len(reduction.eval_points) < 0.2 * reduction.n_total


def test_reduce_offplane_slice_drops_z_mirror():
    group = symmetry_group(CYCLOTRON_SYMMETRIES)
    field = _symmetric_field(group)

    axes = (_sym_axis(40.0, 10.0), _sym_axis(40.0, 10.0), np.array([10.0]))
    reduction = reduce_grid(axes, group)
    assert reduction.n_ops == 8  # z -> -z leaves the z=10 slice

    b_folded = reduction.scatter_vector(field(reduction.eval_points))
    mesh = np.meshgrid(*axes, indexing="ij")
    b_direct = field(np.column_stack([m.ravel() for m in mesh]))
    assert np.allclose(b_folded, b_direct, atol=1e-12)


def test_reduce_no_symmetry_is_identity():
    group = symmetry_group([])
    axes = (_sym_axis(20.0, 10.0), _sym_axis(20.0, 10.0), np.array([0.0]))
    reduction = reduce_grid(axes, group)
    assert reduction.n_ops == 1
    assert len(reduction.eval_points) == reduction.n_total
    bz = _asymmetric_field(reduction.eval_points)
    assert np.allclose(reduction.scatter_vector(bz), bz)


def test_scatter_scalar_z():
    group = symmetry_group(CYCLOTRON_SYMMETRIES)
    field = _symmetric_field(group)

    axes = (_sym_axis(40.0, 5.0), _sym_axis(40.0, 5.0), np.array([0.0]))
    reduction = reduce_grid(axes, group)

    bz_folded = reduction.scatter_scalar_z(field(reduction.eval_points)[:, 2])
    mesh = np.meshgrid(*axes, indexing="ij")
    bz_direct = field(np.column_stack([m.ravel() for m in mesh]))[:, 2]
    assert np.allclose(bz_folded, bz_direct, atol=1e-12)


def test_scatter_scalar_z_rejects_xz_mixing():
    # A mirror with normal in the x-z plane mixes Bx and Bz.
    group = symmetry_group([("perp", [0, 0, 0], [1, 0, 1])])
    axis = _sym_axis(20.0, 10.0)
    reduction = reduce_grid((axis, axis, axis), group)
    assert reduction.n_ops == 2  # x<->-z swap maps the cubic grid onto itself
    try:
        reduction.scatter_scalar_z(np.zeros(len(reduction.eval_points)))
    except ValueError:
        return
    raise AssertionError("Expected ValueError for Bz-only scatter with x-z mixing")


# ---------------------------------------------------------------------------
# Azimuthal folding
# ---------------------------------------------------------------------------
def test_azimuthal_fold():
    n_rot, mirror0 = azimuthal_fold(symmetry_group(CYCLOTRON_SYMMETRIES))
    assert n_rot == 4 and mirror0
    assert np.isclose(azimuthal_sector(CYCLOTRON_SYMMETRIES), np.pi / 4.0)

    # no symmetry -> full circle
    assert np.isclose(azimuthal_sector([]), 2.0 * np.pi)

    # midplane mirror only: in-plane action trivial -> full circle
    assert np.isclose(azimuthal_sector([("para", [0, 0, 0], [0, 0, 1])]), 2.0 * np.pi)

    # single vertical mirror at theta=0 -> half circle
    assert np.isclose(azimuthal_sector([("perp", [0, 0, 0], [0, 1, 0])]), np.pi)


# ---------------------------------------------------------------------------
# Symmetry collection over duck-typed component trees
# ---------------------------------------------------------------------------
class _FakeComponent:
    def __init__(self, symmetries=None, children=None):
        self.symmetries = list(symmetries) if symmetries else []
        self._children = list(children) if children else []

    def iter_cached_children(self):
        return iter(self._children)


def test_collect_field_symmetries():
    iron = _FakeComponent(symmetries=CYCLOTRON_SYMMETRIES)
    coils = _FakeComponent(symmetries=list(reversed(CYCLOTRON_SYMMETRIES)))
    channel = _FakeComponent()

    top_sym = _FakeComponent(children=[iron, coils])
    collected = collect_field_symmetries(top_sym)
    assert canonical_symmetry_set(collected) == canonical_symmetry_set(CYCLOTRON_SYMMETRIES)

    top_mixed = _FakeComponent(children=[iron, coils, channel])
    assert collect_field_symmetries(top_mixed) == []

    # own declaration takes precedence over children
    declared = _FakeComponent(symmetries=CYCLOTRON_SYMMETRIES[:1], children=[channel])
    assert canonical_symmetry_set(collect_field_symmetries(declared)) == \
        canonical_symmetry_set(CYCLOTRON_SYMMETRIES[:1])

    # leaf without anything
    assert collect_field_symmetries(_FakeComponent()) == []
