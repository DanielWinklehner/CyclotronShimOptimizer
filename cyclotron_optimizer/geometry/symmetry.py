"""Field-symmetry algebra for exploiting mirror symmetries in field evaluation.

The geometry components store their symmetries as ``(kind, point, normal)``
tuples ('perp' -> rad.TrfZerPerp, 'para' -> rad.TrfZerPara). For a mirror plane
through the origin with unit normal n, the reflection is ``R = I - 2 n n^T``
and the field of a source symmetric under that mirror transforms as

    'perp':  B(R x) =  R B(x)      (field perpendicular to the plane vanishes
                                    ON the plane; B transforms as a vector)
    'para':  B(R x) = -R B(x)      (field parallel to the plane vanishes on it)

so each symmetry maps to a pair of matrices ``(R, F)`` with ``F = +/-R``.
Compositions multiply, and the full symmetry group is the closure of the
generators under matrix multiplication (16 elements for the standard 8-fold
cyclotron symmetry + midplane mirror).

Field maps are then evaluated only on a fundamental subset of the grid: each
grid point's orbit under the group is computed IN INDEX SPACE, the orbit
member with the smallest flat index is the canonical representative, and after
evaluation the values are scattered back with the appropriate ``F`` transform.
Group elements that do not map the grid onto itself (e.g. a z-mirror on an
asymmetric z range, or any vertical mirror on a slice at z != 0) are dropped;
the surviving elements are the stabilizer subgroup of the grid, so the
reduction is always exact -- worst case only the identity survives and the
full grid is evaluated.

Everything here is pure NumPy (no radia dependency).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

# Matches components.SymmetryTuple: ('perp'|'para', point, normal)
SymmetryTuple = Tuple[str, Any, Any]

# Number of decimals used to key/dedupe matrices and canonicalize normals.
_ROUND = 9

# Safety cap on the closure size (D4h has 16; anything huge means bad input).
_MAX_GROUP_SIZE = 96


# ---------------------------------------------------------------------------
# Canonical form / generators
# ---------------------------------------------------------------------------
def canonical_symmetry(sym: SymmetryTuple) -> Tuple[str, Tuple[float, ...], Tuple[float, ...]]:
    """Return a hashable canonical form of a symmetry tuple for set comparison.

    The normal is normalized to unit length with its first nonzero component
    positive (a mirror plane's normal sign is arbitrary), and kind is lowered.
    """
    kind, point, normal = sym
    kind = str(kind).lower()
    if kind not in ("perp", "para"):
        raise ValueError(f"symmetry kind must be 'perp' or 'para', got {kind!r}")

    p = np.asarray(point, dtype=float)
    n = np.asarray(normal, dtype=float)
    norm = np.linalg.norm(n)
    if norm == 0:
        raise ValueError("symmetry normal must be non-zero")
    n = n / norm
    for comp in n:
        if abs(comp) > 10.0 ** (-_ROUND):
            if comp < 0:
                n = -n
            break

    return (kind, tuple(np.round(p, _ROUND)), tuple(np.round(n, _ROUND)))


def canonical_symmetry_set(symmetries: Optional[Sequence[SymmetryTuple]]) -> frozenset:
    """Canonical, order-independent representation of a list of symmetries."""
    if not symmetries:
        return frozenset()
    return frozenset(canonical_symmetry(s) for s in symmetries)


def reflection_matrix(normal: Sequence[float]) -> np.ndarray:
    """Householder reflection about the plane through the origin with this normal."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    return np.eye(3) - 2.0 * np.outer(n, n)


def _generator_ops(symmetries: Sequence[SymmetryTuple]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """(R, F) generator pairs for the given symmetry tuples (planes must pass
    through the origin -- offset mirror planes are not supported yet)."""
    ops = []
    for sym in symmetries:
        kind, point, normal = canonical_symmetry(sym)
        if any(abs(c) > 10.0 ** (-_ROUND) for c in point):
            raise NotImplementedError(
                f"Symmetry planes must pass through the origin (got point={point}). "
                "Support for offset planes requires affine (R, t) handling."
            )
        r_mat = reflection_matrix(normal)
        f_mat = r_mat if kind == "perp" else -r_mat
        ops.append((r_mat, f_mat))
    return ops


def _mat_key(mat: np.ndarray) -> Tuple[float, ...]:
    return tuple(np.round(mat, _ROUND).ravel())


def symmetry_group(symmetries: Optional[Sequence[SymmetryTuple]]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Closure of the symmetry generators under composition.

    :param symmetries: list of ('perp'|'para', point, normal) tuples (or None).
    :return: list of (R, F) pairs including the identity. For an empty input
             this is just ``[(I, I)]``.
    :raises ValueError: if the same spatial transform R arises with conflicting
             field transforms F (physically inconsistent symmetry set).
    """
    identity = np.eye(3)
    group = {_mat_key(identity): (identity, identity)}
    generators = _generator_ops(symmetries or [])

    changed = True
    while changed:
        changed = False
        for r1, f1 in list(group.values()):
            for r2, f2 in generators:
                r_new = r1 @ r2
                f_new = f1 @ f2
                key = _mat_key(r_new)
                if key in group:
                    if not np.allclose(group[key][1], f_new, atol=10.0 ** (-_ROUND)):
                        raise ValueError(
                            "Inconsistent symmetry set: the same spatial transform "
                            "implies two different field transforms."
                        )
                else:
                    group[key] = (r_new, f_new)
                    changed = True
        if len(group) > _MAX_GROUP_SIZE:
            raise ValueError(
                f"Symmetry group exceeds {_MAX_GROUP_SIZE} elements -- "
                "check the symmetry definitions."
            )

    return list(group.values())


# ---------------------------------------------------------------------------
# Collecting symmetries from component trees
# ---------------------------------------------------------------------------
def collect_field_symmetries(component: Any) -> List[SymmetryTuple]:
    """Field symmetries of a component: its own declared set, or the
    intersection over its (cached) children when it has none of its own.

    A component's ``symmetries`` metadata means "the FIELD of this component
    (as built) is invariant under these operations" -- true both for radia
    TrfZer* mirrors (the model physically contains the mirrored copies) and
    for declared symmetries of intrinsically symmetric sources (e.g. a
    full-revolution +/-z coil pair). Children are only consulted through the
    container's wrapper cache; a container of anonymous leaf ids (e.g. the
    tets of an STP mesh) contributes whatever the container itself declares.
    """
    own = getattr(component, "symmetries", None)
    if own:
        return list(own)

    children = []
    iter_children = getattr(component, "iter_cached_children", None)
    if iter_children is not None:
        children = list(iter_children())
    if not children:
        return []

    first = collect_field_symmetries(children[0])
    common = canonical_symmetry_set(first)
    for child in children[1:]:
        if not common:
            break
        common &= canonical_symmetry_set(collect_field_symmetries(child))

    # Return the original (un-canonicalized) tuples of the first child that
    # survive the intersection, preserving a usable (kind, point, normal) form.
    return [sym for sym in first if canonical_symmetry(sym) in common]


# ---------------------------------------------------------------------------
# Grid reduction
# ---------------------------------------------------------------------------
def _axis_indices(vals: np.ndarray, axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map coordinate values onto grid-node indices of a sorted 1D axis.

    :return: (indices, on_node) -- indices clipped into range and a boolean
             mask of values that actually coincide with a node (within a
             tolerance scaled to the axis spacing).
    """
    n = len(axis)
    if n == 1:
        idx = np.zeros(len(vals), dtype=np.intp)
        tol = 1e-9 * max(1.0, abs(float(axis[0])))
        return idx, np.abs(vals - axis[0]) <= tol

    spacing = np.min(np.diff(axis))
    tol = 1e-6 * spacing
    idx = np.searchsorted(axis, vals - tol)
    idx = np.clip(idx, 0, n - 1)
    on_node = np.abs(axis[idx] - vals) <= tol
    return idx, on_node


class GridReduction:
    """Result of reducing a regular grid by a symmetry group.

    Attributes
    ----------
    eval_points : (M, 3) array
        The canonical representative points (same units as the input axes).
    n_total : int
        Number of points of the full grid.
    n_ops : int
        Number of group elements usable on this grid (stabilizer subgroup size).
    """

    def __init__(self, axes, eval_points, rep_pos, op_index, ops_f):
        self.axes = axes
        self.eval_points = eval_points
        self._rep_pos = rep_pos          # (N,) index into eval_points per grid point
        self._op_index = op_index        # (N,) index into ops_f per grid point
        self._ops_f = ops_f              # list of F matrices (usable ops)
        self.n_total = len(rep_pos)
        self.n_ops = len(ops_f)

    @property
    def grid_shape(self) -> Tuple[int, ...]:
        return tuple(len(a) for a in self.axes)

    def scatter_vector(self, b_eval: np.ndarray) -> np.ndarray:
        """Broadcast (M, 3) field vectors at the eval points to the full grid.

        Uses B(x_i) = F_g^T B(x_rep) for the op g mapping x_i onto its
        representative (row-vector form: b_i = b_rep @ F_g).
        :return: (N, 3) array in flat grid order ('ij' meshgrid, C-ravel).
        """
        b_eval = np.asarray(b_eval, dtype=float).reshape(-1, 3)
        out = np.empty((self.n_total, 3), dtype=float)
        for k, f_mat in enumerate(self._ops_f):
            mask = self._op_index == k
            if np.any(mask):
                out[mask] = b_eval[self._rep_pos[mask]] @ f_mat
        return out

    def scatter_scalar_z(self, bz_eval: np.ndarray) -> np.ndarray:
        """Broadcast (M,) Bz values to the full grid.

        Only valid when no usable op mixes the z field component with x/y
        (F[0,2] = F[1,2] = 0 for all ops); raises otherwise.
        """
        for f_mat in self._ops_f:
            if abs(f_mat[0, 2]) > 1e-12 or abs(f_mat[1, 2]) > 1e-12:
                raise ValueError(
                    "Bz-only scatter invalid: a symmetry op mixes z with x/y "
                    "field components. Use scatter_vector with 'bxbybz'."
                )
        bz_eval = np.asarray(bz_eval, dtype=float).ravel()
        out = np.empty(self.n_total, dtype=float)
        for k, f_mat in enumerate(self._ops_f):
            mask = self._op_index == k
            if np.any(mask):
                out[mask] = f_mat[2, 2] * bz_eval[self._rep_pos[mask]]
        return out


def reduce_grid(axes: Sequence[np.ndarray],
                ops: Sequence[Tuple[np.ndarray, np.ndarray]]) -> GridReduction:
    """Reduce a regular (x, y, z) grid to its fundamental subset under `ops`.

    Ops that do not map every grid node onto a grid node are dropped -- the
    survivors form the stabilizer subgroup of the grid, so correctness never
    depends on the grid being "nice". With only the identity surviving, the
    reduction degenerates to full evaluation.

    :param axes: three sorted 1D coordinate arrays (any units; singleton axes
                 are fine, e.g. z = [0.0] for a midplane slice).
    :param ops: (R, F) pairs from symmetry_group().
    :return: GridReduction
    """
    axes = [np.asarray(a, dtype=float) for a in axes]
    if len(axes) != 3:
        raise ValueError("reduce_grid expects exactly three axes (x, y, z).")

    # Identity first: points that are their own representative keep op_index 0,
    # so usable ops[0] MUST be the identity (F = I on scatter).
    eye = np.eye(3)
    ops = ([(r, f) for r, f in ops if np.allclose(r, eye) and np.allclose(f, eye)][:1] or [(eye, eye)]) \
        + [(r, f) for r, f in ops if not (np.allclose(r, eye) and np.allclose(f, eye))]

    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([m.ravel() for m in mesh])
    n_total = len(points)
    n_yz = len(axes[1]) * len(axes[2])
    n_z = len(axes[2])

    canon = np.arange(n_total)
    op_index = np.zeros(n_total, dtype=np.intp)
    usable_f: List[np.ndarray] = []
    usable_targets: List[np.ndarray] = []

    for r_mat, f_mat in ops:
        transformed = points @ r_mat.T
        ix, ok_x = _axis_indices(transformed[:, 0], axes[0])
        iy, ok_y = _axis_indices(transformed[:, 1], axes[1])
        iz, ok_z = _axis_indices(transformed[:, 2], axes[2])
        if not (ok_x.all() and ok_y.all() and ok_z.all()):
            continue  # op does not preserve this grid -> not in the stabilizer
        usable_targets.append(ix * n_yz + iy * n_z + iz)
        usable_f.append(f_mat)

    for k, target in enumerate(usable_targets):
        better = target < canon
        canon[better] = target[better]
        op_index[better] = k

    reps = np.unique(canon)
    rep_pos = np.searchsorted(reps, canon)
    eval_points = points[reps]

    return GridReduction(axes, eval_points, rep_pos, op_index, usable_f)


# ---------------------------------------------------------------------------
# Azimuthal (r-theta) folding
# ---------------------------------------------------------------------------
def azimuthal_fold(ops: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[int, bool]:
    """Fold factor for Bz sampled on midplane circles.

    Considers only ops that (a) preserve the z = 0 plane without tilting it
    (block-diagonal R), and (b) leave Bz invariant (F z-row = [0, 0, +1]) --
    the requirement for folding an azimuthal Bz average.

    :return: (n_rot, mirror_at_theta0) where n_rot is the number of in-plane
             rotations (>= 1, identity included) and mirror_at_theta0 is True
             when the theta -> -theta mirror is available. The fundamental
             azimuthal sector is then 2*pi/n_rot, halved again if
             mirror_at_theta0 (sector boundary on a mirror axis).
    """
    rotations = set()
    mirror0 = False
    for r_mat, f_mat in ops:
        block_diag = (abs(r_mat[2, 0]) < 1e-12 and abs(r_mat[2, 1]) < 1e-12
                      and abs(r_mat[0, 2]) < 1e-12 and abs(r_mat[1, 2]) < 1e-12)
        bz_invariant = (abs(f_mat[2, 0]) < 1e-12 and abs(f_mat[2, 1]) < 1e-12
                        and abs(f_mat[2, 2] - 1.0) < 1e-12)
        if not (block_diag and bz_invariant):
            continue
        a_mat = r_mat[:2, :2]
        det = np.linalg.det(a_mat)
        if det > 0:
            rotations.add(_mat_key(a_mat))
        elif np.allclose(a_mat, np.diag([1.0, -1.0]), atol=1e-9):
            mirror0 = True

    return max(1, len(rotations)), mirror0


def azimuthal_sector(symmetries: Optional[Sequence[SymmetryTuple]]) -> float:
    """Fundamental azimuthal sector [rad] for midplane Bz sampling."""
    n_rot, mirror0 = azimuthal_fold(symmetry_group(symmetries))
    sector = 2.0 * np.pi / n_rot
    if mirror0:
        sector /= 2.0
    return sector


__all__ = [
    "SymmetryTuple",
    "canonical_symmetry",
    "canonical_symmetry_set",
    "reflection_matrix",
    "symmetry_group",
    "collect_field_symmetries",
    "GridReduction",
    "reduce_grid",
    "azimuthal_fold",
    "azimuthal_sector",
]
