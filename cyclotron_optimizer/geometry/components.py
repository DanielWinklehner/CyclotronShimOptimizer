from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

try:
    import radia as rad  # RadiaCUDA-compatible import
except Exception as _rad_import_error:  # pragma: no cover
    rad = None
else:
    _rad_import_error = None


# -----------------------------
# Type aliases
# -----------------------------
Vertex = Tuple[float, float, float]
BuilderResult = Union[
    int,
    Sequence[int],
    Mapping[str, Any],  # {"id": int, "child_ids": [...], "is_container": bool}
]
SymmetryTuple = Tuple[str, Any, Any]  # ('perp'|'para', point, normal)
SymmetryInput = Union[SymmetryTuple, Sequence[SymmetryTuple]]


# -----------------------------
# Exceptions
# -----------------------------
class RadiaComponentError(RuntimeError):
    pass


class RadiaUnavailableError(RadiaComponentError):
    pass


class ParentAssignmentError(RadiaComponentError):
    pass


# -----------------------------
# Helpers
# -----------------------------
def _require_radia() -> None:
    if rad is None:
        raise RadiaUnavailableError(
            f"radia/RadiaCUDA is unavailable in this environment: {_rad_import_error!r}"
        )


def _is_radia_error(result: Any) -> bool:
    return isinstance(result, str) and result.strip().lower().startswith("error")


def _call_radia(func_name: str, *args: Any, **kwargs: Any) -> Any:
    _require_radia()
    fn = getattr(rad, func_name, None)
    if fn is None:
        raise AttributeError(f"rad.{func_name} does not exist.")
    out = fn(*args, **kwargs)
    if _is_radia_error(out):
        raise RadiaComponentError(f"rad.{func_name} failed: {out}")
    return out


def _validate_radia_id(value: Any, label: str = "id") -> int:
    if not isinstance(value, int):
        raise TypeError(f"{label} must be int, got {type(value).__name__}.")
    if value <= 0:
        raise ValueError(f"{label} must be > 0, got {value}.")
    return value


def _as_symmetry_list(symmetries: Optional[SymmetryInput]) -> List[SymmetryTuple]:
    """
    Only handles:
      - single tuple: ('perp'|'para', point, normal)
      - list of such tuples
    No extra shape/length checks (Radia will validate point/normal).
    """
    if symmetries is None:
        return []

    # single tuple case
    if isinstance(symmetries, (tuple, list)) and len(symmetries) == 3 and isinstance(symmetries[0], str):
        return [tuple(symmetries)]  # type: ignore[arg-type]

    # list-of-tuples case
    return [tuple(s) for s in symmetries]  # type: ignore[arg-type]


# Default tetrahedron -> radia polyhedron faces (1-indexed vertex order).
_TET_FACES = [[1, 2, 3], [1, 4, 2], [2, 4, 3], [3, 4, 1]]


def _tet_to_polyhedron(vertices: Sequence[Vertex]) -> int:
    """Build an (initially unmagnetized) radia polyhedron from 4 tet corner vertices.

    This is the single canonical way to turn a mesh tetrahedron into a radia
    volume: rad.ObjPolyhdr with the fixed tet face connectivity. Magnetization
    starts at zero; the material + relaxation set it.
    """
    return _call_radia("ObjPolyhdr", [list(v) for v in vertices], _TET_FACES)


def _extract_tet_coords() -> List[List[List[float]]]:
    """Extract linear-tetra vertex coordinates from the current (meshed) gmsh model.

    Returns a list of tets, each a list of four [x, y, z] corner coordinates.
    """
    import gmsh

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes: Dict[int, List[float]] = {}
    for i, tag in enumerate(node_tags):
        j = 3 * i
        nodes[int(tag)] = [float(node_coords[j]), float(node_coords[j + 1]), float(node_coords[j + 2])]

    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements()
    tets: List[List[List[float]]] = []
    for elem_type, conn in zip(elem_types, elem_node_tags):
        if int(elem_type) != 4:  # gmsh element type 4 == linear tetrahedron
            continue
        for i in range(0, len(conn), 4):
            tets.append([nodes[int(t)] for t in conn[i:i + 4]])
    return tets


def _resolve_comm(comm: Any):
    """Return the given comm, or fall back to MPI.COMM_WORLD when MPI is initialized.

    Guards the comm=None trap: under a real multi-rank launch a None comm would make
    every rank mesh independently (divergent radia ids). Falling back to COMM_WORLD
    ensures the rank-0-mesh + broadcast path is used. Returns None only when MPI is not
    initialized (genuine single-process / standalone use).
    """
    if comm is not None:
        return comm
    try:
        from mpi4py import MPI
        if MPI.Is_initialized():
            return MPI.COMM_WORLD
    except Exception:
        pass
    return None


def _pin_gmsh_determinism() -> None:
    """Pin gmsh to single-threaded Delaunay so meshing is reproducible run-to-run
    (defense-in-depth for any per-rank meshing path)."""
    import gmsh
    for name, val in (("General.NumThreads", 1),
                      ("Mesh.MaxNumThreads3D", 1),
                      ("Mesh.Algorithm3D", 1)):  # 1 = Delaunay (single-threaded)
        try:
            gmsh.option.setNumber(name, val)
        except Exception:
            pass


# -----------------------------
# Material wrapper
# -----------------------------
_MU0 = 4.0e-7 * 3.141592653589793  # T*m/A

# Unit -> factor converting a value in that unit to the INTERNAL convention:
# H is stored as mu0*H in Tesla (radia's convention), B and M in Tesla.
_H_UNIT_TO_T = {
    "T": 1.0,            # already mu0*H [T]
    "mT": 1.0e-3,
    "A/m": _MU0,
    "kA/m": _MU0 * 1.0e3,
    "Oe": 1.0e-4,        # 1 Oe = 1 G of mu0*H exactly
}
_B_UNIT_TO_T = {
    "T": 1.0,
    "mT": 1.0e-3,
    "G": 1.0e-4,
    "kG": 0.1,
}


def _h_factor(unit: str) -> float:
    try:
        return _H_UNIT_TO_T[unit]
    except KeyError:
        raise ValueError(f"Unknown H unit {unit!r}; supported: "
                         f"{sorted(_H_UNIT_TO_T)}")


def _b_factor(unit: str) -> float:
    try:
        return _B_UNIT_TO_T[unit]
    except KeyError:
        raise ValueError(f"Unknown B/M unit {unit!r}; supported: "
                         f"{sorted(_B_UNIT_TO_T)}")


class RadiaMaterial:
    """Nonlinear isotropic material from a tabulated BH or MH curve.

    Internal storage convention (radia's): H as mu0*H in Tesla, M and B in
    Tesla, with B = mu0*H + M. Curves can be loaded from and exported to
    other units (see _H_UNIT_TO_T / _B_UNIT_TO_T), e.g. the COMSOL export
    convention H in A/m, B in T::

        iron = RadiaMaterial.from_bh_file("COMSOL_1010_BH_T_A-m.csv",
                                          curve="BH", h_unit="A/m")
        hb = iron.get_bh_curve(h_unit="A/m", b_unit="T")   # (N, 2) array
        iron.plot_bh_curve(h_unit="kA/m", kind="both", show=True)
    """

    def __init__(self, name: str = "material", metadata: Optional[Dict[str, Any]] = None) -> None:
        self._name = name
        self._material_object = None
        self._filename = None
        self._metadata = metadata
        self._h_t = None   # (N,) mu0*H [T]
        self._m_t = None   # (N,) M [T]

    @property
    def name(self) -> str:
        return self._name

    @property
    def material(self):
        return self._material_object

    @property
    def filename(self):
        return self._filename

    @property
    def metadata(self):
        return self._metadata

    # ------------------------------------------------------------------
    # Curve access / plotting
    # ------------------------------------------------------------------
    def _require_curve(self):
        if self._h_t is None:
            raise ValueError(
                f"Material {self._name!r} carries no tabulated curve "
                "(only from_bh_file materials do)."
            )

    def get_mh_curve(self, h_unit: str = "T", m_unit: str = "T"):
        """(N, 2) array of (H, M) in the requested units."""
        import numpy as np
        self._require_curve()
        return np.column_stack([self._h_t / _h_factor(h_unit),
                                self._m_t / _b_factor(m_unit)])

    def get_bh_curve(self, h_unit: str = "T", b_unit: str = "T"):
        """(N, 2) array of (H, B) in the requested units (B = mu0*H + M)."""
        import numpy as np
        self._require_curve()
        return np.column_stack([self._h_t / _h_factor(h_unit),
                                (self._h_t + self._m_t) / _b_factor(b_unit)])

    def plot_bh_curve(self, h_unit: str = "A/m", b_unit: str = "T", *,
                      kind: str = "BH", ax=None, logx: bool = False,
                      label: Optional[str] = None, show: bool = False):
        """Plot the material curve.

        :param kind: 'BH', 'MH', or 'both' (both share the H axis).
        :return: (fig, ax)
        """
        import matplotlib.pyplot as plt

        self._require_curve()
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig = ax.figure

        base = label or self._name
        if kind in ("BH", "both"):
            hb = self.get_bh_curve(h_unit=h_unit, b_unit=b_unit)
            ax.plot(hb[:, 0], hb[:, 1], marker=".", ms=4,
                    label=f"{base}  B(H)")
        if kind in ("MH", "both"):
            hm = self.get_mh_curve(h_unit=h_unit, m_unit=b_unit)
            ax.plot(hm[:, 0], hm[:, 1], marker=".", ms=4, ls="--",
                    label=f"{base}  M(H)")
        if kind not in ("BH", "MH", "both"):
            raise ValueError("kind must be 'BH', 'MH' or 'both'")

        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(f"H [{h_unit}]")
        ax.set_ylabel(f"B, M [{b_unit}]")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_radia_material(cls, material: Any, name: str = "material") -> "RadiaMaterial":
        if material is None:
            raise ValueError("material cannot be None")

        try:
            data = _call_radia("UtiDmp", material)
        except Exception as exc:
            raise ValueError(f"Not an active radia ID: {material!r}") from exc

        if "Magnetic material" not in str(data):
            raise ValueError("Object is not a magnetic material.")

        tmp_cls = cls(name)
        tmp_cls._material_object = material
        return tmp_cls

    @classmethod
    def from_bh_file(cls, filename: str, *, curve: str = "BH",
                     h_unit: str = "T", b_unit: str = "T",
                     name: str = "material", delimiter: str = ",",
                     type: Optional[str] = None) -> "RadiaMaterial":
        """Nonlinear isotropic material from a two-column curve file.

        :param filename: CSV with columns (H, B) for curve='BH' or (H, M)
            for curve='MH'. Relative names resolve against the cwd, then the
            repo's resources/ folder. Lines that do not parse as two numbers
            (headers/comments) are skipped.
        :param curve: 'BH' (second column is B; M = B - mu0*H) or 'MH'.
        :param h_unit: unit of the H column ('T' = mu0*H in Tesla -- the
            historical convention of the radia curve files -- or 'A/m',
            'kA/m', 'mT', 'Oe').
        :param b_unit: unit of the B/M column ('T', 'mT', 'G', 'kG').
        :param type: DEPRECATED alias for ``curve`` (old positional API).
        """
        import numpy as np

        if filename is None:
            raise ValueError("filename cannot be None")
        if type is not None:
            curve = type
        if curve not in ("BH", "MH"):
            raise ValueError("curve must be 'BH' or 'MH'")

        tmp_cls = cls(name)
        tmp_cls._filename = filename

        # Resolve relative curve filenames: cwd first (project-local files),
        # then the repo's resources/ (two levels above this file:
        # cyclotron_optimizer/geometry/ -> repo root).
        if os.path.isabs(filename):
            full_path = filename
        else:
            here = os.path.dirname(os.path.realpath(__file__))
            candidates = [
                os.path.abspath(filename),
                os.path.join(here, "..", "..", "resources", filename),
            ]
            full_path = next((c for c in candidates if os.path.exists(c)),
                             candidates[-1])

        raw = np.genfromtxt(full_path, delimiter=delimiter)
        raw = np.atleast_2d(raw)
        raw = raw[~np.isnan(raw).any(axis=1)]  # drop header/comment lines
        if raw.shape[0] < 2 or raw.shape[1] < 2:
            raise ValueError(f"Curve file {full_path!r} needs >= 2 rows of "
                             "two numeric columns (H, B|M)")

        h_t = raw[:, 0] * _h_factor(h_unit)
        second = raw[:, 1] * _b_factor(b_unit)
        m_t = (second - h_t) if curve == "BH" else second

        order = np.argsort(h_t)
        h_t, m_t = h_t[order], m_t[order]
        keep = np.concatenate([[True], np.diff(h_t) > 0])  # dedupe H values
        h_t, m_t = h_t[keep], m_t[keep]

        # Physics sanity: M(H) must be non-decreasing (equivalently, the BH
        # curve must have mu_r >= 1). A defective tail is worse than it
        # looks: radia's tabulated material extrapolates BEYOND the table
        # linearly with the END slope, so one bad last segment gives every
        # relaxation transient an M that falls with H -- measured to cause
        # misfit floors, method disagreements, and Anderson divergence
        # (dillinger_steel.csv's original last segment had mu_r < 1).
        # Tolerance: perfectly saturated tails give secants of 0 minus
        # rounding (M = B - mu0*H with nearly equal increments); only a
        # meaningfully negative slope is a data defect.
        secants = np.diff(m_t) / np.diff(h_t)
        if np.any(secants < -1e-3):
            import warnings
            i = int(np.argmin(secants))
            warnings.warn(
                f"BH/MH curve {filename!r} is NON-MONOTONE in M(H): worst "
                f"segment mu0*H = {h_t[i]:.5f}..{h_t[i + 1]:.5f} T has "
                f"dM/dH = {secants[i]:.4f} (mu_r < 1 for a BH input). The "
                "relaxation extrapolates the END slope beyond the table -- "
                "fix the curve data.", stacklevel=2)

        tmp_cls._h_t = h_t
        tmp_cls._m_t = m_t
        tmp_cls._metadata = {"curve": curve, "h_unit": h_unit,
                             "b_unit": b_unit, "path": full_path,
                             "n_points": int(len(h_t))}
        tmp_cls._material_object = _call_radia(
            "MatSatIsoTab", np.column_stack([h_t, m_t]).tolist())
        return tmp_cls

    @classmethod
    def from_formula(cls, metadata: Dict[str, Any], name: str = "material") -> "RadiaMaterial":
        # TODO: implement formula material
        if metadata is None:
            raise ValueError("metadata cannot be None")
        return cls(name=name, metadata=metadata)


# -----------------------------
# Structural base class
# -----------------------------
class BaseRadiaComponent:
    """
    Structural wrapper:
      - id
      - child ids / container behavior
      - parent
      - color
      - lazy child wrapper generation
      - field-symmetry METADATA (declaration only; applying radia TrfZer*
        transforms is MagnetizedComponent's job)

    The ``symmetries`` metadata means: "the FIELD of this component, as built,
    is invariant under these mirror operations". That holds for magnetized
    parts whose TrfZer* transforms were applied (the model contains the
    mirrored copies) and for intrinsically symmetric sources that merely
    DECLARE it (e.g. a full-revolution +/-z coil pair). The field evaluator
    reads this metadata to decide how a component's field map may be folded.
    """

    def __init__(
        self,
        radia_id: int,
        *,
        child_ids: Optional[Sequence[int]] = None,
        is_container: Optional[bool] = None,
        parent: Optional["BaseRadiaComponent"] = None,
        symmetries: Optional[SymmetryInput] = None,
        color: Optional[Sequence[float]] = None,
        apply_color: bool = False,
    ) -> None:
        self._id = _validate_radia_id(radia_id, "radia_id")
        self._child_ids = list(child_ids) if child_ids is not None else []

        if is_container is None:
            self._is_container = len(self._child_ids) > 0
        else:
            self._is_container = bool(is_container)

        self._parent: Optional["BaseRadiaComponent"] = None
        self._children_cache: Dict[int, BaseRadiaComponent] = {}
        self._symmetries: List[SymmetryTuple] = _as_symmetry_list(symmetries)
        self._tet_coords: Optional[List[List[List[float]]]] = None

        self._color: List[float] = [0.0, 0.5, 1.0]
        if color is not None:
            self.set_color(color, propagate=False, apply_color=apply_color)

        if parent is not None:
            self._set_parent(parent)

    @property
    def id(self) -> int:
        return self._id

    @property
    def is_container(self) -> bool:
        return self._is_container

    @property
    def symmetries(self) -> List[SymmetryTuple]:
        return list(self._symmetries)

    @property
    def tet_coords(self) -> Optional[List[List[List[float]]]]:
        """Tet vertex coordinates this component was meshed from (mesh-based
        components only: from_stp / from_gmsh_occ; None otherwise). One entry
        per child polyhedron, each a list of four [x, y, z] corners -- e.g.
        for building explicitly-replicated (unsymmetrized) copies."""
        return self._tet_coords

    def declare_symmetries(self, symmetries: SymmetryInput) -> None:
        """Record field symmetries as metadata WITHOUT applying radia transforms.

        Use for sources whose field is symmetric by construction (e.g. the
        coil pair). Deliberately does NOT propagate to children: a declaration
        describes this component AS A WHOLE (a +/-z coil pair is z-mirror
        symmetric while its individual coils are not). For magnetized parts
        that need the radia TrfZer* mirrors, use
        MagnetizedComponent.apply_symmetry instead (there the transform DOES
        cascade to the members, so propagation is correct).
        """
        self._symmetries.extend(_as_symmetry_list(symmetries))

    @property
    def color(self) -> List[float]:
        return self._color.copy()

    def get_color(self) -> List[float]:
        return self._color.copy()

    def set_color(self, color: Sequence[float], *, propagate: bool = True,
                  apply_color: bool = False) -> None:
        if len(color) != 3:
            raise ValueError("color must be length 3.")
        c = [float(color[0]), float(color[1]), float(color[2])]
        self._color = c

        if apply_color:
            _call_radia("ObjDrwAtr", self.id, c)

        if propagate:
            for child in self._children_cache.values():
                child._color = c.copy()

    def get_parent(self) -> Optional["BaseRadiaComponent"]:
        return self._parent

    def get_parent_id(self) -> Optional[int]:
        return None if self._parent is None else self._parent.id

    def _set_parent(self, parent: "BaseRadiaComponent") -> None:
        if self._parent is not None and self._parent is not parent:
            raise ParentAssignmentError(
                f"Component {self.id} already has a parent ({self._parent.id})."
            )
        self._parent = parent

    def get_child_ids(self) -> List[int]:
        return list(self._child_ids)

    def _spawn_child_wrapper(self, child_id: int) -> "BaseRadiaComponent":
        return BaseRadiaComponent(
            child_id,
            parent=self,
            symmetries=self._symmetries,
            color=self._color,
        )

    def iter_cached_children(self):
        """Iterate over the child wrappers that already exist (no lazy spawning).

        Used by the symmetry collector: containers built via containerize()
        hold real wrappers here, while anonymous leaf ids (e.g. STP-mesh tets)
        are represented by the container's own metadata.
        """
        return iter(self._children_cache.values())

    def get_children(self) -> List["BaseRadiaComponent"]:
        """
        Materialize wrappers for ALL child ids (eager).

        TODO(future): lazy / filtered access without building every wrapper
        (needed for STP loads with tens of thousands of tetrahedra):
          - get_child(index) / get_child_by_id(cid): synthesize + cache one child
          - num_children / iter_children(): stream via rad.ObjCntSize / rad.ObjCntStuf
          - find_children(predicate): e.g. select tets whose center (rad.ObjM) z < z0
        Also: when child_ids are not supplied, source them from rad.ObjCntStuf(self.id),
        and detect nested containers via (rad.ObjCntStuf(child) != [child]) rather than
        assuming every child is a leaf.
        """
        if not self._is_container:
            return []

        out: List[BaseRadiaComponent] = []
        for cid in self._child_ids:
            child = self._children_cache.get(cid)
            if child is None:
                child = self._spawn_child_wrapper(cid)
                self._children_cache[cid] = child
            out.append(child)
        return out

    def _forget_child(self, child_id: int) -> None:
        """Drop a child from this container's bookkeeping (after the child is disposed)."""
        self._children_cache.pop(child_id, None)
        self._child_ids = [cid for cid in self._child_ids if cid != child_id]

    def dispose(self, *, deep: bool = False) -> None:
        """Delete this component's radia object (``rad.UtiDel``).

        Radia objects are reference-counted handles; ``UtiDel`` removes only THIS
        object's key from the global table -- it does NOT cascade. Hence:
          - deep=False (default): delete only this object. Children keep their own
            keys (they survive); each child's parent pointer is reset so a survivor
            can be re-containerized into a fresh parent. Use on a top container whose
            members you want to keep (e.g. swap the coils, reuse the iron).
          - deep=True: dispose every child first, then this object. Use on a
            throwaway sub-container (e.g. the coils) to also free its leaf objects.
        Idempotent. After disposal ``id`` is None.
        """
        if self._id is None:
            return
        if deep:
            for child in self.get_children():
                child.dispose(deep=True)
        else:
            for child in list(self._children_cache.values()):
                child._parent = None
        try:
            _call_radia("UtiDel", self._id)
        except RuntimeError:
            # Key already gone (e.g. a prior rad.UtiDelAll wiped it) -> idempotent.
            pass
        if self._parent is not None:
            self._parent._forget_child(self._id)
            self._parent = None
        self._children_cache.clear()
        self._child_ids = []
        self._is_container = False
        self._symmetries = []
        self._id = None

    def transform(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("transform() is not implemented yet.")

    @classmethod
    def containerize(cls, components: Sequence["BaseRadiaComponent"]) -> "BaseRadiaComponent":
        comps = list(components)
        if len(comps) < 1:
            raise ValueError("containerize requires at least one component.")

        ids = [c.id for c in comps]
        container_id = _call_radia("ObjCnt", ids)
        container_id = _validate_radia_id(container_id, "container_id")

        container = BaseRadiaComponent(
            container_id,
            child_ids=ids,
            is_container=True,
            color=comps[0].color if len(comps) > 0 else [0.0, 0.5, 1.0],
        )
        container._children_cache = {c.id: c for c in comps}
        for c in comps:
            c._set_parent(container)
        return container

    @staticmethod
    def _coerce_build_result(result: BuilderResult) -> Tuple[int, List[int], bool]:
        if isinstance(result, int):
            rid = _validate_radia_id(result, "build id")
            return rid, [], False

        if isinstance(result, Mapping):
            rid = _validate_radia_id(result["id"], "build id")
            child_ids = list(result.get("child_ids", []))
            is_container = bool(result.get("is_container", len(child_ids) > 0))
            return rid, child_ids, is_container

        ids = list(result)
        if len(ids) == 0:
            raise ValueError("build() returned an empty id sequence.")
        if len(ids) == 1:
            rid = _validate_radia_id(ids[0], "build id")
            return rid, [], False

        ids = [_validate_radia_id(v, "build child id") for v in ids]
        container_id = _call_radia("ObjCnt", ids)
        container_id = _validate_radia_id(container_id, "build container id")
        return container_id, ids, True


# -----------------------------
# Magnetized base class
# -----------------------------
class MagnetizedComponent(BaseRadiaComponent):
    """
    Adds:
      - material metadata / application
      - symmetry APPLICATION (radia TrfZer* transforms; the metadata itself
        lives on BaseRadiaComponent)
    """

    def __init__(
        self,
        radia_id: int,
        *,
        child_ids: Optional[Sequence[int]] = None,
        is_container: Optional[bool] = None,
        parent: Optional[BaseRadiaComponent] = None,
        symmetries: Optional[SymmetryInput] = None,
        material: Optional[RadiaMaterial] = None,
        color: Optional[Sequence[float]] = None,
        apply_sym: bool = False,
        apply_mat: bool = False,
        apply_color: bool = False,
    ) -> None:
        super().__init__(
            radia_id,
            child_ids=child_ids,
            is_container=is_container,
            parent=parent,
            color=color,
            apply_color=apply_color,
        )

        self._material: Optional[RadiaMaterial] = None

        if symmetries is not None:
            self._add_symmetries(symmetries, apply_sym=apply_sym)

        if material is not None:
            self.set_material(material, apply_mat=apply_mat)

    @property
    def material(self) -> Optional[RadiaMaterial]:
        return self._material

    def get_material(self) -> Optional[RadiaMaterial]:
        return self._material

    def _apply_single_symmetry(self, sym: SymmetryTuple) -> None:
        kind, point, normal = sym
        kind = str(kind).lower()

        if kind == "perp":
            _call_radia("TrfZerPerp", self.id, point, normal)
        elif kind == "para":
            _call_radia("TrfZerPara", self.id, point, normal)
        else:
            raise ValueError("symmetry type must be 'perp' or 'para'.")

    def _add_symmetries(self, symmetries: SymmetryInput, *, apply_sym: bool) -> None:
        for sym in _as_symmetry_list(symmetries):
            if apply_sym:
                self._apply_single_symmetry(sym)
            self._symmetries.append(sym)

        for child in self._children_cache.values():
            child._symmetries = list(self._symmetries)

    def apply_symmetry(self, symmetries: SymmetryInput) -> None:
        self._add_symmetries(symmetries, apply_sym=True)

    def set_material(self, material: RadiaMaterial, *, apply_mat: bool = True) -> None:
        if apply_mat:
            if material.material is None:
                raise ValueError(
                    "material.material is None. Use from_radia_material(...) or "
                    "from_bh_file(...), or set apply_mat=False."
                )
            _call_radia("MatApl", self.id, material.material)

        self._material = material

        for child in self._children_cache.values():
            if isinstance(child, MagnetizedComponent):
                child._material = material

    def _spawn_child_wrapper(self, child_id: int) -> "BaseRadiaComponent":
        return MagnetizedComponent(
            child_id,
            parent=self,
            symmetries=self._symmetries,
            material=self._material,
            color=self._color,
            apply_sym=False,
            apply_mat=False,
        )

    @classmethod
    def containerize(cls, components: Sequence["MagnetizedComponent"]) -> "MagnetizedComponent":
        comps = list(components)
        if len(comps) < 1:
            raise ValueError("containerize requires at least one component.")

        ids = [c.id for c in comps]
        container_id = _call_radia("ObjCnt", ids)
        container_id = _validate_radia_id(container_id, "container_id")

        shared_material = comps[0].material
        if not all(c.material is shared_material for c in comps):
            shared_material = None

        shared_sym = comps[0].symmetries
        if not all(c.symmetries == shared_sym for c in comps):
            shared_sym = []

        shared_color = comps[0].color
        if not all(c.color == shared_color for c in comps):
            shared_color = [0.0, 0.5, 1.0]

        container = MagnetizedComponent(
            container_id,
            child_ids=ids,
            is_container=True,
            symmetries=shared_sym,
            material=shared_material,
            color=shared_color,
            apply_sym=False,
            apply_mat=False,
        )
        container._children_cache = {c.id: c for c in comps}
        for c in comps:
            c._set_parent(container)
        return container

    @classmethod
    def from_tet_coords(
        cls,
        tet_coords: List[List[List[float]]],
        *,
        symmetries: Optional[SymmetryInput] = None,
        material: Optional[RadiaMaterial] = None,
        color: Optional[Sequence[float]] = None,
        apply_sym: bool = False,
        apply_mat: bool = False,
        apply_color: bool = False,
    ) -> "MagnetizedComponent":
        """Build the component from an explicit tet vertex list (the common
        tail of from_stp / from_gmsh_occ / build_conforming_group)."""
        if not tet_coords:
            raise RadiaComponentError("No tetrahedra to convert into radia objects.")
        tet_ids = [_tet_to_polyhedron(t) for t in tet_coords]
        container_id = _validate_radia_id(_call_radia("ObjCnt", tet_ids), "container_id")
        comp = cls(
            container_id,
            child_ids=tet_ids,
            is_container=True,
            symmetries=symmetries,
            material=material,
            color=color,
            apply_sym=apply_sym,
            apply_mat=apply_mat,
            apply_color=apply_color,
        )
        comp._tet_coords = tet_coords
        return comp

    @classmethod
    def from_stp(
        cls,
        stp_path: Union[str, Path],
        *,
        mesh_size_min: Optional[float] = None,
        mesh_size_max: Optional[float] = None,
        gmsh_terminal_output: bool = False,
        model_name: Optional[str] = None,
        comm: Any = None,
        symmetries: Optional[SymmetryInput] = None,
        material: Optional[RadiaMaterial] = None,
        color: Optional[Sequence[float]] = None,
        apply_sym: bool = False,
        apply_mat: bool = False,
        apply_color: bool = False,
    ) -> "MagnetizedComponent":
        path = Path(stp_path)
        if not path.exists():
            raise FileNotFoundError(f"STP file not found: {path}")

        try:
            import gmsh
        except Exception as exc:  # pragma: no cover
            raise RadiaComponentError("gmsh is required for from_stp(...).") from exc

        # MPI-safe (mirrors from_gmsh_occ): mesh the STP on rank 0 only and broadcast the
        # tet vertex list, so every rank builds identical radia ids. comm falls back to
        # MPI.COMM_WORLD when MPI is initialized (see _resolve_comm).
        comm = _resolve_comm(comm)
        rank = comm.Get_rank() if comm is not None else 0

        tet_coords: Optional[List[List[List[float]]]] = None
        if rank <= 0:
            gmsh.initialize()
            try:
                gmsh.option.setNumber("General.Terminal", 1 if gmsh_terminal_output else 0)
                _pin_gmsh_determinism()
                gmsh.model.add(model_name or path.stem)
                gmsh.merge(str(path))
                gmsh.model.occ.synchronize()
                if mesh_size_min is not None:
                    gmsh.option.setNumber("Mesh.MeshSizeMin", float(mesh_size_min))
                if mesh_size_max is not None:
                    gmsh.option.setNumber("Mesh.MeshSizeMax", float(mesh_size_max))
                gmsh.model.mesh.generate(3)
                tet_coords = _extract_tet_coords()
            finally:
                gmsh.finalize()

        if comm is not None:
            tet_coords = comm.bcast(tet_coords, root=0)

        return cls.from_tet_coords(
            tet_coords or [],
            symmetries=symmetries, material=material, color=color,
            apply_sym=apply_sym, apply_mat=apply_mat, apply_color=apply_color,
        )

    @classmethod
    def from_stp_structured(
        cls,
        stp_path: Union[str, Path],
        *,
        structure: Optional[Dict[str, Any]] = None,
        mesh_size_min: Optional[float] = None,
        mesh_size_max: Optional[float] = None,
        model_name: Optional[str] = None,
        comm: Any = None,
        symmetries: Optional[SymmetryInput] = None,
        material: Optional[RadiaMaterial] = None,
        color: Optional[Sequence[float]] = None,
        apply_sym: bool = False,
        apply_mat: bool = False,
        apply_color: bool = False,
        gmsh_verbosity: int = 2,
    ) -> "MagnetizedComponent":
        """Structured polar-grid discretization of an STP solid.

        The solid is OCC-fragmented into annular rings snapped to its own
        CAD radii / z-planes; clean interior rings become analytic
        annular-sector PRISM elements (rad.ObjPolyhdr), everything touching
        true CAD detail becomes a conforming tet SKIN. Structured cores
        condition the relaxation far better at a fraction of the element
        count (see geometry/structured.py and RECMAG_GPU_PLAN.md).

        MPI-safe like from_stp: rank 0 slices/meshes, the payload is
        broadcast, every rank builds identical radia ids.
        """
        from cyclotron_optimizer.geometry import structured as _st

        path = Path(stp_path)
        if not path.exists():
            raise FileNotFoundError(f"STP file not found: {path}")

        comm = _resolve_comm(comm)
        rank = comm.Get_rank() if comm is not None else 0

        payload: Optional[Dict[str, Any]] = None
        if rank <= 0:
            payload = _st.slice_stp_polar(
                str(path),
                structure=structure,
                mesh_size_max=mesh_size_max,
                mesh_size_min=mesh_size_min,
                model_name=model_name or path.stem,
                gmsh_verbosity=gmsh_verbosity,
            )
        if comm is not None:
            payload = comm.bcast(payload, root=0)
        assert payload is not None

        prism_ids, cell_defs = _st.emit_prism_cells(payload)
        skin_ids = [_tet_to_polyhedron(t) for t in payload["skin_tets"]]
        child_ids = prism_ids + skin_ids
        if not child_ids:
            raise RadiaComponentError(
                f"Structured slicing of '{path.name}' produced no elements.")

        container_id = _validate_radia_id(_call_radia("ObjCnt", child_ids),
                                          "container_id")
        comp = cls(
            container_id,
            child_ids=child_ids,
            is_container=True,
            symmetries=symmetries,
            material=material,
            color=color,
            apply_sym=apply_sym,
            apply_mat=apply_mat,
            apply_color=apply_color,
        )
        comp._tet_coords = payload["skin_tets"]
        comp._structured_cells = cell_defs
        comp._structured_stats = payload["stats"]
        return comp

    @classmethod
    def from_gmsh_occ(
        cls,
        occ_builder: Callable[[], None],
        *,
        model_name: str = "model",
        mesh_size_min: Optional[float] = None,
        mesh_size_max: Optional[float] = None,
        comm: Any = None,
        symmetries: Optional[SymmetryInput] = None,
        material: Optional[RadiaMaterial] = None,
        color: Optional[Sequence[float]] = None,
        apply_sym: bool = False,
        apply_mat: bool = False,
        apply_color: bool = False,
        gmsh_verbosity: int = 3,
    ) -> "MagnetizedComponent":
        """Build a meshed iron component from an in-memory gmsh-OCC model.

        ``occ_builder`` is a no-arg callable that adds OCC volume(s) to the
        current gmsh model (gmsh is already initialized and a model added). The
        model is then synchronized, meshed to tetrahedra, and each tet becomes a
        radia polyhedron (single container returned).

        With an MPI communicator the mesh is built on rank 0 only and the tet
        vertex list is broadcast, so every rank constructs identical radia ids.
        """
        try:
            import gmsh
        except Exception as exc:  # pragma: no cover
            raise RadiaComponentError("gmsh is required for from_gmsh_occ(...).") from exc

        comm = _resolve_comm(comm)
        rank = comm.Get_rank() if comm is not None else 0

        tet_coords: Optional[List[List[List[float]]]] = None
        if rank <= 0:
            gmsh.initialize()
            try:
                gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
                _pin_gmsh_determinism()
                gmsh.model.add(model_name)
                occ_builder()
                gmsh.model.occ.synchronize()
                if mesh_size_min is not None:
                    gmsh.option.setNumber("Mesh.MeshSizeMin", float(mesh_size_min))
                if mesh_size_max is not None:
                    gmsh.option.setNumber("Mesh.MeshSizeMax", float(mesh_size_max))
                gmsh.model.mesh.generate(3)
                tet_coords = _extract_tet_coords()
            finally:
                gmsh.finalize()

        if comm is not None:
            tet_coords = comm.bcast(tet_coords, root=0)

        if not tet_coords:
            raise RadiaComponentError(f"No tetrahedra generated for '{model_name}'.")

        return cls.from_tet_coords(
            tet_coords,
            symmetries=symmetries, material=material, color=color,
            apply_sym=apply_sym, apply_mat=apply_mat, apply_color=apply_color,
        )


# -----------------------------
# Conforming mesh groups (opt-in via ComponentSpec.mesh_group)
# -----------------------------
def build_conforming_group(
    entries: List[Dict[str, Any]],
    *,
    group_name: str = "group",
    comm: Any = None,
    gmsh_verbosity: int = 3,
    export_stp_path: Optional[Union[str, Path]] = None,
    mesh: bool = True,
) -> Dict[str, List[List[List[float]]]]:
    """Mesh several touching components in ONE gmsh model with CONFORMING
    interfaces, and return per-component tet vertex lists.

    Each entry: ``{"name": str, "stp_path": str | None, "occ": callable |
    None, "mesh_max": float, "mesh_min": float | None}`` — exactly one of
    stp_path / occ per entry. All volumes are boolean-FRAGMENTED against
    each other, so shared (touching) surfaces become single entities with a
    single triangulation: tets on both sides share faces node-for-node.
    This removes the non-conforming contact interfaces that make radia's
    center-collocation equations inconsistent on refined meshes (see
    scripts/perturb_study/FLOOR_ANALYSIS.md).

    Per-component mesh sizes are applied on each part's boundary points,
    LARGEST first, so points on shared interfaces end up with the smallest
    adjoining size (gmsh grades outward from there). Volumes overlapping in
    the CAD (not merely touching) are an error.

    ``export_stp_path``: write the FRAGMENTED (conforming) geometry to a
    STEP file — the exact solids the mesh is generated from, for external
    gold-standard runs (COMSOL imports one file with the interfaces already
    imprinted). ``mesh=False`` skips meshing entirely (export-only call;
    returns empty tet lists).

    MPI-safe like the other factories: rank 0 meshes, the dict of tet lists
    is broadcast.
    """
    try:
        import gmsh
    except Exception as exc:  # pragma: no cover
        raise RadiaComponentError("gmsh is required for build_conforming_group(...).") from exc

    comm = _resolve_comm(comm)
    rank = comm.Get_rank() if comm is not None else 0

    result: Optional[Dict[str, List[List[List[float]]]]] = None
    if rank <= 0:
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
            _pin_gmsh_determinism()
            gmsh.model.add(f"mesh_group_{group_name}")

            owner_of: Dict[int, str] = {}
            for e in entries:
                before = {t for _d, t in gmsh.model.occ.getEntities(3)}
                if e.get("stp_path"):
                    gmsh.model.occ.importShapes(str(e["stp_path"]))
                elif e.get("occ") is not None:
                    e["occ"]()
                else:
                    raise RadiaComponentError(
                        f"mesh_group '{group_name}': entry '{e['name']}' has "
                        "neither stp_path nor occ builder")
                after = {t for _d, t in gmsh.model.occ.getEntities(3)}
                new = after - before
                if not new:
                    raise RadiaComponentError(
                        f"mesh_group '{group_name}': entry '{e['name']}' "
                        "added no OCC volume")
                for t in new:
                    owner_of[t] = e["name"]

            # Fragment everything against everything: conforming interfaces.
            # OCC may emit BOPAlgo_AlertFaceBuilderUnusedEdges here (unused
            # intersection edges while merging coincident faces) -- benign
            # per se, but we VERIFY the invariant that would break if the
            # imprint actually failed: per-component volume conservation.
            vol_in = {e["name"]: 0.0 for e in entries}
            for t, n in owner_of.items():
                vol_in[n] += gmsh.model.occ.getMass(3, t)
            in_tags = sorted(owner_of)
            if len(in_tags) > 1:
                _out, out_map = gmsh.model.occ.fragment(
                    [(3, t) for t in in_tags], [])
                new_owner: Dict[int, str] = {}
                for in_tag, images in zip(in_tags, out_map):
                    for d, t in images:
                        if d != 3:
                            continue
                        prev = new_owner.get(t)
                        if prev is not None and prev != owner_of[in_tag]:
                            raise RadiaComponentError(
                                f"mesh_group '{group_name}': components "
                                f"'{prev}' and '{owner_of[in_tag]}' OVERLAP "
                                "(fragment produced a shared piece); fix the "
                                "geometry so parts only touch")
                        new_owner[t] = owner_of[in_tag]
                owner_of = new_owner

                vol_out = {e["name"]: 0.0 for e in entries}
                for t, n in owner_of.items():
                    vol_out[n] += gmsh.model.occ.getMass(3, t)
                for n, vi in vol_in.items():
                    vo = vol_out[n]
                    rel = abs(vo - vi) / max(abs(vi), 1e-30)
                    if rel > 1e-3:
                        raise RadiaComponentError(
                            f"mesh_group '{group_name}': fragment changed the "
                            f"volume of '{n}' by {rel:.2e} relative "
                            f"({vi:.6g} -> {vo:.6g} mm^3) -- the boolean "
                            "imprint failed; inspect the CAD contact")
                    if rel > 1e-6:
                        print(f"[mesh_group {group_name}] WARNING: volume of "
                              f"'{n}' shifted by {rel:.2e} relative in the "
                              f"fragment ({vi:.6g} -> {vo:.6g} mm^3)",
                              flush=True)
                print(f"[mesh_group {group_name}] volume conservation OK "
                      f"(max rel dev "
                      f"{max(abs(vol_out[n] - vol_in[n]) / max(abs(vol_in[n]), 1e-30) for n in vol_in):.1e})",
                      flush=True)
            gmsh.model.occ.synchronize()

            # Diagnostics: surfaces adjacent to volumes of DIFFERENT owners
            # are the conforming interfaces. If an expected contact reports
            # none, the surfaces did not geometrically coincide (e.g. an
            # approximated face against a true cylinder) and the contact is
            # still non-conforming.
            shared: Dict[tuple, int] = {}
            for _d, s in gmsh.model.getEntities(2):
                up, _down = gmsh.model.getAdjacencies(2, s)
                owners = {owner_of.get(int(v)) for v in up}
                owners.discard(None)
                if len(owners) > 1:
                    key = tuple(sorted(owners))
                    shared[key] = shared.get(key, 0) + 1
            msg = (", ".join(f"{a}~{b}: {n}" for (a, b), n in sorted(shared.items()))
                   if shared else "NONE FOUND")
            print(f"[mesh_group {group_name}] conforming interface surfaces: {msg}",
                  flush=True)

            if export_stp_path is not None:
                out = str(export_stp_path)
                gmsh.write(out)
                print(f"[mesh_group {group_name}] fragmented geometry "
                      f"exported: {out}", flush=True)

            if not mesh:
                result = {e["name"]: [] for e in entries}
            else:
                # Per-part sizes: apply on boundary points, largest size
                # first, so shared-interface points keep the smallest
                # adjoining size.
                sizes = {e["name"]: float(e["mesh_max"]) for e in entries}
                mins = [float(e["mesh_min"]) for e in entries
                        if e.get("mesh_min") is not None]
                gmsh.option.setNumber("Mesh.MeshSizeMax", max(sizes.values()))
                gmsh.option.setNumber("Mesh.MeshSizeMin",
                                      min(mins) if mins else 1.0)
                for e in sorted(entries, key=lambda x: -float(x["mesh_max"])):
                    vol_tags = [(3, t) for t, n in owner_of.items()
                                if n == e["name"]]
                    pts = gmsh.model.getBoundary(vol_tags, combined=False,
                                                 oriented=False, recursive=True)
                    pts = [(d, t) for d, t in pts if d == 0]
                    if pts:
                        gmsh.model.mesh.setSize(pts, float(e["mesh_max"]))

                gmsh.model.mesh.generate(3)

                # Split the tets back per component by volume tag.
                node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
                nodes: Dict[int, List[float]] = {}
                for i, tag in enumerate(node_tags):
                    j = 3 * i
                    nodes[int(tag)] = [float(node_coords[j]),
                                       float(node_coords[j + 1]),
                                       float(node_coords[j + 2])]
                result = {e["name"]: [] for e in entries}
                for t in sorted(owner_of):
                    name = owner_of[t]
                    etypes, _etags, enodes = gmsh.model.mesh.getElements(3, t)
                    for et, conn in zip(etypes, enodes):
                        if int(et) != 4:
                            continue
                        for i in range(0, len(conn), 4):
                            result[name].append(
                                [nodes[int(c)] for c in conn[i:i + 4]])
        finally:
            gmsh.finalize()

    if comm is not None:
        result = comm.bcast(result, root=0)

    assert result is not None
    if mesh:
        empty = [n for n, tets in result.items() if not tets]
        if empty:
            raise RadiaComponentError(
                f"mesh_group '{group_name}': no tets generated for {empty}")
    return result


# -----------------------------
# Current-carrying base class
# -----------------------------
class CurrentCarryingComponent(BaseRadiaComponent):
    """
    Structural + current-carrying behavior.
    No symmetry/material API here.
    """

    pass


# -----------------------------
# Concrete classes
# -----------------------------
class AnnularWedge(MagnetizedComponent):
    """
    Concrete magnetized geometry.
    Override build() with the exact RadiaCUDA calls used in your repository.
    """

    def __init__(
        self,
        *,
        r_inner: float,
        r_outer: float,
        z_min: float,
        z_max: float,
        phi_min_deg: float,
        phi_max_deg: float,
        center_xy: Tuple[float, float] = (0.0, 0.0),
        magnetization: Optional[Sequence[float]] = None,
        angular_resolution_deg: float = 2.5,
        symmetries: Optional[SymmetryInput] = None,
        material: Optional[RadiaMaterial] = None,
        color: Optional[Sequence[float]] = None,
        apply_sym: bool = False,
        apply_mat: bool = False,
        apply_color: bool = False,
    ) -> None:
        if r_outer <= r_inner:
            raise ValueError("r_outer must be greater than r_inner.")
        if z_max <= z_min:
            raise ValueError("z_max must be greater than z_min.")
        if phi_max_deg <= phi_min_deg:
            raise ValueError("phi_max_deg must be greater than phi_min_deg.")

        self.r_inner = r_inner
        self.r_outer = r_outer
        self.z_min = z_min
        self.z_max = z_max
        self.phi_min_deg = phi_min_deg
        self.phi_max_deg = phi_max_deg
        self.center_xy = center_xy
        self.magnetization = magnetization
        self.angular_resolution_deg = angular_resolution_deg

        build_result = self.build()
        rid, child_ids, is_container = self._coerce_build_result(build_result)

        super().__init__(
            rid,
            child_ids=child_ids,
            is_container=is_container,
            symmetries=symmetries,
            material=material,
            color=color,
            apply_sym=apply_sym,
            apply_mat=apply_mat,
            apply_color=apply_color,
        )

    def build(self) -> BuilderResult:
        """
        Build an annular-wedge prism as a uniformly magnetized polyhedron via
        rad.ObjMltExtPgn: an arc-sector polygon (outer arc + reversed inner arc)
        extruded between z_min and z_max.
        """
        import numpy as np

        cx, cy = self.center_xy
        phi0 = np.deg2rad(self.phi_min_deg)
        phi1 = np.deg2rad(self.phi_max_deg)
        n_arc = max(
            1, int(np.ceil((self.phi_max_deg - self.phi_min_deg) / self.angular_resolution_deg))
        )
        angles = np.linspace(phi0, phi1, n_arc + 1)

        outer = [[cx + self.r_outer * np.cos(a), cy + self.r_outer * np.sin(a)] for a in angles]
        inner = [[cx + self.r_inner * np.cos(a), cy + self.r_inner * np.sin(a)] for a in angles[::-1]]
        polygon = outer + inner

        mag = list(self.magnetization) if self.magnetization is not None else [0.0, 0.0, 0.0]
        return _call_radia(
            "ObjMltExtPgn",
            [[polygon, self.z_min], [polygon, self.z_max]],
            mag,
        )


class Coil(CurrentCarryingComponent):
    """
    Racetrack current-carrying coil (radia ObjRaceTrk).
    No symmetry/material handling by design.

    A pure circular coil is a racetrack with zero straight-section lengths
    (the default). `current` is the total (signed) current in Amperes; the
    azimuthal current density is current / (height * (r_outer - r_inner)).
    """

    def __init__(
        self,
        *,
        radius_min_mm: float,
        radius_max_mm: float,
        height_mm: float,
        current: float,
        num_segments: int = 20,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        straight_lengths: Tuple[float, float] = (0.0, 0.0),
        axis: str = "z",
        color: Optional[Sequence[float]] = None,
        apply_color: bool = False,
    ) -> None:
        if radius_max_mm <= radius_min_mm:
            raise ValueError("radius_max_mm must be greater than radius_min_mm.")
        if height_mm <= 0:
            raise ValueError("height_mm must be > 0.")
        if num_segments <= 0:
            raise ValueError("num_segments must be > 0.")

        self.radius_min_mm = float(radius_min_mm)
        self.radius_max_mm = float(radius_max_mm)
        self.height_mm = float(height_mm)
        self.current = float(current)
        self.num_segments = int(num_segments)
        self.center = (float(center[0]), float(center[1]), float(center[2]))
        self.straight_lengths = (float(straight_lengths[0]), float(straight_lengths[1]))
        self.axis = str(axis)

        # Azimuthal current density (A/mm^2) over the rectangular cross-section.
        self.current_density = self.current / (
            self.height_mm * (self.radius_max_mm - self.radius_min_mm)
        )

        build_result = self.build()
        rid, child_ids, is_container = self._coerce_build_result(build_result)

        super().__init__(
            rid,
            child_ids=child_ids,
            is_container=is_container,
            color=color,
            apply_color=apply_color,
        )

    def build(self) -> BuilderResult:
        """Build a racetrack coil via rad.ObjRaceTrk."""
        return _call_radia(
            "ObjRaceTrk",
            list(self.center),
            [self.radius_min_mm, self.radius_max_mm],
            list(self.straight_lengths),
            self.height_mm,
            self.num_segments,
            self.current_density,
            "man",
            self.axis,
        )


__all__ = [
    "RadiaComponentError",
    "RadiaUnavailableError",
    "ParentAssignmentError",
    "RadiaMaterial",
    "BaseRadiaComponent",
    "MagnetizedComponent",
    "CurrentCarryingComponent",
    "AnnularWedge",
    "Coil",
]
