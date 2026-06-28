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


# -----------------------------
# Material wrapper
# -----------------------------
class RadiaMaterial:
    """
    TODO: Implement other material types.
    """

    def __init__(self, name: str = "material", metadata: Optional[Dict[str, Any]] = None) -> None:
        self._name = name
        self._material_object = None
        self._filename = None
        self._metadata = metadata

    @property
    def material(self):
        return self._material_object

    @property
    def filename(self):
        return self._filename

    @property
    def metadata(self):
        return self._metadata

    def get_bh_curve(self, units: str = "TT"):
        # TODO: implement get_bh_curve
        pass

    def get_mh_curve(self, units: str = "TT"):
        # TODO: implement get_mh_curve
        pass

    def plot_bh_curve(self):
        # TODO: implement plot_bh_curve
        pass

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
    def from_bh_file(cls, filename: str, type: str = "BH", name: str = "material") -> "RadiaMaterial":
        if filename is None:
            raise ValueError("filename cannot be None")

        try:
            import numpy as np
        except Exception as exc:
            raise ImportError("numpy is required for from_bh_file") from exc

        tmp_cls = cls(name)
        tmp_cls._filename = filename

        full_radialib_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "radialib"
        )
        full_path = filename if os.path.isabs(filename) else os.path.join(full_radialib_path, filename)

        data = np.genfromtxt(full_path, delimiter=",").tolist()
        if type == "BH":
            data = [[row[0], row[1] - row[0]] for row in data]
        elif type == "MH":
            pass
        else:
            raise ValueError("type must be BH or MH")

        tmp_cls._material_object = _call_radia("MatSatIsoTab", data)
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
    """

    def __init__(
        self,
        radia_id: int,
        *,
        child_ids: Optional[Sequence[int]] = None,
        is_container: Optional[bool] = None,
        parent: Optional["BaseRadiaComponent"] = None,
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
            color=self._color,
        )

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

    def transform(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("transform() is not implemented yet.")

    @classmethod
    def containerize(cls, components: Sequence["BaseRadiaComponent"]) -> "BaseRadiaComponent":
        comps = list(components)
        if len(comps) < 2:
            raise ValueError("containerize requires at least two components.")

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
      - symmetry metadata / application
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

        self._symmetries: List[SymmetryTuple] = []
        self._material: Optional[RadiaMaterial] = None

        if symmetries is not None:
            self._add_symmetries(symmetries, apply_sym=apply_sym)

        if material is not None:
            self.set_material(material, apply_mat=apply_mat)

    @property
    def symmetries(self) -> List[SymmetryTuple]:
        return list(self._symmetries)

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
            if isinstance(child, MagnetizedComponent):
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
        if len(comps) < 2:
            raise ValueError("containerize requires at least two components.")

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
    def from_stp(
        cls,
        stp_path: Union[str, Path],
        *,
        mesh_size_min: Optional[float] = None,
        mesh_size_max: Optional[float] = None,
        gmsh_terminal_output: bool = False,
        model_name: Optional[str] = None,
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

        tet_ids: List[int] = []
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 1 if gmsh_terminal_output else 0)
            gmsh.model.add(model_name or path.stem)
            gmsh.merge(str(path))
            gmsh.model.occ.synchronize()

            if mesh_size_min is not None:
                gmsh.option.setNumber("Mesh.MeshSizeMin", float(mesh_size_min))
            if mesh_size_max is not None:
                gmsh.option.setNumber("Mesh.MeshSizeMax", float(mesh_size_max))

            gmsh.model.mesh.generate(3)

            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            tag_to_xyz: Dict[int, Vertex] = {}
            for i, tag in enumerate(node_tags):
                j = 3 * i
                tag_to_xyz[int(tag)] = (
                    float(node_coords[j]),
                    float(node_coords[j + 1]),
                    float(node_coords[j + 2]),
                )

            elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=3)

            for elem_type, conn in zip(elem_types, elem_node_tags):
                name, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(elem_type)
                if "tetra" not in name.lower():
                    continue

                for i in range(0, len(conn), num_nodes):
                    local_nodes = conn[i : i + num_nodes]
                    corner_tags = local_nodes[:4]
                    vertices = [tag_to_xyz[int(n)] for n in corner_tags]
                    rid = _tet_to_polyhedron(vertices)
                    tet_ids.append(_validate_radia_id(rid, "tet id"))

            if len(tet_ids) == 0:
                raise RadiaComponentError("No tetrahedra converted into radia objects.")

            container_id = _call_radia("ObjCnt", tet_ids)
            container_id = _validate_radia_id(container_id, "container_id")

            return MagnetizedComponent(
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
        finally:
            gmsh.finalize()

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

        rank = comm.Get_rank() if comm is not None else 0

        tet_coords: Optional[List[List[List[float]]]] = None
        if rank <= 0:
            gmsh.initialize()
            try:
                gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
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

        tet_ids = [_tet_to_polyhedron(t) for t in tet_coords]
        container_id = _validate_radia_id(_call_radia("ObjCnt", tet_ids), "container_id")

        return cls(
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
