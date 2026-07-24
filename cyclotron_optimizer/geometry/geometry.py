"""Cyclotron geometry assembly, driven by the config's component list.

Every machine part is a ComponentSpec (name / kind / params / material /
symmetry / mesh / file), and each ``kind`` maps to a builder in the registry
below ('stp', 'wedge', 'lid_upper', 'pole', 'swept_polygon',
'racetrack_pair'). Adding a part is a YAML entry plus -- at most -- one
registered builder. A mirrored piece (e.g. the extraction channel wedge
straddling the median plane) is a single component plus a 'para' symmetry
(e.g. 'median_z') -- the old 'wedge_pair' kind is gone.

Assembly groups the enabled magnetized components by their named symmetry:
each group is containerized and its symmetry set applied (TrfZer*); groups
without a symmetry stay untransformed (e.g. the extraction channel). Current
sources DECLARE their symmetry as metadata for the field evaluator instead.

The build is split for the optimizer's reuse levels (see
ReusableCyclotronSolver): build_static_iron_parts (yoke/lids/channel, built
once) / build_pole_part (the shimmed pole, rebuilt per iterate) /
assemble_iron (containerize + symmetrize, per assembly) / build_coils
(rebuilt per coil current).

Legacy configs are adapted to component specs in config_io.config.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import radia as rad

from cyclotron_optimizer.config_io.config import (
    ComponentSpec,
    CyclotronConfig,
    DEFAULT_CYCLOTRON_SYMMETRIES,
)
from cyclotron_optimizer.geometry.components import (
    BaseRadiaComponent,
    Coil,
    CurrentCarryingComponent,
    MagnetizedComponent,
    RadiaMaterial,
)
from cyclotron_optimizer.geometry import gmsh_builders as gb

# Backward-compatible alias; the source of truth lives in config_io.config
# (the legacy-schema adapter emits it as the named symmetry 'cyclotron_8fold').
CYCLOTRON_SYMMETRIES = DEFAULT_CYCLOTRON_SYMMETRIES

IRON_COLOR = [0.0, 0.5, 1.0]
COIL_COLOR = [1.0, 0.0, 0.0]

# Kinds that are current sources (built per coil current, never symmetrized
# via radia transforms -- they declare their field symmetry instead).
CURRENT_KINDS = {"racetrack_pair"}


# ---------------------------------------------------------------------------
# Build context + builder registry
# ---------------------------------------------------------------------------
class BuildContext:
    """Everything a kind builder may need beyond its own spec."""

    def __init__(self, config: CyclotronConfig,
                 materials: Dict[str, RadiaMaterial],
                 pole_shape=None, comm=None):
        self.config = config
        self.materials = materials
        self.pole_shape = pole_shape
        self.comm = comm

    def material(self, spec: ComponentSpec) -> Optional[RadiaMaterial]:
        if spec.material is None:
            return None
        try:
            return self.materials[spec.material]
        except KeyError:
            raise KeyError(f"Component {spec.name!r} references undefined "
                           f"material {spec.material!r}")


_BUILDERS: Dict[str, Callable] = {}


def register_builder(kind: str):
    """Register a component builder: fn(spec, ctx) -> BaseRadiaComponent."""
    def _decorate(fn):
        _BUILDERS[kind] = fn
        return fn
    return _decorate


def build_component(spec: ComponentSpec, ctx: BuildContext) -> BaseRadiaComponent:
    try:
        builder = _BUILDERS[spec.kind]
    except KeyError:
        raise KeyError(f"Unknown component kind {spec.kind!r} "
                       f"(component {spec.name!r}); registered: "
                       f"{sorted(_BUILDERS)}")
    return builder(spec, ctx)


def build_materials(config: CyclotronConfig) -> Dict[str, RadiaMaterial]:
    """Instantiate the named materials (radia objects; rebuilt after UtiDelAll)."""
    out: Dict[str, RadiaMaterial] = {}
    for name, mdef in (config.materials_def or {}).items():
        mtype = mdef.get("type", "bh_file")
        if mtype == "bh_file":
            out[name] = RadiaMaterial.from_bh_file(
                mdef["file"], curve=mdef.get("curve", "BH"),
                h_unit=mdef.get("h_unit", "T"),
                b_unit=mdef.get("b_unit", "T"), name=name)
        elif mtype == "sat_iso_frm":
            mat_id = rad.MatSatIsoFrm(mdef["saturation_field_t"],
                                      mdef["saturation_curve_m"],
                                      mdef["linear_curve_m"])
            out[name] = RadiaMaterial.from_radia_material(mat_id, name=name)
        else:
            raise ValueError(f"Unknown material type {mtype!r} for {name!r}")
    return out


def _from_stp(stp_filename, max_mesh_size, model_name, material, *, comm=None,
              min_mesh_size=None) -> MagnetizedComponent:
    return MagnetizedComponent.from_stp(
        stp_filename,
        mesh_size_min=min_mesh_size,
        mesh_size_max=max_mesh_size,
        model_name=model_name,
        comm=comm,
        material=material,
        color=IRON_COLOR,
        apply_mat=material is not None,
        apply_color=True,
    )


# ---------------------------------------------------------------------------
# Kind builders
# ---------------------------------------------------------------------------
@register_builder("stp")
def _kind_stp(spec: ComponentSpec, ctx: BuildContext) -> MagnetizedComponent:
    """Magnetized part from an STP file: tet-meshed, or -- with a
    `structure:` block -- polar-grid structured (prism core + tet skin)."""
    if not spec.file:
        raise ValueError(f"Component {spec.name!r} (kind 'stp') needs a file")
    if spec.structure is not None:
        # (grouped structured specs are handled by _build_mesh_group's
        # structured path; this builder only sees ungrouped components)
        mat = ctx.material(spec)
        return MagnetizedComponent.from_stp_structured(
            spec.file,
            structure=spec.structure,
            mesh_size_max=spec.mesh.get("max_size"),
            mesh_size_min=spec.mesh.get("min_size"),
            model_name=spec.name,
            comm=ctx.comm,
            material=mat,
            color=IRON_COLOR,
            apply_mat=mat is not None,
            apply_color=True,
        )
    return _from_stp(spec.file, spec.mesh.get("max_size"), spec.name,
                     ctx.material(spec), comm=ctx.comm,
                     min_mesh_size=spec.mesh.get("min_size"))


@register_builder("wedge")
def _kind_wedge(spec: ComponentSpec, ctx: BuildContext) -> MagnetizedComponent:
    """Programmatic annular wedge (gmsh-OCC); params per gb.build_wedge."""
    if spec.file:
        return _from_stp(spec.file, spec.mesh.get("max_size"), spec.name,
                         ctx.material(spec), comm=ctx.comm,
                         min_mesh_size=spec.mesh.get("min_size"))
    return gb.build_wedge(model_name=spec.name,
                          max_mesh_size=spec.mesh.get("max_size"),
                          min_mesh_size=spec.mesh.get("min_size"),
                          material=ctx.material(spec), comm=ctx.comm,
                          **spec.params)


@register_builder("lid_upper")
def _kind_lid_upper(spec: ComponentSpec, ctx: BuildContext) -> MagnetizedComponent:
    """Two-radius upper lid (gmsh-OCC); params per gb.build_lid_upper."""
    if spec.file:
        return _from_stp(spec.file, spec.mesh.get("max_size"), spec.name,
                         ctx.material(spec), comm=ctx.comm,
                         min_mesh_size=spec.mesh.get("min_size"))
    return gb.build_lid_upper(model_name=spec.name,
                              max_mesh_size=spec.mesh.get("max_size"),
                              min_mesh_size=spec.mesh.get("min_size"),
                              material=ctx.material(spec), comm=ctx.comm,
                              **spec.params)


def _pole_offsets(spec: ComponentSpec, ctx: BuildContext):
    """Resolve the pole's shim offsets from the BuildContext's PoleShape."""
    pole_shape = ctx.pole_shape
    if pole_shape is None:
        raise ValueError(f"Component {spec.name!r} (programmatic pole) needs a "
                         "PoleShape -- pass one to build()/build_geometry()")
    n_segs = pole_shape.num_segments
    cfg = ctx.config
    top = (pole_shape.get_top_offsets_mm() if cfg.top_shim.include
           else np.zeros(n_segs + 1))
    side = (pole_shape.get_side_offsets_deg() if cfg.side_shim.include
            else np.zeros(n_segs + 1))
    return top, side


@register_builder("pole")
def _kind_pole(spec: ComponentSpec, ctx: BuildContext) -> MagnetizedComponent:
    """The (optionally shimmed) pole: STP file or gmsh-OCC with shim offsets."""
    if spec.file:
        return _from_stp(spec.file, spec.mesh.get("max_size"), spec.name,
                         ctx.material(spec), comm=ctx.comm,
                         min_mesh_size=spec.mesh.get("min_size"))

    top, side = _pole_offsets(spec, ctx)
    return gb.build_pole(model_name=spec.name,
                         top_offsets_mm=top, side_offsets_deg=side,
                         max_mesh_size=spec.mesh.get("max_size"),
                         min_mesh_size=spec.mesh.get("min_size"),
                         material=ctx.material(spec), comm=ctx.comm,
                         **spec.params)


@register_builder("swept_polygon")
def _kind_swept_polygon(spec: ComponentSpec, ctx: BuildContext) -> MagnetizedComponent:
    """Solid-of-revolution sector: 2D polygon (N,2) swept about an axis.

    params: polygon (list of [x, y] in the start-plane local frame: x =
    distance from the axis, y = along the axis), axis ([ax, ay, az]),
    start_angle_deg, end_angle_deg, axis_point (optional, default origin).
    """
    if spec.file:
        return _from_stp(spec.file, spec.mesh.get("max_size"), spec.name,
                         ctx.material(spec), comm=ctx.comm,
                         min_mesh_size=spec.mesh.get("min_size"))
    return gb.build_swept_polygon(model_name=spec.name,
                                  max_mesh_size=spec.mesh.get("max_size"),
                                  min_mesh_size=spec.mesh.get("min_size"),
                                  material=ctx.material(spec), comm=ctx.comm,
                                  **spec.params)


@register_builder("racetrack_pair")
def _kind_racetrack_pair(spec: ComponentSpec, ctx: BuildContext) -> CurrentCarryingComponent:
    """+/-z racetrack coil pair.

    Geometry comes from the spec; the CURRENT is read LIVE from
    ``config.coil.current_A`` (the coil-current inner loop mutates it and
    rebuilds this cheap, unmeshed component per current) --
    ``spec.params['current_A']`` is only the initial value.

    The pair's field symmetry is DECLARED (metadata only): equal currents at
    +/-z make the pair symmetric even though the individual coils are not.
    """
    p = spec.params
    z_off = p["midplane_dist"] + 0.5 * p["height_mm"]
    common = dict(
        radius_min_mm=p["radius_min_mm"], radius_max_mm=p["radius_max_mm"],
        height_mm=p["height_mm"], current=ctx.config.coil.current_A,
        num_segments=p["num_segments"],
        color=COIL_COLOR, apply_color=True,
    )
    coil_lower = Coil(center=(0.0, 0.0, z_off), **common)
    coil_upper = Coil(center=(0.0, 0.0, -z_off), **common)
    pair = CurrentCarryingComponent.containerize([coil_lower, coil_upper])
    sym = ctx.config.resolved_symmetry(spec.symmetry)
    if sym:
        pair.declare_symmetries(sym)
    return pair


# ---------------------------------------------------------------------------
# Conforming mesh groups (opt-in via ComponentSpec.mesh_group)
# ---------------------------------------------------------------------------
def _group_occ_entry(spec: ComponentSpec, ctx: BuildContext) -> dict:
    """Turn a grouped ComponentSpec into a build_conforming_group /
    build_structured_group entry."""
    mesh_max = spec.mesh.get("max_size")
    if mesh_max is None:
        raise ValueError(f"Component {spec.name!r} is in mesh_group "
                         f"{spec.mesh_group!r} and needs mesh.max_size")
    entry = {"name": spec.name, "mesh_max": mesh_max,
             "mesh_min": spec.mesh.get("min_size"),
             "structure": spec.structure}
    if spec.file:
        entry["stp_path"] = spec.file
        return entry
    if spec.structure is not None:
        raise ValueError(
            f"Component {spec.name!r}: structure currently requires an STP "
            "file (OCC-callable structured members are a later step)")
    p = dict(spec.params)
    if spec.kind == "wedge":
        entry["occ"] = gb.occ_wedge_callable(**p)
    elif spec.kind == "lid_upper":
        entry["occ"] = gb.occ_lid_upper_callable(**p)
    elif spec.kind == "pole":
        top, side = _pole_offsets(spec, ctx)
        entry["occ"] = gb.occ_pole_callable(top_offsets_mm=top,
                                            side_offsets_deg=side, **p)
    elif spec.kind == "swept_polygon":
        entry["occ"] = gb.occ_swept_polygon_callable(**p)
    else:
        raise ValueError(f"Component kind {spec.kind!r} ({spec.name!r}) is "
                         "not supported in a mesh_group yet")
    return entry


def _build_mesh_group(group_name: str, specs: List[ComponentSpec],
                      ctx: BuildContext, *, rank: int = 0,
                      verbosity: int = 1) -> List[Tuple[ComponentSpec, BaseRadiaComponent]]:
    """Build a conforming mesh group: one gmsh model, fragmented, meshed
    together, split back into per-component MagnetizedComponents.

    If any member carries a `structure:` block the group is built by the
    structured slicer (prism cores + conforming skins/tet members, see
    geometry/structured.py); otherwise by the classic all-tet
    build_conforming_group."""
    from cyclotron_optimizer.geometry.components import build_conforming_group
    from cyclotron_optimizer.geometry.structured import build_structured_group

    if verbosity >= 1 and rank <= 0:
        print(f"Building mesh group '{group_name}' (conforming interfaces): "
              f"{[s.name for s in specs]}...", flush=True)
    entries = [_group_occ_entry(s, ctx) for s in specs]
    out: List[Tuple[ComponentSpec, BaseRadiaComponent]] = []

    if any(s.structure is not None for s in specs):
        group = build_structured_group(entries, group_name=group_name,
                                       comm=ctx.comm)
        for spec in specs:
            mat = ctx.material(spec)
            payload = group["members"][spec.name]
            if spec.structure is not None:
                comp = MagnetizedComponent.from_structured_payload(
                    payload, name=spec.name, material=mat, color=IRON_COLOR,
                    apply_mat=mat is not None, apply_color=True)
            else:
                comp = MagnetizedComponent.from_tet_coords(
                    payload["skin_tets"], material=mat, color=IRON_COLOR,
                    apply_mat=mat is not None, apply_color=True)
            out.append((spec, comp))
        return out

    tets_by_name = build_conforming_group(entries, group_name=group_name,
                                          comm=ctx.comm)
    for spec in specs:
        mat = ctx.material(spec)
        comp = MagnetizedComponent.from_tet_coords(
            tets_by_name[spec.name], material=mat, color=IRON_COLOR,
            apply_mat=mat is not None, apply_color=True)
        out.append((spec, comp))
    return out


def export_iron_stp(config: CyclotronConfig, path, pole_shape=None, *,
                    include_disabled: bool = False, comm=None) -> None:
    """Export ALL (enabled) magnetized components as ONE fragmented STEP file.

    This is the gold-standard geometry contract: the exact conforming solids
    the mesh-group build meshes (same OCC model, same fragment — including
    the shimmed pole at the CURRENT PoleShape and `cylindrical_faces`), for
    import into COMSOL etc. Touching parts arrive pre-imprinted (shared
    interfaces); volume conservation through the fragment is checked.

    Works for grouped and ungrouped configs alike (export always fragments).
    """
    from cyclotron_optimizer.geometry.components import build_conforming_group

    materials = build_materials(config)
    ctx = BuildContext(config, materials, pole_shape=pole_shape, comm=comm)
    entries = []
    for spec in config.components:
        if spec.kind in CURRENT_KINDS:
            continue
        if not spec.enabled and not include_disabled:
            continue
        e = dict(_group_occ_entry_lenient(spec, ctx))
        entries.append(e)
    if not entries:
        raise ValueError("No magnetized components to export")
    build_conforming_group(entries, group_name="stp_export", comm=comm,
                           export_stp_path=path, mesh=False)


def export_component_stp(config: CyclotronConfig, name: str, path,
                         pole_shape=None, *, comm=None) -> None:
    """Export a single component's geometry as a STEP file (e.g. the shimmed
    OCC pole at the current PoleShape, for external gold-standard runs)."""
    from cyclotron_optimizer.geometry.components import build_conforming_group

    spec = config.component(name)
    materials = build_materials(config)
    ctx = BuildContext(config, materials, pole_shape=pole_shape, comm=comm)
    entry = _group_occ_entry_lenient(spec, ctx)
    build_conforming_group([entry], group_name=f"stp_export_{name}",
                           comm=comm, export_stp_path=path, mesh=False)


def _group_occ_entry_lenient(spec: ComponentSpec, ctx: BuildContext) -> dict:
    """_group_occ_entry, but tolerating a missing mesh.max_size and a
    `structure:` block (export-only paths never mesh or discretize -- the
    exported geometry of a structured component is still just its STP)."""
    overrides = {}
    if spec.mesh.get("max_size") is None:
        overrides["mesh"] = {**spec.mesh, "max_size": 1000.0}
    if spec.structure is not None:
        overrides["structure"] = None
    if overrides:
        spec = ComponentSpec(**{**spec.__dict__, **overrides})
    return _group_occ_entry(spec, ctx)


# ---------------------------------------------------------------------------
# Orchestration (the solver's reuse levels)
# ---------------------------------------------------------------------------
def _is_rebuildable(spec: ComponentSpec) -> bool:
    """Shim-dependent components (rebuilt per PoleShape): shimmed and not
    file-based (an STP pole has a frozen shape and counts as static)."""
    return spec.enabled and spec.shimmed and not spec.file


def _rebuildable_group(config: CyclotronConfig) -> Optional[str]:
    """The mesh_group that contains the rebuildable (shimmed) component, if
    any: that ENTIRE group is rebuilt per iterate in build_pole_part."""
    for spec in config.components:
        if _is_rebuildable(spec) and spec.mesh_group:
            return spec.mesh_group
    return None


def build_static_iron_parts(
    config: CyclotronConfig,
    *,
    rank: int = 0,
    comm=None,
    verbosity: int = 1,
) -> dict:
    """Build all enabled magnetized components EXCEPT the rebuildable pole.

    :return: dict with 'parts' (list of (spec, component), in config order)
        and 'materials' ({name: RadiaMaterial}, shared with the pole build).
    """
    say = verbosity >= 1 and rank <= 0
    materials = build_materials(config)
    ctx = BuildContext(config, materials, comm=comm)
    deferred_group = _rebuildable_group(config)

    parts: List[Tuple[ComponentSpec, BaseRadiaComponent]] = []
    built_groups: set = set()
    for spec in config.components:
        if not spec.enabled or spec.kind in CURRENT_KINDS or _is_rebuildable(spec):
            continue
        if spec.mesh_group:
            if spec.mesh_group == deferred_group:
                # rebuilt together with the shimmed pole per iterate
                continue
            if spec.mesh_group in built_groups:
                continue
            group_specs = [s for s in config.components
                           if s.enabled and s.mesh_group == spec.mesh_group
                           and s.kind not in CURRENT_KINDS]
            parts.extend(_build_mesh_group(spec.mesh_group, group_specs, ctx,
                                           rank=rank, verbosity=verbosity))
            built_groups.add(spec.mesh_group)
            continue
        if say:
            print(f"Building static component '{spec.name}' (kind {spec.kind})...",
                  flush=True)
        parts.append((spec, build_component(spec, ctx)))

    return {"parts": parts, "materials": materials}


def build_pole_part(
    config: CyclotronConfig,
    pole_shape=None,
    *,
    comm=None,
    materials: Optional[Dict[str, RadiaMaterial]] = None,
) -> Optional[Tuple[ComponentSpec, BaseRadiaComponent]]:
    """Build the shim-dependent pole (None when the config has none).

    Returns a single ``(spec, component)`` tuple, EXCEPT when the pole is in
    a mesh_group: then the whole group is rebuilt per iterate (conforming
    interfaces need joint meshing) and a LIST of ``(spec, component)`` for
    all group members is returned. assemble_iron accepts both forms.
    """
    rebuildable = [s for s in config.components if _is_rebuildable(s)]
    if not rebuildable:
        return None
    if len(rebuildable) > 1:
        raise ValueError("Multiple shimmed components are not supported yet: "
                         f"{[s.name for s in rebuildable]}")
    spec = rebuildable[0]
    if materials is None:
        materials = build_materials(config)
    ctx = BuildContext(config, materials, pole_shape=pole_shape, comm=comm)
    if spec.mesh_group:
        group_specs = [s for s in config.components
                       if s.enabled and s.mesh_group == spec.mesh_group
                       and s.kind not in CURRENT_KINDS]
        return _build_mesh_group(spec.mesh_group, group_specs, ctx)
    return spec, build_component(spec, ctx)


def assemble_iron(
    config: CyclotronConfig,
    static_parts: dict,
    pole_entry: Optional[Tuple[ComponentSpec, BaseRadiaComponent]] = None,
    *,
    omit_symmetry: bool = False,
    rank: int = 0,
    verbosity: int = 1,
    split_perturbative: bool = False,
):
    """Group the iron by named symmetry, containerize, apply the transforms.

    Every returned container is a per-assembly THROWAWAY: shallow-dispose it
    before re-assembling (the member components survive with parent pointers
    reset). The symmetry is applied to the fresh container each time.

    :param split_perturbative: when True, return ``(main_subs, perturb_subs)``
        with the components flagged ``perturbative: True`` in their own group
        containers (the solver relaxes those in a separate stage); default
        returns the single combined list (main + perturbative).
    """
    say = verbosity >= 1 and rank <= 0

    entries = list(static_parts["parts"])
    if pole_entry is not None:
        if isinstance(pole_entry, list):  # mesh_group: pole + its group
            entries.extend(pole_entry)
        else:
            entries.append(pole_entry)

    groups: Dict[Tuple[Optional[str], bool], List[BaseRadiaComponent]] = {}
    for spec, comp in entries:
        groups.setdefault((spec.symmetry, spec.perturbative), []).append(comp)

    main_subs: List[BaseRadiaComponent] = []
    perturb_subs: List[BaseRadiaComponent] = []
    for (sym_name, perturbative), comps in groups.items():
        container = MagnetizedComponent.containerize(comps)
        if sym_name and not omit_symmetry:
            if say:
                print(f"Applying symmetry '{sym_name}' to "
                      f"{len(comps)} iron component(s)...", flush=True)
            container.apply_symmetry(config.resolved_symmetry(sym_name))
        elif sym_name and omit_symmetry and say:
            print("Symmetry DISABLED (geometry debug mode)", flush=True)
        (perturb_subs if perturbative else main_subs).append(container)

    if split_perturbative:
        return main_subs, perturb_subs
    return main_subs + perturb_subs


def build_iron(
    config: CyclotronConfig,
    pole_shape=None,
    *,
    omit_symmetry: bool = False,
    rank: int = 0,
    comm=None,
    verbosity: int = 1,
) -> List[BaseRadiaComponent]:
    """Build the complete iron assembly (static parts + pole) in one shot.

    Convenience wrapper over build_static_iron_parts / build_pole_part /
    assemble_iron; the solver uses those pieces directly so it can keep the
    static iron across pole rebuilds.
    """
    static_parts = build_static_iron_parts(config, rank=rank, comm=comm,
                                           verbosity=verbosity)
    pole_entry = build_pole_part(config, pole_shape, comm=comm,
                                 materials=static_parts["materials"])
    return assemble_iron(config, static_parts, pole_entry,
                         omit_symmetry=omit_symmetry, rank=rank,
                         verbosity=verbosity)


def build_coils(config: CyclotronConfig) -> BaseRadiaComponent:
    """Build the enabled current-source components (per coil current).

    The coil current is read from ``config.coil.current_A`` at build time, so
    the coil-current inner loop rebuilds this (cheap, unmeshed) sub-container
    per current.
    """
    ctx = BuildContext(config, {}, comm=None)  # current sources need no material
    coils = [build_component(spec, ctx) for spec in config.components
             if spec.enabled and spec.kind in CURRENT_KINDS]
    if not coils:
        raise ValueError("Config defines no enabled current-source components")
    if len(coils) == 1:
        return coils[0]
    return BaseRadiaComponent.containerize(coils)


def build_geometry(
    config: CyclotronConfig,
    pole_shape=None,
    *,
    omit_symmetry: bool = False,
    rank: int = 0,
    comm=None,
    verbosity: int = 1,
) -> BaseRadiaComponent:
    """Build the complete cyclotron model (iron + coils).

    :param config: CyclotronConfig object (either schema; the component list
                   drives the build).
    :param pole_shape: PoleShape providing the pole's shim offsets (required
                       for a programmatically built pole; unused for STP).
    :param omit_symmetry: If True, skip the symmetry transforms (e.g. for
                          visualization of the full model).
    :param rank: MPI rank (only rank 0 prints).
    :param comm: MPI communicator; the mesh builders mesh on rank 0 and
                 broadcast (identical radia ids on every rank).
    :param verbosity: 0 silent, 1 normal, 2 debug.
    :return: a BaseRadiaComponent containerizing the iron and the coils
             (use ``.id`` for the underlying radia object id).
    """
    say = verbosity >= 1 and rank <= 0

    if say:
        print("\n" + "=" * 60, flush=True)
        print("BUILDING CYCLOTRON GEOMETRY", flush=True)
        print("=" * 60 + "\n", flush=True)

    iron_subs = build_iron(
        config, pole_shape,
        omit_symmetry=omit_symmetry, rank=rank, comm=comm, verbosity=verbosity,
    )

    if say:
        print("Building current sources (coils)...", flush=True)
    coils = build_coils(config)

    if say:
        print("Assembling cyclotron...", flush=True)
    cyclotron = BaseRadiaComponent.containerize([*iron_subs, coils])

    if say:
        print("\n" + "=" * 60, flush=True)
        print("GEOMETRY BUILDING COMPLETE", flush=True)
        print("=" * 60 + "\n", flush=True)

    return cyclotron
