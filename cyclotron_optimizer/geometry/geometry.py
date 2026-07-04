"""Cyclotron geometry assembly, driven by the config's component list.

Every machine part is a ComponentSpec (name / kind / params / material /
symmetry / mesh / file), and each ``kind`` maps to a builder in the registry
below ('stp', 'wedge', 'lid_upper', 'pole', 'wedge_pair', 'racetrack_pair').
Adding a part is a YAML entry plus -- at most -- one registered builder.

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
    """Tet-meshed magnetized part from an STP file."""
    if not spec.file:
        raise ValueError(f"Component {spec.name!r} (kind 'stp') needs a file")
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


@register_builder("pole")
def _kind_pole(spec: ComponentSpec, ctx: BuildContext) -> MagnetizedComponent:
    """The (optionally shimmed) pole: STP file or gmsh-OCC with shim offsets."""
    if spec.file:
        return _from_stp(spec.file, spec.mesh.get("max_size"), spec.name,
                         ctx.material(spec), comm=ctx.comm,
                         min_mesh_size=spec.mesh.get("min_size"))

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

    return gb.build_pole(model_name=spec.name,
                         top_offsets_mm=top, side_offsets_deg=side,
                         max_mesh_size=spec.mesh.get("max_size"),
                         min_mesh_size=spec.mesh.get("min_size"),
                         material=ctx.material(spec), comm=ctx.comm,
                         **spec.params)


@register_builder("wedge_pair")
def _kind_wedge_pair(spec: ComponentSpec, ctx: BuildContext) -> MagnetizedComponent:
    """Mirrored wedge pair straddling a horizontal channel (extraction channel)."""
    mat = ctx.material(spec)
    mesh = spec.mesh.get("max_size")
    if spec.file:
        parts = [
            _from_stp(spec.file, mesh, f"{spec.name}_1", mat, comm=ctx.comm),
            _from_stp(spec.file, mesh, f"{spec.name}_2", mat, comm=ctx.comm),
        ]
    else:
        params = dict(spec.params)
        channel_width = params.pop("channel_width_mm")
        height = params["height_mm"]
        common = dict(max_mesh_size=mesh, min_mesh_size=spec.mesh.get("min_size"),
                      material=mat, comm=ctx.comm, **params)
        parts = [
            gb.build_wedge(z_offset_mm=height + channel_width / 2.0,
                           model_name=f"{spec.name}_1", **common),
            gb.build_wedge(z_offset_mm=-channel_width / 2.0,
                           model_name=f"{spec.name}_2", **common),
        ]
    return MagnetizedComponent.containerize(parts)


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
# Orchestration (the solver's reuse levels)
# ---------------------------------------------------------------------------
def _is_rebuildable(spec: ComponentSpec) -> bool:
    """Shim-dependent components (rebuilt per PoleShape): shimmed and not
    file-based (an STP pole has a frozen shape and counts as static)."""
    return spec.enabled and spec.shimmed and not spec.file


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

    parts: List[Tuple[ComponentSpec, BaseRadiaComponent]] = []
    for spec in config.components:
        if not spec.enabled or spec.kind in CURRENT_KINDS or _is_rebuildable(spec):
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
    """Build the shim-dependent pole (None when the config has none)."""
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
