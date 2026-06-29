"""Cyclotron geometry assembly (cache-free).

Builds the iron pieces (yoke wall, lower lid, upper lid, optional extraction
channel, shimmed pole) plus the racetrack coil pair, applies material / 8-fold
symmetry / drawing attributes, and returns a single BaseRadiaComponent.

Each iron piece is built from its STP file when ``stp_filename`` is set, else
programmatically via the gmsh-OCC (tet) builders in ``gmsh_builders.py``. The
previous radia-native segmented builders were intentionally dropped.

The earlier monolithic implementation is preserved in ``geometry_old.py``.
"""

from __future__ import annotations

from typing import List

import numpy as np
import radia as rad

from config_io.config import CyclotronConfig
from geometry.components import (
    BaseRadiaComponent,
    Coil,
    CurrentCarryingComponent,
    MagnetizedComponent,
    RadiaMaterial,
)
from geometry import gmsh_builders as gb

# 8-fold cyclotron symmetry of the iron, as (kind, point, normal). Identical to
# simulation.field_calculator.model_symmetries so the same set can be reused for
# GPU field evaluation.
CYCLOTRON_SYMMETRIES = [
    ("perp", [0, 0, 0], [1, -1, 0]),  # mirror across x=y diagonal
    ("perp", [0, 0, 0], [1, 0, 0]),   # mirror across x=0 plane
    ("perp", [0, 0, 0], [0, 1, 0]),   # mirror across y=0 plane
    ("para", [0, 0, 0], [0, 0, 1]),   # mirror across z=0 plane
]

IRON_COLOR = [0.0, 0.5, 1.0]
COIL_COLOR = [1.0, 0.0, 0.0]


def _build_iron_material(config: CyclotronConfig) -> RadiaMaterial:
    """Create the iron material from a BH file (preferred) or a formula fallback."""
    mat_cfg = config.material
    if mat_cfg.bh_filename is not None:
        return RadiaMaterial.from_bh_file(mat_cfg.bh_filename, type="BH", name="iron")

    mat_id = rad.MatSatIsoFrm(
        mat_cfg.saturation_field_t,
        mat_cfg.saturation_curve_m,
        mat_cfg.linear_curve_m,
    )
    return RadiaMaterial.from_radia_material(mat_id, name="iron")


def _from_stp(stp_filename, max_mesh_size, model_name, material, *, comm=None) -> MagnetizedComponent:
    return MagnetizedComponent.from_stp(
        stp_filename,
        mesh_size_max=max_mesh_size,
        model_name=model_name,
        comm=comm,
        material=material,
        color=IRON_COLOR,
        apply_mat=True,
        apply_color=True,
    )


def _build_yoke_wall(config, material, *, comm) -> MagnetizedComponent:
    cfg = config.yoke
    if cfg.stp_filename:
        return _from_stp(cfg.stp_filename, cfg.max_mesh_size, "yoke", material, comm=comm)
    return gb.build_wedge(
        inner_radius_mm=cfg.inner_radius_mm, outer_radius_mm=cfg.outer_radius_mm,
        height_mm=cfg.height_mm, z_offset_mm=0.0,
        end_ang_deg=config.geometry.yoke_build_angle_deg,
        include_window=True, window_width_mm=cfg.window_width_mm,
        max_mesh_size=cfg.max_mesh_size, model_name="yoke",
        material=material, comm=comm,
    )


def _build_lid_lower(config, material, *, comm) -> MagnetizedComponent:
    cfg = config.lid_lower
    if cfg.stp_filename:
        return _from_stp(cfg.stp_filename, cfg.max_mesh_size, "lid_lower", material, comm=comm)
    return gb.build_wedge(
        inner_radius_mm=cfg.inner_radius_mm, outer_radius_mm=cfg.outer_radius_mm,
        height_mm=cfg.height_mm, z_offset_mm=-config.yoke.height_mm,
        end_ang_deg=config.geometry.yoke_build_angle_deg,
        max_mesh_size=cfg.max_mesh_size, model_name="lid_lower",
        material=material, comm=comm,
    )


def _build_lid_upper(config, material, *, comm) -> MagnetizedComponent:
    cfg = config.lid_upper
    if cfg.stp_filename:
        return _from_stp(cfg.stp_filename, cfg.max_mesh_size, "lid_upper", material, comm=comm)
    z_off = -(config.yoke.height_mm + config.lid_lower.height_mm)
    return gb.build_lid_upper(
        inner_radius_mm=cfg.inner_radius_mm,
        outer_radius_mm_1=cfg.outer_radius_mm_1, outer_radius_mm_2=cfg.outer_radius_mm_2,
        height_mm=cfg.height_mm, z_offset_mm=z_off,
        end_ang_deg=config.geometry.yoke_build_angle_deg,
        seg_theta=cfg.segmentation[1], max_mesh_size=cfg.max_mesh_size,
        material=material, comm=comm,
    )


def _build_pole(config, pole_shape, material, *, comm) -> MagnetizedComponent:
    cfg = config.pole
    if cfg.stp_filename:
        return _from_stp(cfg.stp_filename, cfg.max_mesh_size, "pole", material, comm=comm)

    n_segs = pole_shape.num_segments
    top = pole_shape.get_top_offsets_mm() if config.top_shim.include else np.zeros(n_segs + 1)
    side = pole_shape.get_side_offsets_deg() if config.side_shim.include else np.zeros(n_segs + 1)
    pole_zs = -(config.yoke.height_mm + config.lid_lower.height_mm)

    return gb.build_pole(
        inner_radius_mm=cfg.inner_radius_mm, outer_radius_mm=cfg.outer_radius_mm,
        height_mm=cfg.height_mm, half_angle_deg=cfg.full_angle_deg / 2.0, pole_zs=pole_zs,
        top_offsets_mm=top, side_offsets_deg=side,
        max_mesh_size=cfg.max_mesh_size, material=material, comm=comm,
    )


def _build_extract_channel(config, material, *, comm) -> List[MagnetizedComponent]:
    cfg = config.extract_channel
    if cfg.stp_filename:
        return [
            _from_stp(cfg.stp_filename, cfg.max_mesh_size, "extract_1", material, comm=comm),
            _from_stp(cfg.stp_filename, cfg.max_mesh_size, "extract_2", material, comm=comm),
        ]
    common = dict(
        inner_radius_mm=cfg.inner_radius_mm, outer_radius_mm=cfg.outer_radius_mm,
        height_mm=cfg.height_mm, end_ang_deg=cfg.end_ang_deg, start_ang_deg=cfg.start_ang_deg,
        window_width_mm=cfg.window_width_mm, max_mesh_size=cfg.max_mesh_size,
        material=material, comm=comm,
    )
    return [
        gb.build_wedge(z_offset_mm=cfg.height_mm + cfg.channel_width_mm / 2.0,
                       model_name="extract_1", **common),
        gb.build_wedge(z_offset_mm=-cfg.channel_width_mm / 2.0,
                       model_name="extract_2", **common),
    ]


def build_coils(config: CyclotronConfig) -> CurrentCarryingComponent:
    """Build the upper/lower racetrack coil pair and return them as a container.

    The coil current is read from ``config.coil.current_A`` at build time, so the
    coil-current inner loop rebuilds this (cheap, unmeshed) sub-container per current.
    """
    cfg = config.coil
    z_off = cfg.midplane_dist + 0.5 * cfg.height_mm
    common = dict(
        radius_min_mm=cfg.radius_min_mm, radius_max_mm=cfg.radius_max_mm,
        height_mm=cfg.height_mm, current=cfg.current_A, num_segments=cfg.num_segments,
        color=COIL_COLOR, apply_color=True,
    )
    coil_lower = Coil(center=(0.0, 0.0, z_off), **common)
    coil_upper = Coil(center=(0.0, 0.0, -z_off), **common)
    return CurrentCarryingComponent.containerize([coil_lower, coil_upper])


def build_iron(
    config: CyclotronConfig,
    pole_shape=None,
    *,
    omit_symmetry: bool = False,
    rank: int = 0,
    comm=None,
    verbosity: int = 1,
) -> List[MagnetizedComponent]:
    """Build the iron assembly as one or two sub-containers.

    Returns ``[symmetric_iron]`` (yoke + lids + pole, with the 8-fold cyclotron
    symmetry applied) and, when an extraction channel is configured, additionally a
    non-symmetric iron sub-container (the channel pieces, which break the 8-fold
    symmetry and are therefore NOT symmetrized). Keeping these as standalone
    sub-containers lets the coil-current inner loop swap only the coils while the iron
    -- and its relaxed magnetization -- is reused (see the optimizer rework).

    NOTE: the previous build symmetrized ALL iron including the extraction channel,
    which 8-fold-replicated the channel; splitting it out fixes that.
    """
    say = verbosity >= 1 and rank <= 0
    material = _build_iron_material(config)

    if say:
        print("Building symmetric iron (yoke, lids, pole)...", flush=True)
    symmetric_parts: List[MagnetizedComponent] = [
        _build_yoke_wall(config, material, comm=comm),
        _build_lid_lower(config, material, comm=comm),
        _build_lid_upper(config, material, comm=comm),
        _build_pole(config, pole_shape, material, comm=comm),
    ]
    symmetric_iron = MagnetizedComponent.containerize(symmetric_parts)
    if omit_symmetry:
        if say:
            print("Symmetry DISABLED (geometry debug mode)", flush=True)
    else:
        if say:
            print("Applying 8-fold symmetry to symmetric iron...", flush=True)
        symmetric_iron.apply_symmetry(CYCLOTRON_SYMMETRIES)

    iron_subs: List[MagnetizedComponent] = [symmetric_iron]

    if config.extract_channel.use_extract_chan:
        if say:
            print("Building non-symmetric iron (extraction channel)...", flush=True)
        ext_parts = _build_extract_channel(config, material, comm=comm)
        non_symmetric_iron = (
            MagnetizedComponent.containerize(ext_parts)
            if len(ext_parts) > 1 else ext_parts[0]
        )
        iron_subs.append(non_symmetric_iron)

    return iron_subs


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

    :param config: CyclotronConfig object.
    :param pole_shape: PoleShape providing the pole's shim offsets (required for a
                       programmatically built pole; unused for an STP pole).
    :param omit_symmetry: If True, skip the 8-fold symmetry (e.g. for visualization).
    :param rank: MPI rank (only rank 0 prints).
    :param comm: MPI communicator; passed to the gmsh-OCC builders so they mesh on
                 rank 0 and broadcast (identical radia ids on every rank).
    :param verbosity: 0 silent, 1 normal, 2 debug.
    :return: a BaseRadiaComponent containerizing the iron and the coils
             (use ``.id`` for the underlying radia object id).
    """
    say = verbosity >= 1 and rank <= 0

    if say:
        print("\n" + "=" * 60, flush=True)
        print("BUILDING CYCLOTRON GEOMETRY", flush=True)
        print("=" * 60 + "\n", flush=True)

    # ---------- Iron (symmetric + optional non-symmetric sub-containers) ----------
    iron_subs = build_iron(
        config, pole_shape,
        omit_symmetry=omit_symmetry, rank=rank, comm=comm, verbosity=verbosity,
    )

    # ---------- Coils ----------
    if say:
        print("Building racetrack coils...", flush=True)
    coils = build_coils(config)

    # ---------- Assemble: [symmetric_iron, (non_symmetric_iron), coils] ----------
    if say:
        print("Assembling cyclotron...", flush=True)
    cyclotron = BaseRadiaComponent.containerize([*iron_subs, coils])

    if say:
        print("\n" + "=" * 60, flush=True)
        print("GEOMETRY BUILDING COMPLETE", flush=True)
        print("=" * 60 + "\n", flush=True)

    return cyclotron
