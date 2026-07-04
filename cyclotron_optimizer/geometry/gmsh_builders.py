"""Programmatic gmsh-OCC iron builders (meshed to tets -> radia polyhedra).

Only the OCC/tet path is provided; the radia-native segmented builders were
intentionally dropped (they can be reintroduced later if ever needed).

Each public ``build_*`` returns a ``MagnetizedComponent`` whose radia object is a
container of tetrahedral polyhedra. Symmetry is normally applied once to the
assembled iron container, so these default to ``apply_sym=False``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import gmsh

from cyclotron_optimizer.geometry.components import MagnetizedComponent, RadiaMaterial

IRON_COLOR = [0.0, 0.5, 1.0]


# -----------------------------
# OCC geometry helpers (operate on the current gmsh model)
# -----------------------------
def _fuse_volumes(volumes: List[int]) -> None:
    """Boolean-fuse a list of OCC volume tags into one (in place)."""
    if len(volumes) <= 1:
        return
    base = (3, volumes[0])
    tools = [(3, v) for v in volumes[1:]]
    gmsh.model.occ.fuse([base], tools, removeTool=True)
    gmsh.model.occ.synchronize()


def _occ_wedge(r_inner, r_outer, ang_start, ang_end, z_bottom, height,
               include_window=False, window_width=300.0) -> int:
    """Simple annular wedge via cylinder difference, with an optional window cut."""
    tot_ang = ang_end - ang_start
    inner_cyl = gmsh.model.occ.addCylinder(0.0, 0.0, z_bottom, 0.0, 0.0, height, r_inner, angle=tot_ang)
    volume = gmsh.model.occ.addCylinder(0.0, 0.0, z_bottom, 0.0, 0.0, height, r_outer, angle=tot_ang)
    new_tags, _ = gmsh.model.occ.cut([(3, volume)], [(3, inner_cyl)])
    volume = new_tags[0][1]
    gmsh.model.occ.rotate([(3, volume)], 0, 0, 0, 0, 0, 1, ang_start)

    if include_window:
        box_length = r_outer * 4
        box_height = height + 10
        window_box = gmsh.model.occ.addBox(-box_length / 2, 0, z_bottom - 5,
                                           box_length, window_width, box_height)
        gmsh.model.occ.rotate([(3, window_box)], 0, 0, 0, 0, 0, 1, np.pi / 4)
        y_offset = -(np.sqrt(2) / 2) * window_width
        gmsh.model.occ.translate([(3, window_box)], 0, y_offset, 0)
        new_tags2, _ = gmsh.model.occ.cut([(3, volume)], [(3, window_box)])
        volume = new_tags2[0][1]

    return volume


def _occ_pole_segment(r_inner, r_outer, ang_inner, ang_outer, pole_zs, base_height,
                      top_inner, top_outer) -> int:
    """One shimmed pole radial segment (explicit 8-corner wedge with tapered top)."""
    h_inner = pole_zs + base_height + top_inner
    h_outer = pole_zs + base_height + top_outer

    # Bottom face (z = pole_zs)
    p1 = gmsh.model.occ.addPoint(r_inner, 0.0, pole_zs)
    p2 = gmsh.model.occ.addPoint(r_outer, 0.0, pole_zs)
    p3 = gmsh.model.occ.addPoint(r_outer * np.cos(ang_outer), r_outer * np.sin(ang_outer), pole_zs)
    p4 = gmsh.model.occ.addPoint(r_inner * np.cos(ang_inner), r_inner * np.sin(ang_inner), pole_zs)

    # Top face (tapered: h_inner / h_outer)
    p5 = gmsh.model.occ.addPoint(r_inner, 0.0, h_inner)
    p6 = gmsh.model.occ.addPoint(r_outer, 0.0, h_outer)
    p7 = gmsh.model.occ.addPoint(r_outer * np.cos(ang_outer), r_outer * np.sin(ang_outer), h_outer)
    p8 = gmsh.model.occ.addPoint(r_inner * np.cos(ang_inner), r_inner * np.sin(ang_inner), h_inner)

    center_bottom = gmsh.model.occ.addPoint(0.0, 0.0, pole_zs)
    center_top_inner = gmsh.model.occ.addPoint(0.0, 0.0, h_inner)
    center_top_outer = gmsh.model.occ.addPoint(0.0, 0.0, h_outer)

    e1 = gmsh.model.occ.addLine(p1, p2)
    e2 = gmsh.model.occ.addCircleArc(p2, center_bottom, p3)
    e3 = gmsh.model.occ.addLine(p3, p4)
    e4 = gmsh.model.occ.addCircleArc(p4, center_bottom, p1)
    e5 = gmsh.model.occ.addLine(p5, p6)
    e6 = gmsh.model.occ.addCircleArc(p6, center_top_outer, p7)
    e7 = gmsh.model.occ.addLine(p7, p8)
    e8 = gmsh.model.occ.addCircleArc(p8, center_top_inner, p5)
    e9 = gmsh.model.occ.addLine(p1, p5)
    e10 = gmsh.model.occ.addLine(p2, p6)
    e11 = gmsh.model.occ.addLine(p3, p7)
    e12 = gmsh.model.occ.addLine(p4, p8)

    loop_bottom = gmsh.model.occ.addCurveLoop([e1, e2, e3, e4])
    face_bottom = gmsh.model.occ.addPlaneSurface([loop_bottom])

    loop_top = gmsh.model.occ.addCurveLoop([e5, e6, e7, e8])
    face_top = gmsh.model.occ.addSurfaceFilling(loop_top)

    loop_side1 = gmsh.model.occ.addCurveLoop([e1, e10, -e5, -e9])
    if abs(h_inner - h_outer) < 0.001:
        face_side1 = gmsh.model.occ.addPlaneSurface([loop_side1])
    else:
        face_side1 = gmsh.model.occ.addSurfaceFilling(loop_side1)

    loop_side2 = gmsh.model.occ.addCurveLoop([e2, e11, -e6, -e10])
    face_side2 = gmsh.model.occ.addSurfaceFilling(loop_side2)

    loop_side3 = gmsh.model.occ.addCurveLoop([e3, e12, -e7, -e11])
    if abs(h_inner - h_outer) < 0.001:
        face_side3 = gmsh.model.occ.addPlaneSurface([loop_side3])
    else:
        face_side3 = gmsh.model.occ.addSurfaceFilling(loop_side3)

    loop_side4 = gmsh.model.occ.addCurveLoop([e4, e9, -e8, -e12])
    face_side4 = gmsh.model.occ.addSurfaceFilling(loop_side4)

    surface_loop = gmsh.model.occ.addSurfaceLoop(
        [face_bottom, face_top, face_side1, face_side2, face_side3, face_side4]
    )
    return gmsh.model.occ.addVolume([surface_loop])


def _occ_lid_upper(r_inner, r_outer_1, r_outer_2, ang_start, ang_end, seg_theta,
                   z_bottom, base_height, include_window=False, window_width=300.0) -> int:
    """Tapered upper-lid wedge (two outer radii) with arc subdivision + optional window."""
    h_inner = z_bottom + base_height
    h_outer = z_bottom + base_height

    if include_window:
        cut_pln = 0.5 * window_width * np.sqrt(2.0)
        ang_end_outer = np.deg2rad(45) - np.arcsin(cut_pln * np.sqrt(2.0) * 0.5 / r_outer_1)
        ang_end_inner = np.deg2rad(45) - np.arcsin(cut_pln * np.sqrt(2.0) * 0.5 / r_inner)
    else:
        ang_end_inner = ang_end
        ang_end_outer = ang_end

    segment_angs_outer = np.linspace(ang_start, ang_end_outer, seg_theta)
    intermediate_angs_outer = segment_angs_outer[1:-1]

    p1 = gmsh.model.occ.addPoint(r_inner * np.cos(ang_start), r_inner * np.sin(ang_start), z_bottom)
    p2 = gmsh.model.occ.addPoint(r_outer_2 * np.cos(ang_start), r_outer_2 * np.sin(ang_start), z_bottom)
    p3 = gmsh.model.occ.addPoint(r_outer_2 * np.cos(ang_end_outer), r_outer_2 * np.sin(ang_end_outer), z_bottom)
    p4 = gmsh.model.occ.addPoint(r_inner * np.cos(ang_end_inner), r_inner * np.sin(ang_end_inner), z_bottom)

    p5 = gmsh.model.occ.addPoint(r_inner * np.cos(ang_start), r_inner * np.sin(ang_start), h_inner)
    p6 = gmsh.model.occ.addPoint(r_outer_1 * np.cos(ang_start), r_outer_1 * np.sin(ang_start), h_outer)
    p7 = gmsh.model.occ.addPoint(r_outer_1 * np.cos(ang_end_outer), r_outer_1 * np.sin(ang_end_outer), h_outer)
    p8 = gmsh.model.occ.addPoint(r_inner * np.cos(ang_end_inner), r_inner * np.sin(ang_end_inner), h_inner)

    center_bottom = gmsh.model.occ.addPoint(0.0, 0.0, z_bottom)
    center_top_inner = gmsh.model.occ.addPoint(0.0, 0.0, h_inner)
    center_top_outer = gmsh.model.occ.addPoint(0.0, 0.0, h_outer)

    e1 = gmsh.model.occ.addLine(p1, p2)
    e2 = gmsh.model.occ.addCircleArc(p2, center_bottom, p3)
    e3 = gmsh.model.occ.addLine(p3, p4)
    e4 = gmsh.model.occ.addCircleArc(p4, center_bottom, p1)
    e5 = gmsh.model.occ.addLine(p5, p6)
    e6 = gmsh.model.occ.addCircleArc(p6, center_top_outer, p7)
    e7 = gmsh.model.occ.addLine(p7, p8)
    e8 = gmsh.model.occ.addCircleArc(p8, center_top_inner, p5)
    e9 = gmsh.model.occ.addLine(p1, p5)
    e10 = gmsh.model.occ.addLine(p2, p6)
    e11 = gmsh.model.occ.addLine(p3, p7)
    e12 = gmsh.model.occ.addLine(p4, p8)

    loop_bottom = gmsh.model.occ.addCurveLoop([e1, e2, e3, e4])
    face_bottom = gmsh.model.occ.addPlaneSurface([loop_bottom])

    loop_top = gmsh.model.occ.addCurveLoop([e5, e6, e7, e8])
    face_top = gmsh.model.occ.addSurfaceFilling(loop_top)

    loop_side1 = gmsh.model.occ.addCurveLoop([e1, e10, -e5, -e9])
    if abs(h_inner - h_outer) < 0.001:
        face_side1 = gmsh.model.occ.addPlaneSurface([loop_side1])
    else:
        face_side1 = gmsh.model.occ.addSurfaceFilling(loop_side1)

    # Face 4: outer curved side, optionally subdivided into seg_theta arcs
    if intermediate_angs_outer.size > 0:
        p14_arr = [p2]
        p16_arr = [p6]
        face_side2 = []
        i = 0
        for i in range(intermediate_angs_outer.size):
            ang_mid_outer = intermediate_angs_outer[i]
            p14_arr.append(gmsh.model.occ.addPoint(r_outer_2 * np.cos(ang_mid_outer),
                                                   r_outer_2 * np.sin(ang_mid_outer), z_bottom))
            p16_arr.append(gmsh.model.occ.addPoint(r_outer_1 * np.cos(ang_mid_outer),
                                                   r_outer_1 * np.sin(ang_mid_outer), h_outer))
            e15 = gmsh.model.occ.addCircleArc(p14_arr[i], center_bottom, p14_arr[i + 1])
            e17 = gmsh.model.occ.addCircleArc(p16_arr[i], center_top_outer, p16_arr[i + 1])
            e22 = gmsh.model.occ.addLine(p14_arr[i], p16_arr[i])
            e23 = gmsh.model.occ.addLine(p14_arr[i + 1], p16_arr[i + 1])
            loop_side2_a = gmsh.model.occ.addCurveLoop([e15, e22, -e17, -e23])
            face_side2.append(gmsh.model.occ.addSurfaceFilling(loop_side2_a))
        p14_arr.append(p3)
        p16_arr.append(p7)
        e15 = gmsh.model.occ.addCircleArc(p14_arr[i + 1], center_bottom, p14_arr[i + 2])
        e17 = gmsh.model.occ.addCircleArc(p16_arr[i + 1], center_top_outer, p16_arr[i + 2])
        e22 = gmsh.model.occ.addLine(p14_arr[i + 1], p16_arr[i + 1])
        e23 = gmsh.model.occ.addLine(p14_arr[i + 2], p16_arr[i + 2])
        loop_side2_a = gmsh.model.occ.addCurveLoop([e15, e22, -e17, -e23])
        face_side2.append(gmsh.model.occ.addSurfaceFilling(loop_side2_a))
    else:
        loop_side2 = gmsh.model.occ.addCurveLoop([e2, e11, -e6, -e10])
        face_side2 = [gmsh.model.occ.addSurfaceFilling(loop_side2)]

    loop_side3 = gmsh.model.occ.addCurveLoop([e3, e12, -e7, -e11])
    if abs(h_inner - h_outer) < 0.001:
        face_side3 = gmsh.model.occ.addPlaneSurface([loop_side3])
    else:
        face_side3 = gmsh.model.occ.addSurfaceFilling(loop_side3)

    loop_side4 = gmsh.model.occ.addCurveLoop([e4, e9, -e8, -e12])
    face_side4 = gmsh.model.occ.addSurfaceFilling(loop_side4)

    surface_loop = gmsh.model.occ.addSurfaceLoop(
        [face_bottom, face_top, face_side1, *face_side2, face_side3, face_side4], sewing=True
    )
    return gmsh.model.occ.addVolume([surface_loop])


# -----------------------------
# Public builders -> MagnetizedComponent
# -----------------------------
def build_wedge(
    *,
    inner_radius_mm: float,
    outer_radius_mm: float,
    height_mm: float,
    z_offset_mm: float,
    end_ang_deg: float,
    max_mesh_size: float,
    model_name: str,
    start_ang_deg: float = 0.0,
    include_window: bool = False,
    window_width_mm: float = 300.0,
    min_mesh_size: Optional[float] = None,
    material: Optional[RadiaMaterial] = None,
    symmetries=None,
    color: Sequence[float] = IRON_COLOR,
    comm=None,
    apply_mat: bool = True,
    apply_color: bool = True,
    apply_sym: bool = False,
) -> MagnetizedComponent:
    """Simple annular-wedge iron piece (yoke wall, lower lid, extraction channel)."""
    z_bottom = z_offset_mm - height_mm
    a0 = np.deg2rad(start_ang_deg)
    a1 = np.deg2rad(end_ang_deg)

    def occ():
        _occ_wedge(inner_radius_mm, outer_radius_mm, a0, a1, z_bottom, height_mm,
                   include_window, window_width_mm)

    return MagnetizedComponent.from_gmsh_occ(
        occ, model_name=model_name,
        mesh_size_min=1.0 if min_mesh_size is None else min_mesh_size,
        mesh_size_max=max_mesh_size,
        comm=comm, material=material, symmetries=symmetries, color=color,
        apply_mat=apply_mat, apply_color=apply_color, apply_sym=apply_sym,
    )


def build_lid_upper(
    *,
    inner_radius_mm: float,
    outer_radius_mm_1: float,
    outer_radius_mm_2: float,
    height_mm: float,
    z_offset_mm: float,
    end_ang_deg: float,
    seg_theta: int,
    max_mesh_size: float,
    model_name: str = "lid_upper",
    start_ang_deg: float = 0.0,
    include_window: bool = False,
    window_width_mm: float = 300.0,
    min_mesh_size: Optional[float] = None,
    material: Optional[RadiaMaterial] = None,
    symmetries=None,
    color: Sequence[float] = IRON_COLOR,
    comm=None,
    apply_mat: bool = True,
    apply_color: bool = True,
    apply_sym: bool = False,
) -> MagnetizedComponent:
    """Tapered upper-lid iron piece (two outer radii)."""
    z_bottom = z_offset_mm - height_mm
    a0 = np.deg2rad(start_ang_deg)
    a1 = np.deg2rad(end_ang_deg)

    def occ():
        _occ_lid_upper(inner_radius_mm, outer_radius_mm_1, outer_radius_mm_2, a0, a1,
                       seg_theta, z_bottom, height_mm, include_window, window_width_mm)

    return MagnetizedComponent.from_gmsh_occ(
        occ, model_name=model_name,
        mesh_size_min=1.0 if min_mesh_size is None else min_mesh_size,
        mesh_size_max=max_mesh_size,
        comm=comm, material=material, symmetries=symmetries, color=color,
        apply_mat=apply_mat, apply_color=apply_color, apply_sym=apply_sym,
    )


def build_pole(
    *,
    inner_radius_mm: float,
    outer_radius_mm: float,
    height_mm: float,
    half_angle_deg: float,
    pole_zs: float,
    top_offsets_mm: Sequence[float],
    side_offsets_deg: Sequence[float],
    max_mesh_size: float,
    min_mesh_size: Optional[float] = None,
    model_name: str = "pole",
    material: Optional[RadiaMaterial] = None,
    symmetries=None,
    color: Sequence[float] = IRON_COLOR,
    comm=None,
    apply_mat: bool = True,
    apply_color: bool = True,
    apply_sym: bool = False,
) -> MagnetizedComponent:
    """Shimmed pole base built as a stack of radial segments (top + side shims).

    ``top_offsets_mm`` and ``side_offsets_deg`` each have ``num_segments + 1``
    entries (the shim values at the radial-segment boundaries). ``pole_zs`` is
    the z of the pole base (bottom); segments extrude upward by ``height_mm``
    plus the per-boundary top-shim offset.
    """
    top_offsets = list(top_offsets_mm)
    side_offsets = list(side_offsets_deg)
    n_segs = len(top_offsets) - 1
    if n_segs < 1 or len(side_offsets) != len(top_offsets):
        raise ValueError("top_offsets_mm and side_offsets_deg must be equal-length, len >= 2.")

    def occ():
        volumes = []
        for i in range(n_segs):
            r_in = inner_radius_mm + (outer_radius_mm - inner_radius_mm) * i / n_segs
            r_out = inner_radius_mm + (outer_radius_mm - inner_radius_mm) * (i + 1) / n_segs
            ang_in = np.deg2rad(half_angle_deg + side_offsets[i])
            ang_out = np.deg2rad(half_angle_deg + side_offsets[i + 1])
            volumes.append(
                _occ_pole_segment(r_in, r_out, ang_in, ang_out, pole_zs, height_mm,
                                  top_offsets[i], top_offsets[i + 1])
            )
            gmsh.model.occ.synchronize()
        _fuse_volumes(volumes)

    return MagnetizedComponent.from_gmsh_occ(
        occ, model_name=model_name,
        mesh_size_min=10.0 if min_mesh_size is None else min_mesh_size,
        mesh_size_max=max_mesh_size,
        comm=comm, material=material, symmetries=symmetries, color=color,
        apply_mat=apply_mat, apply_color=apply_color, apply_sym=apply_sym,
    )
