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
               include_window=False, window_width=300.0,
               window_center_deg=45.0, access_holes=None) -> int:
    """Annular wedge via cylinder difference, with optional cuts.

    - window: rectangular slot cut FULLY through the piece vertically,
      centered on the ``window_center_deg`` azimuth (legacy yoke-wall
      geometry: exact 45-degree behavior preserved, generalized by rotating
      the legacy box to other azimuths).
    - access_holes: vertical cylindrical bores through the full height,
      each ``{"center_xy": [x, y], "diameter_mm": d}`` (RF stems, survey /
      access holes -- from the old lid-upper generator).
    """
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
        extra = np.deg2rad(window_center_deg - 45.0)
        if abs(extra) > 1e-12:
            gmsh.model.occ.rotate([(3, window_box)], 0, 0, 0, 0, 0, 1, extra)
        new_tags2, _ = gmsh.model.occ.cut([(3, volume)], [(3, window_box)])
        volume = new_tags2[0][1]

    for hole in (access_holes or []):
        cx, cy = hole["center_xy"]
        r = 0.5 * hole["diameter_mm"]
        bore = gmsh.model.occ.addCylinder(cx, cy, z_bottom - 5.0,
                                          0.0, 0.0, height + 10.0, r)
        new_tags3, _ = gmsh.model.occ.cut([(3, volume)], [(3, bore)])
        volume = new_tags3[0][1]

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
# OCC-only callables (for conforming mesh groups: geometry WITHOUT meshing;
# the group meshes everything together in one gmsh model)
# -----------------------------
def occ_wedge_callable(*, inner_radius_mm, outer_radius_mm, height_mm,
                       z_offset_mm, end_ang_deg, start_ang_deg=0.0,
                       include_window=False, window_width_mm=300.0,
                       window_center_deg=45.0, access_holes=None):
    z_bottom = z_offset_mm - height_mm
    a0 = np.deg2rad(start_ang_deg)
    a1 = np.deg2rad(end_ang_deg)

    def occ():
        _occ_wedge(inner_radius_mm, outer_radius_mm, a0, a1, z_bottom,
                   height_mm, include_window, window_width_mm,
                   window_center_deg, access_holes)
    return occ


def occ_lid_upper_callable(*, inner_radius_mm, outer_radius_mm_1,
                           outer_radius_mm_2, height_mm, z_offset_mm,
                           end_ang_deg, seg_theta, start_ang_deg=0.0,
                           include_window=False, window_width_mm=300.0):
    z_bottom = z_offset_mm - height_mm
    a0 = np.deg2rad(start_ang_deg)
    a1 = np.deg2rad(end_ang_deg)

    def occ():
        _occ_lid_upper(inner_radius_mm, outer_radius_mm_1, outer_radius_mm_2,
                       a0, a1, seg_theta, z_bottom, height_mm,
                       include_window, window_width_mm)
    return occ


def occ_pole_callable(*, inner_radius_mm, outer_radius_mm, height_mm,
                      half_angle_deg, pole_zs, top_offsets_mm,
                      side_offsets_deg, cylindrical_faces=False):
    """OCC builder for the shimmed pole (segments + fuse).

    cylindrical_faces=True: the pole's radial extremes become TRUE cylinder
    patches at inner_radius/outer_radius (the segment stack is built
    radially oversized and boolean-intersected with the exact annulus).
    Needed for conforming mesh groups: only exactly-coincident surfaces
    (e.g. the pole rim against an STP lid bore, both true cylinders at the
    same radius) fragment into a single shared face. The default False
    keeps today's SurfaceFilling approximation (legacy geometry).
    """
    top_offsets = list(top_offsets_mm)
    side_offsets = list(side_offsets_deg)
    n_segs = len(top_offsets) - 1
    if n_segs < 1 or len(side_offsets) != len(top_offsets):
        raise ValueError("top_offsets_mm and side_offsets_deg must be "
                         "equal-length, len >= 2.")

    def occ():
        # May run inside a SHARED group model: track our own volumes by
        # before/after diff, never by "latest entity".
        existing = {t for _d, t in gmsh.model.occ.getEntities(3)}
        pad = 2.0 if cylindrical_faces else 0.0
        r0 = max(inner_radius_mm - pad, 1e-3)
        r1 = outer_radius_mm + pad
        volumes = []
        for i in range(n_segs):
            r_in = inner_radius_mm + (outer_radius_mm - inner_radius_mm) * i / n_segs
            r_out = inner_radius_mm + (outer_radius_mm - inner_radius_mm) * (i + 1) / n_segs
            if i == 0:
                r_in = r0
            if i == n_segs - 1:
                r_out = r1
            ang_in = np.deg2rad(half_angle_deg + side_offsets[i])
            ang_out = np.deg2rad(half_angle_deg + side_offsets[i + 1])
            volumes.append(
                _occ_pole_segment(r_in, r_out, ang_in, ang_out, pole_zs,
                                  height_mm, top_offsets[i], top_offsets[i + 1])
            )
            gmsh.model.occ.synchronize()
        _fuse_volumes(volumes)
        if cylindrical_faces:
            mine = [(3, t) for _d, t in gmsh.model.occ.getEntities(3)
                    if t not in existing]
            z0 = pole_zs - 10.0
            dz = height_mm + max(top_offsets) + 60.0
            outer_cyl = gmsh.model.occ.addCylinder(0, 0, z0, 0, 0, dz,
                                                   outer_radius_mm)
            inner_cyl = gmsh.model.occ.addCylinder(0, 0, z0, 0, 0, dz,
                                                   inner_radius_mm)
            ann, _ = gmsh.model.occ.cut([(3, outer_cyl)], [(3, inner_cyl)])
            gmsh.model.occ.intersect(mine, ann)
            gmsh.model.occ.synchronize()
    return occ


def occ_swept_polygon_callable(*, polygon, axis, start_angle_deg,
                               end_angle_deg, axis_point=(0.0, 0.0, 0.0)):
    """OCC builder for a solid of revolution: a 2D polygon swept about an axis.

    ``polygon`` is (N, 2) — local coordinates in the STARTING PLANE, which
    contains the (3D) axis and the radial direction at ``start_angle_deg``:
      - local x: distance from the axis along that radial direction (keep
        x > 0 — the polygon must not cross the axis);
      - local y: position ALONG the axis (from ``axis_point``).
    The angular zero reference is the projection of global +x onto the plane
    perpendicular to the axis (global +y if the axis is ±x). For axis
    [0, 0, 1] this is standard cylindrical coordinates: local x = r,
    local y = z, angles = azimuth from +x.

    The sweep ``end_angle_deg - start_angle_deg`` must be in (0, 360).
    """
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        raise ValueError("polygon must be an (N, 2) array with N >= 3")
    a = np.asarray(axis, dtype=float)
    na = np.linalg.norm(a)
    if na < 1e-12:
        raise ValueError("axis must be a nonzero 3-vector")
    a = a / na
    p0 = np.asarray(axis_point, dtype=float)
    sweep = np.deg2rad(end_angle_deg - start_angle_deg)
    if not (0.0 < sweep < 2.0 * np.pi):
        raise ValueError("end_angle_deg - start_angle_deg must be in (0, 360); "
                         "build a full ring as two half sweeps")

    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(ref, a))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e0 = ref - float(np.dot(ref, a)) * a
    e0 = e0 / np.linalg.norm(e0)
    e1 = np.cross(a, e0)
    th0 = np.deg2rad(start_angle_deg)
    rhat = np.cos(th0) * e0 + np.sin(th0) * e1

    def occ():
        pts = []
        for x, y in poly:
            p = p0 + x * rhat + y * a
            pts.append(gmsh.model.occ.addPoint(float(p[0]), float(p[1]),
                                               float(p[2])))
        lines = [gmsh.model.occ.addLine(pts[i], pts[(i + 1) % len(pts)])
                 for i in range(len(pts))]
        loop = gmsh.model.occ.addCurveLoop(lines)
        surf = gmsh.model.occ.addPlaneSurface([loop])
        gmsh.model.occ.revolve([(2, surf)], float(p0[0]), float(p0[1]),
                               float(p0[2]), float(a[0]), float(a[1]),
                               float(a[2]), sweep)
    return occ


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
    window_center_deg: float = 45.0,
    access_holes: Optional[List[dict]] = None,
    min_mesh_size: Optional[float] = None,
    material: Optional[RadiaMaterial] = None,
    symmetries=None,
    color: Sequence[float] = IRON_COLOR,
    comm=None,
    apply_mat: bool = True,
    apply_color: bool = True,
    apply_sym: bool = False,
) -> MagnetizedComponent:
    """Annular-wedge iron piece (yoke wall, lower lid, extraction channel),
    with optional full-vertical window slot and vertical access-hole bores."""
    occ = occ_wedge_callable(
        inner_radius_mm=inner_radius_mm, outer_radius_mm=outer_radius_mm,
        height_mm=height_mm, z_offset_mm=z_offset_mm,
        end_ang_deg=end_ang_deg, start_ang_deg=start_ang_deg,
        include_window=include_window, window_width_mm=window_width_mm,
        window_center_deg=window_center_deg, access_holes=access_holes)

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
    occ = occ_lid_upper_callable(
        inner_radius_mm=inner_radius_mm, outer_radius_mm_1=outer_radius_mm_1,
        outer_radius_mm_2=outer_radius_mm_2, height_mm=height_mm,
        z_offset_mm=z_offset_mm, end_ang_deg=end_ang_deg,
        seg_theta=seg_theta, start_ang_deg=start_ang_deg,
        include_window=include_window, window_width_mm=window_width_mm)

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
    cylindrical_faces: bool = False,
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
    occ = occ_pole_callable(
        inner_radius_mm=inner_radius_mm, outer_radius_mm=outer_radius_mm,
        height_mm=height_mm, half_angle_deg=half_angle_deg, pole_zs=pole_zs,
        top_offsets_mm=top_offsets_mm, side_offsets_deg=side_offsets_deg,
        cylindrical_faces=cylindrical_faces)

    return MagnetizedComponent.from_gmsh_occ(
        occ, model_name=model_name,
        mesh_size_min=10.0 if min_mesh_size is None else min_mesh_size,
        mesh_size_max=max_mesh_size,
        comm=comm, material=material, symmetries=symmetries, color=color,
        apply_mat=apply_mat, apply_color=apply_color, apply_sym=apply_sym,
    )


def build_swept_polygon(
    *,
    polygon,
    axis,
    start_angle_deg: float,
    end_angle_deg: float,
    max_mesh_size: float,
    axis_point=(0.0, 0.0, 0.0),
    min_mesh_size: Optional[float] = None,
    model_name: str = "swept_polygon",
    material: Optional[RadiaMaterial] = None,
    symmetries=None,
    color: Sequence[float] = IRON_COLOR,
    comm=None,
    apply_mat: bool = True,
    apply_color: bool = True,
    apply_sym: bool = False,
) -> MagnetizedComponent:
    """Iron piece from a 2D polygon swept about an axis (solid of revolution
    sector). See occ_swept_polygon_callable for the local-frame convention."""
    occ = occ_swept_polygon_callable(
        polygon=polygon, axis=axis, start_angle_deg=start_angle_deg,
        end_angle_deg=end_angle_deg, axis_point=axis_point)

    return MagnetizedComponent.from_gmsh_occ(
        occ, model_name=model_name,
        mesh_size_min=1.0 if min_mesh_size is None else min_mesh_size,
        mesh_size_max=max_mesh_size,
        comm=comm, material=material, symmetries=symmetries, color=color,
        apply_mat=apply_mat, apply_color=apply_color, apply_sym=apply_sym,
    )
