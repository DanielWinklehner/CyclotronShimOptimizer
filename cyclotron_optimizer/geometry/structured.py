"""Structured polar-grid slicing of STP solids ("Option C").

Discretizes a revolved-ish iron solid (yoke) loaded from an STP file into
a STRUCTURED core of annular-sector prisms plus a CONFORMING tetrahedral
skin, instead of an unstructured all-tet mesh. Structured cores condition
the relaxation dramatically better at a fraction of the element count
(see scripts/perturb_study/RECMAG_GPU_PLAN.md and validate_recmag.py) and
put the 60 MeV machine back inside the GPU's dense-IM element budget.

Pipeline (rank 0, inside one gmsh session):
  1. Import the STP; detect its z-axis cylinder radii and z-plane
     positions (snap anchors).
  2. Build the grid edges: anchors + equal-spaced fill to the dr/dz
     targets. No theta cuts -- rings only; the azimuthal subdivision is
     purely analytic at emission time.
  3. Fragment the solid with the cutting surfaces (full cylinders +
     disks), checking volume conservation.
  4. Classify every fragment by its own FACE INVENTORY: a clean interior
     ring has exactly two z-cylinders + two z-planes + the wedge side
     planes, all snapping to grid edges, and its volume matches the
     analytic ring volume. Everything else (touches a cone, a tilted
     cylinder, any true CAD detail) is SKIN.
     (Deliberately NOT center-of-mass binning: the COM of a thin wide
     annular sector falls outside its own radial interval.)
  5. Remove interior rings from the model and tet-mesh only the skin.

Emission (all ranks, from the broadcast payload): interior rings are
subdivided azimuthally into single-chord annular-trapezoid prism cells
built as rad.ObjPolyhdr with vertices ON the true radii (inscribed
chords; volume deficit ~dtheta^2/6 per cell, reported in the stats --
dtheta is the ONLY faceting knob).

Two radia constraints shape the emission (measured, 2026-07-21):
- ObjPolyhdr (radTPolyhedron) is REQUIRED -- ObjMltExtPgn would build
  extruded-polygon elements that the GPU interaction-matrix assembly
  cannot pack (CPU fallback).
- Cells MUST be CONVEX: radia rejects non-convex polyhedra ("Non-convex
  polyhedron encountered", one winding orientation) or heap-CRASHES on
  them (the other orientation). Hence exactly one chord per cell; a
  multi-chord cell has a reflex inner edge and dies.

The structured cells never require the on-plane jitter repair to be off:
keep rad.FldLenRndSw at its default ('on'); cell centers of structured
grids lie exactly on neighbor face-extension planes and the deterministic
AbsRandMagnitude guard is what makes that well-defined.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "slice_stp_polar",
    "emit_prism_cells",
    "structured_defaults",
]


# ---------------------------------------------------------------------------
# Defaults for the ComponentSpec `structure:` dict
# ---------------------------------------------------------------------------
def structured_defaults() -> Dict[str, Any]:
    return {
        "type": "polar_grid",
        "dr_mm": 120.0,          # target radial fill spacing between anchors
        "dz_mm": 120.0,          # target axial fill spacing between anchors
        "dtheta_deg": 2.5,       # azimuthal cell size = arc faceting knob
                                 # (volume deficit ~ dtheta^2/6: 0.03% @ 2.5 deg)
        "theta_span_deg": (0.0, 45.0),  # azimuthal extent of the folded solid
        "snap": True,            # detect CAD radii / z-planes as grid anchors
        "element": "prism",      # 'prism' (ObjPolyhdr); 'recmag' reserved
        "min_fill_frac": 0.35,   # never place a fill edge closer than this
                                 # fraction of the target to an anchor
    }


def _merge_structure(structure: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = structured_defaults()
    for k, v in (structure or {}).items():
        if k not in out:
            raise ValueError(f"Unknown structure option {k!r}; "
                             f"known: {sorted(out)}")
        out[k] = v
    if out["type"] != "polar_grid":
        raise ValueError(f"Unsupported structure type {out['type']!r} "
                         "(only 'polar_grid' for now)")
    if out["element"] != "prism":
        raise NotImplementedError(
            "structure.element='recmag' is reserved for a later step; "
            "use 'prism' (same conditioning win, exact annular geometry)")
    return out


# ---------------------------------------------------------------------------
# Rank-0 gmsh work
# ---------------------------------------------------------------------------
_COINCIDENCE_TOL = 0.05   # mm: snap/dedupe tolerance for detected features
_THETA_TOL_DEG = 0.01     # deg: snap tolerance for azimuthal cut planes
_VOL_RTOL = 1e-6          # relative tolerance for the cell volume test


def _surface_info(gmsh, tag: int) -> Tuple[str, Dict[str, float]]:
    """Classify one surface: ('zcyl', {'r'}), ('zplane', {'z'}),
    ('plane', {'theta_deg'}) or ('other', {})."""
    stype = gmsh.model.getType(2, tag)
    if stype not in ("Cylinder", "Plane"):
        return ("other", {})
    lo, hi = gmsh.model.getParametrizationBounds(2, tag)
    u = 0.5 * (lo[0] + hi[0])
    v = 0.5 * (lo[1] + hi[1])
    xyz = gmsh.model.getValue(2, tag, [u, v])
    n = gmsh.model.getNormal(tag, [u, v])
    nn = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) or 1.0
    nz = n[2] / nn
    if stype == "Cylinder":
        if abs(nz) > 1e-6:
            return ("other", {})           # tilted cylinder
        # Verify the axis is the z-axis (not just any vertical cylinder):
        # the normal must be radial at the sample point.
        r = math.hypot(xyz[0], xyz[1])
        if r < 1e-9:
            return ("other", {})
        rhat = (xyz[0] / r, xyz[1] / r)
        radial = abs(n[0] / nn * rhat[0] + n[1] / nn * rhat[1])
        if radial < 1.0 - 1e-6:
            return ("other", {})           # off-axis cylinder
        return ("zcyl", {"r": r})
    # Plane:
    if abs(nz) > 1.0 - 1e-9:
        return ("zplane", {"z": xyz[2]})
    if abs(nz) < 1e-9:
        theta = math.degrees(math.atan2(n[1], n[0]))
        return ("plane", {"theta_deg": theta})
    return ("other", {})


def _dedupe(vals: Sequence[float], tol: float) -> List[float]:
    out: List[float] = []
    for v in sorted(vals):
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


def _build_edges(anchors: Sequence[float], lo: float, hi: float,
                 target: float, min_fill_frac: float) -> List[float]:
    """Grid edges: [lo, hi] + interior anchors + equal-spaced fill.

    Fill edges are equally spaced inside each anchor interval, so by
    construction they keep >= (interval/n)/2 clearance from the anchors;
    intervals shorter than min_fill_frac*target simply get no fill (one
    structured layer -- thin but perfectly shaped).
    """
    core = [a for a in anchors if lo + _COINCIDENCE_TOL < a < hi - _COINCIDENCE_TOL]
    edges = _dedupe([lo] + core + [hi], _COINCIDENCE_TOL)
    out: List[float] = [edges[0]]
    for a, b in zip(edges, edges[1:]):
        gap = b - a
        n = max(1, int(round(gap / target)))
        if gap / n < min_fill_frac * target and n > 1:
            n -= 1
        for j in range(1, n):
            out.append(a + gap * j / n)
        out.append(b)
    return out


def _snap_to(edges: Sequence[float], v: float) -> Optional[float]:
    for e in edges:
        if abs(v - e) <= _COINCIDENCE_TOL:
            return e
    return None


def slice_stp_polar(
    stp_path: str,
    *,
    structure: Optional[Dict[str, Any]] = None,
    mesh_size_max: Optional[float] = None,
    mesh_size_min: Optional[float] = None,
    model_name: str = "structured",
    gmsh_verbosity: int = 2,
) -> Dict[str, Any]:
    """Rank-0 slicing: returns a picklable payload.

    Payload keys:
      rings:      [(r0, r1, z0, z1)] interior rings (full theta span each)
      skin_tets:  [[[x,y,z] x4]] skin tetrahedra
      theta_span: (t0_deg, t1_deg)
      structure:  the merged structure options
      stats:      diagnostics dict (also printed)
    """
    import gmsh

    from cyclotron_optimizer.geometry.components import (  # local import: avoid cycle
        _pin_gmsh_determinism,
    )

    st = _merge_structure(structure)
    t0, t1 = (float(st["theta_span_deg"][0]), float(st["theta_span_deg"][1]))
    span = math.radians(t1 - t0)
    if span <= 0:
        raise ValueError("theta_span_deg must be increasing")

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        _pin_gmsh_determinism()
        gmsh.model.add(model_name)
        gmsh.model.occ.importShapes(str(stp_path))
        gmsh.model.occ.synchronize()

        vols = gmsh.model.getEntities(3)
        if len(vols) != 1:
            raise ValueError(
                f"{stp_path}: expected ONE solid for polar slicing, found "
                f"{len(vols)} (slice multi-solid files per part)")
        yoke = vols[0][1]
        v_cad = gmsh.model.occ.getMass(3, yoke)
        bb = gmsh.model.getBoundingBox(3, yoke)
        r_lo = 0.0
        r_hi = math.hypot(max(abs(bb[0]), abs(bb[3])),
                          max(abs(bb[1]), abs(bb[4]))) + 1.0
        z_lo, z_hi = bb[2], bb[5]

        # --- detect snap anchors from the CAD faces ---
        radii: List[float] = []
        zplanes: List[float] = []
        for _d, s in gmsh.model.getBoundary([(3, yoke)], combined=False,
                                            oriented=False):
            kind, info = _surface_info(gmsh, s)
            if kind == "zcyl":
                radii.append(info["r"])
            elif kind == "zplane":
                zplanes.append(info["z"])
        radii = _dedupe(radii, _COINCIDENCE_TOL)
        zplanes = _dedupe(zplanes, _COINCIDENCE_TOL)
        if not st["snap"]:
            radii, zplanes = [], []

        # Radial extent of the iron: innermost/outermost detected cylinder
        # if available, else bbox-derived.
        r_min = radii[0] if radii else 0.0
        r_max = radii[-1] if radii else r_hi - 1.0
        r_edges = _build_edges(radii, r_min, r_max, float(st["dr_mm"]),
                               float(st["min_fill_frac"]))
        z_edges = _build_edges(zplanes, z_lo, z_hi, float(st["dz_mm"]),
                               float(st["min_fill_frac"]))
        n_theta = max(1, int(round((t1 - t0) / float(st["dtheta_deg"]))))
        th_edges = [t0 + (t1 - t0) * j / n_theta for j in range(n_theta + 1)]

        # --- cutting tools: SOLID primitives (robust OCC booleans; the
        # earlier surface tools tripped BOPAlgo self-intersections) ---
        # A solid tool cuts with every face that intersects the iron, so
        # each tool is arranged to intersect it with EXACTLY ONE face:
        #   r cut:  coaxial cylinder, z-padded beyond the solid
        #   z cut:  fat cylinder capped AT the cut plane, radially padded
        #   th cut: rotated box whose y'=0 face is the half-plane through
        #           the axis; every other face lies outside the iron (its
        #           x'=0 face contains the axis at theta+90 deg, which for
        #           a <=180 deg part never re-enters the material)
        tools: List[Tuple[int, int]] = []
        pad = 2.0
        zp0, zp1 = z_lo - pad, z_hi + pad
        big = r_hi + 10.0
        for r in r_edges[1:-1]:
            tools.append((3, gmsh.model.occ.addCylinder(
                0.0, 0.0, zp0, 0.0, 0.0, zp1 - zp0, r)))
        for z in z_edges[1:-1]:
            tools.append((3, gmsh.model.occ.addCylinder(
                0.0, 0.0, zp0, 0.0, 0.0, z - zp0, big)))
        for th in th_edges[1:-1]:
            b = gmsh.model.occ.addBox(0.0, 0.0, zp0, big, big, zp1 - zp0)
            gmsh.model.occ.rotate([(3, b)], 0, 0, 0, 0, 0, 1,
                                  math.radians(th))
            tools.append((3, b))

        out_map = None
        if tools:
            _out, out_map = gmsh.model.occ.fragment([(3, yoke)], tools)
        pieces = ([(d, t) for d, t in out_map[0] if d == 3]
                  if out_map is not None else [(3, yoke)])

        v_pieces = sum(gmsh.model.occ.getMass(3, t) for _d, t in pieces)
        rel = abs(v_pieces - v_cad) / max(v_cad, 1e-30)
        if rel > 1e-3:
            raise RuntimeError(
                f"polar slicing of {stp_path}: fragment changed the solid "
                f"volume by {rel:.2e} relative -- OCC boolean failed")
        gmsh.model.occ.synchronize()

        # --- classify pieces by their own face inventory (NOT by binning
        # centers-of-mass: the COM of a wide annular sector can fall outside
        # its own radial interval). A clean interior cell has exactly two
        # grid cylinders, two grid z-planes and two grid theta-planes, and
        # the analytic sector volume. ---
        cells: List[Tuple[float, float, float, float, float, float]] = []
        skin_tags: List[int] = []
        v_interior = 0.0
        for _d, t in pieces:
            faces = gmsh.model.getBoundary([(3, t)], combined=False,
                                           oriented=False)
            cyl_r: List[float] = []
            pl_z: List[float] = []
            pl_th: List[float] = []
            clean = True
            for _fd, f in faces:
                kind, info = _surface_info(gmsh, f)
                if kind == "zcyl":
                    r = _snap_to(r_edges, info["r"])
                    if r is None:
                        clean = False
                        break
                    cyl_r.append(r)
                elif kind == "zplane":
                    z = _snap_to(z_edges, info["z"])
                    if z is None:
                        clean = False
                        break
                    pl_z.append(z)
                elif kind == "plane":
                    # normal is +-90 deg from the face's azimuth
                    th = info["theta_deg"]
                    got = None
                    for cand in (th - 90.0, th + 90.0):
                        cand %= 360.0
                        for e in th_edges:
                            if abs(cand - (e % 360.0)) <= _THETA_TOL_DEG:
                                got = e
                                break
                        if got is not None:
                            break
                    if got is None:
                        clean = False
                        break
                    pl_th.append(got)
                else:
                    clean = False
                    break
            cyl_r = _dedupe(cyl_r, _COINCIDENCE_TOL)
            pl_z = _dedupe(pl_z, _COINCIDENCE_TOL)
            pl_th = _dedupe(pl_th, _THETA_TOL_DEG)
            if (clean and len(cyl_r) == 2 and len(pl_z) == 2
                    and len(pl_th) == 2):
                a, b = cyl_r
                za, zb = pl_z
                ta, tb = pl_th
                v_exp = (0.5 * math.radians(tb - ta) * (b * b - a * a)
                         * (zb - za))
                v_act = gmsh.model.occ.getMass(3, t)
                if abs(v_act - v_exp) <= _VOL_RTOL * v_exp:
                    cells.append((a, b, ta, tb, za, zb))
                    v_interior += v_act
                    continue
            skin_tags.append(t)

        # --- mesh only the skin ---
        skin_tets: List[List[List[float]]] = []
        v_skin = v_cad - v_interior
        n_skin_pieces = len(skin_tags)
        if skin_tags:
            # FUSE the skin pieces back together before meshing: the
            # fragment imprinted every grid cut onto them (2.5-deg sector
            # faces, fill-cylinder slivers against drafted CAD walls...),
            # which forces absurdly fine tets -- measured 15k skin tets vs
            # 16k for the WHOLE all-tet yoke. Fusing dissolves all internal
            # imprints (slivers included); the compound's outer boundary
            # still coincides exactly with the kept interior cells' faces,
            # so the no-gap/no-overlap tiling is preserved.
            if len(skin_tags) > 1:
                v_skin_pre = sum(gmsh.model.occ.getMass(3, t)
                                 for t in skin_tags)
                fused, _fmap = gmsh.model.occ.fuse(
                    [(3, skin_tags[0])],
                    [(3, t) for t in skin_tags[1:]])
                gmsh.model.occ.synchronize()
                skin_tags = [t for d, t in fused if d == 3]
                v_skin_post = sum(gmsh.model.occ.getMass(3, t)
                                  for t in skin_tags)
                srel = abs(v_skin_post - v_skin_pre) / max(v_skin_pre, 1e-30)
                if srel > 1e-6:
                    print(f"[structured {model_name}] WARNING: skin fuse "
                          f"shifted the skin volume by {srel:.2e} relative",
                          flush=True)
            # Mesh ONLY the skin volumes via visibility masking. (Removing
            # the interior volumes from the OCC model instead leaves
            # orphaned boundary faces that poison the 3D mesher --
            # observed: tetgen 'segment and facet intersect' PLC error.)
            if cells:
                gmsh.model.setVisibility(gmsh.model.getEntities(3), 0, True)
                gmsh.model.setVisibility([(3, t) for t in skin_tags], 1, True)
                gmsh.option.setNumber("Mesh.MeshOnlyVisible", 1)
            if mesh_size_max is not None:
                gmsh.option.setNumber("Mesh.MeshSizeMax", float(mesh_size_max))
            if mesh_size_min is not None:
                gmsh.option.setNumber("Mesh.MeshSizeMin", float(mesh_size_min))
            gmsh.model.mesh.generate(3)

            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            nodes: Dict[int, List[float]] = {}
            for i, tag in enumerate(node_tags):
                j = 3 * i
                nodes[int(tag)] = [float(node_coords[j]),
                                   float(node_coords[j + 1]),
                                   float(node_coords[j + 2])]
            for t in skin_tags:
                etypes, _etags, enodes = gmsh.model.mesh.getElements(3, t)
                for et, conn in zip(etypes, enodes):
                    if int(et) != 4:
                        continue
                    for i in range(0, len(conn), 4):
                        skin_tets.append([nodes[int(c)] for c in conn[i:i + 4]])

        # --- stats (inscribed-chord deficit of the emitted prisms) ---
        sub = span / n_theta
        v_model = (math.sin(sub) / sub) * v_interior + v_skin

        stats = {
            "cad_volume_mm3": v_cad,
            "interior_cells": len(cells),
            "skin_pieces": n_skin_pieces,
            "skin_volumes_after_fuse": len(skin_tags),
            "skin_tets": len(skin_tets),
            "skin_volume_frac": v_skin / v_cad,
            "n_theta": n_theta,
            "elements_total": len(cells) + len(skin_tets),
            "inscribed_volume_deficit_frac": (v_cad - v_model) / v_cad,
            "r_edges": len(r_edges),
            "z_edges": len(z_edges),
            "detected_radii": radii,
            "detected_zplanes": zplanes,
            "min_cell_dr": min((b - a for a, b, *_rest in cells),
                               default=float("nan")),
            "min_cell_dz": min((z1 - z0 for *_rest, z0, z1 in cells),
                               default=float("nan")),
        }
        print(f"[structured {model_name}] {stats['interior_cells']} prism "
              f"cells + {stats['skin_tets']} skin tets "
              f"({stats['skin_pieces']} skin pieces, "
              f"{100 * stats['skin_volume_frac']:.2f}% of volume); "
              f"inscribed-chord volume deficit "
              f"{100 * stats['inscribed_volume_deficit_frac']:.4f}%",
              flush=True)

        return {
            "cells": cells,
            "skin_tets": skin_tets,
            "theta_span": (t0, t1),
            "structure": st,
            "stats": stats,
        }
    finally:
        gmsh.finalize()


# ---------------------------------------------------------------------------
# Emission: analytic prism cells -> radia polyhedra (runs on ALL ranks)
# ---------------------------------------------------------------------------
def _prism_polyhedron(r0: float, r1: float, thetas_rad: Sequence[float],
                      z0: float, z1: float,
                      magn: Optional[Sequence[float]] = None) -> int:
    """One annular-sector prism as rad.ObjPolyhdr (planar faces only).

    Vertices sit ON the true radii at the given angles (inscribed chords).
    Faces: bottom/top polygons + inner/outer wall quads + 2 radial ends.
    """
    from cyclotron_optimizer.geometry.components import _call_radia

    c = len(thetas_rad) - 1
    cos = [math.cos(t) for t in thetas_rad]
    sin = [math.sin(t) for t in thetas_rad]
    verts: List[List[float]] = []
    for z in (z0, z1):
        verts += [[r0 * cos[j], r0 * sin[j], z] for j in range(c + 1)]  # inner
        verts += [[r1 * cos[j], r1 * sin[j], z] for j in range(c + 1)]  # outer
    # 1-based ids: bottom inner I=1..c+1, bottom outer O=c+2..2c+2,
    # top inner TI=2c+3..3c+3, top outer TO=3c+4..4c+4
    I = lambda j: 1 + j                    # noqa: E741
    O = lambda j: c + 2 + j                # noqa: E741
    TI = lambda j: 2 * c + 3 + j
    TO = lambda j: 3 * c + 4 + j

    faces: List[List[int]] = []
    faces.append([I(j) for j in range(c + 1)]
                 + [O(j) for j in reversed(range(c + 1))])      # bottom
    faces.append([O(j) for j in range(c + 1)]
                 + [I(j) for j in reversed(range(c + 1))]
                 )                                              # top (indices shifted below)
    faces[1] = [v + 2 * (c + 1) for v in faces[1]]
    for j in range(c):                                          # inner wall
        faces.append([I(j), TI(j), TI(j + 1), I(j + 1)])
    for j in range(c):                                          # outer wall
        faces.append([O(j), O(j + 1), TO(j + 1), TO(j)])
    faces.append([I(0), O(0), TO(0), TI(0)])                    # side at t0
    faces.append([I(c), TI(c), TO(c), O(c)])                    # side at t1
    if magn is not None:
        return _call_radia("ObjPolyhdr", verts, faces, list(magn))
    return _call_radia("ObjPolyhdr", verts, faces)


def emit_prism_cells(payload: Dict[str, Any]) -> Tuple[List[int], List[Tuple]]:
    """Build all structured prism cells; returns (radia_ids, cell_defs).

    One chord per cell -- radia's ObjPolyhdr requires CONVEX polyhedra
    (see module docstring); dtheta_deg alone controls the arc faceting.
    """
    ids: List[int] = []
    for (r0, r1, ta, tb, z0, z1) in payload["cells"]:
        ids.append(_prism_polyhedron(r0, r1,
                                     [math.radians(ta), math.radians(tb)],
                                     z0, z1))
    return ids, list(payload["cells"])
