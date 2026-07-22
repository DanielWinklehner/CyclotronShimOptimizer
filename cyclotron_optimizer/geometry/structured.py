"""Structured polar-grid slicing of STP/OCC solids ("Option C", v2).

Discretizes revolved-ish iron parts into a STRUCTURED core of annular-
sector prisms plus a CONFORMING tetrahedral skin, instead of an
unstructured all-tet mesh. Structured cores condition the relaxation far
better at a fraction of the element count (RECMAG_GPU_PLAN.md,
validate_recmag.py) and put the 60 MeV machine back inside the GPU's
dense-IM element budget.

v2 adds (see the memory note structured-stp-slicing):
  - MESH-GROUP support: several components (structured and/or tet-only)
    built in ONE gmsh model with conforming contacts
    (``build_structured_group``, used by geometry._build_mesh_group).
  - Multi-solid members (e.g. base pole + side shim + VP insert in one
    STP): the grid is applied to every solid; non-revolved solids simply
    classify as skin.
  - ENVELOPE rule ``core_clip``: cylindrical-coordinate bounds outside
    which no cell may be core (e.g. ``{z_max: -140}`` keeps the pole's
    gap face, shim range and tip steps in the tet skin).
  - MARGIN rule ``skin_margin_deg``: per-ring azimuthal dilation -- core
    cells within the margin of any non-core azimuth (spiral wall, hole)
    are demoted, so shim-envelope walls stay inside the skin.
  - DEMOTE-AND-MERGE rule ``min_skin_thickness_mm``: a skin piece
    thinner than the threshold absorbs its interior neighbor cell(s), so
    the mesher always has >= a cell of grading room ("remove the
    adjacent wall").
  - Payload CACHING keyed by input-file digests + parameters
    (output/structured_cache/); a frozen config re-slices only when the
    CAD or the grid changes.

Pipeline (rank 0, one gmsh session):
  1. Import every member (STP or OCC callable); group-fragment all
     volumes (imprints contacts, detects overlaps) -- the mesh_group
     contract.
  2. Per structured member: detect its z-cylinder radii / z-plane
     anchors, build the grid (anchors + fill), fragment its volumes with
     SOLID cutting tools, classify every piece by its own FACE INVENTORY
     (two grid cylinders + two grid z-planes + two grid theta-planes +
     the analytic sector volume; never center-of-mass binning -- the COM
     of a wide annular sector falls outside its own radial interval),
     then apply core_clip, skin_margin_deg and min_skin_thickness_mm.
  3. Fuse each member's skin pieces into a compound (grid imprints and
     slivers would otherwise force absurdly fine tets).
  4. FINAL CONFORMITY FRAGMENT over the MESHED volumes only (skins +
     tet-only members): re-imprints any contact face the fuses rebuilt,
     so every meshed contact is a single shared entity again. Interior
     prisms are never meshed, so no mesh conformity is needed against
     them -- only the exact volume tiling, which the fragments guarantee.
  5. One generate(3) with visibility masking; tets split per owner.

Emission (all ranks, from the broadcast payload): one CONVEX single-
chord prism per cell via rad.ObjPolyhdr. Two radia constraints
(measured 2026-07-21): ObjMltExtPgn elements cannot be packed by the GPU
assembly (CPU fallback), and non-convex polyhedra are rejected or
heap-crash radia -- hence exactly one chord per cell; dtheta_deg is the
arc-faceting knob (volume deficit ~ dtheta^2/6).

Keep rad.FldLenRndSw at its default 'on': structured grids put element
centers exactly on neighbor face-extension planes; the deterministic
AbsRandMagnitude repair is what makes that well-defined.
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "build_structured_group",
    "slice_stp_polar",
    "emit_prism_cells",
    "structured_defaults",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = _REPO_ROOT / "output" / "structured_cache"
_CACHE_VERSION = 2   # bump on any algorithm change that alters payloads


# ---------------------------------------------------------------------------
# Defaults for the ComponentSpec `structure:` dict
# ---------------------------------------------------------------------------
def structured_defaults() -> Dict[str, Any]:
    return {
        "type": "polar_grid",
        "dr_mm": 120.0,          # target radial fill spacing between anchors
        "dz_mm": 120.0,          # target axial fill spacing between anchors
        "dtheta_deg": 2.5,       # azimuthal cell size = arc faceting knob
                                 # (volume deficit ~ dtheta^2/6: 0.03% @ 2.5)
        "theta_span_deg": (0.0, 45.0),  # azimuthal extent of the folded part
        "snap": True,            # detect CAD radii / z-planes as grid anchors
        "element": "prism",      # 'prism' (ObjPolyhdr); 'recmag' reserved
        "min_fill_frac": 0.35,   # never place a fill edge closer than this
                                 # fraction of the target to an anchor
        # v2 rules ---------------------------------------------------------
        "core_clip": None,       # e.g. {"z_max": -140.0}: cells must lie
                                 # wholly inside these cylindrical bounds
                                 # (keys r_min/r_max/z_min/z_max/
                                 # theta_min_deg/theta_max_deg) to be core
        "skin_margin_deg": 0.0,  # demote core cells within this azimuth of
                                 # any non-core position in their ring
                                 # (shim envelope around spiral walls)
        "min_skin_thickness_mm": "auto",  # skin thinner than this absorbs
                                 # its interior neighbor ('auto' =
                                 # 0.5 * mesh_size_max; None/0 disables)
    }


_CLIP_KEYS = {"r_min", "r_max", "z_min", "z_max",
              "theta_min_deg", "theta_max_deg"}


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
    clip = out["core_clip"]
    if clip is not None:
        bad = set(clip) - _CLIP_KEYS
        if bad:
            raise ValueError(f"Unknown core_clip keys {sorted(bad)}; "
                             f"known: {sorted(_CLIP_KEYS)}")
    return out


# ---------------------------------------------------------------------------
# Geometry helpers (rank-0, inside an initialized gmsh session)
# ---------------------------------------------------------------------------
_COINCIDENCE_TOL = 0.05   # mm: snap/dedupe tolerance for detected features
_THETA_TOL_DEG = 0.01     # deg: snap tolerance for azimuthal cut planes
_VOL_RTOL = 1e-6          # relative tolerance for the cell volume test

Cell = Tuple[float, float, float, float, float, float]  # a,b,ta,tb,za,zb


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
        r = math.hypot(xyz[0], xyz[1])
        if r < 1e-9:
            return ("other", {})
        rhat = (xyz[0] / r, xyz[1] / r)
        radial = abs(n[0] / nn * rhat[0] + n[1] / nn * rhat[1])
        if radial < 1.0 - 1e-6:
            return ("other", {})           # off-axis (bolt-hole) cylinder
        return ("zcyl", {"r": r})
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
    """Grid edges: [lo, hi] + interior anchors + equal-spaced fill."""
    core = [a for a in anchors
            if lo + _COINCIDENCE_TOL < a < hi - _COINCIDENCE_TOL]
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


def _detect_anchors(gmsh, vol_tags: Sequence[int]) -> Tuple[List[float], List[float]]:
    radii: List[float] = []
    zplanes: List[float] = []
    for _d, s in gmsh.model.getBoundary([(3, t) for t in vol_tags],
                                        combined=False, oriented=False):
        kind, info = _surface_info(gmsh, s)
        if kind == "zcyl":
            radii.append(info["r"])
        elif kind == "zplane":
            zplanes.append(info["z"])
    return (_dedupe(radii, _COINCIDENCE_TOL),
            _dedupe(zplanes, _COINCIDENCE_TOL))


def _make_tools(gmsh, r_edges, z_edges, th_edges, z_lo, z_hi, r_hi):
    """SOLID cutting tools, each intersecting the iron with exactly one
    face (surface tools trip BOPAlgo self-intersections). The theta box's
    x'=0 face contains the axis at theta+90 deg, which for a <=180 deg
    part never re-enters the material."""
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
        gmsh.model.occ.rotate([(3, b)], 0, 0, 0, 0, 0, 1, math.radians(th))
        tools.append((3, b))
    return tools


def _classify_piece(gmsh, tag: int, r_edges, z_edges, th_edges) -> Optional[Cell]:
    """A piece is a clean core cell iff its face inventory is exactly two
    grid cylinders + two grid z-planes + two grid theta-planes and its
    volume matches the analytic sector volume."""
    faces = gmsh.model.getBoundary([(3, tag)], combined=False, oriented=False)
    cyl_r: List[float] = []
    pl_z: List[float] = []
    pl_th: List[float] = []
    for _fd, f in faces:
        kind, info = _surface_info(gmsh, f)
        if kind == "zcyl":
            r = _snap_to(r_edges, info["r"])
            if r is None:
                return None
            cyl_r.append(r)
        elif kind == "zplane":
            z = _snap_to(z_edges, info["z"])
            if z is None:
                return None
            pl_z.append(z)
        elif kind == "plane":
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
                return None
            pl_th.append(got)
        else:
            return None
    cyl_r = _dedupe(cyl_r, _COINCIDENCE_TOL)
    pl_z = _dedupe(pl_z, _COINCIDENCE_TOL)
    pl_th = _dedupe(pl_th, _THETA_TOL_DEG)
    if not (len(cyl_r) == 2 and len(pl_z) == 2 and len(pl_th) == 2):
        return None
    a, b = cyl_r
    za, zb = pl_z
    ta, tb = pl_th
    v_exp = 0.5 * math.radians(tb - ta) * (b * b - a * a) * (zb - za)
    v_act = gmsh.model.occ.getMass(3, tag)
    if abs(v_act - v_exp) > _VOL_RTOL * v_exp:
        return None
    return (a, b, ta, tb, za, zb)


def _apply_core_clip(interior: Dict[int, Cell], clip: Optional[Dict[str, float]]) -> int:
    """Demote cells not wholly inside the cylindrical clip bounds."""
    if not clip:
        return 0
    demote = []
    for tag, (a, b, ta, tb, za, zb) in interior.items():
        ok = True
        if "r_min" in clip and a < clip["r_min"] - _COINCIDENCE_TOL:
            ok = False
        if "r_max" in clip and b > clip["r_max"] + _COINCIDENCE_TOL:
            ok = False
        if "z_min" in clip and za < clip["z_min"] - _COINCIDENCE_TOL:
            ok = False
        if "z_max" in clip and zb > clip["z_max"] + _COINCIDENCE_TOL:
            ok = False
        if "theta_min_deg" in clip and ta < clip["theta_min_deg"] - _THETA_TOL_DEG:
            ok = False
        if "theta_max_deg" in clip and tb > clip["theta_max_deg"] + _THETA_TOL_DEG:
            ok = False
        if not ok:
            demote.append(tag)
    for tag in demote:
        del interior[tag]
    return len(demote)


def _apply_theta_margin(interior: Dict[int, Cell], margin_deg: float,
                        t0: float, dtheta: float, n_theta: int) -> int:
    """Per-ring 1D dilation along theta: demote core cells within the
    margin of any IN-SPAN azimuth that is not core (spiral wall, hole,
    clipped band). Positions beyond the span are the folded symmetry
    sides, not walls, and do not trigger demotion."""
    if margin_deg <= 0:
        return 0
    m = max(1, int(math.ceil(margin_deg / dtheta - 1e-9)))
    rings: Dict[Tuple, Dict[int, int]] = {}
    for tag, (a, b, ta, tb, za, zb) in interior.items():
        key = (round(a, 4), round(b, 4), round(za, 4), round(zb, 4))
        idx = int(round((ta - t0) / dtheta))
        rings.setdefault(key, {})[idx] = tag
    demote = []
    for key, occ in rings.items():
        present = set(occ)
        for idx, tag in occ.items():
            for j in range(idx - m, idx + m + 1):
                if 0 <= j < n_theta and j not in present:
                    demote.append(tag)
                    break
    for tag in demote:
        del interior[tag]
    return len(demote)


def _apply_thin_skin_demote(gmsh, interior: Dict[int, Cell],
                            seed_tags: Sequence[int],
                            thr: float) -> int:
    """Demote-and-merge: a thin skin piece (thickness estimate 2V/A below
    thr) absorbs its adjacent interior cell(s) -- 'remove the adjacent
    wall' so the tet mesher gets >= a cell of grading room.

    Only ORIGINALLY NON-CLEAN pieces (true CAD-shaved fragments) may seed
    demotion: cells demoted by the clip/margin rules are healthy full
    grid cells whose 2V/A is small merely because inner-radius cells are
    azimuthally narrow -- letting them seed would cascade through the
    whole core (measured)."""
    if thr <= 0:
        return 0
    demote: List[int] = []
    for s in seed_tags:
        v = gmsh.model.occ.getMass(3, s)
        faces = gmsh.model.getBoundary([(3, s)], combined=False,
                                       oriented=False)
        area = sum(gmsh.model.occ.getMass(2, f) for _fd, f in faces)
        if area <= 0 or 2.0 * v / area >= thr:
            continue
        for _fd, f in faces:
            up, _down = gmsh.model.getAdjacencies(2, f)
            for vol in up:
                if int(vol) in interior:
                    demote.append(int(vol))
    demote = sorted(set(demote))
    for tag in demote:
        del interior[tag]
    return len(demote)


def _show_only(gmsh, tags: Sequence[int]) -> None:
    """Visibility-mask EVERYTHING (all dims -- rolled-back fuses leave
    orphan faces that must never reach the mesher), then show the given
    volumes recursively (their faces/edges/points)."""
    gmsh.model.setVisibility(gmsh.model.getEntities(), 0)
    if tags:
        gmsh.model.setVisibility([(3, t) for t in tags], 1, True)
    gmsh.option.setNumber("Mesh.MeshOnlyVisible", 1)


def _try_mesh(gmsh, tags: Sequence[int], dim: int = 3) -> bool:
    """Try to mesh the given volumes in isolation (visibility-masked)."""
    _show_only(gmsh, tags)
    try:
        gmsh.model.mesh.generate(dim)
        return True
    except Exception:
        return False
    finally:
        gmsh.model.mesh.clear()


def _heal_volume(gmsh, tag: int) -> List[int]:
    """occ.healShapes one volume (degenerate/small-edge repair at
    tolerance level); returns the replacement tag(s). Removes the
    original if the kernel kept it around."""
    v_pre = gmsh.model.occ.getMass(3, tag)
    before = {t for _d, t in gmsh.model.occ.getEntities(3)}
    healed = gmsh.model.occ.healShapes(
        [(3, tag)], tolerance=1e-7, fixDegenerated=True,
        fixSmallEdges=True, fixSmallFaces=True, sewFaces=False,
        makeSolids=True)
    gmsh.model.occ.synchronize()
    after = {t for _d, t in gmsh.model.occ.getEntities(3)}
    new = sorted({t for d, t in healed if d == 3} & (after - before))
    if not new:
        new = sorted(after - before)
    if tag in after and new:
        gmsh.model.occ.remove([(3, tag)], recursive=False)
        gmsh.model.occ.synchronize()
    if not new:
        return [tag]        # heal was in-place (or a no-op)
    v_post = sum(gmsh.model.occ.getMass(3, t) for t in new)
    if abs(v_post - v_pre) > 1e-6 * max(v_pre, 1e-30):
        raise RuntimeError(
            f"healShapes changed the volume of piece {tag} by "
            f"{abs(v_post - v_pre) / max(v_pre, 1e-30):.2e} relative")
    return new


def _fuse_robust(gmsh, tags: Sequence[int]) -> Tuple[List[int], int]:
    """Fuse volumes into as few MESHABLE compounds as OCC manages.

    Two OCC failure modes are handled by recursive bisection:
    - the n-ary fuse itself fails ('Courbes non jointives' on heavily
      faceted spiral geometry), or
    - the fuse SUCCEEDS but produces a compound whose shell is broken
      ('The 1D mesh seems not to be forming a closed loop' -- observed
      when cross-member imprint edges from the group fragment are fused).
    Each candidate fusion is therefore performed on COPIES and validated
    with a cheap 2D mesh test; only a meshable result replaces the
    originals, otherwise the copies are discarded (rollback) and the set
    is bisected. Unfusable pieces stay separate (costs some local tets).
    Returns (result_tags, n_failed_joins).
    """
    failures = 0

    def try_fuse(ts: List[int]) -> Optional[List[int]]:
        copies = gmsh.model.occ.copy([(3, t) for t in ts])
        try:
            fused, _fmap = gmsh.model.occ.fuse([copies[0]], copies[1:])
        except Exception:
            try:
                gmsh.model.occ.remove(copies, recursive=True)
                gmsh.model.occ.synchronize()
            except Exception:
                pass
            return None
        gmsh.model.occ.synchronize()
        out = [t for d, t in fused if d == 3]
        if _try_mesh(gmsh, out, dim=2):
            # keep the fusion: drop the originals (faces they shared with
            # other volumes survive; their orphaned boundary is excluded
            # from meshing by _show_only's all-dims masking)
            gmsh.model.occ.remove([(3, t) for t in ts], recursive=False)
            gmsh.model.occ.synchronize()
            return out
        gmsh.model.occ.remove([(3, t) for t in out], recursive=True)
        gmsh.model.occ.synchronize()
        return None

    def rec(ts: List[int]) -> List[int]:
        nonlocal failures
        if len(ts) < 2:
            return list(ts)
        got = try_fuse(ts)
        if got is not None:
            return got
        if len(ts) == 2:
            failures += 1
            return list(ts)
        mid = len(ts) // 2
        return rec(ts[:mid]) + rec(ts[mid:])

    out = rec(list(tags))
    gmsh.model.occ.synchronize()
    return out, failures


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def _entry_digest(e: Dict[str, Any]) -> Optional[str]:
    """Digest of one entry's inputs; None if not cacheable (occ callable)."""
    if e.get("occ") is not None:
        return None
    path = e.get("stp_path")
    if not path:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    h.update(repr((e["name"], e.get("mesh_max"), e.get("mesh_min"),
                   sorted((e.get("structure") or {}).items(),
                          key=lambda kv: kv[0]))).encode())
    return h.hexdigest()


def _group_cache_key(entries: List[Dict[str, Any]]) -> Optional[str]:
    h = hashlib.sha256()
    h.update(str(_CACHE_VERSION).encode())
    for e in entries:
        d = _entry_digest(e)
        if d is None:
            return None
        h.update(d.encode())
    return h.hexdigest()[:16]


def _cache_load(key: Optional[str], group_name: str) -> Optional[Dict]:
    if key is None:
        return None
    path = _CACHE_DIR / f"{group_name}-{key}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            print(f"[structured {group_name}] cache hit: {path.name}",
                  flush=True)
            return payload
        except Exception:
            return None
    return None


def _cache_store(key: Optional[str], group_name: str, payload: Dict) -> None:
    if key is None:
        return
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _CACHE_DIR / f"{group_name}-{key}.pkl"
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[structured {group_name}] cached -> {path.name}", flush=True)
    except Exception as exc:      # cache is best-effort
        print(f"[structured {group_name}] cache write failed: {exc}",
              flush=True)


# ---------------------------------------------------------------------------
# The group builder (rank-0 gmsh work)
# ---------------------------------------------------------------------------
def _build_group_rank0(entries: List[Dict[str, Any]], group_name: str,
                       gmsh_verbosity: int) -> Dict[str, Any]:
    import gmsh

    from cyclotron_optimizer.geometry.components import (  # avoid cycle
        _pin_gmsh_determinism,
    )

    t_start = time.perf_counter()

    def _log(msg):
        print(f"[structured {group_name}] {msg} "
              f"(t={time.perf_counter() - t_start:.0f}s)", flush=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
        _pin_gmsh_determinism()
        gmsh.model.add(f"structured_{group_name}")

        # --- 1. import members ------------------------------------------
        owner_of: Dict[int, str] = {}
        for e in entries:
            before = {t for _d, t in gmsh.model.occ.getEntities(3)}
            if e.get("stp_path"):
                gmsh.model.occ.importShapes(str(e["stp_path"]))
            elif e.get("occ") is not None:
                e["occ"]()
            else:
                raise ValueError(f"entry {e['name']!r}: neither stp_path "
                                 "nor occ builder")
            after = {t for _d, t in gmsh.model.occ.getEntities(3)}
            new = sorted(after - before)
            if not new:
                raise ValueError(f"entry {e['name']!r} added no OCC volume")
            for t in new:
                owner_of[t] = e["name"]
        gmsh.model.occ.synchronize()
        v_in = {e["name"]: 0.0 for e in entries}
        for t, n in owner_of.items():
            v_in[n] += gmsh.model.occ.getMass(3, t)
        _log(f"imported {len(owner_of)} volumes from {len(entries)} members")

        def _refragment(tags: List[int], what: str) -> None:
            """Fragment the given volumes against each other, remapping
            owner_of and detecting cross-member overlaps."""
            nonlocal owner_of
            if len(tags) < 2:
                return
            _out, out_map = gmsh.model.occ.fragment(
                [(3, t) for t in sorted(tags)], [])
            new_owner = dict(owner_of)
            for t in tags:
                new_owner.pop(t, None)
            for in_tag, images in zip(sorted(tags), out_map):
                for d, t in images:
                    if d != 3:
                        continue
                    prev = new_owner.get(t)
                    if prev is not None and prev != owner_of[in_tag]:
                        raise RuntimeError(
                            f"structured group '{group_name}': members "
                            f"'{prev}' and '{owner_of[in_tag]}' OVERLAP "
                            f"({what}); parts must only touch")
                    new_owner[t] = owner_of[in_tag]
            owner_of = new_owner
            gmsh.model.occ.synchronize()

        # --- 2. group imprint fragment (mesh_group contract) ------------
        _refragment(sorted(owner_of), "group imprint")
        _log("group imprint fragment done")

        # --- 3. per-member slicing --------------------------------------
        members: Dict[str, Dict[str, Any]] = {}
        interior_by_member: Dict[str, Dict[int, Cell]] = {}
        for e in entries:
            name = e["name"]
            st = (_merge_structure(e.get("structure"))
                  if e.get("structure") is not None else None)
            if st is None:
                members[name] = {"structure": None}
                continue
            vols = sorted(t for t, n in owner_of.items() if n == name)

            radii, zplanes = (_detect_anchors(gmsh, vols) if st["snap"]
                              else ([], []))
            bbs = [gmsh.model.getBoundingBox(3, t) for t in vols]
            z_lo = min(bb[2] for bb in bbs)
            z_hi = max(bb[5] for bb in bbs)
            r_hi = max(math.hypot(max(abs(bb[0]), abs(bb[3])),
                                  max(abs(bb[1]), abs(bb[4])))
                       for bb in bbs) + 1.0
            r_min = radii[0] if radii else 0.0
            r_max = radii[-1] if radii else r_hi - 1.0
            r_edges = _build_edges(radii, r_min, r_max, float(st["dr_mm"]),
                                   float(st["min_fill_frac"]))
            z_edges = _build_edges(zplanes, z_lo, z_hi, float(st["dz_mm"]),
                                   float(st["min_fill_frac"]))
            t0, t1 = (float(st["theta_span_deg"][0]),
                      float(st["theta_span_deg"][1]))
            if t1 <= t0:
                raise ValueError("theta_span_deg must be increasing")
            n_theta = max(1, int(round((t1 - t0) / float(st["dtheta_deg"]))))
            dtheta = (t1 - t0) / n_theta
            th_edges = [t0 + dtheta * j for j in range(n_theta + 1)]

            tools = _make_tools(gmsh, r_edges, z_edges, th_edges,
                                z_lo, z_hi, r_hi)
            _log(f"{name}: fragmenting {len(vols)} volume(s) with "
                 f"{len(tools)} grid tools...")
            _out, out_map = gmsh.model.occ.fragment(
                [(3, t) for t in vols], tools)
            pieces: List[int] = []
            source_of: Dict[int, int] = {}   # piece -> member-solid index
            for i_src, images in enumerate(out_map[:len(vols)]):
                for d, t in images:
                    if d == 3 and t not in source_of:
                        source_of[t] = i_src
                        pieces.append(t)
            pieces = sorted(set(pieces))
            for t in vols:
                owner_of.pop(t, None)
            for t in pieces:
                owner_of[t] = name
            v_pieces = sum(gmsh.model.occ.getMass(3, t) for t in pieces)
            rel = abs(v_pieces - v_in[name]) / max(v_in[name], 1e-30)
            if rel > 1e-3:
                raise RuntimeError(
                    f"{name}: grid fragment changed the volume by "
                    f"{rel:.2e} relative -- OCC boolean failed")
            gmsh.model.occ.synchronize()
            _log(f"{name}: {len(pieces)} pieces; classifying...")

            interior: Dict[int, Cell] = {}
            for t in pieces:
                cell = _classify_piece(gmsh, t, r_edges, z_edges, th_edges)
                if cell is not None:
                    interior[t] = cell
            n_clean = len(interior)
            # non-clean fragments = true CAD-shaved pieces: the only
            # legitimate seeds for the thin-skin rule below
            cad_fragments = [t for t in pieces if t not in interior]

            n_clip = _apply_core_clip(interior, st["core_clip"])
            n_margin = _apply_theta_margin(interior,
                                           float(st["skin_margin_deg"]),
                                           t0, dtheta, n_theta)
            thr = st["min_skin_thickness_mm"]
            if thr == "auto":
                thr = 0.5 * float(e.get("mesh_max") or 0.0)
            thr = float(thr or 0.0)
            n_thin = _apply_thin_skin_demote(gmsh, interior, cad_fragments,
                                             thr)
            _log(f"{name}: {n_clean} clean cells; demoted {n_clip} (clip) "
                 f"+ {n_margin} (theta margin) + {n_thin} (thin skin) -> "
                 f"{len(interior)} core cells")

            cells = sorted(interior.values(),
                           key=lambda c: (c[4], c[0], c[2]))
            v_interior = sum(
                0.5 * math.radians(tb - ta) * (b * b - a * a) * (zb - za)
                for a, b, ta, tb, za, zb in cells)
            interior_by_member[name] = interior
            members[name] = {
                "structure": st,
                "cells": cells,
                "theta_span": (t0, t1),
                "_v_interior": v_interior,
                "_grid": (len(r_edges), len(z_edges), n_theta),
                "_anchors": (radii, zplanes),
                "_n_skin_pieces": len(pieces) - len(interior),
                "_source_of": source_of,
            }

        # --- 4. PRE-FUSE conformity + contact detection ------------------
        # Cross-member meshed contacts must be single shared face entities.
        # This runs on the RAW pieces (simple, boolean-robust geometry)
        # BEFORE any fuse: fragmenting FUSED compounds proved fragile
        # (broken 1D curve loops; healShapes mangles them). Only pieces
        # whose bounding boxes touch another member are fragmented.
        all_interior = {t for m in interior_by_member.values() for t in m}
        meshed = sorted(t for t in owner_of if t not in all_interior)
        owners_meshed = {owner_of[t] for t in meshed}
        contact_pieces: set = set()
        shared: Dict[tuple, int] = {}
        if len(owners_meshed) > 1:
            boxes = {t: gmsh.model.getBoundingBox(3, t) for t in meshed}
            tol = 0.1

            def _bb_touch(a, b):
                return not (a[3] < b[0] - tol or b[3] < a[0] - tol
                            or a[4] < b[1] - tol or b[4] < a[1] - tol
                            or a[5] < b[2] - tol or b[5] < a[2] - tol)

            by_owner: Dict[str, List[int]] = {}
            for t in meshed:
                by_owner.setdefault(owner_of[t], []).append(t)
            names = sorted(by_owner)
            cands: set = set()
            for i, na in enumerate(names):
                for nb in names[i + 1:]:
                    for ta_ in by_owner[na]:
                        for tb_ in by_owner[nb]:
                            if _bb_touch(boxes[ta_], boxes[tb_]):
                                cands.add(ta_)
                                cands.add(tb_)
            if cands:
                _log(f"conformity: fragmenting {len(cands)} cross-member "
                     "candidate pieces (pre-fuse)...")
                _refragment(sorted(cands), "pre-fuse conformity")
                meshed = sorted(t for t in owner_of
                                if t not in all_interior)
            # shared-face detection -> contact pieces stay OUT of the
            # fuses below, so the conforming entities survive verbatim
            meshed_set = set(meshed)
            for _d, s in gmsh.model.getEntities(2):
                up, _down = gmsh.model.getAdjacencies(2, s)
                owners = {owner_of.get(int(v)) for v in up
                          if int(v) in meshed_set}
                owners.discard(None)
                if len(owners) > 1:
                    shared[tuple(sorted(owners))] = \
                        shared.get(tuple(sorted(owners)), 0) + 1
                    for v in up:
                        if int(v) in meshed_set:
                            contact_pieces.add(int(v))
        msg = (", ".join(f"{a}~{b}: {n}"
                         for (a, b), n in sorted(shared.items()))
               if shared else "NONE")
        _log(f"conforming meshed interfaces: {msg} "
             f"({len(contact_pieces)} contact pieces kept unfused)")

        # --- 5. per-member skin fuse (contact pieces excluded) -----------
        # Fuse PER SOURCE SOLID (base pole / side shim / VP separately):
        # cross-solid fuses of heavily faceted contact geometry are what
        # trip OCC's sewing; _fuse_robust isolates remaining failures.
        for e in entries:
            name = e["name"]
            interior = interior_by_member.get(name, {})
            skin = sorted(t for t, n in owner_of.items()
                          if n == name and t not in interior
                          and t not in contact_pieces)
            if len(skin) > 1 and members[name].get("structure") is not None:
                source_of = members[name].get("_source_of", {})
                by_src: Dict[int, List[int]] = {}
                for t in skin:
                    by_src.setdefault(source_of.get(t, -1), []).append(t)
                v_pre = sum(gmsh.model.occ.getMass(3, t) for t in skin)
                new: List[int] = []
                n_fail = 0
                for _src in sorted(by_src):
                    fused, nf = _fuse_robust(gmsh, sorted(by_src[_src]))
                    new += fused
                    n_fail += nf
                for t in skin:
                    owner_of.pop(t, None)
                for t in new:
                    owner_of[t] = name
                v_post = sum(gmsh.model.occ.getMass(3, t) for t in new)
                srel = abs(v_post - v_pre) / max(v_pre, 1e-30)
                if srel > 1e-6:
                    _log(f"WARNING: {name} skin fuse shifted volume by "
                         f"{srel:.2e} relative")
                _log(f"{name}: fused {len(skin)} skin pieces -> {len(new)}"
                     + (f" ({n_fail} unfusable joins kept separate)"
                        if n_fail else ""))
        meshed = sorted(t for t in owner_of if t not in all_interior)

        # volume conservation per member (interior analytic + meshed OCC)
        for e in entries:
            name = e["name"]
            v_int = members[name].get("_v_interior", 0.0)
            v_meshed = sum(gmsh.model.occ.getMass(3, t)
                           for t in meshed if owner_of[t] == name)
            rel = abs(v_int + v_meshed - v_in[name]) / max(v_in[name], 1e-30)
            if rel > 1e-3:
                raise RuntimeError(
                    f"{name}: volume not conserved through slicing "
                    f"({rel:.2e} relative)")
            if rel > 1e-6:
                _log(f"WARNING: {name} volume shifted by {rel:.2e} relative")

        # --- 6. mesh the meshed volumes ---------------------------------
        sizes = {e["name"]: float(e["mesh_max"]) for e in entries
                 if e.get("mesh_max") is not None}
        if not sizes:
            raise ValueError("at least one entry needs mesh_max")
        mins = [float(e["mesh_min"]) for e in entries
                if e.get("mesh_min") is not None]
        gmsh.option.setNumber("Mesh.MeshSizeMax", max(sizes.values()))
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(mins) if mins else 1.0)
        for e in sorted(entries, key=lambda x: -float(x.get("mesh_max")
                                                      or 0.0)):
            if e.get("mesh_max") is None:
                continue
            vol_tags = [(3, t) for t in meshed if owner_of[t] == e["name"]]
            if not vol_tags:
                continue
            pts = gmsh.model.getBoundary(vol_tags, combined=False,
                                         oriented=False, recursive=True)
            pts = [(d, t) for d, t in pts if d == 0]
            if pts:
                gmsh.model.mesh.setSize(pts, float(e["mesh_max"]))

        if meshed:
            _log(f"meshing {len(meshed)} skin/tet volumes...")
            _show_only(gmsh, meshed)
            try:
                gmsh.model.mesh.generate(3)
            except Exception as exc:
                # Boolean debris (degenerate edges from fuse/fragment on
                # faceted geometry) can break the 1D/2D mesher. Attribute
                # the failure per volume, heal the offenders, retry.
                _log(f"joint mesh failed ({exc}); bisecting per volume "
                     "and healing offenders...")
                gmsh.model.mesh.clear()
                for t in list(meshed):
                    if _try_mesh(gmsh, [t]):
                        continue
                    name = owner_of[t]
                    _log(f"volume {t} ({name}) fails to mesh; healing...")
                    new = _heal_volume(gmsh, t)
                    if t not in new:
                        owner_of.pop(t, None)
                        for nt in new:
                            owner_of[nt] = name
                        meshed = sorted(t2 for t2 in owner_of
                                        if t2 not in all_interior)
                    still_bad = [nt for nt in new
                                 if not _try_mesh(gmsh, [nt])]
                    if still_bad:
                        raise RuntimeError(
                            f"skin volume(s) {still_bad} of '{name}' "
                            "cannot be meshed even after healShapes -- "
                            "inspect the CAD/grid interaction") from exc
                gmsh.model.mesh.clear()
                _show_only(gmsh, meshed)
                gmsh.model.mesh.generate(3)
                _log("mesh succeeded after healing")

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes: Dict[int, List[float]] = {}
        for i, tag in enumerate(node_tags):
            j = 3 * i
            nodes[int(tag)] = [float(node_coords[j]),
                               float(node_coords[j + 1]),
                               float(node_coords[j + 2])]
        tets_by_member: Dict[str, List] = {e["name"]: [] for e in entries}
        for t in meshed:
            etypes, _etags, enodes = gmsh.model.mesh.getElements(3, t)
            for et, conn in zip(etypes, enodes):
                if int(et) != 4:
                    continue
                for i in range(0, len(conn), 4):
                    tets_by_member[owner_of[t]].append(
                        [nodes[int(c)] for c in conn[i:i + 4]])

        # --- 7. per-member payloads + stats ------------------------------
        payload: Dict[str, Any] = {"members": {}}
        tot_cells = tot_tets = 0
        for e in entries:
            name = e["name"]
            m = members[name]
            st = m.get("structure")
            cells = m.get("cells", [])
            tets = tets_by_member[name]
            tot_cells += len(cells)
            tot_tets += len(tets)
            v_cad = v_in[name]
            v_int = m.get("_v_interior", 0.0)
            v_skin = v_cad - v_int
            if st is not None:
                t0, t1 = m["theta_span"]
                n_theta = m["_grid"][2]
                sub = math.radians(t1 - t0) / n_theta
                v_model = (math.sin(sub) / sub) * v_int + v_skin
                radii, zplanes = m["_anchors"]
                stats = {
                    "cad_volume_mm3": v_cad,
                    "interior_cells": len(cells),
                    "skin_pieces": m["_n_skin_pieces"],
                    "skin_tets": len(tets),
                    "skin_volume_frac": v_skin / max(v_cad, 1e-30),
                    "n_theta": n_theta,
                    "elements_total": len(cells) + len(tets),
                    "inscribed_volume_deficit_frac":
                        (v_cad - v_model) / max(v_cad, 1e-30),
                    "r_edges": m["_grid"][0],
                    "z_edges": m["_grid"][1],
                    "detected_radii": radii,
                    "detected_zplanes": zplanes,
                    "min_cell_dr": min((b - a for a, b, *_r in cells),
                                       default=float("nan")),
                    "min_cell_dz": min((z1 - z0 for *_r, z0, z1 in cells),
                                       default=float("nan")),
                }
                print(f"[structured {group_name}/{name}] "
                      f"{len(cells)} prism cells + {len(tets)} skin tets "
                      f"({100 * stats['skin_volume_frac']:.2f}% of volume "
                      f"is skin); chord deficit "
                      f"{100 * stats['inscribed_volume_deficit_frac']:.4f}%",
                      flush=True)
            else:
                stats = {"cad_volume_mm3": v_cad, "interior_cells": 0,
                         "skin_tets": len(tets),
                         "elements_total": len(tets)}
                print(f"[structured {group_name}/{name}] tet member: "
                      f"{len(tets)} tets", flush=True)
            payload["members"][name] = {
                "cells": cells,
                "skin_tets": tets,
                "theta_span": m.get("theta_span"),
                "structure": st,
                "stats": stats,
            }
        payload["group_stats"] = {
            "elements_total": tot_cells + tot_tets,
            "prism_cells": tot_cells,
            "tets": tot_tets,
            "meshed_interfaces": {f"{a}~{b}": n
                                  for (a, b), n in sorted(shared.items())},
            "wall_time_s": time.perf_counter() - t_start,
        }
        _log(f"DONE: {tot_cells} prisms + {tot_tets} tets = "
             f"{tot_cells + tot_tets} elements")
        return payload
    finally:
        gmsh.finalize()


def build_structured_group(
    entries: List[Dict[str, Any]],
    *,
    group_name: str = "group",
    comm: Any = None,
    gmsh_verbosity: int = 2,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Build a conforming group where members may be structured.

    Each entry: ``{"name", "stp_path" | "occ", "mesh_max", "mesh_min",
    "structure"}`` -- ``structure`` is the ComponentSpec dict (None for a
    plain tet member). Returns ``{"members": {name: payload},
    "group_stats": {...}}`` where each member payload has
    cells/skin_tets/structure/stats (emit with emit_prism_cells).

    MPI-safe: rank 0 does the gmsh/OCC work, the payload is broadcast.
    Results are cached under output/structured_cache when every entry is
    file-based (digest-keyed: re-slices only when CAD/params change).
    """
    from cyclotron_optimizer.geometry.components import _resolve_comm

    comm = _resolve_comm(comm)
    rank = comm.Get_rank() if comm is not None else 0

    payload: Optional[Dict[str, Any]] = None
    if rank <= 0:
        key = _group_cache_key(entries) if use_cache else None
        payload = _cache_load(key, group_name)
        if payload is None:
            payload = _build_group_rank0(entries, group_name, gmsh_verbosity)
            _cache_store(key, group_name, payload)
    if comm is not None:
        payload = comm.bcast(payload, root=0)
    assert payload is not None
    return payload


# ---------------------------------------------------------------------------
# Standalone wrapper (kept for tests / direct use; rank-local, no MPI)
# ---------------------------------------------------------------------------
def slice_stp_polar(
    stp_path: str,
    *,
    structure: Optional[Dict[str, Any]] = None,
    mesh_size_max: Optional[float] = None,
    mesh_size_min: Optional[float] = None,
    model_name: str = "structured",
    gmsh_verbosity: int = 2,
    use_cache: bool = False,
) -> Dict[str, Any]:
    """Slice ONE component (possibly multi-solid) on the calling rank.

    Returns the member payload: cells / skin_tets / theta_span /
    structure / stats.
    """
    entry = {"name": model_name, "stp_path": str(stp_path),
             "mesh_max": mesh_size_max, "mesh_min": mesh_size_min,
             "structure": dict(structure or {})}
    key = _group_cache_key([entry]) if use_cache else None
    payload = _cache_load(key, model_name)
    if payload is None:
        payload = _build_group_rank0([entry], model_name, gmsh_verbosity)
        _cache_store(key, model_name, payload)
    return payload["members"][model_name]


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
                 )                                              # top (shifted below)
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
