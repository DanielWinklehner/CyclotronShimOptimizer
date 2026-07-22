"""Structured polar-grid slicing of STP/OCC solids ("Option C", v3).

Discretizes revolved-ish iron parts into a STRUCTURED core of annular-
sector prisms plus a CONFORMING tetrahedral skin, instead of an
unstructured all-tet mesh. Structured cores condition the relaxation far
better at a fraction of the element count (RECMAG_GPU_PLAN.md,
validate_recmag.py) and put the 60 MeV machine back inside the GPU's
dense-IM element budget.

v3 -- "never cut the skin" (see the memory note structured-stp-slicing):
  v2 sliced a member by FRAGMENTING the whole solid with ~50-64 grid
  tools (~1700 pieces), classifying pieces by face inventory, then FUSING
  the ~55-85% skin pieces back together. On the 60 MeV machine that fuse
  ran ~2.2 HOURS (validate-and-rollback bisection around OCC failures on
  the faceted spiral pole). v3 inverts the pipeline so the two slow,
  failure-prone stages (grid fragment + skin fuse) cease to exist:

    1. Decide the CORE cells with NO boolean at all -- margin-dilated
       point sampling (``gmsh.model.isInside``): a cell is core iff every
       point of its DILATED box lies inside the member's solid(s). The
       dilation encodes v2's rules directly -- theta by ``skin_margin_deg``
       (spiral-wall envelope), r/z by ``min_skin_thickness_mm`` (guaranteed
       tet clearance from any true CAD surface) -- with a CLAMP that gives
       a face zero dilation when it sits on a detected CAD anchor (core may
       be flush with snapped faces) or on a ``theta_span_deg`` symmetry
       plane (folded sides are not walls). ``core_clip`` is applied to both
       the raw cell (candidate prefilter) and the dilated cell (so a
       cell whose margin reaches past a clip boundary is demoted, exactly
       as v2's per-ring theta margin dilated against clipped bands).
    2. Merge core cells into maximal index-space blocks (pure Python).
    3. Build each block as an annular-sector solid (occ cylinder wedge:
       true ARC radial faces, exact planar z/theta faces) and remove the
       core with ONE ``occ.cut`` per member. The remainder IS the skin --
       a few connected bodies, untouched by any grid surface, "fused" by
       construction. No skin is ever fragmented or fused.

  NOTE (deliberate refinement of the v3 spec, which proposed *chord*
  extruded-polygon cut blocks): a chord block inscribed in an arc-bounded
  solid leaves a ~0.5 mm arc-vs-chord crescent of solid as skin wrapping
  every anchored cylindrical wall -> thousands of micro-tets (the v2
  "grid imprints force absurdly fine tets" disease). Cutting with the true
  ARC removes the core flush to the CAD surface, so no sliver forms; the
  emitted prisms stay single-chord (the accepted ~dtheta^2/6 volume
  deficit), reproducing v2's proven core geometry exactly. Only cylinder
  radial faces need arcs; z/theta faces are already exact planes.

Pipeline (rank 0, one gmsh session):
  1. Import every member (STP or OCC callable); group-fragment all
     volumes (imprints contacts, detects overlaps) -- the mesh_group
     contract.
  2. Per structured member: detect z-cylinder radii / z-plane anchors,
     build the grid (anchors + fill, tracking which edges are anchors),
     classify cells by dilated isInside sampling, merge the core into
     maximal blocks, cut them out -> skin remainder.
  3. Cross-member conformity scan on the MESHED volumes (skins + tet
     members): the group imprint already made shared contact faces single
     entities; v3 verifies + reports them and warns loudly on any
     coincident-but-non-shared face rather than refragmenting big skin
     compounds (on this machine there are ZERO meshed-meshed contacts --
     everywhere members touch, at least one side is cut-away core).
  4. One generate(3) with visibility masking; tets split per owner
     (heal/bisect backstop for boolean debris).

Emission (all ranks, from the broadcast payload): one CONVEX single-chord
prism per cell via rad.ObjPolyhdr. Two radia constraints (measured
2026-07-21): ObjMltExtPgn elements cannot be packed by the GPU assembly
(CPU fallback), and non-convex polyhedra are rejected or heap-crash radia
-- hence exactly one chord per cell; dtheta_deg is the arc-faceting knob.

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
_CACHE_VERSION = 4   # bump on any algorithm change that alters payloads
                     # (v3: envelope-cut slicing; 4: multi-volume shared-face
                     # double-count fix in the classifier)

# Point-sampling classification tuning -----------------------------------
_SAMPLE_CAP = 9            # max sample points per dimension per cell
_SAMPLE_POINT_BUDGET = 250000   # nominal isInside calls/member before the
                                # lattice is uniformly coarsened (spec:
                                # margins, not density, carry the safety)
_MAX_CLASSIFY_WORKERS = 8       # cap on classification worker processes
_MIN_PARALLEL_CANDIDATES = 500  # below this, process-spawn overhead is not
                                # worth it -- classify serially (also keeps
                                # the small synthetic unit tests single-proc)


def _auto_workers() -> int:
    """Default classification worker count: this node's usable cores,
    capped. gmsh isInside (the slice bottleneck) is single-threaded, so the
    one-time classification fans out over spawned processes. Only rank 0
    builds, and the other MPI ranks idle-wait on the broadcast, so using
    this node's cores does not contend with live MPI work. Override via the
    STRUCTURED_CLASSIFY_WORKERS env var or the classify_workers argument."""
    env = os.environ.get("STRUCTURED_CLASSIFY_WORKERS")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, min((os.cpu_count() or 1) - 2, _MAX_CLASSIFY_WORKERS))


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
        # v2/v3 rules ------------------------------------------------------
        "core_clip": None,       # e.g. {"z_max": -140.0}: cells must lie
                                 # wholly inside these cylindrical bounds
                                 # (keys r_min/r_max/z_min/z_max/
                                 # theta_min_deg/theta_max_deg) to be core;
                                 # a cell whose DILATED box reaches past a
                                 # clip bound is demoted (margin-band rule)
        "skin_margin_deg": 0.0,  # azimuthal dilation: core cells within this
                                 # angle of any non-core azimuth (spiral
                                 # wall, hole, clip band) are demoted
        "min_skin_thickness_mm": "auto",  # radial/axial dilation: guarantee
                                 # >= this much tet room between core and any
                                 # true CAD surface ('auto' = 0.5*mesh_max;
                                 # None/0 disables)
        "sample_spacing_mm": 25.0,  # target isInside sample lattice spacing
                                 # (must stay well below the smallest feature
                                 # to respect -- bolt holes here are Dia50+)
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
    if float(out["sample_spacing_mm"]) <= 0.0:
        raise ValueError("sample_spacing_mm must be > 0")
    return out


# ---------------------------------------------------------------------------
# Geometry helpers (rank-0, inside an initialized gmsh session)
# ---------------------------------------------------------------------------
_COINCIDENCE_TOL = 0.05   # mm: snap/dedupe tolerance for detected features
_THETA_TOL_DEG = 0.01     # deg: snap tolerance for azimuthal cut planes

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
                 target: float, min_fill_frac: float
                 ) -> Tuple[List[float], List[bool]]:
    """Grid edges [lo, hi] + interior anchors + equal-spaced fill, plus a
    parallel bool list marking which edges coincide with a detected CAD
    ANCHOR (vs an interpolated fill). Anchor edges get zero dilation in the
    classifier -- core may sit flush against snapped CAD faces."""
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
    anchor_vals = list(anchors)
    is_anchor = [any(abs(e - a) <= _COINCIDENCE_TOL for a in anchor_vals)
                 for e in out]
    return out, is_anchor


def _detect_anchors(gmsh, vol_tags: Sequence[int]
                    ) -> Tuple[List[float], List[float]]:
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


# ---------------------------------------------------------------------------
# v3 classification: margin-dilated point sampling (no booleans)
# ---------------------------------------------------------------------------
def _cell_in_clip(clip: Optional[Dict[str, float]],
                  a: float, b: float, ta: float, tb: float,
                  za: float, zb: float) -> bool:
    """True iff the cell (a<=r<=b, ta<=theta<=tb deg, za<=z<=zb) lies
    wholly inside the cylindrical clip bounds. Same semantics as v2's
    per-cell clip; called on the raw cell (candidate) AND the dilated cell
    (so a margin reaching past a clip boundary demotes the cell)."""
    if not clip:
        return True
    if "r_min" in clip and a < clip["r_min"] - _COINCIDENCE_TOL:
        return False
    if "r_max" in clip and b > clip["r_max"] + _COINCIDENCE_TOL:
        return False
    if "z_min" in clip and za < clip["z_min"] - _COINCIDENCE_TOL:
        return False
    if "z_max" in clip and zb > clip["z_max"] + _COINCIDENCE_TOL:
        return False
    if "theta_min_deg" in clip and ta < clip["theta_min_deg"] - _THETA_TOL_DEG:
        return False
    if "theta_max_deg" in clip and tb > clip["theta_max_deg"] + _THETA_TOL_DEG:
        return False
    return True


def _dilated_bounds(i: int, j: int, k: int,
                    r_edges, r_anchor, z_edges, z_anchor, th_edges,
                    n_theta: int, margin_deg: float, skin_mm: float
                    ) -> Cell:
    """The cell's box grown for clearance: theta by margin_deg, r/z by
    skin_mm. Two parts of the clamping rule:
      (a) ZERO r/z growth on a face that sits on a detected CAD anchor
          (core may be flush with a snapped CAD surface);
      (b) theta growth is CLAMPED to the [t0, t1] span -- the folded
          symmetry planes are not walls, so a cell near a span edge is
          neither sampled nor clip-tested past the mirror plane (without
          this, an interior cell one step in from the edge dilates to
          theta<t0, samples empty space, and is wrongly demoted)."""
    a0, b0 = r_edges[i], r_edges[i + 1]
    ta0, tb0 = th_edges[j], th_edges[j + 1]
    za0, zb0 = z_edges[k], z_edges[k + 1]
    dr_lo = 0.0 if r_anchor[i] else skin_mm
    dr_hi = 0.0 if r_anchor[i + 1] else skin_mm
    dz_lo = 0.0 if z_anchor[k] else skin_mm
    dz_hi = 0.0 if z_anchor[k + 1] else skin_mm
    t0, t1 = th_edges[0], th_edges[n_theta]
    return (max(0.0, a0 - dr_lo), b0 + dr_hi,
            max(t0, ta0 - margin_deg), min(t1, tb0 + margin_deg),
            za0 - dz_lo, zb0 + dz_hi)


def _sample_counts(dil: Cell, spacing: float) -> Tuple[int, int, int]:
    a, b, ta, tb, za, zb = dil
    r_mid = 0.5 * (a + b)
    arc = r_mid * math.radians(tb - ta)
    nr = min(_SAMPLE_CAP, max(3, int(math.ceil((b - a) / spacing)) + 1))
    nz = min(_SAMPLE_CAP, max(3, int(math.ceil((zb - za) / spacing)) + 1))
    nth = min(_SAMPLE_CAP, max(3, int(math.ceil(arc / spacing)) + 1))
    return nr, nth, nz


def _lin(lo: float, hi: float, n: int, eps: float) -> List[float]:
    """n ascending points spanning [lo+eps, hi-eps]. eps insets off the
    exact boundary (isInside is ambiguous on a CAD surface)."""
    if n <= 1:
        return [0.5 * (lo + hi)]
    step = (hi - lo - 2.0 * eps) / (n - 1)
    return [lo + eps + step * m for m in range(n)]


def _lattice_points(dil: Cell, spacing: float
                    ) -> Tuple[List[float], List[float]]:
    """Flat cartesian coords for the dilated cell, split into a coarse
    3-per-dim subset (the cheap batch-1 reject) and the finer remainder
    (batch-2). Coarse + fine together are the full sample lattice."""
    a, b, ta, tb, za, zb = dil
    nr, nth, nz = _sample_counts(dil, spacing)
    rs = _lin(a, b, nr, 1e-3)
    zs = _lin(za, zb, nz, 1e-3)
    ths = _lin(math.radians(ta), math.radians(tb), nth, 1e-6)
    cs = [(math.cos(t), math.sin(t)) for t in ths]

    def coarse_idx(n):
        return {0, (n - 1) // 2, n - 1}

    cr, ci, ck = coarse_idx(nr), coarse_idx(nth), coarse_idx(nz)
    coarse: List[float] = []
    fine: List[float] = []
    for ir, rr in enumerate(rs):
        for it, (co, si) in enumerate(cs):
            x, y = rr * co, rr * si
            for iz, zz in enumerate(zs):
                dst = (coarse if (ir in cr and it in ci and iz in ck)
                       else fine)
                dst += (x, y, zz)
    return coarse, fine


def _batch_covered(gmsh, vols: Sequence[int],
                   coords: List[float]) -> Tuple[int, int]:
    """(sum of per-volume isInside counts, # isInside calls), short-
    circuiting once the running sum reaches the point count.

    For ONE volume this is the exact union count. For SEVERAL it is an
    UPPER bound on the union count: a point on a shared internal face is
    inside (counts for) both adjacent sub-volumes, so the sum can exceed
    the true number of covered points. It therefore only supports the
    reject direction (sum < n  =>  some point is genuinely outside);
    confirmation of a multi-volume cell needs _all_covered."""
    n = len(coords) // 3
    cov = calls = 0
    for v in vols:
        calls += 1
        cov += gmsh.model.isInside(3, v, coords)
        if cov >= n:
            break
    return cov, calls


def _all_covered(gmsh, vols: Sequence[int],
                 coords: List[float]) -> Tuple[bool, int]:
    """Exact per-point union test: True iff every point is inside at least
    one volume. Short-circuits on the first uncovered point. Used only for
    multi-volume members, where _batch_covered's sum double-counts points
    on shared internal faces and could mask a genuinely-outside point."""
    n = len(coords) // 3
    calls = 0
    for p in range(n):
        b = 3 * p
        pt = coords[b:b + 3]
        hit = False
        for v in vols:
            calls += 1
            if gmsh.model.isInside(3, v, pt):
                hit = True
                break
        if not hit:
            return False, calls
    return True, calls


def _cell_is_core(gmsh, vols: Sequence[int], dil: Cell,
                  spacing: float) -> Tuple[bool, int]:
    """Core iff EVERY lattice point of the dilated cell is inside the
    member's volume(s). A coarse 3^3 pass rejects most exterior cells
    before the finer remainder. For a single volume the batched sum is the
    exact union count; for several it is an upper bound (reject-only), so a
    passing multi-volume cell is confirmed exactly per point (no shared-
    face double-count). Returns (is_core, n_isInside_calls)."""
    if not vols:
        return False, 0
    single = len(vols) == 1
    coarse, fine = _lattice_points(dil, spacing)
    calls = 0
    for coords in (coarse, fine):
        if not coords:
            continue
        n = len(coords) // 3
        cov, c = _batch_covered(gmsh, vols, coords)
        calls += c
        if cov < n:
            return False, calls              # sum >= true union: safe reject
        if not single:
            ok, c2 = _all_covered(gmsh, vols, coords)
            calls += c2
            if not ok:
                return False, calls
    return True, calls


def _sector_bbox(a: float, b: float, ta_deg: float, tb_deg: float,
                 za: float, zb: float) -> Tuple[float, ...]:
    """Cartesian axis-aligned bbox (xlo,xhi,ylo,yhi,zlo,zhi) of an annular
    sector, for cheap per-cell volume pruning. Includes cardinal-angle
    extrema that fall inside [ta,tb]."""
    angs = [ta_deg, tb_deg]
    c = 0.0
    while c <= 360.0:
        if ta_deg < c < tb_deg:
            angs.append(c)
        c += 90.0
    xs: List[float] = []
    ys: List[float] = []
    for r in (a, b):
        for th in angs:
            t = math.radians(th)
            xs.append(r * math.cos(t))
            ys.append(r * math.sin(t))
    return (min(xs), max(xs), min(ys), max(ys), za, zb)


def _bbox_overlap(c: Tuple[float, ...], vb: Sequence[float],
                  tol: float = 0.5) -> bool:
    """Overlap of a sector bbox c=(xlo,xhi,ylo,yhi,zlo,zhi) with a gmsh
    bounding box vb=(xmin,ymin,zmin,xmax,ymax,zmax). Conservative (tol
    grows the box) -- over-inclusion only costs a wasted isInside test,
    under-inclusion would wrongly drop a volume."""
    return not (c[1] < vb[0] - tol or vb[3] < c[0] - tol
                or c[3] < vb[1] - tol or vb[4] < c[2] - tol
                or c[5] < vb[2] - tol or vb[5] < c[4] - tol)


def _classify_cells_serial(gmsh, vols, cand, sp):
    """The isInside classification loop against a live gmsh session."""
    vol_bb = {v: gmsh.model.getBoundingBox(3, v) for v in vols}
    core = set()
    total = 0
    for i, j, k, dil in cand:
        # prune volumes whose bbox cannot contain this cell (e.g. deep core
        # cells never touch the VP insert's shallow bbox) -- fewer isInside
        # calls, and a cell outside every volume is exterior
        cbb = _sector_bbox(*dil)
        rel = [v for v in vols if _bbox_overlap(cbb, vol_bb[v])]
        if not rel:
            continue
        ok, calls = _cell_is_core(gmsh, rel, dil, sp)
        total += calls
        if ok:
            core.add((i, j, k))
    return core, total


def _classify_worker(payload):
    """Top-level (picklable) process worker: import the member STP into a
    fresh gmsh session and classify an assigned shard of candidate cells.
    Independent of the parent's gmsh state -- only reads the STP for
    isInside occupancy (imprint-invariant); the authoritative grid is
    passed in, never recomputed, so shards stay index-consistent."""
    stp_path, grid, margin_deg, skin_mm, sp, shard = payload
    r_edges, r_anchor, z_edges, z_anchor, th_edges, n_theta = grid
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("classify_worker")
        gmsh.model.occ.importShapes(stp_path)
        gmsh.model.occ.synchronize()
        vols = [t for _d, t in gmsh.model.occ.getEntities(3)]
        vol_bb = {v: gmsh.model.getBoundingBox(3, v) for v in vols}
        out = []
        for (i, j, k) in shard:
            dil = _dilated_bounds(i, j, k, r_edges, r_anchor, z_edges,
                                  z_anchor, th_edges, n_theta,
                                  margin_deg, skin_mm)
            cbb = _sector_bbox(*dil)
            rel = [v for v in vols if _bbox_overlap(cbb, vol_bb[v])]
            if not rel:
                continue
            ok, _c = _cell_is_core(gmsh, rel, dil, sp)
            if ok:
                out.append((i, j, k))
        return out
    finally:
        gmsh.finalize()


def _classify_parallel(stp_path, grid, cand, sp, n_workers, margin_deg,
                       skin_mm, log):
    """Classify candidate cells across `n_workers` spawned processes, each
    with its own gmsh session + STP import. Returns the merged core set.
    Round-robin shards for load balance; a set-union merge is order-
    independent, so the result is identical to the serial classifier."""
    import multiprocessing as mp
    cells_idx = [(i, j, k) for (i, j, k, _dil) in cand]
    shards = [s for s in (cells_idx[w::n_workers] for w in range(n_workers))
              if s]
    args = [(stp_path, grid, margin_deg, skin_mm, sp, s) for s in shards]
    ctx = mp.get_context("spawn")
    core = set()
    with ctx.Pool(len(shards)) as pool:
        for sub in pool.map(_classify_worker, args):
            core.update(sub)
    return core


def _classify_core_cells(gmsh, vols, r_edges, r_anchor, z_edges, z_anchor,
                         th_edges, n_theta, clip, margin_deg, skin_mm,
                         spacing, log, stp_path=None, n_workers=1) -> set:
    """Return the set of (i,j,k) core cell indices for one member.

    Candidate enumeration + budget coarsening run on the calling process
    (cheap); the isInside classification is distributed across `n_workers`
    spawned processes when a reusable STP path is available (the one-time
    slice's dominant cost), else run serially against the live session."""
    nr, nz = len(r_edges) - 1, len(z_edges) - 1
    cand: List[Tuple[int, int, int, Cell]] = []
    for i in range(nr):
        a0, b0 = r_edges[i], r_edges[i + 1]
        for k in range(nz):
            za0, zb0 = z_edges[k], z_edges[k + 1]
            for j in range(n_theta):
                ta0, tb0 = th_edges[j], th_edges[j + 1]
                if not _cell_in_clip(clip, a0, b0, ta0, tb0, za0, zb0):
                    continue                       # raw prefilter
                dil = _dilated_bounds(i, j, k, r_edges, r_anchor,
                                      z_edges, z_anchor, th_edges,
                                      n_theta, margin_deg, skin_mm)
                if not _cell_in_clip(clip, *dil):
                    continue                       # margin reaches past clip
                cand.append((i, j, k, dil))
    if not cand:
        log("no candidate cells (all clipped)")
        return set()
    nominal = sum(_sample_counts(dil, spacing)[0]
                  * _sample_counts(dil, spacing)[1]
                  * _sample_counts(dil, spacing)[2] for *_x, dil in cand)
    sp = spacing
    if _SAMPLE_POINT_BUDGET and nominal > _SAMPLE_POINT_BUDGET:
        sp = spacing * (nominal / _SAMPLE_POINT_BUDGET) ** (1.0 / 3.0)
        coarse = sum(_sample_counts(dil, sp)[0] * _sample_counts(dil, sp)[1]
                     * _sample_counts(dil, sp)[2] for *_x, dil in cand)
        log(f"sampling {len(cand)} candidates: nominal {nominal} pts @ "
            f"{spacing:.0f}mm > budget {_SAMPLE_POINT_BUDGET}; coarsened to "
            f"{sp:.0f}mm ({coarse} pts, min 3^3/cell -- margins carry safety)")
    else:
        log(f"sampling {len(cand)} candidates: ~{nominal} pts @ {sp:.0f}mm")

    grid = (r_edges, r_anchor, z_edges, z_anchor, th_edges, n_theta)
    if (n_workers > 1 and stp_path
            and len(cand) >= max(2 * n_workers, _MIN_PARALLEL_CANDIDATES)):
        try:
            core = _classify_parallel(stp_path, grid, cand, sp, n_workers,
                                      margin_deg, skin_mm, log)
            log(f"classified {len(core)}/{len(cand)} candidates core "
                f"({n_workers} processes)")
            return core
        except Exception as exc:                   # never fail the build
            log(f"parallel classify failed ({exc!r}); serial fallback")
    core, total = _classify_cells_serial(gmsh, vols, cand, sp)
    log(f"classified {len(core)}/{len(cand)} candidates core "
        f"({total} isInside calls)")
    return core


def _merge_core_blocks(core: set) -> List[Tuple[int, int, int, int, int, int]]:
    """Greedy maximal-box decomposition of a set of (i,j,k) core cells in
    index space: grow a theta run, then extend across r, then across z.
    Deterministic (sorted seed order). Blocks are only the CUTTING TOOL --
    emission stays strictly per-cell. Returns (i0,i1,j0,j1,k0,k1) inclusive."""
    remaining = set(core)
    blocks: List[Tuple[int, int, int, int, int, int]] = []
    for (i, j, k) in sorted(core):
        if (i, j, k) not in remaining:
            continue
        j1 = j
        while (i, j1 + 1, k) in remaining:
            j1 += 1
        i1 = i
        while all((i1 + 1, jj, k) in remaining for jj in range(j, j1 + 1)):
            i1 += 1
        k1 = k
        while all((ii, jj, k1 + 1) in remaining
                  for ii in range(i, i1 + 1) for jj in range(j, j1 + 1)):
            k1 += 1
        for ii in range(i, i1 + 1):
            for jj in range(j, j1 + 1):
                for kk in range(k, k1 + 1):
                    remaining.discard((ii, jj, kk))
        blocks.append((i, i1, j, j1, k, k1))
    return blocks


def _arc_block(gmsh, r0: float, r1: float, t0_deg: float, t1_deg: float,
               z0: float, z1: float) -> List[int]:
    """One annular-sector solid (true ARC radial faces at r0/r1, exact
    planar z and theta faces) as an occ cylinder wedge minus its bore.
    Returns the resulting volume tag(s)."""
    occ = gmsh.model.occ
    sub = math.radians(t1_deg - t0_deg)
    dz = z1 - z0
    outer = occ.addCylinder(0.0, 0.0, z0, 0.0, 0.0, dz, r1, angle=sub)
    if r0 > _COINCIDENCE_TOL:
        pad = max(1.0, 0.01 * abs(dz))
        bore = occ.addCylinder(0.0, 0.0, z0 - pad, 0.0, 0.0, dz + 2.0 * pad, r0)
        out, _ = occ.cut([(3, outer)], [(3, bore)])
        tags = [t for d, t in out if d == 3]
    else:
        tags = [outer]
    for t in tags:
        occ.rotate([(3, t)], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   math.radians(t0_deg))
    return tags


# ---------------------------------------------------------------------------
# Mesh helpers (kept from v2)
# ---------------------------------------------------------------------------
def _show_only(gmsh, tags: Sequence[int]) -> None:
    """Visibility-mask EVERYTHING (all dims -- boolean removals leave
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
                       gmsh_verbosity: int,
                       classify_workers: int = 1) -> Dict[str, Any]:
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

        # --- 3. per-member envelope-cut slicing -------------------------
        members: Dict[str, Dict[str, Any]] = {}
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
            r_edges, r_anchor = _build_edges(radii, r_min, r_max,
                                             float(st["dr_mm"]),
                                             float(st["min_fill_frac"]))
            z_edges, z_anchor = _build_edges(zplanes, z_lo, z_hi,
                                             float(st["dz_mm"]),
                                             float(st["min_fill_frac"]))
            t0, t1 = (float(st["theta_span_deg"][0]),
                      float(st["theta_span_deg"][1]))
            if t1 <= t0:
                raise ValueError("theta_span_deg must be increasing")
            n_theta = max(1, int(round((t1 - t0) / float(st["dtheta_deg"]))))
            dtheta = (t1 - t0) / n_theta
            th_edges = [t0 + dtheta * j for j in range(n_theta + 1)]

            skin_mm = st["min_skin_thickness_mm"]
            if skin_mm == "auto":
                skin_mm = 0.5 * float(e.get("mesh_max") or 0.0)
            skin_mm = float(skin_mm or 0.0)
            margin = float(st["skin_margin_deg"])
            spacing = float(st["sample_spacing_mm"])

            def _mlog(msg, _n=name):
                _log(f"{_n}: {msg}")

            _mlog(f"grid {len(r_edges)}x{len(z_edges)} edges x {n_theta} theta"
                  f"; classifying (skin_mm={skin_mm:.1f}, margin={margin:.1f})")
            core = _classify_core_cells(
                gmsh, vols, r_edges, r_anchor, z_edges, z_anchor, th_edges,
                n_theta, st["core_clip"], margin, skin_mm, spacing, _mlog,
                stp_path=e.get("stp_path"), n_workers=classify_workers)

            cells = sorted(
                [(r_edges[i], r_edges[i + 1], th_edges[j], th_edges[j + 1],
                  z_edges[k], z_edges[k + 1]) for (i, j, k) in core],
                key=lambda c: (c[4], c[0], c[2]))
            v_int = sum(
                0.5 * math.radians(tb - ta) * (b * b - a * a) * (zb - za)
                for a, b, ta, tb, za, zb in cells)

            blocks = _merge_core_blocks(core)
            block_tags: List[int] = []
            for (i0, i1, j0, j1, k0, k1) in blocks:
                block_tags += _arc_block(
                    gmsh, r_edges[i0], r_edges[i1 + 1],
                    th_edges[j0], th_edges[j1 + 1],
                    z_edges[k0], z_edges[k1 + 1])
            gmsh.model.occ.synchronize()
            _mlog(f"{len(core)} core cells -> {len(blocks)} blocks "
                  f"({len(block_tags)} solids); cutting...")

            v_cad = v_in[name]
            if block_tags:
                v_blocks = sum(gmsh.model.occ.getMass(3, t)
                               for t in block_tags)
                rb = abs(v_blocks - v_int) / max(v_cad, 1e-30)
                if rb > 1e-3:
                    raise RuntimeError(
                        f"{name}: block solids volume differs from the "
                        f"analytic core by {rb:.2e} relative -- arc-block "
                        "build is wrong")
                out, _ = gmsh.model.occ.cut(
                    [(3, t) for t in vols], [(3, t) for t in block_tags])
                gmsh.model.occ.synchronize()
                skin_tags = sorted(t for d, t in out if d == 3)
            else:
                skin_tags = list(vols)

            for t in vols:
                owner_of.pop(t, None)
            for t in skin_tags:
                owner_of[t] = name
            v_skin = sum(gmsh.model.occ.getMass(3, t) for t in skin_tags)
            removed = v_cad - v_skin
            rr = abs(removed - v_int) / max(v_cad, 1e-30)
            if rr > 1e-3:
                raise RuntimeError(
                    f"{name}: cut removed {removed:.6e} mm^3 but the analytic "
                    f"core is {v_int:.6e} ({rr:.2e} relative) -- envelope cut "
                    "did not match the classified core")
            if rr > 1e-6:
                _mlog(f"WARNING: cut/core volume mismatch {rr:.2e} relative")
            _mlog(f"cut -> {len(skin_tags)} skin bodies; core "
                  f"{100 * v_int / max(v_cad, 1e-30):.1f}% / skin "
                  f"{100 * v_skin / max(v_cad, 1e-30):.1f}% of CAD volume")

            members[name] = {
                "structure": st,
                "cells": cells,
                "theta_span": (t0, t1),
                "_v_interior": v_int,
                "_v_skin": v_skin,
                "_grid": (len(r_edges), len(z_edges), n_theta),
                "_anchors": (radii, zplanes),
                "_n_skin_bodies": len(skin_tags),
            }

        # --- 4. cross-member conformity scan (report; never refragment) --
        # The group imprint (step 2) already made every cross-member contact
        # a single shared face entity; the per-member cut only removes deep
        # interior core, so those shared faces survive on the skin. v3 VERIFIES
        # and reports them, and warns loudly on any coincident-but-non-shared
        # face -- it does NOT refragment the big cut-remainder skin compounds.
        meshed = sorted(owner_of)
        owners_meshed = {owner_of[t] for t in meshed}
        shared: Dict[tuple, int] = {}
        n_nonconform = 0
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
            cand_pieces: set = set()
            for ia, na in enumerate(names):
                for nb in names[ia + 1:]:
                    for ta_ in by_owner[na]:
                        for tb_ in by_owner[nb]:
                            if _bb_touch(boxes[ta_], boxes[tb_]):
                                cand_pieces.add(ta_)
                                cand_pieces.add(tb_)
            # shared-face entities (adjacent to volumes of >1 owner)
            meshed_set = set(meshed)
            for _d, s in gmsh.model.getEntities(2):
                up, _down = gmsh.model.getAdjacencies(2, s)
                owners = {owner_of.get(int(vv)) for vv in up
                          if int(vv) in meshed_set}
                owners.discard(None)
                if len(owners) > 1:
                    key = tuple(sorted(owners))
                    shared[key] = shared.get(key, 0) + 1
            # coincident-but-non-shared faces among cross-member candidates:
            # different face entities of different owners at the same place
            if cand_pieces:
                buckets: Dict[tuple, set] = {}
                seen_faces: Dict[int, str] = {}
                for t in cand_pieces:
                    for _d, s in gmsh.model.getBoundary(
                            [(3, t)], combined=False, oriented=False):
                        seen_faces[int(s)] = owner_of[t]
                for s, owner in seen_faces.items():
                    com = gmsh.model.occ.getCenterOfMass(2, s)
                    area = gmsh.model.occ.getMass(2, s)
                    key = (round(com[0], 1), round(com[1], 1),
                           round(com[2], 1), round(area, 1))
                    buckets.setdefault(key, set()).add((int(s), owner))
                for key, group in buckets.items():
                    owners = {o for _s, o in group}
                    if len(owners) > 1 and len(group) > 1:
                        n_nonconform += 1
        msg = (", ".join(f"{a}~{b}: {n}" for (a, b), n in sorted(shared.items()))
               if shared else "NONE")
        _log(f"conforming meshed interfaces: {msg}")
        if n_nonconform:
            _log(f"WARNING: {n_nonconform} coincident-but-NON-shared face "
                 "group(s) between members -- possible non-conforming meshed "
                 "contact; inspect the CAD/grid interaction (NOT refragmenting)")

        # --- 5. final per-member volume conservation --------------------
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

        # --- 6. mesh the meshed (skin + tet-member) volumes -------------
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
                # Boolean debris (degenerate edges from the cut on faceted
                # geometry) can break the 1D/2D mesher. Attribute the failure
                # per volume, heal the offenders, retry.
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
                        meshed = sorted(owner_of)
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
            # analytic skin (v2 semantics): the envelope cut removes the
            # core to the true arc, so CAD - analytic-core is the skin
            # volume exactly, free of the OCC/analytic round-trip noise that
            # would otherwise leak into the chord-deficit stat.
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
                    "skin_pieces": m["_n_skin_bodies"],  # v3: skin bodies
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
    classify_workers: Optional[int] = None,
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

    ``classify_workers`` sets the number of processes the (rank-0) isInside
    classification fans out over (None -> auto from this node's cores; 1 ->
    serial). Only affects a cache MISS (a fresh slice); identical payload.
    """
    from cyclotron_optimizer.geometry.components import _resolve_comm

    comm = _resolve_comm(comm)
    rank = comm.Get_rank() if comm is not None else 0
    workers = _auto_workers() if classify_workers is None else classify_workers

    payload: Optional[Dict[str, Any]] = None
    if rank <= 0:
        key = _group_cache_key(entries) if use_cache else None
        payload = _cache_load(key, group_name)
        if payload is None:
            payload = _build_group_rank0(entries, group_name, gmsh_verbosity,
                                         classify_workers=workers)
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
    classify_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Slice ONE component (possibly multi-solid) on the calling rank.

    Returns the member payload: cells / skin_tets / theta_span /
    structure / stats. ``classify_workers``: processes for the isInside
    classification (None -> auto; 1 -> serial).
    """
    entry = {"name": model_name, "stp_path": str(stp_path),
             "mesh_max": mesh_size_max, "mesh_min": mesh_size_min,
             "structure": dict(structure or {})}
    workers = _auto_workers() if classify_workers is None else classify_workers
    key = _group_cache_key([entry]) if use_cache else None
    payload = _cache_load(key, model_name)
    if payload is None:
        payload = _build_group_rank0([entry], model_name, gmsh_verbosity,
                                     classify_workers=workers)
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
