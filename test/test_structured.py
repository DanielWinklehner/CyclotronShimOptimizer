"""Structured polar-grid slicing (geometry/structured.py, "Option C").

Synthetic revolved solid (90-deg annular wedge with a z-step) written to a
temporary STP, then sliced:
  - classification: every cell interior (zero skin) on a perfectly
    revolved solid, exact expected cell count
  - volume conservation: sum of analytic cell volumes == CAD volume
  - emitted prisms: inscribed-chord deficit matches sin(x)/x theory
  - physics: structured build vs an all-tet twin of the same STP agree
    on B after a small CPU relax (catches face/orientation errors)
  - harvest verification: the mesh we actually GOT is checked against what
    was asked for, and nothing unverified reaches the digest cache
"""

import _testenv  # noqa: F401

import math
import os
import tempfile

import numpy as np
import radia as rad

from cyclotron_optimizer.geometry import structured as _st
from cyclotron_optimizer.geometry.components import MagnetizedComponent
from cyclotron_optimizer.geometry.structured import (
    _prism_polyhedron, _remerge_contact_faces, _tet_volume, _verify_harvest,
    emit_prism_cells, slice_stp_polar)

# Geometry: 90-deg wedge, outer ring r 50..150 z 0..40 + stacked ring
# r 50..100 z 40..60. Anchors: r {50,100,150}, z {0,40,60}.
R0, R1, R2 = 50.0, 100.0, 150.0
Z0, Z1, Z2 = 0.0, 40.0, 60.0
# 90-deg sector: V = (dtheta/2) * (b^2 - a^2) * h with dtheta = pi/2
V_CAD = (math.pi / 4) * ((R2**2 - R0**2) * (Z1 - Z0)
                         + (R1**2 - R0**2) * (Z2 - Z1))

_STP_PATH = None


def _make_stp():
    global _STP_PATH
    if _STP_PATH is not None and os.path.exists(_STP_PATH):
        return _STP_PATH
    import gmsh
    fd, path = tempfile.mkstemp(suffix=".stp", prefix="test_structured_")
    os.close(fd)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("wedge_step")
        occ = gmsh.model.occ
        lower = occ.addCylinder(0, 0, Z0, 0, 0, Z1 - Z0, R2,
                                angle=math.pi / 2)
        upper = occ.addCylinder(0, 0, Z1, 0, 0, Z2 - Z1, R1,
                                angle=math.pi / 2)
        bore = occ.addCylinder(0, 0, Z0 - 1, 0, 0, (Z2 - Z0) + 2, R0)
        out, _ = occ.cut([(3, lower), (3, upper)], [(3, bore)])
        occ.fuse([out[0]], out[1:])
        occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()
    _STP_PATH = path
    return path


# dtheta 7.5 deg keeps the inscribed-chord deficit at 0.29% so the
# field-parity tolerance below is meaningful (22.5 deg would be 2.6%).
STRUCT = {"dr_mm": 30.0, "dz_mm": 20.0, "dtheta_deg": 7.5,
          "theta_span_deg": (0.0, 90.0)}
N_THETA = 12


def _payload():
    return slice_stp_polar(_make_stp(), structure=dict(STRUCT),
                           mesh_size_max=25.0, model_name="test_wedge",
                           gmsh_verbosity=0)


def test_classification_all_interior():
    p = _payload()
    s = p["stats"]
    # grid: r [50,75,100,125,150], z [0,20,40,60]
    # layers z 0..40: 4 radial bins; layer 40..60: 2 bins (r<=100)
    assert s["interior_cells"] == (4 * 2 + 2) * N_THETA, s
    assert s["skin_pieces"] == 0, s
    assert s["skin_tets"] == 0, s
    assert s["skin_volume_frac"] < 1e-9, s


def test_volume_conservation():
    p = _payload()
    v_cells = sum(0.5 * math.radians(tb - ta) * (b * b - a * a) * (z1 - z0)
                  for a, b, ta, tb, z0, z1 in p["cells"])
    # 1e-7: OCC's STEP round-trip carries ~3e-9 relative volume noise
    assert abs(v_cells - V_CAD) / V_CAD < 1e-7, (v_cells, V_CAD)
    assert abs(p["stats"]["cad_volume_mm3"] - V_CAD) / V_CAD < 1e-7


def test_inscribed_deficit_matches_theory():
    p = _payload()
    sub = math.radians(STRUCT["dtheta_deg"])
    expect = 1.0 - math.sin(sub) / sub
    got = p["stats"]["inscribed_volume_deficit_frac"]
    # 1e-9: the stat mixes the analytic cell volumes with OCC's CAD
    # volume, which carries ~1e-9 relative STEP round-trip noise
    assert abs(got - expect) < 1e-9, (got, expect)


def test_prism_volume_via_radia_field_parity():
    """Structured build vs all-tet twin of the same STP: relax a remanent
    linear material (CPU, deterministic) and compare B outside."""
    path = _make_stp()
    pts = [[170.0, 60.0, 20.0], [40.0, 30.0, 80.0], [-60.0, 90.0, -30.0]]

    def solve(structured):
        rad.UtiDelAll()
        mat_id = rad.MatLin([0.1, 0.1], [0.3, 0.5, 0.8])
        if structured:
            comp = MagnetizedComponent.from_stp_structured(
                path, structure=dict(STRUCT), mesh_size_max=25.0,
                model_name="t_struct")
        else:
            comp = MagnetizedComponent.from_stp(
                path, mesh_size_max=12.0, model_name="t_tet")
        rad.MatApl(comp.id, mat_id)
        im = rad.RlxPre(comp.id, use_gpu=False)
        rad.RlxAuto(im, 1e-7, 3000, 4)
        return np.array(rad.Fld(comp.id, 'b', pts, use_gpu=False))

    b_s = solve(True)
    b_t = solve(False)
    scale = np.abs(b_t).max()
    dev = np.abs(b_s - b_t).max() / scale
    # different discretizations of the same solid: agreement at the
    # discretization-error level; wrong prism faces/orientation -> O(1)
    assert dev < 2e-2, (dev, b_s.tolist(), b_t.tolist())


def test_prism_polyhedron_field_against_tet_split():
    """Single prism (fixed M) vs the same shape finely tet-meshed."""
    import gmsh
    rad.UtiDelAll()
    r0, r1, z0, z1 = 300.0, 420.0, -100.0, -20.0
    th = [math.radians(10.0), math.radians(15.0)]
    M = [0.3, -0.2, 0.9]
    pid = _prism_polyhedron(r0, r1, th, z0, z1, magn=M)

    poly = [(r0, th[0]), (r0, th[1]), (r1, th[1]), (r1, th[0])]
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("prism_twin")
        occ = gmsh.model.occ
        tags = [occ.addPoint(r * math.cos(t), r * math.sin(t), z0)
                for r, t in poly]
        lines = [occ.addLine(tags[i], tags[(i + 1) % 4]) for i in range(4)]
        surf = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        occ.extrude([(2, surf)], 0, 0, z1 - z0)
        occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", 20.0)
        gmsh.model.mesh.generate(3)
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes = {int(t): node_coords[3 * i:3 * i + 3].tolist()
                 for i, t in enumerate(node_tags)}
        tet_ids = []
        etypes, _e, enodes = gmsh.model.mesh.getElements(3)
        for et, conn in zip(etypes, enodes):
            if int(et) != 4:
                continue
            for i in range(0, len(conn), 4):
                tet_ids.append(rad.ObjPolyhdr(
                    [nodes[int(c)] for c in conn[i:i + 4]],
                    [[1, 2, 3], [1, 4, 2], [2, 4, 3], [3, 4, 1]], M))
    finally:
        gmsh.finalize()
    twin = rad.ObjCnt(tet_ids)

    pts = [[500.0, 130.0, -60.0], [250.0, 40.0, 30.0], [380.0, 90.0, -170.0]]
    b1 = np.array(rad.Fld(pid, 'b', pts, use_gpu=False))
    b2 = np.array(rad.Fld(twin, 'b', pts, use_gpu=False))
    dev = np.abs(b1 - b2).max() / np.abs(b2).max()
    assert dev < 1e-9, dev


def test_emit_matches_cell_list():
    p = _payload()
    rad.UtiDelAll()
    ids, cells = emit_prism_cells(p)
    assert len(ids) == len(cells) == p["stats"]["interior_cells"]


# ---------------------------------------------------------------------------
# v2 rules
# ---------------------------------------------------------------------------
def test_core_clip_z():
    """core_clip z_max demotes the upper layer into the tet skin."""
    st = dict(STRUCT, core_clip={"z_max": Z1})
    p = slice_stp_polar(_make_stp(), structure=st, mesh_size_max=25.0,
                        model_name="test_clip", gmsh_verbosity=0)
    # layers z 0..40 survive (4 radial bins x 2 layers); z 40..60 demoted
    assert p["stats"]["interior_cells"] == 4 * 2 * N_THETA, p["stats"]
    assert p["stats"]["skin_tets"] > 0
    assert all(zb <= Z1 + 1e-9 for *_r, _z0, zb in p["cells"])


def test_core_clip_z_from_top():
    """Face-relative z clip: z_from_top offsets inward from the part's own
    top (z_hi = Z2), so z_from_top = Z2 - Z1 is equivalent to z_max = Z1."""
    st = dict(STRUCT, core_clip={"z_from_top": Z2 - Z1})
    p = slice_stp_polar(_make_stp(), structure=st, mesh_size_max=25.0,
                        model_name="test_clip_relz", gmsh_verbosity=0)
    assert p["stats"]["interior_cells"] == 4 * 2 * N_THETA, p["stats"]
    assert all(zb <= Z1 + 1e-9 for *_r, _z0, zb in p["cells"])


def test_core_clip_theta_from_max():
    """theta_from_max_deg offsets inward from the span's high edge (t1=90),
    so theta_from_max_deg=45 == theta_max_deg=45 (half the wedge stays
    core). Confirms the angular reference is a span edge, not a fixed 0."""
    st = dict(STRUCT, core_clip={"theta_from_max_deg": 45.0})
    p = slice_stp_polar(_make_stp(), structure=st, mesh_size_max=25.0,
                        model_name="test_clip_relth", gmsh_verbosity=0)
    n_rings = 4 * 2 + 2
    assert p["stats"]["interior_cells"] == n_rings * (N_THETA // 2), \
        p["stats"]
    assert all(tb <= 45.0 + 1e-9 for _a, _b, _ta, tb, *_z in p["cells"])


def test_core_clip_abs_rel_conflict():
    """A bound given both absolutely and face-relative is rejected."""
    st = dict(STRUCT, core_clip={"z_max": Z1, "z_from_top": 10.0})
    try:
        slice_stp_polar(_make_stp(), structure=st, mesh_size_max=25.0,
                        model_name="test_clip_conflict", gmsh_verbosity=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for abs+rel clip conflict")


def test_theta_margin_span_edges_are_not_walls():
    """The folded-symmetry span edges must NOT trigger margin demotion:
    full rings stay fully core even with a large margin."""
    st = dict(STRUCT, skin_margin_deg=15.0)
    p = slice_stp_polar(_make_stp(), structure=st, mesh_size_max=25.0,
                        model_name="test_margin_edge", gmsh_verbosity=0)
    assert p["stats"]["interior_cells"] == (4 * 2 + 2) * N_THETA, p["stats"]


def test_theta_margin_dilates_clipped_band():
    """An in-span non-core region (here made via core_clip theta_max)
    dilates by the margin: one extra theta cell per ring is demoted."""
    st = dict(STRUCT, core_clip={"theta_max_deg": 45.0})
    p0 = slice_stp_polar(_make_stp(), structure=st, mesh_size_max=25.0,
                         model_name="test_margin0", gmsh_verbosity=0)
    st = dict(st, skin_margin_deg=STRUCT["dtheta_deg"])
    p1 = slice_stp_polar(_make_stp(), structure=st, mesh_size_max=25.0,
                         model_name="test_margin1", gmsh_verbosity=0)
    n_rings = 4 * 2 + 2
    assert p0["stats"]["interior_cells"] == n_rings * (N_THETA // 2), \
        p0["stats"]
    assert p1["stats"]["interior_cells"] == n_rings * (N_THETA // 2 - 1), \
        p1["stats"]


def test_structured_group_two_members():
    """Structured wedge + tet-only ring stacked on top: both members come
    back with elements and the meshed contact is a conforming interface."""
    import gmsh
    from cyclotron_optimizer.geometry.structured import build_structured_group

    fd, ring_path = tempfile.mkstemp(suffix=".stp", prefix="test_ring_")
    os.close(fd)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("ring_on_top")
        occ = gmsh.model.occ
        outer = occ.addCylinder(0, 0, Z2, 0, 0, 20.0, R1, angle=math.pi / 2)
        bore = occ.addCylinder(0, 0, Z2 - 1, 0, 0, 22.0, R0)
        occ.cut([(3, outer)], [(3, bore)])
        occ.synchronize()
        gmsh.write(ring_path)
    finally:
        gmsh.finalize()

    # clip the wedge's top layer into skin so the contact at z=Z2 is
    # skin-tet vs tet-member (a MESHED interface that must conform)
    entries = [
        {"name": "wedge", "stp_path": _make_stp(), "mesh_max": 25.0,
         "mesh_min": None, "structure": dict(STRUCT,
                                             core_clip={"z_max": Z1})},
        {"name": "ring", "stp_path": ring_path, "mesh_max": 25.0,
         "mesh_min": None, "structure": None},
    ]
    group = build_structured_group(entries, group_name="test_group",
                                   use_cache=False, gmsh_verbosity=0)
    w = group["members"]["wedge"]
    r = group["members"]["ring"]
    assert w["stats"]["interior_cells"] == 4 * 2 * N_THETA
    assert len(w["skin_tets"]) > 0
    assert w["structure"] is not None and r["structure"] is None
    assert len(r["skin_tets"]) > 0 and not r["cells"]
    iface = group["group_stats"]["meshed_interfaces"]
    assert iface.get("ring~wedge", 0) >= 1, iface
    os.unlink(ring_path)


# ---------------------------------------------------------------------------
# Contact repair
#
# occ.cut rebuilds the cut member's skin and can re-create its side of a
# cross-member contact as a NEW entity, leaving two coincident face copies
# instead of one shared one. gmsh then meshes the two copies independently
# and rejects the overlap. On the 60 MeV tet-yoke + structured-pole group the
# pole's cut split 3 of its 8 yoke contacts and the group would not mesh.
# ---------------------------------------------------------------------------
def _shared_faces(gmsh):
    """Faces adjacent to more than one volume (i.e. conforming contacts)."""
    faces = {}
    for _d, t in gmsh.model.getEntities(3):
        for _dd, s in gmsh.model.getBoundary([(3, t)], combined=False,
                                             oriented=False):
            faces.setdefault(int(s), set()).add(t)
    return sorted(s for s, v in faces.items() if len(v) > 1)


def _two_touching_boxes(fragment):
    """Two stacked boxes sharing the z=10 plane. Fragmented -> one shared
    face (conforming); un-fragmented -> two coincident copies (split)."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("contact")
    occ = gmsh.model.occ
    a = occ.addBox(0, 0, 0, 10, 10, 10)
    b = occ.addBox(0, 0, 10, 10, 10, 10)
    if fragment:
        occ.fragment([(3, a)], [(3, b)])
    occ.synchronize()
    return gmsh


def test_remerge_fuses_a_split_contact():
    """Two coincident copies of a contact face become one shared face, with
    volume tags and masses untouched."""
    gmsh = _two_touching_boxes(fragment=False)
    try:
        assert _shared_faces(gmsh) == [], "boxes should start non-conforming"
        vols_before = sorted(t for _d, t in gmsh.model.occ.getEntities(3))
        mass_before = [gmsh.model.occ.getMass(3, t) for t in vols_before]

        msgs = []
        removed = _remerge_contact_faces(gmsh, msgs.append)

        assert removed > 0, removed
        assert len(_shared_faces(gmsh)) == 1, _shared_faces(gmsh)
        assert len(msgs) == 1 and "re-merged" in msgs[0], msgs
        vols_after = sorted(t for _d, t in gmsh.model.occ.getEntities(3))
        assert vols_after == vols_before, (vols_before, vols_after)
        for t, m in zip(vols_before, mass_before):
            got = gmsh.model.occ.getMass(3, t)
            assert abs(got - m) <= 1e-9 * m, (t, m, got)
    finally:
        gmsh.finalize()


def test_remerge_is_a_noop_when_already_conforming():
    """Nothing to repair -> nothing removed, nothing logged. This is the
    case for every group that meshed before the repair existed."""
    gmsh = _two_touching_boxes(fragment=True)
    try:
        assert len(_shared_faces(gmsh)) == 1, _shared_faces(gmsh)
        n_before = len(gmsh.model.occ.getEntities())
        msgs = []
        assert _remerge_contact_faces(gmsh, msgs.append) == 0
        assert not msgs, msgs
        assert len(gmsh.model.occ.getEntities()) == n_before
        assert len(_shared_faces(gmsh)) == 1, _shared_faces(gmsh)
    finally:
        gmsh.finalize()


# ---------------------------------------------------------------------------
# Harvest verification
#
# Everything upstream compares ANALYTIC quantities against OCC masses; these
# cover the one check that looks at what the mesher actually produced. The
# 60 MeV (tet yoke + structured pole) corner meshed ZERO tets after the heal
# retry -- the whole tet yoke and the structured pole's entire 44% skin --
# and was reported as a success and cached.
# ---------------------------------------------------------------------------
def test_tet_volume():
    assert abs(_tet_volume([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                            [0, 0, 1]]) - 1.0 / 6.0) < 1e-15
    # sign-independent (gmsh connectivity orientation must not matter)
    assert abs(_tet_volume([[0, 0, 0], [0, 1, 0], [1, 0, 0],
                            [0, 0, 1]]) - 1.0 / 6.0) < 1e-15


def _sink():
    msgs = []
    return msgs, msgs.append


def test_verify_harvest_rejects_empty_volume():
    """A meshed volume that came back with no tets is a hard error: this is
    exactly the 'mesh succeeded after healing' -> '0 tets' silent failure."""
    _msgs, log = _sink()
    try:
        _verify_harvest("iron", [(7, "yoke", 1.2e9)],
                        {"yoke": (0.0, 1.2e9, 1.2e9)}, log)
    except RuntimeError as exc:
        assert "ZERO" in str(exc) and "yoke" in str(exc), exc
        return
    raise AssertionError("expected RuntimeError for a tet-less volume")


def test_verify_harvest_rejects_missing_skin():
    """A member that keeps its prism cells but loses its whole tet skin
    raises too -- it would NOT fail downstream (only a pure tet member with
    nothing left does), so this is the check that closes the hole."""
    _msgs, log = _sink()
    try:
        # pole: 44% of a 1e9 mm^3 member is skin, none of it meshed
        _verify_harvest("iron", [], {"pole": (0.0, 4.4e8, 1.0e9)}, log)
    except RuntimeError as exc:
        assert "pole" in str(exc) and "missing" in str(exc), exc
        return
    raise AssertionError("expected RuntimeError for an unmeshed skin")


def test_verify_harvest_accepts_faceting_error():
    """Linear tets inscribe/circumscribe curved CAD walls, so the harvested
    skin never matches analytic exactly. The 60 MeV yoke measures
    0.9949..1.0145 across the ladder runs -- that must pass silently."""
    for ratio in (0.9949, 1.0145):
        msgs, log = _sink()
        got = _verify_harvest("iron", [],
                              {"yoke": (ratio * 1.0e8, 1.0e8, 1.0e9)}, log)
        assert abs(got["yoke"] - ratio) < 1e-12, got
        assert not msgs, msgs


def test_verify_harvest_warns_before_it_raises():
    """Between the warn and fail thresholds: loud log line, no exception."""
    msgs, log = _sink()
    mid = 0.5 * (_st._SKIN_VOLUME_WARN_FRAC + _st._SKIN_VOLUME_FAIL_FRAC)
    got = _verify_harvest("iron", [],
                          {"yoke": ((1.0 + mid) * 1.0e8, 1.0e8, 1.0e9)}, log)
    assert abs(got["yoke"] - (1.0 + mid)) < 1e-12, got
    assert len(msgs) == 1 and "WARNING" in msgs[0], msgs


def test_verify_harvest_skips_all_core_member():
    """An all-core member has no skin to compare (and no tets); the ratio is
    NaN and nothing fires -- including when round-off makes v_skin < 0."""
    msgs, log = _sink()
    got = _verify_harvest("iron", [], {"w": (0.0, -1.0e-6, 1.0e9)}, log)
    assert math.isnan(got["w"]), got
    assert not msgs, msgs


def test_stats_report_measured_skin_volume():
    """The measured counterparts of the analytic stats are published, so a
    drift in the tet/analytic ratio is visible without a crash."""
    st = dict(STRUCT, core_clip={"z_max": Z1})
    p = slice_stp_polar(_make_stp(), structure=st, mesh_size_max=25.0,
                        model_name="test_measured", gmsh_verbosity=0)
    s = p["stats"]
    v_tets = sum(_tet_volume(t) for t in p["skin_tets"])
    assert abs(s["skin_tet_volume_mm3"] - v_tets) < 1e-6 * v_tets, s
    assert abs(s["skin_tet_volume_ratio"]
               - v_tets / s["skin_volume_mm3"]) < 1e-12, s
    # this wedge's skin is a coarse ring: within the faceting band, not zero
    assert 0.97 < s["skin_tet_volume_ratio"] < 1.03, s
    assert abs(s["modelled_vs_cad_frac"]) < 0.01, s


def test_corrupt_payload_is_not_cached():
    """A build that fails verification must leave NOTHING in the digest
    cache -- the corrupt 60 MeV payload was written and would have been
    reused on every later run with the same parameters."""
    import glob

    cache_dir = _st._CACHE_DIR
    name = "test_nocache"
    before = set(glob.glob(os.path.join(str(cache_dir), f"{name}-*.pkl")))

    real = _st._verify_harvest

    def boom(*a, **kw):
        raise RuntimeError("synthetic harvest failure")

    _st._verify_harvest = boom
    try:
        slice_stp_polar(_make_stp(), structure=dict(STRUCT,
                                                    core_clip={"z_max": Z1}),
                        mesh_size_max=25.0, model_name=name,
                        gmsh_verbosity=0, use_cache=True)
    except RuntimeError as exc:
        assert "synthetic harvest failure" in str(exc), exc
    else:
        raise AssertionError("expected the build to propagate the failure")
    finally:
        _st._verify_harvest = real

    after = set(glob.glob(os.path.join(str(cache_dir), f"{name}-*.pkl")))
    assert after == before, sorted(after - before)
