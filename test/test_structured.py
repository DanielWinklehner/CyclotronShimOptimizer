"""Structured polar-grid slicing (geometry/structured.py, "Option C").

Synthetic revolved solid (90-deg annular wedge with a z-step) written to a
temporary STP, then sliced:
  - classification: every cell interior (zero skin) on a perfectly
    revolved solid, exact expected cell count
  - volume conservation: sum of analytic cell volumes == CAD volume
  - emitted prisms: inscribed-chord deficit matches sin(x)/x theory
  - physics: structured build vs an all-tet twin of the same STP agree
    on B after a small CPU relax (catches face/orientation errors)
"""

import _testenv  # noqa: F401

import math
import os
import tempfile

import numpy as np
import radia as rad

from cyclotron_optimizer.geometry.components import MagnetizedComponent
from cyclotron_optimizer.geometry.structured import (
    _prism_polyhedron, emit_prism_cells, slice_stp_polar)

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
