"""End-to-end tests for simulation.field_calculator with a stubbed radia.

Builds a fake cyclotron component tree (symmetric iron + declared-symmetric
coils + asymmetric extraction channel), registers analytic fields on the radia
stub, and checks that the symmetry-folded evaluation reproduces the full
evaluation exactly.
"""

import _testenv  # noqa: F401

import numpy as np

import cyclotron_optimizer.geometry.components as components
import cyclotron_optimizer.simulation.field_calculator as fc
from _radia_stub import RadiaStub
from cyclotron_optimizer.geometry.components import BaseRadiaComponent
from cyclotron_optimizer.geometry.symmetry import symmetry_group

SYMS = [
    ("perp", [0, 0, 0], [1, -1, 0]),
    ("perp", [0, 0, 0], [1, 0, 0]),
    ("perp", [0, 0, 0], [0, 1, 0]),
    ("para", [0, 0, 0], [0, 0, 1]),
]

_REAL_FC_RAD = fc.rad
_REAL_COMP_RAD = components.rad


def _symmetric_field(seed_scale=1.0):
    ops = symmetry_group(SYMS)

    def base(pts):
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        return seed_scale * np.stack([
            0.3 * np.sin(0.011 * x) + 2.1e-4 * y * z,
            0.2 * np.cos(0.013 * y) + 1.7e-4 * x * z,
            0.5 + 0.02 * np.cos(0.008 * z) + 3.0e-5 * x * y,
        ], axis=1)

    def field(pts):
        pts = np.asarray(pts, dtype=float)
        total = np.zeros((len(pts), 3))
        for r_mat, f_mat in ops:
            total += base(pts @ r_mat.T) @ f_mat
        return total / len(ops)

    return field


def _channel_field(pts):
    pts = np.asarray(pts, dtype=float)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return np.stack([
        0.05 * np.sin(0.017 * x + 0.5) + 1e-3 * y,
        0.03 * np.cos(0.019 * y + 1.1) + 2e-3 * z,
        0.01 * np.sin(0.023 * z + 0.3) + 3e-3 * x + 0.005,
    ], axis=1)


def _make_model(with_channel=True):
    """(stub, top_component). Iron/coils get one shared symmetric field each;
    the optional channel gets an asymmetric field."""
    stub = RadiaStub()
    fc.rad = stub
    components.rad = stub

    iron = BaseRadiaComponent(101, symmetries=SYMS)
    coils = BaseRadiaComponent(102, symmetries=SYMS)
    stub.register_field(101, _symmetric_field(1.0))
    stub.register_field(102, _symmetric_field(0.4))

    members = [iron, coils]
    if with_channel:
        channel = BaseRadiaComponent(103)
        stub.register_field(103, _channel_field)
        members.append(channel)

    top = BaseRadiaComponent.containerize(members)
    return stub, top


def _restore():
    fc.rad = _REAL_FC_RAD
    components.rad = _REAL_COMP_RAD


def _axis(limit, step):
    n = int(round(limit / step))
    half = np.arange(n + 1) * float(step)
    return np.concatenate([-half[:0:-1], half])


# ---------------------------------------------------------------------------
# Source grouping
# ---------------------------------------------------------------------------
def test_field_source_groups():
    stub, top = _make_model(with_channel=True)
    try:
        groups = fc._field_source_groups(top, use_symmetry=True)
        assert len(groups) == 2
        by_temp = {g.temp: g for g in groups}
        # iron + coils share the symmetry set -> combined into a temp container
        assert set(stub.containers[by_temp[True].radia_id]) == {101, 102}
        assert len(by_temp[True].symmetries) == 4
        # the channel stands alone with no symmetries
        assert by_temp[False].radia_id == 103
        assert by_temp[False].symmetries == []

        # symmetry off -> single group, whole model, no folding
        groups_off = fc._field_source_groups(top, use_symmetry=False)
        assert len(groups_off) == 1 and groups_off[0].radia_id == top.id
        assert groups_off[0].symmetries == []

        # bare radia id -> single group
        groups_id = fc._field_source_groups(top.id, use_symmetry=True)
        assert len(groups_id) == 1 and groups_id[0].symmetries == []
    finally:
        _restore()


# ---------------------------------------------------------------------------
# Folded vs full evaluation
# ---------------------------------------------------------------------------
def test_get_field_2d_symmetry_matches_full():
    stub, top = _make_model(with_channel=True)
    try:
        axis = _axis(40.0, 5.0)
        f_sym = fc.get_field_2d(top, axis, axis, 0.0, use_symmetry=True,
                                verbosity=0, use_gpu=False)
        f_full = fc.get_field_2d(top, axis, axis, 0.0, use_symmetry=False,
                                 verbosity=0, use_gpu=False)
        for comp in ("x", "y", "z"):
            assert np.allclose(f_sym.grid_values[comp], f_full.grid_values[comp],
                               atol=1e-12), f"component {comp} differs"
        # grid in meters
        assert np.allclose(f_sym.grid["x"], axis * 1e-3)
        # temp group containers were cleaned up
        created = [c[2] for c in stub.calls if c[0] == "ObjCnt" and set(c[1]) == {101, 102}]
        assert created and all(cid in stub.deleted for cid in created)
    finally:
        _restore()


def test_get_field_3d_symmetry_matches_full():
    stub, top = _make_model(with_channel=True)
    try:
        xy = _axis(30.0, 10.0)
        z = np.arange(-20.0, 10.0 + 1e-9, 10.0)  # asymmetric z range
        f_sym = fc.get_field_3d(top, xy, xy, z, use_symmetry=True,
                                verbosity=0, use_gpu=False)
        f_full = fc.get_field_3d(top, xy, xy, z, use_symmetry=False,
                                 verbosity=0, use_gpu=False)
        for comp in ("x", "y", "z"):
            assert np.allclose(f_sym.grid_values[comp], f_full.grid_values[comp],
                               atol=1e-12), f"component {comp} differs"
        assert f_sym.dim == 3

        # Field evaluation at a grid node reproduces the direct sum of sources
        pt_mm = np.array([[10.0, -20.0, 0.0]])
        expected = (_symmetric_field(1.0)(pt_mm) + _symmetric_field(0.4)(pt_mm)
                    + _channel_field(pt_mm))
        got = f_sym(pt_mm * 1e-3)
        assert np.allclose(got, expected, atol=1e-9)
    finally:
        _restore()


def test_three_way_mixed_symmetry_matches_full():
    """Components with FOUR planes, ONE plane, and NO symmetry in one model:
    each source group folds by its own set; the sum matches full evaluation."""
    one_plane = [("perp", [0, 0, 0], [0, 1, 0])]

    stub = RadiaStub()
    fc.rad = stub
    components.rad = stub
    try:
        iron = BaseRadiaComponent(201, symmetries=SYMS)          # 4 planes
        half = BaseRadiaComponent(202, symmetries=one_plane)     # 1 plane
        channel = BaseRadiaComponent(203)                        # none

        def _sym_field(sym_set, scale):
            ops = symmetry_group(sym_set)

            def base(pts):
                x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
                return scale * np.stack([
                    0.2 * np.sin(0.013 * x) + 1e-4 * y * z,
                    0.1 * np.cos(0.017 * y) + 2e-4 * x,
                    0.4 + 0.03 * np.sin(0.011 * z) + 5e-5 * x * y,
                ], axis=1)

            def field(pts):
                pts = np.asarray(pts, dtype=float)
                total = np.zeros((len(pts), 3))
                for r_mat, f_mat in ops:
                    total += base(pts @ r_mat.T) @ f_mat
                return total / len(ops)

            return field

        stub.register_field(201, _sym_field(SYMS, 1.0))
        stub.register_field(202, _sym_field(one_plane, 0.5))
        stub.register_field(203, _channel_field)

        top = BaseRadiaComponent.containerize([iron, half, channel])

        # three distinct symmetry sets -> three separate source groups
        groups = fc._field_source_groups(top, use_symmetry=True)
        assert len(groups) == 3
        assert sorted(len(g.symmetries) for g in groups) == [0, 1, 4]

        axis = _axis(40.0, 5.0)
        f_sym = fc.get_field_2d(top, axis, axis, 0.0, use_symmetry=True,
                                verbosity=0, use_gpu=False)
        f_full = fc.get_field_2d(top, axis, axis, 0.0, use_symmetry=False,
                                 verbosity=0, use_gpu=False)
        for comp in ("x", "y", "z"):
            assert np.allclose(f_sym.grid_values[comp], f_full.grid_values[comp],
                               atol=1e-12), f"component {comp} differs"

        # rz path: the model-wide INTERSECTION is empty -> full-circle sampling
        rz = fc.get_field_rz(top, [100.0], num_angles=800, use_symmetry=True,
                             verbosity=0, use_gpu=False)
        assert rz.bz.shape == (1, 800)
    finally:
        _restore()


def test_symmetric_model_uses_fewer_evaluations():
    stub, top = _make_model(with_channel=False)
    try:
        axis = _axis(40.0, 5.0)
        fc.get_field_2d(top, axis, axis, 0.0, use_symmetry=True,
                        verbosity=0, use_gpu=False)
        n_folded = sum(c[3] for c in stub.calls if c[0] == "Fld")

        stub.calls.clear()
        fc.get_field_2d(top, axis, axis, 0.0, use_symmetry=False,
                        verbosity=0, use_gpu=False)
        n_full = sum(c[3] for c in stub.calls if c[0] == "Fld")

        assert n_full == len(axis) ** 2
        assert n_folded < 0.2 * n_full  # ~1/8 of the plane (octant + boundary)
    finally:
        _restore()


# ---------------------------------------------------------------------------
# r-z (isochronism) path
# ---------------------------------------------------------------------------
def test_get_field_rz_sector_and_values():
    stub, top = _make_model(with_channel=False)
    try:
        radii = [50.0, 100.0, 150.0]
        rz = fc.get_field_rz(top, radii, num_angles=1000, use_symmetry=True,
                             verbosity=0, use_gpu=False)
        # 8-fold: pi/4 sector, 1000/8 = 125 angles
        assert rz.bz.shape == (3, 125)
        assert len(rz.angles) == 125
        assert np.isclose(rz.angles[-1] + rz.angles[1], np.pi / 4.0)  # endpoint=False

        # values match a direct evaluation of the summed sources
        pts = np.zeros((len(radii), len(rz.angles), 3))
        pts[:, :, 0] = np.asarray(radii)[:, None] * np.cos(rz.angles)[None, :]
        pts[:, :, 1] = np.asarray(radii)[:, None] * np.sin(rz.angles)[None, :]
        expected = (_symmetric_field(1.0)(pts.reshape(-1, 3))
                    + _symmetric_field(0.4)(pts.reshape(-1, 3)))[:, 2]
        assert np.allclose(rz.bz.ravel(), expected, atol=1e-12)
    finally:
        _restore()


def test_get_field_rz_falls_back_to_full_circle_with_channel():
    stub, top = _make_model(with_channel=True)
    try:
        rz = fc.get_field_rz(top, [100.0], num_angles=1000, use_symmetry=True,
                             verbosity=0, use_gpu=False)
        # intersection of symmetries is empty -> full circle, no folding
        assert rz.bz.shape == (1, 1000)
        assert np.isclose(rz.angles[-1], 2.0 * np.pi * 999 / 1000)
    finally:
        _restore()


def test_gpu_precision_plumbing():
    """gpu_precision='single' must reach rad.Fld; the default stays 'double'."""
    stub, top = _make_model(with_channel=False)
    try:
        axis = _axis(20.0, 10.0)
        fc.get_field_2d(top, axis, axis, 0.0, use_symmetry=True,
                        verbosity=0, use_gpu=False, gpu_precision="single")
        precisions = {c[4] for c in stub.calls if c[0] == "Fld"}
        assert precisions == {"single"}

        stub.calls.clear()
        fc.get_field_2d(top, axis, axis, 0.0, use_symmetry=True,
                        verbosity=0, use_gpu=False)
        precisions = {c[4] for c in stub.calls if c[0] == "Fld"}
        assert precisions == {"double"}
    finally:
        _restore()


class _FakeSolverConfig:
    """Minimal config for solver tests: live coil current + symmetry lookup."""

    def __init__(self):
        from types import SimpleNamespace
        self.coil = SimpleNamespace(current_A=0.0)
        self.simulation = SimpleNamespace(precision=1e-4, iterations=100)

    def resolved_symmetry(self, name):
        return list(SYMS) if name else []


def test_solver_reuses_static_iron_across_pole_rebuilds():
    """build() #2 must reuse the static parts, rebuild only pole+coils, free
    the old pole, and re-apply the symmetry to fresh assembly containers."""
    from cyclotron_optimizer.config_io.config import ComponentSpec
    from cyclotron_optimizer.geometry.components import MagnetizedComponent

    stub = RadiaStub()
    fc.rad = stub
    components.rad = stub

    calls = {"static": 0, "pole": 0, "coils": 0}
    next_id = [500]

    def _leaf():
        next_id[0] += 1
        return MagnetizedComponent(next_id[0])

    def _spec(name, symmetry, shimmed=False):
        return ComponentSpec(name=name, kind="test", symmetry=symmetry,
                             shimmed=shimmed)

    def fake_static(config, *, rank=0, comm=None, verbosity=1):
        calls["static"] += 1
        return {"parts": [(_spec("yoke", "S8"), _leaf()),
                          (_spec("lid_lower", "S8"), _leaf()),
                          (_spec("lid_upper", "S8"), _leaf()),
                          (_spec("channel", None), _leaf())],
                "materials": {}}

    def fake_pole(config, pole_shape, *, comm=None, materials=None):
        calls["pole"] += 1
        return (_spec("pole", "S8", shimmed=True),
                MagnetizedComponent(9000 + calls["pole"]))

    def fake_coils(config):
        calls["coils"] += 1
        return BaseRadiaComponent(8000 + calls["coils"])

    real = (fc.build_static_iron_parts, fc.build_pole_part, fc.build_coils)
    fc.build_static_iron_parts = fake_static
    fc.build_pole_part = fake_pole
    fc.build_coils = fake_coils
    try:
        config = _FakeSolverConfig()
        solver = fc.ReusableCyclotronSolver(config, [100.0], verbosity=0)

        _, bz, converged, misfit = solver.build("shape1", 1000.0, query=False)
        assert bz is None and converged and misfit == 0.5e-4
        assert calls == {"static": 1, "pole": 1, "coils": 1}
        assert sum(c[0] == "UtiDelAll" for c in stub.calls) == 1
        # two assembly containers: S8 group (statics + pole) and the channel
        assert len(solver._iron_subs) == 2
        sym_container_1 = solver._iron_subs[0].id
        chan_container_1 = solver._iron_subs[1].id
        assert 9001 in stub.containers[sym_container_1]
        # symmetry applied to the S8 container only (4 transforms)
        trf = [c for c in stub.calls if c[0].startswith("TrfZer")]
        assert len(trf) == 4 and {c[1] for c in trf} == {sym_container_1}

        # relax method: GPU default -> 9
        assert [c[4] for c in stub.calls if c[0] == "RlxAuto"] == [9]

        solver.build("shape2", 1200.0, query=False)
        assert calls == {"static": 1, "pole": 2, "coils": 2}   # static NOT rebuilt
        assert sum(c[0] == "UtiDelAll" for c in stub.calls) == 1  # no global wipe
        assert 9001 in stub.deleted                     # old pole freed
        assert sym_container_1 in stub.deleted          # old containers freed
        assert chan_container_1 in stub.deleted
        for _, comp in solver._static_parts["parts"]:
            assert comp.id not in stub.deleted          # statics survive
        sym_container_2 = solver._iron_subs[0].id
        assert sym_container_2 != sym_container_1
        assert 9002 in stub.containers[sym_container_2]
        # symmetry re-applied to the FRESH container
        trf = [c for c in stub.calls if c[0].startswith("TrfZer")]
        assert len(trf) == 8 and {c[1] for c in trf[4:]} == {sym_container_2}
        # reused-iron relax starts from zero magnetization
        zerom_opts = [c[5] for c in stub.calls if c[0] == "RlxAuto"]
        assert "ZeroM->False" in zerom_opts[0]
        assert "ZeroM->True" in zerom_opts[1]

        # opting out forces the full rebuild
        solver.build("shape3", 1200.0, query=False, reuse_static_iron=False)
        assert calls["static"] == 2
        assert sum(c[0] == "UtiDelAll" for c in stub.calls) == 2

        # CPU solver uses relax method 10 (CPU adaptive Jacobi) and CPU assembly
        solver_cpu = fc.ReusableCyclotronSolver(config, [100.0], verbosity=0,
                                                use_gpu=False)
        solver_cpu.build("shape1", 1000.0, query=False, reuse_static_iron=False)
        assert [c[4] for c in stub.calls if c[0] == "RlxAuto"][-1] == 10
        assert [c[3] for c in stub.calls if c[0] == "RlxPre"][-1] is False

        # granular options: CPU assembly + GPU relaxation
        solver_mix = fc.ReusableCyclotronSolver(
            config, [100.0], verbosity=0,
            use_gpu={"assembly": False, "relaxation": True, "field": True})
        assert solver_mix.gpu == fc.GpuOptions(assembly=False, relaxation=True,
                                               field=True)
        assert solver_mix.use_gpu is True  # back-compat property = field switch
        solver_mix.build("shape1", 1000.0, query=False, reuse_static_iron=False)
        assert [c[3] for c in stub.calls if c[0] == "RlxPre"][-1] is False
        assert [c[4] for c in stub.calls if c[0] == "RlxAuto"][-1] == 9
    finally:
        fc.build_static_iron_parts, fc.build_pole_part, fc.build_coils = real
        _restore()


def test_gpu_options_coercion():
    assert fc.GpuOptions.coerce(True) == fc.GpuOptions(True, True, True)
    assert fc.GpuOptions.coerce(False) == fc.GpuOptions(False, False, False)
    assert fc.GpuOptions.coerce({"assembly": False}) == \
        fc.GpuOptions(assembly=False, relaxation=True, field=True)


# ---------------------------------------------------------------------------
# Perturbative-component staging (frozen-background solve + iteration)
# ---------------------------------------------------------------------------
def _perturb_env(stub, **solver_kwargs):
    """Solver over fake builders with a PERTURBATIVE channel; returns
    (solver, restore_fn). The channel spec has perturbative=True, so the
    solver must exclude it from the main relax and stage it separately."""
    from cyclotron_optimizer.config_io.config import ComponentSpec
    from cyclotron_optimizer.geometry.components import MagnetizedComponent

    fc.rad = stub
    components.rad = stub

    next_id = [500]

    def _leaf():
        next_id[0] += 1
        stub.magnetizations[next_id[0]] = [0.0, 0.0, 0.0]
        return MagnetizedComponent(next_id[0])

    def fake_static(config, *, rank=0, comm=None, verbosity=1):
        return {"parts": [
            (ComponentSpec(name="yoke", kind="t", symmetry="S8"), _leaf()),
            (ComponentSpec(name="channel", kind="t", symmetry=None,
                           perturbative=True), _leaf()),
        ], "materials": {}}

    def fake_pole(config, pole_shape, *, comm=None, materials=None):
        return (ComponentSpec(name="pole", kind="t", symmetry="S8",
                              shimmed=True), _leaf())

    def fake_coils(config):
        return BaseRadiaComponent(8000)

    real = (fc.build_static_iron_parts, fc.build_pole_part, fc.build_coils)
    fc.build_static_iron_parts = fake_static
    fc.build_pole_part = fake_pole
    fc.build_coils = fake_coils

    def restore():
        fc.build_static_iron_parts, fc.build_pole_part, fc.build_coils = real
        _restore()

    solver = fc.ReusableCyclotronSolver(_FakeSolverConfig(), [100.0],
                                        verbosity=0, **solver_kwargs)
    return solver, restore


def test_perturbative_stage_ordering_and_srcobj():
    """Main relax excludes the perturbative group; stage 1 relaxes it with
    srcobj = the main container; the cyclotron container holds both."""
    stub = RadiaStub()
    solver, restore = _perturb_env(stub)
    try:
        _, _, converged, misfit = solver.build("shape", 1000.0, query=False)
        assert converged and misfit == 0.5e-4

        pre = [c for c in stub.calls if c[0] == "RlxPre"]
        assert len(pre) == 2
        (_, main_id, im_main, _, src_main), (_, p_id, im_p, _, src_p) = pre
        # stage 0: main target, no srcobj
        assert src_main == 0
        assert main_id == solver._main_cnt.id
        # stage 1: perturbative target with the main container as source
        assert p_id == solver._perturb_cnt.id
        assert src_p == solver._main_cnt.id

        # main container = S8 iron group + coils; channel NOT a member
        main_members = stub.containers[solver._main_cnt.id]
        assert 8000 in main_members
        assert solver._perturb_cnt.id not in main_members
        # cyclotron = [main container, perturbative container] (field eval)
        assert stub.containers[solver._cyclotron.id] == \
            [solver._main_cnt.id, solver._perturb_cnt.id]

        # no stage 2 by default: exactly two relaxations, no RlxUpdSrc
        assert sum(c[0] == "RlxAuto" for c in stub.calls) == 2
        assert not any(c[0] == "RlxUpdSrc" for c in stub.calls)

        # coil-current re-solve runs the same staging on fresh containers
        solver.resolve_at_current(1200.0, query=False)
        pre2 = [c for c in stub.calls if c[0] == "RlxPre"][2:]
        assert len(pre2) == 2 and pre2[1][4] == solver._main_cnt.id
    finally:
        restore()


def test_perturbative_stage2_iterates_via_updsrc():
    """Stage 2: one main-IM rebuild with srcobj=perturb, then per cycle
    RlxUpdSrc(main) -> relax -> RlxUpdSrc(perturb) -> relax."""
    stub = RadiaStub()
    stub.m_step = 1.0  # keep delta-M large: no early tol exit
    solver, restore = _perturb_env(stub, perturb_iterations=2,
                                   perturb_tol=1e-6)
    try:
        solver.build("shape", 1000.0, query=False)

        pre = [c for c in stub.calls if c[0] == "RlxPre"]
        assert len(pre) == 3  # main, perturb, main REBUILD with srcobj
        assert pre[2][1] == solver._main_cnt.id
        assert pre[2][4] == solver._perturb_cnt.id
        im_main2, im_p = pre[2][2], pre[1][2]

        upd = [c[1] for c in stub.calls if c[0] == "RlxUpdSrc"]
        assert upd == [im_main2, im_p, im_main2, im_p]  # 2 full cycles

        # relax count: stage 0 + stage 1 + 2 cycles x (main + perturb)
        assert sum(c[0] == "RlxAuto" for c in stub.calls) == 6
        # stage-2 relaxations are warm (ZeroM->False)
        zerom = [c[5][0] for c in stub.calls if c[0] == "RlxAuto"]
        assert all(z == "ZeroM->False" for z in zerom[2:])
    finally:
        restore()


def test_perturbative_stage2_tol_early_exit():
    """With static magnetizations (delta M = 0 < tol) stage 2 stops after
    the first cycle even though more iterations were allowed."""
    stub = RadiaStub()  # m_step = 0 -> delta M stays 0
    solver, restore = _perturb_env(stub, perturb_iterations=5,
                                   perturb_tol=1e-3)
    try:
        solver.build("shape", 1000.0, query=False)
        assert sum(c[0] == "RlxAuto" for c in stub.calls) == 4  # 2 + 1 cycle
        assert sum(c[0] == "RlxUpdSrc" for c in stub.calls) == 2
    finally:
        restore()


def test_anderson_config_option_sets_environment():
    """simulation.anderson drives the RADIA_ANDERSON / RADIA_NO_ANDERSON
    environment switches read by RadiaCUDA on each RlxAuto call."""
    import os

    stub = RadiaStub()
    solver, restore = _perturb_env(stub)
    try:
        for value, expect_on, expect_off in [
            (True, "1", None), (False, None, "1")]:
            solver.config.simulation.anderson = value
            os.environ.pop("RADIA_ANDERSON", None)
            os.environ.pop("RADIA_NO_ANDERSON", None)
            solver.build("shape", 1000.0, query=False,
                         reuse_static_iron=False)
            assert os.environ.get("RADIA_ANDERSON") == expect_on
            assert os.environ.get("RADIA_NO_ANDERSON") == expect_off
    finally:
        os.environ.pop("RADIA_ANDERSON", None)
        os.environ.pop("RADIA_NO_ANDERSON", None)
        restore()


def test_perturbative_flag_defaults_keep_single_stage():
    """Without perturbative components the solver behaves exactly as before:
    one IM on the full container, one relaxation."""
    from cyclotron_optimizer.config_io.config import ComponentSpec
    from cyclotron_optimizer.geometry.components import MagnetizedComponent

    stub = RadiaStub()
    fc.rad = stub
    components.rad = stub

    def fake_static(config, *, rank=0, comm=None, verbosity=1):
        return {"parts": [(ComponentSpec(name="yoke", kind="t", symmetry="S8"),
                           MagnetizedComponent(501))], "materials": {}}

    real = (fc.build_static_iron_parts, fc.build_pole_part, fc.build_coils)
    fc.build_static_iron_parts = fake_static
    fc.build_pole_part = lambda *a, **k: None
    fc.build_coils = lambda config: BaseRadiaComponent(8000)
    try:
        solver = fc.ReusableCyclotronSolver(_FakeSolverConfig(), [100.0],
                                            verbosity=0)
        solver.build("shape", 1000.0, query=False)
        pre = [c for c in stub.calls if c[0] == "RlxPre"]
        assert len(pre) == 1 and pre[0][4] == 0
        assert pre[0][1] == solver._cyclotron.id
        assert solver._main_cnt is None and solver._perturb_cnt is None
        assert sum(c[0] == "RlxAuto" for c in stub.calls) == 1
    finally:
        fc.build_static_iron_parts, fc.build_pole_part, fc.build_coils = real
        _restore()
    opts = fc.GpuOptions(relaxation=False)
    assert fc.GpuOptions.coerce(opts) is opts


def test_rz_grid_angles_reach_isochronicity():
    """core.isochronicity must use the angles attached to the RZFieldGrid."""
    from types import SimpleNamespace

    from cyclotron_optimizer.core.isochronicity import _grid_and_angles

    cfg = SimpleNamespace(field_evaluation=SimpleNamespace(
        num_points_circle=1000, use_symmetry=True))

    bz = np.ones((2, 7))
    angles = np.linspace(0.0, np.pi, 7, endpoint=False)
    rz = fc.RZFieldGrid(bz=bz, angles=angles, radii_mm=np.array([10.0, 20.0]))

    grid, used_angles = _grid_and_angles(rz, cfg)
    assert grid.shape == (2, 7)
    assert np.allclose(used_angles, angles)

    # plain array falls back to the config-derived octant angles
    grid2, fallback = _grid_and_angles(np.ones((2, 125)), cfg)
    assert len(fallback) == 125
    assert np.isclose(fallback[1], (np.pi / 4.0) / 125)
