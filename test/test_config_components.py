"""Tests for the component-based config schema and the legacy adapter.

The legacy fixture (frozen copy of the old config_muon_smaller.yml) must
adapt to the same component description as the FROZEN component-schema
fixture. The live examples/config_muon_smaller.yml is a workflow file the
user tunes freely (mesh sizes, enabled flags, iteration budgets), so it is
only smoke-tested for parseability/consistency -- never for specific values.
"""

import _testenv  # noqa: F401

import os

from cyclotron_optimizer.config_io.config import (
    DEFAULT_CYCLOTRON_SYMMETRIES,
    CyclotronConfig,
)

LEGACY = os.path.join(_testenv.REPO_ROOT, "test", "fixtures",
                      "legacy_muon_smaller.yml")
V2 = os.path.join(_testenv.REPO_ROOT, "test", "fixtures",
                  "component_muon_smaller.yml")
LIVE_EXAMPLE = os.path.join(_testenv.REPO_ROOT, "examples",
                            "config_muon_smaller.yml")

EXPECTED_NAMES = ["yoke", "lid_lower", "lid_upper", "pole",
                  "extract_channel", "coils"]


def test_legacy_adapter_produces_component_specs():
    cfg = CyclotronConfig.from_yaml(LEGACY)

    assert [s.name for s in cfg.components] == EXPECTED_NAMES
    assert "cyclotron_8fold" in cfg.symmetries_def
    assert cfg.materials_def["iron"]["type"] == "bh_file"

    yoke = cfg.component("yoke")
    assert yoke.kind == "stp"
    assert yoke.file and os.path.exists(yoke.file)
    assert yoke.symmetry == "cyclotron_8fold"
    assert yoke.mesh["max_size"] == 50

    pole = cfg.component("pole")
    assert pole.kind == "pole" and pole.shimmed and pole.file is None
    assert pole.params["half_angle_deg"] == 15.0  # half-wedge; full pole 30 deg
    assert pole.params["pole_zs"] == -(188.5 + 96.5)

    channel = cfg.component("extract_channel")
    assert channel.kind == "wedge"  # single wedge + para-z median mirror
    assert channel.enabled == cfg.extract_channel.use_extract_chan
    assert channel.symmetry == "median_z"
    assert channel.params["z_offset_mm"] == (
        cfg.extract_channel.height_mm + cfg.extract_channel.channel_width_mm / 2.0)

    coils = cfg.component("coils")
    assert coils.kind == "racetrack_pair"
    assert coils.params["current_A"] == cfg.coil.current_A

    resolved = cfg.resolved_symmetry("cyclotron_8fold")
    assert len(resolved) == 4
    assert resolved[0][0] == "perp" and resolved[3][0] == "para"


def test_component_yaml_parses_and_reverse_fills_legacy():
    cfg = CyclotronConfig.from_yaml(V2)

    assert [s.name for s in cfg.components] == EXPECTED_NAMES
    yoke = cfg.component("yoke")
    assert yoke.kind == "stp" and os.path.exists(yoke.file)

    # reverse-filled legacy dataclasses still consumed elsewhere:
    assert cfg.coil.current_A == 15368          # solver's live current source
    assert cfg.coil.radius_min_mm == 460.0
    assert cfg.pole.outer_radius_mm == 400.0    # physics preconditioner
    assert cfg.pole.full_angle_deg == 30.0
    assert cfg.extract_channel.use_extract_chan is True
    assert cfg.material.bh_filename and os.path.exists(cfg.material.bh_filename)

    # workflow sections parse as usual
    assert cfg.field_evaluation.iso_method == "gordon"
    assert cfg.side_shim.num_rad_segments == 14


def test_live_example_config_parses_consistently():
    """The LIVE example yml is user-tunable; assert only structure and
    internal consistency, not specific workflow values."""
    cfg = CyclotronConfig.from_yaml(LIVE_EXAMPLE)
    assert [s.name for s in cfg.components] == EXPECTED_NAMES
    assert cfg.extract_channel.use_extract_chan == \
        cfg.component("extract_channel").enabled
    assert cfg.coil.current_A == cfg.component("coils").params["current_A"]
    assert cfg.material.bh_filename and os.path.exists(cfg.material.bh_filename)


def test_both_schemas_describe_the_same_machine():
    legacy = CyclotronConfig.from_yaml(LEGACY)
    v2 = CyclotronConfig.from_yaml(V2)

    def signature(cfg):
        return [(s.name, s.kind, s.enabled, s.shimmed, s.symmetry,
                 s.mesh.get("max_size")) for s in cfg.components]

    assert signature(legacy) == signature(v2)

    # symmetry sets match the canonical default in both schemas
    for cfg in (legacy, v2):
        resolved = cfg.resolved_symmetry("cyclotron_8fold")
        expected = [(k, list(p), list(n)) for k, p, n in DEFAULT_CYCLOTRON_SYMMETRIES]
        assert resolved == expected

    # the pole build parameters agree (the adapter computes pole_zs from the
    # yoke/lid heights; the v2 file states it explicitly)
    assert legacy.component("pole").params == v2.component("pole").params

    # channel + coil parameters agree
    assert legacy.component("extract_channel").params == \
        v2.component("extract_channel").params
    assert legacy.component("coils").params == v2.component("coils").params


def test_build_coils_from_component_spec():
    """The racetrack_pair builder reads geometry from the spec but the LIVE
    current from config.coil.current_A (mutated by the coil inner loop)."""
    import cyclotron_optimizer.geometry.components as components
    from _radia_stub import RadiaStub
    from cyclotron_optimizer.geometry.geometry import build_coils
    from cyclotron_optimizer.geometry.symmetry import canonical_symmetry_set

    cfg = CyclotronConfig.from_yaml(V2)
    stub = RadiaStub()
    real_rad = components.rad
    components.rad = stub
    try:
        cfg.coil.current_A = 22222.0   # solver-style live mutation
        coils = build_coils(cfg)

        tracks = [c for c in stub.calls if c[0] == "ObjRaceTrk"]
        assert len(tracks) == 2
        # current density = I / (height * dr); geometry from the spec
        expected_j = 22222.0 / (123.5 * (574.5 - 460.0))
        for t in tracks:
            assert abs(t[5] - expected_j) < 1e-12
        assert {t[1][2] for t in tracks} == {55 + 123.5 / 2, -(55 + 123.5 / 2)}

        # declared (not applied) field symmetry from the spec
        assert canonical_symmetry_set(coils.symmetries) == \
            canonical_symmetry_set(cfg.resolved_symmetry("cyclotron_8fold"))
        assert not any(c[0].startswith("TrfZer") for c in stub.calls)
    finally:
        components.rad = real_rad


def test_all_example_configs_load():
    """Every shipped example config parses, resolves its files, and exposes a
    complete machine description."""
    import glob

    ymls = sorted(glob.glob(os.path.join(_testenv.REPO_ROOT, "examples", "*.yml")))
    assert len(ymls) >= 3
    for path in ymls:
        cfg = CyclotronConfig.from_yaml(path)
        assert cfg.components, path
        names = [s.name for s in cfg.components]
        assert len(set(names)) == len(names), path

        # exactly one live current source, built FULL-SIZE (both coils; radia
        # symmetry transforms on current sources are unsupported on the GPU
        # field path -- RadiaCUDA issue #16 -- so they are only declared)
        current = [s for s in cfg.components if s.kind == "racetrack_pair"]
        assert len(current) == 1, path
        assert cfg.coil.current_A == current[0].params["current_A"], path

        # at most one shimmed (rebuildable) component
        assert sum(s.shimmed for s in cfg.components) <= 1, path

        # material files resolve relative to the yml
        for mdef in cfg.materials_def.values():
            if "file" in mdef:
                assert os.path.exists(mdef["file"]), (path, mdef)
        # any STP files resolve too
        for spec in cfg.components:
            if spec.file:
                assert os.path.exists(spec.file), (path, spec.name)

        # every referenced symmetry is defined
        for spec in cfg.components:
            cfg.resolved_symmetry(spec.symmetry)


def test_component_accessors():
    cfg = CyclotronConfig.from_yaml(V2)
    try:
        cfg.component("nonexistent")
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError for unknown component")

    assert cfg.resolved_symmetry(None) == []
    try:
        cfg.resolved_symmetry("undefined_symmetry")
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError for undefined symmetry")