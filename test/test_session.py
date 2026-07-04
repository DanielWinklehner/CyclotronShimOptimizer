"""Smoke tests for the Session/CyclotronModel facade (no field solve).

Uses use_mpi=False throughout so the test runner never initializes MPI.
"""

import _testenv  # noqa: F401

import os

import numpy as np

import cyclotron_optimizer as co

CONFIG_PATH = os.path.join(_testenv.REPO_ROOT, "examples", "config_muon_smaller.yml")


def test_package_surface():
    assert hasattr(co, "Session")
    assert hasattr(co, "CyclotronModel")
    assert hasattr(co, "CyclotronConfig")
    assert co.__version__


def test_serial_session_basics():
    s = co.Session(use_mpi=False)
    assert s.rank == 0
    assert s.is_root
    assert s.comm is None
    s.barrier()  # no-op
    assert s.bcast({"a": 1}) == {"a": 1}

    # context manager form
    with co.Session(use_mpi=False) as s2:
        assert s2.is_root


def test_session_requires_config_for_build():
    s = co.Session(use_mpi=False)
    try:
        s.build()
    except ValueError:
        return
    raise AssertionError("Expected ValueError when building without a config")


def test_session_config_defaults_and_model():
    s = co.Session(CONFIG_PATH, use_mpi=False, verbosity=0)
    fe = s.config.field_evaluation

    radii = s.default_radii_mm()
    assert len(radii) == fe.n_eval_pts
    assert np.isclose(radii[0], fe.radius_min_mm)
    assert np.isclose(radii[-1], fe.radius_max_mm)

    shape = s.default_pole_shape()
    assert shape.num_segments == s.config.side_shim.num_rad_segments

    model = s.build()
    assert model.config is s.config
    assert model.cyclotron is None  # nothing built yet
    assert model.converged is None
    assert len(model.radii_mm) == fe.n_eval_pts

    # field queries before solve() must fail loudly, not crash in radia
    for call in (model.field_rz, model.median_plane_field, model.field_3d,
                 model.isochronism):
        try:
            call()
        except RuntimeError:
            continue
        raise AssertionError(f"{call.__name__} should require a solved model")


def test_config_paths_resolve_against_yaml_dir():
    cfg = co.CyclotronConfig.from_yaml(CONFIG_PATH)
    # STP paths in the yml are relative to the yml's folder; they must come
    # back absolute and existing regardless of the current working directory.
    yoke = cfg.component("yoke")
    assert os.path.isabs(yoke.file)
    assert os.path.exists(yoke.file)
