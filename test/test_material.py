"""RadiaMaterial: unit handling, BH/MH curves, numpy access, plotting."""

import _testenv  # noqa: F401

import os

import matplotlib
matplotlib.use("Agg")
import numpy as np

from cyclotron_optimizer.geometry.components import _MU0, RadiaMaterial

RES = os.path.join(_testenv.REPO_ROOT, "resources")
COMSOL_1010 = os.path.join(RES, "COMSOL_1010_BH_T_A-m.csv")
DILLINGER = os.path.join(RES, "dillinger_steel.csv")


def test_comsol_1010_loads_with_si_units():
    mat = RadiaMaterial.from_bh_file(COMSOL_1010, curve="BH", h_unit="A/m",
                                     name="1010")
    raw = np.genfromtxt(COMSOL_1010, delimiter=",")
    raw = raw[~np.isnan(raw).any(axis=1)]

    hb = mat.get_bh_curve(h_unit="A/m", b_unit="T")
    assert hb.shape == raw.shape
    assert np.allclose(hb[:, 0], np.sort(raw[:, 0]))
    assert np.allclose(hb[:, 1], raw[np.argsort(raw[:, 0]), 1])

    # internal convention: mu0*H in Tesla
    hb_t = mat.get_bh_curve(h_unit="T", b_unit="T")
    assert np.allclose(hb_t[:, 0], hb[:, 0] * _MU0)

    # BH vs MH consistency: B = mu0*H + M
    hm = mat.get_mh_curve(h_unit="T", m_unit="T")
    assert np.allclose(hb_t[:, 1], hb_t[:, 0] + hm[:, 1])

    # unit conversions scale linearly
    kam = mat.get_bh_curve(h_unit="kA/m", b_unit="mT")
    assert np.allclose(kam[:, 0], hb[:, 0] / 1e3)
    assert np.allclose(kam[:, 1], hb[:, 1] * 1e3)

    assert mat.material is not None  # radia MatSatIsoTab created
    assert mat.metadata["n_points"] == len(hb)


def test_legacy_tesla_convention_and_type_alias():
    # historic API: both columns mu0*H / B in Tesla, positional 'type'
    legacy = RadiaMaterial.from_bh_file(DILLINGER, type="BH", name="dillinger")
    modern = RadiaMaterial.from_bh_file(DILLINGER, curve="BH", h_unit="T",
                                        b_unit="T", name="dillinger")
    assert np.allclose(legacy.get_bh_curve(), modern.get_bh_curve())
    raw = np.genfromtxt(DILLINGER, delimiter=",")
    raw = raw[~np.isnan(raw).any(axis=1)]
    assert np.allclose(legacy.get_mh_curve()[:, 1], raw[:, 1] - raw[:, 0])


def test_mh_curve_input():
    hm_file = np.column_stack([np.linspace(0, 2e5, 20),
                               np.tanh(np.linspace(0, 3, 20)) * 1.8])
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mh.csv")
        np.savetxt(path, hm_file, delimiter=",")
        mat = RadiaMaterial.from_bh_file(path, curve="MH", h_unit="A/m",
                                         b_unit="T")
        hm = mat.get_mh_curve(h_unit="A/m", m_unit="T")
        assert np.allclose(hm, hm_file)
        hb = mat.get_bh_curve(h_unit="A/m", b_unit="T")
        assert np.allclose(hb[:, 1], hm_file[:, 1] + hm_file[:, 0] * _MU0)


def test_plot_bh_curve_smoke():
    mat = RadiaMaterial.from_bh_file(COMSOL_1010, curve="BH", h_unit="A/m")
    fig, ax = mat.plot_bh_curve(h_unit="kA/m", b_unit="T", kind="both",
                                logx=False)
    assert len(ax.lines) == 2
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_unknown_units_raise():
    try:
        RadiaMaterial.from_bh_file(COMSOL_1010, curve="BH", h_unit="parsec")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "H unit" in str(exc)


def test_non_monotone_curve_warns():
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        RadiaMaterial.from_bh_file(DILLINGER, curve="BH", h_unit="T")
    assert any("NON-MONOTONE" in str(w.message) for w in caught)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        RadiaMaterial.from_bh_file(COMSOL_1010, curve="BH", h_unit="A/m")
    assert not any("NON-MONOTONE" in str(w.message) for w in caught)


def test_material_units_flow_from_config_yaml():
    """materials: entries carry curve/h_unit/b_unit through parsing and
    build_materials passes them to RadiaMaterial."""
    import tempfile
    import textwrap

    from cyclotron_optimizer.config_io.config import CyclotronConfig
    from cyclotron_optimizer.geometry.geometry import build_materials

    yml = textwrap.dedent(f"""\
        particle_species: "muon"
        max_machine_size_mm: 700.0
        materials:
          iron:
            type: bh_file
            file: "{COMSOL_1010.replace(os.sep, '/')}"
            curve: BH
            h_unit: "A/m"
            b_unit: "T"
        symmetries: {{}}
        components:
          - name: coils
            kind: racetrack_pair
            params: {{radius_min_mm: 460, radius_max_mm: 574.5, height_mm: 123.5,
                      midplane_dist: 55, current_A: 1000, num_segments: 5}}
        side_shim: {{include: false, num_rad_segments: 2,
                     angular_resolution_deg: 2.5, default_offset_deg: 5.0,
                     segmentation: [1, 1, 1]}}
        top_shim: {{include: false, num_rad_segments: 2,
                    angular_resolution_deg: 1.25, default_offset_mm: 5.0,
                    segmentation: [1, 1, 1]}}
        field_evaluation: {{num_points_circle: 360, radius_min_mm: 75,
                            radius_max_mm: 350, n_eval_pts: 4}}
        simulation: {{precision: 1.0e-4, iterations: 10}}
        optimization: {{target_frequency_mhz: 42, frequency_tolerance_mhz: 0.001,
                        max_iterations: 1, coil_current_min_A: 1,
                        coil_current_max_A: 2, side_shim_min_deg: 1,
                        side_shim_max_deg: 2, top_shim_min_mm: 1,
                        top_shim_max_mm: 2, num_workers: 1, n_initial_points: 1,
                        reference_coil_current: 1000, regularization_weight: 0.0,
                        optimizer: 'dfo-ls', random_init: false}}
        visualization: {{show_opengl: false}}
        """)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cfg.yml")
        with open(path, "w") as fh:
            fh.write(yml)
        cfg = CyclotronConfig.from_yaml(path)

    assert cfg.materials_def["iron"]["h_unit"] == "A/m"
    assert cfg.materials_def["iron"]["curve"] == "BH"
    mats = build_materials(cfg)
    assert mats["iron"].metadata["h_unit"] == "A/m"
    # H column correctly interpreted as A/m: internal mu0*H extent ~1.19 T
    hb = mats["iron"].get_bh_curve(h_unit="T")
    assert 1.0 < hb[-1, 0] < 1.5
