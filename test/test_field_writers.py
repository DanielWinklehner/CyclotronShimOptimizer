"""Round-trip tests for the PyPATools field writers and Field save/load.

Every write -> load pair must reproduce the grid and values exactly (up to the
text-format precision). Grids deliberately have UNEQUAL axis lengths so any
axis-ordering / transposition bug fails loudly (the class of bug that scrambled
the old save functions).
"""

import _testenv  # noqa: F401

import os
import tempfile

import h5py
import numpy as np

from PyPATools.field import Field
from PyPATools.field_src.field_writers import write_comsol

RNG = np.random.default_rng(20260702)


def _tmp(name):
    return os.path.join(tempfile.mkdtemp(prefix="cyclo_field_test_"), name)


def test_write_comsol_2d_bz_roundtrip():
    x = np.linspace(-0.4, 0.4, 17)
    y = np.linspace(-0.4, 0.4, 11)  # unequal on purpose
    bz = RNG.normal(1.0, 0.1, size=(17, 11))

    path = _tmp("midplane.comsol")
    write_comsol(path, {"x": x, "y": y, "z": np.array([0.0])}, {"z": bz},
                 components="z", description="test midplane")

    field = Field.from_file(path)
    assert field.dim == 2
    assert np.allclose(field.grid["x"], x)
    assert np.allclose(field.grid["y"], y)
    assert np.allclose(field.grid_values["z"], bz, atol=1e-9)
    # unwritten components load as zeros
    assert np.allclose(field.grid_values["x"], 0.0)
    assert np.allclose(field.grid_values["y"], 0.0)


def test_write_comsol_3d_full_roundtrip():
    x = np.linspace(-0.05, 0.05, 7)
    y = np.linspace(-0.05, 0.05, 5)
    z = np.linspace(-0.10, 0.025, 6)
    values = {c: RNG.normal(0.0, 0.5, size=(7, 5, 6)) for c in ("x", "y", "z")}

    path = _tmp("bore.comsol")
    write_comsol(path, {"x": x, "y": y, "z": z}, values)

    field = Field.from_file(path)
    assert field.dim == 3
    for axis, arr in (("x", x), ("y", y), ("z", z)):
        assert np.allclose(field.grid[axis], arr)
    for comp in ("x", "y", "z"):
        assert np.allclose(field.grid_values[comp], values[comp], atol=1e-9), \
            f"component {comp} scrambled"


def test_field_save_dispatch_comsol():
    x = np.linspace(-0.1, 0.1, 9)
    y = np.linspace(-0.1, 0.1, 13)
    values = {"x": np.zeros((9, 13)),
              "y": np.zeros((9, 13)),
              "z": RNG.normal(1.5, 0.05, size=(9, 13))}
    field = Field.from_arrays({"x": x, "y": y}, values, label="midplane test")

    path = _tmp("dispatch.comsol")
    field.save(path, components="z")

    reloaded = Field.from_file(path)
    assert np.allclose(reloaded.grid_values["z"], values["z"], atol=1e-9)

    # interpolators built from the reloaded file agree at the nodes
    pts = np.column_stack([np.repeat(x, 13), np.tile(y, 9), np.zeros(9 * 13)])
    assert np.allclose(reloaded(pts)[:, 2], values["z"].ravel(), atol=1e-8)


def test_field_pickle_roundtrip_keeps_raw_arrays():
    x = np.linspace(-0.1, 0.1, 5)
    y = np.linspace(-0.1, 0.1, 4)
    z = np.linspace(-0.1, 0.1, 3)
    values = {c: RNG.normal(size=(5, 4, 3)) for c in ("x", "y", "z")}
    field = Field.from_arrays({"x": x, "y": y, "z": z}, values)

    path = _tmp("field.pickle")
    field.save(path)
    reloaded = Field.from_file(path)
    assert reloaded.grid is not None and reloaded.grid_values is not None
    for comp in ("x", "y", "z"):
        assert np.allclose(reloaded.grid_values[comp], values[comp])


def test_save_to_h5part_layout_and_attrs():
    x = np.linspace(-0.02, 0.02, 5)
    y = np.linspace(-0.02, 0.02, 7)
    z = np.linspace(-0.03, 0.01, 9)
    values = {c: RNG.normal(size=(5, 7, 9)) for c in ("x", "y", "z")}
    field = Field.from_arrays({"x": x, "y": y, "z": z}, values)

    path = _tmp("field.h5part")
    field.save(path, resonance_frequency_hz=45.0e6)

    with h5py.File(path, "r") as h5:
        assert np.isclose(h5.attrs["Resonance Frequency(Hz)"][0], 45.0e6)
        hfield = h5["Step#0"]["Block"]["Hfield"]
        hx = np.array(hfield["0"])
        hz = np.array(hfield["2"])
        # file layout is (nz, ny, nx)
        assert hx.shape == (9, 7, 5)
        # spot-check the transposition: file[i, j, k] == values[k, j, i]
        for i, j, k in [(0, 0, 0), (8, 6, 4), (3, 2, 1), (5, 0, 2)]:
            assert np.isclose(hx[i, j, k], values["x"][k, j, i])
            assert np.isclose(hz[i, j, k], values["z"][k, j, i])
        assert np.allclose(hfield.attrs["__Origin__"], [x[0], y[0], z[0]])


def test_write_comsol_asymmetric_field_no_scrambling():
    """The old save path scrambled values whenever the field was NOT x<->y
    symmetric; this pins the fix with a maximally asymmetric field."""
    x = np.linspace(0.0, 0.3, 8)   # asymmetric domain too
    y = np.linspace(-0.2, 0.1, 6)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    bz = 1.0 + 3.0 * xx - 2.0 * yy + 5.0 * xx * yy  # no x<->y symmetry

    path = _tmp("asym.comsol")
    write_comsol(path, {"x": x, "y": y}, {"z": bz}, components="z")

    field = Field.from_file(path)
    assert np.allclose(field.grid_values["z"], bz, atol=1e-9)
    # check a specific off-diagonal pair that x<->y symmetry would NOT mask
    i, j = 2, 4
    pts = np.array([[x[i], y[j], 0.0]])
    assert np.isclose(field(pts)[0, 2], bz[i, j], atol=1e-8)
