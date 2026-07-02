"""Tests for the symmetry metadata on geometry.components (radia stubbed)."""

import _testenv  # noqa: F401

import numpy as np

import geometry.components as components
from _radia_stub import RadiaStub
from geometry.components import BaseRadiaComponent, MagnetizedComponent
from geometry.symmetry import canonical_symmetry_set, collect_field_symmetries

SYMS = [
    ("perp", [0, 0, 0], [1, -1, 0]),
    ("perp", [0, 0, 0], [1, 0, 0]),
    ("perp", [0, 0, 0], [0, 1, 0]),
    ("para", [0, 0, 0], [0, 0, 1]),
]

_REAL_RAD = components.rad


def _with_stub():
    stub = RadiaStub()
    components.rad = stub
    return stub


def _restore():
    components.rad = _REAL_RAD


def test_base_component_stores_symmetries_without_radia_calls():
    stub = _with_stub()
    try:
        comp = BaseRadiaComponent(42, symmetries=SYMS)
        assert canonical_symmetry_set(comp.symmetries) == canonical_symmetry_set(SYMS)
        # metadata only: no radia transform calls
        assert not any(c[0].startswith("TrfZer") for c in stub.calls)
    finally:
        _restore()


def test_declare_symmetries_does_not_propagate_to_children():
    stub = _with_stub()
    try:
        child_a = BaseRadiaComponent(11)
        child_b = BaseRadiaComponent(12)
        container = BaseRadiaComponent.containerize([child_a, child_b])
        container.declare_symmetries(SYMS)

        assert canonical_symmetry_set(container.symmetries) == canonical_symmetry_set(SYMS)
        # the pair is symmetric; the members are NOT (e.g. single coil at +z)
        assert child_a.symmetries == []
        assert child_b.symmetries == []
        assert not any(c[0].startswith("TrfZer") for c in stub.calls)
    finally:
        _restore()


def test_apply_symmetry_applies_and_stores():
    stub = _with_stub()
    try:
        comp = MagnetizedComponent(21)
        comp.apply_symmetry(SYMS)
        assert canonical_symmetry_set(comp.symmetries) == canonical_symmetry_set(SYMS)
        trf_calls = [c for c in stub.calls if c[0].startswith("TrfZer")]
        assert len(trf_calls) == 4
        assert sum(c[0] == "TrfZerPara" for c in trf_calls) == 1
        assert sum(c[0] == "TrfZerPerp" for c in trf_calls) == 3
    finally:
        _restore()


def test_collect_field_symmetries_on_component_tree():
    _with_stub()
    try:
        iron = MagnetizedComponent(31)
        iron.apply_symmetry(SYMS)

        coil_lo = BaseRadiaComponent(32)
        coil_hi = BaseRadiaComponent(33)
        coils = BaseRadiaComponent.containerize([coil_lo, coil_hi])
        coils.declare_symmetries(SYMS)

        channel = MagnetizedComponent(34)  # no symmetries (breaks the 8-fold)

        top_symmetric = BaseRadiaComponent.containerize([iron, coils])
        assert canonical_symmetry_set(collect_field_symmetries(top_symmetric)) == \
            canonical_symmetry_set(SYMS)
    finally:
        _restore()

    _with_stub()
    try:
        iron = MagnetizedComponent(41)
        iron.apply_symmetry(SYMS)
        coils = BaseRadiaComponent(42, symmetries=SYMS)
        channel = MagnetizedComponent(43)

        top_mixed = BaseRadiaComponent.containerize([iron, coils, channel])
        assert collect_field_symmetries(top_mixed) == []
    finally:
        _restore()


def test_dispose_clears_symmetries():
    _with_stub()
    try:
        comp = BaseRadiaComponent(51, symmetries=SYMS)
        comp.dispose()
        assert comp.symmetries == []
        assert comp.id is None
    finally:
        _restore()
