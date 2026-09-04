"""Isochronism methods: 'gordon' (closed orbits), 'gordon_analytic' (formulas),
dispatch names and the dense-grid requirement.

Pure numpy + PyCentralRegion.closed_orbit (no radia, no GPU). The synthetic
field is the exactly isochronous 4-sector flutter field of
scripts/test_isochronism_compare.py:

    B(r, theta) = B_c gamma_iso(r) (1 + a cos 4 theta),
    gamma_iso = 1 / sqrt(1 - (omega0 r / c)^2),  omega0 = (q/m) B_c

sampled on a mirror-folded octant [0, pi/4] (endpoint included) the way
field_calculator.get_field_rz_dense delivers it.
"""
import _testenv  # noqa: F401

from types import SimpleNamespace

import numpy as np

from cyclotron_optimizer.core.isochronicity import (CLIGHT, ISO_METHODS,
                                                    compute_isochronism)
from cyclotron_optimizer.core.species import IonSpecies
from cyclotron_optimizer.simulation.field_calculator import (RZFieldGrid,
                                                             dense_radii_mm)

SPECIES = IonSpecies('H2_1+')
CFG = SimpleNamespace(field_evaluation=SimpleNamespace(
    num_points_circle=720, use_symmetry=True, iso_method='gordon',
    eo_radial_step_mm=2.0, eo_radial_margin=0.15, eo_steps_per_turn=3600))
B_C, A4, N_SEC = 1.0, 0.05, 4
RADII_MM = np.array([500.0, 1000.0, 1500.0, 2000.0])


def _omega0():
    return abs(SPECIES.q_over_m) * B_C          # rad/s


def _b(r_m, theta):
    g = 1.0 / np.sqrt(1.0 - (_omega0() * r_m / CLIGHT) ** 2)
    return B_C * g * (1.0 + A4 * np.cos(N_SEC * theta))


def _dense_octant(radii_mm=RADII_MM, step_mm=2.0):
    r_mm = dense_radii_mm(radii_mm, step_mm=step_mm, margin=0.15)
    n_ang = int(round(720 * (np.pi / 4) / (2 * np.pi)))          # 90 over the octant
    theta = np.linspace(0.0, np.pi / 4, n_ang + 1, endpoint=True)  # + mirror plane sample
    return RZFieldGrid(bz=_b(r_mm[:, None] * 1e-3, theta[None, :]), angles=theta,
                       radii_mm=r_mm, period=np.pi / 2, mirror=True)


def _sparse_octant(radii_mm=RADII_MM):
    theta = np.linspace(0.0, np.pi / 4, 90, endpoint=False)
    return RZFieldGrid(bz=_b(radii_mm[:, None] * 1e-3, theta[None, :]), angles=theta,
                       radii_mm=radii_mm, period=np.pi / 2, mirror=True)


def test_method_names_and_dispatch():
    assert ISO_METHODS == ('circle', 'gordon', 'gordon_analytic', 'seo')
    iso = compute_isochronism('gordon_analytic', _sparse_octant(), RADII_MM, CFG, SPECIES)
    assert iso['method'] == 'gordon_analytic'
    assert iso['tunes'] is not None and 'nu_z_sq' in iso['tunes']
    try:
        compute_isochronism('bogus', _sparse_octant(), RADII_MM, CFG, SPECIES)
    except ValueError as exc:
        assert 'gordon_analytic' in str(exc)
    else:
        raise AssertionError('unknown method must raise')


def test_gordon_needs_dense_grid():
    try:
        compute_isochronism('gordon', _sparse_octant(), RADII_MM, CFG, SPECIES)
    except ValueError as exc:
        assert 'gordon_analytic' in str(exc) and 'dense' in str(exc).lower()
    else:
        raise AssertionError("'gordon' on a sparse rz grid must raise")


def test_gordon_closed_orbits_match_analytic_to_second_order():
    """At a = 0.05 the first-order formulas are exact to O(a^4) ~ 1e-5."""
    iso_eo = compute_isochronism('gordon', _dense_octant(), RADII_MM, CFG, SPECIES)
    iso_an = compute_isochronism('gordon_analytic', _sparse_octant(), RADII_MM, CFG, SPECIES)
    assert iso_eo['method'] == 'gordon'
    f_eo = iso_eo['rev_frequencies_mhz']
    f_an = iso_an['rev_frequencies_mhz']
    assert np.all(np.isfinite(f_eo))
    assert np.max(np.abs(f_eo / f_an - 1.0)) < 3e-5, (f_eo, f_an)
    # the average field is exactly isochronous: closed-orbit f_rev is flat to
    # the O(a^2) scalloping shift (~1e-4) and the shift is the same at all radii
    f0 = _omega0() / (2 * np.pi) * 1e-6
    assert np.max(np.abs(f_eo / f0 - 1.0)) < 2e-4
    assert np.ptp(f_eo) / f0 < 3e-5
    t = iso_eo['tunes']
    for key in ('r_mm', 'nu_r', 'nu_z', 'nu_r_sq', 'nu_z_sq', 'walkinshaw_distance',
                'flutter', 'field_index', 'closed'):
        assert key in t, key
    assert t['closed'].all()
    # nu_r ~ gamma, nu_z^2 = -k + F N^2/(N^2-1) (+O(a^4)); F = a^2/2, k = beta^2 gamma^2
    r_m = RADII_MM * 1e-3
    g = 1.0 / np.sqrt(1.0 - (_omega0() * r_m / CLIGHT) ** 2)
    k = (_omega0() * r_m / CLIGHT) ** 2 * g ** 2
    nu_z_sq = -k + 0.5 * A4 ** 2 * N_SEC ** 2 / (N_SEC ** 2 - 1.0)
    assert np.allclose(t['nu_r'], g, atol=3e-3), (t['nu_r'], g)
    assert np.allclose(t['nu_z_sq'], nu_z_sq, atol=3e-4), (t['nu_z_sq'], nu_z_sq)
    assert np.allclose(t['flutter'], 0.5 * A4 ** 2, rtol=1e-3)


def test_gordon_accepts_full_circle_grid():
    """No symmetry fold (extraction channel present): full circle, endpoint=False."""
    r_mm = dense_radii_mm(RADII_MM[:2], step_mm=2.0, margin=0.15)
    theta = np.linspace(0.0, 2 * np.pi, 720, endpoint=False)
    rz = RZFieldGrid(bz=_b(r_mm[:, None] * 1e-3, theta[None, :]), angles=theta, radii_mm=r_mm)
    iso = compute_isochronism('gordon', rz, RADII_MM[:2], CFG, SPECIES)
    iso_oct = compute_isochronism('gordon', _dense_octant(RADII_MM[:2]), RADII_MM[:2], CFG, SPECIES)
    assert np.allclose(iso['rev_frequencies_mhz'], iso_oct['rev_frequencies_mhz'], rtol=1e-7)


def test_dense_radii_cover_margin():
    r = dense_radii_mm([75.0, 2000.0], step_mm=2.0, margin=0.15)
    assert r[0] <= 0.85 * 75.0 + 1e-9 and r[-1] >= 1.15 * 2000.0 - 1e-9
    assert np.allclose(np.diff(r), 2.0)
