"""Pure-numpy tests for the accumulated-RF-phase-slip module.

Implements the automatable subset of isochronism_reporting_spec.md section 10:
  test 5 (gauge invariance -- THE important one: the reported number is a
          property of the magnet, not of the arbitrary normalization),
  test 6 (harmonic scaling, Phi proportional to h),
  test 7 (stall detection reports, never throws),
plus analytic closed-form checks and the sigma_resid noise-floor split.
No field solve, no radia, no MPI.
"""

import _testenv  # noqa: F401

from types import SimpleNamespace

import numpy as np

from cyclotron_optimizer.core.phase_slip import (
    delta_from_frequencies,
    phase_slip,
    optimize_rf_gauge,
    smooth_fit_delta,
    phase_slip_report,
    headline_line,
)

# The muon machine's RF numbers (user-confirmed 2026-07-29).
H = 4
F_RF = 168.0            # = h * 42 MHz
DE0 = 0.180             # MeV per turn at crest
E = np.linspace(0.03, 5.0, 120)


def _rf_cfg(**over):
    rf = dict(harmonic_h=H, f_rf_mhz=F_RF, de0_kev_per_turn=180.0,
              phi0_deg=0.0, phi_budget_deg=10.0, e_inj_mev=None, e_final_mev=None)
    rf.update(over)
    return SimpleNamespace(
        rf=SimpleNamespace(**rf),
        optimization=SimpleNamespace(target_frequency_mhz=42.0))


def test_delta_referenced_to_rf():
    # f_rev exactly f_RF/h everywhere -> delta identically zero.
    assert np.allclose(delta_from_frequencies(np.full(50, F_RF / H), F_RF, H), 0.0)
    # 42 kHz high on 42 MHz -> delta = 1e-3.
    d = delta_from_frequencies(np.array([42.042]), F_RF, H)
    assert np.isclose(d[0], 1e-3)


def test_constant_delta_matches_closed_form():
    """Constant delta, constant dE0: sin Phi = 2 pi h delta (E - E0) / dE0."""
    d0 = 5e-5
    res = phase_slip(E, np.full_like(E, d0), h=H, de0_mev_per_turn=DE0)
    expected = 2.0 * np.pi * H * d0 * (E - E[0]) / DE0
    assert np.allclose(res['sin_phi'], expected)
    assert not res['stall']
    # turn count ~ (E_end - E_0)/dE0 for tiny phase excursion (cos Phi ~ 1).
    assert np.isclose(res['n_turns'], (E[-1] - E[0]) / DE0, rtol=1e-3)


def test_alternating_error_accumulates_far_less_than_ramp():
    """The reason sigma is not a figure of merit: same RMS, ~an order of
    magnitude different accumulated phase."""
    rng = np.linspace(-1, 1, E.size)
    ramp = 1e-4 * rng
    alternating = 1e-4 * np.sign(np.sin(20 * np.pi * rng))  # same RMS scale
    phi_ramp = phase_slip(E, ramp, h=H, de0_mev_per_turn=DE0)['phi_max_deg']
    phi_alt = phase_slip(E, alternating, h=H, de0_mev_per_turn=DE0)['phi_max_deg']
    assert phi_ramp > 5 * phi_alt


def test_spec5_gauge_invariance():
    """Scale B (hence f_rev) by 1+eps and f_RF by 1+eps: after gauge
    re-optimization max|Phi| is unchanged (property of the magnet)."""
    rng = np.random.default_rng(3)
    f_rev = F_RF / H * (1.0 + 2e-4 * np.cumsum(rng.standard_normal(E.size)) / 10)
    scale = 1.0 + 3e-3

    def optimized_phi_max(f_rev_i, f_rf_i):
        d = delta_from_frequencies(f_rev_i, f_rf_i, H)
        eps = optimize_rf_gauge(E, d, h=H, de0_mev_per_turn=DE0)
        return phase_slip(E, d, h=H, de0_mev_per_turn=DE0, eps=eps)['phi_max_deg']

    a = optimized_phi_max(f_rev, F_RF)
    b = optimized_phi_max(f_rev * scale, F_RF * scale)
    assert np.isclose(a, b, rtol=1e-6)


def test_spec6_harmonic_scaling():
    """At fixed small delta, Phi is proportional to h."""
    d = 1e-5 * np.sin(np.linspace(0, 3, E.size))
    phi1 = phase_slip(E, d, h=1, de0_mev_per_turn=DE0)['phi_max_deg']
    phi4 = phase_slip(E, d, h=4, de0_mev_per_turn=DE0)['phi_max_deg']
    assert np.isclose(phi4, 4.0 * phi1, rtol=1e-3)   # arcsin ~ linear at small arg


def test_spec7_stall_reported_not_raised():
    d = np.full_like(E, 5e-3)          # enormous coherent error -> stall
    res = phase_slip(E, d, h=H, de0_mev_per_turn=DE0)
    assert res['stall'] is True
    assert res['stall_energy_mev'] is not None
    assert 0.03 < res['stall_energy_mev'] < 5.0
    assert np.isnan(res['phi_deg'][-1])      # undefined past the stall
    assert np.isfinite(res['phi_deg'][0])


def test_gauge_beats_mean_zeroing():
    """Spec 3.4: minimax over eps is NOT 'zero the mean of delta'. On an
    asymmetric profile the gauge optimum is strictly better."""
    d = 1e-4 * np.concatenate([np.zeros(E.size - 20), np.ones(20)])  # end kick
    d_meanzero = d - d.mean()
    phi_meanzero = phase_slip(E, d_meanzero, h=H, de0_mev_per_turn=DE0)['phi_max_deg']
    eps = optimize_rf_gauge(E, d, h=H, de0_mev_per_turn=DE0)
    phi_opt = phase_slip(E, d, h=H, de0_mev_per_turn=DE0, eps=eps)['phi_max_deg']
    assert phi_opt < phi_meanzero * 0.95


def test_gauge_balances_extremes():
    """At the minimax optimum the positive and negative Phi extremes match."""
    d = 2e-4 * np.sin(np.linspace(0, 2.0, E.size)) ** 2   # one-signed input
    eps = optimize_rf_gauge(E, d, h=H, de0_mev_per_turn=DE0)
    phi = phase_slip(E, d, h=H, de0_mev_per_turn=DE0, eps=eps)['phi_deg']
    assert np.isclose(phi.max(), -phi.min(), rtol=0.05)


def test_sigma_resid_separates_noise_from_physics():
    smooth = 2e-4 * np.sin(np.linspace(0, 2, E.size))
    noise = 1e-5 * np.random.default_rng(0).standard_normal(E.size)
    fit, sigma = smooth_fit_delta(E, smooth + noise)
    assert 0.5e-5 < sigma < 2e-5                  # recovers the noise scale
    # the fit tracks the physics to well under its amplitude (the deg-12
    # Chebyshev absorbs a little edge noise, so ~the noise scale, not below it)
    assert np.max(np.abs(fit - smooth)) < 2e-5


def test_report_end_to_end_muon_numbers():
    """Sanity on the real machine numbers: sigma = 0.0094 MHz-scale wiggles at
    180 keV/turn, h=4 -> ~27 turns, and Phi_max within the 10-degree budget."""
    rng = np.random.default_rng(1)
    f_rev = 42.0 + 0.0094 * rng.standard_normal(E.size)   # sigma = 0.0094 MHz
    iso = {'energies_mev': E, 'rev_frequencies_mhz': f_rev}
    rep = phase_slip_report(iso, _rf_cfg())
    assert rep is not None
    assert 20 < rep['n_turns'] < 35                       # ~ (5-0.03)/0.180
    assert np.isfinite(rep['phi_max_deg'])
    assert rep['delta_allowed'] > 0
    assert rep['stall_detected'] is False
    assert 'Phi_max' in headline_line(rep, 'mu+')
    # gauge must not LOSE to the nominal frequency
    assert rep['phi_max_deg'] <= rep['phi_max_deg_nominal'] + 1e-9


def test_report_disabled_without_rf_config():
    iso = {'energies_mev': E, 'rev_frequencies_mhz': np.full(E.size, 42.0)}
    assert phase_slip_report(iso, _rf_cfg(harmonic_h=None)) is None
    assert phase_slip_report(iso, _rf_cfg(de0_kev_per_turn=None)) is None
    # and without energies (fabricated test iso dicts)
    assert phase_slip_report({'rev_frequencies_mhz': np.ones(9)}, _rf_cfg()) is None


def test_report_clips_to_stated_energy_range():
    iso = {'energies_mev': E, 'rev_frequencies_mhz': np.full(E.size, 42.0)}
    rep = phase_slip_report(iso, _rf_cfg(e_final_mev=2.0))
    assert rep['energies_mev'].max() <= 2.0
    rep_full = phase_slip_report(iso, _rf_cfg())
    assert rep_full['n_turns'] > rep['n_turns']
