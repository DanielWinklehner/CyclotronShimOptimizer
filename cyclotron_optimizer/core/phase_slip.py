"""Accumulated RF phase slip -- the physically meaningful isochronism metric.

Implements sections 3 and 5.3 of isochronism_reporting_spec.md. The headline
figure of merit for an isochronous cyclotron is NOT the scatter of the
revolution frequency (a sign-alternating error and a monotone ramp with the
same sigma differ by an order of magnitude in accumulated phase) but the RF
phase slip integrated along the acceleration:

    delta(E)   = (f_rev(E) - f_RF/h) / (f_RF/h)          relative iso error
    sin Phi(E) = sin Phi_0 + 2 pi h * int[ delta / dE0 dE' ]

The arcsin is not cosmetic -- it encodes the cos(Phi) feedback in the energy
gain. |sin Phi| > 1 means the beam stalls and turns around: that is reported
as an explicit result (stall energy), never raised as a domain error.

The RF frequency is a GAUGE: f_RF = f_RF0 * (1 + eps) shifts delta -> delta -
eps, and minimizing max|Phi| over eps is a well-behaved 1-D convex minimax
(spec 3.4) that typically halves the peak excursion for free. Zeroing the
mean of delta is NOT the same operation and gives a worse answer.

Everything here is pure numpy on a solved isochronism dict + the RFConfig,
so it is unit-tested without a field solve (test/test_phase_slip.py), same
policy as the residual builder.
"""

import numpy as np
from typing import Dict, Optional


def delta_from_frequencies(f_rev_mhz, f_rf_mhz: float, h: int) -> np.ndarray:
    """Relative isochronism error delta(E), referenced directly to the RF.

    Deliberately NOT constructed from B_avg vs a fitted B_iso(r): computing the
    ideal isochronous field is its own error-prone fitting problem, and folding
    it into the metric contaminates the result (spec 3.1).
    """
    f_ref = float(f_rf_mhz) / int(h)
    return (np.asarray(f_rev_mhz, dtype=float) - f_ref) / f_ref


def _cumtrapz0(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid with a leading 0 (same length as x)."""
    out = np.zeros_like(np.asarray(x, dtype=float))
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def smooth_fit_delta(energies_mev, delta, deg: Optional[int] = None):
    """(delta_smooth, sigma_resid): Chebyshev fit + RMS of the residual.

    Physical delta(E) has no high-spatial-frequency content -- the field is a
    smooth function of radius by construction -- so the fit residual IS the
    solver/mesh noise estimate (spec 5.3), and the INTEGRATION should use the
    smooth fit: integrating the grass accumulates noise into Phi.
    """
    E = np.asarray(energies_mev, dtype=float)
    d = np.asarray(delta, dtype=float)
    if E.size < 8:
        return d.copy(), 0.0
    if deg is None:
        deg = int(np.clip(E.size // 6, 3, 12))
    fit = np.polynomial.chebyshev.Chebyshev.fit(E, d, deg)
    smooth = fit(E)
    return smooth, float(np.sqrt(np.mean((d - smooth) ** 2)))


def phase_slip(energies_mev, delta, *, h: int, de0_mev_per_turn,
               phi0_deg: float = 0.0, eps: float = 0.0) -> Dict:
    """Integrate the longitudinal equations; closed form of spec 3.3.

    :param de0_mev_per_turn: nominal energy gain per turn AT CREST [MeV];
        scalar or per-sample array.
    :param eps: RF gauge shift, f_RF = f_RF0 * (1 + eps) -> delta - eps.
    :return: dict with sin_phi, phi_deg (nan past a stall), turns n(E),
        phi_max_deg / phi_p2p_deg over the un-stalled range, stall flag +
        stall_energy_mev.
    """
    E = np.asarray(energies_mev, dtype=float)
    d = np.asarray(delta, dtype=float) - float(eps)
    dE0 = np.broadcast_to(np.asarray(de0_mev_per_turn, dtype=float), E.shape)
    if np.any(dE0 <= 0):
        raise ValueError("de0_mev_per_turn must be positive")
    if E.size >= 2 and np.any(np.diff(E) <= 0):
        raise ValueError("energies_mev must be strictly increasing")

    sin_phi0 = np.sin(np.deg2rad(float(phi0_deg)))
    I = _cumtrapz0(d / dE0, E)
    sin_phi = sin_phi0 + 2.0 * np.pi * int(h) * I

    stalled = np.abs(sin_phi) > 1.0
    stall = bool(np.any(stalled))
    stall_energy = float(E[np.argmax(stalled)]) if stall else None
    ok = ~stalled
    # arcsin only where physical; past the stall the beam has turned around and
    # Phi(E) is undefined -- nan, not a domain error (spec 3.3).
    phi = np.full_like(sin_phi, np.nan)
    phi[ok] = np.rad2deg(np.arcsin(sin_phi[ok]))

    # Turn number by quadrature with the same Phi (cos Phi energy-gain feedback).
    cos_phi = np.where(ok, np.cos(np.deg2rad(np.where(ok, phi, 0.0))), np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        turns = _cumtrapz0(1.0 / (dE0 * cos_phi), E)

    phi_ok = phi[ok]
    return {
        'sin_phi': sin_phi,
        'phi_deg': phi,
        'turns': turns,
        'phi_max_deg': float(np.max(np.abs(phi_ok))) if phi_ok.size else float('nan'),
        'phi_p2p_deg': float(phi_ok.max() - phi_ok.min()) if phi_ok.size else float('nan'),
        'n_turns': float(turns[ok][-1]) if phi_ok.size else float('nan'),
        'stall': stall,
        'stall_energy_mev': stall_energy,
    }


def optimize_rf_gauge(energies_mev, delta, *, h: int, de0_mev_per_turn,
                      phi0_deg: float = 0.0) -> float:
    """Best RF shift eps minimizing max|Phi| (spec 3.4).

    sin Phi(E; eps) = sin Phi_0 + 2 pi h [I(E) - eps J(E)] with J monotone, so
    max_E |sin Phi| is convex piecewise-linear in eps -- a 1-D unimodal
    minimization. arcsin is monotone odd, so minimizing max|sin Phi| minimizes
    max|Phi| wherever the beam does not stall.
    """
    from scipy.optimize import minimize_scalar

    E = np.asarray(energies_mev, dtype=float)
    d = np.asarray(delta, dtype=float)
    dE0 = np.broadcast_to(np.asarray(de0_mev_per_turn, dtype=float), E.shape)
    c = 2.0 * np.pi * int(h)
    a = np.sin(np.deg2rad(float(phi0_deg))) + c * _cumtrapz0(d / dE0, E)
    b = c * _cumtrapz0(1.0 / dE0, E)

    mask = b > 1e-30
    if not np.any(mask):
        return 0.0
    ratios = a[mask] / b[mask]
    lo, hi = float(ratios.min()), float(ratios.max())
    if lo == hi:
        return lo
    res = minimize_scalar(lambda e: float(np.max(np.abs(a - e * b))),
                          bounds=(lo, hi), method='bounded',
                          options={'xatol': 1e-12})
    return float(res.x)


def phase_slip_report(iso: Dict, config) -> Optional[Dict]:
    """Full spec-style phase-slip result from one isochronism evaluation.

    Returns None (silently) when the RF description is absent or the iso dict
    carries no energies -- old configs and fabricated test dicts are unaffected.
    Integration uses the SMOOTH fit of delta (spec 5.3.4); sigma_resid is
    reported alongside as the numerical-noise diagnostic, the only legitimate
    surviving use of a scatter statistic.
    """
    rf = getattr(config, 'rf', None)
    if rf is None or not getattr(rf, 'harmonic_h', None) \
            or not getattr(rf, 'de0_kev_per_turn', None):
        return None
    energies = iso.get('energies_mev') if hasattr(iso, 'get') else None
    freqs = iso.get('rev_frequencies_mhz') if hasattr(iso, 'get') else None
    if energies is None or freqs is None:
        return None

    E = np.asarray(energies, dtype=float)
    f = np.asarray(freqs, dtype=float)
    if E.size < 3 or E.size != f.size:
        return None
    # Ensure ascending energy (radii are evaluated inner -> outer already).
    order = np.argsort(E)
    E, f = E[order], f[order]
    if np.any(np.diff(E) <= 0):
        return None

    h = int(rf.harmonic_h)
    f_rf = float(rf.f_rf_mhz) if rf.f_rf_mhz else \
        h * float(config.optimization.target_frequency_mhz)
    dE0 = float(rf.de0_kev_per_turn) * 1e-3   # MeV/turn
    phi0 = float(rf.phi0_deg or 0.0)
    budget = float(rf.phi_budget_deg or 10.0)

    # Clip to the stated acceleration range (metadata -> hard integration limits).
    e_lo = float(rf.e_inj_mev) if rf.e_inj_mev is not None else -np.inf
    e_hi = float(rf.e_final_mev) if rf.e_final_mev is not None else np.inf
    keep = (E >= e_lo) & (E <= e_hi)
    if keep.sum() < 3:
        return None
    E, f = E[keep], f[keep]

    delta_raw = delta_from_frequencies(f, f_rf, h)
    delta_smooth, sigma_resid = smooth_fit_delta(E, delta_raw)

    nominal = phase_slip(E, delta_smooth, h=h, de0_mev_per_turn=dE0, phi0_deg=phi0)
    eps = optimize_rf_gauge(E, delta_smooth, h=h, de0_mev_per_turn=dE0, phi0_deg=phi0)
    optimized = phase_slip(E, delta_smooth, h=h, de0_mev_per_turn=dE0,
                           phi0_deg=phi0, eps=eps)

    n_turns = optimized['n_turns']
    delta_allowed = (budget / (360.0 * h * n_turns)
                     if np.isfinite(n_turns) and n_turns > 0 else float('nan'))
    phi_max = optimized['phi_max_deg']
    return {
        'harmonic_h': h,
        'f_rf_mhz_nominal': f_rf,
        'f_rf_mhz_optimized': f_rf * (1.0 + eps),
        'gauge_eps': eps,
        'de0_kev_per_turn': dE0 * 1e3,
        'phi0_deg': phi0,
        'phi_budget_deg': budget,
        'phi_max_deg': phi_max,
        'phi_p2p_deg': optimized['phi_p2p_deg'],
        'phi_max_deg_nominal': nominal['phi_max_deg'],
        'margin': (budget / phi_max if phi_max and np.isfinite(phi_max)
                   and phi_max > 0 else float('nan')),
        'n_turns': n_turns,
        'delta_rms': float(np.sqrt(np.mean(delta_smooth ** 2))),
        'delta_max_abs': float(np.max(np.abs(delta_smooth))),
        'delta_allowed': delta_allowed,
        'sigma_resid': sigma_resid,
        'noise_floor_ok': (bool(sigma_resid < 0.2 * delta_allowed)
                           if np.isfinite(delta_allowed) else None),
        'stall_detected': optimized['stall'],
        'stall_energy_mev': optimized['stall_energy_mev'],
        'energies_mev': E,
        'phi_deg': optimized['phi_deg'],
        'phi_deg_nominal': nominal['phi_deg'],
        'delta': delta_smooth,
        'delta_raw': delta_raw,
    }


def headline_line(report: Dict, particle: str = "", extra: str = "") -> str:
    """One-line summary for logs (spec 6.3)."""
    if report is None:
        return "phase slip: RF description not configured"
    return (f"{particle} h={report['harmonic_h']} "
            f"f_RF {report['f_rf_mhz_optimized']:.4f} MHz "
            f"dE0 {report['de0_kev_per_turn']:.0f} keV | "
            f"{report['n_turns']:.1f} turns | "
            f"Phi_max {report['phi_max_deg']:.2f} deg "
            f"(budget {report['phi_budget_deg']:.0f}, "
            f"margin {report['margin']:.1f}x) | "
            f"delta_rms {report['delta_rms']:.2e} "
            f"(allowed {report['delta_allowed']:.2e}) | "
            f"sigma_resid {report['sigma_resid']:.1e}"
            f"{' | STALL at %.2f MeV' % report['stall_energy_mev'] if report['stall_detected'] else ''}"
            f"{' ' + extra if extra else ''}")
