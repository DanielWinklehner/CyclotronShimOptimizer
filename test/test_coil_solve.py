"""Tests for the inner coil-current match (solve_coil_for_target_frequency).

The point of these is the STOPPING CONDITION. `coil_match_tol_mhz` used to be
converted to a tolerance in AMPS and handed to scipy as a step/bracket
tolerance, so the frequency residual it names was never actually tested: the
solve ran past it and returned an answer 17-1700x tighter than asked, at the
cost of extra full field solves (each one a real magnet solve in production).

A fake solver stands in for radia -- it just returns a monotone f(I), counts
calls, and is deliberately mildly SATURATING (sublinear) so the secant has real
curvature to work with. No radia, no GPU, no MPI.
"""

import _testenv  # noqa: F401

from types import SimpleNamespace

import numpy as np

from cyclotron_optimizer.optimization.objective_function import (
    solve_coil_for_target_frequency,
    CoilBracketError,
)

TARGET = 42.0
ROOT = 13054.8
BRACKET = (8000.0, 20000.0)


class FakeSolver:
    """Counts solves and reports a monotone, mildly saturating f(I)."""

    def __init__(self, root=ROOT, exponent=0.90):
        self.radii_mm = [100.0, 200.0, 300.0]
        self.root, self.exponent = root, exponent
        self.currents = []

    def f_of(self, current):
        return TARGET * (float(current) / self.root) ** self.exponent

    def resolve_at_current(self, current):
        self.currents.append(float(current))
        return None, float(current), True, 1e-6      # (_, bz, converged, misfit)

    @property
    def n_solves(self):
        return len(self.currents)


def _cfg():
    return SimpleNamespace(field_evaluation=SimpleNamespace(iso_method='gordon'))


def _patch_iso(monkey_target, solver):
    """compute_isochronism is called with bz == the current the fake returned."""
    import cyclotron_optimizer.optimization.objective_function as of

    def fake_iso(method, bz, radii, config, species, **kw):
        f = solver.f_of(bz)
        return {'mean_freq_mhz': f, 'std_dev_mhz': 0.005,
                'rev_frequencies_mhz': np.full(len(radii), f), 'tunes': None}

    monkey_target.append((of, 'compute_isochronism', of.compute_isochronism))
    of.compute_isochronism = fake_iso


def _restore(saved):
    for mod, name, orig in saved:
        setattr(mod, name, orig)


def _solve(solver, *, ftol_mhz, xtol_A, seed):
    saved = []
    _patch_iso(saved, solver)
    try:
        return solve_coil_for_target_frequency(
            solver, _cfg(), None, TARGET, BRACKET,
            rank=0, comm=None, verbosity=0,
            xtol_A=xtol_A, ftol_mhz=ftol_mhz, seed_current=seed)
    finally:
        _restore(saved)


# ---------------------------------------------------------------------------
# the residual stop
# ---------------------------------------------------------------------------

def test_seed_already_within_tolerance_costs_one_solve():
    """THE regression. A seed 20 A off is already within 0.1 MHz, so one solve
    (the new geometry has to be evaluated once) is the correct total cost. The
    old x-tolerance path spent four."""
    s = FakeSolver()
    current, iso, converged, misfit = _solve(
        s, ftol_mhz=0.1, xtol_A=23.81, seed=ROOT + 20.0)
    assert s.n_solves == 1
    assert abs(iso['mean_freq_mhz'] - TARGET) <= 0.1
    assert current == ROOT + 20.0          # the EVALUATED point, not an extrapolation
    assert converged is True and misfit == 1e-6


def test_returned_iso_belongs_to_the_returned_current():
    """The early exit must hand back the iso of the point it stopped at -- if it
    returned a different current's iso the flatness would be scored at the wrong
    working point, silently."""
    s = FakeSolver()
    current, iso, _, _ = _solve(s, ftol_mhz=0.1, xtol_A=23.81, seed=ROOT + 20.0)
    assert np.isclose(iso['mean_freq_mhz'], s.f_of(current))


def test_far_seed_still_converges_and_is_cheaper_than_x_tolerance():
    s_res = FakeSolver()
    cur_res, iso_res, _, _ = _solve(s_res, ftol_mhz=0.1, xtol_A=23.81, seed=ROOT * 1.30)
    s_x = FakeSolver()
    cur_x, _, _, _ = _solve(s_x, ftol_mhz=None, xtol_A=23.81, seed=ROOT * 1.30)
    assert abs(iso_res['mean_freq_mhz'] - TARGET) <= 0.1
    assert s_res.n_solves < s_x.n_solves          # strictly fewer magnet solves
    assert abs(cur_res - ROOT) < 200 and abs(cur_x - ROOT) < 200


def test_tight_tolerance_is_honored_not_just_the_loose_one():
    """The final production match must still land on frequency_tolerance_mhz."""
    s = FakeSolver()
    _, iso, _, _ = _solve(s, ftol_mhz=0.001, xtol_A=0.238, seed=ROOT * 1.05)
    assert abs(iso['mean_freq_mhz'] - TARGET) <= 0.001


def test_tight_costs_more_solves_than_loose():
    loose = FakeSolver()
    _solve(loose, ftol_mhz=0.1, xtol_A=23.81, seed=ROOT * 1.05)
    tight = FakeSolver()
    _solve(tight, ftol_mhz=0.001, xtol_A=0.238, seed=ROOT * 1.05)
    assert tight.n_solves > loose.n_solves


def test_ftol_none_keeps_the_old_x_tolerance_behavior():
    """Opt-out path: no residual test, and the returned root is an extrapolated
    point that needed its own extra resolve."""
    s = FakeSolver()
    current, _, _, _ = _solve(s, ftol_mhz=None, xtol_A=23.81, seed=ROOT * 1.05)
    assert abs(current - ROOT) < 50.0
    assert current == s.currents[-1]        # the final extra resolve happened
    assert s.n_solves >= 4


def test_no_seed_falls_back_to_brentq_and_still_stops_on_residual():
    s = FakeSolver()
    _, iso, _, _ = _solve(s, ftol_mhz=0.1, xtol_A=23.81, seed=None)
    assert abs(iso['mean_freq_mhz'] - TARGET) <= 0.1


# ---------------------------------------------------------------------------
# the bracket guard
# ---------------------------------------------------------------------------

def test_unbracketed_target_raises_a_diagnosable_error():
    """Previously a bare scipy ValueError escaped from the middle of the
    optimizer; now it names the bracket, the measured frequencies and the fix."""
    s = FakeSolver(root=ROOT * 4.0)          # target unreachable inside the bracket
    try:
        _solve(s, ftol_mhz=0.1, xtol_A=23.81, seed=None)
    except CoilBracketError as exc:
        msg = str(exc)
        assert '8000' in msg and '20000' in msg
        assert 'MHz' in msg
        assert 'coil_current_max_A' in msg    # both ends below target -> raise the max
        return
    raise AssertionError("an unbracketed target must raise CoilBracketError")


def test_bracket_error_names_the_min_when_both_ends_are_high():
    s = FakeSolver(root=ROOT / 8.0)
    try:
        _solve(s, ftol_mhz=0.1, xtol_A=23.81, seed=None)
    except CoilBracketError as exc:
        assert 'coil_current_min_A' in str(exc)
        return
    raise AssertionError("expected CoilBracketError")
