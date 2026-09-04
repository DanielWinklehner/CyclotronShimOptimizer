"""Pure-numpy tests for the shared residual / objective construction.

Covers the second-difference (roughness) penalty and the unification of the
DFO-LS and Nelder-Mead scoring onto ONE weighted residual vector:

  - D2 is zero on any straight line (constant / linear ramp), large on a spike.
  - build_residual_vector at zero penalty weights == the old freq-only vector.
  - the smoothness block has the right length, is zero on a smooth profile, and
    is NEVER differenced across the side/top block boundary.
  - NM scalar == ||DFO-LS residual vector||^2 (the unification invariant).
  - the residual length is constant across designs / convergence states
    (the DFO-LS requirement).

No field solve, MPI, GPU or radia -- the builder is pure numpy given a solved
isochronism dict, which these tests fabricate.
"""

import _testenv  # noqa: F401

from types import SimpleNamespace

import numpy as np

from cyclotron_optimizer.optimization.objective_function import (
    second_difference,
    field_basis_profiles,
    build_residual_vector,
    compute_objective,
)
from cyclotron_optimizer.optimization.constraints import shim_radial_free_indices

N_SEG = 14           # -> n = 15 offsets per block, D2 length 13
N = N_SEG + 1


def _cfg(smoothness_weight=0.0, smoothness_weight_side=None,
         smoothness_weight_top=None, convergence_penalty_weight=0.0,
         regularization_weight=0.0, precision=1e-4, n_seg=N_SEG,
         inner_radius_mm=50.0, outer_radius_mm=400.0,
         opt_shim_radius_min_mm=None, opt_shim_radius_max_mm=None,
         smoothness_weight_neutral=None, smoothness_weight_active=None,
         smoothness_exchange_ratio=1.0,
         nu_z_ceiling_weight=0.0, nu_z_max=0.42,
         walkinshaw_weight=0.0, walkinshaw_min_distance=0.15,
         nu_z_floor_weight=0.0, nu_z_sq_min=0.0, iso_method='gordon'):
    """Minimal duck-typed config the builder reads (no radia/config import)."""
    return SimpleNamespace(
        side_shim=SimpleNamespace(num_rad_segments=n_seg),
        pole=SimpleNamespace(inner_radius_mm=inner_radius_mm,
                             outer_radius_mm=outer_radius_mm),
        field_evaluation=SimpleNamespace(iso_method=iso_method),
        optimization=SimpleNamespace(
            smoothness_weight=smoothness_weight,
            smoothness_weight_side=smoothness_weight_side,
            smoothness_weight_top=smoothness_weight_top,
            smoothness_weight_neutral=smoothness_weight_neutral,
            smoothness_weight_active=smoothness_weight_active,
            smoothness_exchange_ratio=smoothness_exchange_ratio,
            convergence_penalty_weight=convergence_penalty_weight,
            regularization_weight=regularization_weight,
            opt_shim_radius_min_mm=opt_shim_radius_min_mm,
            opt_shim_radius_max_mm=opt_shim_radius_max_mm,
            nu_z_ceiling_weight=nu_z_ceiling_weight,
            nu_z_max=nu_z_max,
            walkinshaw_weight=walkinshaw_weight,
            walkinshaw_min_distance=walkinshaw_min_distance,
            nu_z_floor_weight=nu_z_floor_weight,
            nu_z_sq_min=nu_z_sq_min,
        ),
        simulation=SimpleNamespace(precision=precision),
    )


def _iso(freqs, tunes=None):
    freqs = np.asarray(freqs, dtype=float)
    return {'rev_frequencies_mhz': freqs,
            'std_dev_mhz': float(np.std(freqs)),
            'mean_freq_mhz': float(np.mean(freqs)),
            'tunes': tunes}


def _tunes(nu_z_sq, nu_r=1.02):
    """Gordon-shaped tunes dict. nu_z is NaN where nu_z_sq < 0, as the real one
    does -- that NaN is exactly what the barriers must survive."""
    nu_z_sq = np.asarray(nu_z_sq, dtype=float)
    nu_r = np.full_like(nu_z_sq, nu_r) if np.isscalar(nu_r) else np.asarray(nu_r, float)
    nu_z = np.where(nu_z_sq > 0.0, np.sqrt(np.clip(nu_z_sq, 0.0, None)), np.nan)
    return {'nu_z_sq': nu_z_sq, 'nu_z': nu_z, 'nu_r': nu_r,
            'nu_r_sq': nu_r ** 2, 'walkinshaw_distance': nu_r - 2.0 * nu_z}


# Vertical tunes shaped like the measured muon design: safe inside, peaking at
# 0.48 (0.02 from nu_z = 1/2) over the outer third.
_NU_Z_SAFE = np.full(36, 0.30)
_NU_Z_HOT = np.concatenate([np.full(24, 0.30), np.full(12, 0.48)])


# A representative, non-flat per-radius frequency vector (n_eval_pts = 36).
_FREQS = 42.0 + 0.1 * np.sin(np.linspace(0, 3.0, 36))
_RAMP = np.linspace(0.0, 1.0, N)                      # linear -> D2 == 0
_CONST = np.full(N, 0.37)                             # constant -> D2 == 0
_SPIKE = _RAMP.copy(); _SPIKE[N // 2] += 0.5          # mid-radius spike
_JAGGED = np.tile([0.2, 0.8], N)[:N]                  # alternating -> big D2


# ---------------------------------------------------------------------------
# second_difference
# ---------------------------------------------------------------------------

def test_d2_zero_on_constant():
    assert np.allclose(second_difference(_CONST), 0.0)


def test_d2_zero_on_linear_ramp():
    assert np.allclose(second_difference(_RAMP), 0.0)


def test_d2_length_is_len_minus_two():
    assert second_difference(_RAMP).shape == (N - 2,)
    assert second_difference(np.zeros(2)).shape == (0,)   # too short -> empty
    assert second_difference(np.zeros(1)).shape == (0,)


def test_d2_large_on_spike_and_jagged():
    # a single spike and an alternating profile both have large roughness,
    # while a smooth ramp of the SAME magnitude range has none.
    assert np.linalg.norm(second_difference(_SPIKE)) > 0.5
    assert np.linalg.norm(second_difference(_JAGGED)) > 1.0
    assert np.linalg.norm(second_difference(_RAMP)) < 1e-12


def test_d2_matches_explicit_formula():
    v = np.array([0.0, 1.0, 0.0, 2.0, -1.0])
    expected = v[:-2] - 2.0 * v[1:-1] + v[2:]
    assert np.allclose(second_difference(v), expected)


# ---------------------------------------------------------------------------
# build_residual_vector -- behavior preservation
# ---------------------------------------------------------------------------

def test_zero_weights_equals_old_freq_only_vector():
    """All penalty weights 0 -> exactly the mean-centered frequency vector."""
    cfg = _cfg(smoothness_weight=0.0, convergence_penalty_weight=0.0,
               regularization_weight=0.0)
    x = np.concatenate([_RAMP, _RAMP])
    r = build_residual_vector(_iso(_FREQS), x, ['side', 'top'],
                              None, None, True, 0.0, cfg)
    assert np.allclose(r, _FREQS - _FREQS.mean())
    assert r.shape == _FREQS.shape


def test_default_conv_weight_converged_is_bit_for_bit_objective():
    """Production default (smoothness 0, magnitude 0, conv 1.0) + converged:
    the objective ||r||^2 equals the old ||f - mean(f)||^2 exactly (the conv
    entry is present but zero, so DFO-LS sees an identical least-squares)."""
    cfg = _cfg(smoothness_weight=0.0, convergence_penalty_weight=1.0,
               regularization_weight=0.0)
    x = np.concatenate([_RAMP, _RAMP])
    r = build_residual_vector(_iso(_FREQS), x, ['side', 'top'],
                              None, None, True, 5e-5, cfg)   # misfit < precision
    old = _FREQS - _FREQS.mean()
    assert np.isclose(float(np.dot(r, r)), float(np.dot(old, old)))
    # extra entry is the (zero) convergence residual
    assert r.shape == (_FREQS.size + 1,)
    assert r[-1] == 0.0


# ---------------------------------------------------------------------------
# build_residual_vector -- smoothness block
# ---------------------------------------------------------------------------

def test_smoothness_block_length_and_zero_on_smooth_profile():
    cfg = _cfg(smoothness_weight=0.1, convergence_penalty_weight=0.0)
    x = np.concatenate([_RAMP, _RAMP])            # both blocks smooth
    r = build_residual_vector(_iso(_FREQS), x, ['side', 'top'],
                              None, None, True, 0.0, cfg)
    # flatness (36) + side D2 (13) + top D2 (13)
    assert r.shape == (36 + (N - 2) + (N - 2),)
    # the smoothness portion is zero on the smooth ramps
    assert np.allclose(r[36:], 0.0)


def test_smoothness_penalizes_jagged_side():
    cfg = _cfg(smoothness_weight=0.1, convergence_penalty_weight=0.0)
    x_smooth = np.concatenate([_RAMP, _RAMP])
    x_spike = np.concatenate([_SPIKE, _RAMP])
    r_smooth = build_residual_vector(_iso(_FREQS), x_smooth, ['side', 'top'],
                                     None, None, True, 0.0, cfg)
    r_spike = build_residual_vector(_iso(_FREQS), x_spike, ['side', 'top'],
                                    None, None, True, 0.0, cfg)
    # same flatness block, but the spiky side raises ||r||
    assert np.allclose(r_smooth[:36], r_spike[:36])
    assert np.dot(r_spike, r_spike) > np.dot(r_smooth, r_smooth) + 1e-6


def test_smoothness_never_crosses_side_top_boundary():
    """A large step BETWEEN the side and top blocks (each constant) must NOT
    register as roughness -- D2 is taken per block, not across the join."""
    cfg = _cfg(smoothness_weight=1.0, convergence_penalty_weight=0.0)
    x = np.concatenate([np.full(N, 0.1), np.full(N, 0.9)])   # 0.1 -> 0.9 jump
    r = build_residual_vector(_iso(_FREQS), x, ['side', 'top'],
                              None, None, True, 0.0, cfg)
    assert np.allclose(r[36:], 0.0)


def test_per_block_smoothness_weight_override():
    cfg = _cfg(smoothness_weight=0.0, smoothness_weight_side=0.5,
               smoothness_weight_top=None, convergence_penalty_weight=0.0)
    x = np.concatenate([_JAGGED, _JAGGED])       # both jagged
    r = build_residual_vector(_iso(_FREQS), x, ['side', 'top'],
                              None, None, True, 0.0, cfg)
    side_block = r[36:36 + (N - 2)]
    top_block = r[36 + (N - 2):]
    # side override 0.5 -> nonzero; top falls back to base 0.0 -> zero
    assert np.linalg.norm(side_block) > 0.0
    assert np.allclose(top_block, 0.0)
    assert np.allclose(side_block, 0.5 * second_difference(_JAGGED))


def test_single_block_only_emits_its_smoothness():
    cfg = _cfg(smoothness_weight=0.2, convergence_penalty_weight=0.0)
    x = _JAGGED.copy()                           # one block of length N
    r = build_residual_vector(_iso(_FREQS), x, ['top'],
                              None, None, True, 0.0, cfg)
    assert r.shape == (36 + (N - 2),)
    assert np.allclose(r[36:], 0.2 * second_difference(_JAGGED))


# ---------------------------------------------------------------------------
# field-basis (neutral / active) smoothness -- the side/top null-space penalty
# ---------------------------------------------------------------------------

# A pair of profiles curving the SAME way (legitimate <B>(r) shaping) and a pair
# curving AGAINST each other (the field-neutral mode: a dip in one cancelled by
# a bulge in the other, which isochronism cannot see).
_BUMP = 0.25 * np.sin(np.linspace(0.0, 3.0 * np.pi, N))   # the only curvature
_BASE = 0.5 * _RAMP + 0.25                                # linear -> D2 == 0
_CURVE = _BASE + _BUMP                                    # stays inside [0, 1]
_TOGETHER_S, _TOGETHER_T = _CURVE, _CURVE
_AGAINST_S, _AGAINST_T = _CURVE, _BASE - _BUMP


def test_field_basis_rotation_is_orthonormal():
    # ||D2(neutral)||^2 + ||D2(active)||^2 == ||D2(side)||^2 + ||D2(top)||^2
    for rho in (0.5, 1.0, 2.0, 3.7):
        neutral, active = field_basis_profiles(_SPIKE, _JAGGED, rho)
        rotated = (np.linalg.norm(second_difference(neutral)) ** 2
                   + np.linalg.norm(second_difference(active)) ** 2)
        blocks = (np.linalg.norm(second_difference(_SPIKE)) ** 2
                  + np.linalg.norm(second_difference(_JAGGED)) ** 2)
        assert np.isclose(rotated, blocks)


def test_field_basis_separates_common_from_anticorrelated_curvature():
    # curving TOGETHER -> all roughness lands in 'active', none in 'neutral'
    neutral, active = field_basis_profiles(_TOGETHER_S, _TOGETHER_T, 1.0)
    assert np.allclose(second_difference(neutral), 0.0)
    assert np.linalg.norm(second_difference(active)) > 0.1
    # curving AGAINST each other -> the reverse
    neutral, active = field_basis_profiles(_AGAINST_S, _AGAINST_T, 1.0)
    assert np.linalg.norm(second_difference(neutral)) > 0.1
    assert np.allclose(second_difference(active), 0.0)


def test_unset_field_basis_weights_are_bit_for_bit_per_block():
    x = np.concatenate([_SPIKE, _JAGGED])
    base = _cfg(smoothness_weight=0.1, convergence_penalty_weight=1.0)
    new = _cfg(smoothness_weight=0.1, convergence_penalty_weight=1.0,
               smoothness_weight_neutral=None, smoothness_weight_active=None)
    r_base = build_residual_vector(_iso(_FREQS), x, ['side', 'top'],
                                   None, None, True, 0.0, base)
    r_new = build_residual_vector(_iso(_FREQS), x, ['side', 'top'],
                                  None, None, True, 0.0, new)
    assert np.array_equal(r_base, r_new)


def test_equal_field_basis_weights_reduce_to_per_block_objective():
    # the rotation is orthonormal, so w_neutral == w_active == w scores exactly
    # the same as the per-block penalty at weight w (vector differs, norm does not)
    x = np.concatenate([_SPIKE, _JAGGED])
    per_block = _cfg(smoothness_weight=0.3, convergence_penalty_weight=0.0)
    rotated = _cfg(smoothness_weight=0.3, convergence_penalty_weight=0.0,
                   smoothness_weight_neutral=0.3, smoothness_weight_active=0.3)
    obj_b, _ = compute_objective(_iso(_FREQS), x, ['side', 'top'],
                                 None, None, True, 0.0, per_block)
    obj_r, _ = compute_objective(_iso(_FREQS), x, ['side', 'top'],
                                 None, None, True, 0.0, rotated)
    assert np.isclose(obj_b, obj_r)


def test_neutral_weight_taxes_the_null_mode_and_spares_common_curvature():
    # THE point of the penalty: with w_neutral >> w_active, a pair that curves
    # together is nearly free while an equally rough anti-correlated pair is not.
    cfg = _cfg(convergence_penalty_weight=0.0,
               smoothness_weight_neutral=1.0, smoothness_weight_active=0.02)
    _, s_together = compute_objective(
        _iso(_FREQS), np.concatenate([_TOGETHER_S, _TOGETHER_T]),
        ['side', 'top'], None, None, True, 0.0, cfg)
    _, s_against = compute_objective(
        _iso(_FREQS), np.concatenate([_AGAINST_S, _AGAINST_T]),
        ['side', 'top'], None, None, True, 0.0, cfg)
    # both designs carry the SAME total per-block roughness ...
    assert np.isclose(np.hypot(s_together['roughness_side'], s_together['roughness_top']),
                      np.hypot(s_against['roughness_side'], s_against['roughness_top']))
    # ... but only the anti-correlated one is penalized
    assert s_against['smoothness_residual_l2'] > 20 * s_together['smoothness_residual_l2']


def test_exchange_ratio_sets_which_pairing_is_neutral():
    # with rho = 2 the field-neutral direction is (1, -1/2): raising the side by
    # 1 normalized unit is cancelled by lowering the top by 1/2.
    neutral, _ = field_basis_profiles(_CURVE, 0.5 * _CURVE, 2.0)
    assert np.allclose(second_difference(neutral), 0.0)
    # ... and that same pair is NOT neutral at rho = 1
    neutral_1, _ = field_basis_profiles(_CURVE, 0.5 * _CURVE, 1.0)
    assert np.linalg.norm(second_difference(neutral_1)) > 0.05


def test_field_basis_ignored_for_a_single_block():
    # one block optimized -> no pair to rotate; falls back to the per-block D2
    cfg = _cfg(smoothness_weight=0.2, convergence_penalty_weight=0.0,
               smoothness_weight_neutral=5.0, smoothness_weight_active=0.0)
    r = build_residual_vector(_iso(_FREQS), _JAGGED.copy(), ['top'],
                              None, None, True, 0.0, cfg)
    assert r.shape == (36 + (N - 2),)
    assert np.allclose(r[36:], 0.2 * second_difference(_JAGGED))


def test_roughness_split_reported_without_enabling_the_penalty():
    # the neutral/active split is a DIAGNOSTIC too: readable (to decide whether
    # to switch penalties) while the per-block penalty is still the one in force.
    cfg = _cfg(smoothness_weight=0.1, convergence_penalty_weight=0.0)
    _, score = compute_objective(
        _iso(_FREQS), np.concatenate([_AGAINST_S, _AGAINST_T]),
        ['side', 'top'], None, None, True, 0.0, cfg)
    assert score['roughness_neutral'] > 0.1
    assert np.isclose(score['roughness_active'], 0.0, atol=1e-12)
    # per-block penalty still in force -> l2 built from the per-block weights
    assert np.isclose(score['smoothness_residual_l2'],
                      np.hypot(0.1 * score['roughness_side'],
                               0.1 * score['roughness_top']))


def test_field_basis_residual_length_constant_across_designs():
    cfg = _cfg(convergence_penalty_weight=1.0, regularization_weight=0.2,
               smoothness_weight_neutral=1.0, smoothness_weight_active=0.05)
    lengths = set()
    for s, t in ((_RAMP, _CONST), (_SPIKE, _JAGGED), (_AGAINST_S, _AGAINST_T)):
        r = build_residual_vector(_iso(_FREQS), np.concatenate([s, t]),
                                  ['side', 'top'], None, None, True, 1e-3, cfg)
        lengths.add(r.shape)
    assert len(lengths) == 1


def test_field_basis_works_with_full_norm_radial_subrange():
    # with a radial sub-range the rotation must act on the FULL profiles
    cfg = _cfg(convergence_penalty_weight=0.0,
               smoothness_weight_neutral=1.0, smoothness_weight_active=0.0)
    free = np.concatenate([_CURVE[3:9], _CURVE[3:9]])       # 6 free stations/block
    full = {'side': _TOGETHER_S, 'top': _TOGETHER_T}
    r = build_residual_vector(_iso(_FREQS), free, ['side', 'top'],
                              None, None, True, 0.0, cfg, full_norm=full)
    # profiles curve together over the FULL range -> neutral roughness vanishes
    assert r.shape == (36 + (N - 2),)
    assert np.allclose(r[36:], 0.0)


# ---------------------------------------------------------------------------
# convergence gate
# ---------------------------------------------------------------------------

def test_convergence_residual_zero_when_converged_positive_when_not():
    cfg = _cfg(convergence_penalty_weight=2.0, precision=1e-4)
    x = np.concatenate([_RAMP, _RAMP])
    r_ok = build_residual_vector(_iso(_FREQS), x, ['side', 'top'],
                                 None, None, True, 1e-3, cfg)  # converged flag
    assert r_ok[-1] == 0.0
    # not converged, misfit = 4x precision -> shortfall 3 -> 2*sqrt(3)
    r_bad = build_residual_vector(_iso(_FREQS), x, ['side', 'top'],
                                  None, None, False, 4e-4, cfg)
    assert np.isclose(r_bad[-1], 2.0 * np.sqrt(4e-4 / 1e-4 - 1.0))
    # length is the SAME whether converged or not (constant residual length)
    assert r_ok.shape == r_bad.shape


# ---------------------------------------------------------------------------
# magnitude regularization
# ---------------------------------------------------------------------------

def test_magnitude_block_present_only_when_weight_nonzero():
    x = np.concatenate([_RAMP, _RAMP])
    cfg0 = _cfg(regularization_weight=0.0)
    r0 = build_residual_vector(_iso(_FREQS), x, ['side', 'top'],
                               None, None, True, 0.0, cfg0)
    assert r0.shape == (36,)                     # no magnitude block
    cfg1 = _cfg(regularization_weight=0.3)
    r1 = build_residual_vector(_iso(_FREQS), x, ['side', 'top'],
                               None, None, True, 0.0, cfg1)
    assert r1.shape == (36 + x.size,)            # magnitude block appended
    assert np.allclose(r1[36:], 0.3 * x)


# ---------------------------------------------------------------------------
# unification invariant + constant length
# ---------------------------------------------------------------------------

def test_nm_scalar_equals_dfols_vector_norm_squared():
    """compute_objective()[0] == ||build_residual_vector||^2 for the SAME
    inputs/weights -- so switching optimizer cannot change the objective."""
    cfg = _cfg(smoothness_weight=0.15, smoothness_weight_side=0.4,
               convergence_penalty_weight=1.0, regularization_weight=0.05,
               precision=1e-4)
    x = np.concatenate([_SPIKE, _JAGGED])
    args = (_iso(_FREQS), x, ['side', 'top'], None, None, False, 3e-4, cfg)
    r = build_residual_vector(*args)
    obj, res = compute_objective(*args)
    assert np.isclose(obj, float(np.dot(r, r)))
    assert np.allclose(res['residual_vector'], r)


def test_compute_objective_reports_roughness_metrics():
    cfg = _cfg(smoothness_weight=0.1, convergence_penalty_weight=0.0)
    x = np.concatenate([_SPIKE, _RAMP])
    _, res = compute_objective(_iso(_FREQS), x, ['side', 'top'],
                               None, None, True, 0.0, cfg)
    assert np.isclose(res['roughness_side'],
                      float(np.linalg.norm(second_difference(_SPIKE))))
    assert np.isclose(res['roughness_top'], 0.0)
    assert res['roughness_side'] > 0.0
    # weighted smoothness residual L2 = hypot(w_side*rough_side, w_top*rough_top)
    assert np.isclose(res['smoothness_residual_l2'], 0.1 * res['roughness_side'])


def test_residual_length_constant_across_designs():
    """DFO-LS needs a fixed-length residual across evaluations."""
    cfg = _cfg(smoothness_weight=0.1, convergence_penalty_weight=1.0,
               regularization_weight=0.05)
    xs = [np.concatenate([_RAMP, _RAMP]),
          np.concatenate([_SPIKE, _JAGGED]),
          np.concatenate([_CONST, _SPIKE])]
    lengths = {build_residual_vector(_iso(_FREQS), x, ['side', 'top'],
                                     None, None, True, 0.0, cfg).size
               for x in xs}
    assert len(lengths) == 1     # identical length for every design


def test_flatness_term_matches_std_scaling():
    """||f - mean(f)||^2 == (n_eval_pts - 1) * std^2 (the note that makes the
    DFO-LS and NM flatness objectives monotonically equivalent)."""
    cfg = _cfg()
    x = np.concatenate([_RAMP, _RAMP])
    r = build_residual_vector(_iso(_FREQS), x, ['side', 'top'],
                              None, None, True, 0.0, cfg)
    n = _FREQS.size
    assert np.isclose(float(np.dot(r, r)), n * float(np.var(_FREQS)))


# ---------------------------------------------------------------------------
# full_norm: D2 over the FULL profile (radial sub-range optimization)
# ---------------------------------------------------------------------------

def test_full_norm_takes_d2_over_full_profile_not_free_params():
    """When only a sub-range is free, x_norm holds the free params but D2 must
    be taken over the FULL normalized profile passed in full_norm."""
    cfg = _cfg(smoothness_weight=0.3, convergence_penalty_weight=0.0)
    x_free = np.array([0.2, 0.5, 0.9])              # 3 free side params
    full = {'side': _JAGGED}                        # full length-N side profile
    r = build_residual_vector(_iso(_FREQS), x_free, ['side'],
                              None, None, True, 0.0, cfg, full_norm=full)
    # flatness (36) + side D2 over the FULL profile (N-2 = 13), NOT over 3 free params
    assert r.shape == (36 + (N - 2),)
    assert np.allclose(r[36:], 0.3 * second_difference(_JAGGED))


def test_full_norm_penalizes_free_frozen_transition():
    """A profile smooth within the free region but with a step at the free/frozen
    boundary has nonzero roughness under D2-over-full (the reason we use it)."""
    cfg = _cfg(smoothness_weight=1.0, convergence_penalty_weight=0.0)
    prof = np.zeros(N); prof[N // 2:] = 0.8         # step in the middle (a transition)
    x_free = prof[:3]
    r = build_residual_vector(_iso(_FREQS), x_free, ['side'],
                              None, None, True, 0.0, cfg, full_norm={'side': prof})
    assert np.linalg.norm(r[36:]) > 0.0
    assert np.allclose(r[36:], second_difference(prof))


def test_full_norm_magnitude_still_uses_free_params():
    """The magnitude penalty regularizes the FREE params (x_norm), while D2 uses
    the full profile -- the two operate on different vectors under masking."""
    cfg = _cfg(smoothness_weight=0.0, regularization_weight=0.5,
               convergence_penalty_weight=0.0)
    x_free = np.array([0.2, 0.4])
    r = build_residual_vector(_iso(_FREQS), x_free, ['side'],
                              None, None, True, 0.0, cfg, full_norm={'side': _RAMP})
    # flatness (36) + magnitude over the 2 free params (no smoothness at weight 0)
    assert r.shape == (36 + 2,)
    assert np.allclose(r[36:], 0.5 * x_free)


def test_full_norm_roughness_diagnostic_reflects_full_profile():
    cfg = _cfg(smoothness_weight=0.1, convergence_penalty_weight=0.0)
    _, res = compute_objective(_iso(_FREQS), np.array([0.1, 0.2]), ['side'],
                               None, None, True, 0.0, cfg, full_norm={'side': _SPIKE})
    assert np.isclose(res['roughness_side'],
                      float(np.linalg.norm(second_difference(_SPIKE))))


def test_full_norm_unmasked_matches_slice_path():
    """Passing full_norm equal to the x_norm block reproduces the no-mask result
    (the DFO-LS path only passes full_norm when a sub-range is actually active)."""
    cfg = _cfg(smoothness_weight=0.2, convergence_penalty_weight=1.0)
    x = np.concatenate([_SPIKE, _JAGGED])
    r_slice = build_residual_vector(_iso(_FREQS), x, ['side', 'top'],
                                    None, None, True, 3e-4, cfg)
    r_full = build_residual_vector(_iso(_FREQS), x, ['side', 'top'],
                                   None, None, True, 3e-4, cfg,
                                   full_norm={'side': _SPIKE, 'top': _JAGGED})
    assert np.allclose(r_slice, r_full)


# ---------------------------------------------------------------------------
# shim_radial_free_indices: the radial-range knob
# ---------------------------------------------------------------------------

def test_radial_indices_all_free_when_unbounded():
    cfg = _cfg(n_seg=19)                              # 20 stations, 50..400 mm
    free, r = shim_radial_free_indices(cfg)
    assert free.tolist() == list(range(20))
    assert np.isclose(r[0], 50.0) and np.isclose(r[-1], 400.0)


def test_radial_indices_max_bound_selects_inner_shims():
    # r < 150 over linspace(50,400,20) -> first 6 stations (the user's scenario)
    cfg = _cfg(n_seg=19, opt_shim_radius_max_mm=150.0)
    free, r = shim_radial_free_indices(cfg)
    assert free.tolist() == [0, 1, 2, 3, 4, 5]
    assert r[free].max() <= 150.0 and r[free[-1] + 1] > 150.0


def test_radial_indices_min_bound_selects_outer_shims():
    cfg = _cfg(n_seg=19, opt_shim_radius_min_mm=300.0)
    free, r = shim_radial_free_indices(cfg)
    assert free.size > 0
    assert r[free].min() >= 300.0
    assert all(r[i] < 300.0 for i in range(free[0]))


def test_radial_indices_band_selects_middle():
    cfg = _cfg(n_seg=19, opt_shim_radius_min_mm=150.0, opt_shim_radius_max_mm=250.0)
    free, r = shim_radial_free_indices(cfg)
    assert free.size > 0
    assert r[free].min() >= 150.0 and r[free].max() <= 250.0


def test_radial_indices_empty_band_raises():
    cfg = _cfg(n_seg=19, opt_shim_radius_max_mm=10.0)   # below inner radius 50
    try:
        shim_radial_free_indices(cfg)
    except ValueError:
        return
    raise AssertionError("an empty radial band should raise ValueError")


# ---------------------------------------------------------------------------
# Betatron-resonance barriers (nu_z ceiling / Walkinshaw / nu_z^2 floor)
#
# All OPT-IN. Motivation is measured, not hypothetical: the muon optimum ran
# nu_z to 0.4804 (0.020 from the half-integer, 0.064 from Walkinshaw) with
# nothing in the objective watching. Because vertical focusing turned out to be
# ABUNDANT here, these are ceilings -- a "maximize nu_z" reward would have
# pushed the design INTO the resonances.
# ---------------------------------------------------------------------------

_BOTH = ['side', 'top']
_XN = np.concatenate([_RAMP, _RAMP])


def test_barriers_off_by_default_is_bit_for_bit():
    """Tunes present, no weights set -> byte-identical to the no-tunes residual."""
    cfg = _cfg()
    with_tunes = build_residual_vector(_iso(_FREQS, _tunes(_NU_Z_HOT ** 2)), _XN,
                                       _BOTH, None, None, True, 0.0, cfg)
    without = build_residual_vector(_iso(_FREQS), _XN, _BOTH,
                                    None, None, True, 0.0, cfg)
    assert np.array_equal(with_tunes, without)


def test_ceiling_inactive_when_below_threshold_costs_nothing():
    """nu_z = 0.30 everywhere, ceiling 0.42 -> the block is present but all zero,
    so the objective equals the barriers-off objective exactly."""
    iso = _iso(_FREQS, _tunes(_NU_Z_SAFE ** 2))
    on = _cfg(nu_z_ceiling_weight=5.0)
    off = _cfg()
    r_on = build_residual_vector(iso, _XN, _BOTH, None, None, True, 0.0, on)
    r_off = build_residual_vector(iso, _XN, _BOTH, None, None, True, 0.0, off)
    assert np.allclose(r_on[r_off.size:], 0.0)
    assert np.isclose(float(np.dot(r_on, r_on)), float(np.dot(r_off, r_off)))


def test_ceiling_active_penalizes_only_the_excess():
    cfg = _cfg(nu_z_ceiling_weight=2.0, nu_z_max=0.42)
    iso = _iso(_FREQS, _tunes(_NU_Z_HOT ** 2))
    r = build_residual_vector(iso, _XN, _BOTH, None, None, True, 0.0, cfg)
    block = r[-len(_NU_Z_HOT):]
    assert np.allclose(block[:24], 0.0)                       # nu_z = 0.30, safe
    assert np.allclose(block[24:], 2.0 * (0.48 - 0.42))       # only the excess


def test_walkinshaw_barrier_binds_on_clearance():
    """nu_r - 2 nu_z with nu_r = 1.02: clearance 0.42 when nu_z = 0.30 (safe),
    0.06 when nu_z = 0.48 (violates a 0.15 minimum by 0.09)."""
    cfg = _cfg(walkinshaw_weight=1.0, walkinshaw_min_distance=0.15)
    r = build_residual_vector(_iso(_FREQS, _tunes(_NU_Z_HOT ** 2, nu_r=1.02)),
                              _XN, _BOTH, None, None, True, 0.0, cfg)
    block = r[-len(_NU_Z_HOT):]
    assert np.allclose(block[:24], 0.0)
    assert np.allclose(block[24:], 0.15 - (1.02 - 2.0 * 0.48))


def test_floor_fires_on_vertical_instability_without_nan():
    """THE regression that matters: nu_z is NaN wherever nu_z^2 < 0, and one NaN
    would destroy DFO-LS's interpolation model for the rest of the run."""
    nu_z_sq = np.full(36, 0.09)
    nu_z_sq[5] = -0.04                       # vertically unstable -> nu_z is NaN
    tunes = _tunes(nu_z_sq)
    assert np.isnan(tunes['nu_z'][5])        # the fixture really does carry a NaN
    cfg = _cfg(nu_z_floor_weight=3.0, nu_z_sq_min=0.01,
               nu_z_ceiling_weight=1.0, walkinshaw_weight=1.0)
    r = build_residual_vector(_iso(_FREQS, tunes), _XN, _BOTH,
                              None, None, True, 0.0, cfg)
    assert np.all(np.isfinite(r))
    floor_block = r[-36:]
    assert np.isclose(floor_block[5], 3.0 * (0.01 - (-0.04)))
    assert np.allclose(np.delete(floor_block, 5), 0.0)   # 0.09 clears the floor


def test_unstable_radius_does_not_trip_the_upper_barriers():
    """nu_z^2 < 0 means no vertical oscillation -- infinitely far BELOW a ceiling,
    not above it. The floor term is what catches that state."""
    nu_z_sq = np.full(36, 0.09)
    nu_z_sq[5] = -0.04
    cfg = _cfg(nu_z_ceiling_weight=1.0, walkinshaw_weight=1.0)
    r = build_residual_vector(_iso(_FREQS, _tunes(nu_z_sq)), _XN, _BOTH,
                              None, None, True, 0.0, cfg)
    assert np.all(np.isfinite(r))
    assert np.allclose(r[len(_FREQS):], 0.0)


def test_residual_length_constant_whether_barriers_bind():
    """DFO-LS requires a fixed-length residual; a hinge must stay present (at 0)
    when inactive rather than dropping out of the vector."""
    cfg = _cfg(nu_z_ceiling_weight=1.0, walkinshaw_weight=1.0, nu_z_floor_weight=1.0)
    safe = build_residual_vector(_iso(_FREQS, _tunes(_NU_Z_SAFE ** 2)), _XN,
                                 _BOTH, None, None, True, 0.0, cfg)
    hot = build_residual_vector(_iso(_FREQS, _tunes(_NU_Z_HOT ** 2)), _XN,
                                _BOTH, None, None, True, 0.0, cfg)
    assert safe.size == hot.size
    assert np.dot(hot, hot) > np.dot(safe, safe)      # the hot design scores worse


def test_barrier_without_tunes_raises_rather_than_silently_dropping():
    cfg = _cfg(nu_z_ceiling_weight=1.0, iso_method='circle')
    try:
        build_residual_vector(_iso(_FREQS), _XN, _BOTH, None, None, True, 0.0, cfg)
    except ValueError as exc:
        assert 'circle' in str(exc)
        return
    raise AssertionError("enabled barriers with no tunes must raise, not no-op")


def test_margins_reported_even_when_barriers_are_off():
    """The margins are diagnostics, not penalties -- readable from any gordon run,
    which is how this class of problem gets caught next time."""
    _, score = compute_objective(_iso(_FREQS, _tunes(_NU_Z_HOT ** 2, nu_r=1.02)),
                                 _XN, _BOTH, None, None, True, 0.0, _cfg())
    assert np.isclose(score['nu_z_peak'], 0.48)
    assert np.isclose(score['walkinshaw_min'], 1.02 - 2.0 * 0.48)
    assert np.isclose(score['nu_z_sq_min'], 0.30 ** 2)
    assert score['tune_penalty_l2'] == 0.0            # off -> no penalty


def test_margins_are_nan_without_tunes():
    _, score = compute_objective(_iso(_FREQS), _XN, _BOTH, None, None,
                                 True, 0.0, _cfg())
    assert np.isnan(score['nu_z_peak'])
    assert np.isnan(score['walkinshaw_min'])


def test_objective_equals_residual_norm_squared_with_barriers_on():
    """The NM-scalar == ||DFO-LS vector||^2 unification must survive the new block."""
    cfg = _cfg(smoothness_weight=0.1, convergence_penalty_weight=1.0,
               nu_z_ceiling_weight=2.0, walkinshaw_weight=1.5, nu_z_floor_weight=3.0)
    iso = _iso(_FREQS, _tunes(_NU_Z_HOT ** 2))
    r = build_residual_vector(iso, _XN, _BOTH, None, None, False, 5e-4, cfg)
    obj, score = compute_objective(iso, _XN, _BOTH, None, None, False, 5e-4, cfg)
    assert np.isclose(obj, float(np.dot(r, r)))
    assert np.isclose(score['tune_penalty_l2'],
                      float(np.linalg.norm(r[len(_FREQS) + (N - 2) * 2:-1])))
