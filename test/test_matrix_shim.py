"""Tests for response-matrix shimming (optimization/matrix_shim.py).

The driver takes plain callables, so the ENTIRE algorithm -- probing, the
regularized linear step, Broyden updates, stopping -- runs here against
synthetic frequency models with zero radia/MPI/GPU. The consistency test
pinning smoothness_operator to build_residual_vector row-for-row is what
guarantees the matrix path optimizes the SAME objective as DFO-LS.
"""

import _testenv  # noqa: F401

from types import SimpleNamespace

import numpy as np

from cyclotron_optimizer.optimization.matrix_shim import (
    MatrixShimOptions,
    broyden_update,
    cosine_basis,
    mean_center_rows,
    probe_directions,
    run_matrix_shim,
    second_difference_matrix,
    smoothness_operator,
    solve_step,
)
from cyclotron_optimizer.optimization.objective_function import (
    build_residual_vector,
    compute_objective,
    second_difference,
)

N_SEG = 14
N = N_SEG + 1
BLOCKS = ['side', 'top']
N_PARAMS = 2 * N
M_EVAL = 36     # frequency samples


def _cfg(**over):
    opt = dict(smoothness_weight=0.0, smoothness_weight_side=None,
               smoothness_weight_top=None, smoothness_weight_neutral=1.0,
               smoothness_weight_active=0.05, smoothness_exchange_ratio=0.7,
               convergence_penalty_weight=0.0, regularization_weight=0.0,
               opt_shim_radius_min_mm=None, opt_shim_radius_max_mm=None,
               nu_z_ceiling_weight=0.0, nu_z_max=0.42, walkinshaw_weight=0.0,
               walkinshaw_min_distance=0.15, nu_z_floor_weight=0.0,
               nu_z_sq_min=0.0)
    opt.update(over)
    return SimpleNamespace(
        side_shim=SimpleNamespace(num_rad_segments=N_SEG),
        pole=SimpleNamespace(inner_radius_mm=50.0, outer_radius_mm=400.0),
        field_evaluation=SimpleNamespace(iso_method='gordon'),
        optimization=SimpleNamespace(**opt),
        simulation=SimpleNamespace(precision=1e-4),
    )


def _iso(freqs):
    freqs = np.asarray(freqs, dtype=float)
    return {'rev_frequencies_mhz': freqs, 'std_dev_mhz': float(np.std(freqs)),
            'mean_freq_mhz': float(np.mean(freqs)), 'tunes': None}


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

def test_second_difference_matrix_matches_function():
    v = np.random.default_rng(0).random(N)
    assert np.allclose(second_difference_matrix(N) @ v, second_difference(v))


def test_cosine_basis_shape_and_smoothness():
    B = cosine_basis(20, 8)
    assert B.shape == (20, 8)
    assert np.allclose(B[:, 0], 1.0)                      # k=0 constant
    D = second_difference_matrix(20)
    rough = [np.linalg.norm(D @ B[:, k]) for k in range(8)]
    assert all(rough[k] <= rough[k + 1] + 1e-12 for k in range(7))  # low-pass order


def test_smoothness_operator_matches_residual_builder_field_basis():
    """THE consistency guarantee: S @ x reproduces the smoothness rows of
    build_residual_vector exactly, so both optimizers score smoothness alike."""
    cfg = _cfg()
    S = smoothness_operator(BLOCKS, N, cfg)
    x = np.random.default_rng(1).random(N_PARAMS)
    f = 42.0 + 0.01 * np.sin(np.linspace(0, 3, M_EVAL))
    r = build_residual_vector(_iso(f), x, BLOCKS, None, None, True, 0.0, cfg)
    assert np.allclose(S @ x, r[M_EVAL:])   # everything after the flatness block


def test_smoothness_operator_matches_residual_builder_per_block():
    cfg = _cfg(smoothness_weight_neutral=None, smoothness_weight_active=None,
               smoothness_weight=0.1, regularization_weight=0.02)
    S = smoothness_operator(BLOCKS, N, cfg)
    x = np.random.default_rng(2).random(N_PARAMS)
    f = 42.0 + 0.01 * np.sin(np.linspace(0, 3, M_EVAL))
    r = build_residual_vector(_iso(f), x, BLOCKS, None, None, True, 0.0, cfg)
    assert np.allclose(S @ x, r[M_EVAL:])


def test_probe_directions_full_respects_bounds():
    x0 = np.zeros(N_PARAMS); x0[3] = 0.98                  # hugging the top bound
    U = probe_directions(x0, 'full', BLOCKS, N, 0.05, 8)
    assert U.shape == (N_PARAMS, N_PARAMS)
    assert np.all(x0[:, None] + U >= -1e-12)
    assert np.all(x0[:, None] + U <= 1.0 + 1e-12)
    assert U[3, 3] < 0                                     # flipped away from 1


def test_probe_directions_reduced_count():
    x0 = np.full(N_PARAMS, 0.5)
    U = probe_directions(x0, 'reduced', BLOCKS, N, 0.05, 8)
    assert U.shape == (N_PARAMS, 16)                       # 8 modes x 2 blocks


def test_broyden_recovers_perturbed_jacobian():
    rng = np.random.default_rng(3)
    G_true = rng.standard_normal((M_EVAL, 6))
    J = G_true + 0.3 * rng.standard_normal((M_EVAL, 6))    # stale estimate
    for _ in range(40):
        dc = rng.standard_normal(6) * 0.1
        J = broyden_update(J, dc, G_true @ dc)             # exact secant pairs
    dc = rng.standard_normal(6)
    assert (np.linalg.norm((J - G_true) @ dc)
            < 0.5 * np.linalg.norm(0.3 * rng.standard_normal((M_EVAL, 6)) @ dc))


# ---------------------------------------------------------------------------
# the driver on synthetic machines
# ---------------------------------------------------------------------------

class SyntheticMachine:
    """f(x) = f0 + G x (+ optional quadratic) with counted 'solves'."""

    def __init__(self, seed=0, curvature=0.0):
        rng = np.random.default_rng(seed)
        # Smooth low-rank-ish sensitivity, like the real gap-filtered machine.
        left = np.stack([np.sin((k + 1) * np.linspace(0, np.pi, M_EVAL))
                         for k in range(6)], axis=1)
        right = rng.standard_normal((6, N_PARAMS))
        self.G = 0.02 * left @ right
        self.f0 = 42.0 + 0.05 * np.sin(np.linspace(0, 2, M_EVAL))
        self.curvature = curvature
        self.rng = rng
        self.n_eval = 0
        self.n_probe = 0

    def f(self, x):
        f = self.f0 + self.G @ x
        if self.curvature:
            f = f + self.curvature * (self.G @ (x * x))
        return f

    def evaluate_fn(self, x):
        self.n_eval += 1
        return 13000.0, _iso(self.f(x)), True, 1e-6

    def probe_fn(self, x):
        self.n_probe += 1
        return self.f(x)


def _drive(machine, cfg, x0, options):
    S = smoothness_operator(BLOCKS, N, cfg)

    def score_fn(x, iso, converged, misfit):
        return compute_objective(iso, np.asarray(x, float), BLOCKS,
                                 None, None, converged, misfit, cfg)

    return run_matrix_shim(machine.evaluate_fn, machine.probe_fn, score_fn,
                           x0, S, BLOCKS, N, options)


def test_linear_machine_converges_in_few_evaluations():
    """On a truly linear machine the relaxed steps head straight for the
    regularized optimum in single-digit evaluation counts. The floor is NOT
    zero: the box bounds and the smoothness penalty leave an irreducible
    residual, so assert a large fixed fraction of the objective is removed."""
    cfg = _cfg()
    m = SyntheticMachine(seed=4)
    x0 = np.full(N_PARAMS, 0.5)
    res = _drive(m, cfg, x0, MatrixShimOptions(mode='full', relaxation=0.8,
                                               max_outer=10, ftol_rel=0.01))
    x_start_obj = compute_objective(_iso(m.f(x0)), x0, BLOCKS, None, None,
                                    True, 1e-6, cfg)[0]
    assert res['best_obj'] < 0.5 * x_start_obj           # >half the objective gone
    assert m.n_eval <= 10
    assert m.n_probe == N_PARAMS + 1                     # columns + base


def test_reduced_mode_is_cheaper_and_nearly_as_good():
    cfg = _cfg()
    m_full = SyntheticMachine(seed=5)
    m_red = SyntheticMachine(seed=5)
    x0 = np.full(N_PARAMS, 0.5)
    res_full = _drive(m_full, cfg, x0, MatrixShimOptions(mode='full'))
    res_red = _drive(m_red, cfg, x0, MatrixShimOptions(mode='reduced', basis_size=8))
    assert m_red.n_probe < 0.6 * m_full.n_probe          # 17 vs 31 probes
    assert res_red['best_obj'] < 1.5 * res_full['best_obj'] + 1e-9


def test_broyden_helps_on_a_nonlinear_machine():
    cfg = _cfg()
    x0 = np.full(N_PARAMS, 0.5)
    opts = dict(mode='reduced', basis_size=8, relaxation=0.7, max_outer=8,
                ftol_rel=0.001)
    m_plain = SyntheticMachine(seed=6, curvature=0.4)
    res_plain = _drive(m_plain, cfg, x0, MatrixShimOptions(**opts, broyden=False))
    m_broy = SyntheticMachine(seed=6, curvature=0.4)
    res_broy = _drive(m_broy, cfg, x0, MatrixShimOptions(**opts, broyden=True))
    assert res_broy['best_obj'] <= res_plain['best_obj'] * 1.001


def test_iterates_stay_inside_bounds():
    cfg = _cfg()
    m = SyntheticMachine(seed=7)
    x0 = np.concatenate([np.full(N, 0.02), np.full(N, 0.97)])   # near both walls
    res = _drive(m, cfg, x0, MatrixShimOptions(mode='full', relaxation=0.8))
    assert np.all(res['best_x'] >= 0.0) and np.all(res['best_x'] <= 1.0)


def test_reported_objective_is_true_not_model_predicted():
    """best_obj must equal compute_objective at best_x on the REAL machine."""
    cfg = _cfg()
    m = SyntheticMachine(seed=8, curvature=0.3)
    res = _drive(m, cfg, np.full(N_PARAMS, 0.5),
                 MatrixShimOptions(mode='reduced'))
    obj_check = compute_objective(_iso(m.f(res['best_x'])), res['best_x'],
                                  BLOCKS, None, None, True, 1e-6, cfg)[0]
    assert np.isclose(res['best_obj'], obj_check, rtol=1e-9)


def test_step_removes_neutral_jag_only_through_the_penalty():
    """The mechanism test for the whole enterprise. Build a flatness Jacobian
    that is EXACTLY blind to the field-neutral direction (like the real
    machine: both shims are levers on the same <B>(r)), and start from a
    design carrying an anti-correlated station jag -- the measured muon null
    mode. The penalty-free step CANNOT remove the jag (it is invisible to
    flatness and lstsq returns the minimum-norm move); the field-basis
    penalty step removes almost all of it."""
    rho = 0.7
    k = 1.0 / np.hypot(1.0, rho)
    rng = np.random.default_rng(9)
    # response only to the ACTIVE combination (rho*s + t): neutral-blind.
    L = np.stack([np.sin((j + 1) * np.linspace(0, np.pi, M_EVAL))
                  for j in range(4)], axis=1) @ rng.standard_normal((4, N))
    P_active = np.zeros((N, N_PARAMS))
    P_active[:, :N] = rho * k * np.eye(N)
    P_active[:, N:] = k * np.eye(N)
    G = 0.02 * L @ P_active
    J_c = mean_center_rows(G)

    # smooth base + an anti-correlated jag = pure NEUTRAL roughness
    jag = 0.15 * np.tile([1.0, -1.0], N)[:N]
    x = np.concatenate([np.full(N, 0.5) + jag, np.full(N, 0.5) - rho * jag])
    x = np.clip(x, 0.0, 1.0)
    r_flat = 0.01 * rng.standard_normal(M_EVAL)
    U = np.eye(N_PARAMS)

    def neutral_rough(v):
        neutral = (v[:N] - rho * v[N:]) * k
        return np.linalg.norm(second_difference(neutral))

    S_on = smoothness_operator(BLOCKS, N, _cfg())          # neutral 1.0 / active .05
    S_off = smoothness_operator(BLOCKS, N, _cfg(smoothness_weight_neutral=0.0,
                                                smoothness_weight_active=0.0))
    x_on = solve_step(J_c, r_flat, S_on, U, x, alpha=1.0)
    x_off = solve_step(J_c, r_flat, S_off, U, x, alpha=1.0)

    assert neutral_rough(x_on) < 0.2 * neutral_rough(x)    # penalty irons the jag
    assert neutral_rough(x_off) > 0.9 * neutral_rough(x)   # blindness leaves it
