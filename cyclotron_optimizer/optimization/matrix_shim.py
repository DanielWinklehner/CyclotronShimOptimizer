"""Response-matrix shimming (isochronism_reporting_spec.md section 7).

Probe the sensitivity matrix M = d f_rev(r_i) / d x_j ONCE around the starting
design at FIXED coil current, then iterate

    regularized linear step  ->  one coil-matched FE evaluation

instead of running a derivative-free optimizer for hundreds of evaluations.
Two probing modes:

  full     one column per free parameter (n_params probe solves). The classic
           unit-cut matrix.
  reduced  one column per SMOOTH basis vector per block (~basis_size columns).
           Physics justification: the midplane field low-passes the shim
           profile on the gap scale (station-alternating modes attenuate by
           ~e^-7), so M is effectively LOW-RANK -- its high-spatial-frequency
           columns are near zero and probing them individually is wasted
           solves. ~10 smooth modes capture essentially all field authority.

Optional Broyden: matrix shimming with a frozen M is Gauss-Newton with a stale
Jacobian; every iteration's evaluation hands back a free secant condition, so
a rank-1 (good Broyden) update lets M track saturation drift at zero extra
solves.

The linear step minimizes the SAME weighted objective as DFO-LS -- the
flatness rows are the linearized mean-centered frequency residual and the
smoothness/magnitude rows come from ``smoothness_operator``, which is
unit-tested to match ``build_residual_vector`` exactly. Switching optimizers
cannot change what "good" means.

The driver (``run_matrix_shim``) takes plain callables and never touches MPI
or radia, so the full algorithm is unit-tested with synthetic models
(test/test_matrix_shim.py); the MPI/solver wiring lives in optimizer.py.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from cyclotron_optimizer.optimization.objective_function import (
    _smoothness_weights,
    _coupling_params,
)


# ---------------------------------------------------------------------------
# Pure building blocks
# ---------------------------------------------------------------------------

def second_difference_matrix(n: int) -> np.ndarray:
    """(n-2, n) matrix D with (D v)[i] = v[i] - 2 v[i+1] + v[i+2]."""
    if n < 3:
        return np.zeros((0, n))
    D = np.zeros((n - 2, n))
    idx = np.arange(n - 2)
    D[idx, idx] = 1.0
    D[idx, idx + 1] = -2.0
    D[idx, idx + 2] = 1.0
    return D


def cosine_basis(n_stations: int, n_modes: int) -> np.ndarray:
    """(n_stations, n_modes) smooth DCT-II-style basis, unit inf-norm columns.

    Column k is cos(pi k (i + 1/2) / n): k=0 the constant, k=1 a half-cosine
    ramp, ... -- the natural low-pass ordering for profiles the gap can
    actually transmit to the midplane.
    """
    n_modes = int(min(n_modes, n_stations))
    i = np.arange(n_stations)
    B = np.stack([np.cos(np.pi * k * (i + 0.5) / n_stations)
                  for k in range(n_modes)], axis=1)
    return B / np.max(np.abs(B), axis=0)


def smoothness_operator(blocks, n: int, config) -> np.ndarray:
    """Linear operator S with S @ x_norm == the smoothness (+ magnitude) rows
    of ``build_residual_vector`` for the same config weights.

    Covers the LINEAR residual rows only: per-block or field-basis D2 blocks
    and the optional magnitude regularization. The convergence gate and the
    resonance barriers are state-dependent (nonlinear) rows -- they are
    excluded from the linear model and only enter through the true objective
    scored at every accepted iterate. Consistency with the builder is
    unit-tested row for row.
    """
    n_params = len(blocks) * n
    slices = {b: slice(i * n, (i + 1) * n) for i, b in enumerate(blocks)}
    D = second_difference_matrix(n)
    rows: List[np.ndarray] = []

    def _select(block) -> np.ndarray:
        P = np.zeros((n, n_params))
        P[:, slices[block]] = np.eye(n)
        return P

    coupling = _coupling_params(config)
    if coupling is not None and 'side' in slices and 'top' in slices:
        w_neutral, w_active, rho = coupling
        k = 1.0 / np.hypot(1.0, rho)
        Ps, Pt = _select('side'), _select('top')
        if w_neutral:
            rows.append(w_neutral * D @ ((Ps - rho * Pt) * k))
        if w_active:
            rows.append(w_active * D @ ((rho * Ps + Pt) * k))
    else:
        w_side, w_top = _smoothness_weights(config)
        if 'side' in slices and w_side:
            rows.append(w_side * D @ _select('side'))
        if 'top' in slices and w_top:
            rows.append(w_top * D @ _select('top'))

    w_mag = getattr(config.optimization, 'regularization_weight', 0.0) or 0.0
    if w_mag:
        rows.append(w_mag * np.eye(n_params))

    return np.vstack(rows) if rows else np.zeros((0, n_params))


def mean_center_rows(F: np.ndarray) -> np.ndarray:
    """Project each column onto the mean-centered subspace: (I - 11^T/m) F.

    The flatness residual is f - mean(f), so its Jacobian is the mean-centered
    raw-frequency Jacobian; centering also removes the (uniform-ish) shift the
    nested coil re-match applies between evaluations, which is what makes
    Broyden secant pairs taken ACROSS coil re-matches valid.
    """
    return F - F.mean(axis=0, keepdims=True)


def probe_directions(x0: np.ndarray, mode: str, blocks, n: int,
                     delta: float, basis_size: int) -> np.ndarray:
    """(n_params, n_probes) matrix of probe steps FROM x0, bound-safe.

    full:    delta * e_j, sign flipped away from whichever bound x0_j hugs.
    reduced: delta * (per-block smooth basis), then clipped into [0, 1]; the
             ACTUAL applied step (after clipping) is what gets returned, and
             the Jacobian columns are measured along these actual directions,
             so bound clipping degrades resolution, never correctness.
    """
    x0 = np.asarray(x0, dtype=float)
    n_params = x0.size
    if mode == 'full':
        signs = np.where(x0 + delta <= 1.0, 1.0, -1.0)
        Dm = np.diag(signs * delta)
    else:
        B = cosine_basis(n, basis_size)
        cols = []
        for bi in range(len(blocks)):
            for k in range(B.shape[1]):
                d = np.zeros(n_params)
                d[bi * n:(bi + 1) * n] = delta * B[:, k]
                cols.append(d)
        Dm = np.stack(cols, axis=1)
    # Clip the landing points into the box; keep the ACTUAL applied steps.
    applied = np.clip(x0[:, None] + Dm, 0.0, 1.0) - x0[:, None]
    keep = np.linalg.norm(applied, axis=0) > 1e-12
    return applied[:, keep]


def broyden_update(J: np.ndarray, dc: np.ndarray, dr: np.ndarray) -> np.ndarray:
    """Good Broyden rank-1 secant update: J += (dr - J dc) dc^T / (dc^T dc)."""
    dc = np.asarray(dc, dtype=float)
    dr = np.asarray(dr, dtype=float)
    denom = float(dc @ dc)
    if denom < 1e-30:
        return J
    return J + np.outer(dr - J @ dc, dc) / denom


def solve_step(J_c: np.ndarray, r_flat: np.ndarray, S: np.ndarray,
               U: np.ndarray, x: np.ndarray, alpha: float) -> np.ndarray:
    """One regularized linear step; returns the new x (clipped into [0,1]).

    Minimizes over coefficients c (steps along the probe directions U):

        || J_c c + r_flat ||^2  +  || S (x + U c) ||^2

    which is the linearization of the shared DFO-LS objective (flatness rows
    linearized through the measured Jacobian, smoothness rows EXACT -- they
    are linear in x). Solved unconstrained, then the relaxed step alpha*U*c is
    clipped into the box; with alpha in [0.6, 0.8] (spec 7.2: several
    iterations with margin -- over-cutting is unrecoverable) and a warm start,
    active bounds at the solution survive the clip.
    """
    A = np.vstack([J_c, S @ U])
    b = -np.concatenate([r_flat, S @ x])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.clip(x + float(alpha) * (U @ c), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@dataclass
class MatrixShimOptions:
    mode: str = 'full'              # 'full' | 'reduced'
    basis_size: int = 8             # smooth modes per block (reduced)
    probe_delta: float = 0.05       # probe step, normalized units
    relaxation: float = 0.7         # alpha per step
    max_outer: int = 10             # max linear-step iterations
    ftol_rel: float = 0.01          # stop when best improves less than this
    broyden: bool = False           # secant-update the Jacobian per iterate


def run_matrix_shim(evaluate_fn: Callable, probe_fn: Callable,
                    score_fn: Callable, x0: np.ndarray, S: np.ndarray,
                    blocks, n: int, options: MatrixShimOptions,
                    log_fn: Optional[Callable] = None) -> Dict:
    """The full matrix-shimming iteration, MPI- and radia-free.

    :param evaluate_fn: x_norm -> (coil_A, iso_dict, converged, misfit); the
        expensive coil-matched evaluation (collective, wired by the caller).
    :param probe_fn: x_norm -> f_rev array [MHz]; a FIXED-coil solve (raw iron
        response -- the coil match would only shift the mean, which the
        centering removes anyway, so probing skips it and saves solves).
    :param score_fn: (x_norm, iso, converged, misfit) -> (objective, results);
        the shared compute_objective, wired by the caller. Called on every
        accepted iterate so the reported objective is always the TRUE one,
        never the linear model's prediction.
    :param x0: starting point, normalized [0, 1].
    :param S: smoothness operator from ``smoothness_operator`` (same config).
    :return: dict with best_x, best_obj, best_coil, history (list of per-
        iterate dicts), n_evaluations, n_probes, jacobian (J_c, final).
    """
    log = log_fn or (lambda *a, **k: None)
    x = np.asarray(x0, dtype=float).copy()
    history = []

    # --- starting point: one coil-matched evaluation -----------------------
    coil, iso, converged, misfit = evaluate_fn(x)
    obj, results = score_fn(x, iso, converged, misfit)
    f = np.asarray(iso['rev_frequencies_mhz'], dtype=float)
    best = {'x': x.copy(), 'obj': float(obj), 'coil': coil, 'results': results}
    history.append({'kind': 'eval', 'obj': float(obj), 'coil': coil})
    log('eval', x=x, obj=obj, coil=coil, results=results, iso=iso,
        converged=converged, misfit=misfit)

    # --- probe the Jacobian around x0 at fixed coil ------------------------
    U = probe_directions(x, options.mode, blocks, n,
                         options.probe_delta, options.basis_size)
    f_base = np.asarray(probe_fn(x), dtype=float)
    log('probe_base', x=x)
    cols = []
    for j in range(U.shape[1]):
        f_j = np.asarray(probe_fn(x + U[:, j]), dtype=float)
        cols.append(f_j - f_base)
        log('probe', x=x + U[:, j], column=j, n_columns=U.shape[1])
    # d f / d c along the probe directions, mean-centered (flatness Jacobian).
    J_c = mean_center_rows(np.stack(cols, axis=1))
    n_probes = U.shape[1] + 1

    # --- iterate: linear step -> true evaluation ---------------------------
    stale = 0
    for k in range(int(options.max_outer)):
        r_flat = f - f.mean()
        x_new = solve_step(J_c, r_flat, S, U, x, options.relaxation)
        if np.linalg.norm(x_new - x) < 1e-12:
            log('stop', reason='step collapsed to zero')
            break

        coil, iso, converged, misfit = evaluate_fn(x_new)
        obj_new, results = score_fn(x_new, iso, converged, misfit)
        f_new = np.asarray(iso['rev_frequencies_mhz'], dtype=float)
        history.append({'kind': 'eval', 'obj': float(obj_new), 'coil': coil})
        log('eval', x=x_new, obj=obj_new, coil=coil, results=results, iso=iso,
            converged=converged, misfit=misfit)

        if options.broyden:
            # Free secant pair from the step just taken. Coefficients of the
            # actual step in the probe-direction basis (U full column rank).
            dc, *_ = np.linalg.lstsq(U, x_new - x, rcond=None)
            J_c = broyden_update(J_c, dc, mean_center_rows(
                (f_new - f)[:, None]).ravel())

        improved = obj_new < best['obj'] * (1.0 - float(options.ftol_rel))
        if obj_new < best['obj']:
            best = {'x': x_new.copy(), 'obj': float(obj_new), 'coil': coil,
                    'results': results}
        x, f = x_new, f_new
        if improved:
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                log('stop', reason=f'no {options.ftol_rel:.0%} improvement '
                                   f'for 2 iterations')
                break

    return {'best_x': best['x'], 'best_obj': best['obj'],
            'best_coil': best['coil'], 'best_results': best['results'],
            'history': history, 'n_evaluations': len(history),
            'n_probes': n_probes, 'jacobian': J_c, 'directions': U}
