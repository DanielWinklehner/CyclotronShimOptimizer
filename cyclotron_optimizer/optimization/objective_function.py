"""Objective function for three-phase optimization."""

import numpy as np
from typing import Tuple, Dict

from cyclotron_optimizer.config_io.config import CyclotronConfig
from cyclotron_optimizer.simulation.field_calculator import evaluate_radii_parallel
from cyclotron_optimizer.core.species import IonSpecies
from cyclotron_optimizer.core.isochronicity import compute_isochronism
from cyclotron_optimizer.geometry.pole_shape import PoleShape
from scipy.optimize import minimize_scalar, brentq, newton


# ---------------------------------------------------------------------------
# Shared residual / objective construction (used by BOTH optimizers)
#
# Both the joint DFO-LS path (minimizes ||residual vector||^2) and the
# three-phase Nelder-Mead path (minimizes a scalar) score a design with the
# SAME weighted residual vector and the SAME config weights, so switching
# optimizers cannot change what "good" means. See build_residual_vector /
# compute_objective below.
#
# Everything here is PURE NUMPY given a solved isochronism dict, so it is
# unit-tested without a field solve (test/test_residual_builder.py).
# ---------------------------------------------------------------------------

def second_difference(v: np.ndarray) -> np.ndarray:
    """Discrete second difference D2(v)[i] = v[i-1] - 2 v[i] + v[i+1].

    This is the machinable, SVD-truncation-style roughness measure used in
    cyclotron shimming: it is ZERO for any straight line (constant or linear
    ramp) and grows with jaggedness, so it penalizes spikes without penalizing
    a smooth profile of any magnitude. Length len(v) - 2 (empty for len < 3).
    """
    v = np.asarray(v, dtype=float)
    if v.size < 3:
        return np.zeros(0, dtype=float)
    return v[:-2] - 2.0 * v[1:-1] + v[2:]


def _smoothness_weights(config) -> "tuple[float, float]":
    """(w_side, w_top) roughness weights: per-block override else the base."""
    opt = config.optimization
    base = getattr(opt, 'smoothness_weight', 0.0) or 0.0
    w_side = getattr(opt, 'smoothness_weight_side', None)
    w_top = getattr(opt, 'smoothness_weight_top', None)
    return (base if w_side is None else float(w_side),
            base if w_top is None else float(w_top))


def _block_slices(x_norm: np.ndarray, blocks, n: int) -> Dict[str, np.ndarray]:
    """Map each optimized block name -> its length-n normalized [0,1] slice.

    ``x_norm`` is the concatenation of the optimized blocks, in ``blocks``
    order (side before top, matching how the DFO-LS param vector is built).
    """
    x_norm = np.asarray(x_norm, dtype=float)
    out, i = {}, 0
    for b in (blocks or []):
        out[b] = x_norm[i:i + n]
        i += n
    return out


def _convergence_shortfall(converged, misfit, config) -> float:
    """max(0, misfit/precision - 1); 0 when converged or misfit unknown.

    Grows smoothly with the relaxation shortfall so the continuous convergence
    gate never silently accepts an unconverged field (see the DFO-LS/NM notes).
    """
    prec = getattr(config.simulation, 'precision', None)
    if converged or misfit is None or not prec or prec <= 0:
        return 0.0
    return max(0.0, float(misfit) / float(prec) - 1.0)


def _smooth_profiles(x_norm, blocks, n, full_norm):
    """Normalized per-block profile to take D2 over.

    Without ``full_norm`` this is just the optimized-params slice of x_norm
    (the whole block). With a radial sub-range only a subset of the block is
    free, so ``full_norm`` carries the FULL normalized profile (free params
    inserted into the frozen base); taking D2 over the full profile penalizes
    roughness INCLUDING the transition into the frozen neighbours, which is
    what keeps the optimized sub-range blending smoothly into the fixed part.
    """
    slices = _block_slices(x_norm, blocks, n)
    out = {}
    for name in ('side', 'top'):
        if full_norm is not None and name in full_norm:
            out[name] = np.asarray(full_norm[name], dtype=float)
        elif name in slices:
            out[name] = slices[name]
    return out


def build_residual_vector(iso: Dict, x_norm: np.ndarray, blocks,
                          lo, hi, converged, misfit, config,
                          full_norm=None) -> np.ndarray:
    """The full weighted least-squares residual for one design.

    Layout (concatenated; each block's length is fixed for a run, so DFO-LS
    sees a constant-length vector):

        [ f - mean(f)                      isochronism (MHz), len n_eval_pts
          w_side * D2(side_norm)           if 'side' optimized & w_side != 0
          w_top  * D2(top_norm)            if 'top'  optimized & w_top  != 0
          w_conv * sqrt(max(0, shortfall)) convergence gate, len 1 if w_conv!=0
          w_mag  * x_norm ]                magnitude reg, len n_params if !=0

    The second differences are taken PER BLOCK on the NORMALIZED offsets (both
    side and top live in [0,1], so one dimensionless weight scale works, and a
    difference is NEVER taken across the side/top boundary -- they are different
    physical quantities, deg vs mm). ``lo``/``hi`` are the physical bounds of
    the optimized blocks; they are accepted for interface symmetry (the
    penalties act on the already-normalized ``x_norm``). ``full_norm`` (a dict
    block -> full normalized profile) is passed when a radial sub-range freezes
    part of a block, so D2 is taken over the full profile rather than only the
    free params; None -> D2 over the x_norm slice (the whole optimized block).

    With every added-penalty weight 0 and a converged solve this reduces
    EXACTLY to the mean-centered frequency vector ``f - mean(f)`` -- i.e. the
    previous DFO-LS residual, bit for bit.
    """
    n = config.side_shim.num_rad_segments + 1
    parts = []

    # (1) isochronism / flatness -- always present.
    f = np.asarray(iso['rev_frequencies_mhz'], dtype=float)
    parts.append(f - f.mean())

    # (2) per-block second-difference roughness on the normalized offsets.
    w_side, w_top = _smoothness_weights(config)
    prof = _smooth_profiles(x_norm, blocks, n, full_norm)
    if 'side' in prof and w_side:
        parts.append(w_side * second_difference(prof['side']))
    if 'top' in prof and w_top:
        parts.append(w_top * second_difference(prof['top']))

    # (3) continuous convergence gate (single residual entry).
    w_conv = getattr(config.optimization, 'convergence_penalty_weight', 0.0) or 0.0
    if w_conv:
        shortfall = _convergence_shortfall(converged, misfit, config)
        parts.append(np.array([w_conv * np.sqrt(shortfall)], dtype=float))

    # (4) magnitude regularization (optional; default OFF).
    w_mag = getattr(config.optimization, 'regularization_weight', 0.0) or 0.0
    if w_mag:
        parts.append(w_mag * np.asarray(x_norm, dtype=float))

    return np.concatenate(parts) if parts else np.zeros(0, dtype=float)


def compute_objective(iso: Dict, x_norm: np.ndarray, blocks,
                      lo, hi, converged, misfit, config,
                      full_norm=None) -> Tuple[float, Dict]:
    """Scalar objective (= ||build_residual_vector||^2) plus a diagnostics dict.

    This is the SINGLE scoring entry point: the Nelder-Mead path minimizes the
    returned scalar, and the DFO-LS path returns the residual vector carried in
    ``results['residual_vector']`` -- so ||DFO-LS vector||^2 == NM scalar for
    the same inputs and weights, by construction.

    The diagnostics dict keeps the keys the CSV/plotters expect, with the
    roughness metrics added so an L-curve (roughness vs flatness) can be read
    straight off the diagnostics file. ``full_norm`` -> see build_residual_vector.
    """
    r = build_residual_vector(iso, x_norm, blocks, lo, hi, converged, misfit, config,
                              full_norm=full_norm)
    objective = float(np.dot(r, r))

    n = config.side_shim.num_rad_segments + 1
    prof = _smooth_profiles(x_norm, blocks, n, full_norm)
    w_side, w_top = _smoothness_weights(config)
    # Unweighted per-block roughness (||D2||) -- the machinability-readable
    # metric that drives the L-curve knee, independent of the chosen weight.
    rough_side = (float(np.linalg.norm(second_difference(prof['side'])))
                  if 'side' in prof else 0.0)
    rough_top = (float(np.linalg.norm(second_difference(prof['top'])))
                 if 'top' in prof else 0.0)
    # Weighted smoothness contribution to the residual (L2 of that sub-block).
    smoothness_residual_l2 = float(np.hypot(w_side * rough_side, w_top * rough_top))

    w_mag = getattr(config.optimization, 'regularization_weight', 0.0) or 0.0
    regularization = (float(w_mag * np.linalg.norm(np.asarray(x_norm, dtype=float)))
                      if w_mag else 0.0)
    w_conv = getattr(config.optimization, 'convergence_penalty_weight', 0.0) or 0.0
    shortfall = _convergence_shortfall(converged, misfit, config)
    convergence_penalty = float(w_conv * shortfall)   # linear diagnostic (as before)

    results = {
        'flatness': float(iso['std_dev_mhz']),
        'avg_f': float(iso['mean_freq_mhz']),
        'regularization': regularization,
        'roughness_side': rough_side,
        'roughness_top': rough_top,
        'smoothness_residual_l2': smoothness_residual_l2,
        'convergence_penalty': convergence_penalty,
        'objective': objective,
        'residual_vector': r,
    }
    return objective, results


def evaluate_cyclotron_objective_simplified(surface_params_32d: np.ndarray,
                                            config: CyclotronConfig,
                                            radii_mm: list,
                                            comm,
                                            rank: int = 0,
                                            verbosity: int = 0,
                                            iteration: int = 0,
                                            x_norm: np.ndarray = None,
                                            blocks=None,
                                            lo=None,
                                            hi=None) -> Tuple[float, Dict]:
    """
    Evaluate the shared objective = ||weighted residual vector||^2.

    Scoring is delegated to compute_objective / build_residual_vector so the
    three-phase Nelder-Mead path and the joint DFO-LS path minimize the SAME
    quantity with the SAME config weights (flatness + smoothness + convergence
    + magnitude regularization).

    :param surface_params_32d: [2*(n_seg+1)] denormalized surface offsets
    :param config: CyclotronConfig
    :param radii_mm: List of radii in mm
    :param comm: MPI communicator
    :param rank: MPI rank
    :param verbosity: Verbosity level
    :param iteration: Iteration number
    :param x_norm: Normalized [0,1] params of the optimized block(s); feeds the
        smoothness (second-difference) and magnitude penalties.
    :param blocks: which block(s) x_norm holds, e.g. ['side'] or ['top'] (the
        current Nelder-Mead phase). None -> no shim penalties, pure flatness.
    :param lo, hi: physical bounds of the optimized block(s) (interface symmetry
        with the DFO-LS builder; the penalties act on the normalized x_norm).
    :return: (objective, results_dict)
    """

    reference_coil_current = config.optimization.reference_coil_current

    if verbosity >= 1 and rank <= 0:
        print(f"    Evaluating with ref coil={reference_coil_current:.0f}A...", flush=True)

    n_segments = config.side_shim.num_rad_segments
    side_offsets_deg = surface_params_32d[:n_segments + 1]
    top_offsets_mm = surface_params_32d[n_segments + 1:2 * (n_segments + 1)]

    pole_shape = PoleShape(n_segments,
                           side_offsets=side_offsets_deg,
                           top_offsets=top_offsets_mm)

    original_current = config.coil.current_A
    config.coil.current_A = reference_coil_current

    try:
        if verbosity >= 2:
            print(f"[RANK {rank}] Before barrier 1...", flush=True)
        comm.Barrier()

        radii_out, bz_values, converged, _, misfit = evaluate_radii_parallel(
            config,
            pole_shape,
            radii_mm,
            rank=rank,
            comm=comm,
            verbosity=verbosity
        )

        if verbosity >= 2:
            print(f"[RANK {rank}] After evaluate_radii_parallel, before barrier 2...", flush=True)
        comm.Barrier()

        converged = comm.bcast(converged, root=0)

        if not converged:
            if rank <= 0 and verbosity >= 1:
                print(f"    [WARNING] Radia convergence failed", flush=True)
            # return 1e6, {
            #     'flatness': 1e6,
            #     'avg_f': 0.0,
            #     'bz_values': None,
            #     'rev_frequencies_mhz': None,
            #     'regularization': 0.0,
            #     'objective': 1e6,
            # }

        flatness = 1e6
        avg_f = 0.0
        regularization = 0.0
        objective = 1e6
        frequencies = None
        convergence_penalty = 0.0
        roughness_side = 0.0
        roughness_top = 0.0
        smoothness_residual_l2 = 0.0

        # bz_values is a (Nr, Ntheta) array (circle/gordon) or a PyPATools Field (seo),
        # and is None off-root -- test identity, not len().
        if rank <= 0 and bz_values is not None:
            if verbosity >= 2:
                print(f"[RANK 0] Computing isochronism ({config.field_evaluation.iso_method})...", flush=True)

            # Configured method (circle / gordon / seo) via the shared dispatch.
            iso = compute_isochronism(config.field_evaluation.iso_method, bz_values, radii_out,
                                      config, IonSpecies(config.particle_species),
                                      rank=rank, comm=comm, verbose=(verbosity >= 2))
            frequencies = iso['rev_frequencies_mhz']
            # Per-radius field for the result/plot (the raw input may be a grid or a Field).
            bz_values = iso['bz_for_plot']

            # Shared scoring: flatness + smoothness + convergence + magnitude, all
            # weighted from config -- identical to what DFO-LS minimizes. blocks/x_norm
            # None (e.g. coil-only phase) -> pure flatness (+ convergence gate).
            x_for_score = x_norm if x_norm is not None else np.zeros(0)
            objective, score = compute_objective(
                iso, x_for_score, blocks, lo, hi, converged, misfit, config)
            flatness = score['flatness']
            avg_f = score['avg_f']
            regularization = score['regularization']
            convergence_penalty = score['convergence_penalty']
            roughness_side = score['roughness_side']
            roughness_top = score['roughness_top']
            smoothness_residual_l2 = score['smoothness_residual_l2']

            if verbosity >= 1:
                print(f"      flatness={flatness:.2e}, avg_f={avg_f:.4f}, "
                      f"reg={regularization:.4f}, smooth={smoothness_residual_l2:.4f}, "
                      f"conv_pen={convergence_penalty:.4f} "
                      f"(converged={converged}, misfit={misfit:.2e}) → obj={objective:.6f}", flush=True)

        if verbosity >= 2:
            print(f"[RANK {rank}] Broadcasting from rank 0...", flush=True)

        objective = comm.bcast(objective, root=0)
        flatness = comm.bcast(flatness, root=0)
        avg_f = comm.bcast(avg_f, root=0)
        regularization = comm.bcast(regularization, root=0)
        frequencies = comm.bcast(frequencies, root=0)
        convergence_penalty = comm.bcast(convergence_penalty, root=0)
        roughness_side = comm.bcast(roughness_side, root=0)
        roughness_top = comm.bcast(roughness_top, root=0)
        smoothness_residual_l2 = comm.bcast(smoothness_residual_l2, root=0)
        misfit = comm.bcast(misfit, root=0)

        return objective, {
            'flatness': flatness,
            'bz_values': bz_values,
            'rev_frequencies_mhz': frequencies,
            'avg_f': avg_f,
            'regularization': regularization,
            'roughness_side': roughness_side,
            'roughness_top': roughness_top,
            'smoothness_residual_l2': smoothness_residual_l2,
            'objective': objective,
            'converged': converged,
            'misfit': misfit,
            'convergence_penalty': convergence_penalty,
        }

    finally:
        config.coil.current_A = original_current


def optimize_coil_final(best_surface_params: np.ndarray,
                        config: CyclotronConfig,
                        radii_mm: list,
                        comm,
                        rank: int = 0,
                        verbosity: int = 1) -> Tuple[float, float, int]:
    """
    Phase 3: Coil current optimization.
    """

    target_f = config.optimization.target_frequency_mhz
    coil_bounds = (config.optimization.coil_current_min_A, config.optimization.coil_current_max_A)

    if verbosity >= 1 and rank <= 0:
        print(f"Finding coil current to achieve target avg_f={target_f:.4f}MHz...\n", flush=True)

    n_segments = config.side_shim.num_rad_segments
    side_offsets_deg = best_surface_params[:n_segments + 1]
    top_offsets_mm = best_surface_params[n_segments + 1:2 * (n_segments + 1)]

    pole_shape = PoleShape(n_segments,
                           side_offsets=side_offsets_deg,
                           top_offsets=top_offsets_mm)

    n_evals = [0]

    def evaluate_at_coil(coil_current):
        """Evaluate avg frequency at given coil current. All ranks participate."""

        if verbosity >= 2 and rank <= 0:
            print(f"[RANK 0] Evaluating coil={coil_current:.0f}A...", flush=True)

        if rank <=0:
            coil_current = comm.bcast(coil_current, root=0)

        original_current = config.coil.current_A
        config.coil.current_A = coil_current

        try:
            if verbosity >= 2:
                print(f"[RANK {rank}] Before barrier...", flush=True)
            comm.Barrier()

            radii_out, bz_values, converged, _, _ = evaluate_radii_parallel(
                config,
                pole_shape,
                radii_mm,
                rank=rank,
                comm=comm,
                verbosity=verbosity
            )

            if verbosity >= 2:
                print(f"[RANK {rank}] After evaluate_radii_parallel, before barrier...", flush=True)
            comm.Barrier()

            converged = comm.bcast(converged, root=0)

            if rank <= 0 and bz_values is not None:
                iso = compute_isochronism(config.field_evaluation.iso_method, bz_values, radii_out,
                                          config, IonSpecies(config.particle_species),
                                          rank=rank, comm=comm)
                avg_f = iso['mean_freq_mhz']
                error = (avg_f - target_f) ** 2
                n_evals[0] += 1

                if verbosity >= 1:
                    print(f"    [Coil-eval {n_evals[0]}] I={coil_current:.0f}A -> avg_f={avg_f:.4f}MHz, "
                          f"err={np.sqrt(error):.4f}MHz", flush=True)
            else:
                error = None

            if verbosity >= 2:
                print(f"[RANK {rank}] Broadcasting error from rank 0...", flush=True)

            error = comm.bcast(error, root=0)

            if verbosity >= 2:
                print(f"[RANK {rank}] Received error={error}", flush=True)

            return error

        finally:
            config.coil.current_A = original_current

    # 1D bounded search
    if rank <= 0:
        if verbosity >= 2:
            print(f"[RANK 0] Starting minimize_scalar...", flush=True)

        result = minimize_scalar(
            evaluate_at_coil,
            bounds=coil_bounds,
            method='bounded',
            # tol=1e-6,
            options={
                'maxiter': config.optimization.max_iterations,
                'xatol': 1e-2,
            }
        )

        optimal_coil = result.x
        final_error = np.sqrt(result.fun)

        if verbosity >= 2:
            print(f"[RANK 0] minimize_scalar complete: coil={optimal_coil:.0f}A, error={final_error:.4f}MHz",
                  flush=True)

        if verbosity >= 2:
            print(f"[RANK 0] Broadcasting termination signal to ranks 1-{comm.Get_size() - 1}...", flush=True)

        # Signal completion to all other ranks
        comm.bcast(None, root=0)

        if verbosity >= 2:
            print(f"[RANK 0] Termination signal sent", flush=True)

    else:
        # ===== RANKS 1+: Evaluation loop =====
        if verbosity >= 2:
            print(f"[RANK {rank}] Entering idle evaluation loop...", flush=True)

        iteration_local = 0

        while True:
            if verbosity >= 2:
                print(f"[RANK {rank}] Waiting for coil_current broadcast...", flush=True)

            # Receive coil current from rank 0 (or None to terminate)
            coil_current_received = comm.bcast(None, root=0)

            if coil_current_received is None:
                if verbosity >= 2:
                    print(f"[RANK {rank}] Received termination signal", flush=True)
                break

            iteration_local += 1

            if verbosity >= 2:

                print(f"[RANK {rank}] >>> Received coil_current={coil_current_received:.0f}A", flush=True)

            # Call with received value
            error = evaluate_at_coil(coil_current_received)

            if verbosity >= 2:
                print(f"[RANK {rank}] <<< Eval {iteration_local} complete\n", flush=True)

        optimal_coil = None
        final_error = None

    if verbosity >= 2:
        print(f"[RANK {rank}] Broadcasting optimal_coil from rank 0...", flush=True)

    optimal_coil = comm.bcast(optimal_coil, root=0)
    final_error = comm.bcast(final_error, root=0)

    if verbosity >= 1 and rank <= 0:
        print(f"\n  --> Optimal coil current: {optimal_coil:.0f}A", flush=True)
        print(f"  --> Frequency error: {final_error:.4f}MHz", flush=True)
        print(f"  --> Total evaluations: {n_evals[0]}", flush=True)

    return optimal_coil, final_error, n_evals[0]


def solve_coil_for_target_frequency(solver, config, species, target_f_mhz, bracket,
                                    *, rank: int = 0, comm=None, verbosity: int = 1,
                                    xtol_A: float = 1.0, seed_current=None):
    """Find the coil current whose mean revolution frequency equals the target.

    Uses an already-built ReusableCyclotronSolver, so each trial current re-solves
    cheaply (reuses the meshed iron; cold relax). Mean frequency is monotone in current,
    so a bracketed root-find (brentq) is robust. This pins the mean frequency exactly,
    so the shim flatness is always evaluated at the true operating (saturation) point --
    fixing the working-point flaw of the old fixed-reference-current scheme.

    MPI-collective: rank 0 drives brentq and broadcasts each trial current; ranks 1+
    follow in a resolve loop until a None sentinel.

    :param bracket: (I_min, I_max) Amps; must bracket the target frequency.
    :return: (optimal_current_A, iso_dict, converged, misfit). iso_dict is the
        compute_isochronism result at the on-target current (mean_freq_mhz, std_dev_mhz,
        rev_frequencies_mhz, tunes, ...), broadcast to all ranks.
    """
    radii = solver.radii_mm
    method = config.field_evaluation.iso_method

    if rank <= 0:
        n_eval = [0]

        def g(coil_current):
            if comm is not None:
                comm.bcast(float(coil_current), root=0)   # ranks 1+ resolve this current
            _, bz, _, _ = solver.resolve_at_current(coil_current)
            iso_i = compute_isochronism(method, bz, radii, config, species, rank=rank, comm=comm)
            n_eval[0] += 1
            mf = iso_i['mean_freq_mhz']
            if verbosity >= 1:
                print(f"    [coil-solve {n_eval[0]}] I={coil_current:.1f} A -> "
                      f"mean_f={mf:.4f} MHz (err={mf - target_f_mhz:+.4f})", flush=True)
            return mf - target_f_mhz

        # Warm-start from a seed current (e.g. the previous outer iterate's optimum) via
        # the secant method -- few evals when the seed is close; fall back to a bracketed
        # brentq over the full range if it strays out of bounds or fails to converge.
        optimal_current = None
        if seed_current is not None:
            try:
                x1 = float(min(max(seed_current * 1.02, bracket[0]), bracket[1]))
                root = newton(g, x0=float(seed_current), x1=x1, tol=xtol_A, maxiter=10)
                if bracket[0] <= root <= bracket[1]:
                    optimal_current = float(root)
            except (RuntimeError, ValueError):
                optimal_current = None
        if optimal_current is None:
            optimal_current = brentq(g, bracket[0], bracket[1], xtol=xtol_A)

        # one final collective resolve exactly at the root, for the on-target iso
        if comm is not None:
            comm.bcast(float(optimal_current), root=0)
        _, bz, converged, misfit = solver.resolve_at_current(optimal_current)
        iso = compute_isochronism(method, bz, radii, config, species, rank=rank, comm=comm)

        if comm is not None:
            comm.bcast(None, root=0)   # stop the ranks-1+ resolve loop
    else:
        while True:
            ci = comm.bcast(None, root=0)
            if ci is None:
                break
            solver.resolve_at_current(ci)
        optimal_current = iso = converged = misfit = None

    if comm is not None:
        optimal_current = comm.bcast(optimal_current, root=0)
        iso = comm.bcast(iso, root=0)
        converged = comm.bcast(converged, root=0)
        misfit = comm.bcast(misfit, root=0)

    return optimal_current, iso, converged, misfit


def physics_precondition_offsets(config, n_iso_levers=1):
    """Cheap physics-based starting shim offsets (no field solve) for the DFO-LS x0.

    Returns (side_offsets_deg, top_offsets_mm), each length num_rad_segments+1 sampled at the
    pole radial stations (inner -> outer radius).

    TOP (isochronism): an isochronous machine needs the average field B0(r) = B0(0)*gamma(r).
    A simple gap-reluctance model gives B0 ~ 1/gap and the top shim reduces the gap, so the top
    offset should grow with gamma(r). gamma follows analytically from the target revolution
    frequency (v = omega*r): beta(r) = 2*pi*f_target*r/c, gamma = 1/sqrt(1-beta^2). Map gamma(r)
    linearly onto [top_min, top_max].
    SIDE (also first-order isochronism): the hill ANGULAR WIDTH sets the hill/valley duty cycle --
    a wider hill means the orbit spends more azimuth in the high-field hill, so the azimuthal-
    average B0 rises directly (this dominates the small flutter back-reaction). So for isochronism
    the side width should follow the same gamma(r) target as the top; it ALSO sets flutter /
    vertical focusing, a coupling that DFO-LS and the future nu_z constraint sort out.
    """
    clight = 299792458.0
    n = config.side_shim.num_rad_segments + 1
    r_m = np.linspace(config.pole.inner_radius_mm, config.pole.outer_radius_mm, n) * 1e-3
    f_hz = config.optimization.target_frequency_mhz * 1e6
    beta = np.clip(2.0 * np.pi * f_hz * r_m / clight, 0.0, 0.999)
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)

    # Split the gamma(r) B0 rise across the n_iso_levers being optimized (side and top are
    # both first-order B0 levers), so that optimizing both does not provision ~2x the needed
    # rise (double-counting Bz(r)). n_iso_levers=1 (single block) uses the full range.
    share = 1.0 / max(1, n_iso_levers)

    t_lo, t_hi = config.optimization.top_shim_min_mm, config.optimization.top_shim_max_mm
    g_span = gamma.max() - gamma.min()
    g_norm = (gamma - gamma.min()) / g_span if g_span > 0 else np.zeros(n)
    top = t_lo + (t_hi - t_lo) * g_norm * share

    # Side width is a first-order B0 lever via the hill duty cycle, so target the same
    # gamma(r) isochronous rise as the top (wider hill at larger r -> higher average B0).
    s_lo, s_hi = config.optimization.side_shim_min_deg, config.optimization.side_shim_max_deg
    side = s_lo + (s_hi - s_lo) * g_norm * share

    return np.clip(side, s_lo, s_hi), np.clip(top, t_lo, t_hi)
