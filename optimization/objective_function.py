"""Objective function for three-phase optimization."""

import numpy as np
from typing import Tuple, Dict

from config_io.config import CyclotronConfig
from simulation.field_calculator import evaluate_radii_parallel
from core.species import IonSpecies
from core.isochronicity import compute_isochronism
from geometry.pole_shape import PoleShape
from scipy.optimize import minimize_scalar, brentq, newton


def evaluate_cyclotron_objective_simplified(surface_params_32d: np.ndarray,
                                            config: CyclotronConfig,
                                            radii_mm: list,
                                            comm,
                                            rank: int = 0,
                                            verbosity: int = 0,
                                            iteration: int = 0,
                                            x_norm: np.ndarray = None) -> Tuple[float, Dict]:
    """
    Evaluate objective: flatness + regularization (minimizes shimming).

    :param surface_params_32d: [32] denormalized surface offsets
    :param config: CyclotronConfig
    :param radii_mm: List of radii in mm
    :param comm: MPI communicator
    :param rank: MPI rank
    :param verbosity: Verbosity level
    :param iteration: Iteration number
    :param x_norm: Normalized params [0,1] for regularization
    :return: (objective, results_dict)
    """

    reference_coil_current = config.optimization.reference_coil_current
    regularization_weight = config.optimization.regularization_weight

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
            flatness = iso['std_dev_mhz']
            avg_f = iso['mean_freq_mhz']
            # Per-radius field for the result/plot (the raw input may be a grid or a Field).
            bz_values = iso['bz_for_plot']

            if x_norm is not None:
                offset_magnitude = np.linalg.norm(x_norm, ord=2)
            else:
                offset_magnitude = 0.0

            regularization = regularization_weight * offset_magnitude

            # Continuous convergence gate: never silently accept an unconverged solve.
            # Penalty is 0 at convergence (misfit <= precision) and grows smoothly with
            # the relaxation shortfall, keeping the landscape continuous for the optimizer.
            if not converged:
                shortfall = max(0.0, misfit / config.simulation.precision - 1.0)
                convergence_penalty = config.optimization.convergence_penalty_weight * shortfall

            objective = flatness + regularization + convergence_penalty

            if verbosity >= 1:
                print(f"      flatness={flatness:.2e}, avg_f={avg_f:.4f}, "
                      f"reg={regularization:.4f}, conv_pen={convergence_penalty:.4f} "
                      f"(converged={converged}, misfit={misfit:.2e}) → obj={objective:.6f}", flush=True)

        if verbosity >= 2:
            print(f"[RANK {rank}] Broadcasting from rank 0...", flush=True)

        objective = comm.bcast(objective, root=0)
        flatness = comm.bcast(flatness, root=0)
        avg_f = comm.bcast(avg_f, root=0)
        regularization = comm.bcast(regularization, root=0)
        frequencies = comm.bcast(frequencies, root=0)
        convergence_penalty = comm.bcast(convergence_penalty, root=0)
        misfit = comm.bcast(misfit, root=0)

        return objective, {
            'flatness': flatness,
            'bz_values': bz_values,
            'rev_frequencies_mhz': frequencies,
            'avg_f': avg_f,
            'regularization': regularization,
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
