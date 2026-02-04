"""Objective function for three-phase optimization."""

import numpy as np
from typing import Tuple, Dict

from config_io.config import CyclotronConfig
from simulation.field_calculator import evaluate_radii_parallel
from core.frequency import revolution_time_from_radius_and_velocity
from core.species import IonSpecies
from core.particles import ParticleDistribution
from geometry.pole_shape import PoleShape
from scipy.optimize import minimize_scalar


def evaluate_cyclotron_objective_simplified(surface_params_32d: np.ndarray,
                                            config: CyclotronConfig,
                                            radii_mm: list,
                                            comm,
                                            rank: int = 0,
                                            verbosity: int = 0,
                                            iteration: int = 0,
                                            use_cache: bool = False,
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
    :param use_cache
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

        radii_out, bz_values, converged = evaluate_radii_parallel(
            config,
            pole_shape,
            radii_mm,
            rank=rank,
            comm=comm,
            verbosity=verbosity,
            use_cache=use_cache
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

        if rank <= 0 < len(bz_values):
            if verbosity >= 2:
                print(f"[RANK 0] Computing frequencies from {len(bz_values)} B-field values...", flush=True)

            species = IonSpecies(config.particle_species)
            particles = ParticleDistribution(species=species)

            rev_frequencies_mhz = []
            for r_mm, bz_t in zip(radii_out, bz_values):
                b_rho_tmm = r_mm * bz_t
                b_rho_tm = b_rho_tmm * 1e-3
                energy = particles.set_z_momentum_from_b_rho(b_rho_tm)
                v_mean = particles.v_mean_m_per_s
                rev_time = revolution_time_from_radius_and_velocity(r_mm, v_mean)
                rev_freq_hz = 1.0 / rev_time
                rev_freq_mhz = rev_freq_hz / 1e6
                rev_frequencies_mhz.append(rev_freq_mhz)

            frequencies = np.array(rev_frequencies_mhz)
            flatness = np.std(frequencies)
            avg_f = np.mean(frequencies)

            if x_norm is not None:
                offset_magnitude = np.linalg.norm(x_norm, ord=2)
            else:
                offset_magnitude = 0.0

            regularization = regularization_weight * offset_magnitude
            objective = flatness + regularization

            if verbosity >= 1:
                print(f"      flatness={flatness:.2e}, avg_f={avg_f:.4f}, "
                      f"reg={regularization:.4f} → obj={objective:.6f}", flush=True)

        if verbosity >= 2:
            print(f"[RANK {rank}] Broadcasting from rank 0...", flush=True)

        objective = comm.bcast(objective, root=0)
        flatness = comm.bcast(flatness, root=0)
        avg_f = comm.bcast(avg_f, root=0)
        regularization = comm.bcast(regularization, root=0)
        frequencies = comm.bcast(frequencies, root=0)

        return objective, {
            'flatness': flatness,
            'bz_values': bz_values,
            'rev_frequencies_mhz': frequencies,
            'avg_f': avg_f,
            'regularization': regularization,
            'objective': objective,
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

            radii_out, bz_values, converged = evaluate_radii_parallel(
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

            if rank <= 0 < len(bz_values):
                species = IonSpecies(config.particle_species)
                particles = ParticleDistribution(species=species)

                frequencies = []
                for r_mm, bz_t in zip(radii_out, bz_values):
                    b_rho_tmm = r_mm * bz_t
                    b_rho_tm = b_rho_tmm * 1e-3
                    energy = particles.set_z_momentum_from_b_rho(b_rho_tm)
                    v_mean = particles.v_mean_m_per_s
                    rev_time = revolution_time_from_radius_and_velocity(r_mm, v_mean)
                    rev_freq_hz = 1.0 / rev_time
                    rev_freq_mhz = rev_freq_hz / 1e6
                    frequencies.append(rev_freq_mhz)

                avg_f = np.mean(frequencies)
                error = (avg_f - target_f) ** 2
                n_evals[0] += 1

                if verbosity >= 1:
                    print(f"    [Coil-eval {n_evals[0]}] I={coil_current:.0f}A → avg_f={avg_f:.4f}MHz, "
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
