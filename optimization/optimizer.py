"""Three-phase Nelder-Mead optimization."""

import numpy as np
from typing import Tuple, List, Dict
from scipy.optimize import minimize
from tqdm import tqdm
import os
from datetime import datetime
import csv

from config_io.config import CyclotronConfig
from optimization.objective_function import (
    evaluate_cyclotron_objective_simplified,
    optimize_coil_final
)
from optimization.constraints import get_optimization_bounds
from visualization.optimization_progress import OptimizationProgressPlotter


class CyclotronOptimizer:
    """Three-phase optimization with MPI support."""

    def __init__(self,
                 config: CyclotronConfig,
                 radii_mm: List[float],
                 comm,
                 rank: int = 0,
                 verbosity: int = 1,
                 check_convergence: bool = True,
                 max_retries: int = 2,
                 use_cache: bool = False):
        """Initialize optimizer."""
        self.config = config
        self.radii_mm = radii_mm
        self.comm = comm
        self.rank = rank
        self.size = comm.Get_size()
        self.verbosity = verbosity
        self.check_convergence = check_convergence
        self.max_retries = max_retries
        self.use_cache = use_cache

        # Best tracking per phase
        self.best_x = None
        self.actual_x = None
        self.best_y = None
        self.best_y_per_multistart = {}
        self.iteration_count = 0
        self.latest_results = None

        # Get bounds
        lower_bounds, upper_bounds = get_optimization_bounds(self.config)
        self.param_min = np.array(lower_bounds)
        self.param_max = np.array(upper_bounds)

        n_segments = self.config.side_shim.num_rad_segments
        self.n_side = n_segments + 1
        self.n_top = n_segments + 1

        # Separate bounds for each phase
        self.side_min = self.param_min[:self.n_side]
        self.side_max = self.param_max[:self.n_side]
        self.top_min = self.param_min[self.n_side:self.n_side + self.n_top]
        self.top_max = self.param_max[self.n_side:self.n_side + self.n_top]

        # Output directory
        self.output_dir = 'output'
        if self.rank <= 0:
            os.makedirs(self.output_dir, exist_ok=True)
            self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.diagnostics_file = os.path.join(
                self.output_dir,
                f'optimization_diagnostics_{self.timestamp}.csv'
            )
            self._init_diagnostics_csv()

        # Progress bars
        if rank <= 0:
            self.plotter = OptimizationProgressPlotter()

        # Early stopping
        self.plateau_threshold = 10
        self.plateau_counter = 0

    def _init_diagnostics_csv(self):
        """Initialize CSV file with header."""
        with open(self.diagnostics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = (
                ['phase', 'iteration', 'multistart', 'nelder_iter',
                 'avg_frequency_mhz', 'flatness', 'regularization', 'objective'] +
                [f'side_param_{i}' for i in range(self.n_side)] +
                [f'top_param_{i}' for i in range(self.n_top)]
            )
            writer.writerow(header)

    def _write_diagnostics_row(self, phase: int, multistart_idx: int, nelder_iter: int,
                               results: Dict, side_offsets: np.ndarray, top_offsets: np.ndarray):
        """Write one row to diagnostics CSV."""
        with open(self.diagnostics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                phase,
                self.iteration_count,
                multistart_idx,
                nelder_iter,
                results['avg_f'],
                results['flatness'],
                results['regularization'],
                results['objective']
            ] + side_offsets.tolist() + top_offsets.tolist()
            writer.writerow(row)

    def denormalize_params(self, x_norm: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
        """Denormalize parameters from [0,1] to physical."""
        return x_min + x_norm * (x_max - x_min)

    def normalize_params(self, x_physical: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
        """Normalize parameters from physical to [0,1]."""
        return (x_physical - x_min) / (x_max - x_min)

    def optimize(self) -> Dict:
        """
        Run three-phase optimization:
        - Phase 1: Optimize top shims for flatness
        - Phase 2: Optimize side shims for flatness
        - Phase 3: Optimize coil current for target frequency
        """

        # Determine which shims to optimize
        opt_top = self.config.optimization.opt_top
        opt_side = self.config.optimization.opt_side
        opt_coil = self.config.optimization.opt_coil

        if not self.config.top_shim.include:
            opt_top = False
        if not self.config.side_shim.include:
            opt_side = False


        if self.rank <= 0 and self.verbosity >= 1:
            print("\n" + "=" * 100, flush=True)
            print("THREE-PHASE CYCLOTRON OPTIMIZATION", flush=True)
            print("=" * 100, flush=True)
            print(f"Phase 1: Optimize TOP shims for frequency flatness", flush=True)
            if not opt_top:
                print(f"---Omitting phase 1 as per config.yml", flush=True)
            print(f"Phase 2: Optimize SIDE shims for frequency flatness (keeping best top shims)", flush=True)
            if not opt_side:
                print(f"---Omitting phase 2 as per config.yml", flush=True)
            print(f"Phase 3: Optimize coil current for target frequency", flush=True)
            if not opt_coil:
                print(f"---Omitting phase 3 as per config.yml", flush=True)
            print("=" * 100 + "\n", flush=True)

        plotter_running = False

        # ===== PHASE 1: Top shims =====
        if opt_top:
            if self.rank <= 0 and self.verbosity >= 1:
                print(f"\n{'='*100}\nPHASE 1: TOP SHIM OPTIMIZATION\n{'='*100}\n", flush=True)

                self.plotter.setup(
                    figsize=(20, 7),
                    inner_radius_mm=self.config.pole.inner_radius_mm,
                    outer_radius_mm=self.config.pole.outer_radius_mm,
                    pole_angle_deg=self.config.pole.full_angle_deg,
                    target_frequency=self.config.optimization.target_frequency_mhz
                )

                plotter_running = True

            best_top, flatness_top = self.optimize_phase(
                phase=1,
                param_type='top',
                n_params=self.n_top,
                param_min=self.top_min,
                param_max=self.top_max,
                fixed_side=None,
                fixed_top=None,
                n_multistart=self.config.optimization.n_initial_points,
                max_iter_per_start=self.config.optimization.max_iterations
            )
        else:

            if self.config.top_shim.top_offsets_mm is not None:
                best_top = np.array(self.config.top_shim.top_offsets_mm)
            else:
                best_top = np.ones(self.n_side) * self.config.top_shim.default_offset_mm

            flatness_top = -1

        # ===== PHASE 2: Side shims (with best top fixed) =====
        if opt_side:
            if self.rank <= 0 and self.verbosity >= 1:
                print(f"\n{'='*100}\nPHASE 2: SIDE SHIM OPTIMIZATION (Top shims fixed)\n{'='*100}\n", flush=True)

                if not plotter_running:
                    self.plotter.setup(
                        figsize=(20, 7),
                        inner_radius_mm=self.config.pole.inner_radius_mm,
                        outer_radius_mm=self.config.pole.outer_radius_mm,
                        pole_angle_deg=self.config.pole.full_angle_deg,
                        target_frequency=self.config.optimization.target_frequency_mhz
                    )

            best_side, flatness_side = self.optimize_phase(
                phase=2,
                param_type='side',
                n_params=self.n_side,
                param_min=self.side_min,
                param_max=self.side_max,
                fixed_side=None,
                fixed_top=best_top,
                n_multistart=self.config.optimization.n_initial_points,
                max_iter_per_start=self.config.optimization.max_iterations
            )
        else:
            if self.config.side_shim.side_offsets_deg is not None:
                best_side = np.array(self.config.side_shim.side_offsets_deg)
            else:
                best_side = np.ones(self.n_side) * self.config.side_shim.default_offset_deg

            flatness_side = -1

        # Reconstruct full surface params
        best_full_surface = np.concatenate([best_side, best_top])

        if opt_coil:
            # ===== PHASE 3: Coil optimization (with best side and top fixed) =====
            if self.rank <= 0 and self.verbosity >= 1:
                print(f"\n{'='*100}\nPHASE 3: COIL CURRENT OPTIMIZATION\n{'='*100}\n", flush=True)

            optimal_coil, coil_error, n_coil_evals = optimize_coil_final(
                best_full_surface,
                self.config,
                self.radii_mm,
                self.comm,
                self.rank,
                self.verbosity
            )

        else:
            optimal_coil = self.config.optimization.reference_coil_current
            coil_error = -1
            n_coil_evals = 0

        self.comm.Barrier()

        if self.rank <= 0 and self.verbosity >= 1:
            print(f"\n{'='*100}", flush=True)
            print(f"THREE-PHASE OPTIMIZATION COMPLETE", flush=True)
            print(f"{'='*100}", flush=True)
            print(f"Phase 1 (Top shims):", flush=True)
            print(f"  ├─ Best flatness: {flatness_top:.6f} MHz", flush=True)
            print(f"  └─ Top offsets: {best_top.tolist()}", flush=True)
            print(f"Phase 2 (Side shims):", flush=True)
            print(f"  ├─ Best flatness: {flatness_side:.6f} MHz", flush=True)
            print(f"  └─ Side offsets: {best_side.tolist()}", flush=True)
            print(f"Phase 3 (Coil current):", flush=True)
            print(f"  ├─ Optimal coil: {optimal_coil:.0f}A", flush=True)
            print(f"  └─ Frequency error: {coil_error:.4f}MHz", flush=True)
            print(f"{'='*100}\n", flush=True)

        return {
            'best_side_shims': best_side,
            'best_top_shims': best_top,
            'flatness_phase1': flatness_top,
            'flatness_phase2': flatness_side,
            'optimal_coil': optimal_coil,
            'coil_error': coil_error,
            'n_coil_evals': n_coil_evals,
            'diagnostics_file': self.diagnostics_file if self.rank <= 0 else None
        }


    def optimize_phase(self,
                       phase: int,
                       param_type: str,
                       n_params: int,
                       param_min: np.ndarray,
                       param_max: np.ndarray,
                       fixed_side: np.ndarray,
                       fixed_top: np.ndarray,
                       n_multistart: int = 1,
                       max_iter_per_start: int = 100) -> Tuple[np.ndarray, float]:
        """Optimize a single phase with config-based or random initialization."""

        if self.rank <= 0 and self.verbosity >= 1:
            print(f"Configuration:", flush=True)
            print(f"  ├─ Optimizing: {param_type.upper()} shims ({n_params} parameters)", flush=True)
            if fixed_side is not None:
                print(f"  ├─ Side shims: FIXED", flush=True)
            if fixed_top is not None:
                print(f"  ├─ Top shims: FIXED", flush=True)
            print(f"  ├─ Multi-start: {n_multistart} initializations", flush=True)
            print(f"  ├─ Random init: {self.config.optimization.random_init}", flush=True)
            print(f"  └─ Max iterations per start: {max_iter_per_start}\n", flush=True)

        self.comm.Barrier()

        self.iteration_count = 0
        self.best_y = None
        self.best_x = None
        self.plateau_counter = 0
        self.best_y_per_multistart = {}

        # ===== RANK 0: Run optimization =====
        if self.rank <= 0:

            pbar = tqdm(total=max_iter_per_start * n_multistart,
                        desc=f"Phase {phase}: {param_type.upper()} optimization",
                        disable=(self.verbosity == 0),
                        ncols=120)

            all_results = []

            for ms_idx in range(n_multistart):
                if self.verbosity >= 1:
                    print(f"\nMulti-start {ms_idx + 1}/{n_multistart}", flush=True)

                # Initialize from config or random
                if self.config.optimization.random_init:
                    x0 = np.random.uniform(0, 1, size=n_params)
                    # If optimizing top with random init, use side=param_min
                    if phase == 1:
                        self._phase1_side_init = self.normalize_params(self.side_min, self.side_min, self.side_max)
                    if self.verbosity >= 1:
                        print(f"Random initialization", flush=True)
                else:
                    # Use config values
                    if param_type == 'top':
                        config_vals = np.array(self.config.top_shim.top_offsets_mm)
                    else:  # 'side'
                        config_vals = np.array(self.config.side_shim.side_offsets_deg)

                    if ms_idx == 0:
                        x0 = self.normalize_params(config_vals, param_min, param_max)
                        # If optimizing top with config init, also use config side
                        if phase == 1:
                            side_config = np.array(self.config.side_shim.side_offsets_deg)
                            self._phase1_side_init = self.normalize_params(side_config, self.side_min, self.side_max)
                        if self.verbosity >= 1:
                            print(f"Initialize from config {param_type} values", flush=True)
                    else:
                        # Subsequent: perturb config values
                        x0_norm = self.normalize_params(config_vals, param_min, param_max)
                        x0 = x0_norm + np.random.normal(0, 0.05, size=n_params)
                        x0 = np.clip(x0, 0, 1)
                        if self.verbosity >= 1:
                            print(f"Perturb from config {param_type} values", flush=True)

                self.plateau_counter = 0
                nelder_iter_counter = [0]

                def objective_wrapper(x_norm):
                    nelder_iter_counter[0] += 1
                    return self._objective_wrapper_phase(
                        x_norm, phase, param_type,
                        param_min, param_max,
                        fixed_side, fixed_top,
                        ms_idx, nelder_iter_counter[0],
                        pbar
                    )

                result = minimize(
                    objective_wrapper,
                    x0,
                    method='Nelder-Mead',
                    bounds=[(0, 1) for _ in range(n_params)],  # ← Add bounds
                    options={
                        'maxiter': max_iter_per_start,
                        'xatol': 1e-4,
                        'fatol': 1e-6,
                        'adaptive': True,
                        'initial_simplex': self._get_initial_simplex(x0, scale=0.5)
                    }
                )

                all_results.append(result)
                self.best_y_per_multistart[ms_idx] = result.fun

                if self.verbosity >= 1:
                    print(f"Multi-start {ms_idx + 1} complete: obj={result.fun:.6f}", flush=True)

                if self.plateau_counter >= self.plateau_threshold:
                    if self.verbosity >= 1:
                        print(f"Early stopping: plateau detected", flush=True)
                    break

            pbar.close()

            # Signal completion to ranks 1+
            self.comm.bcast(None, root=0)

            if self.verbosity >= 1:
                print(f"\nPhase {phase} complete:", flush=True)
                print(f"  ├─ Best objective: {self.best_y:.6f}", flush=True)
                print(f"  ├─ Total evaluations: {self.iteration_count}", flush=True)
                print(f"  └─ Best {param_type}: {self.best_x.tolist()}", flush=True)

        # ===== RANKS 1+: Evaluation loop =====
        else:
            iteration_local = 0
            while True:
                x = self.comm.bcast(None, root=0)
                if x is None:
                    break

                iteration_local += 1
                objective, results_dict = evaluate_cyclotron_objective_simplified(
                    x,
                    self.config,
                    self.radii_mm,
                    comm=self.comm,
                    rank=self.rank,
                    verbosity=self.verbosity,
                    iteration=iteration_local
                )

        self.comm.Barrier()

        # Broadcast best result to all ranks
        best_params = self.comm.bcast(self.best_x, root=0)
        best_flatness = self.comm.bcast(self.best_y, root=0)

        return best_params, best_flatness

    def _objective_wrapper_phase(self,
                                 x_norm_phase: np.ndarray,
                                 phase: int,
                                 param_type: str,
                                 param_min: np.ndarray,
                                 param_max: np.ndarray,
                                 fixed_side: np.ndarray,
                                 fixed_top: np.ndarray,
                                 multistart_idx: int,
                                 nelder_iter: int,
                                 pbar,
                                 use_cache: bool = False) -> float:
        """
        Objective wrapper for a single phase.
        Reconstructs full surface from phase parameters + fixed parameters.
        """

        # Denormalize this phase's parameters
        x_phase_phys = self.denormalize_params(x_norm_phase, param_min, param_max)

        # Reconstruct full surface
        if param_type == 'top':
            # Phase 1: optimizing top
            if fixed_side is None:
                # Use stored side init (either config or param_min based on random_init)
                if hasattr(self, '_phase1_side_init'):
                    fixed_side = self.denormalize_params(
                        self._phase1_side_init,
                        self.side_min,
                        self.side_max
                    )
                else:
                    fixed_side = self.side_min  # Fallback
            x_surface_full = np.concatenate([fixed_side, x_phase_phys])
            x_norm_phase_for_reg = x_norm_phase
        else:  # 'side'
            # Phase 2: optimizing side, using best top from phase 1
            x_surface_full = np.concatenate([x_phase_phys, fixed_top])
            x_norm_phase_for_reg = x_norm_phase

        self.actual_x = x_surface_full
        self.iteration_count += 1

        if self.verbosity >= 1 and self.rank <= 0:
            print(f"  [MS {multistart_idx}, NM {nelder_iter}] Eval {self.iteration_count}", flush=True)

        # Write to CSV early
        if self.rank <= 0:
            side, top = x_surface_full[:self.n_side], x_surface_full[self.n_side:]
            self._write_diagnostics_row(
                phase, multistart_idx, nelder_iter,
                {'avg_f': 0.0, 'flatness': 0.0, 'regularization': 0.0, 'objective': -1},
                side, top
            )

        # Broadcast full surface to all ranks
        x_surface_full = self.comm.bcast(x_surface_full, root=0)

        # Evaluate
        for attempt in range(self.max_retries + 1):
            try:
                objective, results_dict = evaluate_cyclotron_objective_simplified(
                    x_surface_full,
                    self.config,
                    self.radii_mm,
                    comm=self.comm,
                    rank=self.rank,
                    verbosity=self.verbosity,
                    use_cache=use_cache,
                    iteration=self.iteration_count,
                    x_norm=x_norm_phase_for_reg if self.rank <= 0 else None
                )
                break
            except Exception as e:
                if attempt < self.max_retries:
                    if self.verbosity >= 1:
                        print(f"[RANK {self.rank}] Attempt {attempt + 1} failed, retrying", flush=True)
                    continue
                else:
                    return 1e6

        # Rank 0: Track best
        if self.rank <= 0:
            if self.best_y is None or objective < self.best_y:
                self.best_y = objective
                self.best_x = x_phase_phys
                self.plateau_counter = 0
                if self.verbosity >= 1:
                    print(f"    ✓ NEW BEST: {self.best_y:.6f}", flush=True)
            else:
                self.plateau_counter += 1

            # Rewrite CSV with actual results
            side, top = x_surface_full[:self.n_side], x_surface_full[self.n_side:]
            self._write_diagnostics_row(phase, multistart_idx, nelder_iter, results_dict, side, top)

            pbar.update(1)
            pbar.set_postfix({'best': f"{self.best_y:.2e}", 'avg_f': f"{results_dict['avg_f']:.4f}MHz"})

            # ===== UPDATE PLOT =====
            if self.best_x is not None and results_dict['bz_values'] is not None:
                n_side = self.config.side_shim.num_rad_segments + 1
                side_radii = np.linspace(self.config.pole.inner_radius_mm,
                                         self.config.pole.outer_radius_mm, n_side)
                top_radii = np.linspace(self.config.pole.inner_radius_mm,
                                        self.config.pole.outer_radius_mm, n_side)

                self.plotter.update(
                    iteration=self.iteration_count,
                    shim_offsets_best=np.concatenate([self.best_x if param_type == 'side' else fixed_side,
                                                      self.best_x if param_type == 'top' else fixed_top]),
                    shim_offsets_actual=np.concatenate([side, top]),
                    n_segments=self.config.side_shim.num_rad_segments,
                    current_objective=objective,
                    best_objective=self.best_y,
                    radii_mm=np.array(self.radii_mm),
                    bz_values=results_dict['bz_values'],
                    bz_values_best=results_dict.get('bz_values_best'),
                    rev_frequencies_mhz=results_dict['rev_frequencies_mhz'],
                    rev_frequencies_best_mhz=results_dict.get('rev_frequencies_best_mhz'),
                    side_radii_mm=side_radii,
                    top_radii_mm=top_radii
                )

        self.latest_results = results_dict
        return objective

    @staticmethod
    def _get_initial_simplex(x0: np.ndarray, scale: float = 0.2) -> np.ndarray:
        """Create initial simplex for Nelder-Mead."""
        n_dim = len(x0)
        simplex = np.zeros((n_dim + 1, n_dim))
        simplex[0] = x0
        for i in range(n_dim):
            simplex[i + 1] = x0.copy()
            if abs(x0[i]) > 1e-3:
                simplex[i + 1, i] += scale * x0[i]
            else:
                simplex[i + 1, i] += scale
        return np.clip(simplex, 0, 1)