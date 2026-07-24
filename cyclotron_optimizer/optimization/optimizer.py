"""Three-phase Nelder-Mead optimization."""

import numpy as np
from typing import Tuple, List, Dict
from scipy.optimize import minimize
from tqdm import tqdm
import os
import time
from datetime import datetime
import csv

from cyclotron_optimizer.config_io.config import CyclotronConfig
from cyclotron_optimizer.optimization.objective_function import (
    evaluate_cyclotron_objective_simplified,
    optimize_coil_final,
    solve_coil_for_target_frequency,
    physics_precondition_offsets,
    compute_objective,
)
from cyclotron_optimizer.optimization.constraints import (
    get_optimization_bounds,
    shim_radial_free_indices,
)
from cyclotron_optimizer.visualization.optimization_progress import OptimizationProgressPlotter
from cyclotron_optimizer.simulation.field_calculator import ReusableCyclotronSolver
from cyclotron_optimizer.geometry.pole_shape import PoleShape
from cyclotron_optimizer.core.species import IonSpecies


class CyclotronOptimizer:
    """Three-phase optimization with MPI support."""

    def __init__(self,
                 config: CyclotronConfig,
                 radii_mm: List[float],
                 comm,
                 rank: int = 0,
                 verbosity: int = 1,
                 check_convergence: bool = True,
                 max_retries: int = 2):
        """Initialize optimizer."""
        self.config = config
        self.radii_mm = radii_mm
        self.comm = comm
        self.rank = rank
        self.size = comm.Get_size()
        self.verbosity = verbosity
        self.check_convergence = check_convergence
        self.max_retries = max_retries

        # Reproducibility: seed the global RNG used by multistart / random init.
        np.random.seed(config.seed)

        # Best tracking per phase
        self.best_x = None
        self.actual_x = None
        self.best_y = None
        self.worst_y = None
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
                 'avg_frequency_mhz', 'flatness', 'regularization',
                 'roughness_side', 'roughness_top', 'smoothness_residual_l2',
                 'convergence_penalty', 'objective',
                 'converged', 'misfit', 'eval_seconds'] +
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
                results.get('roughness_side', ''),
                results.get('roughness_top', ''),
                results.get('smoothness_residual_l2', ''),
                results.get('convergence_penalty', ''),
                results['objective'],
                results.get('converged', ''),
                results.get('misfit', ''),
                results.get('eval_seconds', ''),
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

        opt_name = (self.config.optimization.optimizer or 'nelder-mead').lower().replace('_', '-')
        if opt_name in ('dfo-ls', 'dfols', 'least-squares'):
            return self.optimize_joint_least_squares()

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
            print(f"  --> Best flatness: {flatness_top:.6f} MHz", flush=True)
            print(f"  --> Top offsets: {best_top.tolist()}", flush=True)
            print(f"Phase 2 (Side shims):", flush=True)
            print(f"  --> Best flatness: {flatness_side:.6f} MHz", flush=True)
            print(f"  --> Side offsets: {best_side.tolist()}", flush=True)
            print(f"Phase 3 (Coil current):", flush=True)
            print(f"  --> Optimal coil: {optimal_coil:.0f}A", flush=True)
            print(f"  --> Frequency error: {coil_error:.4f}MHz", flush=True)
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


    def optimize_joint_least_squares(self) -> Dict:
        """Joint DFO-LS over ALL shims; the coil is nested to hold the mean frequency
        on target inside every evaluation.

        Minimizes the per-radius frequency residual vector r_i = f_rev(r_i) - f_target
        (a nonlinear least-squares problem -- DFO-LS exploits that structure) over the
        2*(num_rad_segments+1) shim parameters. For every shim vector the coil current
        is re-solved so the mean frequency == target, so flatness is always measured at
        the true operating (saturation) point; coil reuse keeps that affordable.

        MPI-collective: rank 0 drives DFO-LS, ranks 1+ follow each shim vector (and its
        nested coil solve). Returns the same dict shape as the three-phase optimize().
        """
        import time as _time
        try:
            import dfols
        except ImportError as exc:  # pragma: no cover
            raise ImportError("DFO-LS is not installed in this env. `pip install DFO-LS`.") from exc

        cfg = self.config
        species = IonSpecies(cfg.particle_species)
        target = cfg.optimization.target_frequency_mhz
        bracket = (cfg.optimization.coil_current_min_A, cfg.optimization.coil_current_max_A)
        seed_current = cfg.optimization.reference_coil_current
        # Coil-match tolerances (df/dI ~ f/I): loose during optimization (fast; saturation
        # stays correct), tight for the final production match. Mean-centered residuals make
        # the loose match harmless to the flatness objective.
        coil_match_tol = getattr(cfg.optimization, 'coil_match_tol_mhz', 0.05)
        loose_xtol_A = max(1e-3, coil_match_tol * seed_current / target)
        final_xtol_A = max(1e-4, cfg.optimization.frequency_tolerance_mhz * seed_current / target)
        n_seg = cfg.side_shim.num_rad_segments
        n = n_seg + 1

        # Which shim blocks to optimize (respect opt_side / opt_top); default to both.
        blocks = [b for b, flag in (('side', cfg.optimization.opt_side),
                                    ('top', cfg.optimization.opt_top)) if flag]
        if not blocks:
            blocks = ['side', 'top']
        # An optimized shim must exist in the geometry, else _build_pole zeros its offsets
        # (making the parameters inert). Force include=True for the optimized blocks.
        if 'side' in blocks and not cfg.side_shim.include:
            if self.rank <= 0:
                print("[DFO-LS] side_shim.include was False -> enabling it (required to optimize side shims)", flush=True)
            cfg.side_shim.include = True
        if 'top' in blocks and not cfg.top_shim.include:
            if self.rank <= 0:
                print("[DFO-LS] top_shim.include was False -> enabling it (required to optimize top shims)", flush=True)
            cfg.top_shim.include = True

        # Fixed (non-optimized) offsets come from config (or the default offset).
        side_fixed = (np.array(cfg.side_shim.side_offsets_deg, dtype=float)
                      if cfg.side_shim.side_offsets_deg is not None
                      else np.full(n, cfg.side_shim.default_offset_deg))
        top_fixed = (np.array(cfg.top_shim.top_offsets_mm, dtype=float)
                     if cfg.top_shim.top_offsets_mm is not None
                     else np.full(n, cfg.top_shim.default_offset_mm))

        # Bounds + x0 for the optimized blocks only.
        s_lo, s_hi = cfg.optimization.side_shim_min_deg, cfg.optimization.side_shim_max_deg
        t_lo, t_hi = cfg.optimization.top_shim_min_mm, cfg.optimization.top_shim_max_mm

        # x0 basis for the optimized blocks. Precedence:
        #   1) x0_from_config  -> resume from the config's saved shim offsets
        #   2) precondition    -> the cheap physics preconditioner (gamma(r) B0 target)
        #   3) otherwise       -> the config offsets (same as x0_from_config, implicitly)
        if getattr(cfg.optimization, 'x0_from_config', False):
            side_pre, top_pre = side_fixed, top_fixed
            if self.rank <= 0 and self.verbosity >= 1:
                print("[DFO-LS] starting from the config shim offsets "
                      "(x0_from_config: side_offsets_deg / top_offsets_mm)", flush=True)
                for name, arr, opted in (('side', cfg.side_shim.side_offsets_deg, 'side' in blocks),
                                         ('top', cfg.top_shim.top_offsets_mm, 'top' in blocks)):
                    if opted and arr is None:
                        print(f"[DFO-LS] note: {name} offsets are unset in the config -> x0 "
                              f"falls back to default_offset for that block", flush=True)
        elif getattr(cfg.optimization, 'precondition', False):
            # Split the gamma(r) B0 target across the optimized levers so side+top don't
            # both provision the full rise (double-counting Bz(r)).
            side_pre, top_pre = physics_precondition_offsets(cfg, n_iso_levers=len(blocks))
            if self.rank <= 0 and self.verbosity >= 1:
                print(f"[DFO-LS] starting from the physics preconditioner "
                      f"(gamma(r) B0 target split across {len(blocks)} lever(s): {blocks})", flush=True)
        else:
            side_pre, top_pre = side_fixed, top_fixed

        # Optional radial sub-range: optimize only the shim stations inside
        # [opt_shim_radius_min_mm, opt_shim_radius_max_mm]; stations OUTSIDE it stay
        # frozen at the x0-basis (starting) value (side and top share stations).
        free_idx, r_stations = shim_radial_free_indices(cfg)
        n_free = int(free_idx.size)
        radial_subset = n_free < n
        if radial_subset and self.rank <= 0 and self.verbosity >= 1:
            print(f"[DFO-LS] radial sub-range: optimizing {n_free}/{n} shim stations at "
                  f"r = {r_stations[free_idx[0]]:.0f}..{r_stations[free_idx[-1]]:.0f} mm "
                  f"(indices {free_idx[0]}..{free_idx[-1]}); the rest stay at the x0 values",
                  flush=True)

        # Base (full-length) offsets that the frozen stations keep: the x0 basis for an
        # OPTIMIZED block (config offsets under x0_from_config, else the preconditioner --
        # so a radial sub-range refines a slice of the SAME starting design it seeds from),
        # and the config offsets for a non-optimized block.
        side_base = np.asarray(side_pre if 'side' in blocks else side_fixed, dtype=float)
        top_base = np.asarray(top_pre if 'top' in blocks else top_fixed, dtype=float)

        # Bounds + x0 for the FREE stations only (x0 value from the x0 basis).
        lo_list, hi_list, x0p = [], [], []
        if 'side' in blocks:
            lo_list += [s_lo] * n_free; hi_list += [s_hi] * n_free
            x0p += list(np.clip(side_base[free_idx], s_lo, s_hi))
        if 'top' in blocks:
            lo_list += [t_lo] * n_free; hi_list += [t_hi] * n_free
            x0p += list(np.clip(top_base[free_idx], t_lo, t_hi))
        lo = np.array(lo_list, dtype=float)
        hi = np.array(hi_list, dtype=float)
        x0 = np.clip((np.array(x0p, dtype=float) - lo) / (hi - lo), 0.0, 1.0)
        n_params = len(lo)

        def _split(x_phys):
            """Reconstruct full (side, top) offset arrays from the optimized subvector.

            Frozen (out-of-range) stations keep their x0-basis value; free stations
            take the optimizer's values. Without a radial sub-range every station is
            free, so the base is fully overwritten (unchanged whole-block behavior).
            """
            side, top, i = side_base.copy(), top_base.copy(), 0
            if 'side' in blocks:
                side[free_idx] = x_phys[i:i + n_free]; i += n_free
            if 'top' in blocks:
                top[free_idx] = x_phys[i:i + n_free]
            return side, top

        def _full_norm(side_full, top_full):
            """Full normalized [0,1] profile per optimized block, for D2-over-full
            (roughness including the transition into the frozen neighbours). Only
            needed when a radial sub-range is active; None otherwise keeps the
            whole-block behavior bit-for-bit."""
            if not radial_subset:
                return None
            fn = {}
            if 'side' in blocks:
                fn['side'] = np.clip((np.asarray(side_full, float) - s_lo) / (s_hi - s_lo), 0.0, 1.0)
            if 'top' in blocks:
                fn['top'] = np.clip((np.asarray(top_full, float) - t_lo) / (t_hi - t_lo), 0.0, 1.0)
            return fn

        solver = ReusableCyclotronSolver(cfg, self.radii_mm, rank=self.rank,
                                         comm=self.comm, verbosity=self.verbosity)
        best = {'norm': np.inf, 'x': None, 'coil': None, 'flatness': -1.0, 'bz': None, 'freq': None}
        last_coil = [seed_current]   # warm-start the inner coil solve across outer iterations

        # Optional live 3-panel progress plot (rank 0 only; degrade gracefully if headless).
        plotter = None
        if self.rank <= 0 and getattr(cfg.visualization, 'live_plot', False):
            try:
                from cyclotron_optimizer.visualization.optimization_progress import DFOLSProgressPlotter
                plotter = DFOLSProgressPlotter()
                plotter.setup(inner_radius_mm=cfg.pole.inner_radius_mm,
                              outer_radius_mm=cfg.pole.outer_radius_mm,
                              half_angle_deg=cfg.pole.full_angle_deg / 2.0,
                              n_seg=n_seg, target_frequency=target)
            except Exception as exc:
                print(f"[DFO-LS] live plot disabled ({exc})", flush=True)
                plotter = None

        def _evaluate(x_norm, xtol_A):
            side, top = _split(lo + x_norm * (hi - lo))
            pole = PoleShape(n_seg, side_offsets=side, top_offsets=top)
            solver.build(pole, last_coil[0])
            coil, iso, converged, misfit = solve_coil_for_target_frequency(
                solver, cfg, species, target, bracket,
                rank=self.rank, comm=self.comm, verbosity=self.verbosity,
                seed_current=last_coil[0], xtol_A=xtol_A)
            last_coil[0] = coil
            return coil, iso, converged, misfit

        if self.rank <= 0:
            self.iteration_count = 0
            if self.verbosity >= 1:
                print(f"\n{'=' * 100}\nJOINT DFO-LS SHIM OPTIMIZATION "
                      f"({n_params} params, {len(self.radii_mm)} residuals)\n{'=' * 100}\n", flush=True)

            def residual(x_norm):
                nonlocal plotter
                x_norm = np.asarray(x_norm, dtype=float)
                self.comm.bcast((x_norm, loose_xtol_A), root=0)   # ranks 1+ evaluate this shim vector
                t0 = _time.time()
                coil, iso, converged, misfit = _evaluate(x_norm, loose_xtol_A)
                self.iteration_count += 1
                f = np.asarray(iso['rev_frequencies_mhz'], dtype=float)
                # Shared weighted residual (flatness + smoothness + convergence + magnitude).
                # With the default weights (smoothness 0, magnitude 0) and a converged
                # solve this is exactly the mean-centered frequency vector f - mean(f).
                # full_norm (radial sub-range only) makes D2 span the full shim profile,
                # including the transition into the frozen stations.
                side_full, top_full = _split(lo + x_norm * (hi - lo))
                obj, score = compute_objective(
                    iso, x_norm, blocks, lo, hi, converged, misfit, cfg,
                    full_norm=_full_norm(side_full, top_full))
                r = score['residual_vector']
                nrm = float(np.sqrt(obj))            # ||r|| of the FULL residual
                if nrm < best['norm']:
                    best.update(norm=nrm, x=x_norm.copy(), coil=coil, flatness=iso['std_dev_mhz'],
                                bz=np.asarray(iso['bz_for_plot'], dtype=float), freq=f.copy())
                self._write_diagnostics_row(
                    0, 0, self.iteration_count,
                    {'avg_f': iso['mean_freq_mhz'], 'flatness': iso['std_dev_mhz'],
                     'regularization': score['regularization'], 'objective': obj,
                     'roughness_side': score['roughness_side'],
                     'roughness_top': score['roughness_top'],
                     'smoothness_residual_l2': score['smoothness_residual_l2'],
                     'convergence_penalty': score['convergence_penalty'],
                     'converged': converged, 'misfit': misfit,
                     'eval_seconds': _time.time() - t0},
                    side_full, top_full)
                if self.verbosity >= 1:
                    print(f"  [DFO-LS eval {self.iteration_count}] coil={coil:.0f}A "
                          f"flatness={iso['std_dev_mhz']:.5f} MHz smooth={score['smoothness_residual_l2']:.4f} "
                          f"||r||={nrm:.5f} (best {best['norm']:.5f})", flush=True)
                if plotter is not None:
                    try:
                        cside, ctop = _split(lo + x_norm * (hi - lo))
                        bside, btop = _split(lo + np.asarray(best['x'], dtype=float) * (hi - lo))
                        plotter.update(eval_idx=self.iteration_count,
                                       side_cur=cside, top_cur=ctop, side_best=bside, top_best=btop,
                                       radii=np.asarray(self.radii_mm, dtype=float),
                                       bz_cur=np.asarray(iso['bz_for_plot'], dtype=float), freq_cur=f,
                                       bz_best=best['bz'], freq_best=best['freq'],
                                       obj_cur=float(iso['std_dev_mhz']), obj_best=best['flatness'], coil=coil)
                    except Exception as exc:
                        print(f"[DFO-LS] live plot update failed, disabling ({exc})", flush=True)
                        plotter = None
                return r

            opt = cfg.optimization
            maxfun = (opt.dfols_maxfun if getattr(opt, 'dfols_maxfun', None)
                      else max(opt.max_iterations, n_params + 2))
            rhobeg = float(getattr(opt, 'dfols_rhobeg', 0.1) or 0.1)
            rhoend = float(getattr(opt, 'dfols_rhoend', 1e-3) or 1e-3)
            has_noise = bool(getattr(opt, 'dfols_objfun_has_noise', False))
            seek_global = bool(getattr(opt, 'dfols_seek_global_minimum', False))
            # DFO-LS 1.6.5 has NO seek_global_minimum kwarg; global behavior = hard restarts
            # via user_params. objfun_has_noise on its own enables (soft) restarts.
            user_params = {}
            if seek_global:
                user_params.update({
                    'restarts.use_restarts': True,
                    'restarts.use_soft_restarts': False,   # hard restarts explore -> global
                    'restarts.max_unsuccessful_restarts': 20,
                    'restarts.increase_npt': True,
                })
            solve_kwargs = dict(
                bounds=(np.zeros(n_params), np.ones(n_params)),
                maxfun=maxfun, rhobeg=rhobeg, rhoend=rhoend,
                objfun_has_noise=has_noise,
            )
            if user_params:
                solve_kwargs['user_params'] = user_params
            if self.verbosity >= 1:
                print(f"[DFO-LS] rhobeg={rhobeg} rhoend={rhoend} maxfun={maxfun} "
                      f"objfun_has_noise={has_noise} seek_global_minimum={seek_global}", flush=True)
            soln = dfols.solve(residual, x0, **solve_kwargs)
            # Final high-accuracy coil match at the best shims -> production current + flatness.
            if best['x'] is not None:
                self.comm.bcast((np.asarray(best['x'], dtype=float), final_xtol_A), root=0)
                coil_f, iso_f, _, _ = _evaluate(np.asarray(best['x'], dtype=float), final_xtol_A)
                best.update(coil=coil_f, flatness=iso_f['std_dev_mhz'])
            self.comm.bcast(None, root=0)   # stop ranks 1+
            if self.verbosity >= 1:
                print(f"\nDFO-LS finished ({soln.flag}) after {self.iteration_count} evals; "
                      f"final tight match: coil={best['coil']:.0f}A, "
                      f"flatness={best['flatness']:.5f} MHz", flush=True)
            if plotter is not None:
                plotter.finalize(savepath=os.path.join(
                    self.output_dir, f'dfols_progress_{self.timestamp}.png'))
        else:
            while True:
                msg = self.comm.bcast(None, root=0)
                if msg is None:
                    break
                x_norm, xtol_A = msg
                _evaluate(np.asarray(x_norm, dtype=float), xtol_A)

        solver.dispose()

        best_x = self.comm.bcast(best['x'] if self.rank <= 0 else None, root=0)
        best_coil = self.comm.bcast(best['coil'] if self.rank <= 0 else None, root=0)
        best_flatness = self.comm.bcast(best['flatness'] if self.rank <= 0 else None, root=0)

        best_side, best_top = _split(lo + np.asarray(best_x, dtype=float) * (hi - lo))
        return {
            'best_side_shims': best_side,
            'best_top_shims': best_top,
            'flatness_phase1': -1,
            'flatness_phase2': best_flatness,
            'optimal_coil': best_coil,
            'coil_error': -1,
            'n_coil_evals': 0,
            'diagnostics_file': self.diagnostics_file if self.rank <= 0 else None,
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
            print(f"  --> Optimizing: {param_type.upper()} shims ({n_params} parameters)", flush=True)
            if fixed_side is not None:
                print(f"  --> Side shims: FIXED", flush=True)
            if fixed_top is not None:
                print(f"  --> Top shims: FIXED", flush=True)
            print(f"  --> Multi-start: {n_multistart} initializations", flush=True)
            print(f"  --> Random init: {self.config.optimization.random_init}", flush=True)
            print(f"  --> Max iterations per start: {max_iter_per_start}\n", flush=True)

        self.comm.Barrier()

        self.iteration_count = 0
        self.best_y = None
        self.worst_y = None
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
                        'initial_simplex': self._get_initial_simplex(x0, scale=0.1)
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
                print(f"  --> Best objective: {self.best_y:.6f}", flush=True)
                print(f"  --> Total evaluations: {self.iteration_count}", flush=True)
                print(f"  --> Best {param_type}: {self.best_x.tolist()}", flush=True)

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

        # Save the final progress frame (rank 0). Works headless: with the Agg
        # backend the live window was never shown, but the figure was still
        # built, so it can be written to disk.
        if self.rank <= 0 and getattr(self, 'plotter', None) is not None:
            self.plotter.finalize(savepath=os.path.join(
                self.output_dir, f'nm_progress_{self.timestamp}.png'))

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
                                 pbar) -> float:
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
        t_eval0 = time.time()
        for attempt in range(self.max_retries + 1):
            try:
                objective, results_dict = evaluate_cyclotron_objective_simplified(
                    x_surface_full,
                    self.config,
                    self.radii_mm,
                    comm=self.comm,
                    rank=self.rank,
                    verbosity=self.verbosity,
                    iteration=self.iteration_count,
                    x_norm=x_norm_phase_for_reg if self.rank <= 0 else None,
                    # The current phase optimizes a single block; smoothness (D2) is
                    # applied to it via the shared builder, same weights as DFO-LS.
                    blocks=[param_type] if self.rank <= 0 else None,
                    lo=param_min if self.rank <= 0 else None,
                    hi=param_max if self.rank <= 0 else None,
                )
                break
            except Exception as e:
                if attempt < self.max_retries:
                    if self.verbosity >= 1:
                        print(f"[RANK {self.rank}] Attempt {attempt + 1} failed, retrying", flush=True)
                    continue
                else:
                    # Finite, ordered penalty (not an infinite cliff) so a failed
                    # build/solve does not wreck the simplex / trust-region geometry.
                    return 1e3 if self.worst_y is None else max(1e3, 10.0 * self.worst_y)

        eval_seconds = time.time() - t_eval0

        # Rank 0: Track best
        if self.rank <= 0:
            results_dict['eval_seconds'] = eval_seconds
            self.worst_y = objective if self.worst_y is None else max(self.worst_y, objective)

            if self.best_y is None or objective < self.best_y:
                self.best_y = objective
                self.best_x = x_phase_phys
                self.plateau_counter = 0
                if self.verbosity >= 1:
                    print(f"    [OK] NEW BEST: {self.best_y:.6f}", flush=True)
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
