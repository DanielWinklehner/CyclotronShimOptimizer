"""Main entry point for cyclotron optimizer with MPI support."""

import os
import sys
import time
from pathlib import Path
import argparse
from visualization.field_comparison import compare_fields

# Add radialib to path for radia
RADIA_PATH = os.path.join(Path(__file__).resolve().parent, 'radialib')
# RADIA_PATH = r"D:\Dropbox (Personal)\Code\Python\cyclotron_optimizer\radialib"
if RADIA_PATH not in sys.path:
    sys.path.insert(0, RADIA_PATH)

import numpy as np
import matplotlib.pyplot as plt
import radia as rad
# from io import StringIO
from simulation.magnetization_cache import RadiaCache

from config_io.config import CyclotronConfig

from geometry.geometry import build_geometry
from geometry.pole_shape import PoleShape
from geometry.inventor_export import InventorPoleExporter

from simulation.field_calculator import evaluate_radii_parallel
from core.frequency import revolution_time_from_radius_and_velocity, isochronism_deviation
from visualization.plots import plot_isochronism_results, plot_isochronism_metric
from core.species import IonSpecies
from core.particles import ParticleDistribution
from optimization.optimizer import CyclotronOptimizer


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str, rank: int = 0, verbosity: int = 1):
        """
        Initialize timer.

        :param name: Name of the timed operation
        :param rank: MPI rank (only rank 0 prints)
        :param verbosity: Verbosity level (0=silent, 1=print time)
        """
        self.name = name
        self.rank = rank
        self.verbosity = verbosity
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.rank <= 0 and self.verbosity >= 1:
            print(f"  {self.name}: {self.elapsed:.3f}s")


def main(rank: int = 0, comm=None, verbosity: int = 1, run_optimization: bool = False, test_geometry: bool = False,
         use_cache: bool = False, conf: str = 'config.yml'):
    """
    Main workflow: Load config, create geometry, calculate isochronism, optionally optimize.

    MPI-aware: Only rank 0 prints, performs I/O, and does post-processing calculations.
    All ranks participate in geometry building and field solving.

    :param rank: MPI rank (0 for sequential)
    :param comm:
    :param verbosity: Verbosity level (0=silent, 1=normal, 2=debug)
    :param run_optimization: Whether to run Bayesian optimization
    :param test_geometry: If called, just show the geometry in OpenGL w/o symmetries applied
    :param conf: Path/Name of config.yml file
    :param use_cache
    """

    t_total = Timer("Total execution time", rank, verbosity)
    t_total.__enter__()

    if rank <= 0 and verbosity >= 1:
        print("\n" + "="*60, flush=True)
        print("CYCLOTRON OPTIMIZER v0.1", flush=True)
        print("="*60 + "\n", flush=True)

    # ========== CONFIGURATION ==========
    with Timer("Load configuration", rank, verbosity):
        if rank <= 0 and verbosity >= 1:
            print(f"Loading configuration...", flush=True)
        config = CyclotronConfig.from_yaml(conf)
        if rank <= 0 and verbosity >= 1:
            print(f"[OK] Configuration loaded", flush=True)
            print(f"  Species: {config.particle_species}", flush=True)
            print(f"  Target frequency: {config.optimization.target_frequency_mhz} MHz", flush=True)
            print(f"\nSetting up radius evaluation points...", flush=True)

    # ========== DEFINE RADII ==========
    r_min_mm = config.field_evaluation.radius_min_mm
    r_max_mm = config.field_evaluation.radius_max_mm
    n_radii = config.field_evaluation.n_eval_pts
    radii_mm = np.linspace(r_min_mm, r_max_mm, n_radii).tolist()

    if rank <= 0 and verbosity >= 1:
        print(f"  Evaluating {n_radii} radii from {r_min_mm:.1f} to {r_max_mm:.1f} mm", flush=True)
        print("", flush=True)

    # ========== OPTIMIZATION ==========
    if run_optimization and not test_geometry:

        # --- Solve base configuration (using default offsets) and save magnetizations
        if use_cache:
            rad.UtiDelAll()
            # If for_caching = True, returns (yoke with base pole, shims)
            cyclotron_for_cache, _ = build_geometry(config,
                                                    pole_shape=None, rank=rank, comm=comm, for_caching=True,
                                                    use_cache=True, verbosity=verbosity)

            # Solve magnetostatics problem
            if rank <=0 and verbosity >=1:
                print("\nSolving base model for caching...", flush=True)

            # Create interaction matrix
            im_id = rad.RlxPre(cyclotron_for_cache)
            result = rad.RlxAuto(im_id, config.simulation.precision, config.simulation.iterations, 4, 'ZeroM->False')
            converged = (result[0] <= config.simulation.precision)

            if not converged and rank <= 0:
                print(f"Radia did not converge. Result: {result}\n", flush=True)
                print(f"You may want to increase the number of iterations or change the geometry resolution", flush=True)
                response = input(f"Save cache anyway and continue (YES|no)? ")
                if response == "no":
                    rad.UtiMPI('off')
                    exit(1)
            else:
                if rank <= 0:
                    print(f"Radia converged. Result: {result}\n Saving magnetization cache...", flush=True)

            comm.Barrier()

            # --- Save base configuration magnetizations (all ranks keep a copy of the full geometry with magnetizations)
            if rank <= 0:

                cache = RadiaCache(rank=rank, verbosity=verbosity)
                magnetizations = {}

                def traverse_comps(obj_id):
                    info_txt = rad.UtiDmp(obj_id)
                    if "Magnetic field source object: Container" in info_txt or \
                            "Magnetic field source object: Subdivided Polyhedron" in info_txt:
                        for sub_obj_id in rad.ObjCntStuf(obj_id):
                            traverse_comps(sub_obj_id)
                    elif "Magnetic field source object: Relaxable: Polyhedron" in info_txt:
                        try:
                            magnetizations[obj_id] = rad.ObjM(obj_id)
                        except Exception as e:
                            print(f"Exception happened while traversing Radia objects: {e}", flush=True)
                            print("\nElement Info:", flush=True)
                            print(info_txt, flush=True)
                            print("", flush=True)
                            rad.UtiMPI('off')
                            exit(1)

                # Iterating through yoke_wall_comp, lid_lower_comp, lid_upper_comp
                for _id in rad.ObjCntStuf(rad.ObjCntStuf(cyclotron_for_cache)[0])[:3]:
                    traverse_comps(_id)

                # Save to cache
                cache.save_magnetizations(config, magnetizations)

                if verbosity >= 1:
                    print(f"\n[RadiaCache] Saved {len(magnetizations)} magnetizations", flush=True)

        # Optimize
        with Timer("Optimization", rank, verbosity):
            optimizer = CyclotronOptimizer(config, radii_mm, comm=comm, rank=rank,
                                           verbosity=verbosity, use_cache=use_cache)
            opt_res = optimizer.optimize()
            side_offsets = opt_res['best_side_shims']
            top_offsets = opt_res['best_top_shims']
            coil_current = opt_res['optimal_coil']

        if rank <= 0:
            print(f"Best coil: {coil_current}")

    else:
        # Use defaults from config
        side_offsets = None
        top_offsets = None
        coil_current = config.coil.current_A

    comm.Barrier()

    # ========== GEOMETRY ==========
    with Timer("Create pole shape", rank, verbosity):

        if rank <= 0 and verbosity >= 1:
            print(f"Creating pole shape with {config.side_shim.num_rad_segments} segments", flush=True)

        if side_offsets is None or top_offsets is None:
            if config.side_shim.side_offsets_deg is None:
                pole_shape = PoleShape(config.side_shim.num_rad_segments,
                                       default_offset_deg=config.side_shim.default_offset_deg,
                                       default_offset_mm=config.top_shim.default_offset_mm)
            else:
                pole_shape = PoleShape(config.side_shim.num_rad_segments,
                                       side_offsets=np.array(config.side_shim.side_offsets_deg),
                                       top_offsets=np.array(config.top_shim.top_offsets_mm))
        else:
            pole_shape = PoleShape(config.side_shim.num_rad_segments,
                                   side_offsets=side_offsets,
                                   top_offsets=top_offsets)

        if rank <= 0 and verbosity >= 1:
            print(f"[OK] Pole shape created", flush=True)
            print(flush=True)

    if test_geometry:
        if rank <= 0:
            # Build geometry on rank 0 only
            rad.UtiDelAll()
            cyclotron_vis, _ = build_geometry(config, pole_shape, rank=rank, comm=comm, omit_symmetry=True, verbosity=verbosity)
            rad.ObjDrwOpenGL(cyclotron_vis)
            input("Hit Enter...")

            # After optimization
            exporter = InventorPoleExporter(config, rank=rank, verbosity=verbosity)

            macro_file = exporter.export_macro(
                pole_shape=pole_shape,
                output_path='output/cyclotron_pole.txt'
            )

        comm.Barrier()

        return None

    # ========== PARTICLE SPECIES (Rank 0 only) ==========
    species = None
    if rank <= 0:
        with Timer("Initialize particle species", rank, verbosity):
            if verbosity >= 1:
                print(f"Initializing {config.particle_species}...", flush=True)
            species = IonSpecies(config.particle_species)
            if verbosity >= 1:
                print(f"[OK] {species.name}: q/m = {species.q_over_m:.3e} C/kg", flush=True)
                print(flush=True)

    # ========== B-FIELD CALCULATION ==========
    with Timer("Calculate B-fields (all radii, single call)", rank, verbosity):
        if rank <= 0 and verbosity >= 1:
            print(f"Calculating B-fields in parallel...", flush=True)
            # pole_offsets_array = pole_shape.get_side_offsets_deg()

        config.coil.current_A = coil_current
        radii_out, bz_values, converged = evaluate_radii_parallel(
            config, pole_shape, radii_mm,
            rank=rank, comm=comm
        )

        if rank <= 0 and verbosity >= 1:
            if len(bz_values) > 0:
                print(f"[OK] B-field calculation complete", flush=True)
                print(f"  B-field range: {min(bz_values):.4f} to {max(bz_values):.4f} T", flush=True)
                print(f"  Convergence: {'[OK]' if converged else '[FAILED]'}", flush=True)
                print(flush=True)
            else:
                print(f"[OK] B-field calculation complete (rank {rank}, no results)", flush=True)

    # ========== ENERGY AND FREQUENCY CALCULATION (Rank 0 only) ==========
    energies_mev = []
    rev_times_s = []
    rev_frequencies_mhz = []
    mean_freq_mhz = None
    std_dev_mhz = None
    percent_dev = None

    if rank <= 0 < len(bz_values):
        with Timer("Calculate energies and frequencies", rank, verbosity):
            if verbosity >= 1:
                print(f"Calculating energies and revolution times...", flush=True)

            # Create particle distribution
            particles = ParticleDistribution(species=species)

            for r_mm, bz_t in zip(radii_out, bz_values):
                # Scale B-field by current ratio
                current_ratio = coil_current / config.coil.current_A
                bz_scaled = bz_t * current_ratio

                # Calculate b_rho from r and Bz
                b_rho_tmm = r_mm * bz_scaled
                b_rho_tm = b_rho_tmm * 1e-3  # Convert to T·m

                # Set particle momentum from b_rho
                energy = particles.set_z_momentum_from_b_rho(b_rho_tm)
                energies_mev.append(energy)

                # Calculate revolution time
                v_mean = particles.v_mean_m_per_s
                rev_time = revolution_time_from_radius_and_velocity(r_mm, v_mean)
                rev_times_s.append(rev_time)

                # Calculate frequency
                rev_freq = 1.0 / rev_time
                rev_frequencies_mhz.append(rev_freq / 1e6)

            if verbosity >= 1:
                print(f"[OK] Calculations complete", flush=True)
                print(f"  Energy range: {min(energies_mev):.2f} to {max(energies_mev):.2f} MeV", flush=True)
                print(f"  Frequency range: {min(rev_frequencies_mhz):.3f} to {max(rev_frequencies_mhz):.3f} MHz",
                      flush=True)
                print(flush=True)

        # ========== ISOCHRONISM METRIC (Rank 0 only) ==========
        with Timer("Calculate isochronism metrics", rank, verbosity):
            if verbosity >= 1:
                print(f"Computing isochronism quality...", flush=True)
            mean_freq_mhz, std_dev_mhz, percent_dev = isochronism_deviation(np.array(rev_frequencies_mhz))
            if verbosity >= 1:
                print(f"[OK] Isochronism Analysis:", flush=True)
                print(f"  Mean frequency: {mean_freq_mhz:.6f} MHz", flush=True)
                print(f"  Std. deviation: {std_dev_mhz:.6f} MHz", flush=True)
                print(f"  % Deviation: {percent_dev:.4f}%", flush=True)
                print(flush=True)

    # ========== OPENGL VISUALIZATION (Rank 0 only) ==========
    if config.visualization.show_opengl:
        with Timer("Display geometry in OpenGL", rank, verbosity):
            if rank <= 0 and verbosity >= 1:
                print(f"Opening OpenGL viewer...", flush=True)
            # Rebuild geometry for visualization
            rad.UtiDelAll()
            cyclotron_vis, _ = build_geometry(config, pole_shape, rank=rank, comm=comm,
                                              omit_symmetry=True, verbosity=verbosity)
            if rank <= 0:
                rad.ObjDrwOpenGL(cyclotron_vis)
                if verbosity >= 1:
                    print(f"[OK] OpenGL viewer closed", flush=True)
                    print(flush=True)

        # ========== VISUALIZATION (Rank 0 Only) ==========
    if rank <= 0 < len(bz_values):
        with Timer("Generate plots", rank, verbosity):
            if verbosity >= 1:
                print(f"Generating plots...", flush=True)

            # Plot 1: Main isochronism plot
            fig1, ax1 = plot_isochronism_results(
                radii_out,
                bz_values,
                energies_mev,
                rev_times_s,
                rev_frequencies_mhz,
                title=f"Cyclotron Isochronism: {config.particle_species.capitalize()}",
                show=False
            )

            # Plot 2: Frequency deviation
            fig2, ax2 = plot_isochronism_metric(
                radii_out,
                rev_frequencies_mhz,
                mean_freq_mhz,
                std_dev_mhz,
                percent_dev,
                show=False
            )

            if config.visualization.comsol_filename is not None:
                compare_fields(
                    external_field_filename=config.visualization.comsol_filename,
                    config=config,
                    radii_mm_radia=np.array(radii_out),
                    bz_values_radia=np.array(bz_values),
                    rev_frequencies_radia_mhz=np.array(rev_frequencies_mhz),
                    mean_freq_radia_mhz=mean_freq_mhz,
                    std_dev_radia_mhz=std_dev_mhz,
                    percent_dev_radia=percent_dev,
                    pole_shape=pole_shape,
                    verbosity=verbosity
                )

            if verbosity >= 1:
                print(f"[OK] Plots generated", flush=True)
                print(flush=True)

        # Display plots
        if verbosity >= 1:
            print(f"Displaying plots...", flush=True)
        plt.show()

    t_total.__exit__(None, None, None)

    if rank <= 0 and verbosity >= 1:
        print(f"\n[OK] Complete!", flush=True)
        input("Hit Enter")

    comm.Barrier()

    # Return results only from rank 0
    if rank <= 0 < len(bz_values):
        return {
            'radii_mm': radii_out,
            'bz_values': bz_values,
            'energies_mev': energies_mev,
            'rev_times_s': rev_times_s,
            'rev_frequencies_mhz': rev_frequencies_mhz,
            'mean_freq_mhz': mean_freq_mhz,
            'std_dev_mhz': std_dev_mhz,
            'percent_dev': percent_dev,
            'pole_shape': pole_shape,
            'coil_current': coil_current,
        }
    else:
        return None


if __name__ == '__main__':
    # Initialize MPI (works with or without mpiexec)
    rank = rad.UtiMPI('on')
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    parser = argparse.ArgumentParser(description='Cyclotron optimization')
    parser.add_argument('--optimize', action='store_true', help='Run optimization')
    parser.add_argument('--cached', action='store_true', help='Cache cyclotron base field and use as applied field for pole')
    parser.add_argument('--geo_test', action='store_true', help='Visual inspection of geometry only')
    parser.add_argument('--verbosity', type=int, default=1, help='Verbosity level (0-2)')
    parser.add_argument('--config', type=str, help='Path to config file')

    args = parser.parse_args()

    results = main(rank=rank,
                   comm=comm,
                   verbosity=args.verbosity,
                   run_optimization=args.optimize,
                   use_cache=args.cached,
                   test_geometry=args.geo_test,
                   conf=args.config)

    # Finalize MPI
    rad.UtiMPI('off')
