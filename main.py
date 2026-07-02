"""Main entry point for cyclotron optimizer with MPI support."""
import os
# gmsh (the conda-forge *_intel build) links the LLVM OpenMP runtime
# (libomp140.x86_64.dll), while MKL-backed numpy/scipy link Intel's libiomp5md.dll.
# When the field calc loads both into one process you get:
#   OMP: Error #15: Initializing libiomp5md.dll, but found libomp140.x86_64.dll
#   already initialized.
# --geo_test never runs the threaded MKL math, so it only shows up on a full run.
# Forcing MKL to a non-OpenMP threading layer means the second OpenMP runtime is
# never loaded, avoiding the clash. Must run before gmsh/numpy/mkl are imported.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import gmsh
import radia as rad

import os
import sys
import time
from pathlib import Path
import argparse
from visualization.field_comparison import compare_fields
from PyRadia import ObjDrwPyVista

# # Add radialib to path for radia
# RADIA_PATH = os.path.join(Path(__file__).resolve().parent, 'radialib')
# # RADIA_PATH = r"D:\Dropbox (Personal)\Code\Python\cyclotron_optimizer\radialib"
# if RADIA_PATH not in sys.path:
#     sys.path.insert(0, RADIA_PATH)

import numpy as np
import matplotlib.pyplot as plt

# from io import StringIO

from config_io.config import CyclotronConfig

from geometry.geometry import build_geometry
from geometry.pole_shape import PoleShape
from geometry.inventor_export import InventorPoleExporter

from simulation.field_calculator import (evaluate_radii_parallel, get_median_plane_field,
                                          save_median_plane_field, save_bore_field)
from visualization.plots import plot_isochronism_results, plot_isochronism_metric, plot_final_summary
from visualization.field_maps import plot_median_plane_field, show_model_with_median_plane_field
from core.species import IonSpecies
from optimization.optimizer import CyclotronOptimizer
from core.isochronicity import compute_isochronism


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
         conf: str = 'config.yml'):
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

        # Optimize
        with Timer("Optimization", rank, verbosity):
            optimizer = CyclotronOptimizer(config, radii_mm, comm=comm, rank=rank,
                                           verbosity=verbosity)
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
        # Build COLLECTIVELY: build_geometry -> from_gmsh_occ/from_stp mesh on rank 0 and
        # broadcast, so every rank must call it (else rank 0 hangs forever in the bcast).
        # Only the visualization is rank-0-only.
        rad.UtiDelAll()
        cyclotron_vis = build_geometry(config, pole_shape, rank=rank, comm=comm,
                                       omit_symmetry=True, verbosity=verbosity)

        if rank <= 0:
            ObjDrwPyVista(cyclotron_vis.id)
            # rad.ObjDrwOpenGL(cyclotron_vis)

            # After optimization
            # exporter = InventorPoleExporter(config, rank=rank, verbosity=verbosity)
            #
            # macro_file = exporter.export_macro(
            #     pole_shape=pole_shape,
            #     output_path='output/cyclotron_pole.txt'
            # )

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
    with Timer("Calculate B-fields", rank, verbosity):
        if rank <= 0 and verbosity >= 1:
            print(f"Calculating B-field...", flush=True)
            # pole_offsets_array = pole_shape.get_side_offsets_deg()

        config.coil.current_A = coil_current

        radii_out, bz_values, converged, cyclotron, _ = evaluate_radii_parallel(
            config, pole_shape, radii_mm,
            rank=rank, comm=comm
        )

        if config.field_evaluation.save_median_plane_field:
            save_median_plane_field(config, cyclotron,
                                    output_path=config.field_evaluation.median_plane_field_output,
                                    rank=rank, comm=comm)

        if config.field_evaluation.save_bore_field:
            save_bore_field(config, cyclotron,
                            output_path=config.field_evaluation.bore_field_output,
                            rank=rank, comm=comm)

        # if rank <= 0 and verbosity >= 1:
        #     if len(bz_values) > 0:
        #         print(f"[OK] B-field calculation complete", flush=True)
        #         print(f"  B-field range: {min(bz_values):.4f} to {max(bz_values):.4f} T", flush=True)
        #         print(f"  Convergence: {'[OK]' if converged else '[FAILED]'}", flush=True)
        #         print(flush=True)
        #     else:
        #         print(f"[OK] B-field calculation complete (rank {rank}, no results)", flush=True)

    # ========== MEDIAN-PLANE FIELD FOR VISUALIZATION (collective) ==========
    # Grab the map from the SOLVED model now -- the OpenGL section below rebuilds
    # the geometry (rad.UtiDelAll) and would wipe it. For the seo method,
    # bz_values already IS this median-plane Field (same config limits), so no
    # extra rad.Fld call is needed; for circle/gordon all ranks do a collective
    # get_median_plane_field on the existing cyclotron handle.
    median_plane_field = None
    if config.visualization.show_median_plane_field:
        with Timer("Calculate median-plane field", rank, verbosity):
            if config.field_evaluation.iso_method == "seo":
                # reuse the solver's (full-precision) map -- no extra rad.Fld
                median_plane_field = bz_values if rank <= 0 else None
            else:
                # DISPLAY map only: coarser resolution + the fp32 GPU kernel
                # (visualization-grade, ~1e-4 relative) -- never used for
                # tracking or isochronism.
                display_res = (config.visualization.field_map_resolution_mm
                               or config.field_evaluation.median_plane_resolution_mm)
                median_plane_field = get_median_plane_field(
                    cyclotron,
                    limit_mm=config.field_evaluation.median_plane_limit_mm,
                    resolution_mm=display_res,
                    use_symmetry=config.field_evaluation.use_symmetry,
                    gpu_precision="single",
                    rank=rank, comm=comm, verbosity=verbosity,
                )

    # ========== ENERGY AND FREQUENCY CALCULATION (Rank 0 only) ==========
    energies_mev = rev_times_s = rev_frequencies_mhz = None
    mean_freq_mhz = std_dev_mhz = percent_dev = None
    tunes = None

    # bz_values is a (Nr, Ntheta) array for circle/gordon or a PyPATools Field for seo,
    # and is None on non-root ranks -- so test identity, not len().
    if rank <= 0 and bz_values is not None:
        with Timer("Compute isochronism", rank, verbosity):
            # Single dispatch for circle / gordon / seo (see core.isochronicity).
            # SEO solver knob: 'newton' (full-turn fixed point, no residual betatron),
            # 'symmetric' (mirror-plane shooting), or 'centroid' (legacy averaging).
            iso = compute_isochronism(
                config.field_evaluation.iso_method,
                bz_values, radii_out, config, species,
                solver='newton', rank=rank, comm=comm, verbose=(verbosity >= 1),
            )
            energies_mev = iso['energies_mev']
            rev_times_s = iso['rev_times_s']
            rev_frequencies_mhz = iso['rev_frequencies_mhz']
            bz_values = iso['bz_for_plot']
            mean_freq_mhz = iso['mean_freq_mhz']
            std_dev_mhz = iso['std_dev_mhz']
            percent_dev = iso['percent_dev']
            tunes = iso.get('tunes')

            # SEO returns the tracked orbits -> show them
            if iso['orbits'] is not None:
                for orb in iso['orbits']:
                    plt.plot(orb.trajectory[:, 0], orb.trajectory[:, 1])
                plt.show()

            if verbosity >= 1:
                print(f"[OK] Isochronism ({iso['method']}): "
                      f"mean = {mean_freq_mhz:.6f} MHz, std = {std_dev_mhz:.6f} MHz, "
                      f"%dev = {percent_dev:.4f}%", flush=True)
                print(flush=True)

    # ========== OPENGL VISUALIZATION (Rank 0 only) ==========
    if config.visualization.show_opengl:
        with Timer("Display geometry in OpenGL", rank, verbosity):
            if rank <= 0 and verbosity >= 1:
                print("Opening OpenGL viewer...", flush=True)
            # Rebuild geometry for visualization (full magnet, no symmetry
            # transforms, for better visibility)
            rad.UtiDelAll()
            cyclotron_vis = build_geometry(config, pole_shape, rank=rank, comm=comm,
                                           omit_symmetry=True, verbosity=verbosity)

            if rank <= 0:
                if median_plane_field is not None:
                    show_model_with_median_plane_field(cyclotron_vis.id,
                                                       median_plane_field)
                else:
                    ObjDrwPyVista(cyclotron_vis.id)

                if verbosity >= 1:
                    print(f"[OK] OpenGL viewer closed", flush=True)
                    print(flush=True)

    # ========== MEDIAN-PLANE FIELD 2D PLOT (Rank 0 only) ==========
    # Always produced when show_median_plane_field is set (with show_opengl the
    # field additionally appears in the 3D window above). Shown together with
    # the other matplotlib figures at the end.
    if rank <= 0 and median_plane_field is not None:
        plot_median_plane_field(median_plane_field, show=False)

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

            # Plot 3: Final 4-panel design summary (Bz+Energy, frequency, tunes, flutter)
            fig3, ax3 = plot_final_summary(
                radii_out, bz_values, energies_mev, rev_frequencies_mhz, tunes,
                target_freq_mhz=config.optimization.target_frequency_mhz,
                title=f"Final design summary: {config.particle_species.capitalize()}",
                show=False,
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
    # rank = rad.UtiMPI('on')
    # rank = 0
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    parser = argparse.ArgumentParser(description='Cyclotron optimization')
    parser.add_argument('--optimize', action='store_true', help='Run optimization')
    parser.add_argument('--geo_test', action='store_true', help='Visual inspection of geometry only')
    parser.add_argument('--verbosity', type=int, default=1, help='Verbosity level (0-2)')
    parser.add_argument('--config', type=str, help='Path to config file')

    args = parser.parse_args()

    results = main(rank=rank,
                   comm=comm,
                   verbosity=args.verbosity,
                   run_optimization=args.optimize,
                   test_geometry=args.geo_test,
                   conf=args.config)

    # Finalize MPI
    rad.UtiMPI('off')
