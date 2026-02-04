"""Field calculation using Radia with MPI support."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
from io import StringIO

# Import radia
import radia as rad

from config_io.config import CyclotronConfig
from geometry.geometry import build_geometry
from geometry.pole_shape import PoleShape
from simulation.magnetization_cache import RadiaCache

def query_b_field_all_radii_at_once(cyclotron_id: int,
                                    radii_mm: List[float],
                                    num_angles: int = 1000,
                                    rank: int = 0,
                                    use_symmetry: bool = True) -> np.ndarray:
    """
    Query Bz field at all radii simultaneously with single rad.Fld() call.

    Creates a 2D grid of points at each radius, queries all at once,
    then averages along the angular direction.

    NOTE: Radia returns field values only on rank 0; other ranks receive empty list.

    :param cyclotron_id: Radia object ID for cyclotron
    :param radii_mm: List of radii in mm
    :param num_angles: Number of angles per radius (default 1000)
    :param use_symmetry:
    :param rank: MPI rank
    :return: Array of averaged Bz values at each radius (rank 0 only), empty array on other ranks
    """
    n_radii = len(radii_mm)

    # Create angles array [0; 2π[ or [0; π/4[ (using symmetry)
    if use_symmetry:
        num_angles = int(num_angles / 8.0)
        angles = np.linspace(0.0, 0.25 * np.pi, num_angles, endpoint=False)
    else:
        angles = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False)

    # Create 2D grid of points: (n_radii, num_angles, 3)
    points_grid = np.zeros((n_radii, num_angles, 3))

    for i, r_mm in enumerate(radii_mm):
        points_grid[i, :, 0] = r_mm * np.cos(angles)  # x
        points_grid[i, :, 1] = r_mm * np.sin(angles)  # y
        points_grid[i, :, 2] = 0.0  # z = 0 (midplane)

    # Flatten to (n_radii * num_angles, 3)
    points_flat = points_grid.reshape(-1, 3).tolist()

    # Single Radia call to query all points
    # NOTE: Only rank 0 receives results; other ranks get empty list
    bz_flat = rad.Fld(cyclotron_id, 'bz', points_flat)

    # Check if we got results (rank 0) or nothing (other ranks)
    if rank > 0:
        # Non-rank-0 process, return empty array
        return np.array([])

    # Reshape to (n_radii, num_angles)
    bz_grid = np.array(bz_flat).reshape(n_radii, num_angles)

    # Average along angular direction
    bz_avg = np.mean(bz_grid, axis=1)

    return bz_avg


def evaluate_radii_parallel(config: CyclotronConfig,
                            pole_shape: PoleShape,
                            radii_mm: List[float],
                            rank: int = 0,
                            comm=None,
                            use_cache: bool = False,
                            verbosity=1):
    """
    Evaluate B-field at multiple radii using single rad.Fld() call for all points.

    All processes execute this in parallel (Radia MPI handles parallelization).
    Only rank 0 receives field results from Radia.

    :param config: CyclotronConfig object
    :param pole_shape: a PoleShape instance
    :param radii_mm: List of radii to evaluate (mm)
    :param rank: MPI rank (0 for sequential)
    :param use_cache
    :param verbosity
    :return: Tuple of (radii_mm, bz_avg_values, converged_flag)
                Note: bz_avg_values is empty list on non-rank-0 processes
    """
    if isinstance(radii_mm, np.ndarray):
        radii_mm = radii_mm.tolist()

    if not isinstance(radii_mm, list):
        radii_mm = [radii_mm]

    # Clear previous Radia objects
    rad.UtiDelAll()

    # Build geometry
    yoke, pole = build_geometry(config, pole_shape, rank=rank, comm=comm, verbosity=verbosity, use_cache=use_cache)

    if rank <=0 and verbosity >= 1:
        print("Building Interaction Matrix...", flush=True)
    if use_cache:
        im_id = rad.RlxPre(pole, yoke)  # use precalculated yoke magnetization as applied field source
    else:
        im_id = rad.RlxPre(yoke)  # calculate all from scratch
    if rank <=0 and verbosity >= 1:
        print("Done!", flush=True)


    # TODO: Remove after debug
    # prec = []
    # for i in range(int(config.simulation.iterations/10)):
    #     result = rad.RlxAuto(im_id, config.simulation.precision, 10, 4, 'ZeroM->False')  # Do 10 iterations at a time
    #     prec.append(result[0])
    #
    # if rank <= 0:
    #     plt.plot(prec)
    #     plt.show()
    #     rad.UtiMPI('off')
    #     exit(0)
    # TODO: End Debug

    if rank <=0 and verbosity >= 1:
        print("Solving...", flush=True)
    result = rad.RlxAuto(im_id, config.simulation.precision, config.simulation.iterations, 4, 'ZeroM->False')
    if rank <=0 and verbosity >= 1:
        print("Done!", flush=True)
        print("Result:", result, flush=True)

    converged = (result[0] <= config.simulation.precision)  # Note: first result item is precision reached

    # Query all radii at once with single rad.Fld() call
    num_angles = config.field_evaluation.num_points_circle

    if use_cache:
        cyclotron = rad.ObjCnt([yoke, pole])
    else:
        cyclotron = yoke

    bz_values = query_b_field_all_radii_at_once(cyclotron,
                                                radii_mm,
                                                num_angles,
                                                use_symmetry=config.field_evaluation.use_symmetry,
                                                rank=rank)

    return radii_mm, bz_values, converged
