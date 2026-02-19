"""Field calculation using Radia with MPI support."""
import datetime
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


import numpy as np
import radia as rad
import datetime
from config_io.config import CyclotronConfig


def save_median_plane_field(config: CyclotronConfig,
                            cyclotron_id: int = None,
                            output_path: str = "output/midplane_field.txt",
                            rank=0,
                            comm=None):
    """
    Save Bz field on median plane (z=0) exploiting 8-fold symmetry.

    Only calculates in octant: 0 ≤ y ≤ x, x ≥ 0
    Mirrors across x-axis, y-axis, and x=y plane to fill full domain.
    """

    # TODO: If cyclotron_id is None: delete all radia objects and rebuild/solve the cyclotron

    # TODO: Get limits and spacing (and filename?) from config

    # Domain limits and spacing
    xmin = ymin = -400  # mm
    xmax = ymax = 400  # mm
    dxy = 0.5  # mm

    if rank <= 0:
        print("Calculating Midplane field (with 8-fold symmetry)...", flush=True)

    # ===== GENERATE OCTANT POINTS (0 ≤ y ≤ x, x ≥ 0) =====
    x_octant = np.arange(0, xmax + dxy / 2, dxy)

    points_octant = []
    for xi in x_octant:
        for yi in np.arange(0, xi + dxy / 2, dxy):
            points_octant.append([xi, yi])

    points_octant = np.array(points_octant)
    points_octant_3d = np.column_stack((points_octant, np.zeros(len(points_octant))))

    if rank <= 0:
        print(f"  Calculating {len(points_octant)} points in octant...", flush=True)

    # Query Radia for octant only
    bz_octant = np.array(rad.Fld(cyclotron_id, 'bz', points_octant_3d.tolist()))

    if rank <= 0:
        print(f"  Received {len(bz_octant)} field values", flush=True)

        # ===== APPLY SYMMETRIES =====
        print("  Applying 8-fold symmetry...", flush=True)

        points_full = []
        bz_full = []

        for i, (xy, bz) in enumerate(zip(points_octant, bz_octant)):
            x, y = xy

            # 8 symmetric copies via mirror operations
            symmetric_points = [
                (x, y),  # Octant 1: original
                (y, x),  # Octant 2: mirror across x=y
                (x, -y),  # Octant 3: mirror across x-axis
                (y, -x),  # Octant 4: mirror across x=y then x-axis
                (-x, y),  # Octant 5: mirror across y-axis
                (-y, x),  # Octant 6: mirror across y-axis then x=y
                (-x, -y),  # Octant 7: mirror across both axes
                (-y, -x),  # Octant 8: mirror across both axes then x=y
            ]

            for (xi, yi) in symmetric_points:
                points_full.append([xi, yi])
                bz_full.append(bz)

        points_full = np.array(points_full)
        bz_full = np.array(bz_full)

        # Remove duplicates (points on axes counted multiple times)
        points_full_tuple = [tuple(p) for p in points_full]
        unique_points, unique_indices = np.unique(np.array(points_full_tuple), axis=0, return_index=True)

        points_full = np.array(unique_points)
        bz_full = bz_full[unique_indices]

        # Sort by y, then by x
        points_full = points_full[np.lexsort((points_full[:, 0], points_full[:, 1]))]
        # Reorder bz_full to match sorted points
        sort_idx = np.lexsort((points_full[:, 0], points_full[:, 1]))
        bz_full = bz_full[sort_idx]

        print(f"  Total unique points after symmetry: {len(points_full)}", flush=True)
        print("Done!", flush=True)

        header_text = f"""% Model:              uCyclo_v2
% Version:            Cyclotron Optimizer v0.1
% Date:               {datetime.date.today()}
% Dimension:          2
% Nodes:              {len(bz_full)}
% Expressions:        1
% Description:        Magnetic flux density, z-component
% Length unit:        m
% x                   y                    Bz (T)
"""

        points_m = points_full * 1e-3  # mm to m for OPAL
        data = np.column_stack((points_m, bz_full))

        print("Writing Midplane field...", flush=True)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as _of:
            _of.write(header_text)
            for _d in data:
                _of.write(f"{_d[0]}            {_d[1]}            {_d[2]}\n")
        print("Done!", flush=True)

    return 0


# def save_median_plane_field(config: CyclotronConfig,
#                             cyclotron_id=None,
#                             rank=0,
#                             comm=None):
#
#     # TODO: If cyclotron_id is None: delete all radia objects and rebuild/solve the cyclotron
#
#     # TODO: Get limits and spacing (and filename?) from config
#
#     # TODO: Exploit symmetry!
#
#     # Create a list of points to evaluate in Radia
#     # Create x and y coordinate arrays
#     xmin = ymin = -400  # -400
#     xmax = ymax = 400  # 400
#     dxy = 0.5
#
#     x = np.arange(xmin, xmax + dxy / 2, dxy)  # +dxy/2 for floating point safety
#     y = np.arange(ymin, ymax + dxy / 2, dxy)
#
#     # Create mesh grid
#     xx, yy = np.meshgrid(x, y)
#
#     # Stack into (N, 2) array
#     points = np.column_stack([xx.ravel(), yy.ravel()])
#
#     # Sort by y, then by x
#     points = points[np.lexsort((points[:, 0], points[:, 1]))]
#     points = np.column_stack((points, np.zeros(len(points))))
#
#     if rank <= 0:
#         print("Calculating Midplane field...", flush=True)
#
#     # Note: For some reason, Radia only returns results on rank 0.
#     # So we have to either bcast the results to other ranks or restrict
#     # the rest of the function to rank 0.
#     bz = np.array(rad.Fld(cyclotron_id, 'bz', points.tolist()))
#
#     if rank <= 0:
#         print("Done!", flush=True)
#         # print(points.tolist(), flush=True)
#         # print(bz, flush=True)
#         # print(len(bz), flush=True)
#
#         header_text = f"""% Model:              uCyclo_v2
# % Version:            Cyclotron Optimizer v0.1
# % Date:               {datetime.date.today()}
# % Dimension:          2
# % Nodes:              {len(bz)}
# % Expressions:        1
# % Description:        Magnetic flux density, z-component
# % Length unit:        m
# % x                   y                    Bz (T)
# """
#
#         points *= 1e-3  # mm to m for OPAL
#         data = np.column_stack((points, bz))
#
#         print("Writing Midplane field...", flush=True)
#         with open(r"output/midplane_field.dat", "w") as _of:
#             _of.write(header_text)
#             for _d in data:
#                 _of.write(f"{_d[0]}            {_d[1]}            {_d[2]}\n")
#         print("Done!", flush=True)
#
#     return 0


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

    return radii_mm, bz_values, converged, cyclotron
