"""Field calculation using Radia with MPI support."""
import datetime
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
from io import StringIO

# Import radia
import radia as rad
from PyRadia import FldGPU

from config_io.config import CyclotronConfig
from geometry.geometry import build_geometry
from geometry.pole_shape import PoleShape
from PyPATools.field import Field


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter


def check_bz_outliers(
    x_range,
    y_range,
    bz,
    kernel_size=5,
    threshold=5.0,
    ax=None,
):
    """
    Detect point-like spikes in a 2D magnetic-field map and plot them.

    The detector uses a local median filter to absorb the legitimate
    smooth radial trend and azimuthal flutter, then flags points whose
    residual exceeds *threshold* × σ_MAD (robust standard deviation
    estimated from the median absolute deviation of the residuals).

    Parameters
    ----------
    x_range : array-like, shape (Nx,)
        Horizontal coordinates of the grid.
    y_range : array-like, shape (Ny,)
        Vertical coordinates of the grid.
    bz : ndarray, shape (Nx, Ny)
        Magnetic field on the regular grid.
    kernel_size : int, optional
        Side length of the square median-filter window.  Must be odd.
        Larger values tolerate broader legitimate features but may miss
        clusters of bad points.  Default 5.
    threshold : float, optional
        Number of robust standard deviations above which a residual is
        flagged as an outlier.  Default 5.0.
    ax : pair of matplotlib Axes or None, optional
        If given, plot into ``ax[0]`` (field) and ``ax[1]`` (residual).
        If *None* a new two-panel figure is created.

    Returns
    -------
    outlier_mask : ndarray of bool, shape (Nx, Ny)
        *True* where an outlier was detected.
    outlier_coords : ndarray, shape (N_outliers, 2)
        (x, y) physical coordinates of every flagged point.
    """
    x = np.asarray(x_range)
    y = np.asarray(y_range)
    bz = np.asarray(bz, dtype=float)

    # --- detection --------------------------------------------------------
    if kernel_size % 2 == 0:
        kernel_size += 1  # median_filter needs odd size

    print("using filter", flush=True)
    bz_smooth = median_filter(bz, size=kernel_size)
    residual = bz - bz_smooth

    # robust scale: MAD → σ  (factor 1.4826 for normal-equivalent σ)
    mad = np.median(np.abs(residual - np.median(residual)))
    sigma = mad * 1.4826 if mad > 0 else np.std(residual)

    outlier_mask = np.abs(residual) > threshold * sigma

    # physical coordinates of flagged points
    ix, iy = np.nonzero(outlier_mask)
    outlier_x = x[ix]
    outlier_y = y[iy]
    outlier_coords = np.column_stack([outlier_x, outlier_y]) if ix.size else np.empty((0, 2))

    print("creating axis", flush=True)
    # --- plotting ---------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    else:
        fig = ax[0].figure

    print("generating meshgrid", flush=True)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # left panel: raw field
    c0 = ax[0].pcolormesh(X, Y, bz, shading="auto", cmap="viridis")
    fig.colorbar(c0, ax=ax[0], label="$B_z$")
    if outlier_x.size:
        ax[0].scatter(
            outlier_x, outlier_y,
            s=120, facecolors="none", edgecolors="red", linewidths=1.5,
            label=f"{outlier_x.size} outlier(s)",
        )
        ax[0].legend(loc="upper right", fontsize=9)
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Magnetic field  $B_z(x,y)$")
    ax[0].set_aspect("equal")

    # right panel: residual after median subtraction
    res_lim = max(np.abs(residual).max(), 1e-30)
    c1 = ax[1].pcolormesh(
        X, Y, residual, shading="auto",
        cmap="RdBu_r", vmin=-res_lim, vmax=res_lim,
    )
    fig.colorbar(c1, ax=ax[1], label="residual")
    if outlier_x.size:
        ax[1].scatter(
            outlier_x, outlier_y,
            s=120, facecolors="none", edgecolors="red", linewidths=1.5,
        )
    ax[1].axhline(0, color="grey", lw=0.3)
    ax[1].axvline(0, color="grey", lw=0.3)
    ax[1].set_xlabel("x")
    ax[1].set_title(
        f"Residual  (median kernel={kernel_size}, "
        f"threshold={threshold}σ,  σ_MAD={sigma:.3g})"
    )
    ax[1].set_aspect("equal")

    fig.tight_layout()
    
    print("Right before show()", flush=True)
    
    plt.show()

    # summary to console
    n = int(outlier_mask.sum())
    if n:
        peak = np.abs(residual[outlier_mask]).max()
        print(
            f"Found {n} outlier(s).  "
            f"Largest residual: {peak:.4g}  ({peak / sigma:.1f}σ)"
        )
    else:
        print("No outliers detected.")

    return outlier_mask, outlier_coords


def get_median_plane_field_rz(cyclotron_id: int,
                              radii_mm: List[float],
                              num_angles: int = 1000,
                              rank: int = 0,
                              comm=None,
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

    # NOTE: Only rank 0 receives results; other ranks get empty list
    # bz_flat = rad.Fld(cyclotron_id, 'bz', points_flat)

    # if rank <= 0:
    #     print("Gathering field values with CuPy")

    # These are the model's symmetries. TODO: Need to keep them in a central spot
    model_symmetries = [
        ('perp', [0, 0, 0], [1, -1, 0]),
        ('perp', [0, 0, 0], [1, 0, 0]),
        ('perp', [0, 0, 0], [0, 1, 0]),
        ('para', [0, 0, 0], [0, 0, 1]),
    ]

    bz_flat = FldGPU(cyclotron_id, points_flat, component='bz', symmetries=model_symmetries, verbose=(rank==0))

    # Check if we got results (rank 0) or nothing (other ranks)
    if rank > 0:
        # Non-rank-0 process, return empty array
        return None

    # Reshape to (n_radii, num_angles)
    bz_grid = np.array(bz_flat).reshape(n_radii, num_angles)

    return bz_grid


def get_median_plane_field_2d(config: CyclotronConfig,
                              cyclotron_id: int = None,
                              limit=400,        # mm
                              resolution=2.0,   # mm
                              rank=0,
                              comm=None):
    """
    Calculate Bz field on median plane (z=0) exploiting 8-fold symmetry (if set).
    Only calculates in octant: 0 ≤ y ≤ x, x ≥ 0, then mirrors to fill the full domain.
    """
    use_symmetry = config.field_evaluation.use_symmetry
    dxy = resolution

    # --- Build the full regular grid definition ---
    nxy = int(2 * limit / dxy) + 1
    x_range = np.linspace(-limit, limit, nxy)  # mm
    y_range = np.linspace(-limit, limit, nxy)  # mm

    # --- Generate sample points (vectorised) ---
    if use_symmetry:
        if rank <= 0:
            print("Calculating Midplane field (with 8-fold symmetry)...", flush=True)

        # Octant grid: 0 ≤ y ≤ x, x ≥ 0
        x_oct = np.arange(0, limit + dxy / 2, dxy)
        y_oct = np.arange(0, limit + dxy / 2, dxy)
        gx, gy = np.meshgrid(x_oct, y_oct, indexing='ij')
        mask = gy <= gx + 1e-12  # 0 ≤ y ≤ x
        sample_points = np.column_stack((gx[mask], gy[mask]))
    else:
        if rank <= 0:
            print("Calculating Midplane field (without symmetry)...", flush=True)

        gx, gy = np.meshgrid(x_range, y_range, indexing='ij')
        sample_points = np.column_stack((gx.ravel(), gy.ravel()))

    # Add z=0 for 3D field query
    sample_points_3d = np.column_stack((sample_points, np.zeros(len(sample_points))))

    # # Grid for non-symmetric elements
    # gx, gy = np.meshgrid(x_range, y_range, indexing='ij')
    # sample_points_full = np.column_stack((gx.ravel(), gy.ravel()))
    # sample_points_full_3d = np.column_stack((sample_points_full, np.zeros(len(sample_points_full))))

    if rank <= 0:
        print(f"  Calculating {len(sample_points)} points...", flush=True)

    # --- These are the model's geometric symmetries ---
    model_symmetries = [
        ('perp', [0, 0, 0], [1, -1, 0]),
        ('perp', [0, 0, 0], [1, 0, 0]),
        ('perp', [0, 0, 0], [0, 1, 0]),
        ('para', [0, 0, 0], [0, 0, 1]),
    ]

    # main_objs = rad.ObjCntStuf(cyclotron_id)
    #
    # info = rad.UtiDmp(main_objs[0], 'asc')
    # print(info)
    #
    # tr1 = rad.UtiDmpPrs(rad.UtiDmp(10589, 'bin'))
    # tr2 = rad.UtiDmpPrs(rad.UtiDmp(10590, 'bin'))
    # tr3 = rad.UtiDmpPrs(rad.UtiDmp(10591, 'bin'))
    # tr4 = rad.UtiDmpPrs(rad.UtiDmp(10592, 'bin'))
    #
    # print(rad.UtiDmp(tr1, 'asc'))
    #
    # # sub_objs = rad.ObjCntStuf(main_objs[0])
    # # for sub_obj in sub_objs:
    # #     print(rad.UtiDmp(sub_obj, 'asc'))
    #
    # exit()

    # geo_symmetric = rad.ObjCnt([main_objs[0], main_objs[2]])  # symmetric cyclotron parts + coil
    # geo_non_symmetric = rad.ObjCnt([main_objs[1]])  # non-symmetric extraction channel parts

    # bz_sample = FldGPU(cyclotron_id,
    #                 sample_points_3d.tolist(),
    #                 component='bz',
    #                 symmetries=model_symmetries,
    #                 verbose=(rank == 0))

    # bz_non_sym = FldGPU(geo_non_symmetric,
    #                     sample_points_full_3d.tolist(),
    #                     component='bz',
    #                     symmetries=None,
    #                     verbose=(rank == 0))

    bz_sample = rad.Fld(cyclotron_id, 'bz', sample_points_3d.tolist(), use_gpu=True)

    # new_outliers = [[129, 79, 0],
    #                 [258, 96, 0],
    #                 [305, 181, 0],
    #                 [311, 263, 0],
    #                 [365, 262, 0],
    #                 [383, 356, 0]]
    #
    # gpu_results = rad.Fld(cyclotron_id, 'bz', new_outliers, use_gpu=True)
    # cpu_results = rad.Fld(cyclotron_id, 'bz', new_outliers, use_gpu=False)

    # gpu_results = rad.Fld(cyclotron_id, 'bz', [[270, 14, 0], [270, 15, 0]], use_gpu=True)
    # cpu_results = rad.Fld(cyclotron_id, 'bz', [[270, 14, 0], [270, 15, 0]], use_gpu=False)

    # print("===========")
    # print("Outliers on GPU: ", gpu_results)
    # print("Outliers on CPU: ", cpu_results)
    # print("===========")

    # print("===========")
    # print("single outlier: ", gpu_results[0])
    # print("single neighbor: ", gpu_results[1])
    # print("single outlier CPU: ", cpu_results[0])
    # print("single neighbor CPU: ", cpu_results[1])
    # print("===========")
    #
    # rad.UtiMPI('off')
    # exit()

    if rank <= 0:
        print(f"  Received {len(bz_sample)} field values", flush=True)

        # --- Place values onto the full grid ---
        bz_grid = np.zeros((nxy, nxy))

        # Convert mm coordinates to grid indices
        def mm_to_idx(coords_mm):
            return np.round((coords_mm + limit) / dxy).astype(int)

        if use_symmetry:
            print("  Applying 8-fold symmetry...", flush=True)
            xs, ys = sample_points[:, 0], sample_points[:, 1]

            # All 8 reflections at once: (±x,±y) and (±y,±x)
            all_x = np.concatenate([ xs,  ys,  xs,  ys, -xs, -ys, -xs, -ys])
            all_y = np.concatenate([ ys,  xs, -ys, -xs,  ys,  xs, -ys, -xs])
            all_bz = np.tile(bz_sample, 8)

            ix = mm_to_idx(all_x)
            iy = mm_to_idx(all_y)

            # Clip to valid range (handles floating-point edge cases)
            valid = (ix >= 0) & (ix < nxy) & (iy >= 0) & (iy < nxy)
            bz_grid[ix[valid], iy[valid]] = all_bz[valid]
        else:
            ix = mm_to_idx(sample_points[:, 0])
            iy = mm_to_idx(sample_points[:, 1])
            bz_grid[ix, iy] = bz_sample

        # Convert to metres for tracking
        x_range_m = x_range * 1e-3
        y_range_m = y_range * 1e-3

        count = np.count_nonzero(bz_grid) if use_symmetry else nxy * nxy
        print(f"  Total filled grid points: {count}", flush=True)
        print("Done!", flush=True)

        # bz_grid += np.array(bz_non_sym).reshape(nxy, nxy)

    #     # TODO: Remove after testing
    #     outlier_mask, outlier_coords = check_bz_outliers(x_range_m, y_range_m, bz_grid, kernel_size=7, threshold=10.0)
    #
    #     for coords in outlier_coords:
    #         print(f"x/y = ({coords[0]}, {coords[1]})")
    #
    # rad.UtiMPI('off')
    # exit()
    #     # TODO END

        b_field = Field.from_arrays(
            {"x": x_range_m, "y": y_range_m},
            {"x": np.zeros_like(bz_grid),
             "y": np.zeros_like(bz_grid),
             "z": bz_grid}
        )
    else:
        b_field = None  # non-root ranks

    return b_field


def save_3d_field(config: CyclotronConfig,
                  cyclotron_id=None,
                  zmin=-100,
                  zmax=25,
                  rank=0,
                  comm=None):
    """
    Save 3D Bx, By, Bz field exploiting 8-fold symmetry in x-y plane.

    Only calculates in octant: 0 ≤ y ≤ x, x ≥ 0
    Mirrors across x-axis, y-axis, and x=y plane to fill full domain.

    :param config: CyclotronConfig
    :param cyclotron_id: Radia cyclotron object ID
    :param zmin: Minimum z (mm)
    :param zmax: Maximum z (mm)
    :param rank: MPI rank
    :param comm: MPI communicator
    """

    # TODO: If cyclotron_id is None: delete all radia objects and rebuild/solve the cyclotron

    # TODO: Get limits and spacing (and filename?) from config

    # Domain limits and spacing
    xmin = ymin = -50  # mm
    xmax = ymax = 50  # mm
    dxy = 0.5  # mm
    dz = 0.5  # mm (adjustable)

    if rank <= 0:
        print("Calculating 3D field (with 8-fold x-y symmetry)...", flush=True)

    # ===== GENERATE OCTANT POINTS (0 ≤ y ≤ x, x ≥ 0) =====
    x_octant = np.arange(0, xmax + dxy / 2, dxy)
    z_vals = np.arange(zmin, zmax + dz / 2, dz)

    points_octant = []
    for xi in x_octant:
        for yi in np.arange(0, xi + dxy / 2, dxy):
            for zi in z_vals:
                points_octant.append([xi, yi, zi])

    points_octant = np.array(points_octant)

    if rank <= 0:
        print(f"  Calculating {len(points_octant)} points in octant...", flush=True)

    # Query Radia for all three components
    b_octant = np.array(rad.Fld(cyclotron_id, 'bxbybz', points_octant.tolist()))

    if rank <= 0:
        print(f"  Received {len(b_octant)} field vectors", flush=True)

        # ===== APPLY SYMMETRIES =====
        print("  Applying 8-fold symmetry...", flush=True)

        points_full = []
        bx_full = []
        by_full = []
        bz_full = []

        for i, (xyz, b) in enumerate(zip(points_octant, b_octant)):
            x, y, z = xyz
            bx, by, bz = b

            # 8 symmetric copies via mirror operations
            # Note: Bx, By change sign under certain mirrors; Bz does not
            symmetric_copies = [
                ((x, y, z), (bx, by, bz)),  # Octant 1: original
                ((y, x, z), (by, bx, bz)),  # Octant 2: mirror across x=y (swap x↔y, Bx↔By)
                ((x, -y, z), (bx, -by, bz)),  # Octant 3: mirror across x-axis (negate y, negate By)
                ((y, -x, z), (-by, bx, bz)),  # Octant 4: mirror across x=y then x-axis
                ((-x, y, z), (-bx, by, bz)),  # Octant 5: mirror across y-axis (negate x, negate Bx)
                ((-y, x, z), (by, -bx, bz)),  # Octant 6: mirror across y-axis then x=y
                ((-x, -y, z), (-bx, -by, bz)),  # Octant 7: mirror across both axes
                ((-y, -x, z), (-by, -bx, bz)),  # Octant 8: mirror across both axes then x=y
            ]

            for (xi, yi, zi), (bxi, byi, bzi) in symmetric_copies:
                points_full.append([xi, yi, zi])
                bx_full.append(bxi)
                by_full.append(byi)
                bz_full.append(bzi)

        points_full = np.array(points_full)
        bx_full = np.array(bx_full)
        by_full = np.array(by_full)
        bz_full = np.array(bz_full)

        # Remove duplicates
        points_full_tuple = np.array([tuple(p) for p in points_full])
        unique_points, unique_indices = np.unique(points_full_tuple, axis=0, return_index=True)

        points_full = np.array([list(p) for p in unique_points])
        bx_full = bx_full[unique_indices]
        by_full = by_full[unique_indices]
        bz_full = bz_full[unique_indices]

        # Sort by z, then y, then x
        points_full = points_full[np.lexsort((points_full[:, 0], points_full[:, 1], points_full[:, 2]))]
        sort_idx = np.lexsort((points_full[:, 0], points_full[:, 1], points_full[:, 2]))
        bx_full = bx_full[sort_idx]
        by_full = by_full[sort_idx]
        bz_full = bz_full[sort_idx]


        print(f"  Total unique points after symmetry: {len(points_full)}", flush=True)
        print("Done!", flush=True)

        header_text = f"""% Model:              uCyclo_v2
% Version:            Cyclotron Optimizer v0.1
% Date:               {datetime.date.today()}
% Dimension:          3
% Nodes:              {len(bx_full)}
% Expressions:        3
% Description:        Magnetic flux density components
% Length unit:        m
% x                   y                   z                   Bx (T)              By (T)              Bz (T)
"""

        points_m = points_full * 1e-3  # mm to m for OPAL
        data = np.column_stack((points_m, bx_full, by_full, bz_full))

        print("Writing 3D field...", flush=True)
        with open(r"output/field_3d.dat", "w") as _of:
            _of.write(header_text)
            for _d in data:
                _of.write(
                    f"{_d[0]}            {_d[1]}            {_d[2]}            {_d[3]}            {_d[4]}            {_d[5]}\n")
        print("Done!", flush=True)

    return 0


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
    symmetries = config.field_evaluation.use_symmetry
    # Domain limits and spacing
    xmin = ymin = -400  # mm
    xmax = ymax = 400  # mm
    dxy = 0.5  # mm

    points_octant = []
    if symmetries:
        if rank <= 0:
            print("Calculating Midplane field (with 8-fold symmetry)...", flush=True)
        # ===== GENERATE OCTANT POINTS (0 ≤ y ≤ x, x ≥ 0) =====
        x_octant = np.arange(0, xmax + dxy / 2, dxy)
        for xi in x_octant:
            for yi in np.arange(0, xi + dxy / 2, dxy):
                points_octant.append([xi, yi])
    else:
        if rank <= 0:
            print("Calculating Midplane field (without symmetry)...", flush=True)
        # ===== GENERATE ALL POINTS  =====
        x_octant = np.arange(xmin, xmax + dxy / 2, dxy)
        for xi in x_octant:
            for yi in np.arange(ymin, ymax + dxy / 2, dxy):
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
        if symmetries:
            print("  Applying 8-fold symmetry...", flush=True)

        points_full = []
        bz_full = []

        for i, (xy, bz) in enumerate(zip(points_octant, bz_octant)):
            x, y = xy
            if symmetries:
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
            else:
                # all already included
                symmetric_points = [
                    (x, y),  
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
                            verbosity=1):
    """
    Evaluate B-field at multiple radii using single rad.Fld() call for all points.

    All processes execute this in parallel (Radia MPI handles parallelization).
    Only rank 0 receives field results from Radia.

    :param config: CyclotronConfig object
    :param pole_shape: a PoleShape instance
    :param radii_mm: List of radii to evaluate (mm)
    :param rank: MPI rank (0 for sequential)
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
    cyclotron = build_geometry(config, pole_shape, rank=rank, comm=comm, verbosity=verbosity).id

    if rank <=0 and verbosity >= 1:
        print("Building Interaction Matrix...", flush=True)
        t0 = time.time()
    im_id = rad.RlxPre(cyclotron)
    if rank <=0 and verbosity >= 1:
        print(f"Done! Assembling took {time.time()- t0} s.", flush=True)

    if rank <=0 and verbosity >= 1:
        print("Solving...", flush=True)
        t0 = time.time()
    result = rad.RlxAuto(im_id, config.simulation.precision, config.simulation.iterations,
                         9, 'ZeroM->False', 'omega->0.3')
    if rank <=0 and verbosity >= 1:
        print(f"Done! Auto-Relaxation took {time.time() - t0} s", flush=True)
        print("Result:", result, flush=True)
        print(f"\ntarget={config.simulation.precision}: iter={result[3]:.0f}, misfitM={result[0]:.6e}")

    converged = (result[0] <= config.simulation.precision)  # Note: first result item is precision reached

    # Query all radii at once with single rad.Fld() call
    num_angles = config.field_evaluation.num_points_circle

    if config.field_evaluation.iso_method != "seo":
        bz_values = get_median_plane_field_rz(cyclotron,
                                              radii_mm,
                                              num_angles,
                                              use_symmetry=config.field_evaluation.use_symmetry,
                                              rank=rank,
                                              comm=comm)
    else:
        bz_values = get_median_plane_field_2d(config,
                                              cyclotron,
                                              limit=400,  #
                                              resolution=1.0,  # resolution in x and y (mm)
                                              rank=rank,
                                              comm=comm)

    return radii_mm, bz_values, converged, cyclotron
