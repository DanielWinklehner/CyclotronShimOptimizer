"""Constraint handling for optimization."""

import numpy as np
from typing import List, Tuple


def get_optimization_bounds(config) -> Tuple[List[float], List[float]]:
    """
    Get bounds for optimization (shim offsets + coil current).

    Returns bounds for N shim offsets + 1 coil current parameter.

    :param config: CyclotronConfig object
    :return: Tuple of (lower_bounds, upper_bounds)
    """
    n_segments = config.side_shim.num_rad_segments

    side_shim_lower = config.optimization.side_shim_min_deg  # deg
    side_shim_upper = config.optimization.side_shim_max_deg  # deg

    top_shim_lower = config.optimization.top_shim_min_mm  # mm
    top_shim_upper = config.optimization.top_shim_max_mm  # mm

    # Coil current bounds (Amps)
    coil_lower = config.optimization.coil_current_min_A
    coil_upper = config.optimization.coil_current_max_A

    lower_bounds = [side_shim_lower] * (n_segments + 1) + [top_shim_lower] * (n_segments + 1) + [coil_lower]
    upper_bounds = [side_shim_upper] * (n_segments + 1) + [top_shim_upper] * (n_segments + 1) + [coil_upper]

    return lower_bounds, upper_bounds


def shim_radial_free_indices(config) -> Tuple[np.ndarray, np.ndarray]:
    """Shim stations the optimizer is free to move, from the radial-range knob.

    The shim boundary points sit at
    ``linspace(pole.inner_radius_mm, pole.outer_radius_mm, num_rad_segments+1)``
    -- the POLE radius, the same convention the physics preconditioner and the
    progress plotter use. Stations whose radius lies within
    ``[opt_shim_radius_min_mm, opt_shim_radius_max_mm]`` (None = unbounded on
    that side) are free; the rest stay frozen at their config offsets. Side and
    top share stations. With both bounds unset every station is free (the
    default).

    :return: (free_idx, r_stations) -- integer indices into the length-(n+1)
        shim arrays, and the station radii [mm].
    :raises ValueError: if the band selects no station.
    """
    n = config.side_shim.num_rad_segments + 1
    r_stations = np.linspace(config.pole.inner_radius_mm,
                             config.pole.outer_radius_mm, n)
    lo = getattr(config.optimization, 'opt_shim_radius_min_mm', None)
    hi = getattr(config.optimization, 'opt_shim_radius_max_mm', None)
    mask = ((r_stations >= (-np.inf if lo is None else float(lo))) &
            (r_stations <= (np.inf if hi is None else float(hi))))
    free_idx = np.where(mask)[0]
    if free_idx.size == 0:
        raise ValueError(
            f"opt_shim_radius range [{lo}, {hi}] mm selects no shim stations "
            f"(stations span {r_stations[0]:.0f}..{r_stations[-1]:.0f} mm at "
            f"{r_stations[1] - r_stations[0]:.1f} mm spacing)")
    return free_idx, r_stations
