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

    side_shim_lower = 5.0  # deg
    side_shim_upper = 15.0  # deg

    top_shim_lower = 5.0  # mm
    top_shim_upper = 17.0  # mm

    # Coil current bounds (Amps)
    coil_lower = config.optimization.coil_current_min_A
    coil_upper = config.optimization.coil_current_max_A

    lower_bounds = [side_shim_lower] * (n_segments + 1) + [top_shim_lower] * (n_segments + 1) + [coil_lower]
    upper_bounds = [side_shim_upper] * (n_segments + 1) + [top_shim_upper] * (n_segments + 1) + [coil_upper]

    return lower_bounds, upper_bounds
