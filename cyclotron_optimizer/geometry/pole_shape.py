"""Pole shape parameterization for shimming."""

import numpy as np
from typing import List


class PoleShape:
    """
    Represents the shimmed pole shape parameterized by angular offsets.

    The pole edge is defined by N radial points. Each point has an angular
    offset relative to the base pole edge angle, in degrees.
    """

    def __init__(self,
                 num_segments: int,
                 default_offset_deg: float = 0.5,
                 default_offset_mm: float = 0.5,
                 side_offsets: np.ndarray = None,
                 top_offsets: np.ndarray = None):
        """
        Initialize pole shape.

        :param num_segments: Number of shim segments (N)
        :param default_offset_deg: Default angular offset in degrees (must be > 0)
        :param default_offset_mm: Default top offset in mm (must be > 0)
        :param side_offsets: Array of N angular offsets in degrees. If None, use defaults.
        :param top_offsets: Array of N angular offsets in mm. If None, use defaults.
        """
        if num_segments < 1:
            raise ValueError("num_segments must be >= 1")

        if default_offset_deg <= 0:
            raise ValueError("default_offset_deg must be > 0")

        if default_offset_mm <= 0:
            raise ValueError("default_offset_mm must be > 0")

        self.num_segments = num_segments
        self.default_offset_deg = default_offset_deg
        self.default_offset_mm = default_offset_mm

        # Initialize side offsets in degrees
        if side_offsets is None:
            self.side_offsets_deg = np.ones(num_segments + 1) * default_offset_deg
        else:
            side_offsets = np.asarray(side_offsets)
            if side_offsets.shape != (num_segments + 1,):
                raise ValueError(f"side offsets must have shape ({num_segments + 1},)")
            if np.any(side_offsets <= 0):
                raise ValueError("All side offsets must be > 0")
            self.side_offsets_deg = side_offsets.copy()

        # Initialize top offsets in degrees
        if top_offsets is None:
            self.top_offsets_mm = np.ones(num_segments + 1) * default_offset_mm
        else:
            top_offsets = np.asarray(top_offsets)
            if top_offsets.shape != (num_segments + 1,):
                raise ValueError(f"top offsets must have shape ({num_segments + 1},)")
            if np.any(top_offsets <= 0):
                raise ValueError("All top offsets must be > 0")
            self.top_offsets_mm = top_offsets.copy()

    def get_side_offsets_deg(self) -> np.ndarray:
        """Get the offset array in degrees."""
        return self.side_offsets_deg.copy()

    def get_side_offsets_rad(self) -> np.ndarray:
        """Get the offset array in radians."""
        return self.side_offsets_deg * np.pi / 180.0

    def get_top_offsets_mm(self) -> np.ndarray:
        """Get the offset array in degrees."""
        return self.top_offsets_mm.copy()

    def set_side_offsets_deg(self, offsets: np.ndarray) -> None:
        """
        Set the offset array in degrees.

        :param offsets: Array of N angular offsets in degrees. Must all be > 0.
        """
        offsets = np.asarray(offsets)
        if offsets.shape != (self.num_segments,):
            raise ValueError(f"offsets must have shape ({self.num_segments},)")
        if np.any(offsets <= 0):
            raise ValueError("All offsets must be > 0")
        self.side_offsets_deg = offsets.copy()
