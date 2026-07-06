"""Pole shape parameterization for shimming."""

import numpy as np
from typing import List, Optional


class PoleShape:
    """
    Represents the shimmed pole shape parameterized by angular offsets.

    The pole edge is defined by N radial points. Each point has an angular
    offset relative to the base pole edge angle, in degrees.
    """

    def __init__(self,
                 num_segments: int,
                 default_offset_deg: Optional[float] = 0.0,
                 default_offset_mm: Optional[float] = 0.0,
                 side_offsets: np.ndarray = None,
                 top_offsets: np.ndarray = None):
        """
        Initialize pole shape (N radial segments -> N+1 boundary values).

        ``side_offsets`` are HALF-ANGLE offsets: each is added to the pole's
        half-wedge angle, and the built wedge is completed by the azimuthal
        symmetry mirror, so the FULL pole widens by 2x the offset (offset on
        each radial face). ``top_offsets`` are the per-boundary top-shim
        heights [mm].

        :param num_segments: Number of radial shim segments (N)
        :param default_offset_deg: Default side half-angle offset [deg] (>= 0)
        :param default_offset_mm: Default top-shim offset [mm] (>= 0)
        :param side_offsets: (N+1,) side half-angle offsets [deg] (>= 0); None -> defaults
        :param top_offsets: (N+1,) top-shim offsets [mm] (>= 0); None -> defaults
        """
        if num_segments < 1:
            raise ValueError("num_segments must be >= 1")

        # The shimmed pole is now built as a single OCC solid, so shim
        # offsets are deltas from the base pole and may be zero (no minimum).
        if default_offset_deg is None:
            default_offset_deg = 0.0
        if default_offset_mm is None:
            default_offset_mm = 0.0
        if default_offset_deg < 0:
            raise ValueError("default_offset_deg must be >= 0")

        if default_offset_mm < 0:
            raise ValueError("default_offset_mm must be >= 0")

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
            if np.any(side_offsets < 0):
                raise ValueError("All side offsets must be >= 0")
            self.side_offsets_deg = side_offsets.copy()

        # Initialize top offsets in mm
        if top_offsets is None:
            self.top_offsets_mm = np.ones(num_segments + 1) * default_offset_mm
        else:
            top_offsets = np.asarray(top_offsets)
            if top_offsets.shape != (num_segments + 1,):
                raise ValueError(f"top offsets must have shape ({num_segments + 1},)")
            if np.any(top_offsets < 0):
                raise ValueError("All top offsets must be >= 0")
            self.top_offsets_mm = top_offsets.copy()

    @classmethod
    def from_shim_configs(cls, num_segments, side_shim, top_shim) -> "PoleShape":
        """Build from the SideShimConfig / TopShimConfig dataclasses.

        Handles the side and top offset arrays INDEPENDENTLY: either may be
        None (that dimension then falls back to its ``default_offset``). The
        previous callers branched only on ``side_offsets_deg is None``, which
        crashed (``np.array(None)``) when only the top array was omitted and
        silently dropped the top array when only the side was omitted -- the
        "top and side shim loading" coupling bug (issue #1).
        """
        side = (np.asarray(side_shim.side_offsets_deg, dtype=float)
                if side_shim.side_offsets_deg is not None else None)
        top = (np.asarray(top_shim.top_offsets_mm, dtype=float)
               if top_shim.top_offsets_mm is not None else None)
        return cls(num_segments,
                   default_offset_deg=side_shim.default_offset_deg,
                   default_offset_mm=top_shim.default_offset_mm,
                   side_offsets=side, top_offsets=top)

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
