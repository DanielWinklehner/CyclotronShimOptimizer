"""Geometry components for modular cyclotron construction."""

from abc import ABC, abstractmethod
import numpy as np
from sympy import Point3D, Line3D, Plane, N
from sympy.geometry import intersection
from typing import List, Optional, Dict, Any
import radia as rad
from scipy.interpolate import interp1d

from config_io.config import CyclotronConfig
from geometry.pole_shape import PoleShape
from geometry.primitives import PolygonBuilder, SegmentationStrategy, ObjectFilter

MIN_POLYGON_AREA_MM2 = 0.001  # Minimum area in mm²

class GeometricComponent(ABC):
    """Abstract base class for geometry components."""

    def __init__(self,
                 config: CyclotronConfig,
                 material_id: Optional[int] = None,
                 rank: int = 0,
                 verbosity: int = 0):
        """
        Initialize component.

        :param config: CyclotronConfig object
        :param material_id: Radia material ID (optional)
        :param rank: MPI rank
        :param verbosity: Verbosity level
        """
        self.config = config
        self.material_id = material_id
        self.radia_id = None
        self.rank = rank
        self.verbosity = verbosity

    @abstractmethod
    def build(self) -> int:
        """
        Build and return Radia object ID.
        Must be implemented by subclasses.
        """
        pass

    def apply_material(self):
        """Apply material to this component if material_id is set."""
        if self.material_id and self.radia_id:
            rad.MatApl(self.radia_id, self.material_id)

    def set_drawing_attributes(self, color: List[float]):
        """Set drawing color."""
        if self.radia_id:
            rad.ObjDrwAtr(self.radia_id, color)

    def _log(self, msg: str, verbosity_level: int = 1):
        """Log message if rank == 0 and verbosity sufficient."""
        if self.rank <= 0 and self.verbosity >= verbosity_level:
            print(f"  {msg}", flush=True)


class AnnularWedgeComponent(GeometricComponent):
    """Annular wedge (wall or lid) with optional window cutout and segmentation."""

    def __init__(self,
                 config: CyclotronConfig,
                 params: Dict[str, Any],
                 material_id: Optional[int] = None,
                 rank: int = 0,
                 verbosity: int = 0):
        """
        Initialize annular wedge component.

        :param config: CyclotronConfig object
        :param params: Dict with keys:
            - 'outer_radius_mm': Outer radius
            - 'inner_radius_mm': Inner radius
            - 'height_mm': Thickness
            - 'z_offset_mm': Z position of top surface
            - 'segmentation': [r_div, theta_div, z_div]
            - 'include_window': Boolean, if True, cuts rectangular window
            - 'window_width_mm': Width of window (if include_window=True)
            - 'component_name': For logging
        :param material_id: Radia material ID
        :param rank: MPI rank
        :param verbosity: Verbosity level
        """
        super().__init__(config, material_id, rank, verbosity)
        self.params = params

    def build(self) -> int:
        """Build annular wedge."""
        name = self.params.get('component_name', 'AnnularWedge')
        self._log(f"Building {name}...")

        or_mm = self.params['outer_radius_mm']
        ir_mm = self.params['inner_radius_mm']
        h_mm = self.params['height_mm']
        z_top = self.params['z_offset_mm']
        z_bottom = z_top - h_mm
        include_window = self.params.get('include_window', False)

        # Determine angular resolution to use
        if self.params.get('use_pole_resolution', False):
            ang_res_deg = self.config.pole.angular_resolution_deg
            ang_res = int(np.ceil(self.config.pole.full_angle_deg / 2.0 / ang_res_deg))
            yoke_angle_deg = self.config.pole.full_angle_deg / 2.0
        else:
            yoke_angle_deg = self.config.geometry.yoke_build_angle_deg
            ang_res = self.config.geometry.angular_resolution

        # Create full wedge polygon
        angles = np.linspace(0.0, np.deg2rad(yoke_angle_deg), ang_res + 1)
        wedge_pgn = PolygonBuilder.arc_polygon(or_mm, angles, origin=True)

        # Create 3D object (prism)
        wedge = rad.ObjMltExtPgn([[wedge_pgn, z_top], [wedge_pgn, z_bottom]])

        # Cut rectangular window if requested
        if include_window:
            window_width = self.params.get('window_width_mm', 300.0)
            cut_pln_p = [0.5 * window_width * np.sqrt(2.0), 0, 0]
            cut_pln_v = [1, -1, 0]
            result = rad.ObjCutMag(wedge, cut_pln_p, cut_pln_v, 'Frame->Lab')
            wedge = result[1] if isinstance(result, list) and len(result) > 1 else result

        # Cut out inner bore using angular resolution
        seg_strategy = SegmentationStrategy([1, ang_res, 1])
        seg_strategy.apply_center_bore_cutout(wedge, ir_mm)

        # Filter: keep outside bore radius
        wedge = ObjectFilter.keep_outside_radius(
            wedge, ir_mm,
            verbose=(self.verbosity >= 2)
        )

        self.radia_id = wedge
        self._log(f"{name} built (ID={self.radia_id})")
        return self.radia_id

    def segment(self,
                segmentation: List[float],
                ratios: List[float]=None,
                size: bool=False,
                coords: str='cyl',
                frame: str='Lab'):
        """
        # Apply additional "fine" segmentation if needed
        :param segmentation:
        :param size: True if segmentation contains a list of sizes rather than number of segments
        :param ratios:
        :param coords: 'cart|cyl'
        :param frame: 'Loc|Lab|LabTot'
        :return: self.radia_id
        """
        if size:
            print("Segmentation by average segment size is buggy and not currently implemented.")
            return self.radia_id

        if ratios is None:
            ratios = [1, 1, 1]

        size_str = 'kxkykz->Size;' if size else 'kxkykz->Numb;'
        frame_str = 'Frame->'+frame

        option_str = size_str + frame_str

        nesting = 1

        if coords == 'cyl':

            # Apply z segmentation only
            if segmentation[2] != 1:
                nesting += 1

                seg_strategy = SegmentationStrategy([1, 1, segmentation[2]],
                                                    [1, 1, ratios[2]])
                seg_strategy.apply_cylindrical(
                    self.radia_id,
                    options=option_str)

            # TODO: Apply additional fine segmentation of theta

            # Apply r segmentation only and force ratio parameters to 1
            if segmentation[0] != 1:

                nesting += 1

                or_mm = self.params['outer_radius_mm']
                ir_mm = self.params['inner_radius_mm']
                r_seg_len = (or_mm - ir_mm) / segmentation[0]
                seg_strategy = SegmentationStrategy([2, 1, 1], [1, 1, 1])
                seg_strategy.apply_cylindrical(
                    self.radia_id,
                    radial_direction=[r_seg_len, 0, 0],
                    options=option_str)

        else:
            seg_strategy = SegmentationStrategy(segmentation, ratios)
            seg_strategy.apply_cartesian(
                self.radia_id)

        if nesting >= 2:
            self.radia_id = ObjectFilter.flatten_nested(self.radia_id, max_depth=nesting,
                                                        verbose=(self.verbosity >= 2))

        return self.radia_id


class LidUpperComponent(GeometricComponent):
    """Upper lid with optional RF stem hole."""

    def __init__(self,
                 config: CyclotronConfig,
                 material_id: Optional[int] = None,
                 rank: int = 0,
                 verbosity: int = 0):
        """
        Initialize upper lid component.

        :param config: CyclotronConfig object
        :param material_id: Radia material ID
        :param rank: MPI rank
        :param verbosity: Verbosity level
        """
        super().__init__(config, material_id, rank, verbosity)

    def build(self) -> int:
        """Build upper lid."""
        self._log("Building upper lid...")

        cfg = self.config.lid_upper
        geom_cfg = self.config.geometry
        yoke_h = self.config.yoke.height_mm
        lid_lower_h = self.config.lid_lower.height_mm

        lid_or1 = cfg.outer_radius_mm_1
        lid_or2 = cfg.outer_radius_mm_2
        lid_ir = cfg.inner_radius_mm
        lid_h = cfg.height_mm
        yoke_angle_deg = geom_cfg.yoke_build_angle_deg
        ang_res = geom_cfg.angular_resolution
        seg_div = cfg.segmentation

        cut_rf_hole = cfg.cut_out_rf_stem_hole
        hole_diam = cfg.hole_diameter_mm
        hole_center = cfg.hole_center_xy
        hole_radius = hole_diam / 2.0

        zl = -(yoke_h + lid_lower_h)
        zu = zl - lid_h

        # Create wedge polygons using angular_resolution
        lid_angles = np.linspace(0.0, np.deg2rad(yoke_angle_deg), ang_res + 1)
        arc1 = np.array([lid_or1 * np.cos(lid_angles),
                         lid_or1 * np.sin(lid_angles)]).T
        arc2 = np.array([lid_or2 * np.cos(lid_angles),
                         lid_or2 * np.sin(lid_angles)]).T
        origin = np.array([[0.0, 0.0]])

        lower_pgn = np.vstack([arc1, origin]).tolist()
        upper_pgn = np.vstack([arc2, origin]).tolist()

        # Create base lid
        lid_temp = rad.ObjMltExtPgn([[lower_pgn, zl], [upper_pgn, zu]])

        # Cut RF stem hole if requested
        if cut_rf_hole:
            seg_strategy = SegmentationStrategy([2, ang_res, 1], [2000, 1, 1])
            seg_strategy.apply_cylindrical(
                lid_temp,
                [hole_center[0], hole_center[1], 0],
                [hole_center[0] + hole_radius, hole_center[1], 0],
                options='Frame->Lab'
            )

            lid_temp = ObjectFilter.keep_outside_circle(
                lid_temp, hole_center, hole_radius,
                verbose=(self.verbosity >= 2)
            )

        # Cut center bore
        # Cut out inner bore using angular resolution
        seg_strategy = SegmentationStrategy([2, ang_res, 1], [2000, 1, 1])
        seg_strategy.apply_center_bore_cutout(lid_temp, lid_ir)

        # Filter: keep outside center bore
        lid_temp = ObjectFilter.keep_outside_radius(
            lid_temp, lid_ir,
            verbose=(self.verbosity >= 2)
        )

        self.radia_id = lid_temp
        self._log(f"Upper lid built (ID={self.radia_id})")
        return self.radia_id


    def segment(self,
                segmentation: List[float],
                ratios: List[float]=None,
                size: bool=False,
                coords: str='cyl',
                frame: str='Lab'):
        """
        # Apply additional "fine" segmentation if needed
        :param segmentation:
        :param size: True if segmentation contains a list of sizes rather than number of segments
        :param ratios:
        :param coords: 'cart|cyl'
        :param frame: 'Loc|Lab|LabTot'
        :return: self.radia_id
        """
        if size:
            print("Segmentation by average segment size is buggy and not currently implemented.")
            return self.radia_id

        if ratios is None:
            ratios = [1, 1, 1]

        size_str = 'kxkykz->Size;' if size else 'kxkykz->Numb;'
        frame_str = 'Frame->'+frame

        option_str = size_str + frame_str

        cfg = self.config.lid_upper
        or_mm = cfg.outer_radius_mm_1
        ir_mm = cfg.inner_radius_mm

        nesting = 1

        if coords == 'cyl':

            # Apply z segmentation only
            if segmentation[2] != 1:
                nesting += 1

                seg_strategy = SegmentationStrategy([1, 1, segmentation[2]],
                                                    [1, 1, ratios[2]])
                seg_strategy.apply_cylindrical(
                    self.radia_id,
                    options=option_str)

            # TODO: Apply additional fine segmentation of theta

            # Apply r segmentation only and force ratio parameters to 1
            if segmentation[0] != 1:

                nesting += 1

                r_seg_len = (or_mm - ir_mm) / segmentation[0]
                seg_strategy = SegmentationStrategy([2, 1, 1], [1, 1, 1])
                seg_strategy.apply_cylindrical(
                    self.radia_id,
                    radial_direction=[r_seg_len, 0, 0],
                    options=option_str)

        else:
            seg_strategy = SegmentationStrategy(segmentation, ratios)
            seg_strategy.apply_cartesian(
                self.radia_id)

        if nesting >= 2:
            self.radia_id = ObjectFilter.flatten_nested(self.radia_id, max_depth=nesting,
                                                        verbose=(self.verbosity >= 2))

        return self.radia_id


class TopShimComponent(GeometricComponent):
    """Top shim surface with angled cuts."""

    def __init__(self,
                 config: CyclotronConfig,
                 pole_shape: PoleShape,
                 material_id: Optional[int] = None,
                 rank: int = 0,
                 verbosity: int = 0):
        super().__init__(config, material_id, rank, verbosity)
        self.pole_shape = pole_shape

    def build(self) -> int:
        """Build top shimming surface."""
        self._log("Building top shim...")

        cfg = self.config.top_shim
        pole_cfg = self.config.pole
        geom_cfg = self.config.geometry
        yoke_h = self.config.yoke.height_mm
        lid_lower_h = self.config.lid_lower.height_mm

        pole_or = pole_cfg.outer_radius_mm
        pole_ir = pole_cfg.inner_radius_mm
        pole_h = pole_cfg.height_mm
        pole_full_angle = pole_cfg.full_angle_deg
        pole_half_angle = pole_full_angle / 2.0

        pole_zs = -(yoke_h + lid_lower_h) + pole_h

        # Get shim elevations and create radial grid
        shim_elevations_temp = self.pole_shape.get_top_offsets_mm()
        shim_n_segs_temp = self.pole_shape.num_segments

        # Now generate num_segments * num_subdivisions elevations by interpolation
        # This is a workaround because Radia's radial segmentation is broken
        shim_n_segs = shim_n_segs_temp * cfg.segmentation[0]
        shim_elevations_interp = interp1d(np.linspace(pole_ir, pole_or, shim_n_segs_temp + 1), shim_elevations_temp)
        r_shim = np.linspace(pole_ir, pole_or, shim_n_segs + 1)
        shim_elevations = shim_elevations_interp(r_shim)

        # Use angular_resolution for arc approximation
        ang_res = int(np.ceil(pole_half_angle / cfg.angular_resolution_deg))
        pole_angles = np.linspace(0.0, np.deg2rad(pole_half_angle), ang_res + 1)

        # Create x/y grids
        top_shim_x, top_shim_y = PolygonBuilder.rectangular_grid(r_shim, pole_angles)

        # Build blocks
        top_shim_segments = []
        for i in range(top_shim_x.shape[0] - 1):
            for j in range(top_shim_x.shape[1] - 1):
                block = self._build_shim_block(
                    i, j,
                    top_shim_x, top_shim_y,
                    shim_elevations,
                    r_shim,
                    pole_zs
                )
                if block:
                    top_shim_segments.append(block)

        self.radia_id = rad.ObjCnt(top_shim_segments)
        self._log(f"Top shim built ({len(top_shim_segments)} blocks, ID={self.radia_id})")
        return self.radia_id


    def segment(self,
                segmentation: List[float],
                ratios: List[float] = None,
                size: bool = False,
                coords: str = 'cyl',
                frame: str = 'Lab'):
        """
        # Apply additional "fine" segmentation if needed
        :param segmentation:
        :param size: True if segmentation contains a list of sizes rather than number of segments
        :param ratios:
        :param coords: 'cart|cyl'
        :param frame: 'Loc|Lab|LabTot'
        :return: self.radia_id
        """
        pass


    def _build_shim_block(self, i, j, x_grid, y_grid, elevations, r_shim, pole_zs):
        """Build a single shim block with optional angled cut."""
        cfg = self.config.top_shim
        pole_ze = pole_zs - self.config.pole.height_mm

        # Create polygon
        polygon = [
            [x_grid[i, j], y_grid[i, j]],
            [x_grid[i + 1, j], y_grid[i + 1, j]],
            [x_grid[i + 1, j + 1], y_grid[i + 1, j + 1]],
            [x_grid[i, j + 1], y_grid[i, j + 1]]
        ]

        block_h = np.max([elevations[i:i + 2]])

        # Create prism
        block = rad.ObjMltExtPgn([
            [polygon, pole_zs],
            [polygon, pole_zs + block_h]
        ])

        # Check if angled cut needed
        if not np.isclose(elevations[i], elevations[i + 1]):
            # Three points on angled surface
            p1 = np.array([x_grid[i, j], y_grid[i, j], pole_zs + elevations[i]])
            p2 = np.array([x_grid[i + 1, j], y_grid[i + 1, j], pole_zs + elevations[i + 1]])
            p3 = np.array([x_grid[i, j + 1], y_grid[i, j + 1], pole_zs + elevations[i]])

            v1 = p2 - p1
            v2 = p3 - p1
            v_norm = np.cross(v1, v2)

            result = rad.ObjCutMag(block, p1, v_norm, 'Frame->Lab')
            block = result if not isinstance(result, list) else result[0]

        # Apply segmentation
        self._apply_block_segmentation(block, r_shim)

        return block

    def _apply_block_segmentation(self, block: int, r_shim: np.ndarray):
        """Apply standard segmentation to a top shim block."""
        cfg = self.config.top_shim
        seg_div = cfg.segmentation

        # --- r-division is now applied by creating more blocks! --- #

        # seg_strategy = SegmentationStrategy(seg_div)
        #
        # r_seg_len = (r_shim[1] - r_shim[0]) / seg_div[0] if seg_div[0] > 1 else 1.0
        #
        # # Cylindrical division
        # seg_strategy.apply_cylindrical(
        #     block,
        #     [0, 0, 0],
        #     [r_seg_len, 0, 0],
        #     options='Frame->Lab'
        # )

        # Z-division if needed
        if seg_div[2] > 1:
            rad.ObjDivMag(block, [[1, 1], [1, 1], [seg_div[2], 0.1]], 'Frame->Lab')

class SideShimComponent(GeometricComponent):
    """
    Side shim with angular offset and complex boundary segment handling.
    """

    def __init__(self,
                 config: CyclotronConfig,
                 pole_shape: PoleShape,
                 material_id: Optional[int] = None,
                 rank: int = 0,
                 verbosity: int = 0):
        super().__init__(config, material_id, rank, verbosity)
        self.pole_shape = pole_shape


    @staticmethod
    def polygon_area_2d(vertices):
        """
        Calculate area of a 2D polygon using the Shoelace formula.
        Works for 3-5 vertices (triangles, quads, pentagons).

        :param vertices: List of [x, y] lists or Nx2 array
        :return: Absolute area in mm²
        """
        vertices = np.asarray(vertices)
        if len(vertices) < 3:
            return 0.0

        x = vertices[:, 0]
        y = vertices[:, 1]

        # Shoelace formula
        area = 0.5 * abs(sum(x[i] * y[(i + 1) % len(vertices)] -
                             x[(i + 1) % len(vertices)] * y[i]
                             for i in range(len(vertices))))
        return area


    @staticmethod
    def is_valid_polygon(vertices, min_area=MIN_POLYGON_AREA_MM2, verbosity=0):
        """
        Check if polygon has sufficient area (not degenerate).

        :param vertices: List of [x, y] lists or Nx2 array
        :param min_area: Minimum allowed area in mm²
        :param verbosity: Verbosity level
        :return: (is_valid, area)
        """
        area = SideShimComponent.polygon_area_2d(vertices)

        if area < min_area:

            return False, area

        return True, area


    def build(self) -> int:
        """Build side shimming surface."""
        self._log("Building side shim...")

        cfg = self.config.side_shim
        pole_cfg = self.config.pole
        geom_cfg = self.config.geometry
        yoke_h = self.config.yoke.height_mm
        lid_lower_h = self.config.lid_lower.height_mm

        pole_or = pole_cfg.outer_radius_mm
        pole_ir = pole_cfg.inner_radius_mm
        pole_h = pole_cfg.height_mm
        pole_full_angle = pole_cfg.full_angle_deg
        pole_half_angle = pole_full_angle / 2.0

        pole_zs = -(yoke_h + lid_lower_h) + pole_h
        pole_ze = -(yoke_h + lid_lower_h)

        # Get shim elevations and create radial grid
        shim_elevations_temp = self.pole_shape.get_top_offsets_mm()
        shim_n_segs_temp = self.pole_shape.num_segments
        ang_shim_rel_temp = self.pole_shape.get_side_offsets_rad()

        # Now generate num_segments * num_subdivisions elevations by interpolation
        # This is a workaround because Radia's radial segmentation is broken
        shim_n_segs = shim_n_segs_temp * cfg.segmentation[0]
        shim_elevations_interp = interp1d(np.linspace(pole_ir, pole_or, shim_n_segs_temp + 1), shim_elevations_temp)
        ang_shim_rel_interp = interp1d(np.linspace(pole_ir, pole_or, shim_n_segs_temp + 1), ang_shim_rel_temp)
        r_shim = np.linspace(pole_ir, pole_or, shim_n_segs + 1)
        shim_elevations = shim_elevations_interp(r_shim)
        ang_shim_rel = ang_shim_rel_interp(r_shim)

        # Get shim parameters
        # shim_elevations = self.pole_shape.get_top_offsets_mm()
        # shim_n_segs = self.pole_shape.num_segments
        # ang_shim_rel = self.pole_shape.get_side_offsets_rad()

        # Create radial array
        # r_shim = np.linspace(pole_ir, pole_or, shim_n_segs + 1)

        # Use angular_resolution for arc approximation
        ang_res = int(np.ceil(pole_half_angle / pole_cfg.angular_resolution_deg))
        pole_angles = np.linspace(0.0, np.deg2rad(pole_half_angle), ang_res + 1)

        # Absolute angles for shim (relative + pole edge angle)
        ang_shim_abs = ang_shim_rel + pole_angles[-1]

        # Build blocks
        side_shim_segments = []
        for i in range(shim_n_segs):
            blocks = self._build_segment_blocks(
                i,
                r_shim,
                pole_angles,
                ang_shim_abs,
                shim_elevations,
                pole_zs,
                pole_ze,
                cfg.angular_resolution_deg
            )
            side_shim_segments.extend(blocks)

        self.radia_id = rad.ObjCnt(side_shim_segments)
        self._log(f"Side shim built ({len(side_shim_segments)} blocks, ID={self.radia_id})")
        return self.radia_id


    # def segment(self,
    #             segmentation: List[float],
    #             ratios: List[float]=None,
    #             size: bool=False,
    #             coords: str='cyl',
    #             frame: str='Lab'):
    #     """
    #     # Apply additional "fine" segmentation if needed
    #     :param segmentation:
    #     :param size: True if segmentation contains a list of sizes rather than number of segments
    #     :param ratios:
    #     :param coords: 'cart|cyl'
    #     :param frame: 'Loc|Lab|LabTot'
    #     :return: self.radia_id
    #     """
    #     if size:
    #         print("Segmentation by average segment size is buggy and not currently implemented.")
    #         return self.radia_id
    #
    #     if ratios is None:
    #         ratios = [1, 1, 1]
    #
    #     size_str = 'kxkykz->Size;' if size else 'kxkykz->Numb;'
    #     frame_str = 'Frame->'+frame
    #
    #     option_str = size_str + frame_str
    #
    #     nesting = 1
    #
    #     if coords == 'cyl':
    #
    #         # Apply z segmentation only
    #         if segmentation[2] != 1:
    #             nesting += 1
    #
    #             seg_strategy = SegmentationStrategy([1, 1, segmentation[2]],
    #                                                 [1, 1, ratios[2]])
    #             seg_strategy.apply_cylindrical(
    #                 self.radia_id,
    #                 options=option_str)
    #
    #         # TODO: Apply additional fine segmentation of theta
    #
    #         # Apply r segmentation only and force ratio parameters to 1
    #         if segmentation[0] != 1:
    #
    #             nesting += 1
    #
    #             or_mm = self.config.pole.outer_radius_mm
    #             ir_mm = self.config.pole.inner_radius_mm
    #
    #             r_seg_len = (or_mm - ir_mm) / segmentation[0]
    #             seg_strategy = SegmentationStrategy([2, 1, 1], [1, 1, 1])
    #             seg_strategy.apply_cylindrical(
    #                 self.radia_id,
    #                 radial_direction=[r_seg_len, 0, 0],
    #                 options=option_str)
    #
    #     else:
    #         seg_strategy = SegmentationStrategy(segmentation, ratios)
    #         seg_strategy.apply_cartesian(
    #             self.radia_id)
    #
    #     if nesting >= 2:
    #         self.radia_id = ObjectFilter.flatten_nested(self.radia_id, max_depth=nesting,
    #                                                     verbose=(self.verbosity >= 2))
    #
    #     return self.radia_id


    def _build_segment_blocks(self,
                              segment_idx: int,
                              r_shim: np.ndarray,
                              pole_angles: np.ndarray,
                              ang_shim_abs: np.ndarray,
                              shim_elevations: np.ndarray,
                              pole_zs: float,
                              pole_ze: float,
                              ang_res_deg: float) -> List[int]:
        """
        Build all radial-angular blocks for one radial segment.

        Handles regular, boundary (BD), and extra segments intelligently.
        """

        blocks = []
        cfg = self.config.side_shim
        seg_div = cfg.segmentation

        # Generate angular positions for inner and outer radii
        _n_angs_ir = round((ang_shim_abs[segment_idx] - pole_angles[-1]) /
                           np.deg2rad(ang_res_deg), 5)
        _angs_ir = np.asarray(pole_angles[-1] + np.arange(_n_angs_ir) *
                              np.deg2rad(ang_res_deg)).tolist() + [ang_shim_abs[segment_idx]]

        _n_angs_or = round((ang_shim_abs[segment_idx + 1] - pole_angles[-1]) /
                           np.deg2rad(ang_res_deg), 5)
        _angs_or = np.asarray(pole_angles[-1] + np.arange(_n_angs_or) *
                              np.deg2rad(ang_res_deg)).tolist() + [ang_shim_abs[segment_idx + 1]]

        # Create line for plane intersections
        p1 = Point3D(r_shim[segment_idx] * np.cos(_angs_ir[-1]),
                     r_shim[segment_idx] * np.sin(_angs_ir[-1]),
                     pole_zs + shim_elevations[segment_idx])
        p2 = Point3D(r_shim[segment_idx + 1] * np.cos(_angs_or[-1]),
                     r_shim[segment_idx + 1] * np.sin(_angs_or[-1]),
                     pole_zs + shim_elevations[segment_idx + 1])
        l1 = Line3D(p1, p2)

        # Determine segment classification
        len_ir = len(_angs_ir)
        len_or = len(_angs_or)
        _n_ang_segs = max(len_ir, len_or) - 1

        long_arc_idx = 0 if len_ir > len_or else 1
        short_arc_idx = 1 - long_arc_idx
        high_elev_idx = 0 if shim_elevations[segment_idx] > shim_elevations[segment_idx + 1] else 1
        # low_elev_idx = 1 - high_elev_idx

        _angs_iror = [_angs_ir, _angs_or]
        _rads_iror = [r_shim[segment_idx], r_shim[segment_idx + 1]]
        _z_iror = [pole_zs + shim_elevations[segment_idx],
                   pole_zs + shim_elevations[segment_idx + 1]]

        block_h = max(shim_elevations[segment_idx:segment_idx + 2])

        # Determine boundary/extra segment counts
        # len_long_arc = len(_angs_iror[long_arc_idx])
        len_short_arc = len(_angs_iror[short_arc_idx])

        if len_ir != len_or:
            _n_bd_segs = 0 if np.isclose(np.diff(_angs_iror[short_arc_idx])[-1],
                                         np.diff(_angs_iror[long_arc_idx])[len_short_arc - 2]) else 1
            _n_reg_segs = len_short_arc - _n_bd_segs - 1
            _n_extra_segs = _n_ang_segs - _n_reg_segs - _n_bd_segs
        else:
            _n_reg_segs = _n_ang_segs
            _n_bd_segs = 0
            _n_extra_segs = 0

        # Build each angular subsegment
        for j in range(_n_ang_segs):

            if j < _n_reg_segs:
                # Regular polygon
                block = self._build_regular_block(
                    j, _angs_ir, _angs_or, _rads_iror,
                    pole_zs, shim_elevations[segment_idx:segment_idx + 2],
                    block_h, pole_ze
                )
            elif j < _n_reg_segs + _n_bd_segs:
                # Boundary segment
                block = self._build_boundary_block(
                    j, _angs_iror, _rads_iror, _z_iror,
                    long_arc_idx, short_arc_idx,
                    l1, pole_zs, block_h, pole_ze
                )
            else:
                # Extra segment
                block = self._build_extra_block(
                    j, _angs_iror, _rads_iror, _z_iror,
                    long_arc_idx, short_arc_idx,
                    l1, pole_zs, block_h, pole_ze,
                    _n_ang_segs
                )

            if block:
                blocks.append(block)

        return blocks

    def _build_regular_block(self,
                             j: int,
                             angs_ir: List[float],
                             angs_or: List[float],
                             rads_iror: List[float],
                             pole_zs: float,
                             elevations: np.ndarray,
                             block_h: float,
                             pole_ze: float) -> Optional[int]:
        """Build a regular (non-boundary) angular subsegment."""
        if self.rank <= 0 and self.verbosity >= 2:
            print(f"Building Side Shim Regular Block (Element {j})", flush=True)

        # cfg = self.config.side_shim
        # seg_div = cfg.segmentation

        # Create polygon
        polygon = [
            [rads_iror[0] * np.cos(angs_ir[j]), rads_iror[0] * np.sin(angs_ir[j])],
            [rads_iror[1] * np.cos(angs_or[j]), rads_iror[1] * np.sin(angs_or[j])],
            [rads_iror[1] * np.cos(angs_or[j + 1]), rads_iror[1] * np.sin(angs_or[j + 1])],
            [rads_iror[0] * np.cos(angs_ir[j + 1]), rads_iror[0] * np.sin(angs_ir[j + 1])]
        ]

        # ===== CHECK POLYGON VALIDITY =====
        is_valid, area = self.is_valid_polygon(polygon)
        if not is_valid:
            if self.rank <= 0 and self.verbosity >= 1:
                print(f"  Skipping extra block {j}: Degenerate polygon (area = {area:.2e} mm²)", flush=True)
            return None

        bottom_polygon = np.array(polygon)[::-1].tolist()

        # Check if angled cut needed
        if not np.isclose(elevations[0], elevations[1]):

            # Create prism with extra heigh so plane cut will not be ambiguous in edge cases
            block = rad.ObjMltExtPgn([
                [polygon, pole_zs + 10000],
                [bottom_polygon, pole_ze]
            ])

            p1 = np.array([polygon[0][0], polygon[0][1], pole_zs + elevations[0]])
            p2 = np.array([polygon[1][0], polygon[1][1], pole_zs + elevations[1]])
            p3 = np.array([polygon[3][0], polygon[3][1], pole_zs + elevations[0]])

            v1 = p2 - p1
            v2 = p3 - p1
            v_norm = np.cross(v1, v2)

            result = rad.ObjCutMag(block, p2, -v_norm, 'Frame->Lab')

            if len(result) == 2:
                obj1, obj2 = result

                # Select the lower part of the cut
                z_cent1 = rad.ObjM(obj1)[0][2]
                z_cent2 = rad.ObjM(obj2)[0][2]
                block = obj1 if z_cent1 < z_cent2 else obj2

        else:
            # Create prism with correct height
            block = rad.ObjMltExtPgn([
                [polygon, pole_zs + block_h],
                [bottom_polygon, pole_ze]
            ])

        # Apply segmentation (Note that radia replaces uses the old object ID for the created container,
        # so we don't need to pass back block
        self._apply_block_segmentation(block, rads_iror)

        return block

    def _build_boundary_block(self,
                              j: int,
                              angs_iror: List[List[float]],
                              rads_iror: List[float],
                              z_iror: List[float],
                              long_arc_idx: int,
                              short_arc_idx: int,
                              line_l1,
                              pole_zs: float,
                              block_h: float,
                              pole_ze: float) -> Optional[int]:
        """Build a boundary segment where arc lengths differ."""
        if self.rank <= 0 and self.verbosity >= 2:
            print(f"Building Side Shim BD Block (Element {j})", flush=True)

        from sympy import Point3D, Plane
        from sympy.geometry import intersection

        cfg = self.config.side_shim
        seg_div = cfg.segmentation

        # Point on longer arc
        p3a = Point3D(
            rads_iror[long_arc_idx] * np.cos(angs_iror[long_arc_idx][j + 1]),
            rads_iror[long_arc_idx] * np.sin(angs_iror[long_arc_idx][j + 1]),
            z_iror[long_arc_idx]
        )

        # Create plane through p3a aligned with boundary
        p4a = Point3D(
            (rads_iror[long_arc_idx] - 50) * np.cos(angs_iror[long_arc_idx][j + 1]),
            (rads_iror[long_arc_idx] - 50) * np.sin(angs_iror[long_arc_idx][j + 1]),
            z_iror[long_arc_idx]
        )
        p5a = Point3D(
            rads_iror[long_arc_idx] * np.cos(angs_iror[long_arc_idx][j + 1]),
            rads_iror[long_arc_idx] * np.sin(angs_iror[long_arc_idx][j + 1]),
            z_iror[long_arc_idx] + 50
        )

        rad_plane1a = Plane(p3a, p4a, p5a)
        p_intersect1a = intersection(rad_plane1a, line_l1)[0]

        # Create polygon
        polygon = [
            [rads_iror[long_arc_idx] * np.cos(angs_iror[long_arc_idx][j]),
             rads_iror[long_arc_idx] * np.sin(angs_iror[long_arc_idx][j])],
            [p3a.x, p3a.y],
            [p_intersect1a.x, p_intersect1a.y],
            [rads_iror[short_arc_idx] * np.cos(angs_iror[short_arc_idx][j + 1]),
             rads_iror[short_arc_idx] * np.sin(angs_iror[short_arc_idx][j + 1])],
            [rads_iror[short_arc_idx] * np.cos(angs_iror[short_arc_idx][j]),
             rads_iror[short_arc_idx] * np.sin(angs_iror[short_arc_idx][j])]
        ]

        # ===== CHECK POLYGON VALIDITY =====
        is_valid, area = self.is_valid_polygon(polygon)
        if not is_valid:
            if self.rank <= 0 and self.verbosity >= 1:
                print(f"  Skipping extra block {j}: degenerate polygon (area={area:.2e})", flush=True)
            return None

        bottom_polygon = np.array(polygon)[::-1].tolist()

        # # Create prism
        # block = rad.ObjMltExtPgn([
        #     [polygon, pole_zs + block_h],
        #     [bottom_polygon, pole_ze]
        # ])

        # Check if angled cut needed
        if not np.isclose(z_iror[0], z_iror[1]):

            # Create prism with extra height for cut
            block = rad.ObjMltExtPgn([
                [polygon, pole_zs + 10000],
                [bottom_polygon, pole_ze]
            ])

            r_ir = rads_iror[0]
            r_or = rads_iror[1]
            ang_ir_0 = angs_iror[long_arc_idx][j]
            ang_ir_1 = angs_iror[long_arc_idx][j+1]
            ang_or_1 = angs_iror[long_arc_idx][j+1]
            z_ir = z_iror[0]
            z_or = z_iror[1]

            p1 = np.array([r_ir * np.cos(ang_ir_0),
                           r_ir * np.sin(ang_ir_0),
                           z_ir])
            p2 = np.array([r_ir * np.cos(ang_ir_1),
                           r_ir * np.sin(ang_ir_1),
                           z_ir])
            p3 = np.array([r_or * np.cos(ang_or_1),
                           r_or * np.sin(ang_or_1),
                           z_or])

            v1 = p2 - p1
            v2 = p3 - p1
            v_norm = np.cross(v1, v2)

            result = rad.ObjCutMag(block, p2, -v_norm, 'Frame->Lab')

            if len(result) == 2:
                obj1, obj2 = result

                # Select the lower part of the cut
                z_cent1 = rad.ObjM(obj1)[0][2]
                z_cent2 = rad.ObjM(obj2)[0][2]
                block = obj1 if z_cent1 < z_cent2 else obj2

        else:
            # Create prism with flat top
            block = rad.ObjMltExtPgn([
                [polygon, pole_zs + block_h],
                [bottom_polygon, pole_ze]
            ])

        # Apply segmentation (Note that radia replaces uses the old object ID for the created container,
        # so we don't need to pass back block
        self._apply_block_segmentation(block, rads_iror)

        return block

    def _build_extra_block(self,
                           j: int,
                           angs_iror: List[List[float]],
                           rads_iror: List[float],
                           z_iror: List[float],
                           long_arc_idx: int,
                           short_arc_idx: int,
                           line_l1,
                           pole_zs: float,
                           block_h: float,
                           pole_ze: float,
                           n_ang_segs: int) -> Optional[int]:
        """Build an extra segment where arc length mismatch is large."""
        if self.rank <= 0 and self.verbosity >= 2:
            print(f"Building Side Shim Extra Block (Element {j})", flush=True)

        # cfg = self.config.side_shim
        # seg_div = cfg.segmentation

        # Two consecutive points on longer arc
        p3 = Point3D(
            rads_iror[long_arc_idx] * np.cos(angs_iror[long_arc_idx][j]),
            rads_iror[long_arc_idx] * np.sin(angs_iror[long_arc_idx][j]),
            z_iror[long_arc_idx]
        )

        p3a = Point3D(
            rads_iror[long_arc_idx] * np.cos(angs_iror[long_arc_idx][j + 1]),
            rads_iror[long_arc_idx] * np.sin(angs_iror[long_arc_idx][j + 1]),
            z_iror[long_arc_idx]
        )

        # Create planes for both boundaries
        p4 = Point3D(
            (rads_iror[long_arc_idx] - 50) * np.cos(angs_iror[long_arc_idx][j]),
            (rads_iror[long_arc_idx] - 50) * np.sin(angs_iror[long_arc_idx][j]),
            z_iror[long_arc_idx]
        )
        p5 = Point3D(
            rads_iror[long_arc_idx] * np.cos(angs_iror[long_arc_idx][j]),
            rads_iror[long_arc_idx] * np.sin(angs_iror[long_arc_idx][j]),
            z_iror[long_arc_idx] + 50
        )

        p4a = Point3D(
            (rads_iror[long_arc_idx] - 50) * np.cos(angs_iror[long_arc_idx][j + 1]),
            (rads_iror[long_arc_idx] - 50) * np.sin(angs_iror[long_arc_idx][j + 1]),
            z_iror[long_arc_idx]
        )
        p5a = Point3D(
            rads_iror[long_arc_idx] * np.cos(angs_iror[long_arc_idx][j + 1]),
            rads_iror[long_arc_idx] * np.sin(angs_iror[long_arc_idx][j + 1]),
            z_iror[long_arc_idx] + 50
        )

        rad_plane1 = Plane(p3, p4, p5)
        rad_plane1a = Plane(p3a, p4a, p5a)

        p_intersect1 = intersection(rad_plane1, line_l1)[0]
        p_intersect1a = intersection(rad_plane1a, line_l1)[0]

        # Create polygon (triangle for last, quad otherwise)
        if j == n_ang_segs - 1:
            # Final triangle
            polygon = [
                [p3.x, p3.y],
                [p3a.x, p3a.y],
                [p_intersect1.x, p_intersect1.y]
            ]
        else:
            # Regular quad
            polygon = [
                [p3.x, p3.y],
                [p3a.x, p3a.y],
                [p_intersect1a.x, p_intersect1a.y],
                [p_intersect1.x, p_intersect1.y]
            ]

        # ===== CHECK POLYGON VALIDITY =====
        is_valid, area = self.is_valid_polygon(polygon)
        if not is_valid:
            if self.rank <= 0 and self.verbosity >= 1:
                print(f"  [SideShim] Skipping degenerate polygon in extra block {j}: "
                      f"area = {area:.2e} mm² < {MIN_POLYGON_AREA_MM2} mm²", flush=True)
            return None

        bottom_polygon = np.array(polygon)[::-1].tolist()

        # Check if angled cut needed
        if not np.isclose(z_iror[0], z_iror[1]):

            # Create prism with large extra height for angle cut
            try:
                block = rad.ObjMltExtPgn([
                    [polygon, pole_zs + 10000],
                    [bottom_polygon, pole_ze]
                ])
            except RuntimeError:
                print("Radia runtime error is polygon a line?")
                for p in polygon:
                    print(N(p[0]), N(p[1]))
                exit()

            r_ir = rads_iror[0]
            r_or = rads_iror[1]
            ang_ir_0 = angs_iror[long_arc_idx][j]
            ang_ir_1 = angs_iror[long_arc_idx][j+1]
            ang_or_1 = angs_iror[long_arc_idx][j+1]
            z_ir = z_iror[0]
            z_or = z_iror[1]

            p1 = np.array([r_ir * np.cos(ang_ir_0),
                           r_ir * np.sin(ang_ir_0),
                           z_ir])
            p2 = np.array([r_ir * np.cos(ang_ir_1),
                           r_ir * np.sin(ang_ir_1),
                           z_ir])
            p3 = np.array([r_or * np.cos(ang_or_1),
                           r_or * np.sin(ang_or_1),
                           z_or])

            v1 = p2 - p1
            v2 = p3 - p1
            v_norm = np.cross(v1, v2)

            result = rad.ObjCutMag(block, p2, -v_norm, 'Frame->Lab')

            if len(result) == 2:
                obj1, obj2 = result

                # Select the lower part of the cut
                z_cent1 = rad.ObjM(obj1)[0][2]
                z_cent2 = rad.ObjM(obj2)[0][2]
                block = obj1 if z_cent1 < z_cent2 else obj2

        else:
            # Create prism with flat top
            block = rad.ObjMltExtPgn([
                [polygon, pole_zs + block_h],
                [bottom_polygon, pole_ze]
            ])

        # Apply segmentation (Note that radia replaces uses the old object ID for the created container,
        # so we don't need to pass back block
        self._apply_block_segmentation(block, rads_iror)

        return block

    def _apply_block_segmentation(self, block: int, rads_iror: List[float]):
        """
        Apply standard segmentation to a side shim block.

        TODO: Make more generic and pull ratio values from config.yml
        """
        cfg = self.config.side_shim
        segmentation = cfg.segmentation

        nesting = 1

        # Apply z segmentation only
        if segmentation[2] != 1:
            nesting += 1

            seg_strategy = SegmentationStrategy([1, 1, segmentation[2]],
                                                [1, 1, 0.1])
            seg_strategy.apply_cylindrical(
                block,
                options='Frame->Lab')

        # TODO: Apply additional fine segmentation of theta

        # --- r-division is now applied by creating more blocks! --- #

        # # Apply r segmentation only and force ratio parameters to 1
        # if segmentation[0] != 1:
        #
        #     nesting += 1
        #
        #     or_mm = self.config.pole.outer_radius_mm
        #     ir_mm = self.config.pole.inner_radius_mm
        #
        #     r_seg_len = (or_mm - ir_mm) / segmentation[0]
        #     seg_strategy = SegmentationStrategy([2, 1, 1], [1, 1, 1])
        #     seg_strategy.apply_cylindrical(
        #         block,
        #         radial_direction=[r_seg_len, 0, 0],
        #         options='Frame->Lab')

        if nesting >= 2:
            ObjectFilter.flatten_nested(block, max_depth=nesting,
                                        verbose=(self.verbosity >= 2))



class CoilComponent(GeometricComponent):
    """
    Field-generating coils (non-magnetic).
    """

    def __init__(self,
                 config: CyclotronConfig,
                 rank: int = 0,
                 verbosity: int = 0):
        # Coils don't need material or segmentation
        super().__init__(config, material_id=None, rank=rank, verbosity=verbosity)

    def build(self) -> int:
        """Build coil pair."""
        self._log("Building coils...")

        cfg = self.config.coil

        Rmin = cfg.radius_min_mm
        Rmax = cfg.radius_max_mm
        Nseg = cfg.num_segments
        h = cfg.height_mm
        midplane_dist = cfg.midplane_dist
        current = cfg.current_A

        CurDens = current / h / (Rmax - Rmin)

        # Lower coil
        pc1 = [0, 0, midplane_dist + 0.5 * h]
        coil1 = rad.ObjRaceTrk(pc1, [Rmin, Rmax], [0.0, 0.0], h, Nseg, CurDens)

        # Upper coil
        pc2 = [0, 0, -(midplane_dist + 0.5 * h)]
        coil2 = rad.ObjRaceTrk(pc2, [Rmin, Rmax], [0.0, 0.0], h, Nseg, CurDens)

        # Set color
        coilcolor = [1, 0, 0]
        rad.ObjDrwAtr(coil1, coilcolor)
        rad.ObjDrwAtr(coil2, coilcolor)

        # Combine into collection
        self.radia_id = rad.ObjCnt([coil1, coil2])
        self._log(f"Coils built (ID={self.radia_id})")

        return self.radia_id