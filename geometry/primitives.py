"""Geometry primitives and utilities for Radia components."""

import numpy as np
from typing import List, Tuple, Dict, Any
import radia as rad


class PolygonBuilder:
    """Builder for 2D/3D polygons."""

    @staticmethod
    def arc_polygon(radius: float,
                    angles: np.ndarray,
                    origin: bool = True) -> List[List[float]]:
        """
        Create arc polygon from radius and angles.

        :param radius: Radius of arc
        :param angles: Array of angles in radians
        :param origin: If True, close polygon at origin
        :return: List of [x, y] coordinates
        """
        arc = np.array([radius * np.cos(angles),
                        radius * np.sin(angles)]).T

        if origin:
            arc = np.vstack([arc, [[0.0, 0.0]]])

        return arc.tolist()

    @staticmethod
    def annular_segment(r_inner: float,
                        r_outer: float,
                        angle_start: float,
                        angle_end: float,
                        num_points: int = 20) -> List[List[float]]:
        """
        Create annular segment polygon (arc from inner to outer radius).

        :param r_inner: Inner radius
        :param r_outer: Outer radius
        :param angle_start: Start angle (radians)
        :param angle_end: End angle (radians)
        :param num_points: Number of points along each arc
        :return: Counterclockwise polygon
        """
        angles = np.linspace(angle_start, angle_end, num_points)

        # Outer arc (forward)
        outer_arc = r_outer * np.array([np.cos(angles), np.sin(angles)]).T

        # Inner arc (backward)
        inner_arc = r_inner * np.array([np.cos(angles[::-1]),
                                        np.sin(angles[::-1])]).T

        return np.vstack([outer_arc, inner_arc]).tolist()

    @staticmethod
    def rectangular_grid(radii: np.ndarray,
                         angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create meshgrid for radial/angular blocks.

        :param radii: Array of radii
        :param angles: Array of angles (radians)
        :return: Tuple of (x_grid, y_grid) with shape (n_radii, n_angles)
        """
        radii_grid, angles_grid = np.meshgrid(radii, angles, indexing='ij')
        x = np.round(radii_grid * np.cos(angles_grid), 5)
        y = np.round(radii_grid * np.sin(angles_grid), 5)
        return x, y


class SegmentationStrategy:
    """Encapsulates Radia segmentation logic."""

    def __init__(self,
                 divisions: List[float],
                 size_factors: List[float]=None):
        """
        Initialize segmentation strategy.

        :param divisions: [r_div, theta_div, z_div]
        :param size_factors: Ratios of first to last element in the subdivision. Can be used to suppress
                             further subdivision if the purpose is cutting out a hole. A large size factor
                             will prevent any more subdivisions.
                             A size factor below 1 will make succesively smaller elements towards the end.
        """
        if len(divisions) != 3:
            raise ValueError("divisions must have 3 elements [r, theta, z]")

        if size_factors is None:
            size_factors = [1, 1, 1]
        if len(size_factors) != 3:
            raise ValueError("size_factors must have 3 elements [r, theta, z]")

        self.r_div = divisions[0]
        self.theta_div = divisions[1]
        self.z_div = divisions[2]
        self.size_factors = size_factors

    def apply_center_bore_cutout(self,
                                 obj_id: int,
                                 radius: float):
        """
        Apply a cylindrical cutout centered at the origin that suppresses
        additional segmentation beyond the desired center cut.

        size factors and r/z divisions are ignored
        Some theta division is needed to approximate a circle.

        :param obj_id:
        :param radius:
        :return:
        """

        center = [0, 0, 0]
        z_axis = [0, 0, 1]

        rad.ObjDivMagCyl(
            obj_id,
            [[2, 2000], [self.theta_div, 1], [1, 1]],
            center,
            z_axis,
            [radius, 0, 0],
            1.0,
            'Frame->Lab'
        )


    def apply_cylindrical(self,
                          obj_id: int,
                          center: List[float]=None,
                          radial_direction: List[float]=None,
                          axis_direction: List[float]=None,
                          options: str = 'Frame->Lab'):
        """
        Apply cylindrical segmentation to object.

        Radia signature:
        ObjDivMagCyl(obj,
                     [[k1,q1],[k2,q2],[k3,q3]],
                     [ax,ay,az],
                     [vx,vy,vz],
                     [px,py,pz],
                     rat,
                     'kxkykz->Numb|Size,Frame->Loc|Lab|LabTot')

        :param obj_id: Radia object ID
        :param center: Cylinder center [x, y, z]
        :param radial_direction: Radial direction [x, y, z]
        :param axis_direction: Cylinder axis [x, y, z] (default [0,0,1])
        :param options: Radia option string 'Frame->Loc|Lab|LabTot' in the future maybe also 'kxkykz->Numb|Size

        TODO: Option strings could be wrapped in pythonic way

        """
        if axis_direction is None:
            axis_direction = [0, 0, 1]

        if center is None:
            center = [0, 0, 0]

        if radial_direction is None:
            radial_direction = [1, 0, 0]

        rad.ObjDivMag(
            obj_id,
            [[self.r_div, self.size_factors[0]],
             [self.theta_div, self.size_factors[1]],
             [self.z_div, self.size_factors[2]]],
            'cyl',
            [center,
             axis_direction,
             radial_direction,
             1.0],  # For now we assume segmentation is circular (other option would be elliptical)
            options
        )

    def apply_cartesian(self,
                        obj_id: int,
                        scale_factor: float = 0.1,
                        frame: str = 'Frame->Lab'):
        """
        Apply Cartesian (z-only) segmentation to object.

        Radia signature:
        ObjDivMag(obj,
                  [[k1,q1],[k2,q2],[k3,q3]],
                  'pln|cyl',
                  [[n1x,n1y,n1z],[n2x,n2y,n2z],[n3x,n3y,n3z]]|[[ax,ay,az],[vx,vy,vz],[px,py,pz],rat],
                  'kxkykz->Numb|Size,Frame->Loc|Lab|LabTot')

        :param obj_id: Radia object ID
        :param scale_factor: Scale factor for z-division (< 1 = finer)
        :param frame: Radia frame string
        """
        if self.z_div > 1:
            rad.ObjDivMag(
                obj_id,
                [[1, 1], [1, 1], [self.z_div, scale_factor]],
                frame
            )


class ObjectFilter:
    """Filter and manipulate Radia object containers."""

    @staticmethod
    def keep_outside_radius(container: int,
                            min_radius: float,
                            verbose: bool = False) -> int:
        """
        Filter: keep only objects outside a radius (from origin).

        :param container: Radia container object ID
        :param min_radius: Minimum radius threshold
        :param verbose: Print filtering info
        :return: New container with filtered objects
        """
        objs_to_keep = []
        objs_deleted = 0

        for obj in rad.ObjCntStuf(container):
            obj_center = rad.ObjM(obj)[0]
            r_center = np.sqrt(obj_center[0] ** 2 + obj_center[1] ** 2)

            if r_center > min_radius:
                objs_to_keep.append(obj)
            else:
                rad.UtiDel(obj)
                objs_deleted += 1

        if verbose:
            print(f"    ObjectFilter: kept {len(objs_to_keep)}, deleted {objs_deleted} (r < {min_radius:.1f})")

        return rad.ObjCnt(objs_to_keep)

    @staticmethod
    def keep_outside_circle(container: int,
                            center: List[float],
                            radius: float,
                            verbose: bool = False) -> int:
        """
        Filter: keep only objects outside a circular region.

        :param container: Radia container object ID
        :param center: Circle center [x, y]
        :param radius: Circle radius
        :param verbose: Print filtering info
        :return: New container with filtered objects
        """
        objs_to_keep = []
        objs_deleted = 0

        for obj in rad.ObjCntStuf(container):
            obj_center = rad.ObjM(obj)[0]
            dist = np.sqrt((obj_center[0] - center[0]) ** 2 +
                           (obj_center[1] - center[1]) ** 2)

            if dist > radius:
                objs_to_keep.append(obj)
            else:
                rad.UtiDel(obj)
                objs_deleted += 1

        if verbose:
            print(f"    ObjectFilter: kept {len(objs_to_keep)}, deleted {objs_deleted} (dist < {radius:.1f})")

        return rad.ObjCnt(objs_to_keep)

    @staticmethod
    def flatten_nested(container: int,
                       max_depth: int = 2,
                       verbose: bool = False) -> int:
        """
        Flatten nested containers for cleaner OpenGL display.

        :param container: Root container
        :param max_depth: How many levels to flatten
        :param verbose: Print flattening info
        :return: Flattened container
        """
        flattened = []

        def recurse(obj, depth):
            if depth == 0:
                # print(rad.UtiDmp(obj))
                flattened.append(obj)
            else:
                for sub_obj in rad.ObjCntStuf(obj):
                    try:
                        recurse(sub_obj, depth - 1)
                    except:
                        # print(rad.UtiDmp(sub_obj))
                        flattened.append(sub_obj)

        recurse(container, max_depth)

        if verbose:
            print(f"    ObjectFilter: flattened to {len(flattened)} objects")

        return rad.ObjCnt(flattened)
