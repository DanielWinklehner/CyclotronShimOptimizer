"""Simplified cyclotron geometry building with component-based architecture."""

# import numpy as np
import radia as rad
import numpy as np

from config_io.config import CyclotronConfig
from geometry.pole_shape import PoleShape
from geometry.components import (
    AnnularWedgeComponent,
    LidUpperComponent,
    TopShimComponent,
    SideShimComponent,
    CoilComponent
)
from simulation.magnetization_cache import RadiaCache


def build_geometry(config: CyclotronConfig,
                   pole_shape: PoleShape = None,
                   omit_symmetry: bool = False,
                   rank: int = 0,
                   verbosity: int = 1,
                   use_cache: bool = False,
                   for_caching: bool = False) -> (int, int):
    """
    Build complete cyclotron geometry using component-based architecture.

    All dimensions in mm.
    All ranks participate in geometry building.

    :param config: CyclotronConfig object
    :param pole_shape: PoleShape object. If None, creates default from config.
    :param omit_symmetry: If True, skip symmetry operations (for visualization)
    :param rank: MPI rank
    :param verbosity: Verbosity level (0=silent, 1=normal, 2=debug)
    :param use_cache
    :param for_caching: Flag whether we are creating just the base geometry for caching and reusing later (speedup)
    :return: Radia object ID for complete cyclotron
    """

    if rank <= 0 and verbosity >= 1:
        print("\n" + "=" * 60, flush=True)
        print("BUILDING CYCLOTRON GEOMETRY", flush=True)
        print("=" * 60 + "\n", flush=True)

    # Create default pole shape if not provided
    if pole_shape is None:
        if rank <= 0 and verbosity >= 1:
            print("Creating default pole shape...", flush=True)

        pole_shape = PoleShape(
            config.side_shim.num_rad_segments,
            default_offset_deg=config.side_shim.default_offset_deg,
            default_offset_mm=config.top_shim.default_offset_mm
        )

    include_side = config.side_shim.include
    include_top = config.top_shim.include
    if not include_top:
        include_side = False

    # ========== MATERIAL ==========
    if rank <= 0 and verbosity >= 1:
        print("Creating material...", flush=True)

    # mat_cfg = config.material
    # ironmat = rad.MatSatIsoFrm(
    #     mat_cfg.saturation_field_t,
    #     mat_cfg.saturation_curve_m,
    #     mat_cfg.linear_curve_m
    # )

    # Dillinger BH curve (mu0*A/m (=T), T)
    if config.material.bh_filename is not None:

        dillinger_data = np.genfromtxt(r"./radialib/dillinger_steel.csv", delimiter=",").tolist()

        if rank <= 0 and verbosity >=1:
            print(f"BH Curve loaded from file {config.material.bh_filename} (in mu0*A/m and T):", flush=True)
            for pair in dillinger_data:
                print(pair, flush=True)
            print("", flush=True)

        ironmat = rad.MatSatIsoTab(dillinger_data)

    # ironmat = rad.MatSatIsoTab([[0.0, 0.0],
    #                             [0.000302, 0.64],
    #                             [0.000397, 0.89],
    #                             [0.000694, 1.25],
    #                             [0.000993, 1.39],
    #                             [0.001497, 1.49],
    #                             [0.00199, 1.54],
    #                             [0.003, 1.61],
    #                             [0.003945, 1.65],
    #                             [0.004973, 1.69],
    #                             [0.006966, 1.72],
    #                             [0.009965, 1.78],
    #                             [0.015026, 1.85],
    #                             [0.019967, 1.91],
    #                             [0.029791, 2.01],
    #                             [0.049908, 2.12],
    #                             [0.069173, 2.16],
    #                             [0.098953, 2.19],
    #                             [0.124749, 2.22],
    #                             [0.251327, 2.3]])

    else:
        mat_cfg = config.material

        ironmat = rad.MatSatIsoFrm(
            mat_cfg.saturation_field_t,
            mat_cfg.saturation_curve_m,
            mat_cfg.linear_curve_m
        )

    ironcolor = [0, 0.5, 1]

    # ========== BUILD COMPONENTS ==========
    if rank <= 0 and verbosity >= 1:
        print("\nBuilding components:", flush=True)

    # Yoke wall (annular wedge with window)
    yoke_wall_params = {
        'outer_radius_mm': config.yoke.outer_radius_mm,
        'inner_radius_mm': config.yoke.inner_radius_mm,
        'height_mm': config.yoke.height_mm,
        'z_offset_mm': 0.0,
        'segmentation': config.yoke.segmentation,
        'include_window': True,
        'window_width_mm': config.yoke.window_width_mm,
        'component_name': 'Yoke Wall'
    }
    yoke_wall_comp = AnnularWedgeComponent(
        config,
        yoke_wall_params,
        material_id=ironmat,
        rank=rank,
        verbosity=verbosity
    )

    yoke_wall_comp.build()
    yoke_wall_comp.segment(config.yoke.segmentation, [1, 1, 1])
    yoke_wall_comp.apply_material()
    yoke_wall_comp.set_drawing_attributes(ironcolor)

    # Lower lid (annular wedge without window)
    lid_lower_params = {
        'outer_radius_mm': config.lid_lower.outer_radius_mm,
        'inner_radius_mm': config.lid_lower.inner_radius_mm,
        'height_mm': config.lid_lower.height_mm,
        'z_offset_mm': -config.yoke.height_mm,
        'segmentation': config.lid_lower.segmentation,
        'include_window': False,
        'component_name': 'Lower Lid'
    }
    lid_lower_comp = AnnularWedgeComponent(
        config,
        lid_lower_params,
        material_id=ironmat,
        rank=rank,
        verbosity=verbosity
    )

    lid_lower_comp.build()
    lid_lower_comp.segment(config.lid_lower.segmentation, [1, 1, 1])
    lid_lower_comp.apply_material()
    lid_lower_comp.set_drawing_attributes(ironcolor)

    # Upper lid
    lid_upper_comp = LidUpperComponent(
        config,
        material_id=ironmat,
        rank=rank,
        verbosity=verbosity
    )

    lid_upper_comp.build()
    lid_upper_comp.segment(config.lid_upper.segmentation, [1, 1, 1])
    lid_upper_comp.apply_material()
    lid_upper_comp.set_drawing_attributes(ironcolor)

    # Pole base (annular wedge with pole-specific angular resolution)
    yoke_h = config.yoke.height_mm
    lid_lower_h = config.lid_lower.height_mm
    pole_h = config.pole.height_mm

    pole_params = {
        'outer_radius_mm': config.pole.outer_radius_mm,
        'inner_radius_mm': config.pole.inner_radius_mm,
        'height_mm': pole_h,
        'z_offset_mm': -(yoke_h + lid_lower_h) + pole_h,
        'segmentation': config.pole.segmentation,
        'include_window': False,
        'use_pole_resolution': True,  # Uses pole.angular_resolution_deg
        'component_name': 'Pole Base'
    }

    pole_comp = AnnularWedgeComponent(
        config,
        pole_params,
        material_id=ironmat,
        rank=rank,
        verbosity=verbosity
    )

    pole_comp.build()
    pole_comp.segment(config.pole.segmentation, [1, 1, 0.1])
    pole_comp.apply_material()
    pole_comp.set_drawing_attributes(ironcolor)

    if include_top:
        # Top shim
        top_shim_comp = TopShimComponent(
            config,
            pole_shape,
            material_id=ironmat,
            rank=rank,
            verbosity=verbosity
        )
        top_shim_comp.build()
        top_shim_comp.apply_material()
        top_shim_comp.set_drawing_attributes(ironcolor)

    if include_side:
        # Side shim
        side_shim_comp = SideShimComponent(
            config,
            pole_shape,
            material_id=ironmat,
            rank=rank,
            verbosity=verbosity
        )
        side_shim_comp.build()
        side_shim_comp.apply_material()
        side_shim_comp.set_drawing_attributes(ironcolor)

    if not use_cache:

        yoke_components = [
            yoke_wall_comp.radia_id,
            lid_lower_comp.radia_id,
            lid_upper_comp.radia_id,
            pole_comp.radia_id,
        ]

        if include_top:
            yoke_components.append(top_shim_comp.radia_id)

        if include_side:
            yoke_components.append(side_shim_comp.radia_id)

        pole_components = None

    else:
        if for_caching:
            # We calculate the field with the pole base but we only save the magnetizations for the yoke
            yoke_components = [
                yoke_wall_comp.radia_id,
                lid_lower_comp.radia_id,
                lid_upper_comp.radia_id,
                pole_comp.radia_id]

            # Note: for caching, the shims are not actually needed nor used, so nothing to exclude here...
            pole_components= None

        else:
            yoke_components = [
                yoke_wall_comp.radia_id,
                lid_lower_comp.radia_id,
                lid_upper_comp.radia_id
                ]

            pole_components = [
                pole_comp.radia_id
            ]

            if include_top:
                pole_components.append(top_shim_comp.radia_id)

            if include_side:
                pole_components.append(side_shim_comp.radia_id)

            # --- Now we apply the cached magnetizations (saved after the solve step later)
            # We only apply what's in the cache, which is yoke components only...
            cache = RadiaCache(rank=rank, verbosity=verbosity)
            cached_mags = cache.load_magnetizations(config)
            if cached_mags is not None:
                cache.apply_magnetizations(cached_mags)

    # --- Note: Yoke and Pole components must be built in the above order so that the IDs are --- #
    # --- consistent between cached version and current version! Shims are never cached             --- #
    # TODO: May cache shims in the future as well (cave: nuymber of shim blocks can change during optimization)
    # TODO: Think about a base level magnetization (maybe based on average in pole base) as a starting point

    yoke = rad.ObjCnt(yoke_components)

    if use_cache:
        pole = rad.ObjCnt(pole_components)
    else:
        pole = None

    # ========== BUILD COILS ==========
    if rank <= 0 and verbosity >= 1:
        print("Building coils...", flush=True)

    coil_comp = CoilComponent(config, rank=rank, verbosity=verbosity)
    coil_comp.build()

    # ========== APPLY SYMMETRIES ==========

    if rank <= 0 and verbosity >= 1:
        print("Applying 8-fold symmetry...", flush=True)


    # 2-fold symmetry in z-direction
    if not omit_symmetry:
        rad.TrfZerPerp(yoke, [0, 0, 0], [1, -1, 0])  # Mirror across xy diagonal
        rad.TrfZerPerp(yoke, [0, 0, 0], [1, 0, 0])  # Mirror across x-axis
        rad.TrfZerPerp(yoke, [0, 0, 0], [0, 1, 0])  # Mirror across y-axis
        rad.TrfZerPara(yoke, [0, 0, 0], [0, 0, 1])  # Mirror across z-plane
    else:
        if rank <= 0 and verbosity >= 1:
            print("Symmetry DISABLED (geometry debug mode)", flush=True)
    if use_cache:
        rad.TrfZerPerp(pole, [0, 0, 0], [1, -1, 0])  # Mirror across xy diagonal
        rad.TrfZerPerp(pole, [0, 0, 0], [1, 0, 0])  # Mirror across x-axis
        rad.TrfZerPerp(pole, [0, 0, 0], [0, 1, 0])  # Mirror across y-axis
        # 2-fold symmetry in z-direction
        if not omit_symmetry:
            rad.TrfZerPara(pole, [0, 0, 0], [0, 0, 1])  # Mirror across z-plane
        else:
            if rank <= 0 and verbosity >= 1:
                print("Symmetry DISABLED (geometry debug mode)", flush=True)

    # ========== ASSEMBLE COMPLETE CYCLOTRON ==========
    if rank <= 0 and verbosity >= 1:
        print("\nAssembling cyclotron...", flush=True)

    yoke_w_coil = rad.ObjCnt([yoke, coil_comp.radia_id])

    if rank <= 0 and verbosity >= 1:
        print("\n" + "=" * 60, flush=True)
        print("GEOMETRY BUILDING COMPLETE", flush=True)
        print("=" * 60 + "\n", flush=True)

    return yoke_w_coil, pole
