"""Simplified cyclotron geometry building with component-based architecture."""

# import numpy as np
import radia as rad
import numpy as np
import os

from config_io.config import CyclotronConfig
from geometry.pole_shape import PoleShape
from geometry.components import (
    AnnularWedgeComponent,
    LidUpperComponent,
    TopShimComponent,
    SideShimComponent,
    CoilComponent, FromSTPFileComponent, GmshPoleComponent, GmshWedgeComponent,GmshLidUpperComponent
)
from simulation.magnetization_cache import RadiaCache


def build_geometry(config: CyclotronConfig,
                   pole_shape: PoleShape = None,
                   omit_symmetry: bool = False,
                   rank: int = 0,
                   comm = None,
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
    :param comm: MPI communicator
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

    # Dillinger BH curve (mu0*A/m (=T), T)
    if config.material.bh_filename is not None:

        dillinger_data = np.genfromtxt(os.path.join("radialib", config.material.bh_filename), delimiter=",").tolist()

        if rank <= 0 and verbosity >=1:
            print(f"BH Curve loaded from file {config.material.bh_filename} (in mu0*A/m and T):", flush=True)
            for pair in dillinger_data:
                print(pair, flush=True)
            print("", flush=True)

        ironmat = rad.MatSatIsoTab(dillinger_data)

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
    if config.yoke.stp_filename is not None:
        yoke_wall_params = {
            'stp_filename': config.yoke.stp_filename,
            'component_name': 'Yoke Wall',
            'elem_size':config.yoke.max_mesh_size
        }

        yoke_wall_comp = FromSTPFileComponent(
            config,
            yoke_wall_params,
            material_id=ironmat,
            rank=rank,
            comm=comm,
            verbosity=verbosity
        )

    else:
        yoke_wall_params = {
            'outer_radius_mm': config.yoke.outer_radius_mm,
            'inner_radius_mm': config.yoke.inner_radius_mm,
            'height_mm': config.yoke.height_mm,
            'z_offset_mm': 0.0,
            'segmentation': config.yoke.segmentation,
            'include_window': True,
            'window_width_mm': config.yoke.window_width_mm,
            'component_name': 'Yoke_Wall',
            'max_mesh_size': config.yoke.max_mesh_size,
            'stp_output': config.yoke.stp_output
        }
        if config.geometry.use_gmsh_occ_yoke:
            yoke_wall_comp = GmshWedgeComponent(config,
                                              yoke_wall_params,
                                              material_id=ironmat,
                                              rank=rank,
                                              comm=comm,
                                              verbosity=verbosity
            )
        else:
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
    if config.lid_lower.stp_filename is not None:

        lid_lower_params = {
            'stp_filename': config.lid_lower.stp_filename,
            'component_name': 'Lower Lid',
            'elem_size':config.lid_lower.max_mesh_size
        }

        lid_lower_comp = FromSTPFileComponent(
            config,
            lid_lower_params,
            material_id=ironmat,
            rank=rank,
            comm=comm,
            verbosity=verbosity
        )

    else:

        lid_lower_params = {
            'outer_radius_mm': config.lid_lower.outer_radius_mm,
            'inner_radius_mm': config.lid_lower.inner_radius_mm,
            'height_mm': config.lid_lower.height_mm,
            'z_offset_mm': -config.yoke.height_mm,
            'segmentation': config.lid_lower.segmentation,
            'include_window': False,
            'component_name': 'Lower_Lid',
            'max_mesh_size': config.lid_lower.max_mesh_size,
            'stp_output': config.lid_lower.stp_output
        }
        if config.geometry.use_gmsh_occ_yoke:
            lid_lower_comp = GmshWedgeComponent(config,
                                              lid_lower_params,
                                              material_id=ironmat,
                                              rank=rank,
                                              comm=comm,
                                              verbosity=verbosity)
        else:
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
    if config.lid_upper.stp_filename is not None:
        lid_upper_params = {
            'stp_filename': config.lid_upper.stp_filename,
            'component_name': 'Upper Lid',
            'elem_size':config.lid_upper.max_mesh_size
        }

        lid_upper_comp = FromSTPFileComponent(
            config,
            lid_upper_params,
            material_id=ironmat,
            rank=rank,
            comm=comm,
            verbosity=verbosity
        )

    else:
        if config.geometry.use_gmsh_occ_yoke:
            lid_upper_params = {
                'outer_radius_mm_1': config.lid_upper.outer_radius_mm_1,
                'outer_radius_mm_2': config.lid_upper.outer_radius_mm_2,
                'inner_radius_mm': config.lid_upper.inner_radius_mm,
                'height_mm': config.lid_upper.height_mm,
                'z_offset_mm': -(config.yoke.height_mm+config.lid_lower.height_mm),
                'segmentation': config.lid_upper.segmentation,
                'include_window': False,
                'hole_diameter_mm': config.lid_upper.hole_diameter_mm,
                'hole_center_xy': config.lid_upper.hole_center_xy,
                'cut_out_rf_stem_hole':config.lid_upper.cut_out_rf_stem_hole,
                # 'channel_width_mm': config.extract_channel.channel_width_mm,
                'component_name': 'lid_upper',
                'max_mesh_size': config.lid_upper.max_mesh_size,
                'stp_output': config.lid_upper.stp_output
            }

            lid_upper_comp = GmshLidUpperComponent(config,
                                              lid_upper_params,
                                              material_id=ironmat,
                                              rank=rank,
                                              comm=comm,
                                              verbosity=verbosity)
        else:
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

 # Extraction Channel 1 wall (annular wedge with window)
 #not set up for combined extraction channel stp
    if config.extract_channel.use_extract_chan:
        if config.extract_channel.stp_filename is not None:
            extract_channel_params = {
                'stp_filename': config.extract_channel.stp_filename,
                'component_name': 'Extraction Channel',
                'elem_size':config.extract_channel.max_mesh_size
            }

            extract_channel_comp_1 = FromSTPFileComponent(
                config,
                extract_channel_params,
                material_id=ironmat,
                rank=rank,
                comm=comm,
                verbosity=verbosity
            )
            extract_channel_comp_2 = FromSTPFileComponent(
                config,
                extract_channel_params,
                material_id=ironmat,
                rank=rank,
                comm=comm,
                verbosity=verbosity
            )

        else:
            extract_channel_params_1 = {
                'outer_radius_mm': config.extract_channel.outer_radius_mm,
                'inner_radius_mm': config.extract_channel.inner_radius_mm,
                'height_mm': config.extract_channel.height_mm,
                'z_offset_mm': config.extract_channel.height_mm+config.extract_channel.channel_width_mm/2,
                'segmentation': config.extract_channel.segmentation,
                'include_window': False,
                'window_width_mm': config.extract_channel.window_width_mm,
                'start_ang_deg': config.extract_channel.start_ang_deg,
                'end_ang_deg': config.extract_channel.end_ang_deg,
                'component_name': 'Extraction_Channel_1',
                'max_mesh_size': config.extract_channel.max_mesh_size
            }

            extract_channel_params_2 = {
                'outer_radius_mm': config.extract_channel.outer_radius_mm,
                'inner_radius_mm': config.extract_channel.inner_radius_mm,
                'height_mm': config.extract_channel.height_mm,
                'z_offset_mm': -config.extract_channel.channel_width_mm/2,
                'segmentation': config.extract_channel.segmentation,
                'include_window': False,
                'window_width_mm': config.extract_channel.window_width_mm,
                'start_ang_deg': config.extract_channel.start_ang_deg,
                'end_ang_deg': config.extract_channel.end_ang_deg,
                # 'channel_width_mm': config.extract_channel.channel_width_mm,
                'component_name': 'Extraction_Channel_2',
                'max_mesh_size': config.extract_channel.max_mesh_size
            }
            if config.geometry.use_gmsh_occ_yoke:
                extract_channel_comp_1 = GmshWedgeComponent(config,
                                              extract_channel_params_1,
                                              material_id=ironmat,
                                              rank=rank,
                                              comm=comm,
                                              verbosity=verbosity)

                extract_channel_comp_2 = GmshWedgeComponent(config,
                                              extract_channel_params_2,
                                              material_id=ironmat,
                                              rank=rank,
                                              comm=comm,
                                              verbosity=verbosity)
            else:                                  
                extract_channel_comp_1 = AnnularWedgeComponent(
                    config,
                    extract_channel_params_1,
                    material_id=ironmat,
                    rank=rank,
                    verbosity=verbosity
                )
                extract_channel_comp_2 = AnnularWedgeComponent(
                    config,
                    extract_channel_params_2,
                    material_id=ironmat,
                    rank=rank,
                    verbosity=verbosity
                )


        extract_channel_comp_1.build()
        extract_channel_comp_1.segment(config.extract_channel.segmentation, [1, 1, 1])
        extract_channel_comp_1.apply_material()
        extract_channel_comp_1.set_drawing_attributes(ironcolor)

        extract_channel_comp_2.build()
        extract_channel_comp_2.segment(config.extract_channel.segmentation, [1, 1, 1])
        extract_channel_comp_2.apply_material()
        extract_channel_comp_2.set_drawing_attributes(ironcolor)

    # Pole base (annular wedge with pole-specific angular resolution)
    if config.pole.stp_filename is not None:
        pole_params = {
            'stp_filename': config.pole.stp_filename,
            'component_name': 'Pole Base',
            'elem_size': config.pole.max_mesh_size,
        }

        pole_comp = FromSTPFileComponent(
            config,
            pole_params,
            material_id=ironmat,
            rank=rank,
            comm=comm,
            verbosity=verbosity
        )

    else:
        if config.geometry.use_gmsh_occ_pole:
            pole_comp = GmshPoleComponent(config,
                                          pole_shape,
                                          material_id=ironmat,
                                          rank=rank,
                                          comm=comm,
                                          verbosity=verbosity)
        else:
            pole_params = {
                'outer_radius_mm': config.pole.outer_radius_mm,
                'inner_radius_mm': config.pole.inner_radius_mm,
                'height_mm': config.pole.height_mm,
                'z_offset_mm': -(config.yoke.height_mm + config.lid_lower.height_mm) + config.pole.height_mm,
                'segmentation': config.pole.segmentation,
                'include_window': False,
                'use_pole_resolution': True,  # Uses pole.angular_resolution_deg
                'component_name': 'Pole Base',
                'stp_output': config.pole.stp_output
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

    if include_top and not config.geometry.use_gmsh_occ_pole:  # Note: in Gmsh building, the shims are included
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

    if include_side and not config.geometry.use_gmsh_occ_pole:
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

        if include_top and not config.geometry.use_gmsh_occ_pole:
            yoke_components.append(top_shim_comp.radia_id)

        if include_side and not config.geometry.use_gmsh_occ_pole:
            yoke_components.append(side_shim_comp.radia_id)

        pole_components = None
        if config.extract_channel.use_extract_chan:
            extract_components = [
                extract_channel_comp_1.radia_id, 
                extract_channel_comp_2.radia_id
            ]
        else:
            extract_components = None

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
            if config.extract_channel.use_extract_chan:
                extract_components = [
                    extract_channel_comp_1.radia_id, 
                    extract_channel_comp_2.radia_id
                ]
            else:
                extract_components = None
        else:
            yoke_components = [
                yoke_wall_comp.radia_id,
                lid_lower_comp.radia_id,
                lid_upper_comp.radia_id
                ]

            pole_components = [
                pole_comp.radia_id
            ]
            if config.extract_channel.use_extract_chan:
                extract_components = [
                    extract_channel_comp_1.radia_id, 
                    extract_channel_comp_2.radia_id
                ]
            else:
                extract_components = None

            if include_top and not config.geometry.use_gmsh_occ_pole:
                pole_components.append(top_shim_comp.radia_id)

            if include_side and not config.geometry.use_gmsh_occ_pole:
                pole_components.append(side_shim_comp.radia_id)

            # --- Now we apply the cached magnetizations (saved after the solve step later)
            # We only apply what's in the cache, which is yoke components only...
            cache = RadiaCache(rank=rank, verbosity=verbosity)
            cached_mags = cache.load_magnetizations(config)
            if cached_mags is not None:
                cache.apply_magnetizations(cached_mags)

    # --- Note: Yoke and Pole components must be built in the above order so that the IDs are --- #
    # --- consistent between cached version and current version! Shims are never cached             --- #
    # TODO: May cache shims in the future as well (cave: number of shim blocks can change during optimization)
    # TODO: Think about a base level magnetization (maybe based on average in pole base) as a starting point

    yoke = rad.ObjCnt(yoke_components)
    if config.extract_channel.use_extract_chan:
        extract = rad.ObjCnt(extract_components)
    else:
        extract = None
    if use_cache:
        pole = rad.ObjCnt(pole_components)
    else:
        pole = None

    # ========== BUILD COILS ==========
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
    if config.extract_channel.use_extract_chan:
        yoke_w_coil = rad.ObjCnt([yoke, coil_comp.radia_id, extract])#_channel_comp_1.radia_id,extract_channel_comp_2.radia_id
    else:
        yoke_w_coil = rad.ObjCnt([yoke, coil_comp.radia_id])

    if rank <= 0 and verbosity >= 1:
        print("\n" + "=" * 60, flush=True)
        print("GEOMETRY BUILDING COMPLETE", flush=True)
        print("=" * 60 + "\n", flush=True)

    return yoke_w_coil, pole
