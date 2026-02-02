"""Configuration loading and validation."""

import yaml
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import sys

# Add radialib to path for radia
RADIA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'radialib')
if RADIA_PATH not in sys.path:
    sys.path.insert(0, RADIA_PATH)


@dataclass
class GeometryConfig:
    yoke_build_angle_deg: float
    angular_resolution: int


@dataclass
class FieldEvaluationConfig:
    num_points_circle: int
    radius_min_mm: float
    radius_max_mm: float
    n_eval_pts: int
    use_symmetry: bool = True


@dataclass
class YokeConfig:
    outer_radius_mm: float
    inner_radius_mm: float
    height_mm: float
    segmentation: List[int]
    window_width_mm: float


@dataclass
class LidLowerConfig:
    outer_radius_mm: float
    inner_radius_mm: float
    height_mm: float
    segmentation: List[int]


@dataclass
class LidUpperConfig:
    outer_radius_mm_1: float
    outer_radius_mm_2: float
    inner_radius_mm: float
    height_mm: float
    segmentation: List[int]
    hole_diameter_mm: float
    hole_center_xy: List[float]
    cut_out_rf_stem_hole: bool


@dataclass
class PoleConfig:
    outer_radius_mm: float
    inner_radius_mm: float
    height_mm: float
    full_angle_deg: float
    angular_resolution_deg: float
    segmentation: List[int]


@dataclass
class SideShimConfig:
    num_rad_segments: int
    angular_resolution_deg: float
    default_offset_deg: float
    segmentation: List[int]
    side_offsets_deg: Optional[List[float]] = None
    include: Optional[bool] = True


@dataclass
class TopShimConfig:
    num_rad_segments: int
    angular_resolution_deg: float
    default_offset_mm: float
    segmentation: List[int]
    top_offsets_mm:  Optional[List[float]] = None
    include: Optional[bool] = True


@dataclass
class CoilConfig:
    radius_min_mm: float
    radius_max_mm: float
    height_mm: float
    current_A: float
    num_segments: int


@dataclass
class MaterialConfig:
    saturation_field_t: list
    saturation_curve_m: list
    linear_curve_m: list
    bh_filename: Optional[str] = None
    

@dataclass
class SimulationConfig:
    precision: float
    iterations: int


@dataclass
class OptimizationConfig:
    target_frequency_mhz: float
    frequency_tolerance_mhz: float
    max_iterations: int
    coil_current_min_A: float
    coil_current_max_A: float
    side_shim_min_deg: float
    side_shim_max_deg: float
    top_shim_min_mm: float
    top_shim_max_mm: float
    num_workers: int
    n_initial_points: int
    reference_coil_current: float
    regularization_weight: float
    optimizer: str
    random_init: bool
    opt_top: Optional[bool] = True
    opt_side: Optional[bool] = True
    opt_coil: Optional[bool] = True


@dataclass
class VisualizationConfig:
    show_opengl: bool
    comsol_filename: Optional[str] = None


@dataclass
class CyclotronConfig:
    """Complete cyclotron configuration."""
    particle_species: str
    max_machine_size_mm: float
    seed: int
    geometry: GeometryConfig
    field_evaluation: FieldEvaluationConfig
    yoke: YokeConfig
    lid_lower: LidLowerConfig
    lid_upper: LidUpperConfig
    pole: PoleConfig
    coil: CoilConfig
    material: MaterialConfig
    side_shim: SideShimConfig
    top_shim: TopShimConfig
    simulation: SimulationConfig
    optimization: OptimizationConfig
    visualization: VisualizationConfig

    @classmethod
    def from_yaml(cls, filepath: str) -> 'CyclotronConfig':
        """Load configuration from YAML file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError("Config file is empty")

        try:
            return cls(
                seed=data.get('seed', 42),
                particle_species=data.get('particle_species', 'muon'),
                max_machine_size_mm=data.get('max_machine_size_mm', 860.0),
                geometry=GeometryConfig(**data['geometry']),
                field_evaluation=FieldEvaluationConfig(**data['field_evaluation']),
                yoke=YokeConfig(**data['yoke']),
                lid_lower=LidLowerConfig(**data['lid_lower']),
                lid_upper=LidUpperConfig(**data['lid_upper']),
                pole=PoleConfig(**data['pole']),
                coil=CoilConfig(**data['coil']),
                material=MaterialConfig(**data['material']),
                side_shim=SideShimConfig(**data['side_shim']),
                top_shim=TopShimConfig(**data['top_shim']),
                simulation=SimulationConfig(**data['simulation']),
                optimization=OptimizationConfig(**data['optimization']),
                visualization=VisualizationConfig(**data['visualization']),
            )
        except KeyError as e:
            raise ValueError(f"Missing required config key: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'particle_species': self.particle_species,
            'max_machine_size_mm': self.max_machine_size_mm,
            'seed': self.seed,
            'geometry': {
                'yoke_build_angle_deg': self.geometry.yoke_build_angle_deg,
                'angular_resolution': self.geometry.angular_resolution,
            },
            'field_evaluation': {
                'num_points_circle': self.field_evaluation.num_points_circle,
                'radius_min_mm': self.field_evaluation.radius_min_mm,
                'radius_max_mm': self.field_evaluation.radius_max_mm,
                'n_eval_pts': self.field_evaluation.n_eval_pts,
                'use_symmetry':  self.field_evaluation.use_symmetry,
            },
            'yoke': {
                'outer_radius_mm': self.yoke.outer_radius_mm,
                'inner_radius_mm': self.yoke.inner_radius_mm,
                'height_mm': self.yoke.height_mm,
                'segmentation': self.yoke.segmentation,
                'window_width_mm': self.yoke.window_width_mm,
            },
            'lid_lower': {
                'outer_radius_mm': self.lid_lower.outer_radius_mm,
                'inner_radius_mm': self.lid_lower.inner_radius_mm,
                'height_mm': self.lid_lower.height_mm,
                'segmentation': self.lid_lower.segmentation,
            },
            'lid_upper': {
                'outer_radius_mm_1': self.lid_upper.outer_radius_mm_1,
                'outer_radius_mm_2': self.lid_upper.outer_radius_mm_2,
                'inner_radius_mm': self.lid_upper.inner_radius_mm,
                'height_mm': self.lid_upper.height_mm,
                'segmentation': self.lid_upper.segmentation,
                'hole_diameter_mm': self.lid_upper.hole_diameter_mm,
                'hole_center_xy': self.lid_upper.hole_center_xy,
                'cut_out_rf_stem_hole': self.lid_upper.cut_out_rf_stem_hole,
            },
            'pole': {
                'outer_radius_mm': self.pole.outer_radius_mm,
                'inner_radius_mm': self.pole.inner_radius_mm,
                'height_mm': self.pole.height_mm,
                'full_angle_deg': self.pole.full_angle_deg,
                'segmentation': self.pole.segmentation,
            },
            'coil': {
                'radius_min_mm': self.coil.radius_min_mm,
                'radius_max_mm': self.coil.radius_max_mm,
                'height_mm': self.coil.height_mm,
                'current_A': self.coil.current_A,
                'num_segments': self.coil.num_segments,
            },
            'material': {
                'saturation_field_t': self.material.saturation_field_t,
                'saturation_curve_m': self.material.saturation_curve_m,
                'linear_curve_m': self.material.linear_curve_m,
                'bh_filename': self.material.bh_filename,
            },
            'side_shim': {
                'num_rad_segments': self.side_shim.num_rad_segments,
                'ang_resulution_deg': self.side_shim.angular_resolution_deg,
                'default_offset_deg': self.side_shim.default_offset_deg,
                'side_offsets_deg': self.side_shim.side_offsets_deg,
                'segmentation': self.side_shim.segmentation,
                'include': self.side_shim.include,
            },
            'top_shim': {
                'num_rad_segments': self.top_shim.num_rad_segments,
                'ang_resulution_deg': self.top_shim.angular_resolution_deg,
                'default_offset_mm': self.top_shim.default_offset_mm,
                'top_offsets_mm': self.top_shim.top_offsets_mm,
                'segmentation': self.top_shim.segmentation,
                'include': self.top_shim.include,
            },
            'simulation': {
                'precision': self.simulation.precision,
                'iterations': self.simulation.iterations,
            },
            'optimization': {
                'target_frequency_mhz': self.optimization.target_frequency_mhz,
                'frequency_tolerance_mhz': self.optimization.frequency_tolerance_mhz,
                'max_iterations': self.optimization.max_iterations,
                'coil_current_min_A': self.optimization.coil_current_min_A,
                'coil_current_max_A': self.optimization.coil_current_max_A,
                'num_workers': self.optimization.num_workers,
                'n_initial_points': self.optimization.n_initial_points,
                'reference_coil_current': self.optimization.reference_coil_current,
                'regularization_weight': self.optimization.regularization_weight,
                'optimizer': self.optimization.optimizer,
                'random_init': self.optimization.random_init,
                'opt_top': self.optimization.opt_top,
                'opt_side': self.optimization.opt_top,
                'opt_coil': self.optimization.opt_top,
                'side_shim_min_deg': self.optimization.side_shim_min_deg,
                'side_shim_max_deg': self.optimization.side_shim_max_deg,
                'top_shim_min_mm': self.optimization.top_shim_min_mm,
                'top_shim_max_mm': self.optimization.top_shim_max_mm,
            },
            'visualization': {
                'show_opengl': self.visualization.show_opengl,
                'comsol_filename': self.visualization.comsol_filename,
            },
        }