"""Configuration loading and validation.

Two YAML schemas are supported:

- LEGACY: fixed per-part sections (yoke:, lid_lower:, ..., coil:, material:).
  Loaded into the per-part dataclasses as before, and ADDITIONALLY adapted
  into the generic component list (ComponentSpec) that the geometry builders
  consume.
- COMPONENT-BASED (new): named ``materials:`` and ``symmetries:`` plus a
  ``components:`` list (name / kind / params / material / symmetry / mesh /
  file / enabled / shimmed). The geometry is built by a kind-keyed builder
  registry (geometry.geometry); adding a part is a YAML entry, not a new
  dataclass. The few legacy dataclasses still read elsewhere (coil, pole,
  extract_channel, material) are synthesized from the specs.

Workflow-ish sections (field_evaluation, simulation, optimization,
visualization, side_shim/top_shim) are common to both schemas.
"""

import yaml
import os
from dataclasses import dataclass, field, fields as _dc_fields


def _from_dict(cls, d):
    """Construct a config dataclass, IGNORING unknown keys.

    Keeps parsing tolerant when fields are removed from the schema (e.g. the
    export/output side-effect flags, now driven by explicit script-level API
    calls, not the config) so older or external ymls that still carry those
    keys load without error.
    """
    d = d or {}
    valid = {f.name for f in _dc_fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in valid})


def _coerce_numeric_fields(section_obj, section_name: str) -> None:
    """Coerce string values in numeric-annotated fields; raise CLEARLY if bogus.

    YAML turns several innocent-looking values into STRINGS -- quoted numbers,
    exponent floats without a decimal point ('1e-1' is a string in YAML 1.1),
    or editing accidents like pasting the dataclass syntax into the yml
    ('bore_xy_limit_mm: float = 200.0'). Those used to survive loading and
    blow up arbitrarily late -- e.g. a TypeError in the bore-field export
    AFTER the ~100 s solve. Coercing at load time makes an honest numeric
    string work and a bogus one fail immediately, naming the yml key.
    """
    for f in _dc_fields(section_obj):
        value = getattr(section_obj, f.name, None)
        if not isinstance(value, str):
            continue
        ann = str(f.type)
        if 'float' not in ann and 'int' not in ann:
            continue   # str/bool/list-typed fields keep their strings
        try:
            coerced = float(value) if 'float' in ann else int(value)
        except ValueError:
            raise ValueError(
                f"config value {section_name}.{f.name} = {value!r} is not a "
                f"number (annotated {ann}). Common causes: pasted Python "
                f"syntax ('key: float = 1.0' -> write 'key: 1.0'), or a YAML "
                f"exponent float without a decimal point ('1e-1' -> '1.0e-1').")
        setattr(section_obj, f.name, coerced)
from typing import Dict, Any, List, Optional


# The standard 8-fold cyclotron symmetry + midplane mirror, as
# (kind, point, normal) with 'perp' -> rad.TrfZerPerp, 'para' -> rad.TrfZerPara.
# Single source of truth: the legacy-schema adapter emits it as the named
# symmetry 'cyclotron_8fold'; component-based configs define their own sets.
DEFAULT_CYCLOTRON_SYMMETRIES = [
    ("perp", [0, 0, 0], [1, -1, 0]),  # mirror across x=y diagonal
    ("perp", [0, 0, 0], [1, 0, 0]),   # mirror across x=0 plane
    ("perp", [0, 0, 0], [0, 1, 0]),   # mirror across y=0 plane
    ("para", [0, 0, 0], [0, 0, 1]),   # mirror across z=0 plane
]


@dataclass
class ComponentSpec:
    """One machine component in the generic geometry description.

    :param name: unique identifier (also the gmsh model name).
    :param kind: builder-registry key ('stp', 'wedge', 'lid_upper', 'pole',
        'wedge_pair', 'racetrack_pair', ...).
    :param enabled: disabled components are skipped entirely.
    :param file: STP file (kinds that support file-based geometry).
    :param material: name into the config's materials.
    :param symmetry: name into the config's symmetries -- the component's
        FIELD symmetry: applied (TrfZer*) for magnetized parts, declared for
        current sources.
    :param mesh: meshing options, e.g. {'max_size': 50}.
    :param params: kind-specific build parameters (passed to the builder).
    :param shimmed: marks the pole whose shape follows the PoleShape shims;
        rebuilt per optimizer iterate (unless file-based).
    :param perturbative: solve this component perturbatively (e.g. the
        extraction channel): it is EXCLUDED from the main relaxation and
        relaxed afterwards in the frozen field of the main machine
        (rad.RlxPre srcobj), optionally iterating main <-> perturbative.
        Field evaluation is unaffected (the component stays a top-level
        field source with its own symmetry). See ReusableCyclotronSolver.
    :param mesh_group: OPT-IN conforming meshing. Components sharing a
        mesh_group name are built into ONE gmsh model, boolean-FRAGMENTED
        (touching surfaces become a single shared triangulation -- no
        non-conforming contact interfaces, the cause of the refined-mesh
        relaxation floor) and meshed together with per-component sizes;
        the tets are then split back per component, so materials/symmetry/
        perturbative semantics are unchanged. If the group contains the
        shimmed pole, the WHOLE group is rebuilt per optimizer iterate.
    """
    name: str
    kind: str
    enabled: bool = True
    file: Optional[str] = None
    material: Optional[str] = None
    symmetry: Optional[str] = None
    mesh: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    shimmed: bool = False
    perturbative: bool = False
    # Structured polar-grid discretization for STP components (Option C):
    # e.g. {'type': 'polar_grid', 'dr_mm': 120, 'dz_mm': 120,
    # 'dtheta_deg': 2.5, 'core_clip': {'z_max': -140},
    # 'skin_margin_deg': 5, 'theta_span_deg': [0, 45]}.
    # Clean revolved cells become analytic prism elements; CAD detail,
    # the core_clip band (shim envelope!) and thin regions become a
    # conforming tet skin (geometry/structured.py). Works standalone and
    # inside a mesh_group (structured cores + skins + tet members are
    # built conforming in one gmsh model, results digest-cached under
    # output/structured_cache).
    structure: Optional[Dict[str, Any]] = None
    mesh_group: Optional[str] = None


@dataclass
class GeometryConfig:
    # Defaults allow component-based configs to omit this legacy section.
    yoke_build_angle_deg: float = 45.0
    angular_resolution: int = 15
    use_gmsh_occ_pole: Optional[bool] = False
    use_gmsh_occ_yoke: Optional[bool] = False

@dataclass
class FieldEvaluationConfig:
    num_points_circle: int
    radius_min_mm: float
    radius_max_mm: float
    n_eval_pts: int
    use_symmetry: bool = True
    # 'circle' | 'gordon' (closed orbits; tunes) | 'gordon_analytic' (first-order
    # scalloping formulas; tunes) | 'seo' (tracked orbits) -- see core.isochronicity
    iso_method: Optional[str] = "circle"
    # Median-plane map extent/resolution (seo isochronism input + midplane save)
    median_plane_limit_mm: float = 400.0
    median_plane_resolution_mm: float = 1.0
    # Closed-orbit ('gordon') method: dense polar grid radial step and the
    # margin (fraction) beyond [radius_min, radius_max] it must cover (the
    # scalloped orbit reaches ~ +7 % / -6 % of its mean radius), and the RK4
    # steps per turn of the Gordon integrator (1800 = 0.2 deg; converged to
    # 1e-10 in f_rev on the smooth radia grids, halve the cost of 3600).
    eo_radial_step_mm: float = 2.0
    eo_radial_margin: float = 0.15
    eo_steps_per_turn: int = 1800
    # angular density of the dense grid (full-circle-equivalent count; 360 =
    # 1 deg reproduces 720 to 0.01 kHz at half the radia query cost). The
    # spline needs a smooth field, so it is a separate knob from
    # num_points_circle, which the analytic/circle averages use.
    eo_num_points_circle: int = 360
    # 3D bore-field map domain (x, y in [-limit, limit], z in [z_min, z_max])
    bore_xy_limit_mm: float = 50.0
    bore_z_min_mm: float = -100.0
    bore_z_max_mm: float = 25.0
    bore_resolution_mm: float = 0.5

    ISO_METHODS = ('circle', 'gordon', 'gordon_analytic', 'seo')

    def __post_init__(self):
        if self.iso_method is not None:
            method = str(self.iso_method).strip().lower()
            if method not in self.ISO_METHODS:
                raise ValueError(
                    f"field_evaluation.iso_method '{self.iso_method}' is not one of "
                    f"{list(self.ISO_METHODS)}")
            self.iso_method = method

@dataclass
class YokeConfig:
    # Defaults allow synthesis from ComponentSpecs (component-based configs);
    # the programmatic builders always receive explicit values.
    outer_radius_mm: float = 0.0
    inner_radius_mm: float = 0.0
    height_mm: float = 0.0
    segmentation: List[int] = field(default_factory=list)
    window_width_mm: float = 0.0
    max_mesh_size: float = 50.0
    stp_filename: Optional[str] = None


@dataclass
class LidLowerConfig:
    outer_radius_mm: float = 0.0
    inner_radius_mm: float = 0.0
    height_mm: float = 0.0
    segmentation: List[int] = field(default_factory=list)
    max_mesh_size: float = 50.0
    stp_filename: Optional[str] = None


@dataclass
class LidUpperConfig:
    outer_radius_mm_1: float = 0.0
    outer_radius_mm_2: float = 0.0
    inner_radius_mm: float = 0.0
    height_mm: float = 0.0
    segmentation: List[int] = field(default_factory=list)
    hole_diameter_mm: float = 0.0
    hole_center_xy: List[float] = field(default_factory=list)
    cut_out_rf_stem_hole: bool = False
    max_mesh_size: float = 50.0
    stp_filename: Optional[str] = None


@dataclass
class ExtractChannelConfig:
    outer_radius_mm: float = 0.0
    inner_radius_mm: float = 0.0
    height_mm: float = 0.0
    segmentation: List[int] = field(default_factory=list)
    channel_width_mm: float = 0.0
    max_mesh_size: float = 20.0
    window_width_mm: Optional[float] = 0
    start_ang_deg: Optional[float] = 0
    end_ang_deg: Optional[float] = 0
    use_extract_chan: Optional[bool] = False
    stp_filename: Optional[str] = None


@dataclass
class PoleConfig:
    outer_radius_mm: float = 0.0
    inner_radius_mm: float = 0.0
    height_mm: float = 0.0
    full_angle_deg: float = 0.0
    angular_resolution_deg: float = 2.5
    segmentation: List[int] = field(default_factory=list)
    max_mesh_size: float = 50.0
    stp_filename: Optional[str] = None


@dataclass
class SideShimConfig:
    num_rad_segments: int
    angular_resolution_deg: float
    segmentation: List[int]
    # Fallback side half-angle offset when side_offsets_deg is omitted. The
    # shimmed pole is one OCC solid now, so offsets are deltas from the base
    # pole with no minimum -- default 0 (no shim).
    default_offset_deg: Optional[float] = 0.0
    side_offsets_deg: Optional[List[float]] = None
    include: Optional[bool] = True


@dataclass
class TopShimConfig:
    num_rad_segments: int
    angular_resolution_deg: float
    segmentation: List[int]
    default_offset_mm: Optional[float] = 0.0
    top_offsets_mm:  Optional[List[float]] = None
    include: Optional[bool] = True


@dataclass
class CoilConfig:
    radius_min_mm: float
    radius_max_mm: float
    height_mm: float
    midplane_dist: float
    current_A: float
    num_segments: int


@dataclass
class MaterialConfig:
    saturation_field_t: list = field(default_factory=list)
    saturation_curve_m: list = field(default_factory=list)
    linear_curve_m: list = field(default_factory=list)
    bh_filename: Optional[str] = None
    

@dataclass
class SimulationConfig:
    precision: float
    iterations: int
    # Relaxation method for rad.RlxAuto. None -> auto (9 = GPU adaptive
    # Jacobi when GPU relaxation is on, else 10 = its CPU port). Explicit
    # values override, e.g. 4 = radia's classic Gauss-Seidel (CPU): on some
    # models the adaptive-Jacobi |delta M| misfit creeps below target while
    # the magnetization is still far from the true fixed point (check with
    # an M-vs-H residual when in doubt), where method 4 converges honestly.
    relax_method: Optional[int] = None
    # Anderson acceleration of the relaxation (methods 9/10). None leaves
    # the RadiaCUDA default / RADIA_ANDERSON environment variable in charge
    # (currently default OFF); True/False set it explicitly per solve.
    # Benchmarked: ~3x fewer iterations to the same endpoint on
    # well-conditioned models; on ill-conditioned tet meshes it can stall
    # at a higher misfit floor than the plain damped iteration.
    anderson: Optional[bool] = None
    # Perturbative-component solve (components flagged perturbative: True).
    # perturb_iterations = 0: stage-1 only (single frozen-background solve of
    # the perturbative parts -- first-order exact). N > 0: up to N additional
    # main <-> perturbative back-reaction cycles, stopping early when the
    # perturbative magnetization changes by less than perturb_tol [T].
    perturb_iterations: int = 0
    perturb_tol: float = 0.0
    # What radia does when the GPU cannot service the interaction matrix --
    # almost always: the dense matrix (36*N^2 bytes) does not fit in VRAM.
    # None leaves whatever the radia build defaults to ('cpu'); otherwise
    #   'cpu'            fall back to the CPU assembly/relaxation. Correct, but
    #                    can turn a seconds-long solve into an hours-long one.
    #   'break'          raise instead, so a model that has outgrown the card
    #                    is caught immediately and the mesh can be coarsened
    #                    deliberately rather than discovered overnight.
    #   'gpu_streaming'  keep the matrix in host RAM and move it to the device
    #                    one row block per matvec: lifts the VRAM limit on model
    #                    size at the cost of PCIe traffic every iteration
    #                    (results are bit-identical to an in-core solve).
    # Requires a RadiaCUDA build with rad.UtiGpuFallback; older builds ignore
    # the setting with a warning.
    gpu_fallback: Optional[str] = None

    def __post_init__(self):
        allowed = ("cpu", "gpu_streaming", "break")
        if self.gpu_fallback is not None and self.gpu_fallback not in allowed:
            raise ValueError(
                f"simulation.gpu_fallback must be one of {allowed} "
                f"(got {self.gpu_fallback!r})")


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
    convergence_penalty_weight: Optional[float] = 1.0
    # Second-difference (roughness) smoothness penalty on the shim profile,
    # added to the least-squares RESIDUAL vector as w * D2(normalized offsets)
    # per block. Unlike regularization_weight (a MAGNITUDE penalty that shrinks
    # all offsets toward 0), this penalizes only JAGGEDNESS -- a smooth profile,
    # large or small, has zero roughness residual. It removes the mid-radius
    # spike DFO-LS parks in the isochronism null space. Default 0 -> identical
    # to prior behavior (existing runs unaffected). The per-block overrides
    # (side is where the spike lives) fall back to smoothness_weight when None.
    # Tune via the L-curve script (examples/l_curve_smoothness.py); ~0.1 is a
    # sensible starting scale. See objective_function.build_residual_vector.
    smoothness_weight: Optional[float] = 0.0
    smoothness_weight_side: Optional[float] = None
    smoothness_weight_top: Optional[float] = None
    # FIELD-BASIS smoothness (replaces the per-block weights above when either
    # of these is set AND both blocks are optimized). Side and top are BOTH
    # first-order levers on <B>(r) while isochronism constrains only that one
    # function, so the pair has a field-neutral null direction -- measured on
    # the muon design as corr(D2(side), D2(top)) = -0.94, i.e. a dip in one
    # profile cancelled by a bulge in the other. Per-block D2 cannot see that
    # mode apart from legitimate curvature. These weights split the pair into
    #   neutral = (side_norm - rho*top_norm)/sqrt(1+rho^2)   <B>-invariant
    #   active  = (rho*side_norm + top_norm)/sqrt(1+rho^2)   carries <B>(r)
    # so the neutral (wasted) roughness can be penalized hard and the active
    # (needed) curvature left nearly free. rho = d<B>/d(side_norm) over
    # d<B>/d(top_norm); ~1 when the two bound ranges have comparable field
    # authority -- measure with examples/measure_shim_exchange_ratio.py.
    # Equal weights reduce EXACTLY to the per-block penalty (orthonormal
    # rotation). Both None (default) -> per-block behavior, bit for bit.
    smoothness_weight_neutral: Optional[float] = None
    smoothness_weight_active: Optional[float] = None
    smoothness_exchange_ratio: Optional[float] = 1.0
    # BETATRON-RESONANCE BARRIERS (all OPT-IN: weight 0 = off = prior behavior).
    # One-sided hinges added to the residual, zero while the design stays inside
    # the safe band, so they cost nothing until they bind. Measured motivation:
    # the muon optimum ran nu_z up to 0.4804 at r = 294 mm -- 0.020 from the
    # nu_z = 1/2 half-integer and 0.064 from Walkinshaw (nu_r = 2 nu_z) -- with
    # NOTHING in the objective watching. Vertical focusing is abundant here, so
    # these are ceilings; a "maximize nu_z" reward would push INTO the
    # resonances. Requires iso_method: gordon or gordon_analytic (the methods
    # producing tunes).
    #   ceiling      w * max(0, nu_z - nu_z_max)
    #   walkinshaw   w * max(0, walkinshaw_min_distance - (nu_r - 2 nu_z))
    #   floor        w * max(0, nu_z_sq_min - nu_z_sq)   (on nu_z^2, stays finite)
    # Isochronism pins nu_r = gamma, so the walkinshaw term is really a
    # radius-dependent nu_z ceiling at (gamma(r) - delta)/2 and overlaps the flat
    # one -- weight ONE as primary. See objective_function.tune_barriers.
    nu_z_ceiling_weight: Optional[float] = 0.0
    nu_z_max: Optional[float] = 0.42
    walkinshaw_weight: Optional[float] = 0.0
    walkinshaw_min_distance: Optional[float] = 0.15
    nu_z_floor_weight: Optional[float] = 0.0
    nu_z_sq_min: Optional[float] = 0.0
    precondition: Optional[bool] = False
    # Seed the optimizer's starting point (x0) from the config's saved shim
    # offsets (side_offsets_deg / top_offsets_mm) instead of the physics
    # preconditioner -- i.e. warm-start / resume from a known-good design.
    # Takes PRECEDENCE over `precondition` when True. False (default) keeps
    # the prior behavior (preconditioner if precondition else config offsets).
    # (The Nelder-Mead path starts from the config offsets whenever
    # random_init is False; this flag is the DFO-LS equivalent.)
    x0_from_config: Optional[bool] = False
    # Optimize only the shim stations whose radius falls in this band [mm];
    # stations OUTSIDE it stay frozen at their x0 value (the config offsets when
    # paired with x0_from_config, else the preconditioner). The stations sit at
    # linspace(pole.inner_radius_mm, pole.outer_radius_mm, num_rad_segments+1) --
    # the POLE radius, not the field-eval radii. None -> no bound on that side
    # (default None/None optimizes all stations). Pair with x0_from_config to
    # refine a sub-range of a saved design (e.g. re-smooth the inner shims:
    # opt_shim_radius_max_mm: 150). Applies to the DFO-LS optimizer.
    opt_shim_radius_min_mm: Optional[float] = None
    opt_shim_radius_max_mm: Optional[float] = None
    coil_match_tol_mhz: Optional[float] = 0.05
    # DFO-LS solver knobs (joint least-squares path). Defaults preserve prior behavior.
    # rhoend should sit ABOVE the mesh-quantization dead zone (~0.026 norm at mesh=50) or
    # the trust region collapses into a zero-gradient pocket and the optimizer quits early.
    dfols_rhobeg: Optional[float] = 0.1
    dfols_rhoend: Optional[float] = 1e-3
    dfols_maxfun: Optional[int] = None  # None -> max(max_iterations, n_params + 2)
    dfols_objfun_has_noise: Optional[bool] = False   # enables (soft) restarts + gentler termination
    dfols_seek_global_minimum: Optional[bool] = False  # hard restarts via user_params (global search)
    # RESPONSE-MATRIX shimming modes (optimizer: 'matrix-full' | 'matrix-reduced').
    # Probe the sensitivity matrix M = d f_rev(r_i) / d x_j ONCE around x0 at
    # FIXED coil current (full: one column per station; reduced: one column per
    # smooth basis vector -- the gap low-passes station-scale structure, so M is
    # effectively low-rank and ~basis_size columns capture the field authority),
    # then iterate: regularized linear step -> one coil-matched FE evaluation.
    # The step minimizes the SAME weighted residual as DFO-LS (flatness +
    # smoothness from the shared builder), so the two paths agree on what good
    # means; the DFO-LS path itself is untouched.
    matrix_broyden: Optional[bool] = False      # secant-update M from iteration solves
    matrix_basis_size: Optional[int] = 8        # smooth modes per block (reduced mode)
    matrix_probe_delta: Optional[float] = 0.05  # probe step [normalized units]
    matrix_relaxation: Optional[float] = 0.7    # alpha applied to each step (spec 7.2)
    matrix_max_outer: Optional[int] = 10        # max linear-step iterations
    matrix_ftol_rel: Optional[float] = 0.01     # stop when objective improves less


@dataclass
class RFConfig:
    """RF / longitudinal-dynamics description for phase-slip reporting.

    These are STATED INPUTS (isochronism_reporting_spec.md section 6.1): a phase
    slip number without them is uninterpretable. All optional -- with
    ``harmonic_h`` or ``de0_kev_per_turn`` unset the phase-slip report is
    disabled and nothing changes for existing configs.

    ``f_rf_mhz`` defaults to harmonic_h * optimization.target_frequency_mhz.
    ``de0_kev_per_turn`` is the nominal energy gain per turn AT CREST (the
    cos(phi) feedback is applied by the phase-slip integral itself).
    ``phi0_deg`` is the RF phase at the exit of the central region -- carried
    explicitly so the central bump's contribution is visible, not assumed zero.
    ``phi_budget_deg`` is a design CHOICE (typical 10 tight .. 30 accepted;
    hard physical limit 90, where the beam decelerates).
    """
    harmonic_h: Optional[int] = None
    f_rf_mhz: Optional[float] = None
    de0_kev_per_turn: Optional[float] = None
    phi0_deg: Optional[float] = 0.0
    phi_budget_deg: Optional[float] = 10.0
    e_inj_mev: Optional[float] = None
    e_final_mev: Optional[float] = None


@dataclass
class VisualizationConfig:
    show_opengl: bool
    comsol_filename: Optional[str] = None
    live_plot: Optional[bool] = True
    # Median-plane Bz visualization: always makes a 2D matplotlib contour plot;
    # with show_opengl also overlaid as a semi-transparent plane in the 3D viewer.
    show_median_plane_field: Optional[bool] = False
    # Resolution of the DISPLAY map [mm]; None falls back to
    # field_evaluation.median_plane_resolution_mm (which also feeds the seo
    # solver, so keep that one fine and relax only this).
    field_map_resolution_mm: Optional[float] = 2.0


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
    extract_channel: ExtractChannelConfig
    pole: PoleConfig
    coil: CoilConfig
    material: MaterialConfig
    side_shim: SideShimConfig
    top_shim: TopShimConfig
    simulation: SimulationConfig
    optimization: OptimizationConfig
    visualization: VisualizationConfig
    rf: RFConfig = field(default_factory=RFConfig)
    # Generic machine description (single source for the geometry builders).
    # Populated by BOTH schemas: parsed directly from component-based YAMLs,
    # synthesized by the legacy adapter otherwise.
    components: List[ComponentSpec] = field(default_factory=list)
    materials_def: Dict[str, dict] = field(default_factory=dict)
    symmetries_def: Dict[str, list] = field(default_factory=dict)
    # Optional local mesh refinement for conforming mesh groups: a list of
    # axis-aligned boxes {x_min, x_max, y_min, y_max, z_min, z_max, size_mm,
    # thickness_mm} (gmsh Box size fields; inside a box the mesh size is the
    # minimum of size_mm and the per-component caps). Used e.g. to grade the
    # gap-facing 100 mm of a pole finer than its base.
    mesh_refinement: Optional[List[dict]] = None

    # ------------------------------------------------------------------
    # Component accessors
    # ------------------------------------------------------------------
    def component(self, name: str) -> ComponentSpec:
        for spec in self.components:
            if spec.name == name:
                return spec
        raise KeyError(f"No component named {name!r} in config")

    def resolved_symmetry(self, name: Optional[str]) -> list:
        """(kind, point, normal) tuples for a named symmetry set ([] for None)."""
        if not name:
            return []
        try:
            entries = self.symmetries_def[name]
        except KeyError:
            raise KeyError(f"Component references undefined symmetry {name!r}")
        return [(str(kind), list(point), list(normal))
                for kind, point, normal in entries]

    @classmethod
    def from_yaml(cls, filepath: str) -> 'CyclotronConfig':
        """Load configuration from YAML file (legacy or component-based schema).

        Relative file paths inside the config (STP geometry, BH material
        curve, comparison maps) are resolved against the YAML's own
        directory when they exist there, so project configs work no matter
        where the script runs from.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError("Config file is empty")

        config_dir = os.path.dirname(os.path.abspath(filepath))

        def _resolve_path(value: Optional[str]) -> Optional[str]:
            if value and not os.path.isabs(value):
                candidate = os.path.join(config_dir, value)
                if os.path.exists(candidate):
                    return candidate
            return value

        def _resolve(section: str, key: str) -> None:
            value = (data.get(section) or {}).get(key)
            resolved = _resolve_path(value)
            if resolved is not value:
                data[section][key] = resolved

        _resolve('visualization', 'comsol_filename')

        if 'components' in data:
            config = cls._from_component_yaml(data, _resolve_path)
            config.mesh_refinement = data.get('mesh_refinement')
            return config

        for part in ('yoke', 'lid_lower', 'lid_upper', 'pole', 'extract_channel'):
            _resolve(part, 'stp_filename')
        _resolve('material', 'bh_filename')

        try:
            config = cls(
                seed=data.get('seed', 42),
                particle_species=data.get('particle_species', 'muon'),
                max_machine_size_mm=data.get('max_machine_size_mm', 860.0),
                geometry=_from_dict(GeometryConfig, data.get('geometry')),
                field_evaluation=_from_dict(FieldEvaluationConfig, data['field_evaluation']),
                yoke=YokeConfig(**data['yoke']),
                lid_lower=LidLowerConfig(**data['lid_lower']),
                lid_upper=LidUpperConfig(**data['lid_upper']),
                extract_channel=ExtractChannelConfig(**data['extract_channel']),
                pole=PoleConfig(**data['pole']),
                coil=CoilConfig(**data['coil']),
                material=MaterialConfig(**data['material']),
                side_shim=SideShimConfig(**data['side_shim']),
                top_shim=TopShimConfig(**data['top_shim']),
                simulation=SimulationConfig(**data['simulation']),
                optimization=OptimizationConfig(**data['optimization']),
                visualization=VisualizationConfig(**data['visualization']),
                rf=_from_dict(RFConfig, data.get('rf')),
            )
        except KeyError as e:
            raise ValueError(f"Missing required config key: {e}")

        config._validate_numeric_types()
        config._adapt_legacy_to_components()
        return config

    # ------------------------------------------------------------------
    # Legacy schema -> component specs (single place mapping old sections
    # onto the generic machine description consumed by the builders)
    # ------------------------------------------------------------------
    def _adapt_legacy_to_components(self) -> None:
        self.materials_def = {'iron': (
            {'type': 'bh_file', 'file': self.material.bh_filename}
            if self.material.bh_filename is not None else
            {'type': 'sat_iso_frm',
             'saturation_field_t': self.material.saturation_field_t,
             'saturation_curve_m': self.material.saturation_curve_m,
             'linear_curve_m': self.material.linear_curve_m}
        )}
        self.symmetries_def = {
            'cyclotron_8fold': [
                [kind, list(point), list(normal)]
                for kind, point, normal in DEFAULT_CYCLOTRON_SYMMETRIES
            ],
            # median-plane mirror only (e.g. the extraction channel: breaks
            # the azimuthal 8-fold but is exactly z-symmetric)
            'median_z': [['para', [0, 0, 0], [0, 0, 1]]],
        }

        build_ang = self.geometry.yoke_build_angle_deg
        pole_zs = -(self.yoke.height_mm + self.lid_lower.height_mm)
        specs: List[ComponentSpec] = []

        specs.append(ComponentSpec(
            name='yoke', kind='stp' if self.yoke.stp_filename else 'wedge',
            file=self.yoke.stp_filename, material='iron',
            symmetry='cyclotron_8fold',
            mesh={'max_size': self.yoke.max_mesh_size},
            params={} if self.yoke.stp_filename else dict(
                inner_radius_mm=self.yoke.inner_radius_mm,
                outer_radius_mm=self.yoke.outer_radius_mm,
                height_mm=self.yoke.height_mm, z_offset_mm=0.0,
                end_ang_deg=build_ang, include_window=True,
                window_width_mm=self.yoke.window_width_mm)))

        specs.append(ComponentSpec(
            name='lid_lower', kind='stp' if self.lid_lower.stp_filename else 'wedge',
            file=self.lid_lower.stp_filename, material='iron',
            symmetry='cyclotron_8fold',
            mesh={'max_size': self.lid_lower.max_mesh_size},
            params={} if self.lid_lower.stp_filename else dict(
                inner_radius_mm=self.lid_lower.inner_radius_mm,
                outer_radius_mm=self.lid_lower.outer_radius_mm,
                height_mm=self.lid_lower.height_mm,
                z_offset_mm=-self.yoke.height_mm, end_ang_deg=build_ang)))

        specs.append(ComponentSpec(
            name='lid_upper', kind='stp' if self.lid_upper.stp_filename else 'lid_upper',
            file=self.lid_upper.stp_filename, material='iron',
            symmetry='cyclotron_8fold',
            mesh={'max_size': self.lid_upper.max_mesh_size},
            params={} if self.lid_upper.stp_filename else dict(
                inner_radius_mm=self.lid_upper.inner_radius_mm,
                outer_radius_mm_1=self.lid_upper.outer_radius_mm_1,
                outer_radius_mm_2=self.lid_upper.outer_radius_mm_2,
                height_mm=self.lid_upper.height_mm, z_offset_mm=pole_zs,
                end_ang_deg=build_ang,
                seg_theta=self.lid_upper.segmentation[1])))

        specs.append(ComponentSpec(
            name='pole', kind='pole', file=self.pole.stp_filename,
            material='iron', symmetry='cyclotron_8fold', shimmed=True,
            mesh={'max_size': self.pole.max_mesh_size},
            params={} if self.pole.stp_filename else dict(
                inner_radius_mm=self.pole.inner_radius_mm,
                outer_radius_mm=self.pole.outer_radius_mm,
                height_mm=self.pole.height_mm,
                half_angle_deg=self.pole.full_angle_deg / 2.0,
                pole_zs=pole_zs)))

        ec = self.extract_channel
        # Single wedge ABOVE the median plane + exact para-z mirror (the
        # channel's environment is z-symmetric): replaces the old wedge_pair,
        # halving the channel's relax elements.
        specs.append(ComponentSpec(
            name='extract_channel', kind='wedge',
            enabled=bool(ec.use_extract_chan), file=ec.stp_filename,
            material='iron', symmetry='median_z',
            mesh={'max_size': ec.max_mesh_size},
            params=dict(inner_radius_mm=ec.inner_radius_mm,
                        outer_radius_mm=ec.outer_radius_mm,
                        height_mm=ec.height_mm,
                        z_offset_mm=ec.height_mm + ec.channel_width_mm / 2.0,
                        start_ang_deg=ec.start_ang_deg,
                        end_ang_deg=ec.end_ang_deg)))

        specs.append(ComponentSpec(
            name='coils', kind='racetrack_pair', symmetry='cyclotron_8fold',
            params=dict(radius_min_mm=self.coil.radius_min_mm,
                        radius_max_mm=self.coil.radius_max_mm,
                        height_mm=self.coil.height_mm,
                        midplane_dist=self.coil.midplane_dist,
                        current_A=self.coil.current_A,
                        num_segments=self.coil.num_segments)))

        self.components = specs

    # ------------------------------------------------------------------
    # Component-based schema
    # ------------------------------------------------------------------
    @classmethod
    def _from_component_yaml(cls, data: dict, resolve_path) -> 'CyclotronConfig':
        """Parse the component-based schema and synthesize the legacy
        dataclasses that other code still reads (coil is the LIVE value the
        solver mutates per current; pole feeds the physics preconditioner)."""
        specs: List[ComponentSpec] = []
        for entry in data['components']:
            entry = dict(entry)
            entry['file'] = resolve_path(entry.get('file'))
            specs.append(ComponentSpec(
                name=entry['name'], kind=entry['kind'],
                enabled=entry.get('enabled', True), file=entry.get('file'),
                material=entry.get('material'), symmetry=entry.get('symmetry'),
                mesh=entry.get('mesh') or {}, params=entry.get('params') or {},
                shimmed=entry.get('shimmed', False),
                perturbative=entry.get('perturbative', False),
                mesh_group=entry.get('mesh_group'),
                structure=entry.get('structure')))

        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate component names in config: {names}")

        materials_def = {}
        for mname, mdef in (data.get('materials') or {}).items():
            mdef = dict(mdef)
            if 'file' in mdef:
                mdef['file'] = resolve_path(mdef['file'])
            materials_def[mname] = mdef
        symmetries_def = dict(data.get('symmetries') or {})

        def _spec_params(name, default=None):
            for s in specs:
                if s.name == name:
                    return s
            return default

        # Reverse-fill the legacy dataclasses still consumed elsewhere.
        coil_spec = next((s for s in specs if s.kind == 'racetrack_pair'), None)
        coil = CoilConfig(**coil_spec.params) if coil_spec else CoilConfig(
            radius_min_mm=0, radius_max_mm=1, height_mm=1, midplane_dist=0,
            current_A=0.0, num_segments=1)

        pole_spec = next((s for s in specs if s.shimmed), None)
        pole_kwargs = {}
        if pole_spec:
            valid = {'inner_radius_mm', 'outer_radius_mm', 'height_mm',
                     'angular_resolution_deg'}
            pole_kwargs = {k: v for k, v in pole_spec.params.items() if k in valid}
            if 'half_angle_deg' in pole_spec.params:
                pole_kwargs['full_angle_deg'] = 2.0 * pole_spec.params['half_angle_deg']
            pole_kwargs['max_mesh_size'] = pole_spec.mesh.get('max_size', 50.0)
            pole_kwargs['stp_filename'] = pole_spec.file

        channel_spec = _spec_params('extract_channel')
        extract = ExtractChannelConfig(
            use_extract_chan=bool(channel_spec and channel_spec.enabled))

        iron_def = next(iter(materials_def.values()), {})
        material = MaterialConfig(bh_filename=iron_def.get('file'))

        try:
            config = cls(
                seed=data.get('seed', 42),
                particle_species=data.get('particle_species', 'muon'),
                max_machine_size_mm=data.get('max_machine_size_mm', 860.0),
                geometry=_from_dict(GeometryConfig, data.get('geometry')),
                field_evaluation=_from_dict(FieldEvaluationConfig, data['field_evaluation']),
                yoke=YokeConfig(),
                lid_lower=LidLowerConfig(),
                lid_upper=LidUpperConfig(),
                extract_channel=extract,
                pole=PoleConfig(**pole_kwargs),
                coil=coil,
                material=material,
                side_shim=SideShimConfig(**data['side_shim']),
                top_shim=TopShimConfig(**data['top_shim']),
                simulation=SimulationConfig(**data['simulation']),
                optimization=OptimizationConfig(**data['optimization']),
                visualization=VisualizationConfig(**data['visualization']),
                rf=_from_dict(RFConfig, data.get('rf')),
                components=specs,
                materials_def=materials_def,
                symmetries_def=symmetries_def,
            )
        except KeyError as e:
            raise ValueError(f"Missing required config key: {e}")
        config._validate_numeric_types()
        return config

    def _validate_numeric_types(self) -> None:
        """Load-time guard: numeric fields carrying YAML strings are coerced or
        rejected with the yml key named (see _coerce_numeric_fields)."""
        for name in ('geometry', 'field_evaluation', 'yoke', 'lid_lower',
                     'lid_upper', 'extract_channel', 'pole', 'coil', 'material',
                     'side_shim', 'top_shim', 'simulation', 'optimization',
                     'visualization', 'rf'):
            section = getattr(self, name, None)
            if section is not None:
                _coerce_numeric_fields(section, name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'particle_species': self.particle_species,
            'max_machine_size_mm': self.max_machine_size_mm,
            'seed': self.seed,
            'geometry': {
                'yoke_build_angle_deg': self.geometry.yoke_build_angle_deg,
                'angular_resolution': self.geometry.angular_resolution,
                'use_gmsh_occ_pole': self.geometry.use_gmsh_occ_pole,
                'use_gmsh_occ_yoke': self.geometry.use_gmsh_occ_yoke,
            },
            'field_evaluation': {
                'num_points_circle': self.field_evaluation.num_points_circle,
                'radius_min_mm': self.field_evaluation.radius_min_mm,
                'radius_max_mm': self.field_evaluation.radius_max_mm,
                'n_eval_pts': self.field_evaluation.n_eval_pts,
                'use_symmetry':  self.field_evaluation.use_symmetry,
                'iso_method': self.field_evaluation.iso_method,
                'median_plane_limit_mm': self.field_evaluation.median_plane_limit_mm,
                'median_plane_resolution_mm': self.field_evaluation.median_plane_resolution_mm,
                'eo_radial_step_mm': self.field_evaluation.eo_radial_step_mm,
                'eo_radial_margin': self.field_evaluation.eo_radial_margin,
                'eo_steps_per_turn': self.field_evaluation.eo_steps_per_turn,
                'eo_num_points_circle': self.field_evaluation.eo_num_points_circle,
                'bore_xy_limit_mm': self.field_evaluation.bore_xy_limit_mm,
                'bore_z_min_mm': self.field_evaluation.bore_z_min_mm,
                'bore_z_max_mm': self.field_evaluation.bore_z_max_mm,
                'bore_resolution_mm': self.field_evaluation.bore_resolution_mm,
            },
            'yoke': {
                'outer_radius_mm': self.yoke.outer_radius_mm,
                'inner_radius_mm': self.yoke.inner_radius_mm,
                'height_mm': self.yoke.height_mm,
                'segmentation': self.yoke.segmentation,
                'window_width_mm': self.yoke.window_width_mm,
                'max_mesh_size': self.yoke.max_mesh_size,
                'stp_filename': self.yoke.stp_filename,
            },
            'lid_lower': {
                'outer_radius_mm': self.lid_lower.outer_radius_mm,
                'inner_radius_mm': self.lid_lower.inner_radius_mm,
                'height_mm': self.lid_lower.height_mm,
                'segmentation': self.lid_lower.segmentation,
                'max_mesh_size': self.lid_lower.max_mesh_size,
                'stp_filename': self.lid_lower.stp_filename,
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
                'max_mesh_size': self.lid_upper.max_mesh_size,
                'stp_filename': self.lid_upper.stp_filename,
            },
            'extract_channel': {
                'outer_radius_mm': self.extract_channel.outer_radius_mm,
                'inner_radius_mm': self.extract_channel.inner_radius_mm,
                'height_mm': self.extract_channel.height_mm,
                'segmentation': self.extract_channel.segmentation,
                'channel_width_mm': self.extract_channel.channel_width_mm,
                'max_mesh_size': self.extract_channel.max_mesh_size,
                'window_width_mm': self.extract_channel.window_width_mm,
                'start_ang_deg': self.extract_channel.start_ang_deg,
                'end_ang_deg': self.extract_channel.end_ang_deg,
                'use_extract_chan': self.extract_channel.use_extract_chan,
                'stp_filename': self.extract_channel.stp_filename,
            },
            'pole': {
                'outer_radius_mm': self.pole.outer_radius_mm,
                'inner_radius_mm': self.pole.inner_radius_mm,
                'height_mm': self.pole.height_mm,
                'full_angle_deg': self.pole.full_angle_deg,
                'segmentation': self.pole.segmentation,
                'max_mesh_size': self.pole.max_mesh_size,
                'stp_filename': self.pole.stp_filename,
            },
            'coil': {
                'radius_min_mm': self.coil.radius_min_mm,
                'radius_max_mm': self.coil.radius_max_mm,
                'height_mm': self.coil.height_mm,
                'midplane_dist': self.coil.midplane_dist,
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
                'opt_side': self.optimization.opt_side,
                'opt_coil': self.optimization.opt_coil,
                'convergence_penalty_weight': self.optimization.convergence_penalty_weight,
                'smoothness_weight': self.optimization.smoothness_weight,
                'smoothness_weight_side': self.optimization.smoothness_weight_side,
                'smoothness_weight_top': self.optimization.smoothness_weight_top,
                'smoothness_weight_neutral': self.optimization.smoothness_weight_neutral,
                'smoothness_weight_active': self.optimization.smoothness_weight_active,
                'smoothness_exchange_ratio': self.optimization.smoothness_exchange_ratio,
                'nu_z_ceiling_weight': self.optimization.nu_z_ceiling_weight,
                'nu_z_max': self.optimization.nu_z_max,
                'walkinshaw_weight': self.optimization.walkinshaw_weight,
                'walkinshaw_min_distance': self.optimization.walkinshaw_min_distance,
                'nu_z_floor_weight': self.optimization.nu_z_floor_weight,
                'nu_z_sq_min': self.optimization.nu_z_sq_min,
                'precondition': self.optimization.precondition,
                'x0_from_config': self.optimization.x0_from_config,
                'opt_shim_radius_min_mm': self.optimization.opt_shim_radius_min_mm,
                'opt_shim_radius_max_mm': self.optimization.opt_shim_radius_max_mm,
                'coil_match_tol_mhz': self.optimization.coil_match_tol_mhz,
                'dfols_rhobeg': self.optimization.dfols_rhobeg,
                'dfols_rhoend': self.optimization.dfols_rhoend,
                'dfols_maxfun': self.optimization.dfols_maxfun,
                'dfols_objfun_has_noise': self.optimization.dfols_objfun_has_noise,
                'dfols_seek_global_minimum': self.optimization.dfols_seek_global_minimum,
                'matrix_broyden': self.optimization.matrix_broyden,
                'matrix_basis_size': self.optimization.matrix_basis_size,
                'matrix_probe_delta': self.optimization.matrix_probe_delta,
                'matrix_relaxation': self.optimization.matrix_relaxation,
                'matrix_max_outer': self.optimization.matrix_max_outer,
                'matrix_ftol_rel': self.optimization.matrix_ftol_rel,
                'side_shim_min_deg': self.optimization.side_shim_min_deg,
                'side_shim_max_deg': self.optimization.side_shim_max_deg,
                'top_shim_min_mm': self.optimization.top_shim_min_mm,
                'top_shim_max_mm': self.optimization.top_shim_max_mm,
            },
            'visualization': {
                'show_opengl': self.visualization.show_opengl,
                'comsol_filename': self.visualization.comsol_filename,
                'live_plot': self.visualization.live_plot,
                'show_median_plane_field': self.visualization.show_median_plane_field,
                'field_map_resolution_mm': self.visualization.field_map_resolution_mm,
            },
            'rf': {
                'harmonic_h': self.rf.harmonic_h,
                'f_rf_mhz': self.rf.f_rf_mhz,
                'de0_kev_per_turn': self.rf.de0_kev_per_turn,
                'phi0_deg': self.rf.phi0_deg,
                'phi_budget_deg': self.rf.phi_budget_deg,
                'e_inj_mev': self.rf.e_inj_mev,
                'e_final_mev': self.rf.e_final_mev,
            },
        }
