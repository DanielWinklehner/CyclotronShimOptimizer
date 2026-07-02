"""Field evaluation and export using Radia (MPI-aware, symmetry-exploiting).

Symmetry handling is generic: the field functions read each source's declared
field symmetries from the geometry components (see geometry.components /
geometry.symmetry), group the top-level sources by symmetry set, evaluate each
group only on the fundamental subset of the requested grid, and fold the
values back with the proper vector transforms. Sources without symmetries
(e.g. the extraction channel) are automatically evaluated on the full grid --
nothing about "octants" or 8-fold is hardcoded.

Public API:
  - get_field_3d / get_field_2d / get_median_plane_field: grid maps returned
    as PyPATools Field objects (meters / Tesla).
  - get_field_rz: Bz on midplane circles (isochronism input), returned as an
    RZFieldGrid carrying the azimuthal sample angles actually used.
  - save_median_plane_field / save_bore_field: thin wrappers that obtain the
    field via the getters and write it through Field.save().
  - ReusableCyclotronSolver / evaluate_radii_parallel: build + relax + query.

NOTE: Radia returns field values only on MPI rank 0; the field functions
return None on all other ranks (the rad.Fld calls themselves are collective).
"""

import time
from typing import List, NamedTuple, Optional, Union

import numpy as np
import radia as rad

from config_io.config import CyclotronConfig
from geometry.components import BaseRadiaComponent
from geometry.geometry import build_coils, build_iron
from geometry.pole_shape import PoleShape
from geometry.symmetry import (
    SymmetryTuple,
    azimuthal_sector,
    canonical_symmetry_set,
    collect_field_symmetries,
    reduce_grid,
    symmetry_group,
)
from PyPATools.field import Field

ComponentOrId = Union[BaseRadiaComponent, int]


class RZFieldGrid(NamedTuple):
    """Bz sampled on midplane circles, with the azimuthal angles actually used.

    ``angles`` may span less than the full circle when the field's symmetry
    allowed folding (e.g. [0, pi/4) for the 8-fold cyclotron); consumers
    (core.isochronicity) read the angles from here instead of re-deriving them.
    """
    bz: np.ndarray        # (n_radii, n_angles)
    angles: np.ndarray    # (n_angles,) [rad]
    radii_mm: np.ndarray  # (n_radii,)


class _SourceGroup(NamedTuple):
    radia_id: int
    symmetries: List[SymmetryTuple]
    temp: bool  # True if radia_id is a throwaway container to UtiDel afterwards


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fld(radia_id: int, components: str, points, use_gpu: bool = True,
         *, precision: str = "double", rank: int = 0, verbosity: int = 1):
    """rad.Fld with the RadiaCUDA use_gpu/precision kwargs, falling back to
    plain radia.

    precision='single' selects RadiaCUDA's fp32 polygon-face kernel
    (visualization-grade, ~1e-4 relative; much faster on GeForce-class GPUs).
    Keep the default 'double' for anything feeding tracking or isochronism.

    Warns when a GPU-requested evaluation was actually serviced by the CPU
    backend (e.g. an unsupported component id -- RadiaCUDA's GPU gate accepts
    only 'b'/'bx'/'by'/'bz'), since that fallback is silent and can be
    orders of magnitude slower.
    """
    kwargs = {"use_gpu": use_gpu}
    if precision != "double":
        kwargs["precision"] = precision
    try:
        result = rad.Fld(radia_id, components, points, **kwargs)
    except TypeError:
        return rad.Fld(radia_id, components, points)

    if use_gpu and rank <= 0 and verbosity >= 1 and hasattr(rad, "UtiFldLastBackend"):
        if rad.UtiFldLastBackend() == "cpu":
            print(f"  WARNING: rad.Fld('{components}', {len(points)} points) fell back "
                  "to the CPU backend despite use_gpu=True!", flush=True)
    return result


def _component_id(component: ComponentOrId) -> int:
    if isinstance(component, (int, np.integer)):
        return int(component)
    return component.id


def symmetric_axis(limit_mm: float, resolution_mm: float) -> np.ndarray:
    """Exactly mirror-symmetric axis [-limit, limit] with the given spacing.

    Built by negating the positive half, so x[i] == -x[-1-i] bit-exactly
    (required for lossless symmetry folding). The extent is rounded to the
    nearest multiple of the resolution.
    """
    n_half = int(round(limit_mm / resolution_mm))
    half = np.arange(n_half + 1) * float(resolution_mm)
    return np.concatenate([-half[:0:-1], half])


def _field_source_groups(component: ComponentOrId,
                         use_symmetry: bool) -> List[_SourceGroup]:
    """Group the top-level field sources by their declared field-symmetry set.

    Children with identical symmetry sets are combined into one temporary
    radia container (single rad.Fld call, folded once); each distinct set gets
    its own group folded by its own symmetries. A bare radia id, a component
    with its own declaration, or use_symmetry=False all yield a single group.
    """
    if isinstance(component, (int, np.integer)):
        return [_SourceGroup(int(component), [], False)]
    if not use_symmetry:
        return [_SourceGroup(component.id, [], False)]

    children = list(component.iter_cached_children())
    if component.symmetries or not children:
        return [_SourceGroup(component.id, list(component.symmetries), False)]

    grouped: dict = {}
    for child in children:
        syms = collect_field_symmetries(child)
        key = canonical_symmetry_set(syms)
        grouped.setdefault(key, (syms, []))[1].append(child.id)

    groups: List[_SourceGroup] = []
    for syms, ids in grouped.values():
        if len(ids) == 1:
            groups.append(_SourceGroup(ids[0], syms, False))
        else:
            groups.append(_SourceGroup(int(rad.ObjCnt(ids)), syms, True))
    return groups


def _evaluate_b_grid(component: ComponentOrId,
                     axes_mm,
                     *,
                     use_symmetry: bool = True,
                     rank: int = 0,
                     comm=None,
                     verbosity: int = 1,
                     use_gpu: bool = True,
                     gpu_precision: str = "double") -> Optional[np.ndarray]:
    """Evaluate (Bx, By, Bz) on a regular (x, y, z) grid, folding symmetries.

    :param axes_mm: three sorted 1D coordinate arrays [mm] (singletons allowed).
    :return: (Nx, Ny, Nz, 3) array [T] on rank 0, None on other ranks.
    """
    axes = [np.asarray(a, dtype=float) for a in axes_mm]
    shape = tuple(len(a) for a in axes)
    say = rank <= 0 and verbosity >= 1

    groups = _field_source_groups(component, use_symmetry)
    b_total = np.zeros(shape + (3,)) if rank <= 0 else None

    try:
        for group in groups:
            ops = symmetry_group(group.symmetries)
            reduction = reduce_grid(axes, ops)
            if say:
                print(f"  Field source group ({len(group.symmetries)} symmetries, "
                      f"{reduction.n_ops} usable ops): evaluating "
                      f"{len(reduction.eval_points)} of {reduction.n_total} grid points...",
                      flush=True)
            # Component id 'b' (NOT 'bxbybz'): RadiaCUDA's GPU gate in
            # RadFld only accepts 'b'/'bx'/'by'/'bz' -- 'bxbybz' silently
            # falls through to the (very slow) CPU path.
            b_eval = _fld(group.radia_id, 'b',
                          reduction.eval_points.tolist(), use_gpu=use_gpu,
                          precision=gpu_precision, rank=rank, verbosity=verbosity)
            if rank <= 0:
                b_full = reduction.scatter_vector(np.asarray(b_eval, dtype=float))
                b_total += b_full.reshape(shape + (3,))
    finally:
        for group in groups:
            if group.temp:
                try:
                    rad.UtiDel(group.radia_id)
                except RuntimeError:
                    pass  # already gone (e.g. rad.UtiDelAll) -> idempotent

    return b_total


# ---------------------------------------------------------------------------
# Field getters
# ---------------------------------------------------------------------------
def get_field_3d(component: ComponentOrId,
                 x_mm, y_mm, z_mm,
                 *,
                 use_symmetry: bool = True,
                 rank: int = 0,
                 comm=None,
                 verbosity: int = 1,
                 use_gpu: bool = True,
                 gpu_precision: str = "double",
                 label: str = "Radia B-field (3D)") -> Optional[Field]:
    """Full 3D B-field on a regular grid, exploiting declared symmetries.

    :param component: geometry component (preferred; carries the symmetry
        metadata) or a raw radia id (no folding).
    :param x_mm, y_mm, z_mm: sorted 1D coordinate arrays [mm].
    :param use_symmetry: master switch; what "symmetry" means is read from the
        component metadata, and group elements that do not map the given grid
        onto itself are dropped automatically (e.g. z-mirror on an asymmetric
        z range).
    :param gpu_precision: 'double' (default) or 'single' (fp32 GPU kernel,
        visualization-grade only -- do NOT use for tracking/isochronism maps).
    :return: PyPATools Field (grid in meters, values in Tesla) on rank 0,
        None on other ranks.
    """
    b_grid = _evaluate_b_grid(component, (x_mm, y_mm, z_mm),
                              use_symmetry=use_symmetry, rank=rank, comm=comm,
                              verbosity=verbosity, use_gpu=use_gpu,
                              gpu_precision=gpu_precision)
    if rank > 0:
        return None

    grid_m = {'x': np.asarray(x_mm, dtype=float) * 1e-3,
              'y': np.asarray(y_mm, dtype=float) * 1e-3,
              'z': np.asarray(z_mm, dtype=float) * 1e-3}
    values = {'x': b_grid[..., 0], 'y': b_grid[..., 1], 'z': b_grid[..., 2]}
    return Field.from_arrays(grid_m, values, label=label)


def get_field_2d(component: ComponentOrId,
                 x_mm, y_mm,
                 z_mm: float = 0.0,
                 *,
                 use_symmetry: bool = True,
                 rank: int = 0,
                 comm=None,
                 verbosity: int = 1,
                 use_gpu: bool = True,
                 gpu_precision: str = "double",
                 label: str = "Radia B-field (plane)") -> Optional[Field]:
    """B-field on the horizontal plane z = z_mm (2D Field over x, y).

    A slice at z != 0 automatically loses the midplane-mirror fold but keeps
    the in-plane symmetries; z = 0 uses the full declared set.
    """
    z_axis = np.array([float(z_mm)])
    b_grid = _evaluate_b_grid(component, (x_mm, y_mm, z_axis),
                              use_symmetry=use_symmetry, rank=rank, comm=comm,
                              verbosity=verbosity, use_gpu=use_gpu,
                              gpu_precision=gpu_precision)
    if rank > 0:
        return None

    grid_m = {'x': np.asarray(x_mm, dtype=float) * 1e-3,
              'y': np.asarray(y_mm, dtype=float) * 1e-3,
              'z': z_axis * 1e-3}
    values = {'x': b_grid[:, :, 0, 0],
              'y': b_grid[:, :, 0, 1],
              'z': b_grid[:, :, 0, 2]}
    return Field.from_arrays(grid_m, values, label=label)


def get_median_plane_field(component: ComponentOrId,
                           limit_mm: float = 400.0,
                           resolution_mm: float = 1.0,
                           *,
                           use_symmetry: bool = True,
                           rank: int = 0,
                           comm=None,
                           verbosity: int = 1,
                           use_gpu: bool = True,
                           gpu_precision: str = "double") -> Optional[Field]:
    """B-field on the median plane (z=0) over [-limit, limit]^2 [mm]."""
    if rank <= 0 and verbosity >= 1:
        print(f"Calculating median-plane field (limit={limit_mm} mm, "
              f"resolution={resolution_mm} mm, symmetry={'on' if use_symmetry else 'off'}, "
              f"gpu_precision={gpu_precision})...",
              flush=True)
    axis = symmetric_axis(limit_mm, resolution_mm)
    field = get_field_2d(component, axis, axis, 0.0,
                         use_symmetry=use_symmetry, rank=rank, comm=comm,
                         verbosity=verbosity, use_gpu=use_gpu,
                         gpu_precision=gpu_precision,
                         label="Radia B-field (median plane)")
    if rank <= 0 and verbosity >= 1:
        print("Done!", flush=True)
    return field


def get_field_rz(component: ComponentOrId,
                 radii_mm,
                 num_angles: int = 1000,
                 *,
                 use_symmetry: bool = True,
                 rank: int = 0,
                 comm=None,
                 verbosity: int = 1,
                 use_gpu: bool = True) -> Optional[RZFieldGrid]:
    """Bz on midplane circles at the given radii (isochronism input).

    The azimuthal fundamental sector is derived from the component's declared
    symmetries (pi/4 for the 8-fold cyclotron, full circle when no usable
    symmetry, e.g. with the extraction channel present); ``num_angles`` is the
    full-circle-equivalent count, so the effective angular resolution is
    independent of the fold.

    :return: RZFieldGrid(bz (Nr, Ntheta), angles, radii_mm) on rank 0, None on
        other ranks.
    """
    radii = np.atleast_1d(np.asarray(radii_mm, dtype=float))

    syms: List[SymmetryTuple] = []
    if use_symmetry and not isinstance(component, (int, np.integer)):
        syms = collect_field_symmetries(component)
    sector = azimuthal_sector(syms)

    n_ang = max(1, int(round(num_angles * sector / (2.0 * np.pi))))
    angles = np.linspace(0.0, sector, n_ang, endpoint=False)

    if rank <= 0 and verbosity >= 1:
        print(f"Calculating Bz on {len(radii)} circles "
              f"({n_ang} angles over {np.degrees(sector):.1f} deg sector)...", flush=True)

    points = np.zeros((len(radii), n_ang, 3))
    points[:, :, 0] = radii[:, None] * np.cos(angles)[None, :]
    points[:, :, 1] = radii[:, None] * np.sin(angles)[None, :]

    bz_flat = _fld(_component_id(component), 'bz',
                   points.reshape(-1, 3).tolist(), use_gpu=use_gpu,
                   rank=rank, verbosity=verbosity)

    if rank > 0:
        return None

    return RZFieldGrid(bz=np.asarray(bz_flat, dtype=float).reshape(len(radii), n_ang),
                       angles=angles,
                       radii_mm=radii)


# ---------------------------------------------------------------------------
# Save functions (obtain the field via the getters, write via Field.save)
# ---------------------------------------------------------------------------
_OPAL_HEADER_KWARGS = dict(model="uCyclo_v2", version="Cyclotron Optimizer v0.1")


def save_median_plane_field(config: CyclotronConfig,
                            component: ComponentOrId,
                            output_path: Optional[str] = None,
                            rank: int = 0,
                            comm=None,
                            verbosity: int = 1) -> Optional[Field]:
    """Compute and save the median-plane field (Bz only for .comsol output)."""
    fe = config.field_evaluation
    field = get_median_plane_field(
        component,
        limit_mm=fe.median_plane_limit_mm,
        resolution_mm=fe.median_plane_resolution_mm,
        use_symmetry=fe.use_symmetry,
        rank=rank, comm=comm, verbosity=verbosity,
    )

    if rank <= 0:
        path = output_path or fe.median_plane_field_output or "output/midplane_field.comsol"
        if verbosity >= 1:
            print(f"Writing median-plane field to '{path}'...", flush=True)
        kwargs = {}
        if path.lower().endswith(".comsol"):
            kwargs = dict(components='z',
                          description="Magnetic flux density, z-component (median plane z=0)",
                          **_OPAL_HEADER_KWARGS)
        field.save(path, **kwargs)
        if verbosity >= 1:
            print("Done!", flush=True)

    return field


def save_bore_field(config: CyclotronConfig,
                    component: ComponentOrId,
                    output_path: Optional[str] = None,
                    rank: int = 0,
                    comm=None,
                    verbosity: int = 1) -> Optional[Field]:
    """Compute and save the 3D bore field (Bx, By, Bz)."""
    fe = config.field_evaluation
    xy_axis = symmetric_axis(fe.bore_xy_limit_mm, fe.bore_resolution_mm)
    n_z = int(round((fe.bore_z_max_mm - fe.bore_z_min_mm) / fe.bore_resolution_mm)) + 1
    z_axis = fe.bore_z_min_mm + np.arange(n_z) * fe.bore_resolution_mm

    if rank <= 0 and verbosity >= 1:
        print(f"Calculating 3D bore field ({len(xy_axis)}x{len(xy_axis)}x{n_z} points)...",
              flush=True)

    field = get_field_3d(component, xy_axis, xy_axis, z_axis,
                         use_symmetry=fe.use_symmetry,
                         rank=rank, comm=comm, verbosity=verbosity,
                         label="Radia B-field (bore)")

    if rank <= 0:
        path = output_path or fe.bore_field_output or "output/bore_field.comsol"
        if verbosity >= 1:
            print(f"Writing bore field to '{path}'...", flush=True)
        kwargs = {}
        if path.lower().endswith(".comsol"):
            kwargs = dict(description="Magnetic flux density components (bore)",
                          **_OPAL_HEADER_KWARGS)
        field.save(path, **kwargs)
        if verbosity >= 1:
            print("Done!", flush=True)

    return field


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class ReusableCyclotronSolver:
    """Stateful cyclotron solver that enables coil-current reuse.

    Builds the iron + coils once; ``resolve_at_current`` then rebuilds ONLY the coils
    (cheap, unmeshed) at a new current and re-relaxes WARM from the iron's retained
    magnetization, reusing the meshed iron. This makes the nested coil-current solve
    cheap. (Radia has no in-place coil-current setter, so each current still re-runs
    RlxPre; the saving is skipping the gmsh mesh + ObjPolyhdr rebuild, which dominates.)

    Disposal goes through the component wrappers: the throwaway top container is
    disposed shallow (keeping the iron sub-containers) and the coils deep; the iron and
    its magnetization persist across currents.
    """

    def __init__(self, config: CyclotronConfig, radii_mm, *, rank: int = 0,
                 comm=None, verbosity: int = 1):
        if isinstance(radii_mm, np.ndarray):
            radii_mm = radii_mm.tolist()
        if not isinstance(radii_mm, list):
            radii_mm = [radii_mm]
        self.config = config
        self.radii_mm = radii_mm
        self.rank = rank
        self.comm = comm
        self.verbosity = verbosity
        self._iron_subs = None   # kept across coil currents (with their magnetization)
        self._coils = None
        self._cyclotron = None
        self._im = None          # interaction matrix (rad.RlxPre handle)

    @property
    def cyclotron(self) -> Optional[BaseRadiaComponent]:
        """The assembled cyclotron component (iron + coils), or None before build()."""
        return self._cyclotron

    def build(self, pole_shape, coil_current):
        """Full (re)build from scratch (new shims), then solve."""
        rad.UtiDelAll()
        self._iron_subs = None
        self._coils = None
        self._cyclotron = None
        self._im = None
        self._iron_subs = build_iron(self.config, pole_shape, rank=self.rank,
                                     comm=self.comm, verbosity=self.verbosity)
        return self._solve_and_query(coil_current, zero_magnetization=False)

    def resolve_at_current(self, coil_current, *, warm=False):
        """Reuse the meshed iron; rebuild only the coils and re-solve at a new current.

        warm=False (default): zero the magnetization and relax COLD -- gives the SAME
        result as a full rebuild while skipping the (dominant) gmsh mesh + ObjPolyhdr
        rebuild. warm=True keeps the previous magnetization (faster in principle), but
        for large current jumps RlxAuto's per-iteration convergence criterion can trip
        during slow creep and return a wrong field -- only use with tightened precision.
        """
        if self._iron_subs is None:
            raise RuntimeError("ReusableCyclotronSolver.build() must be called first.")
        self._teardown_coils()
        return self._solve_and_query(coil_current, zero_magnetization=not warm)

    def dispose(self):
        """Free the per-current coils / top container / interaction matrix and drop refs.

        The meshed iron objects are intentionally left for the next ``rad.UtiDelAll()``
        (issued by ``build()``) rather than deep-disposed tet-by-tet here.
        """
        self._teardown_coils()
        self._iron_subs = None

    def _teardown_coils(self):
        # Dispose the top container (shallow -> keeps the iron sub-containers), the
        # coils (deep), and the interaction matrix. Iron + magnetization persist.
        if self._cyclotron is not None:
            self._cyclotron.dispose(deep=False)
            self._cyclotron = None
        if self._coils is not None:
            self._coils.dispose(deep=True)
            self._coils = None
        if self._im is not None:
            try:
                rad.UtiDel(self._im)
            except RuntimeError:
                pass  # already gone (e.g. a prior rad.UtiDelAll) -> idempotent
            self._im = None

    def _solve_and_query(self, coil_current, *, zero_magnetization):
        self.config.coil.current_A = coil_current
        self._coils = build_coils(self.config)
        self._cyclotron = BaseRadiaComponent.containerize([*self._iron_subs, self._coils])
        cid = self._cyclotron.id

        say = self.rank <= 0 and self.verbosity >= 1
        if say:
            print("Building Interaction Matrix...", flush=True)
            t0 = time.time()
        self._im = rad.RlxPre(cid)
        if say:
            print(f"Done! Assembling took {time.time() - t0} s.", flush=True)
            print("Solving...", flush=True)
            t0 = time.time()
        zerom = 'ZeroM->True' if zero_magnetization else 'ZeroM->False'
        result = rad.RlxAuto(self._im, self.config.simulation.precision,
                             self.config.simulation.iterations, 9, zerom, 'omega->0.3')
        if say:
            print(f"Done! Auto-Relaxation took {time.time() - t0} s", flush=True)
            print(f"target={self.config.simulation.precision}: "
                  f"iter={result[3]:.0f}, misfitM={result[0]:.6e}", flush=True)

        converged = (result[0] <= self.config.simulation.precision)
        misfit = float(result[0])

        fe = self.config.field_evaluation
        if fe.iso_method != "seo":
            bz_values = get_field_rz(
                self._cyclotron, self.radii_mm, fe.num_points_circle,
                use_symmetry=fe.use_symmetry,
                rank=self.rank, comm=self.comm, verbosity=self.verbosity)
        else:
            bz_values = get_median_plane_field(
                self._cyclotron,
                limit_mm=fe.median_plane_limit_mm,
                resolution_mm=fe.median_plane_resolution_mm,
                use_symmetry=fe.use_symmetry,
                rank=self.rank, comm=self.comm, verbosity=self.verbosity)

        return self.radii_mm, bz_values, converged, misfit


def evaluate_radii_parallel(config: CyclotronConfig,
                            pole_shape: PoleShape,
                            radii_mm: List[float],
                            rank: int = 0,
                            comm=None,
                            verbosity=1):
    """Build the cyclotron, relax, and evaluate the field for the isochronism method.

    Thin wrapper around ReusableCyclotronSolver.build() (single source of truth
    for the build -> RlxPre -> RlxAuto -> query sequence). All processes execute
    this collectively (Radia MPI handles parallelization); only rank 0 receives
    field results from Radia.

    :param config: CyclotronConfig object
    :param pole_shape: a PoleShape instance
    :param radii_mm: List of radii to evaluate (mm)
    :param rank: MPI rank (0 for sequential)
    :param comm: MPI communicator
    :param verbosity: 0 silent, 1 normal, 2 debug
    :return: Tuple (radii_mm, bz_values, converged, cyclotron, misfit).
        bz_values is an RZFieldGrid (circle/gordon) or a PyPATools Field (seo)
        on rank 0, None on other ranks; cyclotron is the assembled
        BaseRadiaComponent (use .id for the radia object id); misfit is the
        achieved relaxation misfit (valid on all ranks).
    """
    solver = ReusableCyclotronSolver(config, radii_mm, rank=rank, comm=comm,
                                     verbosity=verbosity)
    radii_out, bz_values, converged, misfit = solver.build(
        pole_shape, config.coil.current_A)
    return radii_out, bz_values, converged, solver.cyclotron, misfit
