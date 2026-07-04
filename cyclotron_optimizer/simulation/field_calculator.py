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

import os
import time
from typing import List, NamedTuple, Optional, Union

import numpy as np
import radia as rad

from cyclotron_optimizer.config_io.config import CyclotronConfig
from cyclotron_optimizer.geometry.components import BaseRadiaComponent
from cyclotron_optimizer.geometry.geometry import (assemble_iron, build_coils,
                                                   build_pole_part,
                                                   build_static_iron_parts)
from cyclotron_optimizer.geometry.pole_shape import PoleShape
from cyclotron_optimizer.geometry.symmetry import (
    SymmetryTuple,
    azimuthal_sector,
    canonical_symmetry_set,
    collect_field_symmetries,
    reduce_grid,
    symmetry_group,
)
from PyPATools.field import Field

ComponentOrId = Union[BaseRadiaComponent, int]


class GpuOptions(NamedTuple):
    """Granular GPU switches for the three radia stages.

    - assembly:   interaction-matrix assembly (rad.RlxPre(use_gpu=...));
                  under MPI, rank 0 assembles on the GPU while workers wait;
                  False = classic CPU assembly (MPI-distributed when radia
                  MPI is active -- the CPU-cluster mode).
    - relaxation: rad.RlxAuto method 9 (CUDA) vs 4 (CPU); always rank 0 only.
    - field:      rad.Fld(use_gpu=...) for the field evaluations.

    Anywhere a ``use_gpu`` argument is accepted, a plain bool (all three),
    a dict (e.g. {'assembly': False}), or a GpuOptions instance works.
    """
    assembly: bool = True
    relaxation: bool = True
    field: bool = True

    @classmethod
    def coerce(cls, value) -> "GpuOptions":
        if isinstance(value, GpuOptions):
            return value
        if isinstance(value, dict):
            return cls(**value)
        return cls(assembly=bool(value), relaxation=bool(value), field=bool(value))


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


# Points per rad.Fld call on the CPU path. CPU evaluations of big grids take
# minutes; chunking gives rank-0 progress output (and abort points) without
# measurable overhead -- each chunk is a fresh MPI work distribution. The GPU
# path stays unchunked (it batches internally and finishes in seconds).
CPU_FLD_CHUNK = 4000


def _fld_chunked(radia_id: int, components: str, points: np.ndarray,
                 *, use_gpu: bool = True, precision: str = "double",
                 rank: int = 0, verbosity: int = 1) -> Optional[np.ndarray]:
    """_fld over a (N, 3) point array, chunked on the CPU path with progress
    prints on rank 0. Returns a flat value array on rank 0, None elsewhere.

    MPI-collective: every rank walks the same chunk boundaries.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    n_pts = len(points)
    if use_gpu or n_pts <= CPU_FLD_CHUNK:
        result = _fld(radia_id, components, points.tolist(), use_gpu=use_gpu,
                      precision=precision, rank=rank, verbosity=verbosity)
        return None if rank > 0 else np.asarray(result, dtype=float).ravel()

    t0 = time.time()
    parts = []
    for start in range(0, n_pts, CPU_FLD_CHUNK):
        chunk = points[start:start + CPU_FLD_CHUNK]
        part = _fld(radia_id, components, chunk.tolist(), use_gpu=use_gpu,
                    precision=precision, rank=rank, verbosity=verbosity)
        if rank <= 0:
            parts.append(np.asarray(part, dtype=float).ravel())
            if verbosity >= 1:
                done = min(start + CPU_FLD_CHUNK, n_pts)
                rate = done / max(time.time() - t0, 1e-9)
                print(f"    ... {done}/{n_pts} points "
                      f"({rate:.0f} pts/s, ~{(n_pts - done) / rate:.0f} s left)",
                      flush=True)
    return None if rank > 0 else np.concatenate(parts)


def _rlx_pre(radia_id: int, *, srcobj: int = 0, use_gpu: bool = True) -> int:
    """rad.RlxPre with the RadiaCUDA use_gpu (GPU IM assembly) kwarg,
    falling back to plain radia builds without it.

    ``srcobj`` (an additional frozen external field source) enters radia as a
    per-element external-field vector evaluated on the CPU -- it composes with
    GPU assembly (the interaction matrix itself is unaffected).
    """
    try:
        return rad.RlxPre(radia_id, srcobj=srcobj, use_gpu=use_gpu)
    except TypeError:
        return rad.RlxPre(radia_id, srcobj) if srcobj else rad.RlxPre(radia_id)


def _magnetizations(radia_id: int) -> np.ndarray:
    """(N, 3) magnetization vectors of an object's relaxable elements."""
    data = rad.ObjM(radia_id)
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 2:  # single element: [point, M]
        arr = arr[None, :, :]
    return arr[:, 1, :]


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
            t0 = time.time()
            # Component id 'b' (NOT 'bxbybz'): RadiaCUDA's GPU gate in
            # RadFld only accepts 'b'/'bx'/'by'/'bz' -- 'bxbybz' silently
            # falls through to the (very slow) CPU path.
            b_eval = _fld_chunked(group.radia_id, 'b',
                                  reduction.eval_points, use_gpu=use_gpu,
                                  precision=gpu_precision, rank=rank,
                                  verbosity=verbosity)
            if say:
                print(f"    rad.Fld: {len(reduction.eval_points)} points in "
                      f"{time.time() - t0:.3f} s", flush=True)
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
    t0 = time.time()
    axis = symmetric_axis(limit_mm, resolution_mm)
    field = get_field_2d(component, axis, axis, 0.0,
                         use_symmetry=use_symmetry, rank=rank, comm=comm,
                         verbosity=verbosity, use_gpu=use_gpu,
                         gpu_precision=gpu_precision,
                         label="Radia B-field (median plane)")
    if rank <= 0 and verbosity >= 1:
        print(f"Done! ({time.time() - t0:.3f} s)", flush=True)
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

    t0 = time.time()
    bz_flat = _fld_chunked(_component_id(component), 'bz',
                           points.reshape(-1, 3), use_gpu=use_gpu,
                           rank=rank, verbosity=verbosity)
    if rank <= 0 and verbosity >= 1:
        print(f"  rad.Fld: {len(radii) * n_ang} points in {time.time() - t0:.3f} s",
              flush=True)

    if rank > 0:
        return None

    return RZFieldGrid(bz=bz_flat.reshape(len(radii), n_ang),
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
                            verbosity: int = 1,
                            use_gpu: bool = True) -> Optional[Field]:
    """Compute and save the median-plane field (Bz only for .comsol output)."""
    fe = config.field_evaluation
    field = get_median_plane_field(
        component,
        limit_mm=fe.median_plane_limit_mm,
        resolution_mm=fe.median_plane_resolution_mm,
        use_symmetry=fe.use_symmetry, use_gpu=use_gpu,
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
                    verbosity: int = 1,
                    use_gpu: bool = True) -> Optional[Field]:
    """Compute and save the 3D bore field (Bx, By, Bz)."""
    fe = config.field_evaluation
    xy_axis = symmetric_axis(fe.bore_xy_limit_mm, fe.bore_resolution_mm)
    n_z = int(round((fe.bore_z_max_mm - fe.bore_z_min_mm) / fe.bore_resolution_mm)) + 1
    z_axis = fe.bore_z_min_mm + np.arange(n_z) * fe.bore_resolution_mm

    if rank <= 0 and verbosity >= 1:
        print(f"Calculating 3D bore field ({len(xy_axis)}x{len(xy_axis)}x{n_z} points)...",
              flush=True)

    field = get_field_3d(component, xy_axis, xy_axis, z_axis,
                         use_symmetry=fe.use_symmetry, use_gpu=use_gpu,
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
    """Stateful cyclotron solver with two levels of geometry reuse.

    - Across COIL CURRENTS (``resolve_at_current``): rebuilds only the coils
      (cheap, unmeshed); the meshed iron persists. Radia has no in-place
      coil-current setter, so each current still re-runs RlxPre; the saving is
      skipping the gmsh mesh + ObjPolyhdr rebuild, which dominates.
    - Across SHIM SHAPES (``build`` with reuse_static_iron=True, the default):
      the yoke, lids, and extraction channel are built ONCE; only the pole
      (shim-dependent) and the coils are rebuilt, and the iron container is
      re-assembled + re-symmetrized. The relaxation then starts from zero
      magnetization (ZeroM->True) so the result is identical to a full
      rebuild -- only mesh/build time is saved, not relaxation iterations.

    PERTURBATIVE COMPONENTS (ComponentSpec.perturbative, e.g. the extraction
    channel, whose asymmetry intentionally breaks the machine symmetry): they
    are EXCLUDED from the main relaxation and solved in stages:

      stage 0  main solve: RlxPre/RlxAuto on the symmetric iron + coils only
               (the perturbative parts are invisible -- the main iron is not
               constrained by them, so the TrfZer symmetry stays consistent).
      stage 1  frozen-background solve: RlxPre(perturb, srcobj=main) +
               RlxAuto -- the perturbative parts relax in the frozen field of
               the solved machine. First-order exact: their own magnetization
               and direct field are correct in the true symmetric background.
      stage 2  (optional, config.simulation.perturb_iterations > 0) up to N
               main <-> perturbative back-reaction cycles via rad.RlxUpdSrc
               (no interaction-matrix rebuilds), stopping early when the
               perturbative magnetization changes less than perturb_tol [T].

    SEMANTICS CAVEAT (stage 2): the main solve remains symmetry-constrained
    during iteration, so the recovered back-reaction is (approximately) the
    SYMMETRIZED part of the true back-reaction -- the isochronism-relevant
    azimuthal average -- while re-introducing phantom ripple of the
    perturbative field at all image azimuths. Stage-1 only (the default) is
    the clean choice: exact symmetric machine + exact first-order channel.

    Field evaluation is unchanged either way: perturbative components stay
    top-level children of the cyclotron container with their own (usually
    empty) field symmetry, handled by the per-group folding.

    Disposal goes through the component wrappers: throwaway containers are
    disposed shallow (members survive with parent pointers reset, ready for
    re-containerizing), replaced parts (coils, old pole) deep.
    """

    def __init__(self, config: CyclotronConfig, radii_mm, *, rank: int = 0,
                 comm=None, verbosity: int = 1, use_gpu=True,
                 perturb_iterations: Optional[int] = None,
                 perturb_tol: Optional[float] = None):
        if isinstance(radii_mm, np.ndarray):
            radii_mm = radii_mm.tolist()
        if not isinstance(radii_mm, list):
            radii_mm = [radii_mm]
        self.config = config
        self.radii_mm = radii_mm
        self.rank = rank
        self.comm = comm
        self.verbosity = verbosity
        # bool | dict | GpuOptions: per-stage GPU switches (assembly /
        # relaxation / field evaluation)
        self.gpu = GpuOptions.coerce(use_gpu)
        # Perturbative-stage controls (None -> config.simulation values)
        sim = config.simulation
        self.perturb_iterations = (getattr(sim, "perturb_iterations", 0)
                                   if perturb_iterations is None
                                   else int(perturb_iterations))
        self.perturb_tol = (getattr(sim, "perturb_tol", 0.0)
                            if perturb_tol is None else float(perturb_tol))
        self._static_parts = None  # yoke/lids/channel, kept across pole rebuilds
        self._pole = None
        self._iron_subs = None   # ALL per-assembly group containers (disposal)
        self._main_subs = None   # non-perturbative groups (main relax)
        self._perturb_subs = None  # perturbative groups (staged relax)
        self._coils = None
        self._main_cnt = None    # main iron + coils (relax target / srcobj)
        self._perturb_cnt = None  # extra wrapper when >1 perturbative group
        self._cyclotron = None
        self._im = None          # main interaction matrix (rad.RlxPre handle)
        self._im_p = None        # perturbative-stage interaction matrix

    @property
    def use_gpu(self) -> bool:
        """Back-compat convenience: the FIELD-evaluation GPU switch."""
        return self.gpu.field

    @property
    def cyclotron(self) -> Optional[BaseRadiaComponent]:
        """The assembled cyclotron component (iron + coils), or None before build()."""
        return self._cyclotron

    def build(self, pole_shape, coil_current, *, query=True,
              reuse_static_iron=True):
        """(Re)build for new shims, then solve.

        On the first call (or with reuse_static_iron=False) everything is
        built from scratch. On subsequent calls the static components
        (yoke/lids/channel) are REUSED and only the shimmed pole is rebuilt
        (gmsh mesh + ObjPolyhdr of the static parts skipped); the relaxation
        then zeroes the magnetization, so the result is identical to a full
        rebuild.

        query=False skips the post-relaxation field evaluation (bz_values in
        the returned tuple is None) -- used by the Session/Model facade, which
        queries fields on demand instead.
        """
        say = self.rank <= 0 and self.verbosity >= 1

        if self._static_parts is None or not reuse_static_iron:
            rad.UtiDelAll()
            self._static_parts = None
            self._pole = None
            self._iron_subs = None
            self._main_subs = None
            self._perturb_subs = None
            self._coils = None
            self._main_cnt = None
            self._perturb_cnt = None
            self._cyclotron = None
            self._im = None
            self._im_p = None
            self._static_parts = build_static_iron_parts(
                self.config, rank=self.rank, comm=self.comm,
                verbosity=self.verbosity)
            zero_magnetization = False  # fresh objects start unmagnetized
        else:
            if say:
                print("Reusing static iron components; rebuilding pole...",
                      flush=True)
            # Tear down everything referencing the old pole: top container +
            # coils + interaction matrix, then every per-assembly iron
            # container (shallow: the static members survive with parent
            # pointers reset), then the old pole itself (deep: tets freed).
            self._teardown_coils()
            for container in (self._iron_subs or []):
                container.dispose(deep=False)
            self._iron_subs = None
            self._main_subs = None
            self._perturb_subs = None
            if self._pole is not None:
                self._pole[1].dispose(deep=True)
                self._pole = None
            # The retained iron magnetization belongs to the OLD pole shape;
            # zero it so the relaxed result matches a from-scratch build.
            zero_magnetization = True

        self._pole = build_pole_part(
            self.config, pole_shape, comm=self.comm,
            materials=self._static_parts["materials"])
        self._main_subs, self._perturb_subs = assemble_iron(
            self.config, self._static_parts, self._pole,
            rank=self.rank, verbosity=self.verbosity,
            split_perturbative=True)
        self._iron_subs = [*self._main_subs, *self._perturb_subs]
        return self._solve_and_query(coil_current,
                                     zero_magnetization=zero_magnetization,
                                     query=query)

    def resolve_at_current(self, coil_current, *, warm=False, query=True):
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
        return self._solve_and_query(coil_current, zero_magnetization=not warm,
                                     query=query)

    def dispose(self):
        """Free the per-current coils / top container / interaction matrix and drop refs.

        The meshed iron objects are intentionally left for the next ``rad.UtiDelAll()``
        (issued by a from-scratch ``build()``) rather than deep-disposed tet-by-tet here.
        """
        self._teardown_coils()
        self._iron_subs = None
        self._main_subs = None
        self._perturb_subs = None
        self._static_parts = None
        self._pole = None

    def _teardown_coils(self):
        # Dispose the top container(s) (shallow -> keeps the iron
        # sub-containers), the coils (deep), and the interaction matrices.
        # Iron + magnetization persist.
        if self._cyclotron is not None:
            self._cyclotron.dispose(deep=False)
            self._cyclotron = None
        if self._main_cnt is not None:
            self._main_cnt.dispose(deep=False)
            self._main_cnt = None
        if self._perturb_cnt is not None:
            # Extra wrapper only exists when there are >1 perturbative groups;
            # the group containers themselves are disposed via _iron_subs.
            if self._perturb_cnt not in (self._perturb_subs or []):
                self._perturb_cnt.dispose(deep=False)
            self._perturb_cnt = None
        if self._coils is not None:
            self._coils.dispose(deep=True)
            self._coils = None
        for attr in ("_im", "_im_p"):
            handle = getattr(self, attr)
            if handle is not None:
                try:
                    rad.UtiDel(handle)
                except RuntimeError:
                    pass  # already gone (e.g. a prior rad.UtiDelAll) -> idempotent
                setattr(self, attr, None)

    def _relax(self, im: int, *, zero_magnetization: bool, label: str = ""):
        """RlxAuto on an interaction matrix; returns (misfit, iterations)."""
        say = self.rank <= 0 and self.verbosity >= 1
        if say:
            t0 = time.time()

        # config.simulation.anderson drives RadiaCUDA's Anderson switch (the
        # C++ reads the RADIA_ANDERSON / RADIA_NO_ANDERSON environment
        # variables on every RlxAuto call); None leaves the environment /
        # built-in default untouched.
        anderson = getattr(self.config.simulation, "anderson", None)
        if anderson is not None:
            os.environ.pop("RADIA_ANDERSON", None)
            os.environ.pop("RADIA_NO_ANDERSON", None)
            os.environ["RADIA_ANDERSON" if anderson else "RADIA_NO_ANDERSON"] = "1"

        zerom = 'ZeroM->True' if zero_magnetization else 'ZeroM->False'
        # Auto: 9 = adaptive under-relaxed Jacobi on the GPU, 10 = the same
        # solver on the CPU (RadiaCUDA). Historically NOT method 4 because its
        # Gauss-Seidel misfit limit-cycles above target on the production muon
        # model -- but on coarse meshes the OPPOSITE failure was measured
        # (methods 9/10 creep: |delta M| < target while the magnetization is
        # ~1e-2 away from the true fixed point; method 4 satisfies the M-vs-H
        # residual to ~1e-4). config.simulation.relax_method overrides.
        relax_method = getattr(self.config.simulation, "relax_method", None)
        if relax_method is None:
            relax_method = 9 if self.gpu.relaxation else 10
        result = rad.RlxAuto(im, self.config.simulation.precision,
                             self.config.simulation.iterations, relax_method,
                             zerom, 'omega->0.3')
        if say:
            print(f"Done! {label or 'Auto-Relaxation'} took "
                  f"{time.time() - t0} s", flush=True)
            print(f"target={self.config.simulation.precision}: "
                  f"iter={result[3]:.0f}, misfitM={result[0]:.6e}", flush=True)
        return float(result[0]), int(result[3])

    def _solve_and_query(self, coil_current, *, zero_magnetization, query=True):
        self.config.coil.current_A = coil_current
        self._coils = build_coils(self.config)
        say = self.rank <= 0 and self.verbosity >= 1

        perturb = self._perturb_subs or []
        if perturb:
            # Main relax target excludes the perturbative parts entirely; they
            # remain top-level field sources of the full cyclotron container.
            self._main_cnt = BaseRadiaComponent.containerize(
                [*self._main_subs, self._coils])
            self._perturb_cnt = (perturb[0] if len(perturb) == 1
                                 else BaseRadiaComponent.containerize(perturb))
            self._cyclotron = BaseRadiaComponent.containerize(
                [self._main_cnt, self._perturb_cnt])
        else:
            self._cyclotron = BaseRadiaComponent.containerize(
                [*self._iron_subs, self._coils])
            self._main_cnt = None
            self._perturb_cnt = None
        relax_target = self._main_cnt if perturb else self._cyclotron

        if say:
            print(f"Building Interaction Matrix "
                  f"({'GPU' if self.gpu.assembly else 'CPU'} assembly)...", flush=True)
            t0 = time.time()
        # Stage 0: the main solve NEVER sees the perturbative parts (no
        # srcobj here: on reuse they still carry the previous solution's
        # magnetization until stage 1 re-relaxes them).
        self._im = _rlx_pre(relax_target.id, use_gpu=self.gpu.assembly)
        if say:
            print(f"Done! Assembling took {time.time() - t0} s.", flush=True)
            print(f"Solving ({'GPU' if self.gpu.relaxation else 'CPU'} relaxation)...",
                  flush=True)
        misfit, _iters = self._relax(self._im,
                                     zero_magnetization=zero_magnetization,
                                     label="Main relaxation")
        converged = misfit <= self.config.simulation.precision

        if perturb:
            p_misfit, p_conv = self._solve_perturbative(
                zero_magnetization=zero_magnetization)
            misfit = max(misfit, p_misfit)
            converged = converged and p_conv

        if not query:
            return self.radii_mm, None, converged, misfit
        return self._query(converged, misfit)

    def _solve_perturbative(self, *, zero_magnetization: bool):
        """Stage 1 (frozen-background solve) + optional stage-2 iteration.

        Returns (worst misfit across stages, converged). See the class
        docstring for the stage semantics and the symmetry caveat.
        """
        say = self.rank <= 0 and self.verbosity >= 1
        prec = self.config.simulation.precision

        # Stage 1: perturbative parts relax in the frozen machine field
        # (srcobj = main iron + coils). The perturbative IM is tiny.
        if say:
            print("Perturbative stage 1: frozen-background solve "
                  f"({'GPU' if self.gpu.assembly else 'CPU'} assembly)...",
                  flush=True)
        self._im_p = _rlx_pre(self._perturb_cnt.id, srcobj=self._main_cnt.id,
                              use_gpu=self.gpu.assembly)
        misfit, _ = self._relax(self._im_p,
                                zero_magnetization=zero_magnetization,
                                label="Perturbative relaxation")
        worst = misfit

        # Stage 2 (optional): main <-> perturbative back-reaction cycles.
        # The main IM is REBUILT once with the perturbative parts as srcobj
        # (their magnetization is now valid); subsequent cycles refresh the
        # external-field data in place via rad.RlxUpdSrc -- no IM rebuilds.
        n_iter = int(self.perturb_iterations or 0)
        if n_iter > 0:
            try:
                rad.UtiDel(self._im)
            except RuntimeError:
                pass
            self._im = _rlx_pre(self._main_cnt.id,
                                srcobj=self._perturb_cnt.id,
                                use_gpu=self.gpu.assembly)
        for cycle in range(1, n_iter + 1):
            m_prev = None
            if self.rank <= 0:
                m_prev = _magnetizations(self._perturb_cnt.id)
            if say:
                print(f"Perturbative stage 2, cycle {cycle}/{n_iter}...",
                      flush=True)
            rad.RlxUpdSrc(self._im)
            m_misfit, _ = self._relax(self._im, zero_magnetization=False,
                                      label="Main re-relaxation")
            rad.RlxUpdSrc(self._im_p)
            p_misfit, _ = self._relax(self._im_p, zero_magnetization=False,
                                      label="Perturbative re-relaxation")
            worst = max(m_misfit, p_misfit)

            delta = None
            if self.rank <= 0:
                m_now = _magnetizations(self._perturb_cnt.id)
                delta = float(np.max(np.linalg.norm(m_now - m_prev, axis=1)))
            if self.comm is not None:
                delta = self.comm.bcast(delta, root=0)
            if say:
                print(f"  perturbative max |delta M| = {delta:.3e} T", flush=True)
            if self.perturb_tol and delta is not None and delta < self.perturb_tol:
                if say:
                    print(f"  converged below perturb_tol="
                          f"{self.perturb_tol:g} T after {cycle} cycle(s)",
                          flush=True)
                break

        return worst, worst <= prec

    def _query(self, converged, misfit):
        fe = self.config.field_evaluation
        if fe.iso_method != "seo":
            bz_values = get_field_rz(
                self._cyclotron, self.radii_mm, fe.num_points_circle,
                use_symmetry=fe.use_symmetry, use_gpu=self.gpu.field,
                rank=self.rank, comm=self.comm, verbosity=self.verbosity)
        else:
            bz_values = get_median_plane_field(
                self._cyclotron,
                limit_mm=fe.median_plane_limit_mm,
                resolution_mm=fe.median_plane_resolution_mm,
                use_symmetry=fe.use_symmetry, use_gpu=self.gpu.field,
                rank=self.rank, comm=self.comm, verbosity=self.verbosity)

        return self.radii_mm, bz_values, converged, misfit


def evaluate_radii_parallel(config: CyclotronConfig,
                            pole_shape: PoleShape,
                            radii_mm: List[float],
                            rank: int = 0,
                            comm=None,
                            verbosity=1,
                            use_gpu: bool = True):
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
                                     verbosity=verbosity, use_gpu=use_gpu)
    radii_out, bz_values, converged, misfit = solver.build(
        pole_shape, config.coil.current_A)
    return radii_out, bz_values, converged, solver.cyclotron, misfit
