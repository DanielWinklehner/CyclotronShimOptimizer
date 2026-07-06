"""Session / CyclotronModel: the high-level scripting facade.

A ``Session`` owns the runtime lifecycle (config, MPI rank/communicator,
radia state); a ``CyclotronModel`` wraps a built cyclotron (via
ReusableCyclotronSolver) and exposes solve / field / isochronism / export /
viewer operations with config-driven defaults.

MPI contract: every Session/Model method is COLLECTIVE (all ranks must call
it, in the same order -- just write the script top-to-bottom as if it were
single-process) and data is returned on rank 0 only (None elsewhere).
Guard only I/O and plotting with ``session.is_root``. Under a plain
``python script.py`` (or ``use_mpi=False``) everything runs single-process.

Typical script:

    import cyclotron_optimizer as co

    with co.Session("machine_muon.yml") as s:
        model = s.build()
        model.solve()
        iso = model.isochronism()
        fmap = model.median_plane_field(resolution_mm=2.0, gpu_precision="single")
        if s.is_root:
            print(f"mean f = {iso['mean_freq_mhz']:.4f} MHz")
            fmap.save("output/midplane.comsol")
            model.show(field=fmap)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import radia as rad

from cyclotron_optimizer.config_io.config import CyclotronConfig
from cyclotron_optimizer.core.isochronicity import compute_isochronism
from cyclotron_optimizer.core.species import IonSpecies
from cyclotron_optimizer.geometry.pole_shape import PoleShape
from cyclotron_optimizer.simulation.field_calculator import (
    GpuOptions,
    ReusableCyclotronSolver,
    get_field_3d,
    get_field_rz,
    get_median_plane_field,
    symmetric_axis,
)

ConfigLike = Union[CyclotronConfig, str, Path, None]


class Session:
    """Runtime context for cyclotron studies: config + MPI + radia lifecycle.

    :param config: path to a machine YAML, a CyclotronConfig, or None (some
        operations then need explicit parameters).
    :param verbosity: 0 silent, 1 normal, 2 debug (rank 0 prints only).
    :param use_mpi: 'auto' (default) initializes mpi4py when available and
        falls back to single-process; True requires it; False forces
        single-process (also handy in notebooks and tests).
    """

    def __init__(self, config: ConfigLike = None, *, verbosity: int = 1,
                 use_mpi: Union[str, bool] = "auto"):
        if isinstance(config, (str, Path)):
            config = CyclotronConfig.from_yaml(str(config))
        self.config = config
        self.verbosity = verbosity

        self.comm = None
        self.rank = 0
        if use_mpi is True or use_mpi == "auto":
            try:
                from mpi4py import MPI  # radia already imported by the package
                self.comm = MPI.COMM_WORLD
                self.rank = self.comm.Get_rank()
            except Exception:
                if use_mpi is True:
                    raise
            else:
                # CRITICAL: tell radia about the (mpi4py-initialized) MPI
                # environment. Radia's interaction matrix / relaxation
                # distribute work via its INTERNAL rank/size, which only
                # UtiMPI sets ('in' = adopt an already-initialized MPI).
                # Without this every rank runs the FULL relaxation and they
                # all contend for the GPU: mpiexec -n N becomes N x SLOWER.
                try:
                    rad.UtiMPI('in')
                except Exception:
                    pass  # plain radia builds without the MPI hook

    # ------------------------------------------------------------------
    # Context / MPI helpers
    # ------------------------------------------------------------------
    @property
    def is_root(self) -> bool:
        return self.rank <= 0

    def barrier(self) -> None:
        if self.comm is not None:
            self.comm.Barrier()

    def bcast(self, value, root: int = 0):
        """Broadcast a picklable value from root to all ranks."""
        if self.comm is not None:
            return self.comm.bcast(value, root=root)
        return value

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # On an exception, do NOT synchronize: the other ranks are mid-collective,
        # so a barrier here deadlocks BEFORE python can even print the traceback
        # (observed as a silent hang under mpiexec). Exit dirty instead -- the
        # traceback prints and mpiexec tears the other ranks down.
        self.close(barrier=exc_type is None)

    def close(self, barrier: bool = True) -> None:
        if not barrier:
            return
        self.barrier()
        if self.comm is not None:
            try:
                rad.UtiMPI('off')
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Config-driven defaults
    # ------------------------------------------------------------------
    def _require_config(self) -> CyclotronConfig:
        if self.config is None:
            raise ValueError("This operation needs a config (Session(config=...)).")
        return self.config

    def default_radii_mm(self) -> np.ndarray:
        fe = self._require_config().field_evaluation
        return np.linspace(fe.radius_min_mm, fe.radius_max_mm, fe.n_eval_pts)

    def default_pole_shape(self) -> PoleShape:
        """PoleShape from the config's shim settings (side and top handled
        independently -- see PoleShape.from_shim_configs)."""
        config = self._require_config()
        return PoleShape.from_shim_configs(config.side_shim.num_rad_segments,
                                           config.side_shim, config.top_shim)

    # ------------------------------------------------------------------
    # Model construction / geometry-only workflows
    # ------------------------------------------------------------------
    def build(self, pole_shape: Optional[PoleShape] = None,
              coil_current: Optional[float] = None,
              radii_mm=None,
              use_gpu=True) -> "CyclotronModel":
        """Create a CyclotronModel (no radia work yet -- call model.solve()).

        :param pole_shape: shim shape; defaults to the config's shims. The
            config value is the initial/default shape -- optimizers pass
            their own here.
        :param coil_current: default coil current for solve(); defaults to
            config.coil.current_A.
        :param radii_mm: default isochronism radii; defaults to the config's
            field_evaluation range.
        :param use_gpu: bool for all stages, or a GpuOptions / dict for
            per-stage control, e.g. {'assembly': True, 'relaxation': True,
            'field': False}. Covers interaction-matrix assembly, relaxation,
            and field evaluation; the field switch is the model-wide default
            that field methods can override per call.
        """
        config = self._require_config()
        if pole_shape is None:
            pole_shape = self.default_pole_shape()
        if radii_mm is None:
            radii_mm = self.default_radii_mm()
        if coil_current is None:
            coil_current = config.coil.current_A

        solver = ReusableCyclotronSolver(
            config, np.asarray(radii_mm, dtype=float).tolist(),
            rank=self.rank, comm=self.comm, verbosity=self.verbosity,
            use_gpu=use_gpu)
        return CyclotronModel(self, solver, pole_shape, coil_current)

    def view_geometry(self, pole_shape: Optional[PoleShape] = None,
                      show_edges: bool = True) -> None:
        """Build the full (symmetry-expanded, unsolved) model and show it.

        Collective: builds on all ranks (rank-0 meshing + broadcast); the
        viewer opens on rank 0 only. Replaces the old ``--geo_test`` flag.
        """
        from cyclotron_optimizer.geometry.geometry import build_geometry

        config = self._require_config()
        if pole_shape is None:
            pole_shape = self.default_pole_shape()

        rad.UtiDelAll()
        display = build_geometry(config, pole_shape, rank=self.rank,
                                 comm=self.comm, omit_symmetry=True,
                                 verbosity=self.verbosity)
        if self.is_root:
            from PyRadia import ObjDrwPyVista
            ObjDrwPyVista(display.id, show_edges=show_edges)
        self.barrier()


class CyclotronModel:
    """A (re)solvable cyclotron: geometry + relaxation state + field queries.

    Created via ``Session.build()``. All methods are MPI-collective; field
    and isochronism results are returned on rank 0 only.
    """

    def __init__(self, session: Session, solver: ReusableCyclotronSolver,
                 pole_shape: Optional[PoleShape], coil_current: float):
        self._session = session
        self._solver = solver
        self._pole_shape = pole_shape
        self._coil_current = float(coil_current)
        self.converged: Optional[bool] = None
        self.misfit: Optional[float] = None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def session(self) -> Session:
        return self._session

    @property
    def config(self) -> CyclotronConfig:
        return self._solver.config

    @property
    def pole_shape(self) -> Optional[PoleShape]:
        return self._pole_shape

    @property
    def cyclotron(self):
        """The assembled geometry component (None before solve())."""
        return self._solver.cyclotron

    @property
    def radii_mm(self) -> np.ndarray:
        return np.asarray(self._solver.radii_mm, dtype=float)

    def _require_solved(self):
        if self._solver.cyclotron is None:
            raise RuntimeError("Model not solved yet -- call model.solve() first.")
        return self._solver.cyclotron

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(self, coil_current: Optional[float] = None, *,
              perturb_iterations: Optional[int] = None,
              perturb_tol: Optional[float] = None) -> "CyclotronModel":
        """Build the geometry (mesh + assemble) and relax the magnetization.

        Components flagged ``perturbative: True`` in the config (e.g. the
        extraction channel) are excluded from the main relaxation and solved
        in a frozen-background stage afterwards, optionally iterating the
        main <-> perturbative back-reaction (see ReusableCyclotronSolver).

        :param perturb_iterations: override config.simulation.perturb_iterations
            (0 = stage-1 only, N > 0 = up to N back-reaction cycles). NOTE:
            during iteration the main solve stays symmetry-constrained, so
            stage 2 recovers only the SYMMETRIZED part of the back-reaction
            and re-introduces ripple at the symmetry-image azimuths.
        :param perturb_tol: override config.simulation.perturb_tol -- early
            exit when the perturbative magnetization changes < tol [T].
        """
        if coil_current is not None:
            self._coil_current = float(coil_current)
        if perturb_iterations is not None:
            self._solver.perturb_iterations = int(perturb_iterations)
        if perturb_tol is not None:
            self._solver.perturb_tol = float(perturb_tol)
        _, _, self.converged, self.misfit = self._solver.build(
            self._pole_shape, self._coil_current, query=False)
        return self

    def resolve_at_current(self, coil_current: float, *,
                           warm: bool = False) -> "CyclotronModel":
        """Re-solve at a new coil current, reusing the meshed iron (cheap)."""
        self._coil_current = float(coil_current)
        _, _, self.converged, self.misfit = self._solver.resolve_at_current(
            coil_current, warm=warm, query=False)
        return self

    # ------------------------------------------------------------------
    # Field queries (collective; data on rank 0 only)
    # ------------------------------------------------------------------
    def _use_gpu(self, override: Optional[bool]) -> bool:
        return self._solver.use_gpu if override is None else override

    def field_rz(self, radii_mm=None, num_angles: Optional[int] = None,
                 use_symmetry: Optional[bool] = None,
                 use_gpu: Optional[bool] = None):
        """Bz on midplane circles (RZFieldGrid with its azimuthal angles)."""
        cyclotron = self._require_solved()
        fe = self.config.field_evaluation
        s = self._session
        return get_field_rz(
            cyclotron,
            self.radii_mm if radii_mm is None else radii_mm,
            fe.num_points_circle if num_angles is None else num_angles,
            use_symmetry=fe.use_symmetry if use_symmetry is None else use_symmetry,
            use_gpu=self._use_gpu(use_gpu),
            rank=s.rank, comm=s.comm, verbosity=s.verbosity)

    def median_plane_field(self, limit_mm: Optional[float] = None,
                           resolution_mm: Optional[float] = None,
                           use_symmetry: Optional[bool] = None,
                           use_gpu: Optional[bool] = None,
                           gpu_precision: str = "double"):
        """Median-plane B-field as a PyPATools Field (meters / Tesla).

        gpu_precision='single' uses the fp32 GPU kernel: much faster,
        visualization-grade only -- keep 'double' for tracking maps.
        """
        cyclotron = self._require_solved()
        fe = self.config.field_evaluation
        s = self._session
        return get_median_plane_field(
            cyclotron,
            limit_mm=fe.median_plane_limit_mm if limit_mm is None else limit_mm,
            resolution_mm=(fe.median_plane_resolution_mm
                           if resolution_mm is None else resolution_mm),
            use_symmetry=fe.use_symmetry if use_symmetry is None else use_symmetry,
            use_gpu=self._use_gpu(use_gpu),
            gpu_precision=gpu_precision,
            rank=s.rank, comm=s.comm, verbosity=s.verbosity)

    def field_3d(self, x_mm=None, y_mm=None, z_mm=None,
                 use_symmetry: Optional[bool] = None,
                 use_gpu: Optional[bool] = None,
                 gpu_precision: str = "double"):
        """3D B-field on a regular grid; axes default to the config's bore box."""
        cyclotron = self._require_solved()
        fe = self.config.field_evaluation
        s = self._session
        if x_mm is None or y_mm is None:
            xy = symmetric_axis(fe.bore_xy_limit_mm, fe.bore_resolution_mm)
            x_mm = xy if x_mm is None else x_mm
            y_mm = xy if y_mm is None else y_mm
        if z_mm is None:
            n_z = int(round((fe.bore_z_max_mm - fe.bore_z_min_mm)
                            / fe.bore_resolution_mm)) + 1
            z_mm = fe.bore_z_min_mm + np.arange(n_z) * fe.bore_resolution_mm
        return get_field_3d(
            cyclotron, x_mm, y_mm, z_mm,
            use_symmetry=fe.use_symmetry if use_symmetry is None else use_symmetry,
            use_gpu=self._use_gpu(use_gpu),
            gpu_precision=gpu_precision,
            rank=s.rank, comm=s.comm, verbosity=s.verbosity)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def isochronism(self, radii_mm=None, method: Optional[str] = None,
                    seo_solver: str = "newton", **kwargs):
        """Isochronism analysis (circle / gordon / seo) at the given radii.

        Collective (performs the field query); the iso dict is returned on
        rank 0, None elsewhere. Extra kwargs go to compute_isochronism.
        """
        config = self.config
        s = self._session
        method = (method or config.field_evaluation.iso_method or "circle").lower()
        radii = self.radii_mm if radii_mm is None else np.asarray(radii_mm, float)

        if method == "seo":
            field = self.median_plane_field()
        else:
            field = self.field_rz(radii_mm=radii)

        if not s.is_root or field is None:
            return None

        species = IonSpecies(config.particle_species)
        return compute_isochronism(
            method, field, radii, config, species,
            solver=seo_solver, rank=s.rank, comm=s.comm,
            verbose=(s.verbosity >= 1), **kwargs)

    # ------------------------------------------------------------------
    # Export / viewer
    # ------------------------------------------------------------------
    def save_median_plane_field(self, path: Optional[str] = None, **kwargs):
        from cyclotron_optimizer.simulation.field_calculator import \
            save_median_plane_field
        s = self._session
        kwargs.setdefault("use_gpu", self._solver.use_gpu)
        return save_median_plane_field(self.config, self._require_solved(),
                                       output_path=path, rank=s.rank,
                                       comm=s.comm, verbosity=s.verbosity, **kwargs)

    def save_bore_field(self, path: Optional[str] = None, **kwargs):
        from cyclotron_optimizer.simulation.field_calculator import save_bore_field
        s = self._session
        kwargs.setdefault("use_gpu", self._solver.use_gpu)
        return save_bore_field(self.config, self._require_solved(),
                               output_path=path, rank=s.rank,
                               comm=s.comm, verbosity=s.verbosity, **kwargs)

    def show(self, field=None, show_edges: bool = True, **viewer_kwargs) -> None:
        """Show the model in the PyVista viewer, optionally with a field overlay.

        Rebuilds a full (symmetry-expanded) display copy of the geometry for
        better visibility WITHOUT touching the solved model (the display copy
        is disposed afterwards), so fields can still be queried after show().
        Collective; the window opens on rank 0 and blocks until closed.

        :param field: a 2D median-plane Field to overlay (e.g. from
            median_plane_field()); None shows the bare model.
        """
        from cyclotron_optimizer.geometry.geometry import build_geometry

        s = self._session
        display = build_geometry(self.config, self._pole_shape, rank=s.rank,
                                 comm=s.comm, omit_symmetry=True,
                                 verbosity=s.verbosity)
        if s.is_root:
            if field is not None:
                from cyclotron_optimizer.visualization.field_maps import \
                    show_model_with_median_plane_field
                show_model_with_median_plane_field(display.id, field,
                                                   show_edges=show_edges,
                                                   **viewer_kwargs)
            else:
                from PyRadia import ObjDrwPyVista
                ObjDrwPyVista(display.id, show_edges=show_edges)
        s.barrier()
        display.dispose(deep=True)

    def dispose(self) -> None:
        """Free the per-current radia objects (iron is left for the next build)."""
        self._solver.dispose()
