"""cyclotron_optimizer: Radia-based compact-cyclotron design and optimization.

Library-first API: write short project scripts that import this package,
instead of editing a monolithic main.py. Typical use:

    import cyclotron_optimizer as co

    with co.Session("machine_muon.yml") as s:
        model = s.build()
        model.solve()
        iso = model.isochronism()
        fmap = model.median_plane_field(resolution_mm=2.0, gpu_precision="single")
        if s.is_root:
            fmap.save("output/midplane.comsol")
            model.show(field=fmap)

Import-order notes handled here (so scripts don't have to know them):
  - MKL is pinned to the sequential threading layer BEFORE numpy/gmsh load,
    avoiding the libiomp5md / libomp140 OpenMP runtime clash.
  - CONDA_PREFIX is defaulted to sys.prefix so cupy imports in
    non-activated shells (its import path joins CONDA_PREFIX unguarded).
  - radia is imported BEFORE mpi4py (Session imports mpi4py lazily), which
    is required for radia's MPI hooks to initialize correctly.
"""

import os as _os
import sys as _sys

_os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
_os.environ.setdefault("CONDA_PREFIX", _sys.prefix)

# Detect a headless/batch environment (e.g. a cluster compute node with no
# $DISPLAY) and pin matplotlib to the non-interactive Agg backend + PyVista
# off-screen BEFORE any pyplot/radia import, so live windows and 3D viewers
# degrade to file output instead of hanging/crashing. Desktops and Jupyter
# notebooks are left untouched (a notebook renders inline -- see runtime.py).
from cyclotron_optimizer.runtime import configure_headless_matplotlib  # noqa: E402
configure_headless_matplotlib()

import radia  # noqa: F401,E402  (must precede any mpi4py import)

from cyclotron_optimizer.config_io.config import CyclotronConfig  # noqa: E402
from cyclotron_optimizer.session import CyclotronModel, GpuOptions, Session  # noqa: E402

__version__ = "0.2.0"

__all__ = [
    "CyclotronConfig",
    "CyclotronModel",
    "GpuOptions",
    "Session",
    "__version__",
]
