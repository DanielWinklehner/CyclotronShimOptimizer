"""Test environment setup: import this FIRST in every test module.

- Sets CONDA_PREFIX (cupy's import raises TypeError without it in
  non-activated shells) and a headless matplotlib backend.
- Puts the repo root and the PyPATools source tree on sys.path so the tests
  exercise the edited sources.
"""

import os
import sys

os.environ.setdefault("CONDA_PREFIX", sys.prefix)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("I_MPI_FABRICS", "shm")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYPATOOLS_SRC = os.path.abspath(os.path.join(REPO_ROOT, "..", "PyPATools", "src"))

for path in (REPO_ROOT, PYPATOOLS_SRC):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)
