"""Runtime environment detection: ``desktop`` / ``notebook`` / ``headless``.

Stdlib-only (``os``, ``sys``) so it is importable from anywhere -- including the
package ``__init__`` *before* radia / matplotlib load -- with zero circular-import
risk. Used to decide, once, whether an interactive display is available, so that
PyVista 3D windows and live matplotlib windows are suppressed on cluster / batch
nodes while a final image is still written to disk.

Three modes (see :func:`display_mode`):

* ``"desktop"``  -- a real display (Windows/macOS, or POSIX with ``$DISPLAY``):
  interactive windows, the historical behaviour.
* ``"notebook"`` -- inside a Jupyter/IPython kernel: render-capable *inline*
  (pyvista Jupyter backend + ``%matplotlib inline``), no X11. This is the
  JupyterHub path -- deliberately left open, NOT disabled.
* ``"headless"`` -- no display and not a notebook (``srun``/``sbatch``/CI):
  disable PyVista, fall back to Agg, and save a final frame instead of
  opening a window.

Override with the ``CYCLOTRON_HEADLESS`` environment variable
(``1``/``true``/``yes``/``on`` -> headless, ``0``/``false``/``no``/``off`` ->
not headless); it wins over autodetection.
"""
from __future__ import annotations

import os
import sys

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}
# matplotlib backends that never open a window -> safe to treat as headless.
_NONINTERACTIVE_MPL = {"agg", "pdf", "svg", "ps", "cairo", "template"}

ENV_OVERRIDE = "CYCLOTRON_HEADLESS"


def _override():
    """Return True/False from ``CYCLOTRON_HEADLESS``, or None if unset/unparsable."""
    val = os.environ.get(ENV_OVERRIDE)
    if val is None:
        return None
    v = val.strip().lower()
    if v in _TRUTHY:
        return True
    if v in _FALSY:
        return False
    return None


def in_notebook() -> bool:
    """True when running inside a Jupyter/IPython ZMQ kernel (notebook/lab/hub).

    A plain terminal IPython shell (``TerminalInteractiveShell``) is NOT a
    notebook -- it has a normal display situation -- so only the ZMQ kernel
    (``ZMQInteractiveShell``) counts as render-capable-inline.
    """
    try:
        from IPython import get_ipython  # type: ignore
    except Exception:
        return False
    try:
        ip = get_ipython()
    except Exception:
        return False
    return ip is not None and type(ip).__name__ == "ZMQInteractiveShell"


def is_headless() -> bool:
    """True when there is no interactive display and we are not in a notebook.

    Precedence:
      1. ``CYCLOTRON_HEADLESS`` overrides everything.
      2. A Jupyter notebook is render-capable -> not headless.
      3. ``PYVISTA_OFF_SCREEN`` truthy or a non-interactive ``MPLBACKEND`` -> headless.
      4. Windows / macOS -> not headless (they do not use ``$DISPLAY``).
      5. POSIX -> headless unless ``$DISPLAY`` or ``$WAYLAND_DISPLAY`` is set
         (covers SLURM compute nodes, which have neither).
    """
    forced = _override()
    if forced is not None:
        return forced
    if in_notebook():
        return False
    if os.environ.get("PYVISTA_OFF_SCREEN", "").strip().lower() in _TRUTHY:
        return True
    if os.environ.get("MPLBACKEND", "").strip().lower() in _NONINTERACTIVE_MPL:
        return True
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return False
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def display_mode() -> str:
    """One of ``'desktop'`` | ``'notebook'`` | ``'headless'``.

    The ``'notebook'`` branch is the hook for JupyterHub inline rendering:
    callers that want to wire it up later (``pv.set_jupyter_backend(...)``,
    ``%matplotlib inline``) can branch on this without reworking the headless
    gating.
    """
    if is_headless():
        return "headless"
    if in_notebook():
        return "notebook"
    return "desktop"


_MPL_CONFIGURED = False


def configure_headless_matplotlib(force: bool | None = None) -> bool:
    """When headless, pin matplotlib to Agg and PyVista to off-screen.

    Selecting a non-interactive backend up front means every later
    ``plt.show()`` becomes a harmless non-blocking no-op instead of trying to
    open (or waiting on) a window. Call this BEFORE ``matplotlib.pyplot`` is
    first imported so the backend is chosen cleanly. Idempotent. Returns True
    iff headless configuration was applied.

    A no-op in ``desktop`` and ``notebook`` modes -- a notebook installs its own
    inline backend and must not be stomped.
    """
    global _MPL_CONFIGURED
    headless = is_headless() if force is None else bool(force)
    if not headless:
        return False
    if _MPL_CONFIGURED:
        return True
    _MPL_CONFIGURED = True
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except Exception:
        pass
    return True
