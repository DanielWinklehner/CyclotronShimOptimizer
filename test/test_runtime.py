"""Tests for headless / notebook / desktop runtime detection."""

import _testenv  # noqa: F401  (pins MPLBACKEND=Agg etc. before heavy imports)

import pytest

from cyclotron_optimizer import runtime


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start each test from a known-empty display environment."""
    for var in ("CYCLOTRON_HEADLESS", "DISPLAY", "WAYLAND_DISPLAY",
                "MPLBACKEND", "PYVISTA_OFF_SCREEN"):
        monkeypatch.delenv(var, raising=False)
    # Isolate from any real IPython kernel so platform/DISPLAY logic is tested.
    monkeypatch.setattr(runtime, "in_notebook", lambda: False)


def test_override_forces_headless(monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")   # would otherwise be non-headless
    monkeypatch.setenv("DISPLAY", ":0")            # would otherwise be non-headless
    monkeypatch.setenv("CYCLOTRON_HEADLESS", "1")
    assert runtime.is_headless() is True
    assert runtime.display_mode() == "headless"


def test_override_forces_display(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")   # would otherwise be headless (no DISPLAY)
    monkeypatch.setenv("CYCLOTRON_HEADLESS", "false")
    assert runtime.is_headless() is False
    assert runtime.display_mode() == "desktop"


def test_posix_without_display_is_headless(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    assert runtime.is_headless() is True


def test_posix_with_display_is_desktop(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("DISPLAY", ":0")
    assert runtime.is_headless() is False
    assert runtime.display_mode() == "desktop"


def test_wayland_display_counts(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert runtime.is_headless() is False


@pytest.mark.parametrize("plat", ["win32", "darwin"])
def test_windows_and_mac_default_to_display(monkeypatch, plat):
    monkeypatch.setattr("sys.platform", plat)
    assert runtime.is_headless() is False


def test_noninteractive_mpl_backend_is_headless(monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")   # even on a display platform
    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert runtime.is_headless() is True


def test_pyvista_off_screen_is_headless(monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setenv("PYVISTA_OFF_SCREEN", "true")
    assert runtime.is_headless() is True


def test_notebook_is_not_headless(monkeypatch):
    # A Jupyter kernel renders inline -> not headless, even with no DISPLAY.
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(runtime, "in_notebook", lambda: True)
    assert runtime.is_headless() is False
    assert runtime.display_mode() == "notebook"


def test_notebook_override_still_wins(monkeypatch):
    # An explicit override beats notebook autodetection.
    monkeypatch.setattr(runtime, "in_notebook", lambda: True)
    monkeypatch.setenv("CYCLOTRON_HEADLESS", "1")
    assert runtime.is_headless() is True


def test_configure_headless_matplotlib_noop_when_not_headless(monkeypatch):
    monkeypatch.setattr(runtime, "_MPL_CONFIGURED", False)
    assert runtime.configure_headless_matplotlib(force=False) is False


def test_configure_headless_matplotlib_applies_agg(monkeypatch):
    monkeypatch.setattr(runtime, "_MPL_CONFIGURED", False)
    assert runtime.configure_headless_matplotlib(force=True) is True
    import matplotlib
    assert matplotlib.get_backend().lower() == "agg"
