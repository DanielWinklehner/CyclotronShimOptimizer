"""Median-plane field visualization.

Two views of a PyPATools Field holding the median-plane B-field (as returned
by simulation.field_calculator.get_median_plane_field):

  - plot_median_plane_field: standard 2D matplotlib filled-contour plot.
  - show_model_with_median_plane_field: the Radia model in a PyVista window
    (via PyRadia's mesh converters) with the field as a semi-transparent
    filled-contour plane at z = 0.

The Field's grid is in meters (tracking convention); both views display mm to
match the Radia model coordinates.
"""

import matplotlib.pyplot as plt
import numpy as np


def _field_plane_mm(field):
    """(x_mm, y_mm, bz) arrays from a 2D median-plane Field."""
    if field.grid is None or field.grid_values is None:
        raise ValueError(
            "Field has no raw grid arrays; median-plane visualization needs a "
            "Field produced by get_median_plane_field / Field.from_arrays."
        )
    x_mm = np.asarray(field.grid["x"]) * 1e3
    y_mm = np.asarray(field.grid["y"]) * 1e3
    bz = np.asarray(field.grid_values["z"])
    if bz.ndim != 2:
        bz = bz.reshape(len(x_mm), len(y_mm))
    return x_mm, y_mm, bz


def plot_median_plane_field(field, *, n_levels=40, cmap="viridis", ax=None,
                            title="Median-plane field  $B_z(x, y, z{=}0)$",
                            show=False):
    """Filled-contour plot of Bz on the median plane.

    :param field: 2D PyPATools Field (grid in meters, Bz in Tesla).
    :param n_levels: number of filled contour levels.
    :param ax: existing matplotlib Axes (a new figure is created if None).
    :param show: call plt.show() (default False; the caller usually collects
        figures and shows them together).
    :return: (fig, ax)
    """
    x_mm, y_mm, bz = _field_plane_mm(field)
    xx, yy = np.meshgrid(x_mm, y_mm, indexing="ij")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6.5))
    else:
        fig = ax.figure

    filled = ax.contourf(xx, yy, bz, levels=n_levels, cmap=cmap)
    ax.contour(xx, yy, bz, levels=n_levels, colors="k",
               linewidths=0.3, alpha=0.4)
    fig.colorbar(filled, ax=ax, label="$B_z$ (T)")

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def build_field_plane(field, *, max_points_per_axis=None):
    """PyVista StructuredGrid of the median-plane Bz at z = 0 (coordinates mm).

    Separate from the viewer so the scalar/point alignment is unit-testable.

    :param max_points_per_axis: subsample the grid to at most this many points
        per axis (None = full resolution). The interactive overlay does not
        need the field-solve resolution -- a ~200x200 plane renders orders of
        magnitude faster than 800x800 with no visible loss.
    """
    import pyvista as pv

    x_mm, y_mm, bz = _field_plane_mm(field)
    if max_points_per_axis is not None:
        step_x = max(1, int(np.ceil(len(x_mm) / max_points_per_axis)))
        step_y = max(1, int(np.ceil(len(y_mm) / max_points_per_axis)))
        x_mm = x_mm[::step_x]
        y_mm = y_mm[::step_y]
        bz = bz[::step_x, ::step_y]

    xx, yy = np.meshgrid(x_mm, y_mm, indexing="ij")
    plane = pv.StructuredGrid(xx, yy, np.zeros_like(xx))
    plane["Bz (T)"] = bz.ravel(order="F")  # F-order matches StructuredGrid points
    return plane


def show_model_with_median_plane_field(radia_id, field, *,
                                       model_opacity=1.0,
                                       show_edges=True,
                                       field_opacity=0.6,
                                       n_levels=24,
                                       cmap="viridis",
                                       overlay_max_points=256,
                                       off_screen=False,
                                       screenshot=None):
    """Show the Radia model in PyVista with the median-plane Bz overlaid.

    The field is rendered as a semi-transparent filled-contour plane at z = 0:
    true geometric contour bands (vtkBandedPolyDataContourFilter via
    pyvista's contour_banded) with the band boundaries drawn as black lines,
    so the bands are crisp and smooth instead of per-triangle color banding.
    The overlay is subsampled to `overlay_max_points` per axis -- rendering a
    full field-solve-resolution plane (800x800 cells, semi-transparent, so it
    is re-blended every frame) is what makes the viewer sluggish.
    Blocks until the interactive window is closed (unless off_screen).

    :param radia_id: Radia object id of the model to draw (typically the
        omit_symmetry visualization rebuild).
    :param field: 2D PyPATools Field of the median plane.
    :param overlay_max_points: max overlay-plane points per axis (None = full).
    :param off_screen: render without opening a window (e.g. batch/screenshot).
    :param screenshot: optional path; the rendered scene is saved there.
    """
    import pyvista as pv
    import radia as rad
    from PyRadia.radia_viewer import _add_vtk_data, _add_vtk_lines

    # ---- model geometry (same path as PyRadia.ObjDrwPyVista) ----
    data = rad.ObjDrwVTK(radia_id, 'EdgeLines->False')
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background("white")
    _add_vtk_data(plotter, data.get("polygons", {}),
                  opacity=model_opacity, show_edges=show_edges)
    _add_vtk_lines(plotter, data.get("lines", {}))

    # ---- median-plane field as a semi-transparent plane at z = 0 ----
    plane = build_field_plane(field, max_points_per_axis=overlay_max_points)
    # algorithm=None is the announced future default (vtkGeometryFilter);
    # passing it explicitly silences the PyVistaFutureWarning.
    surface = plane.extract_surface(algorithm=None)
    rng = surface.get_data_range("Bz (T)")
    scalar_bar_args = dict(title="Bz (T)", color="black", vertical=True,
                           position_x=0.88, position_y=0.25,
                           width=0.07, height=0.5)

    try:
        bands, band_edges = surface.contour_banded(
            n_levels + 1, scalars="Bz (T)", generate_contour_edges=True)
        plotter.add_mesh(bands, scalars="Bz (T)", cmap=cmap, n_colors=n_levels,
                         clim=rng, opacity=field_opacity, lighting=False,
                         scalar_bar_args=scalar_bar_args)
        if band_edges.n_points:
            plotter.add_mesh(band_edges, color="black", line_width=1.5,
                             opacity=min(1.0, field_opacity + 0.3),
                             show_scalar_bar=False)
    except Exception:
        # Fallback (e.g. constant field): plain color-banded surface
        plotter.add_mesh(surface, scalars="Bz (T)", cmap=cmap, n_colors=n_levels,
                         clim=rng, opacity=field_opacity, lighting=False,
                         scalar_bar_args=scalar_bar_args)

    plotter.enable_anti_aliasing("fxaa")
    plotter.add_axes()
    plotter.show(screenshot=screenshot)
