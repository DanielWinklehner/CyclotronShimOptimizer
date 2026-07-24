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

import os

import matplotlib.pyplot as plt
import numpy as np


def _polydata_bounds(pgn_data):
    """Axis-aligned bounds [xmin,xmax,ymin,ymax,zmin,zmax] of the radia polygon
    vertices, or None if there are none. Used to fit the coordinate rulers to
    the SOLID model, so a stray line vertex (radia's draw data occasionally
    carries far-off / degenerate line points) cannot blow the box up.
    """
    v = pgn_data.get("vertices", []) if pgn_data else []
    if len(v) == 0:
        return None
    verts = np.asarray(v, dtype=float).reshape(-1, 3)
    return [float(verts[:, 0].min()), float(verts[:, 0].max()),
            float(verts[:, 1].min()), float(verts[:, 1].max()),
            float(verts[:, 2].min()), float(verts[:, 2].max())]


def _nice_ticks(vmin, vmax, target=5):
    """Round [vmin, vmax] outward to a 'nice' tick step (Heckbert's algorithm)
    so ticks land on round numbers (multiples of 1/2/5 x 10^k -- i.e. tens,
    fifties, hundreds ... depending on the scale). Returns (lo, hi, n_labels)
    where the ticks are lo, lo+step, ..., hi.
    """
    vmin, vmax = float(vmin), float(vmax)
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
        return vmin, vmax, 2

    def _nicenum(x, round_):
        exp = np.floor(np.log10(x))
        f = x / 10.0 ** exp
        if round_:
            nf = 1 if f < 1.5 else 2 if f < 3 else 5 if f < 7 else 10
        else:
            nf = 1 if f <= 1 else 2 if f <= 2 else 5 if f <= 5 else 10
        return nf * 10.0 ** exp

    rng = _nicenum(vmax - vmin, False)
    step = _nicenum(rng / max(1, target - 1), True)
    lo = np.floor(vmin / step) * step
    hi = np.ceil(vmax / step) * step
    n = int(round((hi - lo) / step)) + 1
    return float(lo), float(hi), n


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


def _make_toggle_callbacks(plotter, field_actors, axes_actors, scalar_bar_title):
    """Build the (toggle_field, toggle_axes) key callbacks for the viewer.

    Factored out (and free of any pyvista/VTK import) so the visibility logic
    is unit-testable with fakes -- no GL context needed. ``toggle_field`` hides
    or shows every field actor (the contour bands, their black edges) and the
    field's scalar bar; ``toggle_axes`` hides or shows the ticked bounds axes
    (the CubeAxesActor rulers, NOT the corner orientation tripod). Each is a
    no-argument callable (the shape pyvista's add_key_event expects) and
    re-renders after flipping state. Returns (toggle_field, toggle_axes, state)
    -- ``state`` is exposed for tests.
    """
    state = {"field": True, "axes": True}

    def _set_visible(actor, visible):
        try:
            actor.visibility = visible          # pyvista.Actor property
        except Exception:                       # vtk fallback (e.g. CubeAxesActor)
            actor.SetVisibility(bool(visible))

    def toggle_field():
        state["field"] = not state["field"]
        for actor in field_actors:
            _set_visible(actor, state["field"])
        try:
            plotter.scalar_bars[scalar_bar_title].SetVisibility(state["field"])
        except Exception:                       # pragma: no cover - version/kind guard
            pass
        plotter.render()

    def toggle_axes():
        state["axes"] = not state["axes"]
        for actor in axes_actors:
            _set_visible(actor, state["axes"])
        plotter.render()

    return toggle_field, toggle_axes, state


def show_model_with_median_plane_field(radia_id, field, *,
                                       model_opacity=1.0,
                                       show_edges=True,
                                       field_opacity=0.6,
                                       n_levels=24,
                                       cmap="viridis",
                                       overlay_max_points=256,
                                       off_screen=False,
                                       screenshot=None,
                                       field_toggle_key="b",
                                       axes_toggle_key="a",
                                       draw_model_lines=True):
    """Show the Radia model in PyVista with the median-plane Bz overlaid.

    The field is rendered as a semi-transparent filled-contour plane at z = 0:
    true geometric contour bands (vtkBandedPolyDataContourFilter via
    pyvista's contour_banded) with the band boundaries drawn as black lines,
    so the bands are crisp and smooth instead of per-triangle color banding.
    The overlay is subsampled to `overlay_max_points` per axis -- rendering a
    full field-solve-resolution plane (800x800 cells, semi-transparent, so it
    is re-blended every frame) is what makes the viewer sluggish.
    Blocks until the interactive window is closed (unless off_screen).

    While the window is open, press ``field_toggle_key`` (default 'b') to
    show/hide the B-field overlay and ``axes_toggle_key`` (default 'a') to
    show/hide the ticked bounds axes (the coordinate rulers around the model,
    in mm; the small orientation tripod in the corner stays put), independently.
    These override any pyvista default binding for those keys (pyvista binds 'b'
    by default) so the key performs only the toggle. A one-line hint is drawn in
    the lower-left.

    :param radia_id: Radia object id of the model to draw (typically the
        omit_symmetry visualization rebuild).
    :param field: 2D PyPATools Field of the median plane.
    :param overlay_max_points: max overlay-plane points per axis (None = full).
    :param off_screen: render without opening a window (e.g. batch/screenshot).
    :param screenshot: optional path; the rendered scene is saved there.
    :param field_toggle_key: key that toggles the B-field overlay (None to skip).
    :param axes_toggle_key: key that toggles the ticked bounds axes (None to skip).
    :param draw_model_lines: draw radia's remaining line data. The Cartesian
        frame axes (shafts + arrowhead cones through the origin) are already
        suppressed at the source via ObjDrwVTK 'Axes->False'; this just drops
        any other radia line data. The solid geometry comes from the polygon
        data and is unaffected either way.
    """
    import pyvista as pv
    import radia as rad
    from PyRadia.radia_viewer import _add_vtk_data, _add_vtk_lines

    # ---- model geometry (same path as PyRadia.ObjDrwPyVista) ----
    # Axes->False drops radia's Cartesian frame axes (the shaft lines AND the
    # arrowhead cones) so they don't clutter / block the view. NOTE: radia
    # separates multiple options with ';' (a ',' raises "Improper definition
    # of optional parameters"), despite the docstring using commas to list them.
    data = rad.ObjDrwVTK(radia_id, 'EdgeLines->False;Axes->False')
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background("white")
    _add_vtk_data(plotter, data.get("polygons", {}),
                  opacity=model_opacity, show_edges=show_edges)
    if draw_model_lines:
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

    # Keep handles to the field actors so the key event can toggle them.
    field_actors = []
    try:
        bands, band_edges = surface.contour_banded(
            n_levels + 1, scalars="Bz (T)", generate_contour_edges=True)
        field_actors.append(plotter.add_mesh(
            bands, scalars="Bz (T)", cmap=cmap, n_colors=n_levels,
            clim=rng, opacity=field_opacity, lighting=False,
            scalar_bar_args=scalar_bar_args))
        if band_edges.n_points:
            field_actors.append(plotter.add_mesh(
                band_edges, color="black", line_width=1.5,
                opacity=min(1.0, field_opacity + 0.3),
                show_scalar_bar=False))
    except Exception:
        # Fallback (e.g. constant field): plain color-banded surface
        field_actors.append(plotter.add_mesh(
            surface, scalars="Bz (T)", cmap=cmap, n_colors=n_levels,
            clim=rng, opacity=field_opacity, lighting=False,
            scalar_bar_args=scalar_bar_args))

    plotter.enable_anti_aliasing("fxaa")
    plotter.add_axes()  # small orientation tripod in the corner (always on)

    # ---- ticked bounds axes (coordinate rulers in mm), toggleable via 'a' ----
    # Fit to the SOLID model's own bounds (so a stray vertex can't inflate the
    # box), then round each axis outward to nice round tick steps.
    axes_actors = []
    model_bounds = _polydata_bounds(data.get("polygons", {}))
    try:
        kw = dict(grid="back", location="outer", ticks="both", color="black",
                  xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)", fmt="%.0f")
        if model_bounds is not None:
            xr = _nice_ticks(model_bounds[0], model_bounds[1])
            yr = _nice_ticks(model_bounds[2], model_bounds[3])
            zr = _nice_ticks(model_bounds[4], model_bounds[5])
            kw["bounds"] = [xr[0], xr[1], yr[0], yr[1], zr[0], zr[1]]
            kw["n_xlabels"], kw["n_ylabels"], kw["n_zlabels"] = xr[2], yr[2], zr[2]
        bounds_actor = plotter.show_bounds(**kw)
        if bounds_actor is not None:
            axes_actors.append(bounds_actor)
    except Exception:                           # pragma: no cover - version guard
        pass

    # ---- optional actor dump for debugging stray geometry (env-gated) ----
    if os.environ.get("CYCLO_VIEWER_DEBUG"):
        pgn = data.get("polygons", {}) or {}
        lin = data.get("lines", {}) or {}
        print(f"[viewer] ObjDrwVTK: {len(pgn.get('vertices', [])) // 3} polygon "
              f"verts, {len(lin.get('vertices', [])) // 3} line verts; "
              f"model_bounds={None if model_bounds is None else [round(b) for b in model_bounds]}",
              flush=True)
        try:
            for name, actor in plotter.renderer.actors.items():
                try:
                    b = [round(x, 1) for x in actor.GetBounds()]
                    vis = actor.GetVisibility()
                except Exception:
                    b, vis = None, None
                print(f"[viewer]   actor {name!r}: {type(actor).__name__} "
                      f"bounds={b} visible={vis}", flush=True)
        except Exception as exc:
            print(f"[viewer]   actor dump failed: {exc}", flush=True)

    # ---- interactive toggles (B-field / axes), skipped when off-screen ----
    if not off_screen and (field_toggle_key or axes_toggle_key):
        toggle_field, toggle_axes, _ = _make_toggle_callbacks(
            plotter, field_actors, axes_actors, scalar_bar_args["title"])
        hints = []
        for key, cb, label in ((field_toggle_key, toggle_field, "B-field"),
                               (axes_toggle_key, toggle_axes, "axes")):
            if not key:
                continue
            # add_key_event APPENDS, and pyvista binds a few keys by default
            # (e.g. 'b'); clear first so our toggle is the sole handler and the
            # key does not fire two actions at once.
            try:
                plotter.clear_events_for_key(key)
            except Exception:                   # pragma: no cover - version guard
                pass
            plotter.add_key_event(key, cb)
            hints.append(f"[{key}] {label}")
        plotter.add_text("   ".join(hints), position="lower_left",
                         font_size=9, color="black")

    plotter.show(screenshot=screenshot)
