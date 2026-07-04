"""Diagnostic tools for inspecting computed field maps (plotting, outlier hunts).

Kept separate from field_calculator so the field-evaluation path has no
matplotlib/scipy.ndimage side effects.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter


def check_bz_outliers(
    x_range,
    y_range,
    bz,
    kernel_size=5,
    threshold=5.0,
    ax=None,
    show=False,
):
    """
    Detect point-like spikes in a 2D magnetic-field map and plot them.

    The detector uses a local median filter to absorb the legitimate
    smooth radial trend and azimuthal flutter, then flags points whose
    residual exceeds *threshold* × σ_MAD (robust standard deviation
    estimated from the median absolute deviation of the residuals).

    Parameters
    ----------
    x_range : array-like, shape (Nx,)
        Horizontal coordinates of the grid.
    y_range : array-like, shape (Ny,)
        Vertical coordinates of the grid.
    bz : ndarray, shape (Nx, Ny)
        Magnetic field on the regular grid.
    kernel_size : int, optional
        Side length of the square median-filter window.  Must be odd.
        Larger values tolerate broader legitimate features but may miss
        clusters of bad points.  Default 5.
    threshold : float, optional
        Number of robust standard deviations above which a residual is
        flagged as an outlier.  Default 5.0.
    ax : pair of matplotlib Axes or None, optional
        If given, plot into ``ax[0]`` (field) and ``ax[1]`` (residual).
        If *None* a new two-panel figure is created.
    show : bool, optional
        Call plt.show() after plotting (default False, so this can run in
        headless / batch contexts).

    Returns
    -------
    outlier_mask : ndarray of bool, shape (Nx, Ny)
        *True* where an outlier was detected.
    outlier_coords : ndarray, shape (N_outliers, 2)
        (x, y) physical coordinates of every flagged point.
    """
    x = np.asarray(x_range)
    y = np.asarray(y_range)
    bz = np.asarray(bz, dtype=float)

    # --- detection --------------------------------------------------------
    if kernel_size % 2 == 0:
        kernel_size += 1  # median_filter needs odd size

    bz_smooth = median_filter(bz, size=kernel_size)
    residual = bz - bz_smooth

    # robust scale: MAD → σ  (factor 1.4826 for normal-equivalent σ)
    mad = np.median(np.abs(residual - np.median(residual)))
    sigma = mad * 1.4826 if mad > 0 else np.std(residual)

    outlier_mask = np.abs(residual) > threshold * sigma

    # physical coordinates of flagged points
    ix, iy = np.nonzero(outlier_mask)
    outlier_x = x[ix]
    outlier_y = y[iy]
    outlier_coords = np.column_stack([outlier_x, outlier_y]) if ix.size else np.empty((0, 2))

    # --- plotting ---------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    else:
        fig = ax[0].figure

    X, Y = np.meshgrid(x, y, indexing="ij")

    # left panel: raw field
    c0 = ax[0].pcolormesh(X, Y, bz, shading="auto", cmap="viridis")
    fig.colorbar(c0, ax=ax[0], label="$B_z$")
    if outlier_x.size:
        ax[0].scatter(
            outlier_x, outlier_y,
            s=120, facecolors="none", edgecolors="red", linewidths=1.5,
            label=f"{outlier_x.size} outlier(s)",
        )
        ax[0].legend(loc="upper right", fontsize=9)
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Magnetic field  $B_z(x,y)$")
    ax[0].set_aspect("equal")

    # right panel: residual after median subtraction
    res_lim = max(np.abs(residual).max(), 1e-30)
    c1 = ax[1].pcolormesh(
        X, Y, residual, shading="auto",
        cmap="RdBu_r", vmin=-res_lim, vmax=res_lim,
    )
    fig.colorbar(c1, ax=ax[1], label="residual")
    if outlier_x.size:
        ax[1].scatter(
            outlier_x, outlier_y,
            s=120, facecolors="none", edgecolors="red", linewidths=1.5,
        )
    ax[1].axhline(0, color="grey", lw=0.3)
    ax[1].axvline(0, color="grey", lw=0.3)
    ax[1].set_xlabel("x")
    ax[1].set_title(
        f"Residual  (median kernel={kernel_size}, "
        f"threshold={threshold}σ,  σ_MAD={sigma:.3g})"
    )
    ax[1].set_aspect("equal")

    fig.tight_layout()

    if show:
        plt.show()

    # summary to console
    n = int(outlier_mask.sum())
    if n:
        peak = np.abs(residual[outlier_mask]).max()
        print(
            f"Found {n} outlier(s).  "
            f"Largest residual: {peak:.4g}  ({peak / sigma:.1f}σ)"
        )
    else:
        print("No outliers detected.")

    return outlier_mask, outlier_coords
