"""Real-time optimization progress visualization."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from matplotlib.patches import Rectangle
from typing import Optional

from cyclotron_optimizer.runtime import is_headless, display_mode


class OptimizationProgressPlotter:
    """Real-time visualization of cyclotron optimization progress."""

    def __init__(self):
        """Initialize plotter (no setup yet)."""
        self.fig = None
        self.axes = None
        self.lines = {}
        self.static_elements = {}
        self.config = {}
        self.iteration_history = {'iterations': [], 'objectives_best': [], 'objectives_current': []}

    def setup(self,
              figsize: tuple = (20, 9),
              inner_radius_mm: float = 50.0,
              outer_radius_mm: float = 400.0,
              pole_angle_deg: float = 25.0,
              target_frequency: Optional[float] = None):
        """
        Set up figure and axes with 3-column layout:
        - Left: Side shim (top) and Top shim (bottom)
        - Middle: B-field and Frequency vs radius
        - Right: Objective vs iterations

        :param figsize: Figure size (width, height)
        :param inner_radius_mm: Inner pole radius
        :param outer_radius_mm: Outer pole radius
        :param pole_angle_deg: Full pole angle in degrees
        :param target_frequency: Target RF frequency for reference
        """
        # Only force an interactive GUI backend on a real desktop. In a notebook
        # keep Jupyter's inline backend; headless keeps the Agg backend pinned at
        # package import, so the figure is built off-screen and saved (never shown).
        if display_mode() == "desktop":
            matplotlib.use('TkAgg')

        self.fig = plt.figure(figsize=figsize)
        self.fig.canvas.manager.set_window_title("Cyclotron Optimization Progress")

        # Create grid: 2 rows, 3 columns
        # Increased wspace (width space) and hspace (height space)
        gs = gridspec.GridSpec(2, 3, figure=self.fig, width_ratios=[1, 1, 1],
                               height_ratios=[1, 1], hspace=0.4, wspace=0.5)

        ax_side = self.fig.add_subplot(gs[0, 0])
        ax_top = self.fig.add_subplot(gs[1, 0])
        ax_perf = self.fig.add_subplot(gs[:, 1])
        ax_perf_right = ax_perf.twinx()
        ax_obj = self.fig.add_subplot(gs[:, 2])

        self.axes = {
            'side': ax_side,
            'top': ax_top,
            'perf': ax_perf,
            'perf_right': ax_perf_right,
            'obj': ax_obj,
        }

        # Store config
        self.config = {
            'inner_radius_mm': inner_radius_mm,
            'outer_radius_mm': outer_radius_mm,
            'pole_angle_deg': pole_angle_deg,
            'target_frequency': target_frequency
        }

        # ===== SIDE SHIM PLOT (Left, Top) =====
        ax_side.set_xlabel('Radius (mm)', fontsize=10)
        ax_side.set_ylabel('Side Shim Offset (deg)', fontsize=10)
        ax_side.set_title('Side Shim Profile', fontsize=11, fontweight='bold')
        ax_side.grid(True, alpha=0.3)

        line_side_best, = ax_side.plot([], [], 'b-o', linewidth=2, markersize=4, label='Best so far')
        self.lines['side_best'] = line_side_best

        line_side_actual, = ax_side.plot([], [], 'r-s', linewidth=2, markersize=4, label='Current')
        self.lines['side_actual'] = line_side_actual

        ax_side.legend(fontsize=9, loc='best')

        # ===== TOP SHIM PLOT (Left, Bottom) =====
        ax_top.set_xlabel('Radius (mm)', fontsize=10)
        ax_top.set_ylabel('Top Shim Offset (mm)', fontsize=10)
        ax_top.set_title('Top Shim Profile', fontsize=11, fontweight='bold')
        ax_top.grid(True, alpha=0.3)

        line_top_best, = ax_top.plot([], [], 'b-o', linewidth=2, markersize=4, label='Best so far')
        self.lines['top_best'] = line_top_best

        line_top_actual, = ax_top.plot([], [], 'r-s', linewidth=2, markersize=4, label='Current')
        self.lines['top_actual'] = line_top_actual

        ax_top.legend(fontsize=9, loc='best')

        # ===== B-FIELD & FREQUENCY PLOT (Middle) =====
        ax_perf.set_xlabel('Radius (mm)', fontsize=10)
        ax_perf.set_ylabel('B-field (T)', fontsize=10, color='tab:blue')
        ax_perf.tick_params(axis='y', labelcolor='tab:blue')
        ax_perf.grid(True, alpha=0.3)
        ax_perf.set_title('B-field & Frequency vs Radius', fontsize=11, fontweight='bold')

        # B-field lines
        line_bz_best, = ax_perf.plot([], [], color='darkblue', linewidth=2, marker='o', markersize=3,
                                     label='B (best)', alpha=0.7)
        self.lines['bz_best'] = line_bz_best

        line_bz_actual, = ax_perf.plot([], [], color='tab:blue', linewidth=2, marker='o', markersize=3,
                                       label='B (current)', alpha=1.0)
        self.lines['bz_actual'] = line_bz_actual

        # Frequency axis
        ax_perf_right.spines['right'].set_visible(True)
        ax_perf_right.yaxis.tick_right()
        ax_perf_right.yaxis.set_label_position('right')
        ax_perf_right.set_ylabel('Frequency (MHz)', fontsize=10, color='tab:red')
        ax_perf_right.tick_params(axis='y', labelcolor='tab:red')

        # Frequency lines
        line_freq_best, = ax_perf_right.plot([], [], color='darkred', linewidth=2, marker='s', markersize=3,
                                             label='f (best)', alpha=0.7, linestyle='--')
        self.lines['freq_best'] = line_freq_best

        line_freq_actual, = ax_perf_right.plot([], [], color='tab:red', linewidth=2, marker='s', markersize=3,
                                               label='f (current)', alpha=1.0)
        self.lines['freq_actual'] = line_freq_actual

        # Target frequency line
        if target_frequency is not None:
            line_target = ax_perf_right.axhline(y=target_frequency, color='darkred', linestyle=':',
                                                linewidth=1.5, alpha=0.5, label=f'Target: {target_frequency:.3f}')
            self.static_elements['target_freq'] = line_target

        # Combined legend for perf plot
        lines1, labels1 = ax_perf.get_legend_handles_labels()
        lines2, labels2 = ax_perf_right.get_legend_handles_labels()
        ax_perf.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

        # ===== OBJECTIVE VS ITERATIONS PLOT (Right) =====
        ax_obj.set_xlabel('Iteration', fontsize=10)
        ax_obj.set_ylabel('Objective Value', fontsize=10, color='tab:green')
        ax_obj.tick_params(axis='y', labelcolor='tab:green')
        ax_obj.grid(True, alpha=0.3)
        ax_obj.set_title('Objective Progress', fontsize=11, fontweight='bold')
        ax_obj.set_yscale('log')  # Log scale for better visibility

        line_obj_best, = ax_obj.plot([], [], color='darkgreen', linewidth=2, marker='o', markersize=4,
                                     label='Best', alpha=0.7)
        self.lines['objective_best'] = line_obj_best

        line_obj_current, = ax_obj.plot([], [], color='tab:green', linewidth=1, marker='.', markersize=3,
                                        label='Current', alpha=0.5)
        self.lines['objective_current'] = line_obj_current

        ax_obj.legend(fontsize=9, loc='best')

        # Adjust layout to add padding between columns
        plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.1)

        if not is_headless():
            plt.show(block=False)

    def update(self, iteration: int,
               shim_offsets_best: np.ndarray,
               shim_offsets_actual: np.ndarray,
               n_segments: int,
               current_objective: float = None,
               best_objective: float = None,
               radii_mm: np.ndarray = None,
               bz_values: list = None,
               bz_values_best: list = None,
               rev_frequencies_mhz: list = None,
               rev_frequencies_best_mhz: list = None,
               side_radii_mm: np.ndarray = None,
               top_radii_mm: np.ndarray = None):
        """
        Update all plots with current iteration data.

        :param iteration: Current iteration number
        :param shim_offsets_best: Best side shim offsets (degrees)
        :param shim_offsets_actual: Current side shim offsets (degrees)
        :param n_segments: Total number of shim segments
        :param current_objective: Current objective value
        :param best_objective: Best objective value so far
        :param radii_mm: Radii for B-field/frequency plots
        :param bz_values: Current B-field values
        :param bz_values_best: Best B-field values seen so far
        :param rev_frequencies_mhz: Current frequencies
        :param rev_frequencies_best_mhz: Best frequencies seen so far
        :param side_radii_mm: Radii for side shim plot
        :param top_radii_mm: Radii for top shim plot
        """

        # Extract side and top from best and actual
        n_side = n_segments + 1
        side_best = shim_offsets_best[:n_side]
        top_best = shim_offsets_best[n_side:]
        side_actual = shim_offsets_actual[:n_side]
        top_actual = shim_offsets_actual[n_side:]

        # Generate radii arrays if not provided
        if side_radii_mm is None and radii_mm is not None:
            side_radii_mm = np.linspace(self.config['inner_radius_mm'],
                                        self.config['outer_radius_mm'], n_side)
        if top_radii_mm is None and radii_mm is not None:
            top_radii_mm = np.linspace(self.config['inner_radius_mm'],
                                       self.config['outer_radius_mm'], n_side)

        # ===== UPDATE SIDE SHIM PLOT =====
        if side_radii_mm is not None:
            self.lines['side_best'].set_xdata(side_radii_mm)
            self.lines['side_best'].set_ydata(side_best)

            self.lines['side_actual'].set_xdata(side_radii_mm)
            self.lines['side_actual'].set_ydata(side_actual)

            self.axes['side'].relim()
            self.axes['side'].autoscale_view()

        # ===== UPDATE TOP SHIM PLOT =====
        if top_radii_mm is not None:
            self.lines['top_best'].set_xdata(top_radii_mm)
            self.lines['top_best'].set_ydata(top_best)

            self.lines['top_actual'].set_xdata(top_radii_mm)
            self.lines['top_actual'].set_ydata(top_actual)

            self.axes['top'].relim()
            self.axes['top'].autoscale_view()

        # ===== UPDATE B-FIELD & FREQUENCY PLOT =====
        if radii_mm is not None and len(radii_mm) > 0:

            # Current B-field
            if bz_values is not None and len(bz_values) > 0:
                self.lines['bz_actual'].set_xdata(radii_mm)
                self.lines['bz_actual'].set_ydata(bz_values)

            # Best B-field
            if bz_values_best is not None and len(bz_values_best) > 0:
                self.lines['bz_best'].set_xdata(radii_mm)
                self.lines['bz_best'].set_ydata(bz_values_best)

            # Current frequency
            if rev_frequencies_mhz is not None and len(rev_frequencies_mhz) > 0:
                self.lines['freq_actual'].set_xdata(radii_mm)
                self.lines['freq_actual'].set_ydata(rev_frequencies_mhz)

            # Best frequency
            if rev_frequencies_best_mhz is not None and len(rev_frequencies_best_mhz) > 0:
                self.lines['freq_best'].set_xdata(radii_mm)
                self.lines['freq_best'].set_ydata(rev_frequencies_best_mhz)

            self.axes['perf'].relim()
            self.axes['perf'].autoscale_view()
            self.axes['perf_right'].relim()
            self.axes['perf_right'].autoscale_view()

            # ===== UPDATE OBJECTIVE VS ITERATIONS PLOT =====
            if iteration is not None:
                self.iteration_history['iterations'].append(iteration)
                if best_objective is not None:
                    self.iteration_history['objectives_best'].append(best_objective)
                if current_objective is not None:
                    self.iteration_history['objectives_current'].append(current_objective)

                if len(self.iteration_history['iterations']) > 0:
                    # Plot best objective
                    if len(self.iteration_history['objectives_best']) > 0:
                        self.lines['objective_best'].set_xdata(
                            self.iteration_history['iterations'][:len(self.iteration_history['objectives_best'])])
                        self.lines['objective_best'].set_ydata(self.iteration_history['objectives_best'])

                    # Plot current objective
                    if len(self.iteration_history['objectives_current']) > 0:
                        self.lines['objective_current'].set_xdata(
                            self.iteration_history['iterations'][:len(self.iteration_history['objectives_current'])])
                        self.lines['objective_current'].set_ydata(self.iteration_history['objectives_current'])

                    self.axes['obj'].relim()
                    self.axes['obj'].autoscale_view()

        # ===== UPDATE TITLE WITH METRICS =====
        metrics_str = f'Iteration {iteration}'
        if best_objective is not None:
            metrics_str += f' | Best Obj: {best_objective:.4e}'
        if current_objective is not None and current_objective != best_objective:
            metrics_str += f' | Current Obj: {current_objective:.4e}'

        self.fig.suptitle(metrics_str, fontsize=12, fontweight='bold', y=0.98)

        # ===== RENDER UPDATE =====
        # Interactive refresh only when a window is actually on screen. Headless,
        # the artist data above is enough -- finalize() renders it via savefig.
        if not is_headless():
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)


    def finalize(self, savepath=None):
        """Write the final progress frame to disk (used for headless/batch runs)."""
        if self.fig is not None and savepath:
            try:
                self.fig.savefig(savepath, dpi=120)
            except Exception:
                pass


class DFOLSProgressPlotter:
    """Live 3-panel view for the joint DFO-LS shim optimization.

    Left column (physical projections of the pole):
      - top: meridional (r-z) side view -- the top 5 cm of the base pole as a slab, the
        top-shim profile as the pole-top surface above it (best vs current), vertical end
        lines closing the cross-section; drawn at 1:1 aspect.
      - bottom: plan / top-down (x-y) view -- the upper half of the hill, side edge at
        angle (half_angle + side_offset(r)) so the edge point is (r cos, r sin); inner and
        outer boundary arcs centered at the origin; best vs current.
    Middle: average Bz(r) [left axis] and revolution frequency(r) [right axis] + target.
    Right: optimizer convergence (best + current flatness ||r||).
    """

    def __init__(self):
        self.fig = None
        self.hist = {'eval': [], 'cur': [], 'best': []}

    def setup(self, *, inner_radius_mm, outer_radius_mm, half_angle_deg, n_seg,
              target_frequency=None, base_depth_mm=50.0, z_exaggeration=2.0, figsize=(18, 7)):
        # Only force an interactive GUI backend on a real desktop. In a notebook
        # keep Jupyter's inline backend; headless keeps the Agg backend pinned at
        # package import, so the figure is built off-screen and saved (never shown).
        if display_mode() == "desktop":
            matplotlib.use('TkAgg')
        self.r_in = float(inner_radius_mm)
        self.r_out = float(outer_radius_mm)
        self.half_angle = float(half_angle_deg)
        self.r_shim = np.linspace(self.r_in, self.r_out, n_seg + 1)
        self.target = target_frequency
        self.base_depth = float(base_depth_mm)
        self.z_exag = float(z_exaggeration)

        # constrained_layout keeps axis labels (incl. the twin axis) from overlapping.
        self.fig = plt.figure(figsize=figsize, constrained_layout=True)
        try:
            self.fig.canvas.manager.set_window_title("DFO-LS shim optimization")
        except Exception:
            pass
        # Left column split EXACTLY 50:50; middle and right span both rows.
        gs = gridspec.GridSpec(2, 3, figure=self.fig, width_ratios=[1, 1.15, 1],
                               height_ratios=[1, 1])
        self.ax_mer = self.fig.add_subplot(gs[0, 0])
        self.ax_plan = self.fig.add_subplot(gs[1, 0])
        self.ax_mid = self.fig.add_subplot(gs[:, 1])
        self.ax_mid_r = self.ax_mid.twinx()
        self.ax_conv = self.fig.add_subplot(gs[:, 2])
        if not is_headless():
            plt.show(block=False)

    def _hill_outline(self, side_offsets):
        """Upper-half hill outline in the x-y plane: inner arc -> side edge -> outer arc
        (the closing bottom edge is the centerline y=0)."""
        r = self.r_shim
        th = np.deg2rad(self.half_angle + np.asarray(side_offsets, dtype=float))
        ex, ey = r * np.cos(th), r * np.sin(th)
        ia = np.linspace(0.0, th[0], 24)
        ix, iy = r[0] * np.cos(ia), r[0] * np.sin(ia)
        oa = np.linspace(th[-1], 0.0, 24)
        ox, oy = r[-1] * np.cos(oa), r[-1] * np.sin(oa)
        return np.concatenate([ix, ex, ox]), np.concatenate([iy, ey, oy])

    def _draw_meridional(self, top_cur, top_best):
        ax = self.ax_mer
        ax.cla()
        r = self.r_shim
        ax.add_patch(Rectangle((self.r_in, -self.base_depth), self.r_out - self.r_in,
                               self.base_depth, facecolor='0.85', edgecolor='0.6', lw=0.8))
        ax.axhline(0, color='0.5', lw=0.6, ls='--')
        ax.plot(r, top_best, color='tab:blue', lw=2, marker='o', ms=3, label='best')
        ax.plot(r, top_cur, color='tab:red', lw=1.8, ls='--', marker='s', ms=3, label='current')
        for rr, zz in ((r[0], top_cur[0]), (r[-1], top_cur[-1])):
            ax.plot([rr, rr], [-self.base_depth, zz], color='0.5', lw=1)
        # z exaggerated x{z_exag} vs r; adjustable='datalim' lets the box fill the cell.
        ax.set_aspect(self.z_exag, adjustable='datalim')
        ax.set_xlabel('r (mm)')
        ax.set_ylabel(f'z = top shim (mm), {self.z_exag:g}x')
        ax.set_title(f'pole top - meridional (r-z), z x{self.z_exag:g}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')

    def _draw_plan(self, side_cur, side_best):
        ax = self.ax_plan
        ax.cla()
        xb, yb = self._hill_outline(side_best)
        xc, yc = self._hill_outline(side_cur)
        ax.fill(xb, yb, color='tab:blue', alpha=0.13)
        ax.plot(np.append(xb, xb[0]), np.append(yb, yb[0]), color='tab:blue', lw=1.6, label='best')
        ax.plot(np.append(xc, xc[0]), np.append(yc, yc[0]), color='tab:red', lw=1.6, ls='--', label='current')
        ax.axhline(0, color='0.5', lw=0.6, ls='--')
        # true x-y geometry (round arcs); adjustable='datalim' lets the box fill the cell.
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title('pole hill - top view (x-y), upper half', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')

    def _draw_mid(self, radii, bz_cur, bz_best, freq_cur, freq_best):
        ax, axr = self.ax_mid, self.ax_mid_r
        ax.cla()
        axr.cla()
        # keep the secondary (frequency) axis ticks + label on the RIGHT after cla()
        axr.yaxis.tick_right()
        axr.yaxis.set_label_position('right')
        if bz_best is not None:
            ax.plot(radii, bz_best, color='darkblue', lw=2, marker='o', ms=3, label='<Bz> best')
        ax.plot(radii, bz_cur, color='tab:blue', lw=1.6, ls='--', marker='o', ms=3, label='<Bz> cur')
        if freq_best is not None:
            axr.plot(radii, freq_best, color='darkred', lw=2, marker='s', ms=3, label='f best')
        axr.plot(radii, freq_cur, color='tab:red', lw=1.6, ls='--', marker='s', ms=3, label='f cur')
        if self.target is not None:
            axr.axhline(self.target, color='0.4', lw=1, ls=':', label=f'target {self.target:g}')
        ax.set_xlabel('r (mm)')
        ax.set_ylabel('<Bz> (T)', color='tab:blue')
        axr.set_ylabel('f_rev (MHz)', color='tab:red')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        axr.tick_params(axis='y', labelcolor='tab:red')
        ax.set_title('avg Bz & frequency vs r', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)
        l1, la1 = ax.get_legend_handles_labels()
        l2, la2 = axr.get_legend_handles_labels()
        ax.legend(l1 + l2, la1 + la2, fontsize=7, loc='best')

    def _draw_conv(self):
        ax = self.ax_conv
        ax.cla()
        e = self.hist['eval']
        ax.semilogy(e, self.hist['best'], color='darkgreen', lw=2, marker='o', ms=3, label='best')
        ax.semilogy(e, self.hist['cur'], color='tab:green', lw=1, marker='.', ms=3, alpha=0.5, label='current')
        ax.set_xlabel('evaluation #')
        ax.set_ylabel('flatness sigma (MHz)')
        ax.set_title('convergence', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=8, loc='best')

    def update(self, *, eval_idx, side_cur, top_cur, side_best, top_best,
               radii, bz_cur, freq_cur, bz_best=None, freq_best=None,
               obj_cur=None, obj_best=None, coil=None):
        self.hist['eval'].append(eval_idx)
        self.hist['cur'].append(obj_cur)
        self.hist['best'].append(obj_best)
        self._draw_meridional(np.asarray(top_cur, float), np.asarray(top_best, float))
        self._draw_plan(np.asarray(side_cur, float), np.asarray(side_best, float))
        self._draw_mid(radii, bz_cur, bz_best, freq_cur, freq_best)
        self._draw_conv()
        ttl = f'DFO-LS eval {eval_idx}'
        if obj_best is not None:
            ttl += f' | best flatness sigma={obj_best:.4g} MHz'
        if coil is not None:
            ttl += f' | coil={coil:.0f} A'
        self.fig.suptitle(ttl, fontsize=11, fontweight='bold')
        # Interactive refresh only when a window is actually on screen. Headless,
        # the artist data above is enough -- finalize() renders it via savefig.
        if not is_headless():
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def finalize(self, savepath=None):
        if self.fig is not None and savepath:
            try:
                self.fig.savefig(savepath, dpi=120)
            except Exception:
                pass
