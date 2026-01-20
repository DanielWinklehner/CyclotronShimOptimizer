"""Real-time optimization progress visualization."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from typing import Optional


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
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
