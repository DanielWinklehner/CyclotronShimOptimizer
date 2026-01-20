"""Plotting functions for cyclotron optimization results."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def plot_convergence(radii_mm, history):
    """Visualize convergence like your 30 kHz band → target 1 kHz"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Convergence
    axes[0, 0].semilogy(history['max_error_khz'], 'ro-', label='Max deviation')
    axes[0, 0].semilogy(history['rms_error_khz'], 'bo-', label='RMS deviation')
    axes[0, 0].axhline(1.0, color='green', linestyle='--', label='Target')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Deviation from Mean (kHz)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Frequency span over iterations
    axes[0, 1].plot(history['freq_span_khz'], 'go-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Frequency Span (kHz)')
    axes[0, 1].set_title('Total Frequency Range')
    axes[0, 1].grid(True, alpha=0.3)

    # Final frequency profile
    axes[1, 0].plot(radii_mm, history['freqs'][-1], 'bo-', markersize=8)
    axes[1, 0].axhline(history['mean_freq_mhz'][-1], color='red',
                       linestyle='--', label='Mean')
    axes[1, 0].set_xlabel('Radius (mm)')
    axes[1, 0].set_ylabel('Frequency (MHz)')
    axes[1, 0].set_title('Final Frequency Profile')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Mean frequency evolution
    axes[1, 1].plot(history['mean_freq_mhz'], 'mo-', linewidth=2)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Mean Frequency (MHz)')
    axes[1, 1].set_title('Mean Frequency (depends on coil current)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('isochronous_optimization.png', dpi=150)
    plt.show()

def plot_isochronism_results(radii_mm: List[float],
                             bz_tesla: List[float],
                             energies_mev: List[float],
                             rev_times_s: List[float],
                             rev_frequencies_mhz: List[float],
                             title: str = "Cyclotron Isochronism Analysis",
                             colors: Optional[List[str]] = None,
                             show: bool = True) -> Tuple:
    """
    Plot B-field, energy, revolution time, and resonant frequency vs radius.

    Creates a 2-subplot figure with dual y-axes as per your specification.

    :param radii_mm: List of radii in mm
    :param bz_tesla: List of B-field values in Tesla
    :param energies_mev: List of kinetic energies in MeV
    :param rev_times_s: List of revolution times in seconds
    :param rev_frequencies_mhz: List of resonant frequencies in MHz
    :param title: Title for the figure
    :param colors: List of colors for plotting (left, right)
    :param show: Whether to call plt.show()
    :return: Tuple of (fig, axes)
    """
    if colors is None:
        colors = ['#4B82B8', '#B8474D']  # Blue, Red

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left subplot: B-field and Energy --- #
    plt.sca(ax[0])
    ax0_twin = ax[0].twinx()

    line1 = ax[0].plot(radii_mm, bz_tesla, color=colors[0], linewidth=2, label='B-field')
    ax[0].set_xlabel("Radius (mm)", fontsize=11)
    ax[0].set_ylabel("Average Bz (T)", color=colors[0], fontsize=11)
    ax[0].tick_params(axis='y', labelcolor=colors[0])
    ax[0].grid(True, alpha=0.3)

    line2 = ax0_twin.plot(radii_mm, energies_mev, color=colors[1], linewidth=2,
                          linestyle='--', label='Energy')
    ax0_twin.set_ylabel("Particle Energy (MeV)", color=colors[1], fontsize=11)
    ax0_twin.tick_params(axis='y', labelcolor=colors[1])

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax[0].legend(lines, labels, loc='upper left', fontsize=10)

    # --- Right subplot: Revolution time and Frequency --- #
    plt.sca(ax[1])
    ax1_twin = ax[1].twinx()

    line3 = ax[1].plot(radii_mm, np.array(rev_times_s) * 1e9, color=colors[0],
                       linewidth=2, label='Rev. time')  # Convert to ns for readability
    ax[1].set_xlabel("Radius (mm)", fontsize=11)
    ax[1].set_ylabel("Revolution Time (ns)", color=colors[0], fontsize=11)
    ax[1].tick_params(axis='y', labelcolor=colors[0])
    ax[1].grid(True, alpha=0.3)

    line4 = ax1_twin.plot(radii_mm, rev_frequencies_mhz, color=colors[1],
                          linewidth=2, linestyle='--', label='Frequency')
    ax1_twin.set_ylabel("Resonant Frequency (MHz)", color=colors[1], fontsize=11)
    ax1_twin.tick_params(axis='y', labelcolor=colors[1])

    # Combined legend
    lines = line3 + line4
    labels = [l.get_label() for l in lines]
    ax[1].legend(lines, labels, loc='upper left', fontsize=10)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_isochronism_metric(radii_mm: List[float],
                            rev_frequencies_mhz: List[float],
                            mean_freq: float,
                            std_dev: float,
                            percent_deviation: float,
                            show: bool = True) -> Tuple:
    """
    Plot frequency deviation from mean (isochronism quality).

    :param radii_mm: List of radii in mm
    :param rev_frequencies_mhz: List of resonant frequencies in MHz
    :param mean_freq: Mean frequency in MHz
    :param std_dev: Standard deviation in MHz
    :param percent_deviation: Percent deviation
    :param show: Whether to call plt.show()
    :return: Tuple of (fig, ax)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    deviation_ppm = (np.array(rev_frequencies_mhz) - mean_freq) / mean_freq * 1e6

    ax.plot(radii_mm, deviation_ppm, 'o-', color='#4B82B8', linewidth=2, markersize=6)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Perfect isochronism')
    ax.fill_between(radii_mm, -1, 1, alpha=0.2, color='green', label='±1 ppm')

    ax.set_xlabel("Radius (mm)", fontsize=11)
    ax.set_ylabel("Frequency Deviation (ppm)", fontsize=11)
    ax.set_title(f"Isochronism Quality (σ = {std_dev:.3f} MHz, Δ = {percent_deviation:.3f}%)",
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    if show:
        plt.show()

    return fig, ax