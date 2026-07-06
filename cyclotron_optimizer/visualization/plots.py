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


def plot_final_summary(radii_mm, bz_t, energies_mev, freq_mhz, tunes=None,
                       target_freq_mhz=None, title=None, show=False, savepath=None):
    """Final 4-panel summary of the optimized design.

    Panels: (top-left) average Bz(r) + Energy(r) on a secondary axis; (top-right)
    revolution frequency(r) + target line; (bottom-left) betatron tunes nu_r, nu_z(r)
    with the nu_r=1, nu_z=0 references and the Walkinshaw 2*nu_z line; (bottom-right)
    flutter F(r). ``tunes`` is the compute_isochronism()['tunes'] dict (None for
    circle/seo -> those panels show a placeholder).
    """
    r = np.asarray(radii_mm, dtype=float)
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')

    # ----- top-left: average Bz + Energy -----
    ax = axs[0, 0]
    ax.plot(r, np.asarray(bz_t, dtype=float), color='tab:blue', lw=2, marker='o', ms=3, label='<Bz>')
    ax.set_xlabel("Radius (mm)")
    ax.set_ylabel("<Bz> (T)", color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.grid(True, alpha=0.3)
    ax.set_title("Average field & energy", fontsize=11, fontweight='bold')
    if energies_mev is not None:
        axe = ax.twinx()
        axe.plot(r, np.asarray(energies_mev, dtype=float), color='tab:green', lw=1.8,
                 ls='--', marker='^', ms=3)
        axe.set_ylabel("Energy (MeV)", color='tab:green')
        axe.tick_params(axis='y', labelcolor='tab:green')

    # ----- top-right: frequency + target -----
    ax = axs[0, 1]
    f = np.asarray(freq_mhz, dtype=float)
    ax.plot(r, f, color='tab:red', lw=2, marker='s', ms=3, label='f_rev')
    if target_freq_mhz is not None:
        ax.axhline(target_freq_mhz, color='0.4', lw=1.2, ls=':', label=f'target {target_freq_mhz:g} MHz')
        ax.text(0.02, 0.04, f'std = {np.std(f) * 1e3:.2f} kHz', transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("Radius (mm)")
    ax.set_ylabel("f_rev (MHz)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Isochronism: revolution frequency", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='best')

    # ----- bottom-left: betatron tunes -----
    ax = axs[1, 0]
    if tunes is not None and tunes.get('nu_r') is not None:
        rt = np.asarray(tunes.get('r_mm', r), dtype=float)
        nu_r = np.asarray(tunes['nu_r'], dtype=float)
        nu_z = np.asarray(tunes['nu_z'], dtype=float)
        ax.plot(rt, nu_r, color='tab:purple', lw=2, marker='o', ms=3, label=r'$\nu_r$')
        ax.plot(rt, nu_z, color='tab:orange', lw=2, marker='s', ms=3, label=r'$\nu_z$')
        ax.plot(rt, 2.0 * nu_z, color='0.5', lw=1, ls='--', label=r'$2\nu_z$ (Walkinshaw)')
        ax.axhline(1.0, color='0.6', lw=0.8, ls=':')
        ax.axhline(0.0, color='0.6', lw=0.8, ls=':')
        ax.set_ylabel("tune")
        ax.legend(fontsize=9, loc='best')
    else:
        ax.text(0.5, 0.5, "tunes available with iso_method = gordon",
                ha='center', va='center', transform=ax.transAxes, fontsize=10, color='0.5')
        ax.set_ylabel("tune")
    ax.set_xlabel("Radius (mm)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Betatron tunes", fontsize=11, fontweight='bold')

    # ----- bottom-right: flutter -----
    ax = axs[1, 1]
    if tunes is not None and tunes.get('flutter') is not None:
        rt = np.asarray(tunes.get('r_mm', r), dtype=float)
        ax.plot(rt, np.asarray(tunes['flutter'], dtype=float), color='tab:cyan', lw=2, marker='o', ms=3)
        ax.set_ylabel(r"flutter  F = $\langle (B-\langle B\rangle)^2\rangle / \langle B\rangle^2$")
    else:
        ax.text(0.5, 0.5, "flutter available with iso_method = gordon",
                ha='center', va='center', transform=ax.transAxes, fontsize=10, color='0.5')
        ax.set_ylabel("flutter")
    ax.set_xlabel("Radius (mm)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Flutter", fontsize=11, fontweight='bold')

    fig.tight_layout()
    if savepath:
        try:
            fig.savefig(savepath, dpi=120)
        except Exception:
            pass
    if show:
        plt.show()

    return fig, axs