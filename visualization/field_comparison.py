"""Compare external field map (COMSOL) with Radia-calculated field."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from scipy.interpolate import griddata

from config_io.config import CyclotronConfig
from core.frequency import revolution_time_from_radius_and_velocity
from core.species import IonSpecies
from core.particles import ParticleDistribution


def load_comsol_fieldmap(filename: str, verbosity: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load COMSOL ASCII fieldmap.

    Format: 8 header lines, 1 column name line, then N data lines
    Columns: x (m), y (m), Bz (T)

    :param filename: Path to COMSOL fieldmap file
    :param verbosity: Verbosity level
    :return: (x_m, y_m, bz_t) arrays
    """

    if verbosity >= 1:
        print(f"\nLoading COMSOL fieldmap from {filename}...", flush=True)

    # Load data, skipping 8 header lines + 1 column name line
    data = np.loadtxt(filename, skiprows=9)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    x_m = data[:, 0]
    y_m = data[:, 1]
    bz_t = data[:, 2]

    if verbosity >= 1:
        print(f"  Loaded {len(x_m)} points", flush=True)
        print(f"  X range: {np.min(x_m) * 1e3:.1f} to {np.max(x_m) * 1e3:.1f} mm", flush=True)
        print(f"  Y range: {np.min(y_m) * 1e3:.1f} to {np.max(y_m) * 1e3:.1f} mm", flush=True)
        print(f"  Bz range: {np.min(bz_t):.4f} to {np.max(bz_t):.4f} T", flush=True)

    return x_m, y_m, bz_t


def extract_radii_from_fieldmap(x_m: np.ndarray,
                                y_m: np.ndarray,
                                bz_t: np.ndarray,
                                radii_mm: List[float],
                                num_angles: int = 360,
                                verbosity: int = 1) -> np.ndarray:
    """
    Extract and average Bz at specified radii from COMSOL fieldmap.

    For each radius:
    1. Generate points at angles 0 to 2π
    2. Interpolate Bz using griddata
    3. Average across angles

    :param x_m: X coordinates from fieldmap (m)
    :param y_m: Y coordinates from fieldmap (m)
    :param bz_t: Bz values from fieldmap (T)
    :param radii_mm: List of radii to extract (mm)
    :param num_angles: Number of angles per radius for averaging
    :param verbosity: Verbosity level
    :return: Array of Bz values at each radius
    """

    if verbosity >= 1:
        print(f"\nExtracting B-field at {len(radii_mm)} radii from fieldmap...", flush=True)

    # Convert fieldmap to mm for consistency
    x_mm = x_m * 1e3
    y_mm = y_m * 1e3

    # Create angle array
    angles = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False)

    # Create 2D grid of evaluation points
    points_grid = np.zeros((len(radii_mm), num_angles, 2))

    for i, r_mm in enumerate(radii_mm):
        points_grid[i, :, 0] = r_mm * np.cos(angles)  # x
        points_grid[i, :, 1] = r_mm * np.sin(angles)  # y

    # Flatten for griddata
    points_flat = points_grid.reshape(-1, 2)
    fieldmap_points = np.column_stack([x_mm, y_mm])

    # Interpolate using griddata
    bz_interpolated = griddata(
        fieldmap_points,
        bz_t,
        points_flat,
        method='linear',
        fill_value=np.nan
    )

    # Reshape back
    bz_grid = bz_interpolated.reshape(len(radii_mm), num_angles)

    # Average across angles (ignore NaN from extrapolation)
    bz_avg = np.nanmean(bz_grid, axis=1)

    if verbosity >= 1:
        print(f"  Interpolated {len(radii_mm)} radii", flush=True)
        print(f"  Valid points: {np.sum(~np.isnan(bz_avg))} / {len(radii_mm)}", flush=True)

    return bz_avg


def calculate_frequencies_from_field(config: CyclotronConfig,
                                     radii_mm: np.ndarray,
                                     bz_values: np.ndarray,
                                     verbosity: int = 1) -> Tuple[np.ndarray, float, float, float]:
    """
    Calculate revolution frequencies from B-field values.

    :param config: CyclotronConfig
    :param radii_mm: Array of radii (mm)
    :param bz_values: Array of Bz values (T)
    :param verbosity: Verbosity level
    :return: (frequencies_mhz, mean_freq, std_dev, percent_dev)
    """

    if verbosity >= 1:
        print(f"\nCalculating revolution frequencies...", flush=True)

    species = IonSpecies(config.particle_species)
    particles = ParticleDistribution(species=species)

    rev_frequencies_mhz = []

    for r_mm, bz_t in zip(radii_mm, bz_values):
        if np.isnan(bz_t):
            rev_frequencies_mhz.append(np.nan)
            continue

        # Calculate b_rho from r and Bz
        b_rho_tmm = r_mm * bz_t
        b_rho_tm = b_rho_tmm * 1e-3  # Convert to T·m

        # Set particle momentum from b_rho
        energy = particles.set_z_momentum_from_b_rho(b_rho_tm)

        # Calculate revolution time
        v_mean = particles.v_mean_m_per_s
        rev_time = revolution_time_from_radius_and_velocity(r_mm, v_mean)

        # Calculate frequency
        rev_freq_hz = 1.0 / rev_time
        rev_freq_mhz = rev_freq_hz / 1e6
        rev_frequencies_mhz.append(rev_freq_mhz)

    frequencies = np.array(rev_frequencies_mhz)

    # Remove NaN for statistics
    valid_freqs = frequencies[~np.isnan(frequencies)]

    if len(valid_freqs) > 0:
        mean_freq = np.mean(valid_freqs)
        std_dev = np.std(valid_freqs)
        percent_dev = 100.0 * std_dev / mean_freq
    else:
        mean_freq = np.nan
        std_dev = np.nan
        percent_dev = np.nan

    if verbosity >= 1:
        print(f"  Mean frequency: {mean_freq:.6f} MHz", flush=True)
        print(f"  Std. deviation: {std_dev:.6f} MHz", flush=True)
        print(f"  % Deviation: {percent_dev:.4f}%", flush=True)

    return frequencies, mean_freq, std_dev, percent_dev


def compare_fields(external_field_filename: str,
                   config: CyclotronConfig,
                   radii_mm_radia: np.ndarray,
                   bz_values_radia: np.ndarray,
                   rev_frequencies_radia_mhz: np.ndarray,
                   mean_freq_radia_mhz: float,
                   std_dev_radia_mhz: float,
                   percent_dev_radia: float,
                   pole_shape=None,
                   show: bool = False,
                   verbosity: int = 1):
    """
    Load external fieldmap and compare with Radia-calculated field.

    Creates two subplots:
    - Left: B-field vs radius
    - Right: Revolution frequency vs radius

    :param external_field_filename: Path to COMSOL fieldmap file
    :param config: CyclotronConfig
    :param radii_mm_radia: Radia radii (mm)
    :param bz_values_radia: Radia B-field (T)
    :param rev_frequencies_radia_mhz: Radia frequencies (MHz)
    :param mean_freq_radia_mhz: Radia mean frequency
    :param std_dev_radia_mhz: Radia std deviation
    :param percent_dev_radia: Radia % deviation
    :param pole_shape: PoleShape object (for title only)
    :param show: whether or not to show matplotlib figure
    :param verbosity: Verbosity level
    """

    if verbosity >= 1:
        print("\n" + "=" * 80, flush=True)
        print("FIELD COMPARISON: RADIA vs COMSOL", flush=True)
        print("=" * 80, flush=True)

    # Load external fieldmap
    x_m, y_m, bz_comsol_t = load_comsol_fieldmap(external_field_filename, verbosity)

    # Extract B-field at same radii as Radia
    bz_values_comsol = extract_radii_from_fieldmap(
        x_m, y_m, bz_comsol_t,
        radii_mm_radia.tolist(),
        num_angles=360,
        verbosity=verbosity
    )

    # Calculate frequencies from COMSOL field
    rev_freq_comsol, mean_freq_comsol, std_dev_comsol, percent_dev_comsol = \
        calculate_frequencies_from_field(config, radii_mm_radia, bz_values_comsol, verbosity)

    # ===== CREATE COMPARISON PLOTS =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: B-field comparison
    ax1.plot(radii_mm_radia, bz_values_radia, 'b-', linewidth=2, label='Radia', marker='o', markersize=4)
    ax1.plot(radii_mm_radia, bz_values_comsol, 'r--', linewidth=2, label='COMSOL', marker='s', markersize=4)
    ax1.set_xlabel('Radius (mm)', fontsize=12)
    ax1.set_ylabel('Bz (T)', fontsize=12)
    ax1.set_title('Magnetic Field Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # Add field difference annotation
    valid_mask = ~np.isnan(bz_values_comsol)
    if np.sum(valid_mask) > 0:
        max_diff = np.max(np.abs(bz_values_radia[valid_mask] - bz_values_comsol[valid_mask]))
        ax1.text(0.98, 0.05, f'Max diff: {max_diff:.4f} T',
                 transform=ax1.transAxes, fontsize=10, ha='right', va='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Frequency comparison
    valid_mask = ~np.isnan(rev_freq_comsol)
    ax2.plot(radii_mm_radia, rev_frequencies_radia_mhz, 'b-', linewidth=2, label='Radia', marker='o', markersize=4)
    ax2.plot(radii_mm_radia[valid_mask], rev_freq_comsol[valid_mask], 'r--', linewidth=2,
             label='COMSOL', marker='s', markersize=4)
    ax2.axhline(y=mean_freq_radia_mhz, color='b', linestyle=':', alpha=0.7,
                label=f'Radia mean: {mean_freq_radia_mhz:.4f} MHz')
    ax2.axhline(y=mean_freq_comsol, color='r', linestyle=':', alpha=0.7,
                label=f'COMSOL mean: {mean_freq_comsol:.4f} MHz')
    ax2.set_xlabel('Radius (mm)', fontsize=12)
    ax2.set_ylabel('Revolution Frequency (MHz)', fontsize=12)
    ax2.set_title('Revolution Frequency Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Add statistics box
    stats_text = (
        f"Radia:\n"
        f"  Mean: {mean_freq_radia_mhz:.6f} MHz\n"
        f"  Std: {std_dev_radia_mhz:.6f} MHz\n"
        f"  Flatness: {percent_dev_radia:.4f}%\n\n"
        f"COMSOL:\n"
        f"  Mean: {mean_freq_comsol:.6f} MHz\n"
        f"  Std: {std_dev_comsol:.6f} MHz\n"
        f"  Flatness: {percent_dev_comsol:.4f}%"
    )
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    if show:
        plt.show()

    if verbosity >= 1:
        print("\n" + "=" * 80, flush=True)
        print("COMPARISON SUMMARY", flush=True)
        print("=" * 80, flush=True)
        print(f"\nRadia:")
        print(f"  Mean frequency: {mean_freq_radia_mhz:.6f} MHz")
        print(f"  Std deviation: {std_dev_radia_mhz:.6f} MHz")
        print(f"  Flatness: {percent_dev_radia:.4f}%")
        print(f"\nCOMSOL:")
        print(f"  Mean frequency: {mean_freq_comsol:.6f} MHz")
        print(f"  Std deviation: {std_dev_comsol:.6f} MHz")
        print(f"  Flatness: {percent_dev_comsol:.4f}%")
        print(f"\nDifference:")
        print(f"  Frequency offset: {abs(mean_freq_radia_mhz - mean_freq_comsol):.6f} MHz")
        print(f"  Flatness difference: {abs(percent_dev_radia - percent_dev_comsol):.4f}%")
        print("=" * 80 + "\n", flush=True)
