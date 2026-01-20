"""Cyclotron frequency calculations."""

import numpy as np
from core.species import IonSpecies


def revolution_frequency_from_b_and_velocity(b_field_tesla: float,
                                             species: IonSpecies) -> float:
    """
    Calculate cyclotron revolution frequency from B-field and particle properties.

    f = (q/m) * B / (2π)

    :param b_field_tesla: Magnetic field strength in Tesla
    :param species: IonSpecies object
    :return: Revolution frequency in Hz
    """
    q_over_m = species.q_over_m  # C/kg
    omega = q_over_m * b_field_tesla  # rad/s
    f = omega / (2.0 * np.pi)  # Hz
    return f


def revolution_time_from_radius_and_velocity(radius_mm: float,
                                             velocity_m_per_s: float) -> float:
    """
    Calculate revolution time for circular orbit.

    T = 2πr / v

    :param radius_mm: Orbit radius in mm
    :param velocity_m_per_s: Particle velocity in m/s
    :return: Revolution time in seconds
    """
    radius_m = radius_mm * 1e-3
    if velocity_m_per_s <= 0:
        raise ValueError("Velocity must be positive")
    rev_time = 2.0 * np.pi * radius_m / velocity_m_per_s
    return rev_time


def revolution_frequency_from_radius_and_velocity(radius_mm: float,
                                                  velocity_m_per_s: float) -> float:
    """
    Calculate revolution frequency from radius and velocity.

    f = v / (2πr)

    :param radius_mm: Orbit radius in mm
    :param velocity_m_per_s: Particle velocity in m/s
    :return: Revolution frequency in Hz
    """
    rev_time = revolution_time_from_radius_and_velocity(radius_mm, velocity_m_per_s)
    return 1.0 / rev_time


def isochronism_deviation(rev_frequencies: np.ndarray) -> tuple:
    """
    Calculate isochronism quality metrics.

    For perfect isochronism, all revolution frequencies should be identical.

    :param rev_frequencies: Array of revolution frequencies in Hz
    :return: Tuple of (mean_freq, std_dev, percent_deviation)
    """
    mean_freq = np.mean(rev_frequencies)
    std_dev = np.std(rev_frequencies)
    percent_deviation = 100.0 * std_dev / mean_freq if mean_freq > 0 else np.inf

    return mean_freq, std_dev, percent_deviation