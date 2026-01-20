"""Energy to radius mapping for cyclotron orbits."""

import numpy as np
from scipy import constants
from core.particles import ParticleDistribution
from core.species import IonSpecies


def energy_to_b_rho(energy_mev: float, species: IonSpecies) -> float:
    """
    Convert kinetic energy to magnetic rigidity (B*ρ).

    Uses relativistic relation:
    E_kin = sqrt((p*c)^2 + (m*c^2)^2) - m*c^2

    Where p is momentum and ρ = p / (q*B)

    :param energy_mev: Kinetic energy in MeV
    :param species: IonSpecies object
    :return: Magnetic rigidity B*ρ in Tesla·mm
    """
    mass_mev = species.mass_mev
    q = species.q

    if energy_mev < 0:
        raise ValueError("Energy must be non-negative")

    # Relativistic energy-momentum relation
    # E_kin = sqrt((pc)^2 + (mc^2)^2) - mc^2
    # Solve for pc
    pc_mev = np.sqrt(energy_mev ** 2 + 2.0 * energy_mev * mass_mev)

    # Convert pc (MeV) to p*c (eV)
    pc_ev = pc_mev * 1e6

    # B*ρ = p / q = (pc) / (q*c)
    # In SI: p (kg·m/s) = pc (J) / c (m/s)
    # B*ρ (T·m) = p / q
    #
    # Useful form: B*ρ (T·mm) = pc (MeV) / (q * c_in_SI)
    # where pc is in MeV and we convert appropriately

    c = constants.speed_of_light  # m/s
    e = constants.elementary_charge  # C

    # pc in Joules
    pc_joules = pc_mev * 1e6 * e

    # p in kg·m/s
    p = pc_joules / c

    # B*ρ in T·m
    b_rho_tm = p / (q * e)

    # Convert to T·mm
    b_rho_tmm = b_rho_tm * 1e3

    return b_rho_tmm


def b_rho_to_radius(b_rho_tmm: float, b_field_tesla: float) -> float:
    """
    Convert magnetic rigidity and B-field to orbit radius.

    ρ = B*ρ / B

    :param b_rho_tmm: Magnetic rigidity in Tesla·mm
    :param b_field_tesla: B-field strength in Tesla
    :return: Orbit radius in mm
    """
    if b_field_tesla <= 0:
        raise ValueError("B-field must be positive")

    radius_mm = b_rho_tmm / b_field_tesla
    return radius_mm


def energy_to_radius(energy_mev: float,
                     b_field_tesla: float,
                     species: IonSpecies) -> float:
    """
    Convert particle energy to orbit radius given B-field.

    :param energy_mev: Kinetic energy in MeV
    :param b_field_tesla: B-field strength in Tesla
    :param species: IonSpecies object
    :return: Orbit radius in mm
    """
    b_rho = energy_to_b_rho(energy_mev, species)
    radius_mm = b_rho_to_radius(b_rho, b_field_tesla)
    return radius_mm