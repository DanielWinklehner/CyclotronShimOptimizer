"""Gordon-style isochronism (equilibrium-orbit revolution frequency from a field map)."""

import numpy as np
from scipy.interpolate import CubicSpline

CLIGHT = 299792458.0  # m/s
# p[MeV/c] = K * Z * B[T] * rho[m]   (practical magnetic-rigidity constant, = c/1e6)
RIGIDITY_K = 299.792458


def clean_azimuthal_average(B_octant, theta_octant, n_harmonics_max=6):
    """Leakage-free azimuthal-average field B0(r) from octant field samples.

    Returns the DC term of a least-squares cosine-harmonic fit (n = 4, 8, ...).
    A plain np.mean over a half-open (endpoint=False) octant leaks flutter into
    the average (the sampled cos(n*theta) do not sum to zero); this does not.
    For full-circle sampling it reduces to the plain mean.

    :param B_octant: (Nr, Ntheta) field samples
    :param theta_octant: (Ntheta,) angular samples
    :return: (Nr,) azimuthal-average field
    """
    B_octant = np.asarray(B_octant, dtype=float)
    theta = np.asarray(theta_octant, dtype=float)
    n_list = [4 * k for k in range(1, n_harmonics_max + 1)]
    design = np.column_stack([np.ones_like(theta)] + [np.cos(n * theta) for n in n_list])
    return np.linalg.lstsq(design, B_octant.T, rcond=None)[0][0]


def isochronicity_gordon_octant(
        B_octant,
        r_grid,
        theta_octant,
        species,            # IonSpecies object
        n_harmonics_max=6,
        spiral_angle_deg=0.0,
):
    """
    Compute equilibrium-orbit properties from a single field octant (Gordon's method).

    The orbit is treated in the smooth/scalloping approximation: each sector flutter
    harmonic a_n = A_n / B0 drives a radial scallop of relative amplitude
    x_n = a_n / (n^2 - 1). The revolution period is T = L / v, where L is the
    (longer-than-circular) orbit length and the momentum follows from the magnetic
    rigidity B*rho = (1/2pi) * closed_integral(B ds) = p / q.

    Parameters
    ----------
    B_octant : ndarray, shape (Nr, Ntheta)
        Median-plane |Bz| in one octant [0, pi/4) in Tesla (pass np.abs of the field).
    r_grid : ndarray, shape (Nr,)
        Radial grid in METERS (ascending).
    theta_octant : ndarray, shape (Ntheta,)
        Uniform angular samples of the octant, e.g. linspace(0, pi/4, N, endpoint=False).
        The samples tile the full circle by the 8-fold symmetry, so azimuthal averages
        are arithmetic means (rectangle/DFT rule), not Simpson integrals.
    species : IonSpecies
        Needs .q (signed charge state), .mass_mev, .q_over_m, .q_over_a.
    n_harmonics_max : int
        Number of sector harmonics n = 4, 8, ... to include.

    Returns
    -------
    dict with revolution frequency/period, energy, flutter, harmonics, etc.

    References
    ----------
    M. M. Gordon, "Computation of closed orbits and basic focusing properties...",
    Part. Accel. 16 (1984) 39; and NIM A300 (1991) 453.
    """
    B_octant = np.asarray(B_octant, dtype=float)
    r_grid = np.asarray(r_grid, dtype=float)
    theta = np.asarray(theta_octant, dtype=float)
    Nr, Ntheta = B_octant.shape

    # Sector harmonics (4-fold sector symmetry -> n = 4, 8, 12, ...)
    n_list = np.array([4 * k for k in range(1, n_harmonics_max + 1)], dtype=float)
    n_row = n_list[None, :]

    # --- 1. Azimuthal average + relative flutter harmonics ---
    # Fit B(theta) = B0 + sum_n A_n cos(n theta) by least squares over the octant
    # samples. Exact for the band-limited (cosine, even about 0 and pi/4) field, and
    # -- unlike a plain mean / cos-projection -- immune to the DC<->harmonic leakage
    # from the half-open (endpoint=False) octant sampling.
    design = np.column_stack([np.ones_like(theta)] + [np.cos(n * theta) for n in n_list])
    coeffs, *_ = np.linalg.lstsq(design, B_octant.T, rcond=None)  # (1 + n_harm, Nr)
    B0 = coeffs[0]                                               # (Nr,)
    An = coeffs[1:].T                                           # (Nr, n_harm)
    a_n = An / B0[:, None]                                       # relative harmonics

    # Flutter F = <(B/B0 - 1)^2> = (1/2) sum a_n^2  (reporting / vertical focusing)
    F = 0.5 * np.sum(a_n ** 2, axis=1)

    # --- 2. Equilibrium-orbit scalloping (leading order, nu_r^2 ~ 1) ---
    x_n = a_n / (n_row ** 2 - 1.0)                               # r(theta) = r_bar(1 + sum x_n cos n theta)
    sum_path = np.sum((n_row ** 2) * x_n ** 2, axis=1)          # sum n^2 x_n^2  -> path length
    sum_rho = np.sum(0.5 * a_n * x_n + 0.25 * (n_row ** 2) * x_n ** 2, axis=1)  # closed_integral(B ds) correction

    # --- 3. Orbit length and magnetic rigidity ---
    L = 2.0 * np.pi * r_grid * (1.0 + 0.25 * sum_path)          # m
    b_rho = B0 * r_grid * (1.0 + sum_rho)                       # T*m  (= closed_integral(B ds) / 2pi = p/q)

    # --- 4. Relativistic kinematics from rigidity (|Z| -> momentum magnitude) ---
    p_MeV_c = RIGIDITY_K * abs(species.q) * b_rho
    beta_gamma = p_MeV_c / species.mass_mev
    gamma = np.sqrt(1.0 + beta_gamma ** 2)
    beta = beta_gamma / gamma
    velocity = beta * CLIGHT                                    # m/s

    # --- 5. Revolution time / frequency:  T = L / v ---
    T_rev = L / velocity
    f_rev = 1.0 / T_rev
    omega_rev = 2.0 * np.pi * f_rev

    # Path-averaged field along the orbit: <B>_path = closed_integral(B ds) / L = 2pi*b_rho/L
    B_avg_orbit = 2.0 * np.pi * b_rho / L

    # --- 6. Field index (reporting) ---
    if Nr >= 4:
        dB0_dr = CubicSpline(r_grid, B0, bc_type='natural')(r_grid, nu=1)
        k_index = r_grid * dB0_dr / B0
        k_index[0] = k_index[1]
    else:
        k_index = np.zeros(Nr)

    # --- 6b. Betatron tunes (smooth/hard-edge approximation) ---
    #   nu_r^2 = 1 + k                       (radial; ~ gamma for a well-isochronized field)
    #   nu_z^2 = -k + F (1 + 2 tan^2 xi)     (vertical; xi = sector spiral angle)
    # TODO: the true spiral angle comes from the side-shim radial twist; default to a
    # radial sector (tan xi = 0). Pass spiral_angle_deg to include the spiral focusing boost.
    spiral_tan2 = np.tan(np.deg2rad(spiral_angle_deg)) ** 2
    nu_r_sq = 1.0 + k_index
    nu_z_sq = -k_index + F * (1.0 + 2.0 * spiral_tan2)
    nu_r = np.sqrt(np.clip(nu_r_sq, 0.0, None))
    # nu_z is imaginary where nu_z_sq < 0 (vertical defocusing) -> expose NaN there;
    # the signed nu_z_sq lets callers/plots flag the instability explicitly.
    nu_z = np.where(nu_z_sq > 0.0, np.sqrt(np.clip(nu_z_sq, 0.0, None)), np.nan)
    walkinshaw_distance = nu_r - 2.0 * nu_z   # 0 on the nu_r = 2 nu_z coupling resonance

    # --- 7. Energy ---
    energy_MeV = (gamma - 1.0) * species.mass_mev
    mass_number = species.q / species.q_over_a                  # = A (sign cancels)
    energy_MeV_u = energy_MeV / mass_number
    momentum_MeV_u_c = beta_gamma * species.mass_mev / mass_number

    return {
        'r': r_grid,
        'f_rev': f_rev,
        'T_rev': T_rev,
        'omega_rev': omega_rev,
        'B_avg_orbit': B_avg_orbit,
        'B0_azimuthal': B0,
        'flutter': F,
        'field_index': k_index,
        'nu_r': nu_r,
        'nu_z': nu_z,
        'nu_r_sq': nu_r_sq,
        'nu_z_sq': nu_z_sq,
        'walkinshaw_distance': walkinshaw_distance,
        'spiral_angle_deg': spiral_angle_deg,
        'gamma': gamma,
        'beta': beta,
        'beta_gamma': beta_gamma,
        'scalloping': x_n,
        'orbit_length': L,
        'energy_MeV_per_u': energy_MeV_u,
        'energy_MeV': energy_MeV,
        'momentum_MeV_per_u_c': momentum_MeV_u_c,
        'harmonics': {'n': n_list, 'An': An, 'a_n': a_n, 'Bn': np.zeros_like(An)},
    }


# ============================================================================
# Unified isochronism dispatch (circle / gordon / seo)
# ============================================================================
def _angles_from_config(config):
    """Legacy azimuthal sample angles (pre-RZFieldGrid): hardcoded 8-fold octant.

    Only used as a fallback for plain (Nr, Ntheta) arrays; grids produced by
    simulation.field_calculator.get_field_rz carry their actual angles (which
    are derived from the geometry's symmetries) and take precedence -- see
    _grid_and_angles.
    """
    n = config.field_evaluation.num_points_circle
    if config.field_evaluation.use_symmetry:
        return np.linspace(0.0, 0.25 * np.pi, int(n / 8.0), endpoint=False)
    return np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)


def _grid_and_angles(bz_values, config):
    """(Bz grid, azimuthal angles) for the circle/gordon methods.

    Prefers the angles attached to the field grid (RZFieldGrid from
    get_field_rz); falls back to the config-derived angles for plain arrays.
    """
    angles = getattr(bz_values, 'angles', None)
    grid = getattr(bz_values, 'bz', bz_values)
    if angles is None:
        angles = _angles_from_config(config)
    return np.asarray(grid, dtype=float), np.asarray(angles, dtype=float)


def _iso_metrics(freq_mhz):
    """(mean, std, percent_deviation) of a frequency array [MHz]."""
    freq_mhz = np.asarray(freq_mhz, dtype=float)
    mean = float(np.mean(freq_mhz))
    std = float(np.std(freq_mhz))
    pct = 100.0 * std / mean if mean > 0 else float('inf')
    return mean, std, pct


def _iso_result(method, energies_mev, rev_times_s, freq_mhz, bz_for_plot, orbits=None, tunes=None):
    mean, std, pct = _iso_metrics(freq_mhz)
    return {
        'method': method,
        'energies_mev': np.asarray(energies_mev, dtype=float),
        'rev_times_s': np.asarray(rev_times_s, dtype=float),
        'rev_frequencies_mhz': np.asarray(freq_mhz, dtype=float),
        'bz_for_plot': np.asarray(bz_for_plot, dtype=float),
        'mean_freq_mhz': mean,
        'std_dev_mhz': std,
        'percent_dev': pct,
        'orbits': orbits,
        'tunes': tunes,
    }


def _iso_circle(bz_values, radii_mm, config, species):
    """Circle method: rigidity from the leakage-free azimuthal-average field."""
    bz_grid, angles = _grid_and_angles(bz_values, config)
    B0 = clean_azimuthal_average(bz_grid, angles)
    r_m = np.asarray(radii_mm, dtype=float) * 1e-3
    b_rho = np.abs(B0) * r_m
    beta_gamma = RIGIDITY_K * abs(species.q) * b_rho / species.mass_mev
    gamma = np.sqrt(1.0 + beta_gamma ** 2)
    velocity = (beta_gamma / gamma) * CLIGHT
    rev_times_s = 2.0 * np.pi * r_m / velocity
    freq_mhz = 1.0 / rev_times_s / 1e6
    energies_mev = (gamma - 1.0) * species.mass_mev
    return _iso_result('circle', energies_mev, rev_times_s, freq_mhz, B0)


def _iso_gordon(bz_values, radii_mm, config, species):
    """Gordon method: flutter-corrected equilibrium-orbit frequency."""
    bz_grid, angles = _grid_and_angles(bz_values, config)
    res = isochronicity_gordon_octant(
        np.abs(bz_grid),
        np.asarray(radii_mm, dtype=float) * 1e-3,
        angles,
        species,
        n_harmonics_max=6,
    )
    tunes = {
        'r_mm': np.asarray(radii_mm, dtype=float),
        'nu_r': res['nu_r'],
        'nu_z': res['nu_z'],
        'nu_r_sq': res['nu_r_sq'],
        'nu_z_sq': res['nu_z_sq'],
        'walkinshaw_distance': res['walkinshaw_distance'],
        'flutter': res['flutter'],
        'field_index': res['field_index'],
    }
    return _iso_result('gordon', res['energy_MeV'], res['T_rev'],
                       res['f_rev'] * 1e-6, res['B_avg_orbit'], tunes=tunes)


def _iso_seo(bz_values, radii_mm, config, solver, energy_seeds_kev, verbose):
    """SEO method: track the static equilibrium orbits (PyCentralRegion)."""
    from PyCentralRegion import CentralRegion, SEOFinder  # lazy: heavy import

    design = CentralRegion(name="Isochronism", dimensionality='2D')
    design.set_species(config.particle_species)
    design.set_magnetic_field(bz_values)

    finder = SEOFinder(design, n_turns=20, steps_per_turn=500, n_theta_samples=360,
                       closure_tol_mm=0.1, algorithm='rk4_rel', solver=solver, verbose=verbose)
    orbits = finder.find_seos_at_radii(radii_mm, do_final_tracking=False,
                                       solver=solver, energy_seeds_kev=energy_seeds_kev)

    energies_mev = [1e-3 * o.energy_kev for o in orbits]
    freq_mhz = [1e-6 * o.frequency_hz for o in orbits]
    rev_times_s = [1.0 / o.frequency_hz if o.frequency_hz else 0.0 for o in orbits]
    bz_for_plot = [o.b_field_avg for o in orbits]
    return _iso_result('seo', energies_mev, rev_times_s, freq_mhz, bz_for_plot, orbits=orbits)


def compute_isochronism(method, bz_values, radii_mm, config, species, *,
                        solver='newton', energy_seeds_kev=None,
                        rank=0, comm=None, verbose=False):
    """Compute isochronism (energy / frequency vs radius) by the selected method.

    Single dispatch point for the circle / gordon / seo selection (previously a
    triple-if in main.py), so the optimizer reuses the same path.

    :param method: 'circle', 'gordon', or 'seo'.
    :param bz_values: field as returned by evaluate_radii_parallel for this method --
        an RZFieldGrid (or plain (Nr, Ntheta) array) for 'circle'/'gordon', or a
        2D PyPATools Field map for 'seo'. An RZFieldGrid carries the azimuthal
        sample angles actually used (symmetry-derived sector); plain arrays fall
        back to the config-derived octant/full-circle angles.
    :param radii_mm: evaluation radii [mm].
    :param config: CyclotronConfig.
    :param species: IonSpecies (used by circle/gordon).
    :param solver: SEO fixed-point solver ('newton' | 'symmetric' | 'centroid').
    :param energy_seeds_kev: optional per-radius Gordon energy seeds (seo only).
    :return: dict with keys {method, energies_mev, rev_times_s, rev_frequencies_mhz,
        bz_for_plot, mean_freq_mhz, std_dev_mhz, percent_dev, orbits}.
    """
    method = (method or 'circle').lower()
    if method == 'circle':
        return _iso_circle(bz_values, radii_mm, config, species)
    if method == 'gordon':
        return _iso_gordon(bz_values, radii_mm, config, species)
    if method == 'seo':
        return _iso_seo(bz_values, radii_mm, config, solver, energy_seeds_kev, verbose)
    raise ValueError(f"Unknown isochronism method '{method}' (expected circle|gordon|seo).")


if __name__ == '__main__':
    pass
