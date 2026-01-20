from global_variables import *
import numpy as np
# from numba import jitclass
# from numba import int32, float64
from core.species import IonSpecies

__author__ = "Daniel Winklehner"
__doc__ = "A class that holds particle distribution data and has a few handy functions " \
          "to calculate emittance and Twiss parameters"


class ParticleDistribution(object):
    def __init__(self, species=IonSpecies('proton'),
                 x=np.zeros(1),
                 y=np.zeros(1),
                 z=np.zeros(1),
                 px=np.zeros(1),
                 py=np.zeros(1),
                 pz=np.zeros(1),
                 q=0.0,
                 f=0.0,
                 recalculate=True):
        """
        A class that holds particle distribution data and has a few handy functions
        to calculate emittance and Twiss parameters.

        :param species: a IonSpecies object. Defaults to protons.
        :param x: numpy array of x coordinates (m)
        :param y: numpy array of y coordinates (m)
        :param z: numpy array of z coordinates (m)
        :param px: numpy array of momentum component in x direction (beta * gamma)
        :param py: numpy array of momentum component in y direction (beta * gamma)
        :param pz: numpy array of momentum component in z direction (beta * gamma)
        :param q: bunch charge (C)
        :param f: bunch frequency (Hz)
        :return:
        """
        self._species = species

        # Raw particle data
        self.x = x  # m
        self.y = y  # m
        self.z = z  # m
        self.px = px  # beta * gamma
        self.py = py  # beta * gamma
        self.pz = pz  # beta * gamma
        self.q = q  # C
        self.f = f  # Hz
        
        self.numpart = 0

        # --- Collective data --- #
        # means
        self._xm = 0.0  # m
        self._ym = 0.0  # m
        self._zm = 0.0  # m
        self._pxm = 0.0  # beta * gamma
        self._pym = 0.0  # beta * gamma
        self._pzm = 0.0  # beta * gamma
        self._ekin_mean = None  # Mean Energy (MeV)
        self._ekin_stdev = None  # RMS energy spread (MeV)

        # standard deviations
        self.x_std = 0.0  # (m)
        self.y_std = 0.0  # (m)
        self.z_std = 0.0  # (m)
        self.xp_std = 0.0  # (rad)
        self.yp_std = 0.0  # (rad)
        self.xxp_std = 0.0  # (m * rad)
        self.yyp_std = 0.0  # (m * rad)
        self.xyp_std = 0.0  # (m * rad)
        self.yxp_std = 0.0  # (m * rad)
        # ----------------------- #

        if recalculate:
            self.recalculate_all()
            # self.calculate_emittances()

    def recalculate_all(self):
        self.numpart = len(self.x)
        self.calculate_mean_r()
        self.calculate_mean_pr()
        self.calculate_mean_energy_mev()
        self.calculate_stdevs()

    def generate_random_sample(self, size=100000):

        self.x = np.random.random_sample(size=size)
        self.y = np.random.random_sample(size=size)
        self.z = np.random.random_sample(size=size)
        self.px = np.random.random_sample(size=size)
        self.py = np.random.random_sample(size=size)
        self.pz = np.random.random_sample(size=size)

        self.recalculate_all()

    @property
    def current(self):
        return self.q * self.f

    @property
    def bunch_charge(self):
        return self.q

    @property
    def bunch_freq(self):
        return self.f

    @bunch_charge.setter
    def bunch_charge(self, q):
        self.q = q

    @bunch_freq.setter
    def bunch_freq(self, f):
        self.f = f

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, species):
        if isinstance(species, IonSpecies):
            self._species = species
            self.recalculate_all()
        else:
            print("Expected an IonSpecies object, got {}.".format(type(species)))

    @property
    def mean_energy_mev_per_amu(self):
        if self._ekin_mean is None:
            self.calculate_mean_energy_mev()
        return self._ekin_mean / self.species.a

    @property
    def mean_energy_mev(self):
        if self._ekin_mean is None:
            self.calculate_mean_energy_mev()
        return self._ekin_mean

    @property
    def rms_energy_spread_mev(self):
        if self._ekin_mean is None:
            self.calculate_mean_energy_mev()
        return self._ekin_stdev

    @property
    def mean_momentum_betagamma(self):
        if Z_ENERGY:
            return np.mean(self.pz)
        else:
            return np.mean(np.sqrt(np.square(self.px) + np.square(self.py) + np.square(self.pz)))

    @property
    def vx(self):
        if RELATIVISTIC:
            return CLIGHT * self.px / np.sqrt(np.square(self.px) + 1.0)
        else:
            return CLIGHT * self.px

    @property
    def vy(self):
        if RELATIVISTIC:
            return CLIGHT * self.py / np.sqrt(np.square(self.py) + 1.0)
        else:
            return CLIGHT * self.py

    @property
    def vz(self):
        if RELATIVISTIC:
            return CLIGHT * self.pz / np.sqrt(np.square(self.pz) + 1.0)
        else:
            return CLIGHT * self.pz

    def set_p_from_v(self, vx, vy, vz):
        """
        Set velocity data and calculate momenta
        :param vx: velocity component in x direction (m/s)
        :param vy: velocity component in y direction (m/s)
        :param vz: velocity component in z direction (m/s)
        :return:
        """
        if RELATIVISTIC:
            self.px = vx / np.sqrt(np.square(CLIGHT) - np.square(vx))
            self.py = vy / np.sqrt(np.square(CLIGHT) - np.square(vy))
            self.pz = vz / np.sqrt(np.square(CLIGHT) - np.square(vz))
        else:
            self.px = vx / CLIGHT
            self.py = vy / CLIGHT
            self.pz = vz / CLIGHT

        self.recalculate_all()

        return 0

    @property
    def pz_si(self):
        return self.pz * self._species.mass_kg * CLIGHT

    @property
    def xp(self):
        vz = np.ones(self.vz.shape) * EPSILON
        idx = np.where(self.vz > 1e-10)
        vz[idx] = self.vz[idx]
        return self.vx / vz

    @property
    def yp(self):
        vz = np.ones(self.vz.shape) * EPSILON
        idx = np.where(self.vz > 1e-10)
        vz[idx] = self.vz[idx]
        return self.vy / vz

    @property
    def centroid(self):
        return np.array([self._xm, self._ym, self._zm])

    def calculate_mean_r(self):
        self._xm = np.mean(self.x)
        self._ym = np.mean(self.y)
        self._zm = np.mean(self.z)

    def calculate_mean_pr(self):
        self._pxm = np.mean(self.px)
        self._pym = np.mean(self.py)
        self._pzm = np.mean(self.pz)

    @property
    def ekin_mev(self):
        m_mev = self._species.mass_mev

        if Z_ENERGY:
            pr = self.pz
        else:
            pr = np.sqrt(np.square(self.px) + np.square(self.py) + np.square(self.pz))

        if RELATIVISTIC:
            _ekin = np.sqrt(np.square(pr * m_mev) + np.square(m_mev)) - m_mev
        else:
            _ekin = m_mev * pr**2.0 / 2.0
            
        return _ekin
            
    def calculate_mean_energy_mev(self):
        self._ekin_mean = np.mean(self.ekin_mev)
        self._ekin_stdev = np.std(self.ekin_mev)

    def calculate_stdevs(self):

        x_mean, y_mean, z_mean = self.centroid

        # Calculate standard deviations
        self.x_std = np.std(self.x)  # (m)
        self.y_std = np.std(self.y)  # (m)
        self.z_std = np.std(self.z)  # (m)

        xp_mean = np.mean(self.xp)  # (rad)
        yp_mean = np.mean(self.yp)  # (rad)

        self.xp_std = np.std(self.xp)  # (rad)
        self.yp_std = np.std(self.yp)  # (rad)

        self.xxp_std = np.mean((self.x - x_mean) * (self.xp - xp_mean))  # (m * rad)
        self.yyp_std = np.mean((self.y - y_mean) * (self.yp - yp_mean))  # (m * rad)
        self.xyp_std = np.mean((self.x - x_mean) * (self.yp - yp_mean))  # (m * rad)
        self.yxp_std = np.mean((self.y - y_mean) * (self.xp - xp_mean))  # (m * rad)

        return 0

    def get_beam_edges(self, mode="1rms"):

        x_mean, y_mean, z_mean = self.centroid

        if mode == "1rms":
            return np.array([x_mean - self.x_std, x_mean + self.x_std,
                             y_mean - self.y_std, y_mean + self.y_std,
                             z_mean - self.z_std, z_mean + self.z_std])

        elif mode == "2rms":
            return np.array([x_mean - 2.0 * self.x_std, x_mean + 2.0 * self.x_std,
                             y_mean - 2.0 * self.y_std, y_mean + 2.0 * self.y_std,
                             z_mean - 2.0 * self.z_std, z_mean + 2.0 * self.z_std])

        elif mode == "full":
            return np.array([np.min(self.x), np.max(self.x),
                             np.min(self.y), np.max(self.y),
                             np.min(self.z), np.max(self.z)])

    def get_emittances(self, normalized=True):

        e_xxp_1rms = np.sqrt(np.square((self.x_std * self.xp_std)) - np.square(self.xxp_std))
        e_yyp_1rms = np.sqrt(np.square((self.y_std * self.yp_std)) - np.square(self.yyp_std))
        e_xyp_1rms = np.sqrt(np.square((self.x_std * self.yp_std)) - np.square(self.xyp_std))
        e_yxp_1rms = np.sqrt(np.square((self.y_std * self.xp_std)) - np.square(self.yxp_std))

        if normalized:
            return np.array([e_xxp_1rms, e_yyp_1rms, e_xyp_1rms, e_yxp_1rms]) * self.mean_momentum_betagamma
        else:
            return np.array([e_xxp_1rms, e_yyp_1rms, e_xyp_1rms, e_yxp_1rms])

    def get_twiss_parameters(self):

        e_xxp_1rms, e_yyp_1rms, _, _ = self.get_emittances(normalized=False)

        # Twiss Parameters
        beta_x = (self.x_std ** 2.0) / e_xxp_1rms
        gamma_x = (self.xp_std ** 2.0) / e_xxp_1rms

        if self.xxp_std < 0:

            alpha_x = np.sqrt(beta_x * gamma_x - 1.0)

        else:

            alpha_x = -np.sqrt(beta_x * gamma_x - 1.0)

        beta_y = (self.y_std ** 2.0) / e_yyp_1rms
        gamma_y = (self.yp_std ** 2.0) / e_yyp_1rms

        if self.yyp_std < 0:

            alpha_y = np.sqrt(beta_y * gamma_y - 1.0)

        else:

            alpha_y = -np.sqrt(beta_y * gamma_y - 1.0)

        # Number of particles inside 4-RMS emittance ellipse
        e_xxp_4rms_includes = len(np.where(
            gamma_x * self.x ** 2.0
            + 2.0 * alpha_x * self.x * self.xp
            + beta_x * self.xp ** 2.0 < 4.0 * e_xxp_1rms)[0])
        e_yyp_4rms_includes = len(np.where(
            gamma_y * self.y ** 2.0
            + 2.0 * alpha_y * self.y * self.yp
            + beta_y * self.yp ** 2.0 < 4.0 * e_yyp_1rms)[0])

        e_xxp_4rms_includes_perc = 100.0 * e_xxp_4rms_includes / self.numpart  # percent
        e_yyp_4rms_includes_perc = 100.0 * e_yyp_4rms_includes / self.numpart  # percent

        print("4-RMS emittances include {} and {} percent "
              "of the beam in x and y direction".format(e_xxp_4rms_includes_perc, e_yyp_4rms_includes_perc))

        return np.array([alpha_x, beta_x, gamma_x,
                         alpha_y, beta_y, gamma_y])

        # # Maximum emittance by using the RMS Twiss parameters and the maximum values of x, x', y and y'
        # e_xxp_full = twiss_gamma_x * np.max(self.x) ** 2.0 + 2.0 * twiss_alpha_x * np.max(self.x) * np.max(
        #     self.xp) + twiss_beta_x * np.max(self.xp) ** 2.0
        # e_yyp_full = twiss_gamma_y * np.max(self.y) ** 2.0 + 2.0 * twiss_alpha_y * np.max(self.y) * np.max(
        #     self.yp) + twiss_beta_y * np.max(self.yp) ** 2.0

    @property
    def v_mean_m_per_s(self):

        if Z_ENERGY:
            v = self.vz
        else:
            v = np.sqrt(np.square(self.vx) + np.square(self.vy) + np.square(self.vz))

        return np.mean(v)

    @property
    def v_mean_cm_per_s(self):
        return self.v_mean_m_per_s * 1.0e2

    @property
    def mean_b_rho(self):
        """
        TODO: include non-relativistic case
        :return:
        """
        if Z_ENERGY:
            pr = self.pz
        else:
            pr = np.sqrt(np.square(self.px) + np.square(self.py) + np.square(self.pz))

        return np.mean(pr) * self.species.mass_mev * 1.0e6 / (self.species.q * CLIGHT)

    def set_z_momentum_from_b_rho(self, b_rho):
        """
        Sets all pz entries to correspond to the given b-rho
        TODO: In the future, maybe we toggle keeping the momentum spread? -DW
        """

        self.pz = np.ones(self.pz.shape[0]) * b_rho * self.species.q * CLIGHT / 1.0e6 / self.species.mass_mev

        self.recalculate_all()

        return self.mean_energy_mev

    def set_mean_energy_z_mev(self, energy):
        """
        Sets the mean energy of the distribution to the given value in MeV.
        Currently this only affects the z direction regardless of global variable Z_ENERGY setting.

        TODO: This should be generalized -DW

        :param energy:
        :return:
        """
        ekin_mev = self.ekin_mev - self.mean_energy_mev + energy

        mysign = np.sign(ekin_mev)
        gamma = np.abs(ekin_mev) / self.species.mass_mev + 1.0
        beta = np.sqrt(1.0 - gamma**(-2.0))

        self.pz = mysign * beta * gamma

        # self.pz -= self._pzm
        # self.pz += gamma * beta

        self.recalculate_all()

    def load_from_aima(self, x, xp, y, yp, phi, energy, freq, charge=1.0):
        """
        Import from AIMA lst file with the following units:
        :param x: particle position in cm
        :param xp: particle angle in rad
        :param y: particle position in cm
        :param yp: particle angle in rad
        :param phi: particle phase in deg
        :param energy: particle energy in MeV
        :param freq: bunch frequency in Hz
        :param charge: bunch charge in C
        :return:
        """

        # TODO: include non-relativistic case

        assert len(x) == len(xp) == len(y) == len(yp) == len(phi) == len(energy), \
            "All input arrays must be of same length!"

        self.x = x * 1.0e-2  # m
        self.y = y * 1.0e-2  # m

        gamma = energy / self.species.mass_mev + 1.0
        beta = np.sqrt(1.0 - gamma ** (-2.0))
        beta_lambda = beta / freq * CLIGHT

        self.z = phi * beta_lambda / 360.0  # m

        self.pz = gamma * beta
        self.px = xp * self.pz
        self.py = yp * self.pz

        # print(self.x.shape)
        # print(self.y.shape)
        # print(self.z.shape)
        # print(self.px.shape)
        # print(self.py.shape)
        # print(self.pz.shape)

        self.q = charge * np.ones(self.x.shape)

        self.recalculate_all()


if __name__ == '__main__':
    pass
