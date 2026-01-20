from global_variables import *

__author__ = "Daniel Winklehner, Philip Weigel"
__doc__ = "Simple class to hold and calculate particle data like mass, charge, etc."


PRESETS = {'proton': {'latex_label': r"$\mathrm{p}^+$",
                      'mass_mev': PMASS_MEV,
                      'a': 1.00727647,
                      'z': 1.0,
                      'q': 1.0},
           'H_1-': {'latex_label': r"$\mathrm{H}^1-$",
                    'mass_mev': 1.00837361135 * AMU_MEV,
                    'a': 1.00837361135,
                    'z': 1.0,
                    'q': 1.0},
           'electron': {'latex_label': r"$\mathrm{e}^-$",
                        'mass_mev': EMASS_MEV,
                        'a': EMASS_MEV / AMU_MEV,
                        'z': 0.0,
                        'q': -1.0},
           'muon': {'latex_label': r"$\mu^-$",
                    'mass_mev': MUMASS_MEV,
                    'a': MUMASS_MEV / AMU_MEV,
                    'z': 0.0,
                    'q': -1.0},
           'H2_1+': {'latex_label': r"$\mathrm{H}_2^+$",
                     'mass_mev': 2.01533 * AMU_MEV,
                     'a': 2.01533,
                     'z': 2.0,
                     'q': 1.0},
           '4He_2+': {'latex_label': r"$^4\mathrm{He}^{2+}$",
                      'mass_mev': 4.001506179127 * AMU_MEV,
                      'a': 4.001506179127,
                      'z': 2.0,
                      'q': 2.0}}


class IonSpecies(object):

    def __init__(self,
                 name,
                 label=None,
                 latex_label=None,
                 a=None,
                 z=None,
                 q=None):

        """

        Simple ion species class that holds data and can calculate some basic values.

        :param name: Name of the species, can be one of the presets:
            'protons'
            'electrons'
            'H_1-'
            'H2_1+'
            '4He_2+'

            if it is not a preset, a, z and q have to be defined as well!

        :param label: A plain text label for plotting, defaults to name.
        :param latex_label: A label used in matplotlib, can be in latex shorthand, defaults to name.
        :param a: atomic (molecular) mass in amu
        :param z: number of protons
        :param q: charge state
        """

        # Check if user wants a preset ion species:
        if name in PRESETS.keys():

            species = PRESETS[name]

            # Override label and latex_label if not given
            if label is None:
                label = name
            if latex_label is None:
                latex_label = species["latex_label"]

            mass_mev = species["mass_mev"]
            z = species["z"]
            a = species["a"]
            q = species["q"]

            if DEBUG:
                print("Using preset ion species '{}' with label '{}':".format(name, label))

        # If not, check for missing initial values
        else:

            init_values = [a, z, q]

            if None in init_values:

                print("Sorry, ion species {} was initialized with missing values!".format(name))
                print("a = {}, z = {}, q = {}". format(a, z, q))
                exit(1)

            else:

                if DEBUG:
                    print("Using user defined ion species {}:".format(name))

        # Initialize values (default for a proton)
        self._name = name

        if latex_label is None:
            self._latex_label = name
        else:
            self._latex_label = latex_label

        if label is None:
            self._label = name
        else:
            self._label = label

        self._mass_mev = a * AMU_MEV  # Rest Mass (MeV/c^2)
        self._a = a                   # Mass number A of the ion (amu)
        self._z = z                   # Proton number Z of the ion (unitless)
        self._q = q                   # charge state

        # Calculate mass of the particle in kg
        self._mass_kg = self._mass_mev * ECHARGE * 1.0e6 / CLIGHT**2.0

    def __str__(self):
        return "Ion Species {} with label {}:\n" \
               "A = {}, Z = {}, q = {}\n" \
               "M_0 = {} MeV/c^2 = {} kg, Q = {} C\n" \
               "(For OPERA: M_0 = {} * M_electron)".format(self._name,
                                                           self._label,
                                                           self._a, self._z, self._q,
                                                           self._mass_mev, self._mass_kg, self._q * ECHARGE,
                                                           self._mass_mev / EMASS_MEV)

    @property
    def mass_electron_multiple(self):
        return self._mass_mev / EMASS_MEV

    @property
    def mass_mev(self):
        return self._mass_mev

    @property
    def mass_kg(self):
        return self._mass_kg

    @property
    def mass_opera(self):
        return self._mass_mev / EMASS_MEV

    @property
    def q_over_a(self):
        """
        :return: charge state - to - mass number ratio (unitless) 
        """
        return self._q / self._a

    @property
    def q_over_m(self):
        """
        :return: charge - to - mass ratio (C/kg) 
        """
        return self._q * ECHARGE / self._mass_kg

    @property
    def label(self):
        return self._label

    @property
    def latex_label(self):
        return self._latex_label

    @property
    def name(self):
        return self._name

    @property
    def q(self):
        return self._q

    @property
    def charge(self):
        return self._q * ECHARGE

    @property
    def a(self):
        return self._a


if __name__ == '__main__':
    pass
