from scipy import constants as const
import os
import sys
import shutil
from visualization.colors import MyColors

DEBUG = False
DECIMALS = 12
RELATIVISTIC = True
Z_ENERGY = False  # If true, only the z component is used for beam energy (paraxial approx.)
USE_MULTIPROC = False  # In case we are not using mpi or only using 1 processor, fall back on multiprocessing
ON_WINDOWS = "win" in sys.platform.lower()
ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
MIN_POLYGON_AREA_MM2 = 0.001  # Minimum area in mm²

if ON_WINDOWS:
    LOG_FONT = "Monospac821 BT"
    LOG_FONT_SIZE = 8
else:
    LOG_FONT = "Monospace"
    LOG_FONT_SIZE = 10

# Other variables
COLORS = MyColors()
EPSILON = 1e-10  # A very small number

# --- Set global constants from scipy --- #
CLIGHT = const.speed_of_light
ECHARGE = const.elementary_charge
EPS0 = const.epsilon_0
AMU_MEV = const.value("atomic mass constant energy equivalent in MeV")
EMASS_MEV = const.value("electron mass energy equivalent in MeV")
MUMASS_MEV = const.value("muon mass energy equivalent in MeV")
PMASS_MEV = const.value('proton mass energy equivalent in MeV')

# --- Define factors to go to SI units --- #
kV = 1e3
MV = 1e6
GV = 1e9
kHz = 1e3
MHz = 1e6
km = 1e3
dm = 1e-1
cm = 1e-2
mm = 1e-3
pA = 1e-12
nA = 1e-9
uA = 1e-6
mA = 1e-3
