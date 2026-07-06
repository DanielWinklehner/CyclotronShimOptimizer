"""View the cyclotron geometry in the PyVista viewer (no field solve).

Replaces:  python main.py --geo_test --config <yml>
Run:       [mpiexec -n N] python examples/view_geometry.py [config.yml]
"""

import os
import sys

import cyclotron_optimizer as co

CONFIG = (sys.argv[1] if len(sys.argv) > 1 else
          os.path.join(os.path.dirname(__file__), "config_muon_smaller.yml"))

with co.Session(CONFIG) as s:
    s.view_geometry()
