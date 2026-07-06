"""Export the conforming iron geometry as STEP files (script-level action).

STP export is an explicit script call, not a config flag. The fragmented
STEP carries the pole/lid/yoke contacts pre-imprinted -- exactly the solids
the mesh is built from -- for external tools (COMSOL gold-standard runs, CAD
inspection). Pass a specific PoleShape to export a particular shim solution.

Run:  python examples/export_stp.py [config.yml]
"""

import os
import sys

import cyclotron_optimizer as co
from cyclotron_optimizer.geometry.geometry import (export_component_stp,
                                                   export_iron_stp)

CONFIG = (sys.argv[1] if len(sys.argv) > 1 else
          os.path.join(os.path.dirname(__file__), "config_muon_smaller.yml"))

with co.Session(CONFIG) as s:
    shape = s.default_pole_shape()
    export_iron_stp(s.config, "output/iron_conforming.stp", pole_shape=shape)
    export_component_stp(s.config, "pole", "output/pole_shimmed.stp",
                         pole_shape=shape)
    if s.is_root:
        print("\nExported output/iron_conforming.stp and output/pole_shimmed.stp")
