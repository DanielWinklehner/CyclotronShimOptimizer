"""Solve and export field maps (midplane .comsol + 3D bore field).

Export resolutions/limits come from the config's field_evaluation section;
paths can be overridden here. Full precision (fp64) is used for exported
maps -- they feed tracking.

Run:  [mpiexec -n N] python examples/export_fieldmaps.py [config.yml]
"""

import os
import sys

import cyclotron_optimizer as co

CONFIG = (sys.argv[1] if len(sys.argv) > 1 else
          os.path.join(os.path.dirname(__file__), "config_muon_smaller.yml"))

with co.Session(CONFIG) as s:
    model = s.build()
    model.solve()

    model.save_median_plane_field("output/midplane_field.comsol")
    # model.save_bore_field("output/bore_field.comsol")

    if s.is_root:
        print("\nExported output/midplane_field.comsol and output/bore_field.comsol")
