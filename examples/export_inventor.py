"""Export the shimmed pole as an Autodesk Inventor VBA macro (script-level).

Inventor export is an explicit script call, not a config flag. Pass a
specific PoleShape to export a particular shim solution.

Run:  python examples/export_inventor.py [config.yml]
"""

import os
import sys

import cyclotron_optimizer as co
from cyclotron_optimizer.geometry.inventor_export import InventorPoleExporter

CONFIG = (sys.argv[1] if len(sys.argv) > 1 else
          os.path.join(os.path.dirname(__file__), "config_muon_smaller.yml"))

with co.Session(CONFIG) as s:
    shape = s.default_pole_shape()
    exporter = InventorPoleExporter(s.config, verbosity=1)
    out = exporter.export_macro(pole_shape=shape,
                                output_path="output/cyclotron_pole.bas")
    if s.is_root:
        print(f"\nExported {out}")
