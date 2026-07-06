"""Solve the configured machine, print the isochronism summary, and show the
model with the median-plane field overlaid.

Run:  [mpiexec -n N] python examples/solve_and_plot.py [config.yml]
"""

import os
import sys

import cyclotron_optimizer as co

CONFIG = (sys.argv[1] if len(sys.argv) > 1 else
          os.path.join(os.path.dirname(__file__), "config_muon_smaller.yml"))

with co.Session(CONFIG) as s:
    # use_gpu: bool for all stages, or per-stage control, e.g.
    #   use_gpu={"assembly": True, "relaxation": True, "field": False}
    # (stages: interaction-matrix assembly / relaxation / field evaluation)
    model = s.build(use_gpu=True)
    model.solve()

    iso = model.isochronism()  # method from config (circle/gordon/seo)
    fmap = model.median_plane_field(resolution_mm=2.0, gpu_precision="single")

    if s.is_root:
        print(f"\nConverged: {model.converged} (misfit {model.misfit:.2e})")
        print(f"Isochronism ({iso['method']}): mean = {iso['mean_freq_mhz']:.4f} MHz, "
              f"std = {iso['std_dev_mhz']:.5f} MHz ({iso['percent_dev']:.3f} %)")

    model.show(field=fmap)  # collective (viewer opens on rank 0)
