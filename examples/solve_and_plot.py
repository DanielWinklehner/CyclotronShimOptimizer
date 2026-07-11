"""Solve the configured machine, print the isochronism summary, show the
isochronism / energy / flutter / tune plots, and open the 3D model with the
median-plane field overlaid.

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

    # iso = model.isochronism()  # method from config (circle/gordon/seo)
    fmap = model.median_plane_field(resolution_mm=2.0, gpu_precision="single")

    # if s.is_root:
    #     print(f"\nConverged: {model.converged} (misfit {model.misfit:.2e})")
    #     print(f"Isochronism ({iso['method']}): mean = {iso['mean_freq_mhz']:.4f} MHz, "
    #           f"std = {iso['std_dev_mhz']:.5f} MHz ({iso['percent_dev']:.3f} %)")
    #
    #     # 4-panel design summary: <Bz>(r)+Energy(r), revolution frequency(r) vs
    #     # target, betatron tunes (nu_r/nu_z/Walkinshaw) and flutter F(r). The
    #     # tune/flutter panels need iso_method: gordon (else they show a note).
    #     import matplotlib.pyplot as plt
    #     from cyclotron_optimizer.visualization.plots import plot_final_summary
    #
    #     os.makedirs("output", exist_ok=True)
    #     plot_final_summary(
    #         model.radii_mm,
    #         iso["bz_for_plot"],
    #         iso["energies_mev"],
    #         iso["rev_frequencies_mhz"],
    #         tunes=iso["tunes"],
    #         target_freq_mhz=s.config.optimization.target_frequency_mhz,
    #         title=f"Design summary: {s.config.particle_species} ({iso['method']})",
    #         savepath=os.path.join("output", "design_summary.png"),
    #         show=False,
    #     )
    #     plt.show()  # blocks until closed; then the 3D viewer opens below

    model.show(field=fmap)  # collective (viewer opens on rank 0)
