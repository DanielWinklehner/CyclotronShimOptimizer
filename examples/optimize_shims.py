"""Run the DFO-LS shim optimization, then solve and display the optimum.

Optimization hyperparameters come from the config's optimization section.

Run:  [mpiexec -n N] python examples/optimize_shims.py [config.yml]
"""

import os
import sys

import cyclotron_optimizer as co
from cyclotron_optimizer.geometry.pole_shape import PoleShape
from cyclotron_optimizer.optimization.optimizer import CyclotronOptimizer

CONFIG = (sys.argv[1] if len(sys.argv) > 1 else
          os.path.join(os.path.dirname(__file__), "config_muon_smaller.yml"))

with co.Session(CONFIG) as s:
    radii_mm = s.default_radii_mm().tolist()

    optimizer = CyclotronOptimizer(s.config, radii_mm, comm=s.comm, rank=s.rank,
                                   verbosity=s.verbosity)
    result = optimizer.optimize()

    best_shape = PoleShape(s.config.side_shim.num_rad_segments,
                           side_offsets=result['best_side_shims'],
                           top_offsets=result['best_top_shims'])

    model = s.build(pole_shape=best_shape, coil_current=result['optimal_coil'])
    model.solve()
    iso = model.isochronism()
    fmap = model.median_plane_field(resolution_mm=2.0, gpu_precision="single")

    if s.is_root:
        print(f"\nOptimal coil current: {result['optimal_coil']:.1f} A")
        print(f"Isochronism at optimum: std = {iso['std_dev_mhz']:.5f} MHz "
              f"({iso['percent_dev']:.3f} %)")

    model.show(field=fmap)  # collective (viewer opens on rank 0)
