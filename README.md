# Cyclotron Optimizer

A library-first framework for compact-cyclotron magnet design: GPU-accelerated
[RadiaCUDA](https://github.com/DanielWinklehner/RadiaCUDA) magnetostatics,
component-based geometry (gmsh-OCC / STP tet meshing), symmetry-exploiting
field evaluation, isochronism analysis (circle / Gordon / SEO tracking), and
DFO-LS shim optimization.

Written for the [IsoDAR](https://www.nevis.columbia.edu/isodar/) project.
Currently Windows-only (tested on Win 11) with a single CUDA GPU.

## Quick start

The package installs editable into the `radiacuda2` conda environment (which
provides radia/RadiaCUDA, gmsh, cupy, mpi4py, PyPATools, PyRadia):

```bash
pip install -e . --no-deps
```

Project scripts import the package; the runtime lifecycle (radia/MPI init
ordering, environment quirks) is handled by the `Session` facade:

```python
import cyclotron_optimizer as co

with co.Session("examples/config_muon_smaller.yml") as s:
    model = s.build()                 # defaults (shims, radii, current) from config
    model.solve()                     # mesh + assemble + relax (GPU)
    iso = model.isochronism()         # circle / gordon / seo per config
    fmap = model.median_plane_field(resolution_mm=2.0, gpu_precision="single")
    if s.is_root:
        print(f"mean f = {iso['mean_freq_mhz']:.4f} MHz "
              f"({iso['percent_dev']:.3f} % dev)")
        fmap.save("output/midplane.comsol")
    model.show(field=fmap)            # PyVista viewer + field overlay
```

Ready-made workflow scripts live in `examples/` (each takes an optional
config path argument):

```bash
python examples/view_geometry.py        # geometry only, no solve
python examples/solve_and_plot.py       # solve + isochronism + field overlay
python examples/export_fieldmaps.py     # midplane + 3D bore field maps
python examples/optimize_shims.py       # DFO-LS shim optimization
```

The legacy all-in-one flow (full analysis + plots + comparison maps) remains
available as `python -m cyclotron_optimizer.api --config <yml> [--optimize]`.

**MPI contract**: every `Session`/`CyclotronModel` method is collective and
returns data on rank 0 only -- write scripts top-to-bottom as if
single-process and guard only prints/saves with `session.is_root`. Note that
on a single-GPU machine `mpiexec` does NOT speed up GPU solves (radia's
relaxation runs on rank 0; extra ranks idle); multi-rank helps only the
CPU-cluster mode (`use_gpu={"assembly": False, ...}` distributes the
interaction-matrix assembly).

## Machine configuration (component-based)

A machine is a list of components with named materials and symmetries
(`examples/config_muon_smaller.yml` is the reference):

```yaml
materials:
  iron: {type: bh_file, file: "../resources/dillinger_steel.csv"}

symmetries:
  cyclotron_8fold:
    - [perp, [0, 0, 0], [1, -1, 0]]
    - [perp, [0, 0, 0], [1, 0, 0]]
    - [perp, [0, 0, 0], [0, 1, 0]]
    - [para, [0, 0, 0], [0, 0, 1]]

components:
  - {name: yoke, kind: stp, file: "../resources/uCyclo_v2_YokeWall.stp",
     material: iron, symmetry: cyclotron_8fold, mesh: {max_size: 50}}
  - {name: pole, kind: pole, material: iron, symmetry: cyclotron_8fold,
     shimmed: true, mesh: {max_size: 50},
     params: {inner_radius_mm: 50.0, outer_radius_mm: 400.0, height_mm: 230.0,
              half_angle_deg: 10.0, pole_zs: -285.0}}
  - {name: coils, kind: racetrack_pair, symmetry: cyclotron_8fold,
     params: {radius_min_mm: 460.0, radius_max_mm: 574.5, height_mm: 123.5,
              midplane_dist: 55, current_A: 15368, num_segments: 25}}
```

- `kind` selects a registered builder (`stp`, `wedge`, `lid_upper`, `pole`,
  `wedge_pair`, `racetrack_pair`); adding a part is a YAML entry plus, at
  most, one new builder in `geometry/geometry.py`.
- `symmetry` names the component's FIELD symmetry: applied as radia
  `TrfZer*` mirrors for magnetized parts, only DECLARED (metadata for the
  field-map folding) for current sources. Components with `symmetry: null`
  (e.g. the extraction channel) are evaluated unfolded automatically.
- **Coils are always built full-size** (both coils of the pair): radia
  symmetry transforms on current sources force a full CPU fallback on the
  GPU field path (RadiaCUDA issue #16).
- `shimmed: true` marks the pole whose shape follows the `PoleShape` shim
  offsets; the solver rebuilds only it (and the coils) per optimizer iterate,
  reusing the static yoke/lids meshes.
- Relative file paths resolve against the YAML's own directory, so a config
  travels with its project folder.
- Legacy fixed-section configs still load (adapted internally to the same
  component description).

Workflow settings (field evaluation radii/methods, relaxation precision,
optimizer hyperparameters, visualization) live in the remaining config
sections; most have per-call overrides on the `CyclotronModel` methods.

## GPU / precision control

`use_gpu` is accepted everywhere as a bool, or granular per stage:

```python
model = s.build(use_gpu={"assembly": True,     # interaction matrix (RlxPre)
                         "relaxation": True,   # RlxAuto method 9 (CUDA) vs 4
                         "field": True})       # rad.Fld evaluations
```

Field maps additionally accept `gpu_precision="single"` (fp32 kernel,
~16x faster, visualization-grade ~1e-4; keep the default `"double"` for
anything feeding tracking).

## B-H curves

BH curves load from CSV (comma-separated, units T (mu_0*H) and T) referenced
by the config's `materials:` section; `resources/` ships the BH curves (`dillinger_steel.csv`, `COMSOL_1010_BH_T_A-m.csv`).
A `sat_iso_frm` material type exposes Radia's `MatSatIsoFrm` formula
alternative.

## Repository layout

```
cyclotron_optimizer/     the installable package
  session.py             Session / CyclotronModel facade
  api.py                 legacy full-analysis workflow + CLI
  config_io/             two-schema config loading (components + legacy)
  geometry/              component builders, symmetry algebra, gmsh meshing
  simulation/            field evaluation/export, solver, diagnostics
  optimization/          DFO-LS / Nelder-Mead shim optimization
  visualization/         plots, field maps, PyVista overlays
examples/                machine configs + workflow scripts
test/                    test suite (python test/run_tests.py)
scripts/                 one-off verification/dev scripts
resources/               STP geometry files
resources/               STP geometry + BH curve data
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: No module named 'radia'` | Run inside the `radiacuda2` env (RadiaCUDA installs `radia.pyd`) |
| Radia convergence failures | Increase `simulation.iterations`, relax `precision` |
| `rad.Fld` unexpectedly slow | Check for the CPU-fallback warning (GPU gate accepts only `b`/`bx`/`by`/`bz`) |
| `mpiexec -n N` not faster | Expected for GPU solves (rank-0 relaxation); use ranks for CPU assembly only |
| MPI rank sync errors | All ranks must load the identical config |

## Citation

```bibtex
@software{cyclotron_optimizer_2026,
  title={Cyclotron Optimizer: GPU-Accelerated Cyclotron Magnet Design Framework},
  author={Daniel Winklehner},
  year={2026},
  url={https://github.com/DanielWinklehner/CyclotronShimOptimizer}
}
```

## Contact & Support

- **Issues**: GitHub Issues tracker
- **Email**: winklehn@mit.edu
