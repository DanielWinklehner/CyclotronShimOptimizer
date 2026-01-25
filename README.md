# Cyclotron Optimizer

A three-phase optimization framework for cyclotron magnet design, featuring MPI parallelization, 
[Radia](https://github.com/ochubar/Radia)  magnetostatics integration, and component-based geometry generation with intelligent radial segmentation.
This program can calculate the magnetic field for a given geometry (only round, 4-hill isochronous cyclotrons for now)
display magnetic fields and isochronism and optimize the shims (top and side) for maximum isochronism.

This code was written for the [IsoDAR](https://www.nevis.columbia.edu/isodar/) project.

Note: Currently this code only runs in Windows (tested on Win 11).

## Overview

This project optimizes cyclotron pole geometry to achieve isochronism (frequency flatness across the acceleration radius) by intelligently varying:

- **Top shims**: Vertical displacement at each radius (Phase 1)
- **Side shims**: Angular displacement at each radius (Phase 2)  
- **Coil current**: RF drive amplitude for target frequency (Phase 3)

Advanced features include:
- **Magnetization caching**: Warm-starts subsequent solves via Radia's interaction matrix reuse
- **MPI parallelization**: Distributed objective function evaluation
- **Component-based geometry**: Modular architecture with automatic segmentation handling
- **Real-time visualization**: Live optimization progress plots

## Requirements

### System Requirements
- Python 3.8.x (Note: 3.8 is mandatory for now as the radia binary was compiled for 3.8)
- Radia (magnetostatics solver) - included in `/radialib`
- MPI runtime (optional, for parallel execution)

### Python Dependencies

```
numpy~=1.24.4
scipy~=1.10.1
matplotlib~=3.7.3
sympy~=1.13.3
tqdm~=4.67.1
scikit-optimize~=0.10.2
mpi4py~=4.0.0 (optional, for MPI support)
yaml~=0.2.5
pyyaml~=6.0.2
```

### Installation

```bash
# Clone repository
git clone https://github.com/DanielWinklehner/CyclotronShimOptimizer.git
cd cyclotron_optimizer

# Install dependencies
pip install -r requirements.txt
```
OR

Using Anaconda (recommended): 
```bash
conda env create -n <env_name> -f environment.yml
```

## Capabilities

### Optimization Features

- **Three-Phase Workflow**
  - Phase 1: Optimize top shims for frequency flatness
  - Phase 2: Optimize side shims (top shims fixed) for improved flatness
  - Phase 3: Optimize coil current for target RF frequency

- **Multi-Start Nelder-Mead**
  - Configurable number of random restarts per phase
  - Early stopping via plateau detection
  - Normalized parameter space [0, 1]
  - Bounded optimization with physical constraints

- **Objective Function**
  - Flatness metric: `σ(frequencies)` across acceleration radius
  - Regularization term: `λ × ||x_norm||_L2` to minimize excessive shimming
  - MPI-distributed evaluation across worker ranks

### Geometry Features

- **Component-Based Design**
  - Modular components: yoke, lids, pole base, shims, coils
  - Automatic material assignment and drawing attributes
  - Symmetry operations: 8-fold (4-fold in-plane + 2-fold z)

- **Intelligent Radial Segmentation**
  - Interpolation-based workaround for Radia's segmentation limitations
  - Per-radius elevation and angular offset support
  - Fine mesh generation at optimization scale

- **Complex Geometry Handling**
  - Angled surface cuts for sloped shimming
  - Boundary segment detection for mismatched arc lengths
  - Degenerate polygon filtering (< 0.001 mm²)
  - 3D plane/line intersections via SymPy

### Magnetization Caching

- **Automatic Cache Management**
  - Parameter signature hashing (MD5) for validity checking
  - Tracks: coil current, yoke geometry, lid geometry, pole base geometry
  - Stale cache automatic invalidation

- **Warm-Start Solving**
  - Extract yoke magnetizations after base solve
  - Use as external field source in subsequent solves
  - Radia `RlxAuto(..., 'ZeroM->False')` prevents matrix reset
  - 2-5× speedup on convergence

## Core Implementation

### Architecture

```
┌─ config.yml (CyclotronConfig)
│
├─ main.py
│  └─ Timer, MPI initialization
│
├─ CyclotronOptimizer (optimizer.py)
│  ├─ Phase 1: optimize_phase(top)
│  ├─ Phase 2: optimize_phase(side)
│  └─ Phase 3: optimize_coil_final()
│
├─ evaluate_cyclotron_objective_simplified() (objective_function.py)
│  ├─ Reconstruct pole geometry
│  ├─ Solve magnetostatics (all ranks)
│  ├─ Calculate revolution frequencies (rank 0)
│  └─ Compute flatness + regularization
│
├─ evaluate_radii_parallel() (field_calculator.py)
│  ├─ Build geometry (build_geometry)
│  ├─ Create interaction matrix (rad.RlxPre)
│  ├─ Solve with warm-start (rad.RlxAuto)
│  └─ Query B-field at all radii (single rad.Fld call)
│
├─ RadiaCache (magnetization_cache.py)
│  ├─ Compute parameter signature
│  ├─ Save/load pickle files
│  └─ Apply cached magnetizations
│
└─ GeometricComponent hierarchy (components.py)
   ├─ AnnularWedgeComponent (yoke, lids, pole)
   ├─ LidUpperComponent (tapered upper lid)
   ├─ TopShimComponent (top surface shims)
   ├─ SideShimComponent (side surface shims)
   └─ CoilComponent (RF coils)
```

## Usage Example

### Running Optimization

**2. Single-process (no MPI):**

```bash
python main.py --optimize --verbosity 1
```

**3. With MPI (4 processes):**

```bash
mpiexec -n 4 python main.py --optimize --verbosity 1
```

**4. With magnetization caching (first run):**

```bash
mpiexec -n 4 python main.py --optimize --cached --verbosity 1
```

This creates cache on first run; subsequent runs use warm-start.

**5. Geometry visualization only:**

```bash
python main.py --geo_test --verbosity 2
```

Opens OpenGL viewer of current geometry.

### Output Files

```
output/
├── radia_cache/
│   └── magnetizations_{MD5_SIGNATURE}.pkl
├── cyclotron_pole.txt
├── optimization_diagnostics_{TIMESTAMP}.csv
└── plots/
    ├── isochronism_results.png
    └── frequency_deviation.png
```

**Diagnostics CSV columns:**
```
phase, iteration, multistart, nelder_iter, avg_frequency_mhz, 
flatness, regularization, objective, side_param_0, ..., top_param_0, ...
```

## Python API Example

```python
from config_io.config import CyclotronConfig
from geometry.pole_shape import PoleShape
from geometry.geometry import build_geometry
from simulation.field_calculator import evaluate_radii_parallel
import numpy as np

# Load configuration
config = CyclotronConfig.from_yaml('config.yml')

# Create pole geometry with custom shims
pole_shape = PoleShape(
    n_segments=14,
    side_offsets=np.array([5.0] * 15),  # 5° at all radii
    top_offsets=np.array([7.0] * 15)     # 7 mm at all radii
)

# Build geometry (rank 0, no MPI)
cyclotron_id, pole_id = build_geometry(
    config, 
    pole_shape,
    rank=0,
    verbosity=1
)

# Evaluate B-field
radii_mm = np.linspace(50, 400, 50)
radii_out, bz_values, converged = evaluate_radii_parallel(
    config,
    pole_shape,
    radii_mm.tolist(),
    rank=0,
    use_cache=False
)

print(f"Convergence: {converged}")
print(f"B-field range: {min(bz_values):.4f} to {max(bz_values):.4f} T")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: No module named 'radia'` | Check `RADIA_PATH` in `main.py`, verify radialib present |
| Radia convergence failures | Increase `simulation.iterations`, reduce `precision` |
| Memory exhaustion | Reduce `segmentation` factors, fewer `field_evaluation.n_eval_pts` |
| Slow optimization | Enable `--cached` flag, reduce `n_initial_points` |
| MPI rank sync errors | Ensure all ranks have identical config.yml |

## Citation

If you use this optimizer in published work, please cite:

```bibtex
@software{cyclotron_optimizer_2024,
  title={Cyclotron Optimizer: MPI-Parallel Magnet Design Framework},
  author={Daniel Winklehner},
  year={2026},
  url={https://github.com/DanielWinklehner/CyclotronShimOptimizer}
}
```

## Contact & Support

For issues, questions, or contributions:
- **Issues**: GitHub Issues tracker
- **Email**: [winklehn@mit.edu]

---

**Last Updated**: January 22, 2026  
