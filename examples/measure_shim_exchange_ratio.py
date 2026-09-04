"""Measure rho = d<B>/d(side_norm) / d<B>/d(top_norm) -- the shim exchange ratio.

``rho`` is the one number the FIELD-BASIS smoothness penalty needs
(``optimization.smoothness_exchange_ratio``). It says how many NORMALIZED top-shim
units cancel one NORMALIZED side-shim unit in the azimuthally averaged field:
the field-neutral direction is (side, top) = (1, -rho), and that is exactly the
direction the isochronism residual cannot see -- the one the optimizer wanders
along, producing a dip in one shim profile cancelled by a bulge in the other.

Method: central finite differences about a starting design. Perturb ALL stations
of one block by +/- delta (normalized), re-solve, and read the change in the
azimuthally averaged field. Four field solves (2 blocks x 2 signs); no baseline
solve is needed, since a central difference never evaluates the centre. The coil
current is held FIXED (no nested re-match) -- we want the raw field response of
the iron, not the coil-compensated one.

Also reports rho PER STATION-BAND, because the ratio is not constant with radius
(the gap lever saturates where the iron does, the duty-cycle lever does not). Use
the mid-radius value if you are targeting the mid-radius null mode; the spread
tells you how well a single scalar rho can represent the machine.

LINEARIZATION POINT. rho is a local derivative ratio, so measure it where the
design actually sits. Precedence: --from-csv (the best row of an optimization
diagnostics CSV) > the config's saved side_offsets_deg / top_offsets_mm >
mid-range. Beware stations parked ON a bound at the chosen point: np.clip makes
the -delta leg one-sided there, which biases that station's contribution low. The
run prints how many stations are affected.

Run:  mpiexec -n 1 python examples/measure_shim_exchange_ratio.py [config.yml] [delta] \
                                     [--from-csv output/optimization_diagnostics_*.csv]

Defaults: config_muon_smaller.yml, delta = 0.05 (normalized). Needs GPU + a
PROGRAMMATIC pole (pole kind 'pole', not an STP file) or the shims are inert.
"""

import csv
import os
import sys

import numpy as np

import cyclotron_optimizer as co
from cyclotron_optimizer.geometry.pole_shape import PoleShape
from cyclotron_optimizer.simulation.field_calculator import ReusableCyclotronSolver
from cyclotron_optimizer.core.isochronicity import compute_isochronism
from cyclotron_optimizer.core.species import IonSpecies

_argv = list(sys.argv[1:])
FROM_CSV = None
if "--from-csv" in _argv:
    i = _argv.index("--from-csv")
    FROM_CSV = _argv[i + 1]
    del _argv[i:i + 2]

CONFIG = (_argv[0] if len(_argv) > 0 else
          os.path.join(os.path.dirname(__file__), "config_muon_smaller.yml"))
DELTA = float(_argv[1]) if len(_argv) > 1 else 0.05


def best_design_from_csv(path):
    """(side, top) offsets of the MINIMUM-objective row of a diagnostics CSV.

    Rows written before their evaluation completes carry objective = -1 (the
    optimizer pre-writes a placeholder row, then rewrites it), so those are
    skipped rather than taken as a spectacular optimum.
    """
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if float(r["objective"]) >= 0.0]
    if not rows:
        raise ValueError(f"no completed evaluations in {path}")
    best = min(rows, key=lambda r: float(r["objective"]))
    n = sum(1 for k in best if k.startswith("side_param_"))
    side = np.array([float(best[f"side_param_{i}"]) for i in range(n)])
    top = np.array([float(best[f"top_param_{i}"]) for i in range(n)])
    return side, top, float(best["objective"]), float(best["flatness"])


with co.Session(CONFIG) as s:
    cfg = s.config
    radii = s.default_radii_mm().tolist()
    species = IonSpecies(cfg.particle_species)
    opt = cfg.optimization
    n = cfg.side_shim.num_rad_segments + 1

    s_lo, s_hi = opt.side_shim_min_deg, opt.side_shim_max_deg
    t_lo, t_hi = opt.top_shim_min_mm, opt.top_shim_max_mm

    # Linearization point: --from-csv best row > saved config offsets > mid-range
    # (a perturbation about a bound is one-sided and measures a clipped response).
    origin = "mid-range"
    if FROM_CSV:
        side0, top0, best_obj, best_flat = best_design_from_csv(FROM_CSV)
        origin = (f"best row of {os.path.basename(FROM_CSV)} "
                  f"(objective {best_obj:.6f}, flatness {best_flat:.5f} MHz)")
        if side0.size != n:
            raise ValueError(f"CSV has {side0.size} stations, config has {n}")
    else:
        side0 = (np.array(cfg.side_shim.side_offsets_deg, dtype=float)
                 if cfg.side_shim.side_offsets_deg is not None
                 else np.full(n, 0.5 * (s_lo + s_hi)))
        top0 = (np.array(cfg.top_shim.top_offsets_mm, dtype=float)
                if cfg.top_shim.top_offsets_mm is not None
                else np.full(n, 0.5 * (t_lo + t_hi)))
        if (cfg.side_shim.side_offsets_deg is not None
                or cfg.top_shim.top_offsets_mm is not None):
            origin = "config side_offsets_deg / top_offsets_mm"
    coil = opt.reference_coil_current

    solver = ReusableCyclotronSolver(cfg, radii, rank=s.rank, comm=s.comm,
                                     verbosity=s.verbosity)

    def bz_of(side, top):
        """Azimuthally averaged Bz(r) [T] for one shim design at the fixed coil."""
        solver.build(PoleShape(cfg.side_shim.num_rad_segments,
                               side_offsets=np.clip(side, s_lo, s_hi),
                               top_offsets=np.clip(top, t_lo, t_hi)), coil,
                     solve=False)   # resolve_at_current below does THE solve
        _, bz, converged, misfit = solver.resolve_at_current(coil)
        iso = compute_isochronism(cfg.field_evaluation.iso_method, bz, radii,
                                  cfg, species, rank=s.rank, comm=s.comm)
        if s.is_root and not converged:
            print(f"  [WARNING] unconverged (misfit {misfit:.2e})", flush=True)
        return np.asarray(iso['bz_for_plot'], dtype=float)

    d_side_deg = DELTA * (s_hi - s_lo)
    d_top_mm = DELTA * (t_hi - t_lo)

    if s.is_root:
        print(f"\n{'=' * 88}\nSHIM EXCHANGE RATIO (delta = {DELTA} normalized, "
              f"coil held at {coil:.0f} A)\n{'=' * 88}", flush=True)
        print(f"linearized about: {origin}")
        print(f"side x0: {np.round(side0, 3).tolist()}")
        print(f"top  x0: {np.round(top0, 3).tolist()}")
        # Stations whose -delta leg clips against a bound: their contribution to
        # the central difference is one-sided and therefore biased low.
        clipped_s = int(np.sum((side0 - d_side_deg < s_lo) | (side0 + d_side_deg > s_hi)))
        clipped_t = int(np.sum((top0 - d_top_mm < t_lo) | (top0 + d_top_mm > t_hi)))
        if clipped_s or clipped_t:
            print(f"[NOTE] {clipped_s}/{n} side and {clipped_t}/{n} top stations sit within "
                  f"delta of a bound;\n       their leg of the central difference is clipped, "
                  f"biasing those radii's\n       sensitivity LOW. Read the per-band numbers, "
                  f"not just the average.")
        print(flush=True)

    bz_sp = bz_of(side0 + d_side_deg, top0)
    bz_sm = bz_of(side0 - d_side_deg, top0)
    bz_tp = bz_of(side0, top0 + d_top_mm)
    bz_tm = bz_of(side0, top0 - d_top_mm)

    if s.is_root:
        # d<B>/d(normalized unit): central difference over the 2*DELTA span.
        a = (bz_sp - bz_sm) / (2.0 * DELTA)      # T per normalized side unit
        b = (bz_tp - bz_tm) / (2.0 * DELTA)      # T per normalized top unit
        r = np.asarray(radii, dtype=float)

        print(f"\n{'r[mm]':>8} {'a=d<B>/ds':>12} {'b=d<B>/dt':>12} {'rho=a/b':>10}")
        with np.errstate(divide='ignore', invalid='ignore'):
            rho_r = np.where(np.abs(b) > 1e-12, a / b, np.nan)
        for i in range(len(r)):
            print(f"{r[i]:8.1f} {a[i]:12.5f} {b[i]:12.5f} {rho_r[i]:10.4f}")

        a_tot, b_tot = float(a.mean()), float(b.mean())
        rho = a_tot / b_tot if abs(b_tot) > 1e-12 else float('nan')
        print(f"\nradius-averaged: a = {a_tot:.5f} T/side-unit, "
              f"b = {b_tot:.5f} T/top-unit")
        print(f"  -> smoothness_exchange_ratio (rho) = {rho:.4f}")

        inner = rho_r[r < r.min() + (r.max() - r.min()) / 3.0]
        mid = rho_r[(r >= r.min() + (r.max() - r.min()) / 3.0)
                    & (r <= r.max() - (r.max() - r.min()) / 3.0)]
        outer = rho_r[r > r.max() - (r.max() - r.min()) / 3.0]
        print(f"  per band: inner {np.nanmean(inner):.4f} | "
              f"mid {np.nanmean(mid):.4f} | outer {np.nanmean(outer):.4f}")
        print(f"  spread over radius: {np.nanmin(rho_r):.4f} .. {np.nanmax(rho_r):.4f}")
        print("\nThe field-neutral direction is (side_norm, top_norm) = (1, -rho): "
              "raising every\nside station by one normalized unit and lowering every top "
              "station by rho leaves\n<B>(r) unchanged. Put rho in the config as "
              "optimization.smoothness_exchange_ratio.")
        if not np.isfinite(rho) or rho <= 0:
            print("\n[WARNING] rho is non-positive or undefined -- the two levers do NOT "
                  "both raise <B>\nhere (check that the pole is programmatic and the shims "
                  "are not clipped at a bound).")

    solver.dispose()
