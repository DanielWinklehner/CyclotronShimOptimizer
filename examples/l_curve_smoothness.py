"""L-curve sweep of the shim SMOOTHNESS weight for the DFO-LS shim optimizer.

Re-runs the joint DFO-LS shim optimization at a series of second-difference
(roughness) penalty weights and records, at each converged optimum, the
FLATNESS (isochronism std_dev, MHz) and the ROUGHNESS (||D2|| of the normalized
shim profile). Plotting roughness vs flatness traces an L-curve; the KNEE --
where roughness has dropped sharply but flatness has barely moved -- is the
recommended weight. The right weight is problem-dependent, so this measures it
rather than hardcoding one.

Scale intuition (why ~0.1 is the natural starting point): the flatness block is
||f - mean(f)|| ~ std * sqrt(n_eval_pts) (~0.6 MHz for std 0.1, n=36), while a
jagged NORMALIZED profile has ||D2|| ~ 1, so smoothness_weight ~ 0.1 makes the
roughness residual ~15% of the flatness residual -- a nudge, not a takeover.

Needs GPU + a PROGRAMMATIC pole (pole kind 'pole', not an STP file) or the shims
are inert. Heavy: one full DFO-LS optimization per weight.

Run:  [mpiexec -n N] python examples/l_curve_smoothness.py [config.yml] \
                                  [w0,w1,...] [side|top|both|neutral]

Defaults: config_muon_smaller.yml, weights 0,0.03,0.1,0.3,1.0, block=side.

block=neutral sweeps the FIELD-BASIS weight instead: the penalty then acts on
(side_norm - rho*top_norm), the combination that leaves <B>(r) unchanged and so
is invisible to the isochronism residual -- the side/top null mode. The
<B>-carrying combination is held at ACTIVE_WEIGHT (nearly free) throughout, so
this L-curve trades ONLY wasted roughness against flatness. That is the sweep to
read when roughness_neutral dominates roughness_active in the diagnostics CSV.
"""

import os
import sys
import csv
from datetime import datetime

import numpy as np

import cyclotron_optimizer as co
from cyclotron_optimizer.optimization.optimizer import CyclotronOptimizer
from cyclotron_optimizer.optimization.objective_function import (
    second_difference, field_basis_profiles)

CONFIG = (sys.argv[1] if len(sys.argv) > 1 else
          os.path.join(os.path.dirname(__file__), "config_muon_smaller.yml"))
WEIGHTS = ([float(w) for w in sys.argv[2].split(",")] if len(sys.argv) > 2
           else [0.0, 0.03, 0.1, 0.3, 1.0])
BLOCK = (sys.argv[3].lower() if len(sys.argv) > 3 else "side")   # side|top|both|neutral
# Held fixed while BLOCK="neutral" sweeps the null-mode weight. PER-BLOCK scale
# (0.05), NOT nearly-zero: the midplane field low-passes the shim profile on the
# gap scale (station-alternating modes attenuate ~e^-7 =~ 1e-3), so high-k
# jaggedness is field-invisible in the ACTIVE direction too -- an untaxed active
# component refills with station-scale sawtooth (measured 2026-07-28: active
# weight 0.0025 -> bang-bang bound-railed shims, sigma stuck 3.6x worse).
ACTIVE_WEIGHT = 0.05


def _normalize(offsets, lo, hi):
    offsets = np.asarray(offsets, dtype=float)
    span = hi - lo
    return (offsets - lo) / span if span > 0 else np.zeros_like(offsets)


def _normalized_roughness(offsets, lo, hi):
    """||D2(normalized offsets)|| -- the same metric the penalty acts on."""
    return float(np.linalg.norm(second_difference(_normalize(offsets, lo, hi))))


def _field_basis_roughness(side, top, bounds, rho):
    """(||D2(neutral)||, ||D2(active)||) for a shim pair -- the wasted vs needed split."""
    (s_lo, s_hi), (t_lo, t_hi) = bounds
    neutral, active = field_basis_profiles(_normalize(side, s_lo, s_hi),
                                           _normalize(top, t_lo, t_hi), rho)
    return (float(np.linalg.norm(second_difference(neutral))),
            float(np.linalg.norm(second_difference(active))))


def _apply_weight(opt, block, w):
    """Set the swept smoothness weight on the requested block(s)."""
    if block == "side":
        opt.smoothness_weight_side = w
        opt.smoothness_weight_top = 0.0
        opt.smoothness_weight_neutral = opt.smoothness_weight_active = None
    elif block == "top":
        opt.smoothness_weight_side = 0.0
        opt.smoothness_weight_top = w
        opt.smoothness_weight_neutral = opt.smoothness_weight_active = None
    elif block == "neutral":
        # FIELD BASIS: sweep the weight on the <B>-invariant (null-mode) part
        # and hold the <B>-carrying part nearly free. This is the L-curve to
        # read when the roughness is the side/top pair fighting each other --
        # roughness_neutral should collapse while flatness barely moves.
        opt.smoothness_weight_neutral = w
        opt.smoothness_weight_active = ACTIVE_WEIGHT
    else:  # both
        opt.smoothness_weight = w
        opt.smoothness_weight_side = None
        opt.smoothness_weight_top = None
        opt.smoothness_weight_neutral = opt.smoothness_weight_active = None


def _knee_index(rough, flat):
    """L-curve corner: the point of max distance below the chord joining the
    endpoints, on axes each scaled to [0,1]. Falls back to the min-roughness
    point when the curve is degenerate (all equal)."""
    rough = np.asarray(rough, dtype=float)
    flat = np.asarray(flat, dtype=float)
    if len(rough) < 3:
        return int(np.argmin(rough))

    def _unit(a):
        rng = a.max() - a.min()
        return (a - a.min()) / rng if rng > 0 else np.zeros_like(a)

    x, y = _unit(rough), _unit(flat)          # x: roughness, y: flatness
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    denom = np.hypot(x1 - x0, y1 - y0)
    if denom == 0:
        return int(np.argmin(rough))
    # perpendicular distance of each point from the endpoint chord
    dist = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom
    return int(np.argmax(dist))


with co.Session(CONFIG) as s:
    radii_mm = s.default_radii_mm().tolist()
    opt = s.config.optimization
    s_lo, s_hi = opt.side_shim_min_deg, opt.side_shim_max_deg
    t_lo, t_hi = opt.top_shim_min_mm, opt.top_shim_max_mm

    if s.is_root:
        print(f"\n{'='*90}\nSMOOTHNESS L-CURVE SWEEP ({BLOCK} block)\n"
              f"config={CONFIG}\nweights={WEIGHTS}\n{'='*90}", flush=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("output", exist_ok=True)
        csv_path = os.path.join("output", f"l_curve_smoothness_{ts}.csv")
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["smoothness_weight", "block", "flatness_mhz",
                 "roughness_side_norm", "roughness_top_norm",
                 "roughness_neutral_norm", "roughness_active_norm",
                 "optimal_coil_A", "diagnostics_file"])

    rows = []
    for w in WEIGHTS:
        _apply_weight(opt, BLOCK, w)
        if s.is_root:
            print(f"\n{'-'*90}\n[L-curve] smoothness weight = {w} on {BLOCK}\n{'-'*90}",
                  flush=True)

        optimizer = CyclotronOptimizer(s.config, radii_mm, comm=s.comm,
                                       rank=s.rank, verbosity=s.verbosity)
        result = optimizer.optimize()

        rough_side = _normalized_roughness(result["best_side_shims"], s_lo, s_hi)
        rough_top = _normalized_roughness(result["best_top_shims"], t_lo, t_hi)
        rho = getattr(opt, "smoothness_exchange_ratio", 1.0) or 1.0
        rough_neutral, rough_active = _field_basis_roughness(
            result["best_side_shims"], result["best_top_shims"],
            ((s_lo, s_hi), (t_lo, t_hi)), float(rho))
        flat = float(result["flatness_phase2"])
        rows.append((w, flat, rough_side, rough_top, rough_neutral, rough_active,
                     result["optimal_coil"]))

        if s.is_root:
            print(f"[L-curve] w={w}: flatness={flat:.5f} MHz, "
                  f"roughness_side={rough_side:.4f}, roughness_top={rough_top:.4f}, "
                  f"neutral={rough_neutral:.4f}, active={rough_active:.4f}, "
                  f"coil={result['optimal_coil']:.0f} A", flush=True)
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [w, BLOCK, flat, rough_side, rough_top, rough_neutral,
                     rough_active, result["optimal_coil"],
                     result.get("diagnostics_file", "")])

    if s.is_root:
        flat_arr = [r[1] for r in rows]
        # index of the roughness the sweep actually acts on: side=2, top=3, neutral=4
        col = {"top": 3, "neutral": 4}.get(BLOCK, 2)
        swept_rough = [r[col] for r in rows]
        knee = _knee_index(swept_rough, flat_arr)
        print(f"\n{'='*90}\nL-CURVE SUMMARY ({BLOCK} block)\n{'='*90}", flush=True)
        print(f"{'weight':>10} {'flatness(MHz)':>14} {'roughness':>11} "
              f"{'neutral':>10} {'active':>10}", flush=True)
        for i, (w, flat, rs, rt, rn, ra, _coil) in enumerate(rows):
            mark = "  <-- knee (recommended)" if i == knee else ""
            print(f"{w:>10} {flat:>14.5f} {swept_rough[i]:>11.4f} "
                  f"{rn:>10.4f} {ra:>10.4f}{mark}", flush=True)
        w_rec = rows[knee][0]
        key = ("smoothness_weight_neutral" if BLOCK == "neutral"
               else f"smoothness_weight for the {BLOCK} block")
        print(f"\nRecommended {key}: {w_rec}", flush=True)
        print("The knee is where roughness drops sharply while flatness barely moves; "
              "if flatness at the knee is unacceptable, step one weight toward 0.", flush=True)
        print(f"CSV: {csv_path}", flush=True)
