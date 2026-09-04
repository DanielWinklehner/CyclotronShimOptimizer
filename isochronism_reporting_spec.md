# Isochronism Analysis & Shimming — Design Specification

**Status:** design spec for implementation
**Scope:** AVF (azimuthally varying field) isochronous cyclotrons, any particle species, any harmonic number
**Audience:** implementers (human or agent) working on the shimming/EO analysis code

---

## 1. Purpose

Define the quantities, numerical requirements, plots, and machine-readable outputs that the shimming software must produce so that isochronism results are (a) physically meaningful, (b) comparable across machines and particle species, and (c) immediately legible to anyone in the cyclotron community.

The code currently reports the mean and standard deviation of the closed-orbit revolution frequency over a set of radii. **That is not a valid figure of merit and must be replaced.** Isochronism is a path integral along the acceleration trajectory, not a scatter statistic. A sign-alternating frequency error and a monotonic ramp with identical σ differ by an order of magnitude in accumulated RF phase. σ is retained only as a *solver-noise diagnostic* (§5.3), never as a headline number.

---

## 2. Nomenclature

| Symbol | Meaning | Units |
|---|---|---|
| `N_sec` | number of sectors (hills) / lattice periodicity | — |
| `h` | RF harmonic number, `f_RF / f_rev` | — |
| `f_RF` | RF frequency | Hz |
| `f_rev(E)` | closed-orbit revolution frequency at energy `E` | Hz |
| `δ(E)` | **relative isochronism error** (§3.1) | dimensionless |
| `Φ(E)` | accumulated RF phase slip | RF degrees |
| `ΔΦ_turn` | RF phase slip per turn | RF degrees/turn |
| `Φ_0` | initial RF phase at end of central region | RF degrees |
| `ΔE_0(E)` | nominal energy gain per turn at crest | eV |
| `n(E)` | turn number | — |
| `B̄(r)` | azimuthally averaged median-plane field on the closed orbit | T |
| `B_iso(r)` | ideal isochronous average field for the chosen `f_RF` | T |
| `F(r)` | flutter, `(⟨B²⟩ − ⟨B⟩²)/⟨B⟩²` | — |
| `ν_r, ν_z` | radial / vertical betatron tunes | — |
| `C_n(r)` | harmonic error amplitude, `sqrt(A_n² + B_n²)` | relative |
| `r_1, r_2` | boundaries of the shimmable/useful radial range (§4) | m |

---

## 3. Core definitions

### 3.1 Relative isochronism error

```
δ(E) ≡ [ f_rev(E) − f_RF / h ] / ( f_RF / h )
```

Dimensionless, referenced directly to the RF. **Do not** construct this by comparing `B̄(r)` against a separately computed `B_iso(r)`. Computing the ideal isochronous field is itself an error-prone fitting problem, and folding it into the isochronism metric silently contaminates the result. `B_iso` is a *shimming* variable (§7), not an isochronism metric.

At fixed energy `δ ≈ ΔB̄/B̄`, so the two are interchangeable for shim sensitivity work, but only `δ` should be reported as the isochronism error.

### 3.2 Phase slip per turn

```
ΔΦ_turn(E) = 360° · h · δ(E)          [RF degrees per turn]
```

Always RF degrees. Never particle-revolution degrees — the gaps run at RF and the energy gain depends on RF phase.

### 3.3 Accumulated phase slip (primary result)

Coupled longitudinal equations, per turn:

```
dΦ/dn = 2π h δ(E)
dE/dn = ΔE_0(E) · cos Φ
```

Eliminating `n` and integrating gives the closed form:

```
sin Φ(E) = sin Φ_0 + 2π h ∫[E_inj → E] δ(E′) / ΔE_0(E′) dE′
```

Implement exactly this. Two notes:

- The `arcsin` is not cosmetic — it encodes the `cos Φ` feedback in the energy gain. **If the argument exceeds 1, the beam stalls and turns around.** The code must detect this and report it as an explicit failure mode with the energy at which it occurs, not raise a domain error.
- Carry `Φ_0` from central-region design rather than assuming zero. This makes the central bump's contribution explicit.

**Notational trap:** some references write this with a `(2πh/f_RF)` prefactor and an absolute `Δf` in Hz. In that convention `Δf` is the *RF-equivalent* frequency error `h·f_rev − f_RF`, not `f_rev − f_RF/h`. Using the revolution-frequency error with that prefactor loses a factor of `h`. The dimensionless form above has no such ambiguity — use it.

Turn number follows by quadrature with the same `Φ(E)`:

```
n(E) = ∫[E_inj → E] dE′ / ( ΔE_0(E′) · cos Φ(E′) )
```

### 3.4 RF frequency gauge

`f_RF` is a free parameter and **must be optimized**, not left at the nominal design value. Write `f_RF = f_RF⁰ (1 + ε)`; then `δ → δ − ε` to first order, and

```
sin Φ(E; ε) = sin Φ_0 + 2π h [ I(E) − ε J(E) ],
   I(E) = ∫ δ/ΔE_0 dE ,   J(E) = ∫ dE/ΔE_0
```

`J` is monotonic, so minimizing `max|Φ|` over `ε` is a well-behaved 1-D minimax problem — solve by Brent or bisection. At the optimum the positive and negative extremes of `Φ(E)` are equal in magnitude. This is standard practice and typically **halves** the peak excursion for free.

Do **not** substitute "zero the mean of `δ`" for this. It is a different operation and gives a worse answer.

### 3.5 Design tolerance

Particle-agnostic, one line:

```
δ_allowed = Φ_budget / ( 360° · h · N_turns )
```

This is the worst case (systematic, single-sign error). Report achieved vs. allowed. `Φ_budget` is a stated design choice — typical values in the literature range from ±10° (tight) to ±30° (accepted for some production machines); the physical hard limit is 90°, at which the beam decelerates.

The particle enters in exactly two places, both of which are *stated inputs* rather than embedded assumptions:

1. the shape of `B_iso(r) ∝ γ(r)`, which sets required shim range (~7% for 70 MeV protons, ~1.75% for 35 MeV deuterons, etc.);
2. `N_turns` via `ΔE_0/E`, which sets the tolerance.

Nothing else is species-specific.

---

## 4. Radial partition

**Never report a single global statistic across the full radius.** Three regions with different physics, reported separately:

| Region | Range | Character |
|---|---|---|
| **Central** | `r < r_1` | Deliberately non-isochronous — field bump provides `ν_z` where flutter → 0. Crossed in few turns, so cheap. |
| **Useful / shimmable** | `r_1 ≤ r ≤ r_2` | The figure of merit. This is what pole-edge shimming controls. |
| **Fringe** | `r > r_2` | Rolloff is intrinsic to the pole geometry and does not shim out. Report, don't optimize. |

Defaults: `r_1` = radius where `F(r)` reaches 50% of its plateau value, or the end of the central plug bump, whichever is larger. `r_2` = outer limit of shim authority. Both overridable in config; both must appear in the output metadata.

For each region output: ΔΦ contribution, turn count spent there, and fraction of total |Φ| excursion generated there.

---

## 5. Numerical requirements

### 5.1 Sampling

**Sample uniformly in energy, not in radius.** For constant `ΔE_0`, uniform-in-`E` is uniform-in-turn-number, which is the natural integration variable. Uniform-in-`r` systematically under-weights the outer region — `dE/dr` grows with `r`, so each outer sample represents more turns than each inner one, and the outer region is usually where the slip is generated.

- Minimum 100 closed orbits across the useful range (60 is too few).
- Additionally resample densely (≥30 points) across the outer 10–15% of radius to resolve the fringe rolloff.
- If integrating on a non-uniform-`E` grid, weight by `dn/dE = 1/(ΔE_0 cos Φ)`.

### 5.2 Closed-orbit solver

Gordon-style Newton iteration on `(r, p_r)` for orbit closure over one sector period. Requirements:

- Converge orbit closure to `|Δr| < 1e-9 m` and `|Δp_r/p| < 1e-9`.
- ODE integrator: adaptive RK with `rtol ≤ 1e-10`. A well-converged EO solve resolves `f_rev` to ~`1e-6` relative or better; anything coarser means the solver, not the magnet, sets the reported error.
- Field interpolation off the FE mesh must be at least C¹ (bicubic or tricubic spline). Nearest-neighbour or linear interpolation injects exactly the noise described below.

### 5.3 Noise floor diagnostic (the only legitimate use of σ)

Physical `δ(r)` has no high-spatial-frequency content — the field is a smooth function of radius by construction. Any point-to-point jitter between adjacent closed orbits is numerical.

Procedure:

1. Fit `δ(r)` with a smooth basis (cubic spline with ~`n_pts/6` knots, or a low-order Chebyshev series).
2. `σ_resid` = RMS of the residual. **This** is the solver/mesh noise estimate.
3. Require `σ_resid < 0.2 × δ_allowed`. If violated, tighten integrator tolerance and refine the field map near the median plane, then re-run; if `σ_resid` drops, it was numerical.
4. Integrate the **smooth fit**, not the raw points. Integrating the grass accumulates noise into `Φ`.
5. Report `σ_resid` explicitly labelled as a numerical diagnostic, alongside the convergence study result.

Optional but useful: plot the power spectrum of the residual vs. radial wavenumber. Physical content lives in the first few cycles across the range; everything above that is artifact.

---

## 6. Required numerical outputs

Emit as JSON (diffable across shim iterations) and as a formatted text table.

### 6.1 Configuration block — mandatory, no exceptions

A phase-slip number without these is uninterpretable, which is why the same field map can honestly support wildly different claims.

```json
{
  "config": {
    "particle": "mu+", "charge_state": 1, "mass_MeV": 105.658,
    "E_inj_MeV": null, "E_ext_MeV": null,
    "N_sectors": 4, "harmonic_h": 4,
    "f_RF_MHz_nominal": null, "f_RF_MHz_optimized": null,
    "dE_per_turn_keV": null, "flat_topped": false,
    "N_turns": null,
    "r_inj_m": null, "r_ext_m": null,
    "r1_m": null, "r2_m": null,
    "field_map_source": "", "field_map_grid": "",
    "shim_iteration": 0
  }
}
```

### 6.2 Results block

```json
{
  "results": {
    "phi_max_deg": null,            // max|Φ|, useful range, RF degrees — PRIMARY
    "phi_p2p_deg": null,            // peak-to-peak
    "phi_budget_deg": null,         // stated design budget
    "margin": null,                 // phi_budget / phi_max
    "delta_rms_useful": null,       // RMS of δ over useful range
    "delta_max_abs_useful": null,
    "delta_allowed": null,          // = phi_budget / (360 h N_turns)
    "sigma_resid": null,            // numerical noise floor (§5.3)
    "noise_floor_ok": null,         // sigma_resid < 0.2 * delta_allowed
    "stall_detected": false,
    "stall_energy_MeV": null,
    "by_region": {
      "central": {"dPhi_deg": null, "turns": null, "frac_of_excursion": null},
      "useful":  {"dPhi_deg": null, "turns": null, "frac_of_excursion": null},
      "fringe":  {"dPhi_deg": null, "turns": null, "frac_of_excursion": null}
    },
    "tunes": {
      "nu_r_range": [null, null], "nu_z_min": null,
      "resonance_crossings": [
        {"line": "nu_r = 2 nu_z", "order": 3, "structural": false,
         "E_MeV": null, "turns_in_stopband": null}
      ]
    },
    "harmonics": {"C1_max_gauss": null, "C2_max_gauss": null,
                  "C1_max_radius_m": null},
    "shim": {"iron_removed_max_mm": null, "iron_budget_remaining_mm": null,
             "authority_margin": null}
  }
}
```

### 6.3 Headline summary line

One line, for logs and commit messages:

```
[shim iter 3] mu+ | N=4 h=4 | f_RF 168.024 MHz | dE0 250 keV | 22 turns
  Phi_max = 4.7 deg (budget 10, margin 2.1x) | delta_rms 8.3e-5 (allowed 2.3e-4)
  nu_z_min 0.31 | sigma_resid 1.1e-5 OK
```

---

## 7. Shimming module

### 7.1 Sensitivity matrix

```
M_ij = ∂B̄(r_i) / ∂δ_j
```

where `δ_j` is the cut depth (or angular removal) of the removable pole edge at radial station `j`. Compute once per magnet topology from a series of unit cuts at increasing radius in the FE model, then reuse across iterations.

Hard-edge first estimate, for sanity-checking `M` only:

```
ΔB̄(r) ≈ ( N_sec · δθ(r) / 2π ) · ( B_hill − B_valley )
```

(`δθ` = angular width removed per sector.) This is order-of-magnitude at best — it assumes flux vanishes with the removed iron, whereas in practice flux redistributes to other radii, especially when saturated. Always prefer the FE-computed matrix; flag a warning if the two disagree by more than 2×, since that indicates significant saturation.

### 7.2 Inverse problem

Solve `M δ = −ΔB̄_residual` as regularized least squares:

```
minimize  ||M δ + ΔB̄||²  +  λ_s ||D₂ δ||²  +  λ_t ||δ||²
subject to  0 ≤ δ_j ≤ δ_budget_j
```

- `D₂` = second-difference operator (machinability — the cut profile must be smooth).
- Non-negativity: **you can only remove iron.** If the solution wants to add field, that is a main-coil / trim-coil / flap problem, not a shimming problem. Flag it explicitly rather than returning a negative cut.
- Apply a per-iteration relaxation factor `α ∈ [0.6, 0.8]` and re-solve the FE model. Standard practice is several iterations with margin each time; over-cutting is unrecoverable.
- Track and report cumulative iron removed against `δ_budget` per station.

### 7.3 Correction authority check

Simulation-only shimming has a floor set not by numerics but by the model-to-machine offset, which is empirically **0.3–1%** on the azimuthally averaged field — 30–100× the isochronism requirement. Sources: B–H curve batch variation, machining tolerance (`δB/B ≈ δgap/gap`, so ±0.1 mm on a 100 mm hill gap is already 1e-3), pole deflection under magnetic load, coil positioning, welds. None of this is in the model.

The code must therefore verify, and report:

1. **Assumed build-offset envelope**, default ±0.5% with a saturation-like radial shape (excess at large `r`, deficit at small `r`).
2. **Available correction range** from trim coils / flaps / main-coil headroom, as `ΔB̄(r)` curves.
3. **Authority margin** = available range / assumed offset, per radius. Require ≥ 1.5 across the useful range.

If authority margin is below 1, further in-simulation shimming is worthless — the machine will not be correctable regardless of how good the model looks.

---

## 8. Required plots

All plots: RF degrees for phase, units of 1e-4 for `δ`, gauss for field errors. Shade the three radial regions consistently (light grey = central, white = useful, light red = fringe) in every radius-axis plot. Every figure carries the config summary line in the caption or as a footer.

**Fig. 1 — Accumulated phase slip Φ (PRIMARY).**
x: energy, with secondary top axes for mean radius and turn number. y: `Φ` in RF degrees. Horizontal dashed bands at `±Φ_budget`; dotted lines at ±90° (stall limit). Plot both nominal-`f_RF` and optimized-`f_RF` curves so the gauge gain is visible. Annotate `max|Φ|` and peak-to-peak.

**Fig. 2 — Phase slip per turn.**
x: mean radius. y: `ΔΦ_turn` [deg/turn], twin right axis in `δ` [1e-4]. Shows *where* slip is generated — the diagnostic that tells you where to cut.

**Fig. 3 — Isochronism error and noise floor.**
Two stacked panels sharing x = mean radius. Upper: `δ(r)` in 1e-4 with the smooth fit overlaid, plus a horizontal band at `±δ_allowed`. Lower: fit residual, annotated with `σ_resid` and the `0.2 × δ_allowed` threshold. This figure is what distinguishes "my magnet is imperfect" from "my solver is noisy."

**Fig. 4 — Field error (shimming variable).**
x: mean radius. y: `B̄(r) − B_iso(r)` in gauss, twin right axis in relative units. Overlay previous shim iterations in faded colour to show convergence. Sign convention noted on the plot: positive = too little field.

**Fig. 5 — Tunes and working diagram.**
Left panel: `ν_r`, `ν_z` vs. energy; plot `2ν_z` alongside `ν_r` so the Walkinshaw crossing is visible by inspection. Right panel: working curve in the `(ν_r, ν_z)` plane with resonance lines `k ν_r + l ν_z = m` up to 4th order, structural lines (`m` a multiple of `N_sec`) drawn heavier. Mark each crossing with turns-in-stopband.

**Fig. 6 — Harmonic errors.**
`C_n(r)` for n = 1…4 vs. radius, in gauss with a relative twin axis. Acceptance band at 5–10 G. Expect artifacts at very small and very large radius where radial gradients are steep — annotate rather than suppress.

**Fig. 7 — Shim map and sensitivity.**
Left: applied cut depth `δ(r)` with the remaining iron budget as a filled envelope. Right: heatmap of `M_ij`, radius-of-cut on x, radius-of-effect on y. The off-diagonal structure is the flux-redistribution effect and is worth looking at directly.

**Fig. 8 — Convergence history.**
`max|Φ|` and RMS `ΔB̄` vs. shim iteration number, log y. Horizontal line at `Φ_budget`.

**Fig. 9 — Correction authority.**
Assumed ±0.5% build-offset envelope (shaded) vs. available trim/flap/main-coil range (lines), both as `ΔB̄(r)`. Third curve: authority margin, with a threshold line at 1.5.

---

## 9. Optional module: species lifetime

For unstable species, survival is a hard constraint on `N_turns` and therefore couples directly to the isochronism budget via §3.5.

```
survival = exp( − n(E) / ( f_rev · γ(E) · τ_0 ) )
```

integrated along the acceleration. Muons: `τ_0 = 2.1969811 µs`. At `f_rev = 42 MHz`, one turn costs 23.81 ns of lab time.

Report survival fraction vs. turn number as an inset on Fig. 1, and include `N_turns_max_for_X%_survival` in the results block. For stable species the module is a no-op and `survival = 1`.

The useful framing: for a decay-limited machine the low turn count *buys* isochronism headroom. Quote the spec as `δ_allowed = Φ_budget/(360 h N_turns)` and let the survival requirement set `N_turns`, rather than importing a tolerance from a 500-turn proton machine.

---

## 10. Validation tests

Required before the code is trusted. All should be automated.

| # | Test | Pass criterion |
|---|---|---|
| 1 | Uniform field, zero flutter | `f_rev = qB/(2πγm)` to 1e-8 relative |
| 2 | Analytic isochronous field `B(r) = B_0 γ(r)`, zero flutter | `δ = 0` to solver tolerance; confirms the EO solver, not the magnet, is being tested |
| 3 | Round-trip shim | Perturb a converged map by a known `δ_known(r)`; sensitivity-matrix inversion recovers it to <10% |
| 4 | Turn-count consistency | `n(E)` from quadrature agrees with direct multi-turn tracking to <1% |
| 5 | Gauge invariance | Scale `B` by `(1+ε)` and `f_RF` by `(1+ε)`; `max|Φ|` after re-optimization is unchanged |
| 6 | Harmonic scaling | At fixed `δ(E)`, `Φ ∝ h` |
| 7 | Stall detection | Construct `δ` large enough to drive `|sin Φ| > 1`; code reports stall energy, does not throw |
| 8 | Species independence | Same normalized field shape, different particle; `δ(E/E_ext)` identical |

Test 5 is the important one — it verifies that the reported number is a property of the magnet rather than of the arbitrary normalization.

---

## 11. Deprecations

Remove from all output:

- Mean and standard deviation of `f_rev` as a headline result. σ survives **only** as `sigma_resid` per §5.3, and must be labelled a numerical diagnostic.
- Any single global isochronism statistic spanning the full radius (§4).
- Phase slip quoted without `ΔE_0`, `h`, and `N_turns` (§6.1).
- `f_RF` left at nominal without gauge optimization (§3.4).

---

## 12. References

- W. Kleeven and S. Zaremba, *Cyclotrons: Magnetic Design and Beam Dynamics*, CAS Proceedings (arXiv:1804.08961) — phase-slip formalism, shimming matrix, hard-edge estimate, harmonic-error acceptance.
- M. M. Gordon, *Particle Accelerators* **16** (1984) 39 — closed-orbit and accumulated-phase formalism.
- T. Planche, *Constant-Tune Cyclotrons* (arXiv:2211.02988) — integral form of the isochronism bound.
- IsoDAR@Yemilab Preliminary Design Report Vol. I (arXiv:2404.06281) — worked reporting example (dB in gauss, phase slip, tunes).
- DAEδALUS design study (arXiv:1107.0652) — explicit statement on when chasing model precision stops being useful.
- Isochronism of the ACCEL 250 MeV cyclotron, NIM A — TOSCA-vs-measured offset (~±0.5%).
