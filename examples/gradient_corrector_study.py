"""Do passive gradient correctors (GCs) actually do what we want?

Conceptual check on the muon machine (examples/config_muon_smaller.yml):

  * a HARMONIC BAR is iron above and below the median plane -> its induced
    dipole ADDS to Bz between the plates (a local bump);
  * a GRADIENT CORRECTOR is supposed to WEAKEN Bz at the beam and leave a
    positive dBz/dr behind (restoring radial focusing that the falling
    extraction field destroys).

Both are the same material in the same gap, so the sign of what they do is
pure GEOMETRY. The point of this script is to make that explicit: it drops a
few passive iron elements -- shapes traced from the classic passive
magnetic-channel / gradient-corrector cross-section -- into the valley of the
solved muon machine and plots Bz along a radial line through the element's
azimuthal centre.

Each element is declared ``perturbative: true``, so the main machine relaxes
WITHOUT it and the element then relaxes in that frozen field. The main
solution is therefore bit-identical across variants and

    dBz(r) = Bz(total) - Bz(main) = the element's own field

is a clean difference, not a difference of two noisy solves (the script also
verifies that identity against a separate no-GC run).

Run:  python examples/gradient_corrector_study.py [config.yml]
      python examples/gradient_corrector_study.py --joint   (cross-check one
      variant with the GC inside the main relaxation, i.e. with the machine's
      back-reaction included)
"""

import os
import sys
import time

import numpy as np

import cyclotron_optimizer as co
from cyclotron_optimizer.config_io.config import ComponentSpec
from cyclotron_optimizer.core.species import IonSpecies
from cyclotron_optimizer.simulation.field_calculator import _fld_chunked

# ---------------------------------------------------------------------------
# Where the elements sit
# ---------------------------------------------------------------------------
# The muon machine has four 25-deg hills; theta = 45 deg is the middle of a
# valley (Bz ~ 0.12 T there vs ~0.48 T on the hill), which is where an
# extraction element would live. The GC spans 20 deg of azimuth, so the
# "azimuthal centre of the GC" is theta = 45 deg.
GC_START_DEG, GC_END_DEG = 35.0, 55.0
GC_CENTER_DEG = 0.5 * (GC_START_DEG + GC_END_DEG)

R_EXTR_MM = 335.0        # r(E = 5 MeV) from the base solve
E_EXTR_MEV = 5.0

R0, R1 = 300.0, 370.0    # radial span of the plate structure
GC_MESH_MM = 10.0

# Radial scan line (median plane, theta = GC_CENTER_DEG)
LINE_R = np.arange(240.0, 420.0 + 0.25, 0.25)


# ---------------------------------------------------------------------------
# Element cross-sections: polygons in the (r, z) plane [mm]
# ---------------------------------------------------------------------------
# Shapes traced from the reference figure: an iron stack above and below the
# beam whose CLEARANCE to the median plane shrinks with radius (taper, then
# steps), plus a separate bar sitting radially INSIDE the beam and straddling
# the median plane.
_R13 = R0 + (R1 - R0) / 3.0
_R23 = R0 + 2.0 * (R1 - R0) / 3.0

PLATE_UNIFORM = [(R0, 20.0), (R1, 20.0), (R1, 40.0), (R0, 40.0)]

PLATE_TAPER = [(R0, 30.0), (R1, 12.0), (R1, 32.0), (R0, 50.0)]

PLATE_STEP = [(R0, 30.0), (_R13, 30.0), (_R13, 21.0), (_R23, 21.0),
              (_R23, 12.0), (R1, 12.0), (R1, 46.0), (R0, 46.0)]

INNER_BAR = [(278.0, -25.0), (292.0, -25.0), (292.0, 25.0), (278.0, 25.0)]

# ---- "bigger blocks, closer to the median plane at the outer radius" --------
# Same staircase idea as PLATE_STEP but grown in every direction: radially
# 295-380 instead of 300-370, clearance 30 -> 18 -> 6 mm instead of 30 -> 21 ->
# 12, and 70 mm tall instead of 46.
_B13, _B23 = 323.3, 351.7
PLATE_BIG = [(295.0, 30.0), (_B13, 30.0), (_B13, 18.0), (_B23, 18.0),
             (_B23, 6.0), (380.0, 6.0), (380.0, 70.0), (295.0, 70.0)]

# Same aperture, but the two plates are JOINED by an outer web into a C whose
# mouth faces inward: the flux entering the top plate can run out radially and
# down the web instead of crossing the beam aperture. One solid, so there are
# no non-conforming iron-iron contacts.
C_CHANNEL = [(295.0, -70.0), (400.0, -70.0), (400.0, 70.0), (295.0, 70.0),
             (295.0, 30.0), (_B13, 30.0), (_B13, 18.0), (_B23, 18.0),
             (_B23, 6.0), (380.0, 6.0), (380.0, -6.0), (_B23, -6.0),
             (_B23, -18.0), (_B13, -18.0), (_B13, -30.0), (295.0, -30.0)]

# Classic passive magnetic channel: bars flanking the beam RADIALLY, spanning
# the full vertical aperture, beam in between.
BAR_IN = [(305.0, -50.0), (325.0, -50.0), (325.0, 50.0), (305.0, 50.0)]
BAR_OUT = [(345.0, -50.0), (365.0, -50.0), (365.0, 50.0), (345.0, 50.0)]

# CLOSED frame: iron on all four sides of the aperture (r 315-355, |z| < 18),
# open only along the beam direction -- a magnetic shield around the channel.
# Four abutting bodies (a polygon with a hole is not a simple polygon); the
# contacts are non-conforming, which is acceptable for a small perturbative
# element but would need a mesh_group if this ever entered the main solve.
FRAME = [
    [(295.0, 18.0), (375.0, 18.0), (375.0, 60.0), (295.0, 60.0)],     # top
    [(295.0, -60.0), (375.0, -60.0), (375.0, -18.0), (295.0, -18.0)],  # bottom
    [(295.0, -18.0), (315.0, -18.0), (315.0, 18.0), (295.0, 18.0)],   # inner
    [(355.0, -18.0), (375.0, -18.0), (375.0, 18.0), (355.0, 18.0)],   # outer
]


def _mirror_z(poly):
    """Median-plane mirror of an (r, z) polygon (reversed to keep winding)."""
    return [(r, -z) for r, z in reversed(poly)]


def _pair(poly):
    """A plate and its median-plane mirror image."""
    return [poly, _mirror_z(poly)]


VARIANTS = {
    "harmonic_bar": dict(
        pieces=_pair(PLATE_UNIFORM),
        label="Harmonic bar: uniform plates above/below",
    ),
    "gc_taper": dict(
        pieces=_pair(PLATE_TAPER),
        label="GC (taper): clearance 30 -> 12 mm outward",
    ),
    "gc_step": dict(
        pieces=_pair(PLATE_STEP),
        label="GC (stepped): staircase clearance, thickening outward",
    ),
    "gc_inner_bar": dict(
        pieces=[INNER_BAR],
        label="Inner bar only: iron radially inside the beam",
    ),
    "gc_full": dict(
        pieces=_pair(PLATE_STEP) + [INNER_BAR],
        label="Full GC: stepped plates + inner bar (the figure)",
    ),
    # --- bigger / closer / different material / flux-bypass shapes ----------
    "gc_big": dict(
        pieces=_pair(PLATE_BIG),
        label="Bigger blocks, clearance 30 -> 6 mm outward (1010)",
    ),
    "gc_big_m13": dict(
        pieces=_pair(PLATE_BIG), material="iron_x1p13",
        label="Same, 1.13x stronger material (Ms 2.08 -> 2.35 T, VP-like)",
    ),
    "gc_big_m5": dict(
        pieces=_pair(PLATE_BIG), material="iron_x5",
        label="Same, 5x stronger material -- saturation out of reach",
    ),
    "gc_cbox": dict(
        pieces=[C_CHANNEL],
        label="C-channel: plates joined by an outer web (flux bypass)",
    ),
    "gc_bars": dict(
        pieces=[BAR_IN, BAR_OUT],
        label="Passive magnetic channel: bars flanking the beam radially",
    ),
    "gc_frame": dict(
        pieces=FRAME, mesh_group="gc",
        label="Closed frame: iron on all four sides of the aperture",
    ),
}

MU0 = 4.0e-7 * np.pi


def scaled_bh_materials(base_file, out_dir, factors=(1.13, 5.0)):
    """Surrogate 'better materials': the real 1010 curve with M = B - mu0*H
    scaled by a factor (mu0*H added back), written as new BH files.

    Scaling M scales BOTH the saturation magnetization and the initial
    susceptibility, so x1.13 is a generous stand-in for Vanadium Permendur
    (Ms 2.08 -> 2.35 T) and x5 puts saturation out of reach entirely. This is
    a deliberately generous test: if even x5 barely moves the field, the
    material is not the limiting factor.

    (Radia's MatSatIsoFrm formula is NOT usable for this: M = ms*tanh(ksi*H/ms)
    saturates at H ~ ms/ksi, which for a high-chi soft material is a fraction
    of a mT -- it models a permanent magnet, not soft iron.)
    """
    data = np.loadtxt(base_file, delimiter=",")
    h_am, b_t = data[:, 0], data[:, 1]
    m_t = b_t - MU0 * h_am
    out = {}
    for f in factors:
        name = "iron_x" + f"{f:g}".replace(".", "p")
        path = os.path.join(out_dir, f"bh_{name}.csv")
        np.savetxt(path, np.column_stack([h_am, MU0 * h_am + f * m_t]),
                   delimiter=",")
        out[name] = {"type": "bh_file", "file": path, "curve": "BH",
                     "h_unit": "A/m", "b_unit": "T"}
    return out


def gc_specs(pieces, *, perturbative=True, mesh_mm=GC_MESH_MM, material="iron",
             mesh_group=None):
    """ComponentSpecs for one variant (one swept_polygon per piece).

    No symmetry is declared: both halves are built explicitly, so nothing
    depends on how a median-plane mirror would be interpreted, and pieces that
    straddle z = 0 are handled the same way as pairs.

    ``mesh_group`` puts the pieces in one conforming gmsh model -- required
    when they TOUCH (e.g. the closed frame), since non-conforming iron-iron
    contacts stall the relaxation well above its precision target.
    """
    return [
        ComponentSpec(
            name=f"gc_piece_{i}", kind="swept_polygon", material=material,
            symmetry=None, perturbative=perturbative, mesh_group=mesh_group,
            mesh={"max_size": mesh_mm, "min_size": 0.4 * mesh_mm},
            params=dict(polygon=[list(p) for p in poly], axis=[0.0, 0.0, 1.0],
                        start_angle_deg=GC_START_DEG,
                        end_angle_deg=GC_END_DEG),
        )
        for i, poly in enumerate(pieces)
    ]


# ---------------------------------------------------------------------------
# Field along the radial line
# ---------------------------------------------------------------------------
def line_points(theta_deg=GC_CENTER_DEG, r_mm=LINE_R):
    th = np.deg2rad(theta_deg)
    return np.column_stack([r_mm * np.cos(th), r_mm * np.sin(th),
                            np.zeros_like(r_mm)])


def bz_on_line(radia_id, session, points, use_gpu=True):
    bz = _fld_chunked(radia_id, "bz", points, use_gpu=use_gpu,
                      rank=session.rank, verbosity=0)
    return None if bz is None else np.asarray(bz, dtype=float)


def d_dr(values, r_mm):
    """dValue/dr in T/m (r in mm)."""
    return np.gradient(np.asarray(values, dtype=float),
                       np.asarray(r_mm, dtype=float)) * 1000.0


def _working_point(r, dbz, g, brho):
    """Radius where the element's own field crosses zero on its RISING flank.

    That is where a gradient corrector is meant to be used: the beam sees the
    full gradient and no net dipole kick. Returns (r0, g(r0), 1/f(r0)) or
    (None, None, None) if the element never changes sign (a pure bump/dip).
    """
    sel = (r > 250.0) & (r < 400.0)
    rr, yy = r[sel], np.asarray(dbz)[sel]
    up = np.where((yy[:-1] < 0.0) & (yy[1:] >= 0.0))[0]
    if not len(up):
        return None, None, None
    i = up[-1]
    r0 = rr[i] - yy[i] * (rr[i + 1] - rr[i]) / (yy[i + 1] - yy[i])
    g0 = float(np.interp(r0, r, g))
    arc0 = r0 * 1e-3 * np.deg2rad(GC_END_DEG - GC_START_DEG)
    return float(r0), g0, g0 * arc0 / brho


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = [a for a in argv[1:] if a.startswith("--")]
    joint = "--joint" in flags
    only = next((f.split("=", 1)[1].split(",") for f in flags
                 if f.startswith("--only=")), None)
    tag = next((f.split("=", 1)[1] for f in flags if f.startswith("--tag=")), "")
    config_path = args[0] if args else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config_muon_smaller.yml")

    out_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(out_dir, exist_ok=True)
    stem = "gradient_corrector_study" + (f"_{tag}" if tag else "")

    if "--report-only" in flags:
        # Re-print the tables / redraw the figure from the saved lines (no solves)
        d = np.load(os.path.join(out_dir, stem + ".npz"))
        report({"base": dict(bz_total=d["bz_base"], bz_gc=None, pieces=[])} |
               {k[4:]: dict(bz_total=d["bz_base"] + d[k], bz_gc=d[k],
                            pieces=VARIANTS[k[4:].replace("_joint", "")]["pieces"])
                for k in d.files if k.startswith("dbz_")},
               out_dir, stem, joint=joint)
        return

    points = line_points()
    results = {}

    with co.Session(config_path, verbosity=1) as s:
        base_components = list(s.config.components)
        s.config.materials_def = {
            **s.config.materials_def,
            **scaled_bh_materials(s.config.materials_def["iron"]["file"], out_dir),
        }

        selected = VARIANTS if only is None else {
            k: VARIANTS[k] for k in only if k in VARIANTS}
        cases = [("base", None)] + list(selected.items())
        if joint:
            cases.append(("gc_full_joint", VARIANTS["gc_full"]))

        for name, variant in cases:
            perturbative = not name.endswith("_joint")
            if variant is None:
                s.config.components = list(base_components)
            else:
                s.config.components = list(base_components) + gc_specs(
                    variant["pieces"], perturbative=perturbative,
                    material=variant.get("material", "iron"),
                    mesh_group=variant.get("mesh_group"))

            if s.is_root:
                print(f"\n{'=' * 70}\n[{name}] solving\n{'=' * 70}", flush=True)
            t0 = time.time()
            model = s.build(use_gpu=True)
            model.solve()
            dt = time.time() - t0

            solver = model._solver
            bz_total = bz_on_line(model.cyclotron.id, s, points)
            bz_main = bz_total
            bz_gc = None
            n_elem, m_mean, m_max = 0, 0.0, 0.0
            if solver._perturb_cnt is not None:
                bz_main = bz_on_line(solver._main_cnt.id, s, points)
                bz_gc = bz_on_line(solver._perturb_cnt.id, s, points)
                n_elem, m_mean, m_max = _iron_state(solver._perturb_cnt.id)

            if s.is_root:
                print(f"[{name}] solved in {dt:.1f} s  converged={model.converged} "
                      f"misfit={model.misfit:.2e}  element tets: {n_elem}  "
                      f"|M| mean {m_mean:.3f} T max {m_max:.3f} T", flush=True)
                if bz_gc is not None:
                    # superposition check: container field == main + element
                    err = float(np.abs(bz_total - (bz_main + bz_gc)).max())
                    print(f"[{name}] superposition check "
                          f"max|total - (main + element)| = {err:.2e} T", flush=True)
                results[name] = dict(bz_total=bz_total, bz_main=bz_main,
                                     bz_gc=bz_gc,
                                     pieces=[] if variant is None else variant["pieces"],
                                     label="Base machine (no element)"
                                     if variant is None else variant["label"],
                                     perturbative=perturbative,
                                     m_mean=m_mean, m_max=m_max,
                                     seconds=dt)
            model.dispose()

        if not s.is_root:
            return

    report(results, out_dir, stem, joint=joint)


def _iron_state(radia_id):
    """(element count, mean |M|, max |M|) of a relaxed object -- how hard the
    material is being driven (compare |M|max against the material's Ms)."""
    import radia as rad
    try:
        arr = np.asarray(rad.ObjM(radia_id), dtype=float).reshape(-1, 2, 3)
    except Exception:
        return -1, 0.0, 0.0
    mag = np.linalg.norm(arr[:, 1, :], axis=1)
    return len(mag), float(mag.mean()), float(mag.max())


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report(results, out_dir, stem="gradient_corrector_study", *, joint=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = LINE_R
    base = results["base"]["bz_total"]

    species = IonSpecies("muon")
    m = species.mass_mev
    p_mev = np.sqrt((E_EXTR_MEV + m) ** 2 - m ** 2)
    brho = p_mev / (299.792458 * abs(species.q))          # T*m
    arc_m = R_EXTR_MM * 1e-3 * np.deg2rad(GC_END_DEG - GC_START_DEG)

    i_extr = int(np.argmin(np.abs(r - R_EXTR_MM)))

    print("\n" + "=" * 100)
    print(f"Bz along the radial line at theta = {GC_CENTER_DEG:.0f} deg "
          f"(valley centre), median plane")
    print(f"Beam reference radius r = {R_EXTR_MM:.0f} mm (E = {E_EXTR_MEV} MeV), "
          f"B*rho = {brho:.4f} T m, GC arc length = {arc_m * 1000:.0f} mm")
    print("=" * 100)
    header = (f"{'variant':<16} {'dBz(r_b)':>10} {'kick':>8} {'d(dBz)/dr':>11} "
              f"{'1/f':>8} {'min dBz':>9} {'max dBz':>9}   "
              f"{'r0':>6} {'g(r0)':>8} {'1/f(r0)':>8} {'|M|max':>8}")
    print(header)
    print(f"{'':<16} {'[mT]':>10} {'[deg]':>8} {'[T/m]':>11} {'[1/m]':>8} "
          f"{'[mT]':>9} {'[mT]':>9}   {'[mm]':>6} {'[T/m]':>8} {'[1/m]':>8} {'[T]':>8}")
    print("-" * 120)

    dbdr_base = d_dr(base, r)

    rows = {}
    for name, res in results.items():
        if name == "base":
            continue
        dbz = (res["bz_gc"] if res["bz_gc"] is not None
               else res["bz_total"] - base)
        g = d_dr(dbz, r)
        inv_f = g[i_extr] * arc_m / brho
        kick = np.degrees(dbz[i_extr] * arc_m / brho)
        r0, g0, inv_f0 = _working_point(r, dbz, g, brho)
        rows[name] = dict(dbz=dbz, g=g, inv_f=inv_f, r0=r0)
        wp = ("     --       --       --" if r0 is None else
              f"{r0:>6.0f} {g0:>8.2f} {inv_f0:>8.2f}")
        mmax = res.get("m_max")
        print(f"{name:<16} {dbz[i_extr] * 1e3:>10.2f} {kick:>8.2f} {g[i_extr]:>11.3f} "
              f"{inv_f:>8.2f} {dbz.min() * 1e3:>9.2f} {dbz.max() * 1e3:>9.2f}   {wp} "
              f"{'    --' if not mmax else f'{mmax:>8.2f}'}")
    print("-" * 120)
    print("dBz   = the element's own field; r_b = beam reference radius "
          f"({R_EXTR_MM:.0f} mm, mid-element here).")
    print("kick  = dBz*L/(B*rho): the DIPOLE deflection the element adds "
          "(unwanted -- it steers the orbit).")
    print("1/f   = g*L/(B*rho):   the GRADIENT (radial focusing) strength "
          "(wanted; positive = focusing).")
    print("r0    = radius where the element's dBz crosses ZERO on its rising "
          "flank -- the design working")
    print("        point of a gradient corrector: full gradient, no net dipole "
          "kick. g(r0), 1/f(r0) there.")

    # ---------------- plots ----------------
    order = [k for k in results if k != "base"]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_geo, ax_tot, ax_d, ax_g = axes.ravel()

    # (a) cross-sections
    for i, name in enumerate(order):
        for j, poly in enumerate(results[name]["pieces"]):
            xy = np.array(list(poly) + [poly[0]], dtype=float)
            ax_geo.plot(xy[:, 0], xy[:, 1], color=colors[i], lw=1.6,
                        label=name if j == 0 else None)
    ax_geo.axhline(0.0, color="k", lw=0.8, ls="--")
    ax_geo.axvline(R_EXTR_MM, color="0.4", lw=1.0, ls=":")
    ax_geo.text(R_EXTR_MM, -55, f" beam\n r={R_EXTR_MM:.0f}", fontsize=8, color="0.3")
    ax_geo.set_xlabel("r [mm]")
    ax_geo.set_ylabel("z [mm]")
    ax_geo.set_title("(a) Element cross-sections in the (r, z) plane")
    ax_geo.legend(fontsize=8, loc="upper right")
    ax_geo.grid(alpha=0.3)
    ax_geo.set_aspect("equal")

    # (b) total Bz
    ax_tot.plot(r, base, "k-", lw=2, label="base (no element)")
    for i, name in enumerate(order):
        ax_tot.plot(r, base + rows[name]["dbz"], color=colors[i], lw=1.3, label=name)
    ax_tot.axvline(R_EXTR_MM, color="0.4", lw=1.0, ls=":")
    ax_tot.set_xlabel("r [mm]")
    ax_tot.set_ylabel(r"$B_z$ [T]")
    ax_tot.set_title(f"(b) Total $B_z$ on the radial line "
                     f"($\\theta$ = {GC_CENTER_DEG:.0f}$^\\circ$, z = 0)")
    ax_tot.legend(fontsize=8)
    ax_tot.grid(alpha=0.3)

    # (c) delta Bz
    ax_d.axhline(0.0, color="k", lw=0.8)
    for i, name in enumerate(order):
        ax_d.plot(r, rows[name]["dbz"] * 1e3, color=colors[i], lw=1.6, label=name)
        if rows[name]["r0"] is not None:
            ax_d.plot([rows[name]["r0"]], [0.0], "o", color=colors[i], ms=6,
                      mec="k", mew=0.6, zorder=5)
    ax_d.axvline(R_EXTR_MM, color="0.4", lw=1.0, ls=":")
    ax_d.set_xlabel("r [mm]")
    ax_d.set_ylabel(r"$\Delta B_z$ [mT]")
    ax_d.set_title("(c) Element's own field (enhancement > 0, weakening < 0);\n"
                   "circles = zero-dipole working point")
    ax_d.legend(fontsize=8)
    ax_d.grid(alpha=0.3)

    # (d) gradient
    ax_g.axhline(0.0, color="k", lw=0.8)
    ax_g.plot(r, dbdr_base, "k-", lw=2, label="base $dB_z/dr$")
    for i, name in enumerate(order):
        ax_g.plot(r, dbdr_base + rows[name]["g"], color=colors[i], lw=1.3, label=name)
    ax_g.axvline(R_EXTR_MM, color="0.4", lw=1.0, ls=":")
    ax_g.set_xlabel("r [mm]")
    ax_g.set_ylabel(r"$dB_z/dr$ [T/m]")
    ax_g.set_title("(d) Total radial gradient (more positive = more radial focusing)")
    ax_g.legend(fontsize=8)
    ax_g.grid(alpha=0.3)

    fig.suptitle("Passive gradient correctors vs a harmonic bar "
                 "(muon machine, valley at 45 deg)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = os.path.join(out_dir, stem + ".png")
    fig.savefig(png, dpi=140)
    print(f"\nSaved {png}")

    npz = os.path.join(out_dir, stem + ".npz")
    np.savez(npz, r_mm=r, bz_base=base,
             **{f"dbz_{k}": rows[k]["dbz"] for k in order})
    print(f"Saved {npz}")

    if joint and "gc_full_joint" in rows and "gc_full" in rows:
        d = rows["gc_full_joint"]["dbz"] - rows["gc_full"]["dbz"]
        print(f"\nBack-reaction check (joint - perturbative), gc_full: "
              f"max |diff| = {np.abs(d).max() * 1e3:.3f} mT "
              f"({np.abs(d).max() / max(np.abs(rows['gc_full']['dbz']).max(), 1e-12):.1%} "
              f"of the element's own field)")


if __name__ == "__main__":
    main(sys.argv)
