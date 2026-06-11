"""M3 experiment: Stage B via estimator_new — the headline regression test.

Spec §10 M3 acceptance:
    - Stage B must match Stage A on supported objectives within ~1.5×.
    - u not all-zero.
    - Not worse than open-loop.
    - Save figs 1-3.

Procedure (per scenario):
    1. Calibrate IdentifiedSystem via plant + Uniform[0,1] probe (T_cal=800).
    2. Build Stage A controllers (TRUE params) and Stage B controllers
       (FITTED params) for the same objective.
    3. Run closed-loop on a fresh plant seed; report z-RMS, effort, sat,
       achieved steady-state.

Saves:
    fig1_uncontrolled_vs_controlled.png
    fig2_u_with_saturation.png
    fig3_reachable_zonotope.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent  # week3/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "week 1"))

import Simulator as sim  # noqa: E402

from control.plant import SimulatorPlant  # noqa: E402
from control.observer import SteadyStateKalman  # noqa: E402
from control.controllers import OpenLoop, LQG, MPC  # noqa: E402
from control.closed_loop import run_closed_loop  # noqa: E402
from control.control_interface import IdentifiedSystem  # noqa: E402
from control.reachability import steady_state_gain, zonotope_vertices, feasibility  # noqa: E402
from metrics import rms, control_effort, saturation_fraction  # noqa: E402


# Note: the readout M is now the 2 leading PCs of the calibration measurements,
# computed inside IdentifiedSystem.calibrate. See iface.readout_M after fitting.


def stage_a_run(system, M, ref, T, plant_seed, ctrl_name, ref_const=True):
    """Run a controller built from the TRUE matrices."""
    n = system.A.shape[0]
    a0 = np.zeros(n)
    c0 = np.zeros(system.obs_dim)
    plant = SimulatorPlant(system, seed=plant_seed)
    obs = SteadyStateKalman(system.A, system.B, system.C, system.Q, system.R, a=a0, c=c0)
    if ctrl_name == "OL":
        ctrl = OpenLoop(input_dim=system.input_dim)
    elif ctrl_name == "LQG":
        ctrl = LQG(system.A, system.B, system.C, M, a=a0, c=c0, rho=0.1, ref=ref)
    elif ctrl_name == "MPC":
        ctrl = MPC(system.A, system.B, system.C, M, a=a0, c=c0, horizon=20, rho=0.1, ref=ref)
    else:
        raise ValueError(ctrl_name)
    logs = run_closed_loop(plant, ctrl, obs, T=T, ref_fn=(lambda t: ref) if ref_const else None)
    return logs


def stage_b_run(iface, M, ref, plant, T, ctrl_name, ref_const=True):
    """Run a controller built from the FITTED matrices, on a given plant."""
    obs = iface.make_observer()
    A, B, C, Q, R, a, c = iface.params()
    if ctrl_name == "OL":
        ctrl = OpenLoop(input_dim=B.shape[1])
    elif ctrl_name == "LQG":
        ctrl = LQG(A, B, C, M, a=a, c=c, rho=0.1, ref=ref)
    elif ctrl_name == "MPC":
        ctrl = MPC(A, B, C, M, a=a, c=c, horizon=20, rho=0.1, ref=ref)
    else:
        raise ValueError(ctrl_name)
    logs = run_closed_loop(plant, ctrl, obs, T=T, ref_fn=(lambda t: ref) if ref_const else None)
    return logs


def main():
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_plant = 7
    seed_probe = 42
    T_cal = 800
    T_run = 250

    system = sim.default_neural_system(seed=0, obs_dim=16)
    n = system.A.shape[0]
    a0 = np.zeros(n)
    c0 = np.zeros(system.obs_dim)

    # ---- Calibrate Stage B identified system -------------------------------
    cal_plant = SimulatorPlant(system, seed=seed_probe + 1)
    iface = IdentifiedSystem().calibrate(
        cal_plant, T_cal=T_cal, n=n, m=system.input_dim, seed=seed_probe,
        true_model=dict(A=system.A, B=system.B, C=system.C, a=a0, c=c0),
        x_true=None,  # x_hat alignment is gauge-dependent; not needed here
    )
    # Readout = 2 leading PCs of the probe measurements (spec §3).
    M = iface.readout_M
    print("=" * 80)
    print("M3 — Stage B identification diagnostics (default_neural_system)")
    print("=" * 80)
    print(f"  fit iters: {iface.diagnostics['fit_iters']}")
    print(f"  rho(A_fit): {iface.diagnostics['A_spectral_radius']:.4f}")
    print(f"  ||CB||_F:   {iface.diagnostics['CB_frob']:.4f}")
    print(f"  B col norms (true scale): {[f'{v:.3f}' for v in iface.diagnostics['B_col_norms']]}")
    print(f"  ||a||:      {iface.diagnostics['a_norm']:.4f}")
    print(f"  ||c||:      {iface.diagnostics['c_norm']:.4f}")
    print(f"  degenerate flags: {iface.degenerate_flags or 'none'}")
    print(f"  readout (PCA): explained var = "
          f"{[round(v, 3) for v in iface.readout_explained_var]}")
    print(f"  cond(G_readout) = {iface.readout_cond_G:.2f}  "
          f"({'well-conditioned' if iface.readout_cond_G < 50 else 'ill-conditioned'})")
    if "G_rel_err" in iface.validation:
        print(f"  G rel.err: {iface.validation['G_rel_err']:.4f}")
        print(f"  z0 err:    {iface.validation['z0_err']:.4f}")

    # ---- Reachability ------------------------------------------------------
    G_true, z0_true = steady_state_gain(system.A, system.B, system.C, M, a=a0, c=c0)
    G_fit, z0_fit = steady_state_gain(iface.A, iface.B, iface.C, M, a=iface.a, c=iface.c)
    ref_set_TRUE = G_true @ np.array([0.5, 0.5]) + z0_true
    # When picking the Stage-B setpoint, we have to use the FIT to decide
    # what's reachable — but evaluate it against the TRUE plant. Use the truth
    # for the "headline target" (so Stage A and Stage B share a ref in the
    # same observation-space coordinates), then check feasibility in BOTH
    # the true zonotope and the fit zonotope.
    ref_set = ref_set_TRUE
    inside_true, _ = feasibility(ref_set, G_true, z0_true)
    inside_fit, u_pred_fit = feasibility(ref_set, G_fit, z0_fit)
    print(f"\n  feasible target {ref_set.round(3)}:")
    print(f"     in TRUE zonotope: {inside_true}")
    print(f"     in FIT zonotope:  {inside_fit}  (predicted u_ss in fit ~ {u_pred_fit.round(3)})")

    # ---- Run controllers ---------------------------------------------------
    print("\n  Stage A (TRUE params) vs Stage B (FITTED params) on default_neural_system:")
    rows = []
    runs = {}

    print(f"\n  Suppression (ref = z0_true = 0):")
    print(f"    {'controller':10s}  {'A z-RMS':>10s}  {'B z-RMS':>10s}  {'B/A':>6s}  {'A effort':>10s}  {'B effort':>10s}")
    for cname in ["OL", "LQG", "MPC"]:
        ref_sup = np.zeros(M.shape[0])
        logs_A = stage_a_run(system, M, ref_sup, T_run, seed_plant, cname)
        plant_B = SimulatorPlant(system, seed=seed_plant)
        logs_B = stage_b_run(iface, M, ref_sup, plant_B, T_run, cname)
        zA = logs_A["y"] @ M.T
        zB = logs_B["y"] @ M.T
        rA = rms(zA); rB = rms(zB)
        eA = control_effort(logs_A["u"]); eB = control_effort(logs_B["u"])
        ratio = rB / rA if rA > 0 else float("inf")
        print(f"    {cname:10s}  {rA:10.4f}  {rB:10.4f}  {ratio:6.3f}  {eA:10.2f}  {eB:10.2f}")
        rows.append(("sup", cname, rA, rB, eA, eB))
        runs[("sup", cname, "A")] = logs_A
        runs[("sup", cname, "B")] = logs_B

    print(f"\n  Feasible setpoint ref = G_true·[0.5,0.5] + z0_true = {ref_set.round(3)}:")
    print(f"    {'controller':10s}  {'A rms_err':>10s}  {'B rms_err':>10s}  {'B/A':>6s}  {'A achieved':>20s}  {'B achieved':>20s}")
    for cname in ["OL", "LQG", "MPC"]:
        logs_A = stage_a_run(system, M, ref_set, T_run, seed_plant, cname)
        plant_B = SimulatorPlant(system, seed=seed_plant)
        logs_B = stage_b_run(iface, M, ref_set, plant_B, T_run, cname)
        zA = logs_A["y"] @ M.T
        zB = logs_B["y"] @ M.T
        eA = rms(zA - ref_set); eB = rms(zB - ref_set)
        ach_A = zA[-30:].mean(axis=0); ach_B = zB[-30:].mean(axis=0)
        effA = control_effort(logs_A["u"]); effB = control_effort(logs_B["u"])
        ratio = eB / eA if eA > 0 else float("inf")
        print(f"    {cname:10s}  {eA:10.4f}  {eB:10.4f}  {ratio:6.3f}  {str(ach_A.round(3)):>20s}  {str(ach_B.round(3)):>20s}")
        rows.append(("set", cname, eA, eB, effA, effB))
        runs[("set", cname, "A")] = logs_A
        runs[("set", cname, "B")] = logs_B

    # ---- Acceptance summary ------------------------------------------------
    print("\n" + "=" * 80)
    print("  M3 acceptance — Stage B vs Stage A on default_neural_system")
    print("=" * 80)
    # Setpoint: Stage B within 1.5× Stage A AND u not all-zero AND not worse than OL.
    set_rows = [r for r in rows if r[0] == "set"]
    OL_set = next(r for r in set_rows if r[1] == "OL")
    print(f"\n  Open-loop (zero input) setpoint rms_err: A={OL_set[2]:.3f}  B={OL_set[3]:.3f}")
    for r in set_rows:
        if r[1] == "OL":
            continue
        ratio = r[3] / r[2] if r[2] > 0 else float("inf")
        u_nonzero = runs[("set", r[1], "B")]["u"].any()
        beats_OL = r[3] < OL_set[3]
        print(f"  {r[1]}: B/A ratio = {ratio:.3f}  (<= 1.5? {ratio <= 1.5})"
              f"  |  u != 0? {u_nonzero}  |  beats OL? {beats_OL}")

    # ---- Figures -----------------------------------------------------------
    # Fig 1: uncontrolled vs controlled (Stage A and Stage B both shown)
    logs_OL = runs[("set", "OL", "A")]
    logs_A = runs[("set", "MPC", "A")]
    logs_B = runs[("set", "MPC", "B")]
    t = np.arange(T_run)
    fig, axes = plt.subplots(3, 2, figsize=(14, 9.5), sharex=True)
    for col, (logs, label) in enumerate([(logs_A, "Stage A (TRUE)"), (logs_B, "Stage B (FITTED)")]):
        ax_y, ax_z, ax_u = axes[0, col], axes[1, col], axes[2, col]
        # y(t): show a few channels
        for k in range(min(6, system.obs_dim)):
            ax_y.plot(t, logs["y"][:, k], lw=0.8, alpha=0.65)
            ax_y.plot(t, logs_OL["y"][:, k], lw=0.5, alpha=0.25, color="0.5")
        ax_y.set_title(f"y(t) — {label}  (controlled = solid; open-loop = grey)")
        ax_y.set_ylabel("y[0..5]")

        z = logs["y"] @ M.T
        z_OL = logs_OL["y"] @ M.T
        ax_z.plot(t, z[:, 0], color="tab:orange", lw=1.5, label="z[0]")
        ax_z.plot(t, z[:, 1], color="tab:blue", lw=1.5, label="z[1]")
        ax_z.plot(t, z_OL[:, 0], "--", color="tab:orange", lw=0.8, alpha=0.5, label="open z[0]")
        ax_z.plot(t, z_OL[:, 1], "--", color="tab:blue", lw=0.8, alpha=0.5, label="open z[1]")
        ax_z.axhline(ref_set[0], color="tab:orange", lw=0.8, ls=":", alpha=0.5)
        ax_z.axhline(ref_set[1], color="tab:blue", lw=0.8, ls=":", alpha=0.5)
        ax_z.set_ylabel("readout z(t) vs ref")
        ax_z.legend(loc="best", fontsize=8)

        u = logs["u"]
        ax_u.plot(t, u[:, 0], color="tab:green", lw=1.2, label="u[0]")
        ax_u.plot(t, u[:, 1], color="tab:red", lw=1.2, label="u[1]")
        ax_u.axhline(0.0, color="k", lw=0.5)
        ax_u.axhline(1.0, color="k", lw=0.5)
        ax_u.set_ylim(-0.05, 1.05)
        ax_u.set_xlabel("t")
        ax_u.set_ylabel("u(t) ∈ [0,1]")
        ax_u.legend(loc="best", fontsize=8)
    fig.suptitle(f"Fig 1 — Stage A vs Stage B (MPC), feasible setpoint  "
                 f"ref = ({ref_set[0]:.2f}, {ref_set[1]:.2f})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "fig1_uncontrolled_vs_controlled.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_dir / 'fig1_uncontrolled_vs_controlled.png'}")

    # Fig 2: u(t) with saturation shading (LQG vs MPC, Stage A and B)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)
    for r_idx, ctrl in enumerate(["LQG", "MPC"]):
        for c_idx, stage in enumerate(["A", "B"]):
            ax = axes[r_idx, c_idx]
            logs = runs[("set", ctrl, stage)]
            u = logs["u"]
            ax.plot(t, u[:, 0], color="tab:green", lw=1.2, label="u[0]")
            ax.plot(t, u[:, 1], color="tab:red", lw=1.2, label="u[1]")
            sat_mask = (u <= 1e-6) | (u >= 1.0 - 1e-6)
            sat_any = sat_mask.any(axis=1)
            ax.fill_between(t, -0.05, 1.05, where=sat_any, color="0.85", alpha=0.4, label="saturated")
            ax.axhline(0.0, color="k", lw=0.5)
            ax.axhline(1.0, color="k", lw=0.5)
            ax.set_ylim(-0.05, 1.05)
            sat_frac = saturation_fraction(u)
            ax.set_title(f"{ctrl}  Stage {stage}  (sat = {sat_frac:.2f})")
            ax.set_ylabel("u")
            ax.set_xlabel("t")
            if r_idx == 0 and c_idx == 0:
                ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Fig 2 — control inputs with [0,1] saturation", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "fig2_u_with_saturation.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'fig2_u_with_saturation.png'}")

    # Fig 3: reachable zonotope (TRUE and FIT) with targets + achieved points
    fig, ax = plt.subplots(figsize=(8, 7))
    verts_T = zonotope_vertices(G_true, z0_true)
    verts_F = zonotope_vertices(G_fit, z0_fit)
    poly_order = [0, 1, 3, 2, 0]
    ax.plot(verts_T[poly_order, 0], verts_T[poly_order, 1], "-", color="tab:blue", lw=2, label="TRUE zonotope")
    ax.plot(verts_F[poly_order, 0], verts_F[poly_order, 1], "--", color="tab:orange", lw=2, label="FIT zonotope")
    ax.scatter(verts_T[:, 0], verts_T[:, 1], color="tab:blue", marker="o", s=50)
    ax.scatter(verts_F[:, 0], verts_F[:, 1], color="tab:orange", marker="s", s=40)
    ax.scatter(*ref_set, marker="*", s=180, color="black", label=f"feasible target (G·0.5+z0)")
    # Achieved by MPC Stage A and Stage B
    ach_A = (runs[("set", "MPC", "A")]["y"] @ M.T)[-30:].mean(axis=0)
    ach_B = (runs[("set", "MPC", "B")]["y"] @ M.T)[-30:].mean(axis=0)
    ax.scatter(*ach_A, marker="P", s=140, color="tab:green", label=f"MPC Stage A achieved {ach_A.round(2)}")
    ax.scatter(*ach_B, marker="X", s=140, color="tab:purple", label=f"MPC Stage B achieved {ach_B.round(2)}")
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.set_title("Fig 3 — affine reachable zonotope: TRUE vs FIT + targets + MPC achieved")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_reachable_zonotope.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'fig3_reachable_zonotope.png'}")

    return iface, rows


if __name__ == "__main__":
    main()
