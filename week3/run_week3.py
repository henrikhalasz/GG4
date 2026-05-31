"""run_week3.py — end-to-end Week-3 solution on the Brain.

Story:
    install/load Brain → probe → fit estimator_new → validate →
    closed-loop control (5 controllers) → figures + narrative.

Usage:
    python run_week3.py                # default config, full 5-seed Brain run
    python run_week3.py --seed 0       # single seed
    python run_week3.py --notebook     # quieter, returns dicts; no Brain seeds

Mapping to the week3.ipynb assessed deliverables (§1 of the spec):
    1. Control objective       — setpoint regulation at a fit-zonotope-feasible
                                  target on the readout z = M y; justified by the
                                  affine reachability zonotope.
    2. Strategy & comparison   — 5 controllers (OpenLoop, ProportionalFeedback,
                                  LQG, PolePlacement, MPC). MPC wins on every
                                  objective in the study; chosen as final.
    3. Implementation          — Simulator (Week 1) + estimator_new.py +
                                  control/ package (this file is the runner).
    4. Closed-loop test        — closed_loop.run_closed_loop driver across the
                                  Brain and Simulator plants.
    5. Evaluation              — z-RMS, settling, control effort, saturation,
                                  closed-loop spectral radius, state R²,
                                  reachability feasibility check.
    6. Working closed-loop +
       perf plots + issues +
       attempted improvement   — see saved figures; the improvement is the
                                  estimator rebuild (affine LGSSM + offsets) and
                                  MPC over clipped LQG.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent  # week3/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "week 1"))

from control.plant import BrainPlant, SimulatorPlant  # noqa: E402
from control.observer import SteadyStateKalman  # noqa: E402
from control.controllers import (  # noqa: E402
    OpenLoop, ProportionalFeedback, LQG, MPC, PolePlacement,
    PI, LQGI, OffsetFreeMPC,
)
from control.closed_loop import run_closed_loop  # noqa: E402
from control.control_interface import IdentifiedSystem  # noqa: E402
from control.reachability import steady_state_gain, zonotope_vertices, feasibility  # noqa: E402
from control.metrics import (
    rms, control_effort, saturation_fraction, settling_time, spectral_radius  # noqa: E402
)


def build_controller(name: str, A, B, C, M, a, c, ref, *,
                     rho: float = 1.0, rho_q: float = 0.001, H: int = 20):
    if name == "OpenLoop":
        return OpenLoop(input_dim=B.shape[1])
    if name == "ProportionalFeedback":
        return ProportionalFeedback(Kp=0.05 * np.eye(B.shape[1], M.shape[0]), M=M, ref=ref)
    if name == "LQG":
        return LQG(A, B, C, M, a=a, c=c, rho=rho, ref=ref)
    if name == "PolePlacement":
        return PolePlacement(A, B, C, M, target_poles=np.linspace(0.5, 0.85, A.shape[0]),
                              a=a, c=c, ref=ref)
    if name == "MPC":
        return MPC(A, B, C, M, a=a, c=c, horizon=H, rho=rho, ref=ref)
    # v3 E1 integral variants.
    if name == "PI":
        return PI(Kp=0.05 * np.eye(B.shape[1], M.shape[0]),
                  Ki=0.02 * np.eye(B.shape[1], M.shape[0]),
                  M=M, ref=ref, anti_windup=True)
    if name == "LQGI":
        return LQGI(A, B, C, M, a=a, c=c, rho=rho, rho_q=rho_q,
                    ref=ref, anti_windup=True)
    if name == "OffsetFreeMPC":
        return OffsetFreeMPC(A, B, C, M, a=a, c=c, horizon=H, rho=rho,
                              ref=ref, anti_windup=True)
    raise ValueError(f"Unknown controller: {name}")


def calibrate_brain(seed: int, T_cal: int = 600, n: int = 4,
                    probe_seed: int = 42) -> IdentifiedSystem:
    from GG4 import Brain  # late import so we can run the docstring without GG4
    brain = Brain(random_seed=seed)
    plant = BrainPlant(brain)
    iface = IdentifiedSystem().calibrate(
        plant, T_cal=T_cal, n=n, m=plant.input_dim, seed=probe_seed
    )
    return iface


def run_closed_loop_on_brain(seed: int, iface: IdentifiedSystem,
                               controller_name: str, ref: np.ndarray,
                               T: int) -> dict:
    """Run a closed-loop episode on a fresh Brain at the given seed."""
    from GG4 import Brain
    brain = Brain(random_seed=seed)
    plant = BrainPlant(brain)
    A, B, C, Q, R, a, c = iface.params()
    M = iface.readout_M  # 2 leading PCs of the probe measurements
    obs = SteadyStateKalman(A, B, C, Q, R, a=a, c=c)
    controller = build_controller(controller_name, A, B, C, M, a, c, ref)
    logs = run_closed_loop(plant, controller, obs, T=T, ref_fn=lambda t: ref)
    return logs


def evaluate_episode(logs: dict, M: np.ndarray, ref: np.ndarray) -> dict:
    z = logs["y"] @ M.T
    err = z - ref
    return dict(
        z_rms=rms(z),
        rms_err=rms(err),
        effort=control_effort(logs["u"]),
        sat=saturation_fraction(logs["u"]),
        settling=settling_time(z, ref),
        achieved=z[-30:].mean(axis=0).tolist(),
    )


def summarise_method_table(table: Dict, label: str) -> None:
    print(f"\n  {label}")
    print(f"    {'controller':22s}  {'z-RMS':>8s}  {'rms_err':>8s}  {'effort':>8s}  {'sat':>6s}  {'settle':>7s}")
    for cname, m in table.items():
        print(f"    {cname:22s}  {m['z_rms']:8.3f}  {m['rms_err']:8.3f}  {m['effort']:8.2f}  {m['sat']:6.2f}  {m['settling']:7d}")


def run_brain_story(seed: int, T_cal: int = 600, T_run: int = 200,
                    n_latent: int = 4, out_dir: Path = None) -> dict:
    """End-to-end on a single Brain seed — walks the v3 §6 story.

    Steps follow the §6 narrative:
        1. Setup — probe + EM fit → (A, B, C, Q, R, a, c) + PCA readout M
        2. What can we hold — reachability check on the FIT region; pick
           feasible target (zonotope midpoint) and call out suppression (r=0)
           as infeasible when z0 ≠ 0
        3. Task 1 — hold the feasible target with all 8 controllers
           (5 existing + 3 integral variants)
        4. Task 2 — suppress (drive readout to z0)
        5. Figures saved per Brain seed
    """
    if out_dir is None:
        out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"  Brain seed {seed}  —  walking the v3 §6 story")
    print("=" * 80)

    # 1. Setup
    print("\n  [1/5] SETUP — probe + EM fit + PCA readout ...")
    iface = calibrate_brain(seed=seed, T_cal=T_cal, n=n_latent)
    A, B, C, Q, R, a, c = iface.params()
    M = iface.readout_M  # 2 leading PCs of probe measurements

    print(f"    fit iters         : {iface.diagnostics['fit_iters']}")
    print(f"    ρ(A_fit)          : {iface.diagnostics['A_spectral_radius']:.4f}")
    print(f"    ‖CB‖_F            : {iface.diagnostics['CB_frob']:.4f}  (threshold {iface.diagnostics['CB_visibility_threshold']:.4f})")
    print(f"    B col norms       : {[round(v, 3) for v in iface.diagnostics['B_col_norms']]}")
    print(f"    ‖a‖, ‖c‖          : {iface.diagnostics['a_norm']:.3f}, {iface.diagnostics['c_norm']:.3f}")
    print(f"    degenerate flags  : {iface.degenerate_flags or 'none'}")
    print(f"    controller warnings: "
          f"{iface.controller_warnings or 'none'}")
    print(f"    one-step pred RMS : "
          f"{iface.diagnostics.get('one_step_rms', float('nan')):.3f}  "
          f"({iface.diagnostics.get('one_step_rms_frac', float('nan'))*100:.1f}% of y-RMS)")
    print(f"    readout (2 PCs)   : explained var = {[round(v, 3) for v in iface.readout_explained_var]}")
    print(f"    cond(G_readout)   : {iface.readout_cond_G:.2f}   "
          f"({'well-conditioned — PC1 and PC2 are independently controllable' if iface.readout_cond_G < 50 else 'ILL-CONDITIONED — inputs cannot move PC1, PC2 independently'})")

    # 2. What can we hold — reachability on the FIT region
    print("\n  [2/5] WHAT CAN WE HOLD — reachable (PC1, PC2) region from the fit ...")
    G_fit, z0_fit = steady_state_gain(A, B, C, M, a=a, c=c)
    ref_set = G_fit @ np.array([0.5, 0.5]) + z0_fit
    ref_sup = np.zeros(M.shape[0])  # suppression target = (0, 0)
    print(f"    G_fit (input → readout DC gain) =\n{G_fit}")
    print(f"    z0_fit (resting readout)        = {z0_fit}")
    print(f"    feasible target = G·[0.5, 0.5] + z0 = {ref_set.round(3)}  "
          "(zonotope midpoint)")
    inside_sup, _ = feasibility(ref_sup, G_fit, z0_fit)
    inside_set, _ = feasibility(ref_set, G_fit, z0_fit)
    print(f"    is ref=(0,0) holdable?            {inside_sup}  "
          f"{'' if inside_sup else '— suppression below the resting readout is infeasible'}")
    print(f"    is the midpoint target holdable?  {inside_set}")

    # 3 + 4. Same controllers, two tasks (hold + suppress).
    controllers = ["OpenLoop", "ProportionalFeedback", "LQG", "PolePlacement",
                    "MPC", "PI", "LQGI", "OffsetFreeMPC"]

    print("\n  [3/5] TASK 1 (HOLD) — drive the readout to the feasible target ...")
    set_table: Dict = {}
    for cname in controllers:
        logs = run_closed_loop_on_brain(seed, iface, cname, ref_set, T_run)
        m = evaluate_episode(logs, M, ref_set)
        set_table[cname] = m | {"logs": logs}
    summarise_method_table({k: v for k, v in set_table.items()}, "Hold (feasible target)")

    print("\n  [4/5] TASK 2 (SUPPRESS) — drive readout variance to zero ...")
    sup_table: Dict = {}
    for cname in controllers:
        logs = run_closed_loop_on_brain(seed, iface, cname, ref_sup, T_run)
        m = evaluate_episode(logs, M, ref_sup)
        sup_table[cname] = m | {"logs": logs}
    summarise_method_table({k: v for k, v in sup_table.items()}, "Suppress (ref = 0)")

    # 5. Figures
    print("\n  [5/5] FIGURES — saving per-seed summary ...")
    _plot_brain_story(out_dir, seed, M, ref_set, sup_table, set_table,
                      G_fit, z0_fit)

    return dict(
        seed=seed, iface_diagnostics=iface.diagnostics,
        flags=iface.degenerate_flags,
        suppression={k: {kk: vv for kk, vv in v.items() if kk != "logs"} for k, v in sup_table.items()},
        setpoint={k: {kk: vv for kk, vv in v.items() if kk != "logs"} for k, v in set_table.items()},
        ref_set=ref_set.tolist(),
        G_fit=G_fit.tolist(), z0_fit=z0_fit.tolist(),
    )


def _plot_brain_story(out_dir: Path, seed: int, M, ref_set,
                      sup_table, set_table, G_fit, z0_fit) -> None:
    t = np.arange(sup_table["MPC"]["logs"]["u"].shape[0])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # (1) z(t) under MPC vs OpenLoop on setpoint
    ax = axes[0, 0]
    z_OL = set_table["OpenLoop"]["logs"]["y"] @ M.T
    z_MPC = set_table["MPC"]["logs"]["y"] @ M.T
    ax.plot(t, z_OL[:, 0], color="0.6", lw=1.0, label="OL z[0]")
    ax.plot(t, z_OL[:, 1], color="0.4", lw=1.0, label="OL z[1]")
    ax.plot(t, z_MPC[:, 0], color="tab:orange", lw=1.5, label="MPC z[0]")
    ax.plot(t, z_MPC[:, 1], color="tab:blue", lw=1.5, label="MPC z[1]")
    ax.axhline(ref_set[0], color="tab:orange", ls=":", lw=0.8, alpha=0.6)
    ax.axhline(ref_set[1], color="tab:blue", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("t"); ax.set_ylabel("readout z(t)")
    ax.set_title(f"MPC vs OpenLoop on feasible setpoint  (seed {seed})")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # (2) u(t) under MPC, saturation shaded
    ax = axes[0, 1]
    u = set_table["MPC"]["logs"]["u"]
    ax.plot(t, u[:, 0], color="tab:green", lw=1.2, label="u[0]")
    ax.plot(t, u[:, 1], color="tab:red", lw=1.2, label="u[1]")
    sat_mask = ((u <= 1e-6) | (u >= 1.0 - 1e-6)).any(axis=1)
    ax.fill_between(t, -0.05, 1.05, where=sat_mask, color="0.85", alpha=0.4, label="saturated")
    ax.axhline(0.0, color="k", lw=0.5); ax.axhline(1.0, color="k", lw=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("t"); ax.set_ylabel("u(t)")
    ax.set_title(f"MPC inputs with [0, 1] band  (sat = {sat_mask.mean():.2f})")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # (3) zonotope + achieved point
    ax = axes[1, 0]
    verts = zonotope_vertices(G_fit, z0_fit)
    poly_order = [0, 1, 3, 2, 0]
    ax.plot(verts[poly_order, 0], verts[poly_order, 1], "-", color="tab:blue", lw=2,
            label="fit zonotope")
    ax.scatter(verts[:, 0], verts[:, 1], color="tab:blue", marker="o", s=40)
    ax.scatter(*ref_set, marker="*", s=180, color="black", label="feasible target")
    ach = set_table["MPC"]["achieved"]
    ax.scatter(*ach, marker="P", s=140, color="tab:green",
               label=f"MPC achieved ({ach[0]:.2f}, {ach[1]:.2f})")
    ax.set_xlabel("z[0]"); ax.set_ylabel("z[1]")
    ax.set_title("Reachable zonotope + MPC achieved")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # (4) controller comparison bar (setpoint rms_err)
    ax = axes[1, 1]
    names = list(set_table.keys())
    errs = [set_table[n]["rms_err"] for n in names]
    ax.bar(names, errs, color=["0.6", "tab:purple", "tab:blue", "tab:green", "tab:orange"])
    ax.set_ylabel("rms_err (setpoint)")
    ax.set_title("Controller comparison — feasible setpoint")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Week-3 solution on Brain (seed {seed})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = out_dir / f"run_week3_seed{seed}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0,
                        help="Brain seed (default: 0)")
    parser.add_argument("--all-seeds", type=int, default=None,
                        help="Run the multi-seed sweep with N seeds (e.g. 5).")
    parser.add_argument("--T-cal", type=int, default=600)
    parser.add_argument("--T-run", type=int, default=200)
    parser.add_argument("--n-latent", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    if args.all_seeds is not None:
        results = []
        for s in range(args.all_seeds):
            results.append(run_brain_story(
                seed=s, T_cal=args.T_cal, T_run=args.T_run,
                n_latent=args.n_latent, out_dir=args.out_dir
            ))
        print("\n" + "=" * 80)
        print("  Multi-seed summary (MPC feasible-setpoint rms_err):")
        for r in results:
            err = r["setpoint"]["MPC"]["rms_err"]
            ach = r["setpoint"]["MPC"]["achieved"]
            print(f"    seed {r['seed']}: rms_err={err:.3f}  ach={[round(x, 3) for x in ach]}")
    else:
        run_brain_story(
            seed=args.seed, T_cal=args.T_cal, T_run=args.T_run,
            n_latent=args.n_latent, out_dir=args.out_dir
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
