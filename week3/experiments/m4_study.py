"""M4 experiment: full study (figs 4-7, 9, 10) + Brain seeds.

Per spec §10 M4 acceptance:
    - Full controller matrix across scenarios + ≥5 Brain seeds.
    - Sweeps over noise scale (×Q, ×R), T_cal, n, ρ, MPC H.
    - Figures 4-7, 9, 10 + Brain multi-seed figure.
    - Failure modes documented (suppression floor, hidden-input authority loss).

No dither, no online recalibration (spec hard rule).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "week 1"))

import Simulator as sim  # noqa: E402

from control.plant import SimulatorPlant, BrainPlant  # noqa: E402
from control.observer import SteadyStateKalman  # noqa: E402
from control.controllers import OpenLoop, ProportionalFeedback, LQG, MPC, PolePlacement  # noqa: E402
from control.closed_loop import run_closed_loop  # noqa: E402
from control.control_interface import IdentifiedSystem  # noqa: E402
from control.reachability import steady_state_gain, feasibility  # noqa: E402
from control.control_interface import compute_pca_readout  # noqa: E402
from control.metrics import (
    rms, control_effort, saturation_fraction, state_r2_aligned, spectral_radius  # noqa: E402
)


SCENARIOS = [
    ("default", sim.default_neural_system, 4, 16),
    ("input_aligned", sim.input_aligned_system, 4, 16),
    ("input_blind", sim.input_blind_system, 4, 16),
    ("slow_drift", sim.slow_drift_system, 4, 16),
    ("closed_loop", sim.closed_loop_system, 4, 16),
    ("hidden_input", sim.hidden_input_system, 5, 10),
]


def scenario_pca_readout(system, T_probe: int = 600, probe_seed: int = 42,
                          plant_seed: int = 43) -> np.ndarray:
    """Compute the 2-leading-PC readout from a short open-loop+probe run on this
    scenario. Used by Stage-A experiments so the readout choice matches what
    Stage-B would compute from real probe data.
    """
    plant = SimulatorPlant(system, seed=plant_seed)
    rng = np.random.default_rng(probe_seed)
    U = rng.uniform(0, 1, size=(T_probe, system.input_dim))
    y0 = np.asarray(plant.measure(), dtype=float)
    Y = np.empty((T_probe, system.obs_dim))
    Y[0] = y0
    plant.next_state(U[0])
    for t in range(1, T_probe):
        Y[t] = plant.measure()
        plant.next_state(U[t])
    M, _ = compute_pca_readout(Y, k=2)
    return M


def build_controller(name, A, B, C, M, a, c, ref, *, rho=0.1, H=20):
    if name == "open":
        return OpenLoop(input_dim=B.shape[1])
    if name == "PropFB":
        return ProportionalFeedback(Kp=0.05 * np.eye(B.shape[1], M.shape[0]), M=M, ref=ref)
    if name == "LQG":
        return LQG(A, B, C, M, a=a, c=c, rho=rho, ref=ref)
    if name == "PolePlace":
        return PolePlacement(A, B, C, M, target_poles=np.linspace(0.5, 0.85, A.shape[0]),
                             a=a, c=c, ref=ref)
    if name == "MPC":
        return MPC(A, B, C, M, a=a, c=c, horizon=H, rho=rho, ref=ref)
    raise ValueError(name)


def run_loop(plant, ctrl, A, B, C, Q, R, a, c, T, ref):
    obs = SteadyStateKalman(A, B, C, Q, R, a=a, c=c)
    logs = run_closed_loop(plant, ctrl, obs, T=T, ref_fn=lambda t: ref)
    return logs


def make_stage_a_plant_and_obs(system, seed):
    plant = SimulatorPlant(system, seed=seed)
    return plant


def study_simulator_matrix(out_dir, T=200, seed_plant=7):
    """Run all 5 controllers on 3 objectives across scenarios (Stage A on truth)."""
    print("\n=== Simulator scenarios — all controllers × 3 objectives ===")
    objectives = {}
    state_r2_rows = []
    convergence_traces = {}

    controllers = ["open", "PropFB", "LQG", "PolePlace", "MPC"]
    for sname, fac, n_state, p in SCENARIOS:
        system = fac(seed=0, obs_dim=p)
        n = system.A.shape[0]
        M = scenario_pca_readout(system)
        a0 = np.zeros(n); c0 = np.zeros(system.obs_dim)
        G, z0 = steady_state_gain(system.A, system.B, system.C, M, a=a0, c=c0)
        ref_sup = np.zeros(M.shape[0])
        ref_feas = G @ np.array([0.5, 0.5]) + z0
        ref_inf = 3.0 * (G @ np.array([1.0, 1.0])) + z0
        targets = {"suppression": ref_sup, "feasible_setpoint": ref_feas, "infeasible_setpoint": ref_inf}
        for obj_name, ref in targets.items():
            for cname in controllers:
                plant = make_stage_a_plant_and_obs(system, seed_plant)
                ctrl = build_controller(cname, system.A, system.B, system.C, M, a0, c0, ref)
                logs = run_loop(plant, ctrl, system.A, system.B, system.C, system.Q, system.R, a0, c0, T, ref)
                z = logs["y"] @ M.T
                e = rms(z - ref)
                eff = control_effort(logs["u"])
                sat = saturation_fraction(logs["u"])
                objectives.setdefault((sname, obj_name), []).append((cname, e, eff, sat))
                # Sample convergence trace on the headline (default + feasible_setpoint).
                if sname == "default" and obj_name == "feasible_setpoint" and cname in ("open", "LQG", "MPC"):
                    convergence_traces[cname] = (z[:, 0], ref[0])
                # State R²: only meaningful with same-basis comparison.
                if logs.get("x_true") is not None and logs.get("x_hat") is not None:
                    r2 = state_r2_aligned(logs["x_true"], logs["x_hat"])
                    state_r2_rows.append((sname, cname, obj_name, r2))

    # ---- Fig 4: per-objective achieved error per controller -----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    obj_list = ["suppression", "feasible_setpoint", "infeasible_setpoint"]
    width = 0.15
    x = np.arange(len(SCENARIOS))
    for j, obj_name in enumerate(obj_list):
        ax = axes[j]
        for k, cname in enumerate(controllers):
            errs = [next(e for c, e, *_ in objectives[(s[0], obj_name)] if c == cname) for s in SCENARIOS]
            ax.bar(x + (k - 2) * width, errs, width=width, label=cname)
        ax.set_xticks(x)
        ax.set_xticklabels([s[0] for s in SCENARIOS], rotation=30, ha="right", fontsize=8)
        ax.set_title(obj_name)
        ax.set_ylabel("RMS error (vs ref)")
        if j == 2:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8, ncol=2)
    fig.suptitle("Fig 4 — per-objective achieved error across controllers (Stage A)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "fig4_objective_error_summary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir/'fig4_objective_error_summary.png'}")

    # ---- Fig 5: error vs effort on suppression (default) -------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    rows = objectives[("default", "suppression")]
    for cname, e, eff, _ in rows:
        ax.scatter(eff, e, s=60, label=cname)
        ax.annotate(cname, (eff, e), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("control effort  Σ ||u||")
    ax.set_ylabel("z-RMS")
    ax.set_title("Fig 5 — suppression error vs effort (Stage A, default_neural_system)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_error_vs_effort.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir/'fig5_error_vs_effort.png'}")

    # ---- Fig 9: state R² per scenario (estimator's perspective via x_smooth alignment) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_data = {}
    for sname, fac, n_state, p in SCENARIOS:
        # Calibrate the system; report state R² from the estimator's smoothed states.
        system = fac(seed=0, obs_dim=p)
        n = system.A.shape[0]
        cp = SimulatorPlant(system, seed=43)
        iface = IdentifiedSystem().calibrate(
            cp, T_cal=800, n=n, m=system.input_dim, seed=42,
            true_model=dict(A=system.A, B=system.B, C=system.C,
                             a=np.zeros(n), c=np.zeros(system.obs_dim)),
        )
        # x_true for the same probe.
        cp2 = SimulatorPlant(system, seed=43)
        cp2.reset(seed=43)
        # Re-simulate the same trajectory to grab x_true (same RNG seed -> same noise).
        rng = np.random.default_rng(42)
        U = rng.uniform(0, 1, size=(800, system.input_dim))
        data = system.simulate(800, U=U)
        x_true = data["x"][:800]
        x_smooth = iface.estimator.x_smooth
        # Affine alignment
        Xh_c = x_smooth - x_smooth.mean(axis=0)
        Xt_c = x_true - x_true.mean(axis=0)
        T_align, *_ = np.linalg.lstsq(Xh_c, Xt_c, rcond=None)
        pred = Xh_c @ T_align
        ss_res = float(np.sum((Xt_c - pred) ** 2))
        ss_tot = float(np.sum(Xt_c ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        bar_data[sname] = (r2, iface.diagnostics["CB_frob"], iface.degenerate_flags)
        print(f"  {sname}: state R² = {r2:.4f}  ||CB||={iface.diagnostics['CB_frob']:.3f}  flags={iface.degenerate_flags or 'none'}")
    snames = list(bar_data.keys())
    r2s = [bar_data[s][0] for s in snames]
    colors = ["tab:red" if bar_data[s][2] else "tab:blue" for s in snames]
    ax.bar(snames, r2s, color=colors)
    ax.axhline(0.9, color="0.5", ls=":", lw=0.8, label="0.9 floor")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("aligned state R² (estimator)")
    ax.set_title("Fig 9 — estimator state R² per scenario  (red = degenerate-flag fired)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig9_state_r2_per_scenario.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir/'fig9_state_r2_per_scenario.png'}")

    # ---- Fig 10: convergence/settling on the primary objective (default + feasible_setpoint) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    t = np.arange(T)
    for cname, (trace, ref0) in convergence_traces.items():
        ax.plot(t, trace, label=f"{cname}  z[0]", lw=1.3)
    ax.axhline(list(convergence_traces.values())[0][1], color="k", ls="--", lw=0.8,
               label=f"ref z[0] = {list(convergence_traces.values())[0][1]:.2f}")
    ax.set_xlabel("t")
    ax.set_ylabel("readout z[0](t)")
    ax.set_title("Fig 10 — convergence on the primary objective (Stage A, default, feasible setpoint)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig10_convergence_settling.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir/'fig10_convergence_settling.png'}")

    return objectives


def sweep_noise_and_horizon(out_dir, T=200, seed_plant=7):
    """Robustness vs noise (×Q, ×R) and sensitivity sweeps over T_cal, n, ρ, H."""
    print("\n=== Robustness and sensitivity sweeps (Stage B on default) ===")
    # Noise sweep (Stage A) — multiply Q and R by scale; LQG vs MPC.
    scales = [0.25, 0.5, 1.0, 2.0, 4.0]
    fig, ax = plt.subplots(figsize=(8, 5))
    for cname in ["LQG", "MPC"]:
        rms_vs_scale = []
        for s in scales:
            system = sim.default_neural_system(seed=0, obs_dim=16)
            system.Q = system.Q * s
            system.R = system.R * s
            n = system.A.shape[0]
            M = scenario_pca_readout(system)
            a0 = np.zeros(n); c0 = np.zeros(system.obs_dim)
            G, z0 = steady_state_gain(system.A, system.B, system.C, M, a=a0, c=c0)
            ref = np.zeros(M.shape[0])  # suppression
            plant = SimulatorPlant(system, seed=seed_plant)
            ctrl = build_controller(cname, system.A, system.B, system.C, M, a0, c0, ref)
            logs = run_loop(plant, ctrl, system.A, system.B, system.C, system.Q, system.R, a0, c0, T, ref)
            z = logs["y"] @ M.T
            rms_vs_scale.append(rms(z))
        ax.plot(scales, rms_vs_scale, "-o", label=cname)
    ax.set_xscale("log")
    ax.set_xlabel("noise scale × (Q, R)")
    ax.set_ylabel("z-RMS")
    ax.set_title("Fig 6 — robustness vs noise scale (Stage A, default, suppression)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_robustness_vs_noise.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir/'fig6_robustness_vs_noise.png'}")

    # Sensitivity sweeps (Stage B) — vary T_cal, n, ρ, MPC H on the feasible setpoint.
    system = sim.default_neural_system(seed=0, obs_dim=16)
    n_truth = system.A.shape[0]
    M = scenario_pca_readout(system)
    a0 = np.zeros(n_truth); c0 = np.zeros(system.obs_dim)
    G_true, z0_true = steady_state_gain(system.A, system.B, system.C, M, a=a0, c=c0)
    ref = G_true @ np.array([0.5, 0.5]) + z0_true

    def fit_and_run(T_cal=800, n_lat=n_truth, rho_val=0.1, mpc_H=20, ctrl_name="MPC"):
        cp = SimulatorPlant(system, seed=43)
        iface = IdentifiedSystem().calibrate(cp, T_cal=T_cal, n=n_lat,
                                              m=system.input_dim, seed=42)
        A, B, C, Q, R, a, c = iface.params()
        plant = SimulatorPlant(system, seed=seed_plant)
        ctrl = build_controller(ctrl_name, A, B, C, M, a, c, ref, rho=rho_val, H=mpc_H)
        logs = run_loop(plant, ctrl, A, B, C, Q, R, a, c, T, ref)
        z = logs["y"] @ M.T
        return rms(z - ref), control_effort(logs["u"])

    T_cals = [120, 250, 500, 800, 1500]
    n_lats = [2, 3, 4, 5, 6]
    rhos = [1e-3, 1e-2, 1e-1, 1, 10]
    Hs = [5, 10, 20, 30]
    res_T = [fit_and_run(T_cal=tc) for tc in T_cals]
    res_n = [fit_and_run(n_lat=nl) for nl in n_lats]
    res_r = [fit_and_run(rho_val=r) for r in rhos]
    res_H = [fit_and_run(mpc_H=h) for h in Hs]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, sweep_name, xs, ys in [
        (axes[0, 0], "T_cal", T_cals, res_T),
        (axes[0, 1], "n (latent)", n_lats, res_n),
        (axes[1, 0], "ρ (LQR R-weight)", rhos, res_r),
        (axes[1, 1], "MPC horizon H", Hs, res_H),
    ]:
        rms_err = [r[0] for r in ys]
        eff = [r[1] for r in ys]
        ax.plot(xs, rms_err, "-o", color="tab:red", label="rms_err")
        ax2 = ax.twinx()
        ax2.plot(xs, eff, "--s", color="tab:blue", label="effort")
        ax.set_xlabel(sweep_name)
        ax.set_ylabel("rms_err")
        ax2.set_ylabel("effort")
        if sweep_name == "ρ (LQR R-weight)":
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_title(sweep_name)
    fig.suptitle("Fig 7 — sensitivity sweeps (Stage B on default, MPC feasible setpoint)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "fig7_sensitivity_sweeps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir/'fig7_sensitivity_sweeps.png'}")

    # Print numbers for the writeup.
    print("\n  T_cal sweep:")
    for tc, r in zip(T_cals, res_T): print(f"    T_cal={tc:5d}: rms_err={r[0]:.3f}, eff={r[1]:.2f}")
    print("\n  n sweep:")
    for nl, r in zip(n_lats, res_n): print(f"    n={nl}: rms_err={r[0]:.3f}, eff={r[1]:.2f}")
    print("\n  ρ sweep:")
    for rr, r in zip(rhos, res_r): print(f"    ρ={rr:.1e}: rms_err={r[0]:.3f}, eff={r[1]:.2f}")
    print("\n  H sweep:")
    for h, r in zip(Hs, res_H): print(f"    H={h:2d}: rms_err={r[0]:.3f}, eff={r[1]:.2f}")


def brain_seeds_study(out_dir, n_seeds=5, T=200, T_cal=600, n_lat=4):
    """Run the headline Stage-B suppression and setpoint experiment on the Brain."""
    print("\n=== Brain ≥5-seed study (Stage B via estimator_new) ===")
    from GG4 import Brain  # noqa: E402

    rows = []
    seed_list = list(range(n_seeds))
    for seed in seed_list:
        # CALIBRATE on a fresh brain instance.
        b_cal = Brain(random_seed=seed)
        cp = BrainPlant(b_cal)
        iface = IdentifiedSystem().calibrate(cp, T_cal=T_cal, n=n_lat, m=2, seed=42)
        A, B, C, Q, R, a, c = iface.params()
        M = iface.readout_M

        # OpenLoop suppression baseline on a FRESH brain (same seed, fresh instance).
        b_OL = Brain(random_seed=seed)
        plant_OL = BrainPlant(b_OL)
        ctrl_OL = OpenLoop(input_dim=2)
        obs_OL = SteadyStateKalman(A, B, C, Q, R, a=a, c=c)
        logs_OL = run_closed_loop(plant_OL, ctrl_OL, obs_OL, T=T)

        # LQG suppression with FITTED params on a FRESH brain.
        b_lqg = Brain(random_seed=seed)
        plant_lqg = BrainPlant(b_lqg)
        ctrl_lqg = LQG(A, B, C, M, a=a, c=c, rho=0.1, ref=np.zeros(2))
        obs_lqg = SteadyStateKalman(A, B, C, Q, R, a=a, c=c)
        logs_lqg = run_closed_loop(plant_lqg, ctrl_lqg, obs_lqg, T=T,
                                     ref_fn=lambda t: np.zeros(2))

        # MPC setpoint at the FIT-zonotope midpoint (feasibility decided in fit basis).
        G_fit, z0_fit = steady_state_gain(A, B, C, M, a=a, c=c)
        ref_set = G_fit @ np.array([0.5, 0.5]) + z0_fit
        b_mpc = Brain(random_seed=seed)
        plant_mpc = BrainPlant(b_mpc)
        ctrl_mpc = MPC(A, B, C, M, a=a, c=c, horizon=20, rho=0.1, ref=ref_set)
        obs_mpc = SteadyStateKalman(A, B, C, Q, R, a=a, c=c)
        logs_mpc = run_closed_loop(plant_mpc, ctrl_mpc, obs_mpc, T=T,
                                     ref_fn=lambda t: ref_set)

        z_OL = (logs_OL["y"] @ M.T)
        z_lqg = (logs_lqg["y"] @ M.T)
        z_mpc = (logs_mpc["y"] @ M.T)
        rms_OL = rms(z_OL)
        rms_lqg = rms(z_lqg)
        rms_mpc_set = rms(z_mpc - ref_set)
        ach_mpc = z_mpc[-30:].mean(axis=0)
        eff_lqg = control_effort(logs_lqg["u"])
        eff_mpc = control_effort(logs_mpc["u"])
        flags = iface.degenerate_flags
        rho = iface.diagnostics["A_spectral_radius"]
        cb = iface.diagnostics["CB_frob"]

        rows.append(dict(
            seed=seed,
            rho_A=rho,
            CB=cb,
            flags=flags,
            ref_set=ref_set,
            ach_mpc=ach_mpc,
            rms_OL=rms_OL,
            rms_lqg=rms_lqg,
            rms_mpc_set=rms_mpc_set,
            eff_lqg=eff_lqg,
            eff_mpc=eff_mpc,
        ))
        print(f"  seed {seed}: rho={rho:.3f}  ||CB||={cb:.3f}  flags={flags or 'none'}")
        print(f"    OL z-RMS={rms_OL:.3f}  LQG z-RMS={rms_lqg:.3f}  ratio={rms_lqg/rms_OL:.3f}")
        print(f"    MPC setpoint rms_err={rms_mpc_set:.3f}  ref={ref_set.round(3)} ach={ach_mpc.round(3)}")

    # ---- Figure: per-seed Brain summary ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    seeds = [r["seed"] for r in rows]
    rms_OLs = [r["rms_OL"] for r in rows]
    rms_lqgs = [r["rms_lqg"] for r in rows]
    rms_mpc_sets = [r["rms_mpc_set"] for r in rows]

    w = 0.35
    x = np.arange(len(seeds))
    axes[0].bar(x - w/2, rms_OLs, width=w, label="open-loop", color="0.6")
    axes[0].bar(x + w/2, rms_lqgs, width=w, label="LQG Stage B", color="tab:blue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"seed {s}" for s in seeds])
    axes[0].set_ylabel("z-RMS  (suppression)")
    axes[0].set_title("Brain suppression: open-loop vs LQG Stage B")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(x, rms_mpc_sets, color="tab:orange")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"seed {s}" for s in seeds])
    axes[1].set_ylabel("MPC setpoint rms_err")
    axes[1].set_title("Brain MPC at fit-zonotope midpoint")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Brain multi-seed Stage-B study (T_cal={T_cal}, T_run={T}, n={n_lat})")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "fig_brain_multiseed.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir/'fig_brain_multiseed.png'}")

    # Print summary
    safer_or_equal = sum(1 for r in rows if r["rms_lqg"] <= 1.5 * r["rms_OL"])
    beats_OL = sum(1 for r in rows if r["rms_lqg"] < r["rms_OL"])
    print(f"\n  Summary: {safer_or_equal}/{len(rows)} seeds have LQG ≤ 1.5× OL")
    print(f"           {beats_OL}/{len(rows)} seeds have LQG < OL")
    return rows


def main():
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("M4 — full study (Simulator matrix + sweeps + Brain seeds)")
    print("=" * 84)
    objectives = study_simulator_matrix(out_dir)
    sweep_noise_and_horizon(out_dir)
    brain_rows = brain_seeds_study(out_dir, n_seeds=5)


if __name__ == "__main__":
    main()
