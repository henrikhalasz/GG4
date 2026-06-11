"""M4 — Controller comparison on both 1-D and 2-D readouts, real Brain.

Per the M4 directive:

  - settling is reported via :func:`metrics.band_occupancy_settling` so it
    decouples from steady-state error (the M3 metric saturated whenever
    measurement noise exceeded the band; this one is occupancy-based);
  - both readouts (1-D dominant-controllable, 2-D top PCs) are configured
    via :func:`control.readouts.build_readout` — same code path, just a
    different ``kind=…`` config;
  - all 8 controllers compared on each readout for the *hold* task;
  - the integral conclusion stays Brain-honest: integral does NOT help on
    the Brain (M3 showed noise-limited every T_cal). A separate synthetic
    figure illustrates the mechanism on a deliberately mismatched model
    (clearly labelled — NOT a Brain claim);
  - all figures painted through ``viz/style`` (shared colour map,
    plain-language axes, desired-vs-achieved primitive).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent  # week3/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "week 1"))

from analysis.ground_truth import GroundTruthModel  # noqa: E402
from estimator.identify import EstimatorNew  # noqa: E402
from control.plant import BrainPlant, SimulatorPlant  # noqa: E402
from control.observer import SteadyStateKalman  # noqa: E402
from control.controllers import (  # noqa: E402
    OpenLoop, ProportionalFeedback, PI, LQG, LQGI,
    PolePlacement, MPC, OffsetFreeMPC,
)
from control.closed_loop import run_closed_loop  # noqa: E402
from control.readouts import build_readout  # noqa: E402
from control.reachability import steady_state_gain, feasibility  # noqa: E402
import Simulator as sim  # noqa: E402

from metrics import (  # noqa: E402
    band_occupancy_settling, steady_state_error,
    control_effort, saturation_fraction, rms,
)
from viz.style import CONTROLLER_COLORS  # noqa: E402

from experiments.m3_benchmark_study import _probe, brain_plant_factory  # noqa: E402


# ----------------------------------------------------------------------------
# Controllers — common interface (controller_factory(model, M, ref) → ctrl).
# ----------------------------------------------------------------------------
def make_controller(name: str, model: dict, M: np.ndarray, ref: np.ndarray):
    A, B, C = model["A"], model["B"], model["C"]
    a, c = model["a"], model["c"]
    m = B.shape[1]
    if name == "OpenLoop":
        return OpenLoop(input_dim=m)
    if name == "ProportionalFeedback":
        Kp = 0.05 * np.eye(m, M.shape[0])
        return ProportionalFeedback(Kp=Kp, M=M, ref=ref)
    if name == "PI":
        Kp = 0.05 * np.eye(m, M.shape[0])
        Ki = 0.02 * np.eye(m, M.shape[0])
        return PI(Kp=Kp, Ki=Ki, M=M, ref=ref, anti_windup=True)
    if name == "LQG":
        return LQG(A, B, C, M, a=a, c=c, rho=1.0, ref=ref)
    if name == "LQGI":
        return LQGI(A, B, C, M, a=a, c=c, rho=1.0, rho_q=0.01,
                    ref=ref, anti_windup=True)
    if name == "PolePlacement":
        poles = np.linspace(0.55, 0.9, A.shape[0])
        return PolePlacement(A, B, C, M, target_poles=poles, a=a, c=c, ref=ref)
    if name == "MPC":
        return MPC(A, B, C, M, a=a, c=c, horizon=20, rho=0.1, ref=ref)
    if name == "OffsetFreeMPC":
        return OffsetFreeMPC(A, B, C, M, a=a, c=c, horizon=20, rho=0.1,
                              ref=ref, anti_windup=True)
    raise ValueError(f"unknown controller {name!r}")


CONTROLLERS = (
    "OpenLoop", "ProportionalFeedback", "PI",
    "LQG", "LQGI", "PolePlacement", "MPC", "OffsetFreeMPC",
)


# ----------------------------------------------------------------------------
# Hold-task evaluation with the new metrics.
# ----------------------------------------------------------------------------
def evaluate_hold(logs: dict, M: np.ndarray, ref: np.ndarray,
                  noise_floor_readout: float) -> dict:
    z = logs["y"] @ M.T
    occ = band_occupancy_settling(z, ref, tol_frac=0.10,
                                   reference_scale=max(float(np.linalg.norm(ref)), 1.0))
    return dict(
        settle=int(occ["settle"]),
        band_occupancy=float(occ["band_occupancy"]),
        band=float(occ["band"]),
        steady_err=float(steady_state_error(z, ref, tail_frac=0.5)),
        ss_err_x_noise=float(steady_state_error(z, ref, tail_frac=0.5)
                              / max(noise_floor_readout, 1e-12)),
        effort=float(control_effort(logs["u"])),
        sat=float(saturation_fraction(logs["u"])),
        achieved=z[-30:].mean(axis=0).tolist(),
        z=z, u=logs["u"],
    )


def noise_floor_for_readout(M: np.ndarray, R_yardstick: np.ndarray) -> float:
    """Per-step readout noise scale = ``sqrt(trace(M R M^T))``."""
    return float(np.sqrt(np.trace(M @ R_yardstick @ M.T)))


# ----------------------------------------------------------------------------
# Run a controller on the real Brain (seed-specific).
# ----------------------------------------------------------------------------
def run_hold_on_brain(seed: int, model: dict, controller_name: str,
                       M_readout: np.ndarray, ref: np.ndarray,
                       T: int = 300) -> dict:
    plant = brain_plant_factory(seed)
    obs = SteadyStateKalman(model["A"], model["B"], model["C"],
                              model["Q"], model["R"],
                              a=model["a"], c=model["c"])
    ctrl = make_controller(controller_name, model, M_readout, ref)
    logs = run_closed_loop(plant, ctrl, obs, T=T,
                            ref_fn=lambda t: ref)
    return logs


# ----------------------------------------------------------------------------
# Main comparison: 8 controllers × 2 readouts on the real Brain.
# ----------------------------------------------------------------------------
def comparison_on_brain(
    seeds: List[int] = (0, 1),
    T_cal: int = 1000,
    T_run: int = 300,
    log=print,
) -> dict:
    log("\n[brain] 8 controllers × 2 readouts × hold task")
    rows = []
    for s in seeds:
        log(f"\n  [seed {s}]")
        # 1) Calibrate honest model.
        Y_cal, U_cal = _probe(brain_plant_factory, seed=s, T_cal=T_cal,
                              m=2, probe_seed=42 + s)
        est = EstimatorNew().fit(Y_cal, U_cal, n=6, init="n4sid",
                                  max_iter=80, tol=1e-5)
        model = dict(A=est.A, B=est.B, C=est.C, Q=est.Q, R=est.R,
                     a=est.a, c=est.c)
        # 2) Build BOTH readouts from this seed's data (config, not code fork).
        M_1d = build_readout("dominant", A=model["A"], B=model["B"], C=model["C"])
        M_2d = build_readout("pca", Y_probe=Y_cal, k=2)
        # 3) Targets: feasible inside each readout's zonotope.
        gt = GroundTruthModel.load(ROOT / "results" / "ground_truth" /
                                    f"seed_{s}.npz")
        # 1-D target at half the yardstick's reachable extent on dom dir.
        G_y_1d, z0_y_1d = steady_state_gain(gt.A, gt.B, gt.C, M_1d,
                                             a=gt.a, c=gt.c)
        # extent = max over u ∈ [0,1]^m of G_y_1d @ u  (a scalar)
        g_dir = G_y_1d.flatten()  # shape (m,)
        u_max = np.clip(np.sign(g_dir), 0.0, 1.0)
        extent = float(g_dir @ u_max)
        ref_1d = np.array([z0_y_1d[0] + 0.5 * extent])
        # 2-D target at zonotope midpoint of yardstick.
        G_y_2d, z0_y_2d = steady_state_gain(gt.A, gt.B, gt.C, M_2d,
                                             a=gt.a, c=gt.c)
        ref_2d = G_y_2d @ (0.5 * np.ones(gt.B.shape[1])) + z0_y_2d
        # Noise floors per readout.
        nf_1d = noise_floor_for_readout(M_1d, gt.R)
        nf_2d = noise_floor_for_readout(M_2d, gt.R)
        log(f"    1-D readout: ref = {ref_1d.round(3).tolist()}  "
            f"reachable extent = {extent:.2f}  noise floor = {nf_1d:.3f}")
        log(f"    2-D readout: ref = {ref_2d.round(3).tolist()}  "
            f"noise floor = {nf_2d:.3f}")

        for readout_kind, M, ref, nf in (("1d", M_1d, ref_1d, nf_1d),
                                          ("2d", M_2d, ref_2d, nf_2d)):
            log(f"\n    [readout = {readout_kind}]  hold task")
            log(f"      {'ctrl':>22}  {'ss_err':>7}  "
                f"{'x noise':>7}  {'occ':>5}  {'settle':>6}  "
                f"{'effort':>7}  {'sat':>5}")
            for cname in CONTROLLERS:
                logs = run_hold_on_brain(s, model, cname, M, ref, T=T_run)
                m = evaluate_hold(logs, M, ref, nf)
                log(f"      {cname:>22}  {m['steady_err']:7.3f}  "
                    f"{m['ss_err_x_noise']:7.2f}  "
                    f"{m['band_occupancy']:5.2f}  "
                    f"{m['settle']:6d}  "
                    f"{m['effort']:7.1f}  {m['sat']:5.2f}")
                rows.append(dict(seed=s, readout=readout_kind,
                                  controller=cname,
                                  ref=ref.tolist(),
                                  noise_floor=nf, **m))
    return dict(rows=rows)


# ----------------------------------------------------------------------------
# Integral *mechanism* illustration on a deliberately mismatched synthetic.
# Clearly labelled — NOT a Brain claim.
# ----------------------------------------------------------------------------
def integral_mechanism_illustration(log=print) -> dict:
    """Show LQGI removing a steady-state offset on a synthetic system where
    the controller's model is biased (so feedforward is wrong, and integral
    has something to absorb).

    This is **not** a finding about the Brain — the Brain M3 result shows
    the integral does not help there because the error is noise-limited at
    every T_cal. This figure documents the integral mechanism *when bias
    is the bottleneck*, so the report can show "integral works when bias
    dominates; the Brain just isn't in that regime".
    """
    log("\n[mechanism] integral on deliberately mismatched synthetic")
    rng = np.random.default_rng(0)
    n, m, p, T = 2, 2, 2, 400
    A = 0.8 * np.eye(n)
    B = np.eye(m)
    C = np.eye(p)
    Q = 1e-4 * np.eye(n)
    R = 1e-3 * np.eye(p)
    a_true = np.array([0.5, 0.5])    # The truth has a non-trivial drift.
    a_wrong = np.zeros(n)             # Controller is given a = 0 — biased.
    c0 = np.zeros(p)
    M_r = np.eye(2)
    ref = np.array([3.0, 3.0])

    def rollout(controller_factory, observer_factory):
        x = np.zeros(n); ys = np.empty((T, p)); us = np.empty((T, m))
        ctrl = controller_factory(); obs = observer_factory()
        obs.reset(); ctrl.reset()
        u_prev = np.zeros(m)
        for t in range(T):
            y = C @ x + c0 + rng.multivariate_normal(np.zeros(p), R)
            ys[t] = y
            x_hat = obs.filter_step(y, u_prev)
            if hasattr(ctrl, "observe"):
                ctrl.observe(y, x_hat, u_prev)
            if getattr(ctrl, "uses_raw_y", False):
                u = ctrl.compute(y, ref)
            else:
                u = ctrl.compute(x_hat, ref)
            us[t] = u
            x = A @ x + B @ u + a_true + rng.multivariate_normal(np.zeros(n), Q)
            u_prev = u
        return ys, us

    model_wrong = dict(A=A, B=B, C=C, Q=Q, R=R, a=a_wrong, c=c0)
    rng_state = rng.bit_generator.state
    ys_lqg, us_lqg = rollout(
        controller_factory=lambda: make_controller("LQG", model_wrong, M_r, ref),
        observer_factory=lambda: SteadyStateKalman(A, B, C, Q, R, a=a_wrong, c=c0),
    )
    rng.bit_generator.state = rng_state
    ys_lqgi, us_lqgi = rollout(
        controller_factory=lambda: make_controller("LQGI", model_wrong, M_r, ref),
        observer_factory=lambda: SteadyStateKalman(A, B, C, Q, R, a=a_wrong, c=c0),
    )
    ss_lqg = float(np.linalg.norm(ys_lqg[-50:].mean(axis=0) - ref))
    ss_lqgi = float(np.linalg.norm(ys_lqgi[-50:].mean(axis=0) - ref))
    log(f"  LQG (no integral, biased model): steady-state err = {ss_lqg:.3f}")
    log(f"  LQGI (integral, biased model)  : steady-state err = {ss_lqgi:.3f}")
    log(f"  → integral closes the offset by {ss_lqg / max(ss_lqgi, 1e-12):.1f}× "
        f"on this bias-dominated synthetic")
    return dict(ys_lqg=ys_lqg, ys_lqgi=ys_lqgi, us_lqg=us_lqg, us_lqgi=us_lqgi,
                ref=ref, ss_lqg=ss_lqg, ss_lqgi=ss_lqgi)


# ----------------------------------------------------------------------------
# Figure — one question per panel, painted through viz/style colours.
# ----------------------------------------------------------------------------
def plot_m4(comparison: dict, mechanism: dict, out_path: Path) -> None:
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # Aggregate comparison rows by (seed, readout).
    rows = comparison["rows"]
    seeds = sorted(set(r["seed"] for r in rows))
    readouts = ("1d", "2d")

    # ---- Panel A — steady-state error per controller, both readouts -------
    for ridx, ro in enumerate(readouts):
        ax = fig.add_subplot(gs[0, ridx])
        ax.set_title(
            ("1-D readout (dominant controllable direction)\n"
             "× noise floor; 1.0 = best achievable; "
             "OpenLoop column is the no-control reference"
             if ro == "1d" else
             "2-D readout (top 2 PCs)\n"
             "× noise floor; 1.0 = best achievable; "
             "OpenLoop column at ~11× shows the gap closed-loop closes"),
            fontsize=10,
        )
        sub = [r for r in rows if r["readout"] == ro]
        ctrls = CONTROLLERS
        x = np.arange(len(ctrls))
        w = 0.8 / max(len(seeds), 1)
        for i, s in enumerate(seeds):
            vals = []
            for c in ctrls:
                cand = [r for r in sub if r["seed"] == s and r["controller"] == c]
                vals.append(cand[0]["ss_err_x_noise"] if cand else np.nan)
            ax.bar(x + (i - (len(seeds) - 1) / 2) * w, vals, width=w,
                   color=[CONTROLLER_COLORS.get(c, "#999") for c in ctrls],
                   alpha=0.5 + 0.5 * (i + 1) / len(seeds),
                   label=f"seed {s}")
        ax.set_xticks(x)
        ax.set_xticklabels(ctrls, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("steady-state error\n(× per-readout noise floor)")
        ax.axhline(1.0, color="black", ls=":", lw=1.1,
                    label="noise floor (best achievable)")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")

    # ---- Panel B — band-occupancy per controller, both readouts ----------
    for ridx, ro in enumerate(readouts):
        ax = fig.add_subplot(gs[1, ridx])
        ax.set_title(
            f"band-occupancy — {ro.upper()} readout\n"
            "fraction of final 50% inside ±10% band; "
            "1.0 = always inside, 0.0 = never",
            fontsize=10,
        )
        sub = [r for r in rows if r["readout"] == ro]
        ctrls = CONTROLLERS
        x = np.arange(len(ctrls))
        w = 0.8 / max(len(seeds), 1)
        for i, s in enumerate(seeds):
            vals = []
            for c in ctrls:
                cand = [r for r in sub if r["seed"] == s and r["controller"] == c]
                vals.append(cand[0]["band_occupancy"] if cand else np.nan)
            ax.bar(x + (i - (len(seeds) - 1) / 2) * w, vals, width=w,
                   color=[CONTROLLER_COLORS.get(c, "#999") for c in ctrls],
                   alpha=0.5 + 0.5 * (i + 1) / len(seeds),
                   label=f"seed {s}")
        ax.set_xticks(x)
        ax.set_xticklabels(ctrls, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("band occupancy")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3, axis="y")

    # ---- Panel C — sample desired-vs-achieved on seed 0, 1-D readout -----
    ax = fig.add_subplot(gs[0, 2])
    seed0_1d = [r for r in rows if r["seed"] == 0 and r["readout"] == "1d"]
    cmap_focus = ("OpenLoop", "ProportionalFeedback", "LQG", "MPC")
    for cname in cmap_focus:
        cand = [r for r in seed0_1d if r["controller"] == cname]
        if not cand:
            continue
        z = cand[0]["z"]
        col = CONTROLLER_COLORS.get(cname, "#444")
        ax.plot(np.arange(z.shape[0]), z[:, 0], "-",
                lw=1.4 if cname != "OpenLoop" else 1.0,
                color=col, alpha=0.95 if cname != "OpenLoop" else 0.5,
                label=cname)
    if seed0_1d:
        ref_y = seed0_1d[0]["ref"][0]
        ax.axhline(ref_y, ls="--", color="black", lw=1.0, label="desired (target)")
    ax.set_xlabel("time step")
    ax.set_ylabel("1-D readout")
    ax.set_title("Hold task — seed 0, 1-D readout\n(desired vs achieved)",
                 fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)

    # ---- Panel D — sample desired-vs-achieved on seed 0, 2-D readout -----
    ax = fig.add_subplot(gs[1, 2])
    seed0_2d = [r for r in rows if r["seed"] == 0 and r["readout"] == "2d"]
    for cname in cmap_focus:
        cand = [r for r in seed0_2d if r["controller"] == cname]
        if not cand:
            continue
        z = cand[0]["z"]
        col = CONTROLLER_COLORS.get(cname, "#444")
        ax.plot(z[:, 0], z[:, 1], "-",
                lw=1.2 if cname != "OpenLoop" else 0.8,
                color=col, alpha=0.9 if cname != "OpenLoop" else 0.4,
                label=cname)
    if seed0_2d:
        r0 = seed0_2d[0]["ref"]
        ax.scatter([r0[0]], [r0[1]], marker="*", s=220, color="black",
                   edgecolor="white", lw=0.6, label="target", zorder=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Hold task — seed 0, 2-D readout\n(trajectory in PC1×PC2)",
                 fontsize=10)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # ---- Panel E — integral mechanism on a synthetic mismatched model ----
    ax = fig.add_subplot(gs[2, :2])
    T = mechanism["ys_lqg"].shape[0]
    t = np.arange(T)
    ax.plot(t, mechanism["ys_lqg"][:, 0], "-",
            color=CONTROLLER_COLORS["LQG"], lw=1.6, label="LQG (no integral)")
    ax.plot(t, mechanism["ys_lqgi"][:, 0], "-",
            color=CONTROLLER_COLORS["LQGI"], lw=1.6, label="LQGI (+integral)")
    ax.axhline(mechanism["ref"][0], color="black", ls="--", lw=1.0,
               label="desired (target)")
    ax.set_xlabel("time step")
    ax.set_ylabel("output y[0]")
    ax.set_title(
        "Mechanism illustration — integral on a *deliberately mismatched* synthetic\n"
        f"(controller given a=0; truth has a=[0.5, 0.5]).  "
        f"LQG steady err = {mechanism['ss_lqg']:.3f},  "
        f"LQGI steady err = {mechanism['ss_lqgi']:.3f}  "
        f"({mechanism['ss_lqg'] / max(mechanism['ss_lqgi'], 1e-12):.0f}× reduction).\n"
        "NOT a Brain claim — see Panel F.",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # ---- Panel F — Brain regime note (the honest conclusion) -------------
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    ax.text(0.0, 1.0,
            "Brain conclusion (M3 data-efficiency sweep):\n"
            "LQG ≈ LQGI at every T_cal ∈ {100..2000}.\n"
            "Residual error is noise-limited at every probe\n"
            "budget — there is no feedforward bias for the\n"
            "integrator to absorb.\n\n"
            "Integral does NOT help on the Brain hold task.\n\n"
            "Panel E shows the integral mechanism on a\n"
            "deliberately bias-dominated synthetic so the\n"
            "reader can see what the integrator does when\n"
            "bias *is* the bottleneck. The Brain just isn't\n"
            "in that regime.",
            ha="left", va="top", fontsize=9,
            family="DejaVu Sans",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#fff7e1",
                      edgecolor="#d9b86b"))

    fig.suptitle(
        "M4 — Controller comparison on both readouts; fixed metrics; "
        "honest integral conclusion",
        fontsize=13, y=0.99,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(
    seeds: List[int] = (0, 1),
    T_cal: int = 1000,
    T_run: int = 300,
    out_dir: Path = None,
) -> dict:
    if out_dir is None:
        out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        log_lines.append(msg)

    log("=" * 78)
    log("  M4 — Controller comparison on both readouts + integral mechanism")
    log("=" * 78)
    t0 = time.time()
    comparison = comparison_on_brain(list(seeds), T_cal=T_cal,
                                       T_run=T_run, log=log)
    mechanism = integral_mechanism_illustration(log=log)
    log(f"\n[done]  wall-clock = {time.time() - t0:.1f}s")

    fig_path = out_dir / "m4_controller_comparison.png"
    plot_m4(comparison, mechanism, fig_path)
    log(f"figure: {fig_path}")

    text_path = out_dir / "m4_controller_comparison.txt"
    text_path.write_text("\n".join(log_lines), encoding="utf-8")
    log(f"log:    {text_path}")

    return dict(comparison=comparison, mechanism=mechanism,
                figure=fig_path, text=text_path)


if __name__ == "__main__":
    main()
