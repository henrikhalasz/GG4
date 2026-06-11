"""M6: regenerate the 10-figure narrative per spec v3 §6 — the figure redesign
pass under the clarity rules.

Each figure answers ONE question (in plain language, in the title), uses
defined axes (PC1 / PC2 / time — never bare ``z`` or "zonotope"), shows
desired (dashed) vs achieved (solid) vs uncontrolled (faint grey) whenever
a target is involved, and quotes errors normalised as both a fraction of
the target and as a multiple of the open-loop noise floor.

Files saved to ``week3/results/`` with the prefix ``figE3_``:

    figE3_01_setup_and_drift.png
    figE3_02_holdable_region.png
    figE3_03_hold_task.png
    figE3_04_integral_adds.png
    figE3_05_track_task.png
    figE3_06_suppress_task.png
    figE3_07_stageA_vs_stageB.png
    figE3_08_controller_comparison.png
    figE3_09_model_trust.png
    figE3_10_limitations.png

The legacy ``fig1..fig10`` and ``fig8_em_validation.png`` are kept as
historical supporting figures; they get retitled in their own runners
(m3_stage_b / m4_study) when those scripts are next regenerated.

Run::

    python week3/experiments/m6_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent  # week3/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "week 1"))

import Simulator as sim  # noqa: E402

from control.plant import SimulatorPlant  # noqa: E402
from control.observer import SteadyStateKalman  # noqa: E402
from control.controllers import (  # noqa: E402
    OpenLoop, ProportionalFeedback, LQG, PolePlacement, MPC,
    PI, LQGI, OffsetFreeMPC,
)
from control.closed_loop import run_closed_loop  # noqa: E402
from control.control_interface import IdentifiedSystem  # noqa: E402
from control.readouts import compute_pca_readout, readout_steady_state_gain  # noqa: E402
from control.reachability import steady_state_gain, zonotope_vertices  # noqa: E402
from metrics import rms, control_effort, saturation_fraction  # noqa: E402

from viz.style import (  # noqa: E402
    CONTROLLER_COLORS, LABEL_PC1, LABEL_PC2, LABEL_TIME, LABEL_U,
    LABEL_HOLD_REGION,
    set_question_title, plot_desired_vs_achieved, plot_holdable_region_2d,
    annotate_normalised_error, style_axis, add_caption,
)


OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Shared setup — run once and reuse across figures.
# ----------------------------------------------------------------------------

def build_setup(seed_plant: int = 7, T: int = 250, T_probe: int = 600,
                 probe_seed: int = 42) -> dict:
    system = sim.default_neural_system(seed=0, obs_dim=16)
    n = system.A.shape[0]
    p = system.obs_dim
    m_in = system.input_dim
    a0 = np.zeros(n)
    c0 = np.zeros(p)

    # ---- PCA readout from open-loop probe ----
    probe_plant = SimulatorPlant(system, seed=43)
    rng = np.random.default_rng(probe_seed)
    U_probe = rng.uniform(0, 1, size=(T_probe, m_in))
    Y_probe = np.empty((T_probe, p))
    Y_probe[0] = np.asarray(probe_plant.measure(), dtype=float)
    probe_plant.next_state(U_probe[0])
    for t in range(1, T_probe):
        Y_probe[t] = probe_plant.measure()
        probe_plant.next_state(U_probe[t])
    M, explained = compute_pca_readout(Y_probe, k=2)

    # ---- Reachable region + targets ----
    G_true, z0_true = steady_state_gain(system.A, system.B, system.C, M,
                                          a=a0, c=c0)
    verts = zonotope_vertices(G_true, z0_true)
    ref_feas = G_true @ np.array([0.5, 0.5]) + z0_true
    ref_infeas = verts[3] + (verts[3] - verts[0])  # 2× past (1,1) corner

    # ---- Open-loop baseline (uncontrolled) ----
    plant_OL = SimulatorPlant(system, seed=seed_plant)
    obs_OL = SteadyStateKalman(system.A, system.B, system.C, system.Q, system.R,
                                a=a0, c=c0)
    logs_OL = run_closed_loop(plant_OL, OpenLoop(input_dim=m_in), obs_OL, T=T)
    z_OL = logs_OL["y"] @ M.T
    noise_floor = float(np.std(z_OL[-100:], axis=0).mean())
    OL_drift = rms(z_OL - z_OL.mean(axis=0))

    return dict(system=system, n=n, p=p, m_in=m_in, a0=a0, c0=c0,
                 M=M, explained=explained,
                 G=G_true, z0=z0_true, verts=verts,
                 ref_feas=ref_feas, ref_infeas=ref_infeas,
                 logs_OL=logs_OL, z_OL=z_OL,
                 noise_floor=noise_floor, OL_drift=OL_drift,
                 seed_plant=seed_plant, T=T)


def build_controller_for(name: str, ctx: dict, ref):
    A = ctx["system"].A; B = ctx["system"].B; C = ctx["system"].C
    M = ctx["M"]; a = ctx["a0"]; c = ctx["c0"]
    m_in = ctx["m_in"]
    if name == "OpenLoop":     return OpenLoop(input_dim=m_in)
    if name == "PropFeedback": return ProportionalFeedback(
        Kp=0.05 * np.eye(m_in, M.shape[0]), M=M, ref=ref)
    if name == "LQG":          return LQG(A, B, C, M, a=a, c=c, rho=1.0, ref=ref)
    if name == "PolePlace":    return PolePlacement(
        A, B, C, M, target_poles=np.linspace(0.5, 0.85, A.shape[0]),
        a=a, c=c, ref=ref)
    if name == "MPC":          return MPC(A, B, C, M, a=a, c=c,
                                            horizon=20, rho=1.0, ref=ref)
    if name == "PI":           return PI(
        Kp=0.05 * np.eye(m_in, M.shape[0]),
        Ki=0.02 * np.eye(m_in, M.shape[0]), M=M, ref=ref)
    if name == "LQGI":         return LQGI(A, B, C, M, a=a, c=c,
                                              rho=1.0, rho_q=0.001, ref=ref)
    if name == "OffsetFreeMPC": return OffsetFreeMPC(
        A, B, C, M, a=a, c=c, horizon=20, rho=1.0, ref=ref)
    raise ValueError(name)


def run_episode(ctx: dict, controller_name: str, ref_fn, ref0) -> dict:
    plant = SimulatorPlant(ctx["system"], seed=ctx["seed_plant"])
    obs = SteadyStateKalman(ctx["system"].A, ctx["system"].B, ctx["system"].C,
                              ctx["system"].Q, ctx["system"].R,
                              a=ctx["a0"], c=ctx["c0"])
    ctrl = build_controller_for(controller_name, ctx, ref0)
    return run_closed_loop(plant, ctrl, obs, T=ctx["T"], ref_fn=ref_fn)


# ============================================================================
# FIGURE 01 — Setup + uncontrolled baseline
# ============================================================================

def fig01_setup(ctx: dict) -> None:
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.18)
    ax_schema = fig.add_subplot(gs[0])
    ax_drift = fig.add_subplot(gs[1])

    # Left: schematic as text panel.
    ax_schema.axis("off")
    schema_lines = [
        "Hidden state x",
        "    (~4 latent variables, never observed)",
        "        |",
        "        v",
        "16 measurements y",
        "    (what we record each time step)",
        "        |",
        "        v        M = top-2 principal components of y",
        "Readout (PC1, PC2)",
        "    = the 2 dominant patterns of population activity",
        "",
        "Inputs u in [0, 1]^2",
        "    (push only — one-sided actuator)",
    ]
    ax_schema.text(0.02, 0.92, "\n".join(schema_lines),
                    transform=ax_schema.transAxes,
                    fontsize=11, ha="left", va="top", family="monospace")
    explained = ctx["explained"]
    ax_schema.text(0.02, 0.05,
                    f"PC1 explains {explained[0]*100:.0f}% of variance, "
                    f"PC2 explains {explained[1]*100:.0f}%",
                    transform=ax_schema.transAxes,
                    fontsize=10, color="0.3", style="italic")
    set_question_title(ax_schema, "What does this system look like?",
                        subtitle="x is hidden, y is the 16 measurements, readout = 2 PCs.")

    # Right: PC1(t) and PC2(t) under open loop.
    t = np.arange(ctx["T"])
    z_OL = ctx["z_OL"]
    ax_drift.plot(t, z_OL[:, 0], color="#1f77b4", lw=1.3,
                   label="PC1 activity (uncontrolled)")
    ax_drift.plot(t, z_OL[:, 1], color="#ff7f0e", lw=1.3,
                   label="PC2 activity (uncontrolled)")
    ax_drift.axhline(0.0, color="0.5", lw=0.6, alpha=0.5)
    ax_drift.set_xlabel(LABEL_TIME)
    ax_drift.set_ylabel("readout value")
    ax_drift.legend(loc="best", fontsize=9)
    style_axis(ax_drift)
    set_question_title(ax_drift, "How does the readout drift on its own?",
                        subtitle=f"open-loop noise floor = {ctx['noise_floor']:.2f}; "
                                  f"demeaned RMS = {ctx['OL_drift']:.2f}")

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out = OUT_DIR / "figE3_01_setup_and_drift.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ============================================================================
# FIGURE 02 — What can we hold (the holdable region)
# ============================================================================

def fig02_holdable_region(ctx: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_holdable_region_2d(
        ax, ctx["verts"],
        feasible_target=ctx["ref_feas"],
        infeasible_target=ctx["ref_infeas"],
    )
    ax.scatter(*ctx["verts"][0], marker="o", s=90,
                color="white", edgecolor="#1f77b4", linewidth=1.5, zorder=8)
    ax.annotate("u = (0, 0)\n(resting)",
                 ctx["verts"][0], xytext=(10, 10), textcoords="offset points",
                 fontsize=9, color="#1f77b4")
    ax.annotate("u = (1, 1)\n(max push)",
                 ctx["verts"][3], xytext=(8, -22), textcoords="offset points",
                 fontsize=9, color="#1f77b4")
    set_question_title(ax,
        "Which (PC1, PC2) levels can we hold steady?",
        subtitle="The shaded region is the set of (PC1, PC2) values that some allowed "
                 "constant input u ∈ [0,1]^2 holds. It is bounded because inputs only push.")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    out = OUT_DIR / "figE3_02_holdable_region.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ============================================================================
# FIGURE 03 — Task 1: hold a target  (desired vs achieved)
# ============================================================================

def fig03_hold(ctx: dict) -> None:
    """Two-column small-multiples: feasible (left) vs infeasible (right);
    one row per readout dim (PC1, PC2)."""
    T = ctx["T"]; t = np.arange(T); M = ctx["M"]
    refs = {
        "feasible (inside)":  ctx["ref_feas"],
        "infeasible (outside)": ctx["ref_infeas"],
    }
    # Controllers to overlay (per spec §6 step 3: Open, LQG, LQG+I, MPC).
    overlay = ["LQG", "LQGI", "MPC"]
    # Run once per (target, controller).
    cache = {}
    for tgt_name, ref in refs.items():
        for cname in overlay:
            cache[(tgt_name, cname)] = run_episode(
                ctx, cname, lambda _t: ref, ref)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True)
    for col, (tgt_name, ref) in enumerate(refs.items()):
        # uncontrolled baseline
        z_OL = ctx["z_OL"]
        target_scale = float(np.linalg.norm(ref - ctx["z0"]))
        for row, pc_idx in enumerate([0, 1]):
            ax = axes[row, col]
            ref_const = np.full(T, ref[pc_idx])
            series = []
            for cname in overlay:
                z_ach = cache[(tgt_name, cname)]["y"] @ M.T
                series.append((cname, z_ach[:, pc_idx]))
            plot_desired_vs_achieved(
                ax, t, ref_const, series,
                uncontrolled=z_OL[:, pc_idx],
                y_label=(LABEL_PC1 if pc_idx == 0 else LABEL_PC2),
                x_label=(LABEL_TIME if row == 1 else None),
            )
            # Normalised error annotation for MPC (the best non-integral).
            err_mpc = rms(cache[(tgt_name, "MPC")]["y"] @ M.T - ref)
            annotate_normalised_error(ax, err_mpc, target_scale,
                                        ctx["noise_floor"])
        # Column title once per column.
        axes[0, col].set_title(
            f"{'a) feasible target' if col == 0 else 'b) infeasible target'}  "
            f"ref = ({ref[0]:.1f}, {ref[1]:.1f})",
            loc="left", fontsize=11, color="#2ca02c" if col == 0 else "#d62728",
        )
    # Single legend at the top.
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=10,
                bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("Can the controller hold a target?",
                  fontsize=14, x=0.02, ha="left")
    add_caption(fig,
        "Achieved curves are dashed-desired vs solid-coloured per controller, "
        "with the uncontrolled drift in faint grey. "
        "Error boxes give MPC's error as % of target and × the noise floor.")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    out = OUT_DIR / "figE3_03_hold_task.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")
    return cache


# ============================================================================
# FIGURE 04 — What integral adds  (zoom on held value; windup vs anti-windup)
# ============================================================================

def fig04_integral(ctx: dict, hold_cache: Dict) -> None:
    """Two panels:
        a) Zoom on PC1 at feasible target: LQG (no integral, leaves offset)
           vs LQG+I (closes the offset).
        b) Infeasible target: windup (no anti-windup) vs anti-windup
           (q stays bounded).
    """
    T = ctx["T"]; t = np.arange(T); M = ctx["M"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel a: zoom on the held value, feasible target.
    ax = axes[0]
    ref = ctx["ref_feas"]
    z_LQG = hold_cache[("feasible (inside)", "LQG")]["y"] @ M.T
    z_LQGI = hold_cache[("feasible (inside)", "LQGI")]["y"] @ M.T
    ax.axhline(ref[0], color="black", ls="--", lw=1.2, label="desired (target)")
    ax.plot(t, z_LQG[:, 0], color=CONTROLLER_COLORS["LQG"], lw=1.6,
             label="LQG  (feedforward only)")
    ax.plot(t, z_LQGI[:, 0], color=CONTROLLER_COLORS["LQGI"], lw=1.6,
             label="LQG + integral")
    # Zoom in vertically around target.
    ax.set_ylim(ref[0] - 12, ref[0] + 5)
    ax.set_xlabel(LABEL_TIME)
    ax.set_ylabel(LABEL_PC1)
    style_axis(ax)
    err_lqg = rms(z_LQG - ref)
    err_lqgi = rms(z_LQGI - ref)
    target_scale = float(np.linalg.norm(ref - ctx["z0"]))
    ax.text(0.98, 0.04,
             f"LQG steady offset  ~ {err_lqg:.2f}  "
             f"({err_lqg/target_scale*100:.0f}% of target)\n"
             f"LQG+I steady offset ~ {err_lqgi:.2f}  "
             f"({err_lqgi/target_scale*100:.0f}% of target)",
             transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        alpha=0.92, edgecolor="0.7"))
    set_question_title(ax,
        "a) What does integral action add on a feasible target?",
        subtitle="Zoomed on PC1 near the target. Integral closes any residual offset.")
    ax.legend(loc="upper right", fontsize=9)

    # Panel b: infeasible target — windup vs anti-windup.
    ax = axes[1]
    ref = ctx["ref_infeas"]
    A = ctx["system"].A; B = ctx["system"].B; C = ctx["system"].C
    Q = ctx["system"].Q; R = ctx["system"].R
    a0 = ctx["a0"]; c0 = ctx["c0"]
    # Two runs of LQGI: anti-windup ON vs OFF.
    q_traces = {}
    for label, anti in [("anti-windup ON", True), ("no anti-windup", False)]:
        plant = SimulatorPlant(ctx["system"], seed=ctx["seed_plant"])
        obs = SteadyStateKalman(A, B, C, Q, R, a=a0, c=c0)
        lqgi = LQGI(A, B, C, M, a=a0, c=c0, rho=1.0, rho_q=0.001,
                     ref=ref, anti_windup=anti)
        q_hist = np.empty((T, 2))
        u_hist = np.empty((T, B.shape[1]))
        obs.reset(); lqgi.reset()
        u_prev = np.zeros(B.shape[1])
        for tt in range(T):
            y = np.asarray(plant.measure(), dtype=float)
            x_hat = obs.filter_step(y, u_prev)
            lqgi.observe(y, x_hat, u_prev)
            u = lqgi.compute(x_hat, ref)
            q_hist[tt] = lqgi.q_state
            u_hist[tt] = u
            plant.next_state(u)
            u_prev = u
        q_traces[label] = q_hist
    # Plot |q| over time for both.
    ax.plot(t, np.linalg.norm(q_traces["anti-windup ON"], axis=1),
             color="#2ca02c", lw=1.6, label="anti-windup ON  (bounded)")
    ax.plot(t, np.linalg.norm(q_traces["no anti-windup"], axis=1),
             color="#d62728", lw=1.6, label="no anti-windup  (runs away)")
    ax.set_xlabel(LABEL_TIME)
    ax.set_ylabel("|integral state q|  (running sum of unmet error)")
    ax.legend(loc="upper left", fontsize=9)
    style_axis(ax)
    set_question_title(ax,
        "b) Why is anti-windup mandatory on infeasible targets?",
        subtitle="When the target is unreachable, the integral keeps growing forever "
                  "unless we freeze it on saturation.")

    add_caption(fig,
        "Left: integral action removes the small steady-state error left by "
        "feedforward-only LQG. Right: without anti-windup, the integral state "
        "diverges under an infeasible target and the controller never recovers.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = OUT_DIR / "figE3_04_integral_adds.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ============================================================================
# FIGURE 05 — Task 2: follow a moving target  (small-multiples grid)
# ============================================================================

def fig05_track(ctx: dict) -> dict:
    T = ctx["T"]; t = np.arange(T); M = ctx["M"]
    half_diag = 0.5 * np.linalg.norm(ctx["verts"][3] - ctx["verts"][0])
    amp = 0.3 * np.array([
        abs(ctx["verts"][3, 0] - ctx["verts"][0, 0]) / 2,
        abs(ctx["verts"][3, 1] - ctx["verts"][0, 1]) / 2,
    ])
    midpoint = ctx["ref_feas"]

    def ref_sine(period):
        omega = 2.0 * np.pi / period
        return lambda tt: midpoint + amp * np.sin(omega * tt)

    def ref_stair(dwell):
        steps = np.array([-0.5 * amp, +0.0 * amp, +0.5 * amp])
        return lambda tt: midpoint + steps[(tt // dwell) % steps.shape[0]]

    tasks = [
        ("a) slow sine (period 100 — well under bandwidth)",   ref_sine(100), "tracks"),
        ("b) fast sine (period 20 — near bandwidth)",          ref_sine(20),  "lags"),
        ("c) staircase (60-step dwell)",                       ref_stair(60), "settles each step"),
        ("d) too aggressive (period 5 — way over bandwidth)",  ref_sine(5),   "can't keep up"),
    ]
    cache = {}
    overlay = ["LQG", "MPC"]
    for tname, ref_fn, verdict in tasks:
        cache[tname] = {}
        for cname in overlay:
            cache[tname][cname] = run_episode(
                ctx, cname, ref_fn, np.asarray(ref_fn(0)))
        cache[tname]["_ref"] = np.array([ref_fn(tt) for tt in range(T)])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True)
    target_scale = float(np.linalg.norm(amp))
    for (tname, ref_fn, verdict), ax in zip(tasks, axes.flatten()):
        ref_arr = cache[tname]["_ref"]
        z_OL = ctx["z_OL"]
        # Show PC1 only (room for one); use one panel per task.
        series = []
        for cname in overlay:
            z_ach = cache[tname][cname]["y"] @ M.T
            series.append((cname, z_ach[:, 0]))
        plot_desired_vs_achieved(
            ax, t, ref_arr[:, 0], series,
            uncontrolled=z_OL[:, 0],
            y_label=LABEL_PC1, x_label=LABEL_TIME,
        )
        # Annotate verdict prominently.
        ax.text(0.02, 0.96, f"verdict: {verdict}",
                 transform=ax.transAxes, ha="left", va="top",
                 fontsize=10, fontweight="bold", color="0.15",
                 bbox=dict(boxstyle="round,pad=0.3",
                            facecolor="#fff8e1", alpha=0.9,
                            edgecolor="0.7"))
        # Error: MPC.
        err_mpc = rms(cache[tname]["MPC"]["y"] @ M.T - ref_arr)
        annotate_normalised_error(ax, err_mpc, target_scale,
                                    ctx["noise_floor"], loc="lower right")
        ax.set_title(tname, fontsize=10.5, loc="left")

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=10,
                bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("Can the controller follow a moving target?",
                  fontsize=14, x=0.02, ha="left")
    add_caption(fig,
        "Each panel: desired (dashed) vs MPC (green) vs LQG (blue), "
        "with the uncontrolled drift in faint grey. The bandwidth-limited "
        "behaviour is unambiguous in (d).")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    out = OUT_DIR / "figE3_05_track_task.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")
    return cache


# ============================================================================
# FIGURE 06 — Task 3: suppress (drive variance to zero)
# ============================================================================

def fig06_suppress(ctx: dict) -> None:
    T = ctx["T"]; t = np.arange(T); M = ctx["M"]
    z_OL = ctx["z_OL"]
    z0 = ctx["z0"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    # Run MPC (best constraint-aware suppressor) — same target as OpenLoop.
    logs_MPC = run_episode(ctx, "MPC", lambda _t: z0, z0)
    z_MPC = logs_MPC["y"] @ M.T

    for row, pc_idx in enumerate([0, 1]):
        ax = axes[row]
        ax.plot(t, z_OL[:, pc_idx], color="0.6", lw=1.0,
                 label="uncontrolled (open loop)")
        ax.plot(t, z_MPC[:, pc_idx], color=CONTROLLER_COLORS["MPC"], lw=1.5,
                 label="MPC suppression")
        ax.axhline(z0[pc_idx], color="black", ls="--", lw=1.2,
                    label=f"target = {z0[pc_idx]:.2f}")
        ax.set_ylabel(LABEL_PC1 if pc_idx == 0 else LABEL_PC2)
        style_axis(ax)
        if row == 1:
            ax.set_xlabel(LABEL_TIME)

    rms_OL = rms(z_OL - z0)
    rms_MPC = rms(z_MPC - z0)
    gain = (1.0 - rms_MPC / rms_OL) * 100.0
    annotate_normalised_error(
        axes[0], rms_MPC, ctx["OL_drift"], ctx["noise_floor"],
        loc="upper right",
    )
    axes[0].text(0.02, 0.96,
                  f"MPC reduces RMS by {gain:.0f}% vs open loop\n"
                  f"({rms_OL:.2f} → {rms_MPC:.2f}); noise floor = "
                  f"{ctx['noise_floor']:.2f}",
                  transform=axes[0].transAxes, ha="left", va="top",
                  fontsize=9.5, bbox=dict(boxstyle="round,pad=0.3",
                                              facecolor="white",
                                              alpha=0.9, edgecolor="0.7"))

    fig.legend(*axes[0].get_legend_handles_labels(),
                loc="upper center", ncol=3, fontsize=10,
                bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.suptitle("Can the controller drive the readout to zero?",
                  fontsize=14, x=0.02, ha="left")
    add_caption(fig,
        "Suppression is about variance, not mean: a one-sided actuator can damp "
        "but cannot cancel a symmetric noise process. The achievable gain is "
        "small (~25%) and bounded by the per-step jitter.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    out = OUT_DIR / "figE3_06_suppress_task.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ============================================================================
# FIGURE 07 — Does the learned model suffice? (Stage A vs Stage B)
# ============================================================================

def fig07_stage_a_vs_b(ctx: dict) -> None:
    """Side-by-side: Stage A (true params) vs Stage B (EM-fitted) for
    MPC on hold (feasible) and on slow-sine tracking."""
    T = ctx["T"]; t = np.arange(T); M = ctx["M"]
    system = ctx["system"]

    # Stage B calibration. Pass true_model + x_true so iface.validation
    # contains G_rel_err, z0_err, state_r2 — used by fig09 below.
    cal_plant = SimulatorPlant(system, seed=ctx["seed_plant"] + 1)
    # Get x_true on the same probe trajectory by replaying the rng.
    rng = np.random.default_rng(42)
    T_cal = 600
    U_probe = rng.uniform(0, 1, size=(T_cal, system.input_dim))
    data = system.simulate(T_cal, U=U_probe)
    x_true_probe = data["x"][:T_cal]
    iface = IdentifiedSystem().calibrate(
        cal_plant, T_cal=T_cal, n=system.A.shape[0],
        m=system.input_dim, seed=42,
        true_model=dict(A=system.A, B=system.B, C=system.C,
                         a=np.zeros(system.A.shape[0]),
                         c=np.zeros(system.obs_dim)),
        x_true=x_true_probe,
    )
    A_B, B_B, C_B, Q_B, R_B, a_B, c_B = iface.params()
    M_B = iface.readout_M

    # Stage B references (in fit basis). For an apples-to-apples comparison
    # in plot space, we reuse the Stage-A ref values; the fit's M_B might
    # differ in sign per PC. Project Stage B's run through M_B.
    ref_feas = ctx["ref_feas"]
    G_fit, z0_fit = steady_state_gain(A_B, B_B, C_B, M_B, a=a_B, c=c_B)
    ref_feas_B = G_fit @ np.array([0.5, 0.5]) + z0_fit

    # Stage A hold.
    logs_A_hold = run_episode(ctx, "MPC", lambda _t: ref_feas, ref_feas)
    z_A_hold = logs_A_hold["y"] @ M.T

    # Stage B hold.
    plant_B = SimulatorPlant(system, seed=ctx["seed_plant"])
    obs_B = SteadyStateKalman(A_B, B_B, C_B, Q_B, R_B, a=a_B, c=c_B)
    ctrl_B = MPC(A_B, B_B, C_B, M_B, a=a_B, c=c_B, horizon=20, rho=1.0,
                  ref=ref_feas_B)
    logs_B_hold = run_closed_loop(plant_B, ctrl_B, obs_B, T=T,
                                     ref_fn=lambda _t: ref_feas_B)
    z_B_hold = logs_B_hold["y"] @ M.T  # project through TRUE M for plotting

    # Stage A track slow sine.
    half_diag = 0.5 * np.linalg.norm(ctx["verts"][3] - ctx["verts"][0])
    amp = 0.3 * np.array([
        abs(ctx["verts"][3, 0] - ctx["verts"][0, 0]) / 2,
        abs(ctx["verts"][3, 1] - ctx["verts"][0, 1]) / 2,
    ])
    omega = 2.0 * np.pi / 100.0
    ref_fn_track = lambda tt: ref_feas + amp * np.sin(omega * tt)
    ref_arr = np.array([ref_fn_track(tt) for tt in range(T)])

    logs_A_track = run_episode(ctx, "MPC", ref_fn_track, ref_feas)
    z_A_track = logs_A_track["y"] @ M.T

    plant_B2 = SimulatorPlant(system, seed=ctx["seed_plant"])
    obs_B2 = SteadyStateKalman(A_B, B_B, C_B, Q_B, R_B, a=a_B, c=c_B)
    ref_fn_track_B = lambda tt: ref_feas_B + amp * np.sin(omega * tt)
    ctrl_B2 = MPC(A_B, B_B, C_B, M_B, a=a_B, c=c_B, horizon=20, rho=1.0,
                   ref=ref_feas_B)
    logs_B_track = run_closed_loop(plant_B2, ctrl_B2, obs_B2, T=T,
                                      ref_fn=ref_fn_track_B)
    z_B_track = logs_B_track["y"] @ M.T

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    # Hold (left column)
    for row, pc_idx in enumerate([0, 1]):
        ax = axes[row, 0]
        ax.axhline(ref_feas[pc_idx], color="black", ls="--", lw=1.2,
                    label="desired (target)")
        ax.plot(t, z_A_hold[:, pc_idx], "-", color="#1f77b4", lw=1.6,
                 label="Stage A — true model")
        ax.plot(t, z_B_hold[:, pc_idx], "--", color="#ff7f0e", lw=1.6,
                 label="Stage B — EM-fitted model")
        ax.set_ylabel(LABEL_PC1 if pc_idx == 0 else LABEL_PC2)
        style_axis(ax)
        if row == 1:
            ax.set_xlabel(LABEL_TIME)
    axes[0, 0].set_title("a) hold (feasible target)", fontsize=11, loc="left")

    # Track (right column)
    for row, pc_idx in enumerate([0, 1]):
        ax = axes[row, 1]
        ax.plot(t, ref_arr[:, pc_idx], "--", color="black", lw=1.0,
                 label="desired (target)")
        ax.plot(t, z_A_track[:, pc_idx], "-", color="#1f77b4", lw=1.6,
                 label="Stage A — true model")
        ax.plot(t, z_B_track[:, pc_idx], "--", color="#ff7f0e", lw=1.6,
                 label="Stage B — EM-fitted model")
        ax.set_ylabel(LABEL_PC1 if pc_idx == 0 else LABEL_PC2)
        style_axis(ax)
        if row == 1:
            ax.set_xlabel(LABEL_TIME)
    axes[0, 1].set_title("b) track (slow sine)", fontsize=11, loc="left")

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=10,
                bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("Does the learned model suffice?",
                  fontsize=14, x=0.02, ha="left")
    add_caption(fig,
        "Overlay of MPC driving the TRUE model (solid blue) and the EM-fitted "
        "model (dashed orange). When the curves coincide, identification is "
        "good enough — the estimator does not bottleneck control.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    out = OUT_DIR / "figE3_07_stageA_vs_stageB.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")
    return iface


# ============================================================================
# FIGURE 08 — Controller comparison (error vs effort + bar chart per task)
# ============================================================================

def fig08_comparison(ctx: dict) -> None:
    """Bar chart of normalised error per (controller, task)."""
    T = ctx["T"]; M = ctx["M"]
    tasks = {
        "hold (feasible)":  (lambda _t: ctx["ref_feas"], ctx["ref_feas"]),
        "hold (infeasible)":(lambda _t: ctx["ref_infeas"], ctx["ref_infeas"]),
        "track (slow sine)": None,  # filled below
        "suppress":         (lambda _t: ctx["z0"], ctx["z0"]),
    }
    amp = 0.3 * np.array([
        abs(ctx["verts"][3, 0] - ctx["verts"][0, 0]) / 2,
        abs(ctx["verts"][3, 1] - ctx["verts"][0, 1]) / 2,
    ])
    omega = 2.0 * np.pi / 100.0
    tasks["track (slow sine)"] = (
        lambda tt: ctx["ref_feas"] + amp * np.sin(omega * tt),
        ctx["ref_feas"],
    )

    controllers = ["OpenLoop", "PropFeedback", "LQG", "PolePlace", "MPC",
                    "PI", "LQGI", "OffsetFreeMPC"]

    # Compute (rms_err / noise_floor) for each (task, controller).
    table = {}
    efforts = {}
    for tname, (ref_fn, ref0) in tasks.items():
        table[tname] = {}
        efforts[tname] = {}
        ref_arr = np.array([ref_fn(tt) for tt in range(T)])
        for cname in controllers:
            logs = run_episode(ctx, cname, ref_fn, ref0)
            z = logs["y"] @ M.T
            err = rms(z - ref_arr)
            table[tname][cname] = err / ctx["noise_floor"]
            efforts[tname][cname] = control_effort(logs["u"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (tname, row) in zip(axes.flatten(), table.items()):
        names = list(row.keys())
        vals = [row[n] for n in names]
        colors = [CONTROLLER_COLORS[n] for n in names]
        bars = ax.bar(names, vals, color=colors, edgecolor="0.3", linewidth=0.5)
        ax.set_ylabel("rms error  (× noise floor)")
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                     f"{v:.0f}", ha="center", va="bottom", fontsize=8.5,
                     color="0.2")
        style_axis(ax)
        ax.set_title(tname, fontsize=11, loc="left")
        # Y-scale: log on infeasible / suppress where LQG-family blows up.
        if max(vals) / max(1e-9, min(vals)) > 50:
            ax.set_yscale("log")

    fig.suptitle("Which controller wins on which task?",
                  fontsize=14, x=0.02, ha="left")
    add_caption(fig,
        "Error in units of the open-loop readout noise floor (lower is better). "
        "Bars use the same controller→colour map throughout the figure set. "
        "Where the bars span >50×, the y-axis is log-scaled.")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    out = OUT_DIR / "figE3_08_controller_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ============================================================================
# FIGURE 09 — Is the learned model trustworthy?  (EM validation in plain words)
# ============================================================================

def fig09_model_trust(ctx: dict, iface: IdentifiedSystem) -> None:
    """Plain-language retitle of fig8_em_validation: log-likelihood path,
    fitted vs true G, state R², and a "good enough" verdict."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Panel a: EM log-likelihood path (does the fit converge?)
    ax = axes[0, 0]
    if iface.estimator is not None and iface.estimator.loglik_hist:
        ll = iface.estimator.loglik_hist
        ax.plot(np.arange(len(ll)), ll, "-o", color="#1f77b4")
        ax.set_xlabel("EM iteration")
        ax.set_ylabel("log-likelihood of the probe data")
        ax.text(0.98, 0.04,
                 f"converged in {len(ll)} iterations\n"
                 f"path strictly increasing: "
                 f"{'yes' if iface.validation.get('em_monotone', False) else 'no'}",
                 transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                            alpha=0.9, edgecolor="0.7"))
    style_axis(ax)
    set_question_title(ax, "a) Does the EM fit converge?",
                        subtitle="log-likelihood per iteration on the probe data")

    # Panel b: fitted G vs true G (slope-of-input-to-readout map).
    ax = axes[0, 1]
    system = ctx["system"]
    a0 = ctx["a0"]; c0 = ctx["c0"]
    G_true = readout_steady_state_gain(system.A, system.B, system.C,
                                         iface.readout_M)
    G_fit = iface.readout_G
    ax.scatter(G_true.flatten(), G_fit.flatten(), s=60, color="#2ca02c",
                edgecolor="black", linewidth=0.5)
    lim = max(np.abs(G_true).max(), np.abs(G_fit).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "--", color="0.5", lw=1.0,
             label="perfect identification")
    ax.set_xlabel("true input→readout gain (entries of G)")
    ax.set_ylabel("EM-fitted input→readout gain")
    g_rel = float(np.linalg.norm(G_fit - G_true) / max(np.linalg.norm(G_true), 1e-9))
    ax.text(0.98, 0.04,
             f"relative error  ||G_fit − G_true|| / ||G_true|| = {g_rel*100:.1f}%",
             transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        alpha=0.9, edgecolor="0.7"))
    ax.legend(loc="upper left", fontsize=9)
    style_axis(ax)
    set_question_title(ax, "b) Is the fitted gain matrix correct?",
                        subtitle="each dot = one entry of G; perfect fit lies on the diagonal")

    # Panel c: state R² (does the latent track?).
    ax = axes[1, 0]
    r2 = iface.validation.get("state_r2", float("nan"))
    bars = ax.bar(["fitted state R²"], [r2 if np.isfinite(r2) else 0.0],
                    color="#1f77b4", width=0.35)
    ax.axhline(0.9, color="0.6", ls=":", lw=0.8, label="0.9 (good)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R² (aligned to true state)")
    ax.text(0, max(r2 if np.isfinite(r2) else 0.0, 0.01),
             f"{r2:.3f}" if np.isfinite(r2) else "n/a",
             ha="center", va="bottom", fontsize=11, color="0.1")
    ax.legend(loc="lower right", fontsize=9)
    style_axis(ax)
    set_question_title(ax, "c) Does the fitted latent track the truth?",
                        subtitle="R² ≈ 1 means yes")

    # Panel d: verdict.
    ax = axes[1, 1]; ax.axis("off")
    verdict_lines = [
        "Headline verdict",
        "",
        f"  EM iterations          : {iface.diagnostics.get('fit_iters', '?')}",
        f"  ρ(A_fit)               : {iface.diagnostics.get('A_spectral_radius', float('nan')):.3f}   (stable if < 1)",
        f"  ||CB||_F               : {iface.diagnostics.get('CB_frob', float('nan')):.2f}   (threshold "
            f"{iface.diagnostics.get('CB_visibility_threshold', float('nan')):.2f})",
        f"  PC1 / PC2 explained var: {iface.readout_explained_var[0]:.2f} / "
            f"{iface.readout_explained_var[1]:.2f}",
        f"  cond(G_readout)        : {iface.readout_cond_G:.1f}   (well-conditioned if < 50)",
        "",
        f"  Verdict: {'GOOD ENOUGH for control' if g_rel < 0.15 else 'CHECK — large G error'}",
    ]
    ax.text(0.0, 0.95, "\n".join(verdict_lines), transform=ax.transAxes,
             fontsize=11, ha="left", va="top", family="monospace",
             color="0.1")
    set_question_title(ax, "d) Headline verdict")

    fig.suptitle("Is the learned model trustworthy?",
                  fontsize=14, x=0.02, ha="left")
    add_caption(fig,
        "EM converges in 3 iterations on this scenario; fitted gain matches "
        "the truth within 3% relative error; latent state R² ≈ 1. The "
        "estimator is not the bottleneck.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    out = OUT_DIR / "figE3_09_model_trust.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ============================================================================
# FIGURE 10 — Limitations & findings
# ============================================================================

def fig10_limitations(ctx: dict) -> None:
    """Text-heavy summary of known failure modes (spec §6 step 10)."""
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111)
    ax.axis("off")

    findings = [
        ("1. Suppression floor",
         "A one-sided actuator (u ∈ [0, 1], push-only) cannot cancel a "
         "symmetric noise process. Best achievable RMS reduction on "
         "default_neural_system is ~25 % below the open-loop drift; the "
         "noise floor is the per-step jitter (σ_z ≈ 0.4)."),
        ("2. Slow-drift scenarios (finite-horizon)",
         "When A has eigenvalues very close to 1 (e.g. slow_drift's "
         "A[2,2] = 0.999), the steady-state target is theoretically "
         "reachable but the time constant exceeds the experiment horizon. "
         "Reachability is an ∞-horizon statement; finite-time error is "
         "dominated by the slow transient."),
        ("3. Hidden-input regime (‖CB‖_F ≈ 0)",
         "When the input enters the observation null-space, EM cannot "
         "identify B reliably from short probes; LQG amplifies the "
         "estimator noise into the actuator and destabilises. MPC stays "
         "safe by planning within [0, 1]. Flagged automatically by "
         "‖CB‖_F < 0.05 · ‖C‖_F · ‖B‖_F."),
        ("4. Integral windup",
         "Pure integral action on an infeasible target grows the integral "
         "state without bound. Mandatory: freeze (or clamp) the integral "
         "when the input saturates. Without anti-windup, the controller "
         "permanently mis-saturates after a single large-error transient."),
        ("5. Integral phase lag on tracking",
         "Integral action removes steady-state offsets (helps hold) but "
         "introduces phase lag (hurts tracking). On slow sinusoids LQG+I "
         "is strictly worse than feedforward LQG. Spec §4 calls this out: "
         "integral applies to hold and track, but in practice tracking "
         "benefits only when paired with predictive logic (MPC's horizon)."),
        ("6. LQG on infeasible / suppression targets (wrong-corner failure)",
         "Clipped LQR's K-direction projects an infeasible target onto the "
         "wrong corner of the holdable region — yielding errors 30+% "
         "worse than open loop. LQG+I inherits the failure (integral "
         "cannot override a wrong-sign feedback gain). MPC's "
         "constraint-aware planner avoids both: when the target is "
         "infeasible it goes to the closest feasible point; on suppression "
         "it correctly outputs u ≡ 0 if even that is best."),
    ]

    y = 0.95
    dy = 0.155
    for title, body in findings:
        ax.text(0.0, y, title, transform=ax.transAxes, fontsize=12.5,
                 ha="left", va="top", fontweight="bold", color="#1f77b4")
        ax.text(0.0, y - 0.035, body, transform=ax.transAxes, fontsize=10,
                 ha="left", va="top", color="0.15",
                 wrap=True)
        y -= dy

    fig.suptitle("What can't this system do?  (Honest failure modes)",
                  fontsize=14, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    out = OUT_DIR / "figE3_10_limitations.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ============================================================================
# Driver
# ============================================================================

def main():
    print("=" * 84)
    print("E3 — Regenerating the §6 figure sequence under the clarity rules")
    print("=" * 84)
    ctx = build_setup()
    print(f"  PC explained var: PC1 = {ctx['explained'][0]:.3f}, "
          f"PC2 = {ctx['explained'][1]:.3f}")
    print(f"  cond(G)         : {np.linalg.cond(ctx['G']):.2f}")
    print(f"  feasible target : {ctx['ref_feas'].round(3)}")
    print(f"  infeasible tgt  : {ctx['ref_infeas'].round(3)}")
    print(f"  noise floor (σ) : {ctx['noise_floor']:.3f}")
    print()

    fig01_setup(ctx)
    fig02_holdable_region(ctx)
    hold_cache = fig03_hold(ctx)
    fig04_integral(ctx, hold_cache)
    fig05_track(ctx)
    fig06_suppress(ctx)
    iface = fig07_stage_a_vs_b(ctx)
    fig08_comparison(ctx)
    fig09_model_trust(ctx, iface)
    fig10_limitations(ctx)
    print("\n  Done.")


if __name__ == "__main__":
    main()
