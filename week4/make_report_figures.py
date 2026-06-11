#!/usr/bin/env python
"""make_report_figures.py -- Week-4 Step-2D report figures (real-cascade controller).

Every trajectory in these figures comes from running the genuine brain->muscle->arm
cascade (``CascadePlant`` wrapping ``BMI_and_Hand(Brain(seed))``) through the deployed
cascade controller (``reach_controller.ReachController``), on the SAME path the
evaluation uses: a ``WARMUP``-step DC power pre-latch, then ``ReachController.step``
each step under live ``hand_pos`` feedback. No idealised arm, no hand-drawn or
synthesised trajectories. Any figure whose data cannot be produced prints a
``SKIP: <figure> -- missing <what>`` line instead of being faked.

The per-reach logger ``reach_controller._run_one`` already records hand (x,y), the
input (u0,u1), the active joint, and the target; it does NOT return joint angles, so
``run_reach_logged`` below additionally captures the controller's IK-tracked joint
angles (``ReachController.draw_theta``) and the target joint angles
(``Reacher._theta_star``) -- a logging addition, not synthesis.

Run headless on the GG4 venv (the only interpreter with GG4 + torch):

    C:/Github/GG4/venv/Scripts/python.exe make_report_figures.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")            # headless: we only save figures
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import kinematics as kin
from reacher import Reacher
from reach_controller import ReachController, load_library, muscle_index, SPEED_MODEL
import arm_controller                      # eval constants (WARMUP, DC)
from cascade_plant import CascadePlant

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "fig"
WARMUP = int(arm_controller.WARMUP)        # ~100-step one-time DC power latch
REST = np.asarray(ReachController.REST, float)   # [0.5, 0.5] -- DC on, no AC (arm freezes)
R_INNER, R_OUTER = 5.0, 57.0               # safe annulus (demo.py)
TOL = 2.0

OUTCOME_COLORS = {"land": "#2ca02c", "near": "#e8a200",
                  "limit": "#d62728", "stall": "#7f7f7f"}

PRODUCED: list[str] = []
SKIPPED: list[str] = []


# ---------------------------------------------------------------------------
# IO / style helpers (no shared week4 style module exists; match demo.py)
# ---------------------------------------------------------------------------
def save(fig, name: str) -> None:
    FIGDIR.mkdir(exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.pdf")
    fig.savefig(FIGDIR / f"{name}.png", dpi=130)
    plt.close(fig)
    print(f"  [saved] fig/{name}.pdf  +  fig/{name}.png")


def draw_workspace(ax) -> None:
    """Reachable disk (r=60) + safe annulus (r=5,57), faint, as in demo.py's ArmView."""
    ax.add_patch(plt.Circle((0, 0), kin.REACH, fill=False, ls="--", ec="0.6", lw=1.0))
    ax.add_patch(plt.Circle((0, 0), R_OUTER, fill=False, ls=":", ec="0.8", lw=1.0))
    ax.add_patch(plt.Circle((0, 0), R_INNER, fill=False, ls=":", ec="0.8", lw=1.0))
    ax.set_aspect("equal")
    ax.set_xlim(-65, 65)
    ax.set_ylim(-65, 65)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def classify(final_err: float, switches: int, tol: float = TOL) -> str:
    """Eval outcome taxonomy (evaluate_controller.reach_metrics order)."""
    if final_err <= tol:
        return "land"
    if final_err <= 2 * tol:
        return "near_miss"
    if switches > 20:
        return "limit_cycle"
    return "stall"


def joint_runs(active):
    """Run-length encode the active-joint array -> [(value, start, end, length), ...]."""
    a = np.asarray(active)
    out, i = [], 0
    while i < len(a):
        j = i
        while j < len(a) and a[j] == a[i]:
            j += 1
        out.append((int(a[i]), i, j - 1, j - i))
        i = j
    return out


def max_run(active, val) -> int:
    return max((n for v, s, e, n in joint_runs(active) if v == val), default=0)


# ---------------------------------------------------------------------------
# The logged run -- mirrors evaluate_controller.run_reach + full per-step logging
# ---------------------------------------------------------------------------
def run_reach_logged(target, seed, tol: float = TOL, max_steps: int = 1500, **ctrl_kw):
    """One deployed reach on the real cascade, fully logged (the eval path)."""
    target = np.asarray(target, float)
    plant = CascadePlant(seed=seed)
    for _ in range(WARMUP):                 # one-time power latch (DC on, no AC)
        plant.command(REST)

    ctrl = ReachController(tol=tol, **ctrl_kw)
    ctrl.reset()

    hands, cmds, thetas, active, muscles = [], [], [], [], []
    touch = None
    for k in range(max_steps):
        hand = np.asarray(plant.hand_pos, float)
        u = np.asarray(ctrl.step(hand, target), float)
        plant.command(u)

        hands.append(hand.copy())
        cmds.append(u.copy())
        thetas.append(ctrl.draw_theta.copy())          # IK-tracked joint angle estimate
        mv = ctrl.move
        active.append(mv[0] if mv is not None else -1)
        muscles.append(muscle_index(*mv) if mv is not None else -1)

        d = float(np.linalg.norm(target - hand))
        if touch is None and d < tol:
            touch = k
        if touch is not None and k > touch + 8:         # settled near target -> stop
            recent = np.asarray(hands[-8:])
            if np.max(np.linalg.norm(np.diff(recent, axis=0), axis=1)) < 1e-3:
                break

    hands = np.asarray(hands)
    cmds = np.asarray(cmds)
    thetas = np.asarray(thetas)
    active = np.asarray(active)
    muscles = np.asarray(muscles)
    theta_star = (np.asarray(ctrl.reacher._theta_star, float).copy()
                  if ctrl.reacher._theta_star is not None else None)

    d = np.linalg.norm(target - hands, axis=1)
    tc = int(np.argmin(d))
    min_dist = float(d[tc])
    final_err = float(d[-1])
    overshoot = float(np.max(d[tc:]) - d[tc])
    switches = int(np.sum(muscles[1:] != muscles[:-1]))
    outcome = classify(final_err, switches, tol)

    return dict(hands=hands, cmds=cmds, thetas=thetas, active=active, muscles=muscles,
                theta_star=theta_star, target=target, seed=int(seed), tol=tol,
                min_dist=min_dist, final_err=final_err, touch=touch, overshoot=overshoot,
                switches=switches, outcome=outcome, n_retrig=int(ctrl.n_retrig),
                steps=len(hands), staircase=(max_run(active, 0) >= 8 and max_run(active, 1) >= 8))


# ---------------------------------------------------------------------------
# A single fast-shoulder move from home, on the real cascade, with two stop rules.
# Driving only sh+ keeps the hand on the reach rim, so the shoulder angle is tracked
# cleanly by continuity IK.  NAIVE: full A*, cut AT the target angle (coasts past).
# GRADED: the deployed stop -- matched-coast graded amplitude + live-coast lead.
# ---------------------------------------------------------------------------
def run_single_shoulder(seed, theta_target, mode, muscle: int = 0, max_steps: int = 600,
                        speed_gain: float = 0.75, lead_margin: float = 0.05,
                        vbar_decay: float = 0.6):
    plant = CascadePlant(seed=seed)
    for _ in range(WARMUP):
        plant.command(REST)
    rec = load_library()[muscle]
    f, A_star, coast_steps = rec["f"], rec["A_star"], rec["coast_steps"]
    slope, a0, a_min = SPEED_MODEL[muscle]

    tracker = Reacher(tol=TOL)
    tracker.reset()
    th_s, cmds, radii = [], [], []
    t = 0
    cut = None
    vbar = 0.0
    prev = 0.0
    for k in range(max_steps):
        hand = np.asarray(plant.hand_pos, float)
        th = float(tracker._track(hand)[0])            # tracked shoulder angle
        th_s.append(th)
        radii.append(float(np.hypot(*hand)))
        vbar = vbar_decay * vbar + (1.0 - vbar_decay) * abs(th - prev)
        prev = th
        dgo = float(theta_target) - th                 # remaining (positive move)

        if cut is not None:
            u = REST.copy()
        elif mode == "naive":
            if dgo <= 0.0:                              # NAIVE: cut at the target angle
                cut = k
                u = REST.copy()
            else:
                t += 1
                u = np.array([0.5 + A_star * np.cos(2.0 * np.pi * f * t), 0.5])
        else:                                          # graded (the deployed stop)
            live_coast = vbar * coast_steps
            if dgo <= live_coast + lead_margin:        # live-coast lead: cut early
                cut = k
                u = REST.copy()
            else:
                desired_v = speed_gain * dgo / coast_steps
                A = float(np.clip(a0 + desired_v / slope, a_min, A_star))
                t += 1
                u = np.array([0.5 + A * np.cos(2.0 * np.pi * f * t), 0.5])

        cmds.append(np.asarray(u, float).copy())
        plant.command(u)
        if cut is not None and k > cut + 12:           # coasted to rest -> stop
            recent = np.asarray(th_s[-8:])
            if float(np.max(np.abs(np.diff(recent)))) < 1e-4:
                break

    th_s = np.unwrap(np.asarray(th_s))                 # continuous (monotone rotation)
    return dict(theta_s=th_s, cmds=np.asarray(cmds), cut=cut, mode=mode,
                theta_target=float(theta_target), rest=float(th_s[-1]),
                overshoot=float(np.max(th_s) - float(theta_target)),
                r_min=float(np.min(radii)), r_max=float(np.max(radii)), seed=int(seed))


# ---------------------------------------------------------------------------
# Figure 1 -- reach trajectory, success vs failure
# ---------------------------------------------------------------------------
# success: prefer a clean LAND that genuinely staircases (shoulder leg + elbow leg)
SUCCESS_CANDIDATES = [(0, (40.0, 25.0)), (6, (20.0, 20.0)),
                      (1, (40.0, 25.0)), (0, (38.97, 22.5)), (0, (0.0, 30.0))]
# failure: a clear, large miss (the (30,0) reach loops dramatically and drifts off)
FAILURE_CANDIDATES = [(0, (30.0, 0.0)), (0, (15.0, 25.98)),
                      (2, (30.0, 0.0)), (6, (0.0, 45.0))]


def _search(candidates, want, label):
    tried = []
    for seed, tgt in candidates:
        print(f"    trying seed {seed} target {tgt} ...", flush=True)
        r = run_reach_logged(tgt, seed)
        print(f"      -> outcome={r['outcome']} final_err={r['final_err']:.2f} "
              f"min_dist={r['min_dist']:.2f} switches={r['switches']} "
              f"staircase={r['staircase']} steps={r['steps']}")
        tried.append(r)
        if want(r):
            return r, tried
    return None, tried


def build_fig1():
    print("\n[Figure 1] reach trajectory, success vs failure")
    print("  searching for a clean LAND that staircases:")
    success, land_tried = _search(SUCCESS_CANDIDATES,
                                  lambda r: r["outcome"] == "land" and r["staircase"], "land")
    if success is None:                       # accept any clean land if none staircase
        lands = [r for r in land_tried if r["outcome"] == "land"]
        success = lands[0] if lands else None

    print("  searching for a clear FAILURE (large miss / limit cycle):")
    failure, fail_tried = _search(
        FAILURE_CANDIDATES,
        lambda r: r["outcome"] != "land" and r["final_err"] > 6.0, "failure")
    if failure is None:
        clear = [r for r in fail_tried if r["outcome"] != "land" and r["final_err"] > 2 * TOL]
        failure = max(clear, key=lambda r: r["final_err"]) if clear else None

    if success is None or failure is None:
        miss = [m for m, ok in (("a clean LAND", success), ("a clear FAILURE", failure))
                if ok is None]
        SKIPPED.append(f"Figure 1 (reach_success_fail) -- missing {', '.join(miss)} "
                       f"in the searched seeds/targets")
        print(f"  SKIP: Figure 1 -- missing {', '.join(miss)}")
        return None

    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2))

    # panel (a): the two hand paths
    draw_workspace(ax[0])
    for res, color, label in ((success, "tab:blue", "success (land)"),
                              (failure, "tab:red", f"failure ({failure['outcome']})")):
        h = res["hands"]
        ax[0].plot(h[:, 0], h[:, 1], color=color, lw=1.8, alpha=0.9, label=label, zorder=4)
        ax[0].plot(h[0, 0], h[0, 1], "o", color=color, ms=7, zorder=6)            # start
        ax[0].plot(res["target"][0], res["target"][1], "x", color=color,
                   ms=13, mew=2.6, zorder=7)                                      # target
    ax[0].plot([], [], "o", color="0.3", label="start (home)")
    ax[0].plot([], [], "x", color="0.3", mew=2.6, label="target")
    ax[0].set_title("(a) Hand path: success vs failure  (real cascade)")
    ax[0].legend(loc="lower left", fontsize=8)

    # panel (b): successful reach joint angles -> the one-joint staircase
    th = success["thetas"]
    steps = np.arange(len(th))
    ax[1].plot(steps, th[:, 0], color="tab:blue", lw=1.6, label=r"$\theta_s$ (shoulder)")
    ax[1].plot(steps, th[:, 1], color="tab:green", lw=1.6, label=r"$\theta_e$ (elbow)")
    if success["theta_star"] is not None:
        ts = success["theta_star"]
        ax[1].axhline(ts[0], color="tab:blue", ls="--", lw=1.0, alpha=0.7,
                      label=r"$\theta_s^\star$")
        ax[1].axhline(ts[1], color="tab:green", ls="--", lw=1.0, alpha=0.7,
                      label=r"$\theta_e^\star$")
    ax[1].set_xlabel("control step")
    ax[1].set_ylabel("joint angle (rad)")
    ax[1].set_title("(b) Successful reach: joint angles\n(one joint at a time = staircase)")
    ax[1].legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, "reach_success_fail")
    PRODUCED.append("Figure 1 (reach_success_fail)")

    print(f"  USED  success: seed {success['seed']} target "
          f"({success['target'][0]:.1f},{success['target'][1]:.1f})  "
          f"final_err={success['final_err']:.2f}  staircase={success['staircase']}")
    print(f"  USED  failure: seed {failure['seed']} target "
          f"({failure['target'][0]:.1f},{failure['target'][1]:.1f})  "
          f"outcome={failure['outcome']}  final_err={failure['final_err']:.2f}")
    return success


# ---------------------------------------------------------------------------
# Figure 2 -- the stop mechanism (coast): naive vs graded single-shoulder move
# ---------------------------------------------------------------------------
SHOULDER_SEED = 0
SHOULDER_TARGET_ANGLE = 0.9      # rad; sh+ from home, leaves room for the naive coast


def build_fig2():
    print("\n[Figure 2] the stop mechanism (coast) -- single fast-shoulder move")
    print(f"  graded (deployed stop): seed {SHOULDER_SEED} sh+ to "
          f"{SHOULDER_TARGET_ANGLE} rad", flush=True)
    graded = run_single_shoulder(SHOULDER_SEED, SHOULDER_TARGET_ANGLE, "graded")
    print(f"      cut@{graded['cut']} rest={graded['rest']:.3f} "
          f"overshoot={graded['overshoot']:+.3f} rad  r in "
          f"[{graded['r_min']:.1f},{graded['r_max']:.1f}]")
    print(f"  naive (full A*, cut at target): seed {SHOULDER_SEED}", flush=True)
    naive = run_single_shoulder(SHOULDER_SEED, SHOULDER_TARGET_ANGLE, "naive")
    print(f"      cut@{naive['cut']} rest={naive['rest']:.3f} "
          f"overshoot={naive['overshoot']:+.3f} rad  r in "
          f"[{naive['r_min']:.1f},{naive['r_max']:.1f}]")

    if graded["cut"] is None:
        SKIPPED.append("Figure 2 (stop_coast) -- missing a clean single shoulder-move log "
                       "(deployed graded move never cut)")
        print("  SKIP: Figure 2 -- the graded shoulder move never cut")
        return

    tgt = SHOULDER_TARGET_ANGLE
    overlay = (naive["cut"] is not None
               and naive["overshoot"] > 0.15
               and naive["overshoot"] > graded["overshoot"] + 0.10)
    print(f"  -> {'overlay naive vs graded' if overlay else 'graded single-move fallback'}")

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.axhline(tgt, color="0.4", ls="--", lw=1.1, label=r"target $\theta_s^\star$")

    # graded (deployed) curve
    g = graded["theta_s"]
    ax.plot(np.arange(len(g)), g, color="tab:blue", lw=2.0,
            label="graded approach + live-coast lead (deployed)")
    gc = graded["cut"]
    ax.axvline(gc, color="tab:blue", ls=":", lw=1.2)
    ax.annotate("AC cut", (gc, g[gc]), textcoords="offset points", xytext=(6, -16),
                fontsize=9, color="tab:blue")
    ax.axvspan(gc, len(g) - 1, color="tab:blue", alpha=0.07)
    ax.text((gc + len(g) - 1) / 2, ax.get_ylim()[0], "coast", ha="center", va="bottom",
            fontsize=8, color="tab:blue")
    ax.plot(len(g) - 1, g[-1], "o", color="tab:blue", ms=7)
    ax.annotate(f"rest {g[-1]:.2f} (target {tgt:.2f})", (len(g) - 1, g[-1]),
                textcoords="offset points", xytext=(-6, 10), fontsize=8,
                color="tab:blue", ha="right")

    if overlay:
        n = naive["theta_s"]
        ax.plot(np.arange(len(n)), n, color="tab:red", lw=2.0,
                label=r"naive stop: full $A^*$, cut at target (overshoots)")
        nc = naive["cut"]
        ax.axvline(nc, color="tab:red", ls=":", lw=1.2)
        ax.annotate("naive cut", (nc, n[nc]), textcoords="offset points", xytext=(6, 8),
                    fontsize=9, color="tab:red")
        ax.plot(int(np.argmax(n)), float(np.max(n)), "v", color="tab:red", ms=8)
        ax.annotate(f"overshoot +{naive['overshoot']:.2f} rad",
                    (int(np.argmax(n)), float(np.max(n))), textcoords="offset points",
                    xytext=(6, -2), fontsize=8, color="tab:red")

    ax.set_xlabel("control step")
    ax.set_ylabel(r"shoulder angle $\theta_s$ (rad)")
    ttl = ("Stop mechanism: naive overshoot vs graded landing" if overlay
           else "Stop mechanism: deployed graded approach + live-coast lead")
    ax.set_title(ttl + "  (single sh+ move, real cascade)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, "stop_coast")
    PRODUCED.append("Figure 2 (stop_coast)" + ("  [naive vs graded]" if overlay
                                               else "  [deployed single move, annotated]"))
    if not overlay:
        print("  note: produced the deployed graded single-move fallback "
              "(no clean naive overshoot to overlay).")


# ---------------------------------------------------------------------------
# Figure 3 -- outcome breakdown across controllers (fixed numbers, no runs)
# ---------------------------------------------------------------------------
def build_fig3():
    print("\n[Figure 3] outcome breakdown across controllers (fixed numbers)")
    # (land, near, limit-cycle, stall); ordered best land-rate at the top.
    rows = [
        ("Cascade Kalman+PD (Jash)", (86, 7, 6, 1)),
        ("Joint-space (mine)",       (76, 12, 12, 0)),
        ("Resolved-rate (Henrik)",   (71, 10, 19, 0)),
    ]
    seg_keys = ["land", "near", "limit", "stall"]
    seg_lbl = {"land": "land", "near": "near", "limit": "limit-cycle", "stall": "stall"}
    labels = [r[0] for r in rows]
    y = np.arange(len(rows))[::-1]           # first row at the top

    fig, ax = plt.subplots(figsize=(10, 3.6))
    for yi, (_, vals) in zip(y, rows):
        left = 0.0
        for key, v in zip(seg_keys, vals):
            if v <= 0:
                continue
            ax.barh(yi, v, left=left, color=OUTCOME_COLORS[key], edgecolor="white")
            if v >= 4:
                ax.text(left + v / 2, yi, f"{v}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
            left += v
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of reaches (%)")
    ax.set_title("Outcome breakdown across controllers")
    handles = [plt.Rectangle((0, 0), 1, 1, color=OUTCOME_COLORS[k]) for k in seg_keys]
    ax.legend(handles, [seg_lbl[k] for k in seg_keys], ncol=4, fontsize=8,
              loc="lower center", bbox_to_anchor=(0.5, -0.42), frameon=False)
    fig.tight_layout()
    save(fig, "outcome_breakdown")
    PRODUCED.append("Figure 3 (outcome_breakdown)")


# ---------------------------------------------------------------------------
# Figure 4 -- per-seed landing spread (ranges supplied -> range bars)
# ---------------------------------------------------------------------------
def build_fig4():
    print("\n[Figure 4] per-seed landing spread (supplied ranges)")
    rows = [
        ("Cascade Kalman+PD (Jash)", (71, 100), "tab:green"),
        ("Joint-space (mine)",       (44, 90),  "tab:blue"),
        ("Resolved-rate (Henrik)",   (56, 80),  "tab:orange"),
    ]
    labels = [r[0] for r in rows]
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(9, 3.6))
    for yi, (_, (lo, hi), col) in zip(y, rows):
        ax.hlines(yi, lo, hi, color=col, lw=8, alpha=0.55)
        ax.plot([lo, hi], [yi, yi], "|", color=col, ms=16, mew=2.5)
        ax.text(lo - 1.5, yi, f"{lo}", ha="right", va="center", fontsize=9, color=col)
        ax.text(hi + 1.5, yi, f"{hi}", ha="left", va="center", fontsize=9, color=col)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(35, 105)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("per-seed landing rate (%)  -- min to max")
    ax.set_title("Per-seed landing spread (range across seeds)")
    fig.tight_layout()
    save(fig, "seed_spread")
    PRODUCED.append("Figure 4 (seed_spread)")


# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Week-4 Step-2D report figures -- real cascade controller")
    print(f"output dir: {FIGDIR}")
    print("=" * 70)

    for name, fn in (("Figure 1", build_fig1), ("Figure 2", build_fig2),
                     ("Figure 3", build_fig3), ("Figure 4", build_fig4)):
        try:
            fn()
        except Exception as ex:               # one figure failing must not kill the rest
            tag = {"Figure 1": "reach_success_fail", "Figure 2": "stop_coast",
                   "Figure 3": "outcome_breakdown", "Figure 4": "seed_spread"}[name]
            SKIPPED.append(f"{name} ({tag}) -- error: {ex!r}")
            print(f"  SKIP: {name} -- error: {ex!r}")

    print("\n" + "=" * 70)
    print("PRODUCED:")
    for p in PRODUCED:
        print(f"   + {p}")
    print("SKIPPED:")
    for s in (SKIPPED or ["(none)"]):
        print(f"   - {s}")
    print("=" * 70)


if __name__ == "__main__":
    main()
