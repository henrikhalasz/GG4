"""harness.py -- target-sequence runner, metrics, by-eye trajectory plot (Step 2A).

Runs a controller (the ``ReachingPolicy``) against a plant (the ``IdealActuator``
now; the ``CascadePlant`` later, unchanged) over a sequence of targets, mirroring
the Week-4 ``run_closed_loop`` convention (measure -> compute from PAST
observations -> apply), and logs the hand trajectory and commands. It then
computes the metrics that matter -- **touched within tolerance?** and
**steps-to-touch** -- plus overshoot and path length, and draws the by-eye
trajectory plot over the reachable disk.

The harness is plant-agnostic: it only uses ``plant.hand_pos`` / ``plant.command``
/ ``plant.reset`` and feeds the live hand position back as the observation
(the regime-A assumption, see cascade_plant.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kinematics as kin

HERE = Path(__file__).resolve().parent


def run_single(plant, policy, target, *, tol: float = 2.0, max_steps: int = 600,
               settle_win: int = 5, settle_eps: float = 1e-3) -> dict:
    """Drive ``plant`` from its home pose to one ``target`` via ``policy``.

    Mirrors the Week-4 ``run_closed_loop`` convention (measure -> compute from
    PAST observations -> apply). Returns the hand trajectory, commands, and
    per-target metrics (touch, steps-to-touch, closest approach, overshoot).
    """
    plant.reset()
    policy.set_targets(np.asarray(target, dtype=float).reshape(1, 2))

    observations: list = []
    hands: list = []
    commands: list = []
    touch_step = None
    for _ in range(max_steps):
        hand = np.asarray(plant.hand_pos, dtype=float)
        u = policy(np.array(observations), None)
        plant.command(u)
        observations.append(hand.copy())
        hands.append(hand.copy())
        commands.append(np.asarray(u, dtype=float).copy())
        if touch_step is None and np.linalg.norm(target - hand) < tol:
            touch_step = len(hands) - 1
        if policy.done and len(hands) > settle_win:
            recent = np.asarray(hands[-settle_win:])
            if np.max(np.linalg.norm(np.diff(recent, axis=0), axis=1)) < settle_eps:
                break

    hands = np.asarray(hands)
    d = np.linalg.norm(target - hands, axis=1)
    t_closest = int(np.argmin(d))
    min_dist = float(d[t_closest])
    # overshoot = drift back out after the closest approach (isolated reach).
    overshoot = float(np.max(d[t_closest:]) - min_dist)
    return {"hands": hands, "commands": np.asarray(commands), "target": np.asarray(target),
            "touched": bool(min_dist < tol), "min_dist": min_dist,
            "t_closest": t_closest, "steps_to_touch": touch_step,
            "overshoot": overshoot,
            "path_length": float(np.sum(np.linalg.norm(np.diff(hands, axis=0), axis=1)))}


def run_target_set(plant, policy, targets, *, tol: float = 2.0,
                   max_steps: int = 600) -> dict:
    """Isolated reaches from home to each target; aggregate metrics + trajectories."""
    targets = np.asarray(targets, dtype=float).reshape(-1, 2)
    runs = [run_single(plant, policy, tgt, tol=tol, max_steps=max_steps)
            for tgt in targets]
    touched = [r["touched"] for r in runs]
    metrics = {
        "touched": touched,
        "min_dist": [r["min_dist"] for r in runs],
        "steps_to_touch": [r["steps_to_touch"] for r in runs],
        "overshoot": [r["overshoot"] for r in runs],
        "path_length": [r["path_length"] for r in runs],
        "n_touched": int(sum(touched)),
    }
    return {"runs": runs, "targets": targets, "tol": tol, "metrics": metrics}


def run_cascade_sweep(seeds, targets, *, tol: float = 2.0, max_steps: int = 950) -> dict:
    """Isolated reaches from home over ``targets`` on the real cascade, per Brain seed.

    Each reach rebuilds a fresh (cold-power) Brain, so it pays the ~94-step cold
    onset once. Aggregates touch-success / steps-to-touch / overshoot across seeds.
    """
    from control_policy import ReachingPolicy
    from cascade_plant import CascadePlant

    targets = np.asarray(targets, dtype=float).reshape(-1, 2)
    per_seed = []
    print(f"cascade sweep: {len(seeds)} seeds x {len(targets)} targets (tol={tol})")
    for s in seeds:
        plant = CascadePlant(seed=s)
        policy = ReachingPolicy(mode="one_joint", tol=tol, cascade=True)
        rec = run_target_set(plant, policy, targets, tol=tol, max_steps=max_steps)
        rec["seed"] = s
        per_seed.append(rec)
        m = rec["metrics"]
        steps = [x for x in m["steps_to_touch"] if x is not None]
        print(f"  seed {s}: touched {m['n_touched']}/{len(targets)}  "
              f"steps-to-touch mean {np.mean(steps) if steps else float('nan'):.0f}  "
              f"max overshoot {max(m['overshoot']):.1f}")
    return {"per_seed": per_seed, "targets": targets, "tol": tol}


def summarise_cascade(sweep) -> None:
    per_seed = sweep["per_seed"]
    targets = sweep["targets"]
    tol = sweep["tol"]
    n = len(targets)
    touched = np.array([[r["metrics"]["touched"][i] for i in range(n)] for r in per_seed])
    steps = np.array([[r["metrics"]["steps_to_touch"][i] or np.nan for i in range(n)]
                      for r in per_seed], dtype=float)
    over = np.array([[r["metrics"]["overshoot"][i] for i in range(n)] for r in per_seed])
    print(f"\n=== cascade reach metrics ({len(per_seed)} seeds x {n} targets) ===")
    print(f"  touch-success: {touched.sum()}/{touched.size} = {touched.mean():.0%}")
    svals = steps[np.isfinite(steps)]
    print(f"  steps-to-touch: mean {np.nanmean(steps):.0f}  [{np.nanmin(svals):.0f},"
          f"{np.nanmax(svals):.0f}]  (includes the ~94-step cold onset)")
    print(f"  overshoot: mean {over.mean():.1f}  max {over.max():.1f}  (tol={tol})")
    print(f"  per-target touch rate:")
    for i, tgt in enumerate(targets):
        print(f"    {i:2d} ({tgt[0]:+5.0f},{tgt[1]:+5.0f}) r={np.hypot(*tgt):4.0f}: "
              f"{touched[:, i].sum()}/{len(per_seed)} touched, "
              f"steps {np.nanmean(steps[:, i]):.0f}, overshoot {over[:, i].mean():.1f}")


def plot_trajectories(record, *, title: str = "Reaching trajectories",
                      path: str | Path = HERE / "reach_trajectories.pdf") -> Path:
    """By-eye plot: every isolated reach from home over the disk, targets marked."""
    targets = record["targets"]
    tol = record["tol"]
    m = record["metrics"]

    fig, ax = plt.subplots(figsize=(7, 7))
    # reachable disk + safe annulus bounds
    ax.add_patch(plt.Circle((0, 0), kin.REACH, fill=False, ls="--",
                            ec="0.6", lw=1.2, label="reach r=60"))
    ax.add_patch(plt.Circle((0, 0), 57, fill=False, ls=":", ec="0.75", lw=1,
                            label="safe annulus [5,57]"))
    ax.add_patch(plt.Circle((0, 0), 5, fill=False, ls=":", ec="0.75", lw=1))

    for i, run in enumerate(record["runs"]):
        hands = run["hands"]
        ax.plot(hands[:, 0], hands[:, 1], "-", lw=1.1, alpha=0.85,
                label="hand paths" if i == 0 else None)
    ax.plot(60, 0, "o", color="k", ms=8, zorder=5, label="home (60,0)")

    for i, tgt in enumerate(targets):
        ok = m["touched"][i]
        ax.add_patch(plt.Circle(tgt, tol, fill=False, ec="0.5", lw=0.8))
        ax.plot(*tgt, "x", color="tab:green" if ok else "tab:red", ms=10, mew=2,
                zorder=6)
        ax.annotate(str(i), tgt, textcoords="offset points", xytext=(6, 6),
                    fontsize=9)

    ax.set_aspect("equal")
    ax.set_xlim(-65, 65)
    ax.set_ylim(-65, 65)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title}  ({m['n_touched']}/{len(targets)} touched, tol={tol})")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=110)
    plt.close(fig)
    return path


def _main_cascade(seeds):
    from reacher import _target_set
    targets = _target_set()
    sweep = run_cascade_sweep(seeds, targets, tol=2.0)
    summarise_cascade(sweep)
    # trajectory figure from the first seed
    rec0 = sweep["per_seed"][0]
    out = plot_trajectories(rec0, title=f"Cascade reaches (seed {rec0['seed']}, one-joint)",
                            path=HERE / "reach_trajectories_cascade.pdf")
    print(f"saved figure -> {out}")


def _main():
    import sys
    if "--cascade" in sys.argv:
        n = 4
        if "--seeds" in sys.argv:
            n = int(sys.argv[sys.argv.index("--seeds") + 1])
        _main_cascade(list(range(n)))
        return

    from ideal_actuator import IdealActuator
    from control_policy import ReachingPolicy
    from reacher import _target_set

    print("=== harness.py self-check ===")
    targets = _target_set()
    plant = IdealActuator()
    policy = ReachingPolicy(mode="one_joint", tol=2.0)

    record = run_target_set(plant, policy, targets, tol=2.0)
    m = record["metrics"]
    print(f"{'tgt':>3s} {'r':>6s} {'touch':>6s} {'min_d':>7s} "
          f"{'steps':>6s} {'oshoot':>7s} {'path':>7s}")
    for i, tgt in enumerate(targets):
        print(f"{i:3d} {np.hypot(*tgt):6.2f} {str(m['touched'][i]):>6s} "
              f"{m['min_dist'][i]:7.3f} {str(m['steps_to_touch'][i]):>6s} "
              f"{m['overshoot'][i]:7.3f} {m['path_length'][i]:7.1f}")
    steps = [s for s in m["steps_to_touch"] if s is not None]
    print(f"\ntouched {m['n_touched']}/{len(targets)} within tol={record['tol']}")
    print(f"steps-to-touch: mean {np.mean(steps):.1f}, min {min(steps)}, max {max(steps)}")
    print(f"max overshoot = {max(m['overshoot']):.3f} (tol={record['tol']})")

    out = plot_trajectories(record,
                            title="Reaching trajectories (IdealActuator, one-joint)")
    print(f"saved figure -> {out}")
    assert m["n_touched"] == len(targets), "harness: every target must be touched"
    assert max(m["overshoot"]) < record["tol"], "overshoot must stay within tol"
    print("All assertions passed.")


if __name__ == "__main__":
    _main()
