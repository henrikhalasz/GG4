"""
evaluate_controller.py -- rigorous, reusable evaluation suite for the reach
controller, brain-in-the-loop and multi-seed.

WHY THIS EXISTS
  The offline mock mispredicts the real brain (targets it calls limit cycles
  land fine on the cascade, and vice-versa), and each brain SEED is a different
  linear system (the seed-0 identified constants do not transfer cleanly). So a
  single-seed pass/fail tells us almost nothing. This suite drives the controller
  through the REAL cascade across many seeds and reports, per target, the
  LANDING RATE and an outcome taxonomy -- the honest picture, and a regression
  baseline to measure each future improvement against.

REPRODUCIBILITY
  Every reach uses a fresh GG4.Brain(random_seed=seed); the brain's noise is
  seeded internally, so results are deterministic given (seed, target, controller).

USAGE
  # real-brain multi-seed baseline (the headline run):
  python evaluate_controller.py --real --seeds 0-7
  # quick mock smoke / plumbing check:
  python evaluate_controller.py --seeds 0-2 --quiet
  # A/B a controller change without editing code (kwargs go to the controller):
  python evaluate_controller.py --real --seeds 0-7 --tag dwell --ctrl-kw minseg=50 margin=6
  # then compare two tagged runs:
  python evaluate_controller.py --compare v1 dwell

OUTPUTS (under docs/, suffixed by --tag)
  eval_<tag>_reaches.csv   one row per (seed,target) reach -- the raw record
  eval_<tag>_bytarget.csv  one row per target -- landing rate & aggregates over seeds
  eval_<tag>_map.pdf       workspace maps: landing rate, and path quality
  eval_<tag>_summary.txt   human-readable headline numbers
"""
import argparse
import csv
import json
import os
import numpy as np

# importing test_run (without --live) pins matplotlib to headless Agg
import test_run as T
from arm_controller import ReachController, ENGAGE, DC, WARMUP

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
PROBES = [(40.0, 25.0), (20.0, 20.0), (10.0, 35.0), (-20.0, 5.0), (-10.0, -38.0)]


# ---------------------------------------------------------------------------
# Controller construction goes through one function so a future v2 controller
# (or a parameter A/B) is a one-line swap. ctrl_kwargs come straight from --ctrl-kw.
# ---------------------------------------------------------------------------
def build_controller(target, ctrl_kwargs):
    return ReachController(np.asarray(target, float), **ctrl_kwargs)


# ---------------------------------------------------------------------------
# One reach: warm the plant, drive the controller to the target, reduce to metrics.
# ---------------------------------------------------------------------------
def run_reach(use_mock, seed, target, max_steps, ctrl_kwargs, el_weak=0.6):
    plant = T.make_plant(use_mock, seed, el_weak)
    for _ in range(WARMUP):                              # one-time power latch
        plant.next_state(np.array([DC, 0.5]))

    ctrl = build_controller(target, ctrl_kwargs)
    traj = [np.asarray(plant.hand_pos, float).copy()]
    for _ in range(max_steps):
        u0 = ctrl.command(plant.hand_pos)
        plant.next_state(np.array([u0, 0.5]))
        traj.append(np.asarray(plant.hand_pos, float).copy())
        if ctrl.done:
            break
    return reach_metrics(ctrl, np.array(traj), np.asarray(target, float), seed, max_steps)


def reach_metrics(ctrl, traj, target, seed, max_steps):
    log = ctrl.log
    errs = np.array([r["err"] for r in log], float)
    bands = [r["band"] for r in log]

    switches = sum(1 for i in range(1, len(bands)) if bands[i] != bands[i - 1])
    seglens, cur, n = [], bands[0], 1
    for b in bands[1:]:
        if b == cur:
            n += 1
        else:
            seglens.append((cur, n)); cur, n = b, 1
    seglens.append((cur, n))
    active = [n for b, n in seglens if b is not None]
    n_seg = len(active)
    n_seg_lt_engage = sum(1 for k in active if k < ENGAGE)

    overshoot_frac = float(np.mean(np.diff(errs) > 1e-6)) if len(errs) > 1 else 0.0
    pl = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
    straight = float(np.linalg.norm(traj[-1] - traj[0]))
    path_ratio = pl / max(straight, 1e-9)

    final_err = float(np.linalg.norm(target - traj[-1]))
    db = ctrl.deadband
    landed = final_err <= db
    steps = len(log)
    capped = (steps >= max_steps) and not landed

    # outcome taxonomy: separates the failure MODES, which need different fixes
    if landed:
        outcome = "land"
    elif final_err <= 2 * db:
        outcome = "near_miss"          # stopped just outside the deadband (terminal precision)
    elif switches > 20:
        outcome = "limit_cycle"        # thrashing between bands (switching instability)
    else:
        outcome = "stall"              # gave up / stuck far from target

    return dict(seed=int(seed), x=float(target[0]), y=float(target[1]),
                radius=float(np.hypot(*target)), landed=bool(landed), outcome=outcome,
                final_err=round(final_err, 3), steps=int(steps), switches=int(switches),
                n_seg=int(n_seg), n_seg_lt_engage=int(n_seg_lt_engage),
                overshoot_frac=round(overshoot_frac, 3), path_ratio=round(path_ratio, 3),
                capped=bool(capped))


# ---------------------------------------------------------------------------
# Grid + sweep
# ---------------------------------------------------------------------------
def build_targets(radii, n_ang, with_probes=True):
    pts, seen = [], set()
    for r in radii:
        for k in range(n_ang):
            a = 2 * np.pi * k / n_ang
            p = (round(r * np.cos(a), 2), round(r * np.sin(a), 2))
            if p not in seen:
                seen.add(p); pts.append(p)
    if with_probes:
        for p in PROBES:
            if p not in seen:
                seen.add(p); pts.append(p)
    return pts


def evaluate(use_mock, seeds, targets, max_steps, ctrl_kwargs, quiet=False):
    rows = []
    print(f"[eval] {'MOCK' if use_mock else 'REAL'}  seeds={seeds}  "
          f"{len(targets)} targets  max_steps={max_steps}  ctrl_kwargs={ctrl_kwargs or '{}'}")
    for seed in seeds:
        seed_rows = []
        for tg in targets:
            try:
                m = run_reach(use_mock, seed, tg, max_steps, ctrl_kwargs)
            except Exception as ex:
                if not quiet:
                    print(f"  seed {seed} ({tg[0]:.1f},{tg[1]:.1f}) FAILED: {ex}")
                continue
            seed_rows.append(m); rows.append(m)
            if not quiet:
                print(f"  s{seed} ({m['x']:6.1f},{m['y']:6.1f}) {m['outcome']:11s} "
                      f"|e|={m['final_err']:5.2f} steps={m['steps']:4d} sw={m['switches']:3d} "
                      f"pr={m['path_ratio']:.2f}")
        landed = sum(r["landed"] for r in seed_rows)
        print(f"  [seed {seed}] landed {landed}/{len(seed_rows)} "
              f"({100*landed/max(len(seed_rows),1):.0f}%)")
    return rows


# ---------------------------------------------------------------------------
# Aggregation across seeds (per target)
# ---------------------------------------------------------------------------
def aggregate_by_target(rows):
    bykey = {}
    for r in rows:
        bykey.setdefault((r["x"], r["y"]), []).append(r)
    agg = []
    for (x, y), rs in bykey.items():
        n = len(rs)
        landed = [r for r in rs if r["landed"]]
        from collections import Counter
        oc = Counter(r["outcome"] for r in rs)
        agg.append(dict(
            x=x, y=y, radius=round(float(np.hypot(x, y)), 2), n_seeds=n,
            landing_rate=round(len(landed) / n, 3),
            median_steps_landed=int(np.median([r["steps"] for r in landed])) if landed else -1,
            median_path_ratio_landed=round(float(np.median([r["path_ratio"] for r in landed])), 3) if landed else -1,
            mean_final_err=round(float(np.mean([r["final_err"] for r in rs])), 3),
            limit_cycle_rate=round(oc["limit_cycle"] / n, 3),
            near_miss_rate=round(oc["near_miss"] / n, 3),
            stall_rate=round(oc["stall"] / n, 3),
        ))
    agg.sort(key=lambda d: (d["radius"], np.arctan2(d["y"], d["x"])))
    return agg


def summarize(rows, agg, use_mock, seeds):
    from collections import Counter
    n = len(rows)
    landed = sum(r["landed"] for r in rows)
    oc = Counter(r["outcome"] for r in rows)
    lines = []
    P = lines.append
    P("=" * 60)
    P(f" plant            : {'MOCK' if use_mock else 'REAL'}")
    P(f" seeds            : {seeds}")
    P(f" reaches          : {n}  ({len(agg)} targets x {len(seeds)} seeds)")
    P(f" overall landing  : {landed}/{n}  ({100*landed/max(n,1):.1f}%)")
    P(f" outcome mix      : land={oc['land']} near_miss={oc['near_miss']} "
      f"limit_cycle={oc['limit_cycle']} stall={oc['stall']}")
    lr = np.array([a["landing_rate"] for a in agg])
    P(f" targets reliable (landing_rate==1.0)   : {int((lr==1.0).sum())}/{len(agg)}")
    P(f" targets robust   (landing_rate>=0.8)   : {int((lr>=0.8).sum())}/{len(agg)}")
    P(f" targets fragile  (0<landing_rate<0.8)  : {int(((lr>0)&(lr<0.8)).sum())}/{len(agg)}")
    P(f" targets dead     (landing_rate==0.0)   : {int((lr==0.0).sum())}/{len(agg)}")
    ml = [r["steps"] for r in rows if r["landed"]]
    mp = [r["path_ratio"] for r in rows if r["landed"]]
    if ml:
        P(f" median steps (landed)      : {int(np.median(ml))}")
        P(f" median path ratio (landed) : {np.median(mp):.2f}")
    # per-seed landing
    for s in seeds:
        srs = [r for r in rows if r["seed"] == s]
        if srs:
            P(f"   seed {s}: {sum(r['landed'] for r in srs)}/{len(srs)} landed")
    P("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def write_csv(rows, path, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[csv] wrote {path}")


def plot_map(agg, path, use_mock, seeds):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 6.4))
    th = np.linspace(0, 2 * np.pi, 200)
    for a in ax:
        a.plot(60 * np.cos(th), 60 * np.sin(th), color="lightgray", lw=1)
        a.scatter([60], [0], c="k", marker="s", s=30, zorder=3)        # home
        a.set_aspect("equal"); a.set_xlim(-65, 65); a.set_ylim(-65, 65)
        a.set_xlabel("x"); a.set_ylabel("y")

    xs = [a["x"] for a in agg]; ys = [a["y"] for a in agg]

    # panel 0: landing rate across seeds
    lr = [a["landing_rate"] for a in agg]
    sc = ax[0].scatter(xs, ys, c=lr, cmap="RdYlGn", vmin=0, vmax=1, s=80,
                       edgecolor="k", lw=0.4, zorder=5)
    fig.colorbar(sc, ax=ax[0], label="landing rate over seeds", fraction=0.046, pad=0.04)
    overall = 100 * np.mean([a["landing_rate"] for a in agg])
    ax[0].set_title(f"Landing rate  ({'MOCK' if use_mock else 'REAL'}, {len(seeds)} seeds)\n"
                    f"target-mean {overall:.0f}%")

    # panel 1: path quality among landed; ring targets that fail on a majority of seeds
    pr = [min(a["median_path_ratio_landed"], 6.0) if a["median_path_ratio_landed"] > 0 else np.nan
          for a in agg]
    sc2 = ax[1].scatter(xs, ys, c=pr, cmap="inferno_r", vmin=1.0, vmax=6.0, s=80,
                        edgecolor="k", lw=0.4, zorder=5)
    fig.colorbar(sc2, ax=ax[1], label="median path ratio, landed (clip@6)", fraction=0.046, pad=0.04)
    frag = [a for a in agg if a["landing_rate"] < 0.5]
    if frag:
        ax[1].scatter([a["x"] for a in frag], [a["y"] for a in frag], facecolors="none",
                     edgecolors="red", s=190, lw=1.8, zorder=7, label="lands <50% of seeds")
        ax[1].legend(loc="upper left", fontsize=9)
    ax[1].set_title("Path quality & fragile targets")

    fig.tight_layout(); fig.savefig(path); print(f"[figure] wrote {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# A/B comparison of two tagged runs (for applying improvements one by one)
# ---------------------------------------------------------------------------
def load_reaches(tag):
    path = os.path.join(DOCS, f"eval_{tag}_reaches.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["x"] = float(r["x"]); r["y"] = float(r["y"])
        r["landed"] = r["landed"] == "True"
        r["seed"] = int(r["seed"])
    return rows


def compare(tag_a, tag_b):
    a, b = load_reaches(tag_a), load_reaches(tag_b)
    def rate(rows):
        return sum(r["landed"] for r in rows) / max(len(rows), 1)
    print(f"\n=== compare  A={tag_a}  vs  B={tag_b} ===")
    print(f" overall landing:  A {100*rate(a):.1f}%  ->  B {100*rate(b):.1f}%  "
          f"(Δ {100*(rate(b)-rate(a)):+.1f} pts)")
    # per-target landing rate, by (x,y)
    def bytarget(rows):
        d = {}
        for r in rows:
            d.setdefault((r["x"], r["y"]), []).append(r["landed"])
        return {k: np.mean(v) for k, v in d.items()}
    ra, rb = bytarget(a), bytarget(b)
    keys = sorted(set(ra) & set(rb))
    regress = [(k, ra[k], rb[k]) for k in keys if rb[k] < ra[k] - 1e-9]
    improve = [(k, ra[k], rb[k]) for k in keys if rb[k] > ra[k] + 1e-9]
    print(f" targets improved: {len(improve)}   regressed: {len(regress)}")
    for label, lst in (("REGRESSED", regress), ("improved", improve)):
        for (x, y), va, vb in sorted(lst, key=lambda t: t[2] - t[1])[:12]:
            print(f"   {label:9s} ({x:6.1f},{y:6.1f})  {va:.2f} -> {vb:.2f}")
    print("=" * 50)


# ---------------------------------------------------------------------------
def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-"); out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def parse_ctrl_kw(pairs):
    kw = {}
    for p in pairs or []:
        k, v = p.split("=")
        try:
            kw[k] = int(v)
        except ValueError:
            try:
                kw[k] = float(v)
            except ValueError:
                kw[k] = v
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="real cascade (default: mock)")
    ap.add_argument("--seeds", default="0-4", help="e.g. 0-7 or 0,1,2 (default 0-4)")
    ap.add_argument("--radii", default="15,30,45", help="comma list (default 15,30,45)")
    ap.add_argument("--angles", type=int, default=12, help="angular samples per radius")
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--targets", nargs="+", type=float, default=None, help="explicit x y x y ...")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--tag", default=None, help="output filename suffix (default: v1/real)")
    ap.add_argument("--ctrl-kw", nargs="+", default=None, help="controller kwargs, key=val ...")
    ap.add_argument("--quiet", action="store_true", help="suppress per-reach lines")
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"),
                    help="compare two prior tagged runs and exit")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare); return

    seeds = parse_seeds(args.seeds)
    ctrl_kwargs = parse_ctrl_kw(args.ctrl_kw)
    if args.targets is not None:
        v = args.targets
        if len(v) % 2:
            ap.error("--targets needs x y pairs")
        targets = [(v[i], v[i + 1]) for i in range(0, len(v), 2)]
    else:
        radii = [float(x) for x in args.radii.split(",")]
        targets = build_targets(radii, args.angles, with_probes=not args.no_probes)

    rows = evaluate(use_mock=not args.real, seeds=seeds, targets=targets,
                    max_steps=args.max_steps, ctrl_kwargs=ctrl_kwargs, quiet=args.quiet)
    if not rows:
        print("no reaches recorded"); return
    agg = aggregate_by_target(rows)
    summary = summarize(rows, agg, not args.real, seeds)
    print("\n" + summary)

    os.makedirs(DOCS, exist_ok=True)
    tag = args.tag or ("real" if args.real else "v1")
    write_csv(rows, os.path.join(DOCS, f"eval_{tag}_reaches.csv"),
              ["seed", "x", "y", "radius", "landed", "outcome", "final_err", "steps",
               "switches", "n_seg", "n_seg_lt_engage", "overshoot_frac", "path_ratio", "capped"])
    write_csv(agg, os.path.join(DOCS, f"eval_{tag}_bytarget.csv"),
              ["x", "y", "radius", "n_seeds", "landing_rate", "median_steps_landed",
               "median_path_ratio_landed", "mean_final_err", "limit_cycle_rate",
               "near_miss_rate", "stall_rate"])
    with open(os.path.join(DOCS, f"eval_{tag}_summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"[txt] wrote {os.path.join(DOCS, f'eval_{tag}_summary.txt')}")
    if not args.no_fig:
        plot_map(agg, os.path.join(DOCS, f"eval_{tag}_map.pdf"), not args.real, seeds)


if __name__ == "__main__":
    main()
