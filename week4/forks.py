"""forks.py -- resolve the two 2C design forks on the live cascade (Step 2B, S5).

Fork 1 -- coordinated two-joint motion: drive a TWO-TONE sum (a shoulder band +
an elbow band) at power and ask whether both joints move and whether it
**superposes** (each joint ~ its single-tone velocity) or **interferes** (energy
splits / the wrong muscle cross-fires). Clean superposition would let 2C use the
coordinated reacher; otherwise we stay one-joint-at-a-time (the default).

Fork 2 -- open-loop reliability: across seeds, how often does a primitive fail to
fire / fire the wrong joint within the onset window? In regime A the hand feedback
covers occasional misfires, but the rate tells 2C whether an inner x1 check is ever
needed (expected: not needed).

Reuses calibrate.py's IK tracker / drivers / pose prep, and primitives.py bands.
"""

from __future__ import annotations

import numpy as np

import primitives as prim
import calibrate as cal
from cascade_plant import CascadePlant

# two-tone amplitude per tone: 2*A_TONE < 0.5 so the summed drive never clips
A_TONE = 0.24


def drive_seq(plant, seq, theta0, *, r_stop=cal.ELBOW_RSTOP):
    """Apply a precomputed (T,2) drive; log theta (continuous IK) / hand / power.

    Stops early if the hand radius leaves ``r_stop`` (keeps the elbow off the
    centre/rim singularities where IK tracking would flip)."""
    th = np.asarray(theta0, dtype=float)
    rec = {"theta": [], "hand": [], "power": []}
    for t, u in enumerate(seq):
        hand = plant.command(u)
        th = cal.track_theta(hand, th)
        rec["theta"].append(th.copy())
        rec["hand"].append(hand.copy())
        rec["power"].append(cal.read_power(plant))
        if r_stop is not None and t > 5:
            rr = float(np.linalg.norm(hand))
            if rr < r_stop[0] or rr > r_stop[1]:
                break
    for key in rec:
        rec[key] = np.asarray(rec[key])
    return rec


def velocity_in_annulus(rec, joint, *, r_lo=8.0, r_hi=54.0, skip=40):
    """Slope of a joint angle over the post-onset, in-annulus part of a run."""
    r = np.linalg.norm(rec["hand"], axis=1)
    idx = np.where((r > r_lo) & (r < r_hi))[0]
    idx = idx[idx >= skip]
    if len(idx) < 12:
        n = len(rec["theta"])
        idx = np.arange(max(0, n - 30), n)  # fallback
    seg = rec["theta"][idx, joint]
    return float(np.polyfit(np.arange(len(seg)), seg, 1)[0])


# --- Fork 1: two-tone coordination ------------------------------------------
def fork_two_tone(seed=0, k_sh=0, k_el=2, A=A_TONE, T=260, start_r=50.0):
    """Compare two-tone (sh band + el band) joint velocities to single tones."""
    f_sh, f_el = prim.BAND_F[k_sh], prim.BAND_F[k_el]
    js, je = 0, 1  # shoulder, elbow joint indices

    def run(seq):
        plant = CascadePlant(seed=seed)
        th = cal.prepare(plant, start_r, cool_after=False)  # keep power latched
        return drive_seq(plant, seq, th)

    single_sh = run(prim.drive_sequence(A, f_sh, T))
    single_el = run(prim.drive_sequence(A, f_el, T))
    both = run(prim.two_tone(A, f_sh, f_el, T))

    vs_single = abs(velocity_in_annulus(single_sh, js))
    ve_single = abs(velocity_in_annulus(single_el, je))
    vs_both = abs(velocity_in_annulus(both, js))
    ve_both = abs(velocity_in_annulus(both, je))
    return {"vs_single": vs_single, "ve_single": ve_single,
            "vs_both": vs_both, "ve_both": ve_both,
            "ratio_sh": vs_both / (vs_single + 1e-9),
            "ratio_el": ve_both / (ve_single + 1e-9)}


def run_fork1(seeds=(0, 1, 2, 3)):
    print("=== Fork 1: two-tone coordination (sh+ band + el+ band, A=%.2f each) ===" % A_TONE)
    rs, re = [], []
    for s in seeds:
        r = fork_two_tone(seed=s)
        rs.append(r["ratio_sh"]); re.append(r["ratio_el"])
        print(f"  seed {s}: shoulder {r['vs_both']:.4f}/{r['vs_single']:.4f}="
              f"{r['ratio_sh']:.2f}x   elbow {r['ve_both']:.4f}/{r['ve_single']:.4f}="
              f"{r['ratio_el']:.2f}x")
    rs, re = np.array(rs), np.array(re)
    print(f"  mean retained: shoulder {rs.mean():.2f}x, elbow {re.mean():.2f}x")
    clean = rs.mean() > 0.6 and re.mean() > 0.6
    verdict = ("SUPERPOSE (both joints keep >60% of single-tone speed) -> "
               "2C may use coordinated reach") if clean else \
              ("INTERFERE (a joint loses speed / energy splits) -> "
               "2C stays one-joint-at-a-time")
    print(f"  VERDICT: {verdict}")
    return clean


# --- Fork 2: open-loop misfire rate -----------------------------------------
def run_fork2(seeds=range(8), A=0.49, T=210, fire_thresh=0.004):
    print(f"\n=== Fork 2: open-loop misfire rate ({len(list(seeds))} seeds, A={A}) ===")
    seeds = list(seeds)
    misfire = {k: 0 for k in range(4)}
    detail = {k: [] for k in range(4)}
    for s in seeds:
        for k, (name, joint, sign, antag) in enumerate(prim.MUSCLES):
            jidx = prim.JOINT_INDEX[joint]
            other = 1 - jidx
            r_stop = None if joint == "shoulder" else cal.ELBOW_RSTOP
            plant = CascadePlant(seed=s)
            th = cal.prepare_cold(plant, cal.START_R[k])    # cold start (power off)
            rec = cal.drive_log(plant, A, prim.BAND_F[k], T, th, r_stop=r_stop)
            vd = cal.velocity_in_annulus(rec, jidx, skip=0)
            vo = abs(cal.velocity_in_annulus(rec, other, skip=0))
            fired = abs(vd) >= fire_thresh and np.sign(vd) == sign and abs(vd) >= vo
            detail[k].append(abs(vd))
            if not fired:
                misfire[k] += 1
    print(f"  {'muscle':>9s} {'misfire':>8s} {'mean|v|':>8s}")
    total = 0
    for k, (name, *_ ) in enumerate(prim.MUSCLES):
        rate = misfire[k] / len(seeds)
        total += misfire[k]
        print(f"  {name:>9s} {misfire[k]}/{len(seeds)} = {rate:4.0%}   "
              f"{np.mean(detail[k]):.4f}")
    overall = total / (4 * len(seeds))
    print(f"  overall misfire rate: {overall:.0%}  "
          f"(el- is the expected offender; hand feedback covers it in regime A)")
    return overall


def main():
    clean = run_fork1()
    rate = run_fork2()
    print("\n=== fork verdicts ===")
    print(f"  coordination: {'superpose' if clean else 'interfere -> one-joint default'}")
    print(f"  open-loop misfire rate: {rate:.0%}")


if __name__ == "__main__":
    main()
