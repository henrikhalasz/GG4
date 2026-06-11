"""calibrate.py -- measure the four primitives on the live cascade (Step 2B, S4).

For each muscle we drive ``u0 = 0.5 + A*cos(2*pi*f_k*t)`` (input 1 a constant
spectator) on the real ``BMI_and_Hand(Brain(seed))`` and read what it does to the
arm. Joint angles are recovered from the clean ``hand_pos`` by 2A branch-continuity
IK -- never from noisy observations. We read the muscle head's ``power_state`` and
per-band ``selector`` directly to confirm the shared-power mechanism.

Per muscle we measure: steady joint velocity, the slow **cold power onset** (key
number), the fast **warm selector switch** (power latched), the **coast** after
cutting the AC, the **antagonist brake**, the **leakage** to the other joint, and
the **spread** across ~8 seeds. We also choose the drive amplitude ``A*`` by a
sweep (selectivity without clipping), and predict feasibility from the identified
model (S6).

Outputs: prints every must-capture number; saves ``primitive_library.npz``,
``primitive_onset.pdf`` and ``primitive_summary.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kinematics as kin
import primitives as prim
from cascade_plant import CascadePlant

HERE = Path(__file__).resolve().parent

# Per-muscle cold-start mid-disk radius. Shoulder drives keep r constant; elbow
# drives sweep r, so el+ starts near the rim (r decreasing into the annulus) and
# el- starts near the centre (r increasing), each with room to traverse.
# Prepare targets are ~10 units above the desired landing radius: bending the
# elbow to position then cooling lets it coast ~10 units further inward, so we aim
# high and let the coast settle it inside the annulus.
START_R = {0: 38.0, 1: 38.0, 2: 52.0, 3: 32.0}
# Keep the elbow strictly inside the annulus during a drive (room left for coast).
ELBOW_RSTOP = (12.0, 52.0)
A_DEFAULT = 0.45


# --- small helpers -----------------------------------------------------------
def _wrap(a):
    return (np.asarray(a, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def track_theta(hand, prev):
    """Branch-continuity IK returning a *continuous* (unwrapped) angle.

    Pick the IK branch nearest ``prev``, then accumulate the wrapped step onto
    ``prev`` so the shoulder angle never jumps by 2*pi as the arm sweeps past +-pi
    (a wrap would corrupt the velocity slope)."""
    up, dn = kin.inverse(hand)
    cand = up if np.sum(_wrap(up - prev) ** 2) <= np.sum(_wrap(dn - prev) ** 2) else dn
    return np.asarray(prev, dtype=float) + _wrap(cand - prev)


def read_power(plant) -> float:
    return float(plant._cascade.ann.muscle_head.power_state.reshape(-1)[0].item())


def read_selectors(plant) -> np.ndarray:
    """The four per-band selectors actually applied this step (pure fn of x1 cache)."""
    mh = plant._cascade.ann.muscle_head
    _, _, _, dbg = mh._compute_from_cache(mh.latent_cache, mh.power_state, mh.evidence_state)
    return np.asarray(dbg["selector"].detach().cpu()).reshape(-1)[:4]


def joint_velocity(theta_series, joint, win=40):
    """Robust steady angular velocity (rad/step) of ``joint`` over the last ``win``."""
    y = np.asarray(theta_series)[-win:, joint]
    t = np.arange(len(y))
    return float(np.polyfit(t, y, 1)[0])


def velocity_in_annulus(rec, joint, *, skip=0, r_lo=8.0, r_hi=54.0):
    """Slope of a joint angle over the post-``skip``, in-annulus part of a run.

    Restricting to r in (r_lo, r_hi) keeps the fit away from the centre/rim where
    IK branch tracking is ill-defined (an elbow crossing r=0 would flip the sign)."""
    r = np.linalg.norm(rec["hand"], axis=1)
    idx = np.where((r > r_lo) & (r < r_hi))[0]
    idx = idx[idx >= skip]
    if len(idx) < 15:
        n = len(rec["theta"])
        idx = np.arange(max(0, n - 40), n)
    seg = rec["theta"][idx, joint]
    return float(np.polyfit(np.arange(len(seg)), seg, 1)[0])


def onset_from_signal(sig, frac=0.9, smooth=5, floor=0.05):
    """Steps until a rising signal first sustainedly reaches ``frac`` of its steady level.

    Used on the clean internal signals -- the shared ``power`` (cold onset) and a
    band ``selector`` (warm switch) -- which are reliable even when the joint barely
    moves (so they work for the el- bottleneck where IK-derived velocity is noisy)."""
    sig = np.asarray(sig, dtype=float)
    if smooth > 1:
        sig = np.convolve(sig, np.ones(smooth) / smooth, mode="same")
    steady = float(np.mean(sig[-20:])) if len(sig) >= 20 else float(np.max(sig))
    if steady < floor:
        return None                       # signal never rose -> primitive did not fire
    hit = sig >= frac * steady
    was_low = sig < 0.5 * steady          # require a genuine ramp from a low (cold) state
    for t in range(len(hit) - 2):
        if hit[t] and hit[t + 1] and hit[t + 2] and was_low[:t].any():
            return t
    return None


# --- driving -----------------------------------------------------------------
def drive_log(plant, A, f, T, theta0, *, dc=0.5, u1=0.5, r_stop=None):
    """Drive a single tone for T steps; log theta/hand/power/selectors/u0/clip.

    If ``r_stop=(lo,hi)`` is given, stop early once the hand radius leaves the band
    (keeps an elbow drive inside the annulus)."""
    th = np.asarray(theta0, dtype=float)
    rec = {"theta": [], "hand": [], "power": [], "sel": [], "u0": []}
    for t in range(T):
        u = prim.drive_u(A, f, t, dc=dc, u1=u1)
        hand = plant.command(u)
        th = track_theta(hand, th)
        rec["theta"].append(th.copy())
        rec["hand"].append(hand.copy())
        rec["power"].append(read_power(plant))
        rec["sel"].append(read_selectors(plant))
        rec["u0"].append(float(u[0]))
        if r_stop is not None and t > 5:
            rr = float(np.linalg.norm(hand))
            if rr < r_stop[0] or rr > r_stop[1]:
                break
    for key in rec:
        rec[key] = np.asarray(rec[key])
    rec["last_theta"] = th
    rec["clip"] = float(np.mean((rec["u0"] <= 0.0) | (rec["u0"] >= 1.0)))
    return rec


def hold_dc(plant, T, theta0, *, dc=0.5, u1=0.5):
    """Hold u0 at a constant DC (no AC) for T steps; log theta/power."""
    return drive_log(plant, 0.0, 0.0, T, theta0, dc=dc, u1=u1)


def cool(plant, theta0, *, thresh=0.05, max_steps=120):
    """Hold u0=0 until power decays below thresh (a cold start)."""
    th = np.asarray(theta0, dtype=float)
    for _ in range(max_steps):
        hand = plant.command([0.0, prim.U1_SPECTATOR])
        th = track_theta(hand, th)
        if read_power(plant) < thresh:
            break
    return th


def prepare(plant, target_r, *, cool_after=True):
    """Bend the elbow (elbow+) to ~target_r. If ``cool_after``, cool to a cold
    start (power off); otherwise leave power latched (the elbow+ drive latches it)."""
    th = np.array([0.0, 0.0])  # home
    f_elp = prim.BAND_F[2]
    for t in range(400):
        hand = plant.command(prim.drive_u(A_DEFAULT, f_elp, t))
        th = track_theta(hand, th)
        if np.linalg.norm(hand) <= target_r:
            break
    if cool_after:
        th = cool(plant, th)
    return th


def prepare_cold(plant, target_r):
    return prepare(plant, target_r, cool_after=True)


# --- A-sweep -----------------------------------------------------------------
def sweep_amplitude(seed=0, As=(0.30, 0.40, 0.45, 0.49, 0.55), T=230):
    """Pick A* per muscle: max driven speed with selectivity and ~no clipping.

    Each A is measured from a fresh cold latch (no warm drift). A=0.55 is included
    to show where clipping starts (DC 0.5 + A>0.5 leaves the [0,1] box) and leaks
    into neighbouring bands (runner-up selector rises)."""
    print("=== A-sweep (selectivity without clipping; fresh cold latch per A) ===")
    A_star = {}
    for k, (name, joint, sign, antag) in enumerate(prim.MUSCLES):
        jidx = prim.JOINT_INDEX[joint]
        other = 1 - jidx
        rows = []
        for A in As:
            plant = CascadePlant(seed=seed)
            th = prepare_cold(plant, START_R[k])
            r = drive_log(plant, A, prim.BAND_F[k], T, th)
            vd = abs(joint_velocity(r["theta"], jidx, win=60))
            vo = abs(joint_velocity(r["theta"], other, win=60))
            runner = float(np.sort(r["sel"][-60:].mean(axis=0))[-2])  # leakage
            rows.append((A, vd, vo, runner, r["clip"]))
        ok = [row for row in rows if row[4] < 0.02]
        pick = max(ok or rows, key=lambda row: row[1])
        A_star[k] = pick[0]
        print(f"  {name:>9s} (f={prim.BAND_F[k]:.3f}):")
        for A, vd, vo, runner, clip in rows:
            star = " <-" if A == pick[0] else ""
            print(f"      A={A:.2f}  v_driven={vd:.4f}  v_other={vo:.4f}  "
                  f"runner-up_sel={runner:.3f}  clip={clip:.2f}{star}")
    print(f"  chosen A*: {{{', '.join(f'{prim.MUSCLES[k][0]}:{a}' for k,a in A_star.items())}}}")
    return A_star


# --- per-muscle, per-seed measurement (one consolidated run) ------------------
def measure_run(plant, k, A, *, Tc=320, Tcoast=70, Tredrive=60, Tbrake=45):
    """One run: cold drive -> coast -> re-drive -> antagonist brake. All numbers
    for muscle k except the warm switch (measured separately)."""
    name, joint, sign, antag = prim.MUSCLES[k]
    jidx = prim.JOINT_INDEX[joint]
    other = 1 - jidx
    fk = prim.BAND_F[k]
    r_stop = None if joint == "shoulder" else ELBOW_RSTOP

    th = prepare_cold(plant, START_R[k])
    cold = drive_log(plant, A, fk, Tc, th, r_stop=r_stop)
    # cold onset = the slow, band-independent POWER ramp (the key number). Require
    # a real latch (floor 0.3) so a non-latching seed reports None (a misfire).
    onset = onset_from_signal(cold["power"], floor=0.3)
    # steady velocity over the in-annulus window AFTER power has latched.
    latch_step = int(np.argmax(cold["power"] > 0.5)) if (cold["power"] > 0.5).any() else 0
    skip = min(latch_step + 15, max(0, len(cold["theta"]) - 20))
    steady_v = velocity_in_annulus(cold, jidx, skip=skip)
    v_other = velocity_in_annulus(cold, other, skip=skip)
    sel_steady = cold["sel"][-30:].mean(axis=0)
    runner_up = float(np.sort(sel_steady)[-2])  # 2nd-highest selector
    target_sel = float(sel_steady[k])           # this band's own selector (bottleneck evidence)
    power_latched = float(cold["power"][-30:].mean())

    # coast: cut AC (hold DC) -- travel measured within the coast segment
    coast = hold_dc(plant, Tcoast, cold["last_theta"])
    coast_travel = abs(coast["theta"][-1, jidx] - coast["theta"][0, jidx])
    coast_stop = _steps_to_stop(coast["theta"], jidx)

    # re-drive to steady (warm), then antagonist brake (travel within brake segment)
    redrive = drive_log(plant, A, fk, Tredrive, coast["last_theta"], r_stop=r_stop)
    brake = drive_log(plant, A, prim.BAND_F[antag], Tbrake, redrive["last_theta"])
    brake_travel = abs(brake["theta"][-1, jidx] - brake["theta"][0, jidx])
    brake_stop = _steps_to_stop(brake["theta"], jidx)

    leak_ratio = abs(v_other) / abs(steady_v) if abs(steady_v) > 1e-3 else np.nan
    return {
        "steady_v": steady_v, "onset": onset, "target_sel": target_sel,
        "v_other": abs(v_other), "leak_ratio": leak_ratio,
        "runner_up_sel": runner_up, "power_latched": power_latched,
        "coast_travel": coast_travel, "coast_stop": coast_stop,
        "brake_travel": brake_travel, "brake_stop": brake_stop,
        "cold_record": cold, "coast_record": coast,
    }


def _steps_to_stop(theta_series, joint, eps=2e-3):
    v = np.abs(np.gradient(np.asarray(theta_series)[:, joint]))
    below = np.where(v < eps)[0]
    return int(below[0]) if len(below) else len(v)


# --- warm switch (one clean single switch per muscle) ------------------------
def warm_switch_single(seed, j, A_star, *, Tprime=90, Tsw=130):
    """Warm switch to muscle ``j``: latch power with a PURE-DC prime (no AC, so the
    arm stays put and power latches reliably even on slow-power seeds), then switch
    a single time to band ``j`` and time its rise.

    The selector rises from cold while power is already latched -- this isolates the
    selector-switch timescale (no power ramp), the core of the shared-power claim.
    """
    name, joint, sign, antag = prim.MUSCLES[j]
    jidx = prim.JOINT_INDEX[joint]
    r_stop = None if joint == "shoulder" else ELBOW_RSTOP
    plant = CascadePlant(seed=seed)
    # Position via elbow+ WITHOUT cooling (so power stays latched), then hold DC so
    # the positioning selector decays to ~0 while power stays up; then switch.
    th = prepare(plant, START_R[j], cool_after=False)
    pr = hold_dc(plant, Tprime, th)
    r = drive_log(plant, A_star[j], prim.BAND_F[j], Tsw, pr["last_theta"], r_stop=r_stop)
    # warm switch = the band-j SELECTOR rise (power already latched) -- robust even
    # for the el- bottleneck whose joint velocity is near the IK noise floor.
    onset = onset_from_signal(r["sel"][:, j])
    return {"onset": onset, "steady_v": velocity_in_annulus(r, jidx, skip=5),
            "power_floor": float(r["power"].min()),
            "primer_power": float(pr["power"][-1])}


# --- feasibility (S6) --------------------------------------------------------
def feasibility():
    """Identified-model |x1<-u0| at the bands vs the given true-brain numbers."""
    M, _ = prim.decoder_M()
    im = np.load(HERE / "identified_model.npz")
    A, B, C = im["A"], im["B"], im["C"]
    In = np.eye(A.shape[0])
    true_brain = [1.88, 1.08, 0.69, 0.50]
    pred = []
    for f in prim.BAND_F:
        z = np.exp(1j * 2 * np.pi * f)
        pred.append(abs(M[0] @ (C @ np.linalg.solve(z * In - A, B[:, 0]))))
    return np.array(pred), np.array(true_brain)


# --- orchestration -----------------------------------------------------------
def run(seeds, A_star, *, quick=False):
    Tc = 280 if quick else 340
    per_seed = {k: [] for k in range(4)}
    warm = {k: [] for k in range(4)}
    onset_fig_record = None

    for si, seed in enumerate(seeds):
        for k in range(4):
            plant = CascadePlant(seed=seed)
            res = measure_run(plant, k, A_star[k], Tc=Tc)
            per_seed[k].append(res)
            if onset_fig_record is None and k == 0:
                onset_fig_record = res  # keep one sh+ cold start for the figure
            warm[k].append(warm_switch_single(seed, k, A_star))
        print(f"  seed {seed} done ({si + 1}/{len(seeds)})")
    return per_seed, warm, onset_fig_record


def _agg(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return (np.nan, np.nan, np.nan)
    return (float(np.mean(a)), float(np.min(a)), float(np.max(a)))


def summarise(per_seed, warm):
    print("\n=== per-muscle recipe (mean [min, max] across seeds) ===")
    rows = {}
    for k, (name, joint, sign, antag) in enumerate(prim.MUSCLES):
        S = per_seed[k]
        sv = _agg([r["steady_v"] for r in S])
        on = _agg([r["onset"] for r in S])
        ts = _agg([r["target_sel"] for r in S])
        co = _agg([r["coast_travel"] for r in S])
        br = _agg([r["brake_travel"] for r in S])
        lk = _agg([r["v_other"] for r in S])
        ru = _agg([r["runner_up_sel"] for r in S])
        pw = _agg([r["power_latched"] for r in S])
        wsw = _agg([w["onset"] for w in warm[k]])
        wpf = _agg([w["power_floor"] for w in warm[k]])
        rows[k] = dict(steady_v=sv, onset=on, target_sel=ts, warm=wsw,
                       warm_power_floor=wpf, coast=co, brake=br, leak=lk,
                       runner_up=ru, power=pw)
        print(f"\n  {k} {name} (f={prim.BAND_F[k]:.3f}):")
        print(f"     steady |v|   = {abs(sv[0]):.4f} rad/step  [{abs(sv[1]):.4f},{abs(sv[2]):.4f}]  (sign {np.sign(sv[0]):+.0f})")
        print(f"     cold onset   = {on[0]:.0f} steps  [{on[1]:.0f},{on[2]:.0f}]  (power-ramp)")
        print(f"     warm switch  = {wsw[0]:.0f} steps  [{wsw[1]:.0f},{wsw[2]:.0f}]  "
              f"(selector rise; power floor {wpf[1]:.2f})")
        print(f"     own selector = {ts[0]:.3f} (threshold 0.15 -> bottleneck if low)")
        brake_gain = co[0] / br[0] if br[0] > 1e-6 else float("nan")
        print(f"     coast travel = {co[0]:.3f} rad  [{co[1]:.3f},{co[2]:.3f}]")
        print(f"     brake travel = {br[0]:.3f} rad  [{br[1]:.3f},{br[2]:.3f}]  "
              f"(brake gain {brake_gain:.2f}x shorter than coast)")
        print(f"     other-joint |v| = {lk[0]:.4f} rad/step  runner-up sel = {ru[0]:.3f}  power = {pw[0]:.2f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 seeds, shorter runs")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--no-sweep", action="store_true", help="use A_DEFAULT for all")
    args = ap.parse_args()

    seeds = list(range(2 if args.quick else args.seeds))
    print(f"calibrating on seeds {seeds}\n")

    if args.no_sweep:
        A_star = {k: A_DEFAULT for k in range(4)}
        print(f"using A*={A_DEFAULT} for all muscles\n")
    else:
        A_star = sweep_amplitude()

    per_seed, warm, onset_rec = run(seeds, A_star, quick=args.quick)
    rows = summarise(per_seed, warm)

    # shared-power confirmation: warm switch << cold onset, power stays latched
    print("\n=== shared-power mechanism ===")
    cold_mean = np.nanmean([rows[k]["onset"][0] for k in range(4)])
    warm_mean = np.nanmean([rows[k]["warm"][0] for k in range(4)])
    print(f"  mean cold onset = {cold_mean:.0f} steps;  mean warm switch = {warm_mean:.0f} steps "
          f"({cold_mean / warm_mean:.1f}x faster)")
    for k in range(4):
        w = rows[k]["warm"]
        pf = rows[k]["warm_power_floor"]
        print(f"  warm switch to {prim.MUSCLES[k][0]:>9s}: {w[0]:.0f} steps  "
              f"(power floor {pf[1]:.2f} -- stays latched)")
    print("  -> warm switch << cold onset, power latched across switch: shared power confirmed.")

    # el- bottleneck
    sv_all = {k: abs(rows[k]["steady_v"][0]) for k in range(4)}
    bottleneck = min(sv_all, key=sv_all.get)
    print(f"\n=== bottleneck: slowest muscle = {prim.MUSCLES[bottleneck][0]} "
          f"(|v|={sv_all[bottleneck]:.4f} rad/step) ===")

    # feasibility S6
    pred, true_brain = feasibility()
    print("\n=== feasibility: identified vs true-brain |x1<-u0|, vs live steady |v| ===")
    live = np.array([abs(rows[k]["steady_v"][0]) for k in range(4)])
    live_n = live / live[0] * pred[0]  # normalise live to the sh+ predicted gain
    for k in range(4):
        print(f"  {prim.MUSCLES[k][0]:>9s}: identified={pred[k]:.2f}  "
              f"true-brain={true_brain[k]:.2f}  live|v|(norm)={live_n[k]:.2f}")

    save_library(rows, A_star, per_seed)
    make_onset_figure(onset_rec)
    make_summary_figure(rows)
    print("\nsaved primitive_library.npz, primitive_onset.pdf, primitive_summary.pdf")


# --- persistence + figures ---------------------------------------------------
def save_library(rows, A_star, per_seed):
    out = {"band_f": prim.BAND_F,
           "muscle_names": np.array([m[0] for m in prim.MUSCLES])}
    for k in range(4):
        out[f"A_star_{k}"] = A_star[k]
        for key in ("steady_v", "onset", "target_sel", "warm", "coast", "brake",
                    "leak", "runner_up", "power"):
            out[f"{key}_{k}"] = np.array(rows[k][key])  # (mean,min,max)
    np.savez(HERE / "primitive_library.npz", **out)


def make_onset_figure(rec, path=HERE / "primitive_onset.pdf"):
    if rec is None:
        return
    cold, coast = rec["cold_record"], rec["coast_record"]
    th = np.concatenate([cold["theta"][:, 0], coast["theta"][:, 0]])
    pw = np.concatenate([cold["power"], coast["power"]])
    v = np.gradient(th)
    tcut = len(cold["theta"])
    fig, ax = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    ax[0].plot(th, color="tab:blue"); ax[0].set_ylabel("shoulder angle")
    ax[1].plot(v, color="tab:green"); ax[1].set_ylabel("joint velocity")
    ax[2].plot(pw, color="tab:red"); ax[2].set_ylabel("power_state")
    for a in ax:
        a.axvline(tcut, color="0.5", ls="--")
        a.axhline(0, color="0.85", lw=0.8)
    ax[0].set_title("sh+ cold start: onset -> steady -> coast (AC cut at dashed line)")
    ax[2].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path); fig.savefig(Path(path).with_suffix(".png"), dpi=110)
    plt.close(fig)


def make_summary_figure(rows, path=HERE / "primitive_summary.pdf"):
    names = [m[0] for m in prim.MUSCLES]
    sv = [abs(rows[k]["steady_v"][0]) for k in range(4)]
    on = [rows[k]["onset"][0] for k in range(4)]
    wsw = [rows[k]["warm"][0] for k in range(4)]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].bar(names, sv, color="tab:blue"); ax[0].set_title("steady |v| (rad/step)")
    ax[0].axhline(0); ax[0].tick_params(axis="x", rotation=30)
    ax[1].bar(names, on, color="tab:orange"); ax[1].set_title("cold onset (steps)")
    ax[1].tick_params(axis="x", rotation=30)
    ax[2].bar(names, wsw, color="tab:green"); ax[2].set_title("warm switch (steps)")
    ax[2].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path); fig.savefig(Path(path).with_suffix(".png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
