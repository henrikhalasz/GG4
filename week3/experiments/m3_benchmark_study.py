"""M3 — Full benchmark study: is good prediction enough for good control?

Reframed central question (per the M3 directive): the honest pipeline
predicts at ~1.3× the noise floor but disagrees with the yardstick's
eigenvalues by 18–63 %. The data genuinely doesn't pin down structure on
this sloppy, near-rank-1 system. **So the question is not "how accurate
is identification" but "is good prediction enough for good control on a
system whose structure we can't recover?"**

Three sub-studies — control metrics are the headline; identification
metrics are diagnostic:

    A. Headline      — honest vs yardstick, closed-loop hold on the real
                       Brain, plus held-out prediction error as a side car.
    B. Data efficiency — T_cal ∈ {100, 200, 500, 1000, 2000}: closed-loop
                       settling/steady-state on the hold task, with
                       feedforward-only LQG vs LQGI overlay (integral
                       should help most at small T_cal where the model
                       is biased).
    C. Local minima  — EM from N4SID / output-only-SSI / random inits:
                       spread in BOTH held-out prediction AND closed-loop
                       control performance.

Outputs:
    results/m3_benchmark_study.png   — 4-panel: headline | T_cal sweep |
                                       integral overlay | local-minima
    results/m3_benchmark_study.txt   — printable numbers
    results/m3_benchmark_study.npz   — raw arrays for downstream re-use
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
from control.plant import BrainPlant  # noqa: E402
from control.observer import SteadyStateKalman  # noqa: E402
from control.controllers import LQG, LQGI  # noqa: E402
from control.closed_loop import run_closed_loop  # noqa: E402
from control.readouts import compute_pca_readout  # noqa: E402
from control.reachability import steady_state_gain, feasibility  # noqa: E402
from metrics import (  # noqa: E402
    settling_time, steady_state_error, control_effort,
    saturation_fraction, rms, band_occupancy_settling,
)
from viz.style import CONTROLLER_COLORS  # noqa: E402


# ----------------------------------------------------------------------------
# Brain / plant helpers
# ----------------------------------------------------------------------------
def brain_plant_factory(seed: int):
    from GG4 import Brain
    return BrainPlant(Brain(random_seed=int(seed)))


def _probe(plant_factory, seed: int, T_cal: int, m: int,
           probe_seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(probe_seed)
    U = rng.uniform(0.0, 1.0, size=(T_cal, m))
    plant = plant_factory(seed)
    Y = np.empty((T_cal, len(plant.measure())))
    plant = plant_factory(seed)
    Y[0] = np.asarray(plant.measure(), dtype=float)
    plant.next_state(U[0])
    for t in range(1, T_cal):
        Y[t] = np.asarray(plant.measure(), dtype=float)
        plant.next_state(U[t])
    return Y, U


# ----------------------------------------------------------------------------
# Model dictionary — a unified container for honest and yardstick models.
# ----------------------------------------------------------------------------
def model_from_iface(est: EstimatorNew, source: str = "honest") -> dict:
    return dict(A=est.A, B=est.B, C=est.C, Q=est.Q, R=est.R,
                a=est.a, c=est.c, source=source)


def model_from_yardstick(gt: GroundTruthModel) -> dict:
    return dict(A=gt.A, B=gt.B, C=gt.C, Q=gt.Q, R=gt.R,
                a=gt.a, c=gt.c, source="yardstick")


# ----------------------------------------------------------------------------
# Fair-comparison closed-loop runner — both controllers aim at the SAME
# physical readout and the SAME physical reference.
# ----------------------------------------------------------------------------
def build_controller(name: str, model: dict, M_readout: np.ndarray,
                       ref: np.ndarray, *, rho: float = 1.0,
                       rho_q: float = 0.01) -> object:
    A, B, C = model["A"], model["B"], model["C"]
    a, c = model["a"], model["c"]
    if name == "OpenLoop":
        from control.controllers import OpenLoop as _OL
        return _OL(input_dim=B.shape[1])
    if name == "LQG":
        return LQG(A, B, C, M_readout, a=a, c=c, rho=rho, ref=ref)
    if name == "LQGI":
        return LQGI(A, B, C, M_readout, a=a, c=c,
                    rho=rho, rho_q=rho_q, ref=ref, anti_windup=True)
    raise ValueError(f"unknown controller {name!r}")


def noise_floor_2d(M: np.ndarray, R_yardstick: np.ndarray) -> float:
    """Per-step readout noise scale = ``sqrt(trace(M R M^T))`` —
    the absolute lower bound for steady-state error on this readout."""
    return float(np.sqrt(np.trace(M @ R_yardstick @ M.T)))


def run_hold_on_brain(seed: int, model: dict, controller_name: str,
                        M_readout: np.ndarray, ref: np.ndarray,
                        T: int = 300) -> dict:
    plant = brain_plant_factory(seed)
    obs = SteadyStateKalman(model["A"], model["B"], model["C"],
                              model["Q"], model["R"],
                              a=model["a"], c=model["c"])
    ctrl = build_controller(controller_name, model, M_readout, ref)
    logs = run_closed_loop(plant, ctrl, obs, T=T, ref_fn=lambda t: ref)
    return logs


def evaluate_hold(logs: dict, M_readout: np.ndarray,
                    ref: np.ndarray, noise_floor: float | None = None) -> dict:
    z = logs["y"] @ M_readout.T
    ref_scale = float(np.linalg.norm(ref))
    ss = float(steady_state_error(z, ref, tail_frac=0.5))
    occ = band_occupancy_settling(z, ref, tol_frac=0.10,
                                    reference_scale=max(ref_scale, 1.0))
    out = dict(
        settling=int(settling_time(z, ref, tol_frac=0.10,
                                     reference_scale=max(ref_scale, 1.0))),
        steady_err=ss,
        effort=float(control_effort(logs["u"])),
        sat=float(saturation_fraction(logs["u"])),
        achieved=z[-30:].mean(axis=0).tolist(),
        band_occupancy=float(occ["band_occupancy"]),
    )
    if noise_floor is not None and noise_floor > 0:
        out["ss_err_x_noise"] = float(ss / noise_floor)
        out["noise_floor"] = float(noise_floor)
    return out


# ----------------------------------------------------------------------------
# Held-out prediction helper.
# ----------------------------------------------------------------------------
def held_out_prediction_rms(model: dict, Y_h: np.ndarray, U_h: np.ndarray
                             ) -> float:
    """One-step RMS of ``y_pred(t) = C x_pred(t) + c`` on (Y_h, U_h),
    using a Kalman filter run with the model's own (Q, R)."""
    from estimator.identify import _kalman_filter_affine
    A, B, C = model["A"], model["B"], model["C"]
    Q, R = model["Q"], model["R"]
    a, c = model["a"], model["c"]
    n = A.shape[0]
    try:
        x0 = np.linalg.solve(np.eye(n) - A, a)
    except np.linalg.LinAlgError:
        x0 = np.zeros(n)
    P0 = np.eye(n)
    _xf, _Pf, xp, _Pp, _ll, _Kl = _kalman_filter_affine(
        Y_h, U_h, A, B, C, Q, R, a, c, x0, P0
    )
    y_pred = (C @ xp.T).T + c
    resid = Y_h - y_pred
    return float(np.sqrt(np.mean(resid ** 2)))


# ----------------------------------------------------------------------------
# Common readout + target — computed once from a long Brain probe so all
# models are graded against the SAME physical setpoint.
# ----------------------------------------------------------------------------
def build_common_readout_and_target(
    seed: int = 0, T_probe: int = 2000, target_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(M_readout, ref, G_yardstick, z0_yardstick)``.

    ``M_readout`` is the 2-D PCA of a long probe of seed ``seed``;
    ``ref`` is at ``target_fraction`` of the yardstick zonotope diagonal
    (a target that lives clearly inside the yardstick's reachable region).
    """
    Y, _U = _probe(brain_plant_factory, seed=seed, T_cal=T_probe, m=2)
    M, _ = compute_pca_readout(Y, k=2)
    gt = GroundTruthModel.load(ROOT / "results" / "ground_truth" /
                                f"seed_{seed}.npz")
    G_y, z0_y = steady_state_gain(gt.A, gt.B, gt.C, M, a=gt.a, c=gt.c)
    # Reference at the convex midpoint of the input box → middle of the
    # parallelogram-shaped reachable zonotope.
    ref = G_y @ (target_fraction * np.ones(gt.B.shape[1])) + z0_y
    return M, ref, G_y, z0_y


# ----------------------------------------------------------------------------
# Sub-study A — Headline: honest vs yardstick on the same Brain task.
# ----------------------------------------------------------------------------
def study_a_headline(seeds: List[int], T_cal: int = 1000, T_run: int = 300,
                       log=print) -> dict:
    log("\n[A] Headline — honest vs yardstick, hold task on real Brain")
    M_ref, ref, G_y, z0_y = build_common_readout_and_target(seed=seeds[0])
    log(f"    common readout M from seed {seeds[0]} probe (T=2000)")
    log(f"    yardstick G = {G_y.round(3).tolist()}")
    log(f"    target ref  = {ref.round(3).tolist()} (yardstick zonotope midpoint)")
    rows = []
    for s in seeds:
        log(f"\n  [seed {s}]")
        # 1) Calibrate honest model from a single probe.
        Y_cal, U_cal = _probe(brain_plant_factory, seed=s, T_cal=T_cal,
                              m=2, probe_seed=42 + s)
        est = EstimatorNew().fit(Y_cal, U_cal, n=6, init="n4sid",
                                  max_iter=80, tol=1e-5)
        honest = model_from_iface(est, source="honest")
        # 2) Load yardstick.
        gt = GroundTruthModel.load(ROOT / "results" / "ground_truth" /
                                    f"seed_{s}.npz")
        yard = model_from_yardstick(gt)
        nf = noise_floor_2d(M_ref, gt.R)
        # 3) Held-out prediction for both (on the SAME held-out probe).
        Y_h, U_h = _probe(brain_plant_factory, seed=s, T_cal=400,
                          m=2, probe_seed=9999 + s)
        rms_h = held_out_prediction_rms(honest, Y_h, U_h)
        rms_y = held_out_prediction_rms(yard, Y_h, U_h)
        # 4) Closed-loop hold on real Brain (seed s): honest, yardstick, OL ref.
        logs_h = run_hold_on_brain(s, honest, "LQG", M_ref, ref, T=T_run)
        logs_y = run_hold_on_brain(s, yard, "LQG", M_ref, ref, T=T_run)
        logs_ol = run_hold_on_brain(s, honest, "OpenLoop", M_ref, ref, T=T_run)
        m_h = evaluate_hold(logs_h, M_ref, ref, noise_floor=nf)
        m_y = evaluate_hold(logs_y, M_ref, ref, noise_floor=nf)
        m_ol = evaluate_hold(logs_ol, M_ref, ref, noise_floor=nf)
        log(f"    honest   : pred_rms={rms_h:.3f}  ss×NF={m_h['ss_err_x_noise']:.2f}  "
            f"settle={m_h['settling']:>3}  effort={m_h['effort']:.1f}")
        log(f"    yardstick: pred_rms={rms_y:.3f}  ss×NF={m_y['ss_err_x_noise']:.2f}  "
            f"settle={m_y['settling']:>3}  effort={m_y['effort']:.1f}")
        log(f"    OpenLoop : (ref)                ss×NF={m_ol['ss_err_x_noise']:.2f}  "
            f"  (the gap to honest/yardstick is the closed-loop win)")
        rows.append(dict(seed=s, honest_pred=rms_h, yard_pred=rms_y,
                          honest=m_h, yard=m_y, openloop=m_ol,
                          noise_floor=nf,
                          logs_h=logs_h, logs_y=logs_y))
    return dict(rows=rows, M_ref=M_ref, ref=ref)


# ----------------------------------------------------------------------------
# Sub-study B — Data efficiency: control + prediction vs T_cal.
# ----------------------------------------------------------------------------
def study_b_data_efficiency(
    seeds: List[int],
    T_cals: List[int] = (100, 200, 500, 1000, 2000),
    T_run: int = 300,
    log=print,
) -> dict:
    log("\n[B] Data efficiency — control + prediction vs T_cal")
    M_ref, ref, G_y, z0_y = build_common_readout_and_target(seed=seeds[0])
    log(f"    same readout + target as study A")
    # One OL reference per seed (independent of T_cal — OL outputs u=0).
    ol_ref_per_seed: dict = {}
    nf_per_seed: dict = {}
    for s in seeds:
        gt = GroundTruthModel.load(ROOT / "results" / "ground_truth" /
                                    f"seed_{s}.npz")
        nf = noise_floor_2d(M_ref, gt.R)
        # Use a placeholder model — OL ignores it.
        ph = dict(A=gt.A, B=gt.B, C=gt.C, Q=gt.Q, R=gt.R, a=gt.a, c=gt.c)
        logs_ol = run_hold_on_brain(s, ph, "OpenLoop", M_ref, ref, T=T_run)
        m_ol = evaluate_hold(logs_ol, M_ref, ref, noise_floor=nf)
        ol_ref_per_seed[s] = m_ol["ss_err_x_noise"]
        nf_per_seed[s] = nf
        log(f"    seed {s} OL reference: ss×NF = {m_ol['ss_err_x_noise']:.2f}")
    rows = []
    for T_cal in T_cals:
        log(f"\n  [T_cal = {T_cal}]")
        for s in seeds:
            Y_cal, U_cal = _probe(brain_plant_factory, seed=s, T_cal=T_cal,
                                  m=2, probe_seed=42 + s)
            try:
                est = EstimatorNew().fit(Y_cal, U_cal, n=6, init="n4sid",
                                          max_iter=80, tol=1e-5)
            except Exception as exc:
                log(f"    seed {s}: fit FAILED ({type(exc).__name__}: {exc})")
                continue
            honest = model_from_iface(est, source="honest")
            # Prediction error on a fixed held-out probe.
            Y_h, U_h = _probe(brain_plant_factory, seed=s, T_cal=400,
                              m=2, probe_seed=9999 + s)
            pred_rms = held_out_prediction_rms(honest, Y_h, U_h)
            # Control hold — LQG (feedforward only) AND LQGI (+integral).
            nf = nf_per_seed[s]
            logs_lqg = run_hold_on_brain(s, honest, "LQG", M_ref, ref, T=T_run)
            logs_lqgi = run_hold_on_brain(s, honest, "LQGI", M_ref, ref, T=T_run)
            m_lqg = evaluate_hold(logs_lqg, M_ref, ref, noise_floor=nf)
            m_lqgi = evaluate_hold(logs_lqgi, M_ref, ref, noise_floor=nf)
            log(f"    seed {s}: pred={pred_rms:.3f}  "
                f"LQG  ss×NF={m_lqg['ss_err_x_noise']:.2f}  "
                f"LQGI ss×NF={m_lqgi['ss_err_x_noise']:.2f}")
            rows.append(dict(T_cal=T_cal, seed=s, pred_rms=pred_rms,
                              lqg=m_lqg, lqgi=m_lqgi,
                              fit_iters=est.fit_iters,
                              noise_floor=nf,
                              rho=float(np.max(np.abs(np.linalg.eigvals(est.A))))))
    return dict(rows=rows, T_cals=list(T_cals), M_ref=M_ref, ref=ref,
                 ol_ref_per_seed=ol_ref_per_seed, nf_per_seed=nf_per_seed)


# ----------------------------------------------------------------------------
# Sub-study C — Local minima: EM from different inits.
# ----------------------------------------------------------------------------
def study_c_local_minima(
    seeds: List[int], T_cal: int = 1000, T_run: int = 300,
    inits: List[str] = ("n4sid", "output_ssi", "random"),
    log=print,
) -> dict:
    log("\n[C] Local minima — EM from {n4sid, output_ssi, random} inits")
    M_ref, ref, G_y, z0_y = build_common_readout_and_target(seed=seeds[0])
    rows = []
    ol_ref_per_seed: dict = {}
    nf_per_seed: dict = {}
    for s in seeds:
        log(f"\n  [seed {s}]")
        Y_cal, U_cal = _probe(brain_plant_factory, seed=s, T_cal=T_cal,
                              m=2, probe_seed=42 + s)
        Y_h, U_h = _probe(brain_plant_factory, seed=s, T_cal=400,
                          m=2, probe_seed=9999 + s)
        gt = GroundTruthModel.load(ROOT / "results" / "ground_truth" /
                                    f"seed_{s}.npz")
        nf = noise_floor_2d(M_ref, gt.R)
        nf_per_seed[s] = nf
        # One OL reference per seed for this panel.
        ph = dict(A=gt.A, B=gt.B, C=gt.C, Q=gt.Q, R=gt.R, a=gt.a, c=gt.c)
        logs_ol = run_hold_on_brain(s, ph, "OpenLoop", M_ref, ref, T=T_run)
        m_ol = evaluate_hold(logs_ol, M_ref, ref, noise_floor=nf)
        ol_ref_per_seed[s] = m_ol["ss_err_x_noise"]
        log(f"    OL reference ss×NF = {m_ol['ss_err_x_noise']:.2f}")
        for init in inits:
            try:
                est = EstimatorNew().fit(Y_cal, U_cal, n=6, init=init,
                                          max_iter=80, tol=1e-5)
            except Exception as exc:
                log(f"    init={init}: fit FAILED ({type(exc).__name__})")
                continue
            ll = est.loglik_history()
            honest = model_from_iface(est)
            pred_rms = held_out_prediction_rms(honest, Y_h, U_h)
            logs = run_hold_on_brain(s, honest, "LQG", M_ref, ref, T=T_run)
            m = evaluate_hold(logs, M_ref, ref, noise_floor=nf)
            log(f"    init={init:>11}: iters={est.fit_iters:>3}  "
                f"ll_final={ll[-1]:.1f}  pred={pred_rms:.3f}  "
                f"ss×NF={m['ss_err_x_noise']:.2f}")
            rows.append(dict(seed=s, init=init, fit_iters=est.fit_iters,
                              ll_final=float(ll[-1]),
                              pred_rms=pred_rms, hold=m,
                              noise_floor=nf,
                              rho=float(np.max(np.abs(np.linalg.eigvals(est.A))))))
    return dict(rows=rows, inits=list(inits), M_ref=M_ref, ref=ref,
                 ol_ref_per_seed=ol_ref_per_seed, nf_per_seed=nf_per_seed)


# ----------------------------------------------------------------------------
# Figure — 2x2: headline / T_cal sweep / integral overlay / local minima.
# ----------------------------------------------------------------------------
COLORS = {
    "honest": "#d95f02",   # warm orange
    "yardstick": "#1b9e77", # green
    "LQG":     "#1f77b4",  # blue
    "LQGI":    "#9467bd",  # purple
    "n4sid":   "#1b9e77",
    "output_ssi": "#d95f02",
    "random":  "#7570b3",
}


def plot_m3(a: dict, b: dict, c: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # ---- Panel A: headline — honest vs yardstick, anchored ---------------
    ax = axes[0, 0]
    seeds = [r["seed"] for r in a["rows"]]
    x = np.arange(len(seeds))
    w = 0.27
    ss_ol = [r["openloop"]["ss_err_x_noise"] for r in a["rows"]]
    ss_h = [r["honest"]["ss_err_x_noise"] for r in a["rows"]]
    ss_y = [r["yard"]["ss_err_x_noise"] for r in a["rows"]]
    ax.bar(x - w, ss_ol, w, label="OpenLoop (no control)",
           color=CONTROLLER_COLORS["OpenLoop"])
    ax.bar(x,     ss_h, w, label="honest model LQG",
           color=COLORS["honest"])
    ax.bar(x + w, ss_y, w, label="yardstick model LQG",
           color=COLORS["yardstick"])
    ax.axhline(1.0, color="black", ls=":", lw=1.1,
                label="noise floor (best achievable)")
    ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("steady-state error\n(× per-readout noise floor)")
    ax.set_title("A. Honest vs yardstick model — closed-loop hold on real Brain\n"
                 "× noise floor; 1.0 = best achievable; OL ≈ 11× shown for reference")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    for i, r in enumerate(a["rows"]):
        ratio = r["honest_pred"] / max(r["yard_pred"], 1e-12)
        ax.text(i, max(ss_h[i], ss_y[i]) * 1.6,
                f"pred(h/y) = {ratio:.2f}", ha="center", fontsize=7,
                color="0.35")

    # ---- Panel B: T_cal sweep — control metric, anchored, single axis ----
    ax = axes[0, 1]
    T_cals = b["T_cals"]
    by_tcal = {tc: [r for r in b["rows"] if r["T_cal"] == tc] for tc in T_cals}
    mean_ss_lqg = [np.mean([r["lqg"]["ss_err_x_noise"] for r in by_tcal[tc]])
                    for tc in T_cals]
    mean_ss_lqgi = [np.mean([r["lqgi"]["ss_err_x_noise"] for r in by_tcal[tc]])
                     for tc in T_cals]
    ax.plot(T_cals, mean_ss_lqg, "o-", color=CONTROLLER_COLORS["LQG"],
             lw=2, label="LQG (mean)")
    ax.plot(T_cals, mean_ss_lqgi, "s-", color=CONTROLLER_COLORS["LQGI"],
             lw=2, label="LQGI (mean, +integral)")
    for tc in T_cals:
        for r in by_tcal[tc]:
            ax.plot([tc], [r["lqg"]["ss_err_x_noise"]], ".",
                    color=CONTROLLER_COLORS["LQG"], alpha=0.4, ms=6)
            ax.plot([tc], [r["lqgi"]["ss_err_x_noise"]], ".",
                    color=CONTROLLER_COLORS["LQGI"], alpha=0.4, ms=6)
    # OpenLoop reference (seed-mean) as a dashed horizontal line.
    ol_mean = float(np.mean(list(b["ol_ref_per_seed"].values())))
    ax.axhline(ol_mean, color=CONTROLLER_COLORS["OpenLoop"], ls="--", lw=1.2,
                label=f"OpenLoop (no control) ≈ {ol_mean:.1f}×")
    ax.axhline(1.0, color="black", ls=":", lw=1.1,
                label="noise floor (best achievable)")
    ax.set_xscale("log")
    ax.set_xlabel("T_cal (probe length, log scale)")
    ax.set_ylabel("steady-state error\n(× per-readout noise floor)")
    pred_mean = [np.mean([r["pred_rms"] for r in by_tcal[tc]]) for tc in T_cals]
    pred_str = ", ".join(f"T={tc}: {p:.2f}" for tc, p in zip(T_cals, pred_mean))
    ax.set_title(
        "B. Data efficiency — control metric is primary "
        "(integral overlay)\n"
        f"× noise floor; held-out pred RMS by T_cal: {pred_str}",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    # ---- Panel C: band-occupancy vs T_cal (replaces saturated settling) ---
    ax = axes[1, 0]
    occ_lqg = [np.mean([r["lqg"]["band_occupancy"] for r in by_tcal[tc]])
                for tc in T_cals]
    occ_lqgi = [np.mean([r["lqgi"]["band_occupancy"] for r in by_tcal[tc]])
                 for tc in T_cals]
    ax.plot(T_cals, occ_lqg, "o-", color=CONTROLLER_COLORS["LQG"],
             lw=2, label="LQG")
    ax.plot(T_cals, occ_lqgi, "s-", color=CONTROLLER_COLORS["LQGI"],
             lw=2, label="LQGI")
    for tc in T_cals:
        for r in by_tcal[tc]:
            ax.plot([tc], [r["lqg"]["band_occupancy"]], ".",
                     color=CONTROLLER_COLORS["LQG"], alpha=0.4, ms=6)
            ax.plot([tc], [r["lqgi"]["band_occupancy"]], ".",
                     color=CONTROLLER_COLORS["LQGI"], alpha=0.4, ms=6)
    ax.set_xscale("log")
    ax.set_xlabel("T_cal (log scale)")
    ax.set_ylabel("band occupancy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("C. Band occupancy vs T_cal\n"
                 "(fraction of final 50% inside ±10% band; "
                 "M4-style metric; replaces the saturated settling-time)",
                 fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    # ---- Panel D: Local minima — anchored ss × NF + OL ref ---------------
    ax = axes[1, 1]
    inits = c["inits"]
    seeds_c = sorted(set(r["seed"] for r in c["rows"]))
    x = np.arange(len(seeds_c))
    w = 0.8 / len(inits)
    for i, init in enumerate(inits):
        ss = []; pr = []
        for s in seeds_c:
            cand = [r for r in c["rows"] if r["seed"] == s and r["init"] == init]
            ss.append(cand[0]["hold"]["ss_err_x_noise"] if cand else np.nan)
            pr.append(cand[0]["pred_rms"] if cand else np.nan)
        ax.bar(x + (i - (len(inits) - 1) / 2) * w, ss, w,
                label=init, color=COLORS.get(init, f"C{i}"))
        for xi, val, p in zip(x, ss, pr):
            if np.isnan(val) or np.isnan(p):
                continue
            ax.text(xi + (i - (len(inits) - 1) / 2) * w, val + 0.15,
                     f"pred {p:.2f}", ha="center", fontsize=7, color="0.3")
    # Per-seed OL reference line at the mean OL ss × NF.
    ol_mean_c = float(np.mean(list(c["ol_ref_per_seed"].values())))
    ax.axhline(ol_mean_c, color=CONTROLLER_COLORS["OpenLoop"], ls="--", lw=1.2,
                label=f"OpenLoop ≈ {ol_mean_c:.1f}×")
    ax.axhline(1.0, color="black", ls=":", lw=1.1,
                label="noise floor (1.0 = best)")
    ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in seeds_c])
    ax.set_ylabel("steady-state error\n(× per-readout noise floor)")
    ax.set_title("D. Local minima — bad init breaks control "
                 "even with good prediction\n"
                 "× noise floor; the seed-1 × output_ssi spike is the headline finding",
                 fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("M3 — Is good prediction enough for good control on a "
                  "sloppy, weakly-identifiable system?",
                  fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(
    seeds: List[int] = (0, 1, 2),
    T_cals: List[int] = (100, 200, 500, 1000, 2000),
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
    log("  M3 — Benchmark study: control performance as the headline")
    log("=" * 78)

    t0 = time.time()
    a = study_a_headline(list(seeds), T_cal=1000, T_run=T_run, log=log)
    b = study_b_data_efficiency(list(seeds), T_cals=list(T_cals), T_run=T_run,
                                 log=log)
    c = study_c_local_minima(list(seeds), T_cal=1000, T_run=T_run, log=log)
    log(f"\n[done]  total wall-clock = {time.time() - t0:.1f}s")

    fig_path = out_dir / "m3_benchmark_study.png"
    plot_m3(a, b, c, fig_path)
    log(f"figure: {fig_path}")

    text_path = out_dir / "m3_benchmark_study.txt"
    text_path.write_text("\n".join(log_lines), encoding="utf-8")
    log(f"log:    {text_path}")

    # Save raw arrays for reuse.
    npz_path = out_dir / "m3_benchmark_study.npz"
    np.savez(
        npz_path,
        seeds=np.asarray(list(seeds)),
        T_cals=np.asarray(list(T_cals)),
        ref=a["ref"],
    )

    return dict(a=a, b=b, c=c, figure=fig_path, text=text_path, npz=npz_path)


if __name__ == "__main__":
    main()
