"""M5: three-task exploration (hold / track / suppress) — same controller family,
three references (spec v3 §5).

For each controller (5 existing + 3 integral variants) we run on:

    * **hold (feasible)**      ref constant inside the holdable (PC1, PC2) region
    * **hold (infeasible)**    ref constant outside the holdable region
    * **track (slow sine)**    ref varies on a long period (≫ closed-loop time constant)
    * **track (fast sine)**    ref varies on a period near closed-loop bandwidth
    * **track (staircase)**    piecewise-constant step changes inside the region
    * **track (too aggressive)** amplitude leaves region OR period ≪ bandwidth
    * **suppress**             ref = z0 (open-loop resting readout)

Errors are reported **normalised** — as a fraction of the target's magnitude
*and* as a fraction of the open-loop readout noise floor — never bare absolutes
(spec v3 §6 clarity rules).

Primary scenario: ``default_neural_system`` on the Simulator (Stage A — true
params — to isolate controller behaviour from identification quality; Stage B
parity is the E3 figure). Brain hold-only sweep is appended at the end.

Run::

    python week3/experiments/m5_three_tasks.py

Stage A runs print a single table per task plus a per-task "why" summary.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

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
from control.readouts import compute_pca_readout  # noqa: E402
from control.reachability import steady_state_gain, feasibility, zonotope_vertices  # noqa: E402
from metrics import rms, control_effort, saturation_fraction, settling_time  # noqa: E402


# ----------------------------------------------------------------------------
# Controller factory — built from (A, B, C, M, a, c, ref). All controllers
# in §5 of the spec are constructed identically except for their internal
# integral/disturbance state, which they manage themselves.
# ----------------------------------------------------------------------------

CONTROLLER_NAMES = [
    "OpenLoop", "PropFeedback", "LQG", "PolePlace", "MPC",  # existing 5
    "PI", "LQGI", "OffsetFreeMPC",                          # E1 additions
]


def build_controller(name: str, A, B, C, M, a, c, ref, *,
                      rho: float = 1.0, rho_q: float = 0.001,
                      H: int = 20, dist_gain: float = 0.1):
    # rho_q sets how aggressively the LQR pushes the integral state — too
    # large and K_x blows up alongside K_q (LQR over-uses the actuator to
    # crush q), producing chattering on targets whose PC magnitude is large.
    # 0.001 keeps K_x close to the standard LQR's and lets the integral close
    # the offset gently; the synthetic test_integral still passes because the
    # bias there is small (and the test uses its own rho_q=10 anyway).
    m = B.shape[1]
    q = M.shape[0]
    if name == "OpenLoop":
        return OpenLoop(input_dim=m)
    if name == "PropFeedback":
        return ProportionalFeedback(
            Kp=0.05 * np.eye(m, q), M=M, ref=ref,
        )
    if name == "LQG":
        return LQG(A, B, C, M, a=a, c=c, rho=rho, ref=ref)
    if name == "PolePlace":
        return PolePlacement(A, B, C, M,
                              target_poles=np.linspace(0.5, 0.85, A.shape[0]),
                              a=a, c=c, ref=ref)
    if name == "MPC":
        return MPC(A, B, C, M, a=a, c=c, horizon=H, rho=rho, ref=ref)
    if name == "PI":
        # Match ProportionalFeedback's gain; add modest Ki.
        return PI(Kp=0.05 * np.eye(m, q), Ki=0.02 * np.eye(m, q),
                  M=M, ref=ref, anti_windup=True)
    if name == "LQGI":
        return LQGI(A, B, C, M, a=a, c=c, rho=rho, rho_q=rho_q,
                    ref=ref, anti_windup=True)
    if name == "OffsetFreeMPC":
        return OffsetFreeMPC(A, B, C, M, a=a, c=c,
                              horizon=H, rho=rho, ref=ref,
                              dist_gain=dist_gain, anti_windup=True)
    raise ValueError(f"unknown controller {name}")


# ----------------------------------------------------------------------------
# Reference factories — ``ref_fn(t)`` is called once per step by the driver.
# ----------------------------------------------------------------------------

def ref_constant(value: np.ndarray) -> Callable[[int], np.ndarray]:
    v = np.asarray(value, dtype=float)
    return lambda t: v


def ref_sine(midpoint: np.ndarray, amp: np.ndarray, period: float
             ) -> Callable[[int], np.ndarray]:
    midpoint = np.asarray(midpoint, dtype=float)
    amp = np.asarray(amp, dtype=float)
    omega = 2.0 * np.pi / period
    def f(t):
        return midpoint + amp * np.sin(omega * t)
    return f


def ref_staircase(midpoint: np.ndarray, steps: np.ndarray, dwell: int
                   ) -> Callable[[int], np.ndarray]:
    midpoint = np.asarray(midpoint, dtype=float)
    steps = np.asarray(steps, dtype=float)
    def f(t):
        idx = (t // dwell) % steps.shape[0]
        return midpoint + steps[idx]
    return f


# ----------------------------------------------------------------------------
# Per-episode evaluation: collapse a closed-loop run into a row of metrics.
# All errors are reported normalised: fraction of target magnitude and
# fraction of the open-loop readout noise floor (spec §6).
# ----------------------------------------------------------------------------

def evaluate_episode(logs: dict, M: np.ndarray, ref_array: np.ndarray,
                      noise_floor: float, target_scale: float) -> dict:
    z = logs["y"] @ M.T
    err = z - ref_array
    rms_err = rms(err)
    return dict(
        rms_err=rms_err,
        rms_err_frac_target=rms_err / max(target_scale, 1e-9),
        rms_err_frac_noise=rms_err / max(noise_floor, 1e-9),
        effort=control_effort(logs["u"]),
        sat=saturation_fraction(logs["u"]),
        settling=settling_time(z, ref_array.mean(axis=0))
                 if ref_array.ndim == 2 else settling_time(z, ref_array),
        z_final=z[-30:].mean(axis=0).tolist(),
    )


def print_task_table(task_name: str, target_descr: str, target_scale: float,
                      noise_floor: float, rows: Dict[str, dict]) -> None:
    print(f"\n  -- Task: {task_name} --")
    print(f"     target: {target_descr}")
    print(f"     normalisers: target-magnitude = {target_scale:.3f}, "
          f"noise-floor (OL readout std) = {noise_floor:.3f}")
    print(f"     {'controller':14s}  {'rms_err':>9s}  {'frac.tgt':>9s}  "
          f"{'frac.noise':>10s}  {'effort':>7s}  {'sat':>5s}  {'settle':>7s}")
    for cname, m in rows.items():
        print(f"     {cname:14s}  "
              f"{m['rms_err']:9.3f}  "
              f"{m['rms_err_frac_target']:9.3f}  "
              f"{m['rms_err_frac_noise']:10.3f}  "
              f"{m['effort']:7.2f}  "
              f"{m['sat']:5.2f}  "
              f"{m['settling']:7d}")


# ----------------------------------------------------------------------------
# Main Simulator sweep.
# ----------------------------------------------------------------------------

def run_simulator_sweep(seed_plant: int = 7, T: int = 250,
                          T_probe: int = 600, probe_seed: int = 42) -> Dict:
    system = sim.default_neural_system(seed=0, obs_dim=16)
    n = system.A.shape[0]
    p = system.obs_dim
    m_in = system.input_dim
    a0 = np.zeros(n)
    c0 = np.zeros(p)

    # ---------- PCA readout from a short probe -----------------------------
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
    print("=" * 84)
    print("M5 — three-task exploration on default_neural_system (Stage A)")
    print("=" * 84)
    print(f"  Readout = 2 leading PCs of probe Y.  "
          f"explained var (PC1, PC2) = ({explained[0]:.3f}, {explained[1]:.3f})")

    # ---------- Reachable region & target choice ---------------------------
    G_true, z0_true = steady_state_gain(system.A, system.B, system.C, M,
                                          a=a0, c=c0)
    verts = zonotope_vertices(G_true, z0_true)  # 4 corners
    cond_G = float(np.linalg.cond(G_true))
    print(f"  Holdable region: 4 corners at u ∈ {{0,1}}^2, "
          f"cond(G) = {cond_G:.2f}")
    print(f"    z(u=0,0) = {verts[0].round(3)}")
    print(f"    z(u=1,0) = {verts[1].round(3)}")
    print(f"    z(u=0,1) = {verts[2].round(3)}")
    print(f"    z(u=1,1) = {verts[3].round(3)}")

    # Feasible centre (zonotope midpoint).
    ref_feas = G_true @ np.array([0.5, 0.5]) + z0_true
    inside_feas, _ = feasibility(ref_feas, G_true, z0_true)
    # Infeasible: pull 2× past the (1,1) corner along the diagonal.
    far = verts[3] + (verts[3] - verts[0])
    ref_infeas = far
    inside_infeas, _ = feasibility(ref_infeas, G_true, z0_true)
    print(f"  ref_feasible   = {ref_feas.round(3)}   inside? {inside_feas}")
    print(f"  ref_infeasible = {ref_infeas.round(3)}  inside? {inside_infeas}")

    # ---------- Noise floor on the readout under OpenLoop ------------------
    plant_OL = SimulatorPlant(system, seed=seed_plant)
    obs_OL = SteadyStateKalman(system.A, system.B, system.C, system.Q, system.R,
                                a=a0, c=c0)
    logs_OL = run_closed_loop(plant_OL, OpenLoop(input_dim=m_in), obs_OL, T=T)
    z_OL = logs_OL["y"] @ M.T
    noise_floor = float(np.std(z_OL[-100:], axis=0).mean())
    OL_readout_drift_rms = rms(z_OL - z_OL.mean(axis=0))
    print(f"\n  open-loop readout noise floor (std, last 100 samples) = "
          f"{noise_floor:.3f}")
    print(f"  open-loop readout demeaned-RMS                          = "
          f"{OL_readout_drift_rms:.3f}")

    # ---------- Task helper -----------------------------------------------
    def run_task(ref_fn, ref_array, target_scale, controllers, name: str):
        rows = {}
        for cname in controllers:
            plant = SimulatorPlant(system, seed=seed_plant)
            obs = SteadyStateKalman(system.A, system.B, system.C,
                                     system.Q, system.R, a=a0, c=c0)
            ctrl = build_controller(cname, system.A, system.B, system.C,
                                      M, a0, c0,
                                      ref_array.mean(axis=0)
                                      if ref_array.ndim == 2 else ref_array)
            logs = run_closed_loop(plant, ctrl, obs, T=T, ref_fn=ref_fn)
            rows[cname] = evaluate_episode(logs, M, ref_array,
                                            noise_floor, target_scale)
        return rows

    # ====================================================================
    # Task 1: HOLD a feasible target
    # ====================================================================
    ref_array_h = np.tile(ref_feas, (T, 1))
    target_scale_h = float(np.linalg.norm(ref_feas - z0_true))
    rows_hold = run_task(ref_constant(ref_feas), ref_array_h,
                          target_scale_h,
                          CONTROLLER_NAMES, "hold_feasible")
    print_task_table("hold (feasible)",
                      f"constant ref = {ref_feas.round(3)} "
                      f"(zonotope midpoint, inside)",
                      target_scale_h, noise_floor, rows_hold)

    # ====================================================================
    # Task 2: HOLD an infeasible target
    # ====================================================================
    ref_array_inf = np.tile(ref_infeas, (T, 1))
    target_scale_inf = float(np.linalg.norm(ref_infeas - z0_true))
    rows_inf = run_task(ref_constant(ref_infeas), ref_array_inf,
                          target_scale_inf,
                          CONTROLLER_NAMES, "hold_infeasible")
    print_task_table("hold (infeasible)",
                      f"constant ref = {ref_infeas.round(3)} (outside)",
                      target_scale_inf, noise_floor, rows_inf)

    # ====================================================================
    # Task 3-6: TRACK various references
    # ====================================================================
    # Reference amplitude = 0.3 × half-diagonal of the zonotope.
    half_diag = 0.5 * np.linalg.norm(verts[3] - verts[0])
    amp = 0.3 * np.array([
        abs(verts[3, 0] - verts[0, 0]) / 2,
        abs(verts[3, 1] - verts[0, 1]) / 2,
    ])

    # 3a: slow sine — period ~ 4 × closed-loop time-constant (~20 steps)
    period_slow = 100.0
    ref_fn = ref_sine(ref_feas, amp, period_slow)
    ref_array = np.array([ref_fn(t) for t in range(T)])
    target_scale = float(np.linalg.norm(amp))
    rows_slow = run_task(ref_fn, ref_array, target_scale,
                          CONTROLLER_NAMES, "track_slow_sine")
    print_task_table("track (slow sine)",
                      f"midpoint + {amp.round(3)} * sin(2π t / {period_slow:.0f})",
                      target_scale, noise_floor, rows_slow)

    # 3b: fast sine — period ~ closed-loop time-constant
    period_fast = 20.0
    ref_fn = ref_sine(ref_feas, amp, period_fast)
    ref_array = np.array([ref_fn(t) for t in range(T)])
    rows_fast = run_task(ref_fn, ref_array, target_scale,
                          CONTROLLER_NAMES, "track_fast_sine")
    print_task_table("track (fast sine)",
                      f"midpoint + {amp.round(3)} * sin(2π t / {period_fast:.0f})",
                      target_scale, noise_floor, rows_fast)

    # 3c: staircase — three steps inside the region
    step_levels = np.array([
        -0.5 * amp,
        +0.0 * amp,
        +0.5 * amp,
    ])
    dwell = 60
    ref_fn = ref_staircase(ref_feas, step_levels, dwell)
    ref_array = np.array([ref_fn(t) for t in range(T)])
    rows_stair = run_task(ref_fn, ref_array, target_scale,
                           CONTROLLER_NAMES, "track_staircase")
    print_task_table("track (staircase)",
                      f"midpoint ± {(0.5*amp).round(3)} per {dwell}-step dwell",
                      target_scale, noise_floor, rows_stair)

    # 3d: deliberately too-aggressive — period 5 steps (≪ bandwidth)
    period_aggr = 5.0
    ref_fn = ref_sine(ref_feas, amp, period_aggr)
    ref_array = np.array([ref_fn(t) for t in range(T)])
    rows_aggr = run_task(ref_fn, ref_array, target_scale,
                          CONTROLLER_NAMES, "track_too_aggressive")
    print_task_table("track (too aggressive — period ≪ bandwidth)",
                      f"midpoint + {amp.round(3)} * sin(2π t / {period_aggr:.0f})  "
                      f"— faster than any controller can keep up",
                      target_scale, noise_floor, rows_aggr)

    # ====================================================================
    # Task 7: SUPPRESS — drive readout to its resting value (z0)
    # ====================================================================
    ref_array_s = np.tile(z0_true, (T, 1))
    # Suppression's "target" is the natural resting state; report normalised
    # against the OL readout drift RMS instead of a target magnitude.
    rows_supp = run_task(ref_constant(z0_true), ref_array_s,
                          OL_readout_drift_rms,
                          CONTROLLER_NAMES, "suppress")
    print_task_table("suppress (drive variance to zero)",
                      f"ref = z0 = {z0_true.round(3)}  (resting readout)",
                      OL_readout_drift_rms, noise_floor, rows_supp)
    print(f"     note: 'target' here is the OL readout drift RMS = {OL_readout_drift_rms:.3f};\n"
          f"           an ideal controller would drive that to zero. The 'noise-floor'\n"
          f"           column is the per-step jitter, which is the floor of what's possible.")

    return dict(
        M=M, G=G_true, z0=z0_true, verts=verts,
        ref_feas=ref_feas, ref_infeas=ref_infeas,
        noise_floor=noise_floor, OL_drift=OL_readout_drift_rms,
        results=dict(
            hold_feasible=rows_hold,
            hold_infeasible=rows_inf,
            track_slow_sine=rows_slow,
            track_fast_sine=rows_fast,
            track_staircase=rows_stair,
            track_too_aggressive=rows_aggr,
            suppress=rows_supp,
        ),
    )


# ----------------------------------------------------------------------------
# Brain hold-only sweep — Stage B (fitted params via IdentifiedSystem).
# ----------------------------------------------------------------------------

def run_brain_hold(n_seeds: int = 3, T_cal: int = 600, T: int = 200,
                    n_latent: int = 4) -> List[Dict]:
    try:
        from GG4 import Brain  # noqa: F401
    except ImportError:
        print("\n  [Brain] GG4 not importable in this env — skipping Brain hold sweep.")
        print("          Run this script in an environment with GG4 installed.")
        return []
    from control.plant import BrainPlant
    from GG4 import Brain  # type: ignore

    print("\n" + "=" * 84)
    print(f"M5 — Brain hold-only sweep ({n_seeds} seeds, Stage B)")
    print("=" * 84)
    rows = []
    for seed in range(n_seeds):
        brain = Brain(random_seed=seed)
        plant_cal = BrainPlant(brain)
        iface = IdentifiedSystem().calibrate(
            plant_cal, T_cal=T_cal, n=n_latent, m=plant_cal.input_dim, seed=42,
        )
        A, B, C, Q, R, a, c = iface.params()
        M = iface.readout_M
        G_fit, z0_fit = steady_state_gain(A, B, C, M, a=a, c=c)
        ref_feas = G_fit @ np.array([0.5, 0.5]) + z0_fit
        target_scale = float(np.linalg.norm(ref_feas - z0_fit))
        # Noise floor under OpenLoop on a fresh brain.
        brain_OL = Brain(random_seed=seed)
        plant_OL = BrainPlant(brain_OL)
        obs_OL = SteadyStateKalman(A, B, C, Q, R, a=a, c=c)
        logs_OL = run_closed_loop(plant_OL, OpenLoop(input_dim=2), obs_OL, T=T)
        z_OL = logs_OL["y"] @ M.T
        noise_floor = float(np.std(z_OL[-100:], axis=0).mean())

        print(f"\n  seed {seed}: cond(G_fit)={iface.readout_cond_G:.2f}  "
              f"ref={ref_feas.round(3)}  noise_floor={noise_floor:.3f}")

        seed_rows = {}
        for cname in ["OpenLoop", "LQG", "LQGI", "MPC", "OffsetFreeMPC"]:
            brain_run = Brain(random_seed=seed)
            plant = BrainPlant(brain_run)
            obs = SteadyStateKalman(A, B, C, Q, R, a=a, c=c)
            ctrl = build_controller(cname, A, B, C, M, a, c, ref_feas)
            logs = run_closed_loop(plant, ctrl, obs, T=T,
                                     ref_fn=lambda t: ref_feas)
            seed_rows[cname] = evaluate_episode(
                logs, M, np.tile(ref_feas, (T, 1)),
                noise_floor, target_scale,
            )
        print(f"     {'controller':14s}  {'rms_err':>9s}  {'frac.tgt':>9s}  "
              f"{'frac.noise':>10s}  {'effort':>7s}  {'sat':>5s}")
        for cname, m in seed_rows.items():
            print(f"     {cname:14s}  "
                  f"{m['rms_err']:9.3f}  "
                  f"{m['rms_err_frac_target']:9.3f}  "
                  f"{m['rms_err_frac_noise']:10.3f}  "
                  f"{m['effort']:7.2f}  "
                  f"{m['sat']:5.2f}")
        rows.append(dict(seed=seed, ref=ref_feas, noise_floor=noise_floor,
                          target_scale=target_scale, results=seed_rows))
    return rows


def main():
    sim_results = run_simulator_sweep()
    brain_results = run_brain_hold(n_seeds=3)
    return sim_results, brain_results


if __name__ == "__main__":
    main()
