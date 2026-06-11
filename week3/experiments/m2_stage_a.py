"""M2 experiment: Stage A — control math verified on TRUE params.

Per spec §10 M2 acceptance:
  - Control-math unit tests pass.
  - Stage-A suppression beats no-control (or matches it when the natural
    equilibrium is already at the target — a known one-sided-actuator limit).
  - Feasible setpoint reached by LQG / MPC.
  - MPC obeys 0 ≤ u ≤ 1.

This script runs all 5 controllers on every white-box Simulator scenario
(true a = c = 0) for two objectives:
    1. Suppression (ref = 0).
    2. Feasible setpoint at the zonotope midpoint  G·[0.5, 0.5] + z0.
Reports z-RMS, control effort, saturation, and closed-loop spectral radius.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # week3/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "week 1"))

import Simulator as sim  # noqa: E402

from control.plant import SimulatorPlant  # noqa: E402
from control.observer import SteadyStateKalman  # noqa: E402
from control.controllers import OpenLoop, ProportionalFeedback, LQG, MPC, PolePlacement  # noqa: E402
from control.closed_loop import run_closed_loop  # noqa: E402
from control.reachability import steady_state_gain, feasibility  # noqa: E402
from metrics import rms, control_effort, saturation_fraction, spectral_radius  # noqa: E402


SCENARIOS = [
    ("default", sim.default_neural_system, 4, 16),
    ("input_aligned", sim.input_aligned_system, 4, 16),
    ("input_blind", sim.input_blind_system, 4, 16),
    ("slow_drift", sim.slow_drift_system, 4, 16),
    ("closed_loop_sys", sim.closed_loop_system, 4, 16),
    ("hidden_input", sim.hidden_input_system, 5, 10),
]


# Readout choice: 2 leading principal components of the scenario's measurements
# under a short open-loop probe (spec §3). Each PC is one weighted combination
# of the p measurements, capturing a dominant pattern of population activity.
from control.readouts import compute_pca_readout  # noqa: E402


def scenario_pca_M(system, T_probe: int = 600, probe_seed: int = 42,
                    plant_seed: int = 43) -> np.ndarray:
    """Compute the 2-leading-PC readout matrix from a short open-loop probe."""
    plant = SimulatorPlant(system, seed=plant_seed)
    rng = np.random.default_rng(probe_seed)
    U = rng.uniform(0, 1, size=(T_probe, system.input_dim))
    Y = np.empty((T_probe, system.obs_dim))
    Y[0] = np.asarray(plant.measure(), dtype=float)
    plant.next_state(U[0])
    for t in range(1, T_probe):
        Y[t] = plant.measure()
        plant.next_state(U[t])
    M, _ = compute_pca_readout(Y, k=2)
    return M


def run_one(system, controller_factory, T, seed, M, a, c, ref=None):
    """Single closed-loop run. controller_factory(A, B, C, M, a, c) -> controller."""
    plant = SimulatorPlant(system, seed=seed)
    obs = SteadyStateKalman(system.A, system.B, system.C, system.Q, system.R, a=a, c=c)
    controller = controller_factory(system.A, system.B, system.C, M, a, c, ref)
    logs = run_closed_loop(plant, controller, obs, T=T,
                            ref_fn=(None if ref is None else (lambda t: ref)))
    return logs


def _factory(name, ref):
    if name == "open":
        return lambda A, B, C, M, a, c, ref: OpenLoop(input_dim=B.shape[1])
    if name == "PropFB":
        return lambda A, B, C, M, a, c, ref: ProportionalFeedback(
            Kp=0.05 * np.eye(B.shape[1], M.shape[0]), M=M, ref=(ref if ref is not None else np.zeros(M.shape[0]))
        )
    if name == "LQG":
        return lambda A, B, C, M, a, c, ref: LQG(A, B, C, M, a=a, c=c, rho=0.1,
                                                  ref=(ref if ref is not None else np.zeros(M.shape[0])))
    if name == "PolePlace":
        return lambda A, B, C, M, a, c, ref: PolePlacement(
            A, B, C, M, target_poles=np.linspace(0.5, 0.85, A.shape[0]),
            a=a, c=c, ref=(ref if ref is not None else np.zeros(M.shape[0]))
        )
    if name == "MPC":
        return lambda A, B, C, M, a, c, ref: MPC(A, B, C, M, a=a, c=c, horizon=20, rho=0.1,
                                                   ref=(ref if ref is not None else np.zeros(M.shape[0])))
    raise ValueError(name)


def closed_loop_rho(A, B, K=None):
    if K is None:
        return spectral_radius(A)
    return spectral_radius(A - B @ K)


def main():
    seed = 7
    T = 200
    print("=" * 84)
    print("M2 — Stage A (TRUE params) — all scenarios, all controllers")
    print("=" * 84)

    sup_rows = []
    set_rows = []
    inf_set_rows = []
    for name, factory_fn, n_state, p in SCENARIOS:
        system = factory_fn(seed=0, obs_dim=p)
        M = scenario_pca_M(system)
        n = system.A.shape[0]
        a0 = np.zeros(n)
        c0 = np.zeros(system.obs_dim)
        G_true, z0_true = steady_state_gain(system.A, system.B, system.C, M, a=a0, c=c0)
        ref_set = G_true @ np.array([0.5, 0.5]) + z0_true   # zonotope midpoint
        # Infeasible target: outside the zonotope. Set to scaled-+1 corner ×3.
        ref_inf = 3.0 * (G_true @ np.array([1.0, 1.0])) + z0_true
        inside_set, _ = feasibility(ref_set, G_true, z0_true)
        inside_inf, _ = feasibility(ref_inf, G_true, z0_true)

        print(f"\n--- {name}  (n={n}, p={system.obs_dim})")
        print(f"  G_true =\n{G_true}")
        print(f"  z0_true = {z0_true}")
        print(f"  feasible setpoint inside zonotope? {inside_set}")
        print(f"  infeasible setpoint inside zonotope? {inside_inf}")

        # Suppression (ref = 0)
        print("\n  Suppression (ref = 0):")
        print(f"    {'controller':12s}  {'z-RMS':>8s}  {'effort':>8s}  {'sat':>6s}  {'rho(A-BK)':>10s}")
        for cname in ["open", "PropFB", "LQG", "PolePlace", "MPC"]:
            fac = _factory(cname, np.zeros(M.shape[0]))
            logs = run_one(system, fac, T, seed, M, a0, c0, ref=np.zeros(M.shape[0]))
            z = (logs["y"] @ M.T)
            z_rms = rms(z)
            eff = control_effort(logs["u"])
            sat = saturation_fraction(logs["u"])
            # rho_cl reported when K is available (LQG/PolePlace use K).
            rho_cl_str = "-"
            if cname in ("LQG", "PolePlace"):
                ctrl = fac(system.A, system.B, system.C, M, a0, c0, np.zeros(M.shape[0]))
                rho_cl_str = f"{closed_loop_rho(system.A, system.B, ctrl.K):.3f}"
            print(f"    {cname:12s}  {z_rms:8.4f}  {eff:8.2f}  {sat:6.2f}  {rho_cl_str:>10s}")
            sup_rows.append((name, cname, z_rms, eff, sat))

        # Feasible setpoint
        print(f"\n  Feasible setpoint ref = G·[0.5,0.5] + z0:")
        print(f"    {'controller':12s}  {'rms_err':>8s}  {'achieved z[-1]':>30s}  {'effort':>8s}")
        for cname in ["open", "PropFB", "LQG", "PolePlace", "MPC"]:
            fac = _factory(cname, ref_set)
            if cname == "open":
                # OpenLoop with constant feedforward u = (0.5, 0.5) — best-shot at zonotope midpoint.
                fac_ol = lambda A, B, C, M, a, c, ref: OpenLoop(
                    u_const=np.array([0.5, 0.5]), input_dim=B.shape[1]
                )
                logs = run_one(system, fac_ol, T, seed, M, a0, c0, ref=ref_set)
            else:
                logs = run_one(system, fac, T, seed, M, a0, c0, ref=ref_set)
            z = (logs["y"] @ M.T)
            err = z - ref_set
            rms_err = rms(err)
            ach = z[-20:].mean(axis=0)
            eff = control_effort(logs["u"])
            print(f"    {cname:12s}  {rms_err:8.4f}  {str(ach.round(3)):>30s}  {eff:8.2f}")
            set_rows.append((name, cname, rms_err, eff))

        # Infeasible setpoint (MPC should pick boundary; others go to u=0).
        print(f"\n  Infeasible setpoint ref = 3·G·[1,1] + z0:")
        for cname in ["open", "LQG", "MPC"]:
            fac = _factory(cname, ref_inf)
            logs = run_one(system, fac, T, seed, M, a0, c0, ref=ref_inf)
            z = (logs["y"] @ M.T)
            err = z - ref_inf
            rms_err = rms(err)
            ach = z[-20:].mean(axis=0)
            eff = control_effort(logs["u"])
            print(f"    {cname:12s}  rms_err={rms_err:8.4f}  achieved={ach.round(3)}  effort={eff:.2f}")
            inf_set_rows.append((name, cname, rms_err, eff))

    print("\n" + "=" * 84)
    print("M2 acceptance checks:")
    print("=" * 84)
    # Aggregate: Stage A suppression — best controller per scenario vs OpenLoop.
    print("\n  Suppression: best non-open RMS / OpenLoop RMS (lower = better)")
    for name, _, _, _ in SCENARIOS:
        rows = [r for r in sup_rows if r[0] == name]
        open_rms = next(r[2] for r in rows if r[1] == "open")
        best = min(r for r in rows if r[1] != "open")
        ratio = best[2] / open_rms
        print(f"    {name:18s}  best={best[1]:10s} ratio={ratio:.3f}")

    print("\n  Feasible setpoint: best non-open rms_err per scenario")
    for name, _, _, _ in SCENARIOS:
        rows = [r for r in set_rows if r[0] == name]
        best = min(r for r in rows if r[1] in ("LQG", "MPC"))
        open_rms = next(r[2] for r in rows if r[1] == "open")
        ratio = best[2] / open_rms if open_rms > 0 else float("inf")
        print(f"    {name:18s}  best={best[1]:10s}  rms_err={best[2]:.4f}  (OL ratio={ratio:.3f})")


if __name__ == "__main__":
    main()
