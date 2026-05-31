"""Controller performance study helpers for the week3 notebook.

This module runs closed-loop comparisons for the identified Brain model and
the week1 Simulator test cases, then renders overlay plots for the controllers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Mapping
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from IPython.display import display

try:
    import pandas as pd
except Exception:
    pd = None

ROOT = Path(__file__).resolve().parent
WEEK1_DIR = ROOT.parent / "week 1"

for path in (ROOT, ROOT.parent, WEEK1_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from control.control_interface import IdentifiedSystem  # noqa: E402
from control.controllers import (  # noqa: E402
    LQG,
    LQGI,
    MPC,
    OffsetFreeMPC,
    OpenLoop,
    PI,
    PolePlacement,
    ProportionalFeedback,
)
from control.closed_loop import run_closed_loop  # noqa: E402
from control.metrics import rms, settling_time  # noqa: E402
from control.observer import SteadyStateKalman  # noqa: E402
from control.plotting import LABEL_PC1, LABEL_PC2, LABEL_TIME, style_axis, set_question_title  # noqa: E402
from control.reachability import feasibility, steady_state_gain, zonotope_vertices  # noqa: E402
from control.plant import BrainPlant, SimulatorPlant  # noqa: E402
from Simulator import (  # noqa: E402
    default_neural_system,
    input_aligned_system,
    input_blind_system,
    slow_drift_system,
)

try:
    from GG4 import Brain  # noqa: E402
except Exception:
    Brain = None


CONTROLLER_ORDER = [
    "OpenLoop",
    "ProportionalFeedback",
    "LQG",
    "PolePlacement",
    "MPC",
    "PI",
    "LQGI",
    "OffsetFreeMPC",
]

CONTROLLER_COLORS = {
    "OpenLoop": "#8c8c8c",
    "ProportionalFeedback": "#d62728",
    "LQG": "#1f77b4",
    "PolePlacement": "#9467bd",
    "MPC": "#2ca02c",
    "PI": "#e377c2",
    "LQGI": "#17becf",
    "OffsetFreeMPC": "#bcbd22",
}

BRAIN_SEEDS = [0, 1, 2, 3, 4]
SIMULATOR_CASES = {
    "default_neural_system": default_neural_system,
    "input_aligned_system": input_aligned_system,
    "input_blind_system": input_blind_system,
    "slow_drift_system": slow_drift_system,
}
SIMULATOR_SEEDS = [0]


def _as_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    return array


def make_brain_plant(seed: int) -> BrainPlant:
    if Brain is None:
        raise ImportError("GG4.Brain is not available in this environment")
    return BrainPlant(Brain(random_seed=seed))


def make_simulator_plant(factory: Callable[[int], object], seed: int) -> SimulatorPlant:
    return SimulatorPlant(factory(seed=seed), seed=seed)


def make_controller(name: str, A, B, C, M, a, c, ref):
    ref = np.asarray(ref, dtype=float)
    if name == "OpenLoop":
        return OpenLoop(input_dim=B.shape[1])
    if name == "ProportionalFeedback":
        return ProportionalFeedback(Kp=0.05 * np.eye(B.shape[1], M.shape[0]), M=M, ref=ref)
    if name == "LQG":
        return LQG(A, B, C, M, a=a, c=c, rho=1.0, ref=ref)
    if name == "PolePlacement":
        poles = np.linspace(0.5, 0.85, A.shape[0])
        return PolePlacement(A, B, C, M, target_poles=poles, a=a, c=c, ref=ref)
    if name == "MPC":
        return MPC(A, B, C, M, a=a, c=c, horizon=18, rho=0.2, ref=ref)
    if name == "PI":
        return PI(
            Kp=0.05 * np.eye(B.shape[1], M.shape[0]),
            Ki=0.02 * np.eye(B.shape[1], M.shape[0]),
            M=M,
            ref=ref,
            anti_windup=True,
        )
    if name == "LQGI":
        return LQGI(A, B, C, M, a=a, c=c, rho=1.0, rho_q=0.5, ref=ref, anti_windup=True)
    if name == "OffsetFreeMPC":
        return OffsetFreeMPC(A, B, C, M, a=a, c=c, horizon=18, rho=0.2, ref=ref, anti_windup=True)
    raise ValueError(f"Unknown controller: {name}")


def _choose_infeasible_reference(G: np.ndarray, z0: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    candidate = np.asarray(feasible, dtype=float) + np.array([0.35, -0.2])
    scale = 1.4
    while feasibility(candidate, G, z0)[0] and scale < 6.0:
        candidate = np.asarray(z0, dtype=float) + scale * (np.asarray(feasible, dtype=float) - np.asarray(z0, dtype=float)) + np.array([0.2, -0.15])
        scale *= 1.4
    return candidate


def build_task_suite(ref_feasible: np.ndarray, ref_infeasible: np.ndarray, T_run: int) -> Dict[str, dict]:
    ref_feasible = np.asarray(ref_feasible, dtype=float)
    ref_infeasible = np.asarray(ref_infeasible, dtype=float)
    zeros = np.zeros_like(ref_feasible)

    pulse_start = max(5, T_run // 5)
    pulse_end = max(pulse_start + 8, int(T_run * 0.55))
    pulse = np.zeros((T_run, ref_feasible.size), dtype=float)
    pulse[pulse_start:pulse_end] = ref_feasible

    return {
        "hold_feasible": {
            "title": "How do the controllers hold a feasible target?",
            "ref_series": np.tile(ref_feasible, (T_run, 1)),
            "ref_fn": lambda _t, ref=ref_feasible: ref,
            "settle_from": 0,
            "region": True,
            "feasible_target": ref_feasible,
            "infeasible_target": ref_infeasible,
        },
        "hold_infeasible": {
            "title": "What happens when the target is not holdable?",
            "ref_series": np.tile(ref_infeasible, (T_run, 1)),
            "ref_fn": lambda _t, ref=ref_infeasible: ref,
            "settle_from": 0,
            "region": True,
            "feasible_target": ref_feasible,
            "infeasible_target": ref_infeasible,
        },
        "track_pulse": {
            "title": "How do the controllers track a pulse reference?",
            "ref_series": pulse,
            "ref_fn": lambda t, series=pulse: series[t],
            "settle_from": pulse_end,
            "region": False,
            "feasible_target": ref_feasible,
            "infeasible_target": ref_infeasible,
        },
        "suppress": {
            "title": "How well do the controllers suppress the readout?",
            "ref_series": np.tile(zeros, (T_run, 1)),
            "ref_fn": lambda _t, ref=zeros: ref,
            "settle_from": 0,
            "region": False,
            "feasible_target": ref_feasible,
            "infeasible_target": ref_infeasible,
        },
    }


def compute_metrics(z: np.ndarray, ref_series: np.ndarray, settle_from: int = 0) -> dict:
    z = _as_2d(z)
    ref_series = _as_2d(ref_series)
    if ref_series.shape[0] != z.shape[0]:
        raise ValueError("reference and trajectory must have the same length")
    final_window = max(10, z.shape[0] // 5)
    reference_tail = ref_series[-1]
    settle_idx = settle_from + settling_time(z[settle_from:], reference_tail)
    sse = rms(z[-final_window:] - ref_series[-final_window:])
    return {
        "settling_time": int(settle_idx),
        "steady_state_error": float(sse),
        "rms_error": float(rms(z - ref_series)),
        "final_mean": z[-final_window:].mean(axis=0),
    }


def calibrate_and_run(
    plant_factory: Callable[[int], object],
    seed: int,
    *,
    T_cal: int = 600,
    T_run: int = 200,
    n_latent: int = 4,
    controller_order: list[str] | None = None,
) -> dict:
    controller_order = controller_order or CONTROLLER_ORDER
    calibration_plant = plant_factory(seed)
    iface = IdentifiedSystem().calibrate(
        calibration_plant,
        T_cal=T_cal,
        n=n_latent,
        m=calibration_plant.input_dim,
        seed=seed,
    )
    A, B, C, Q, R, a, c = iface.params()
    M = iface.readout_M
    G, z0 = steady_state_gain(A, B, C, M, a=a, c=c)

    if M.shape[0] < 2:
        raise RuntimeError("This notebook expects a 2D readout (PC1 and PC2)")

    feasible_target = z0 + G @ np.full(B.shape[1], 0.5)
    infeasible_target = _choose_infeasible_reference(G, z0, feasible_target)
    task_suite = build_task_suite(feasible_target, infeasible_target, T_run)

    suite = {
        "seed": seed,
        "iface": iface,
        "G": G,
        "z0": z0,
        "task_suite": task_suite,
        "tasks": {},
    }

    for task_name, task_spec in task_suite.items():
        task_results = {}
        for controller_name in controller_order:
            run_plant = plant_factory(seed)
            observer = SteadyStateKalman(A, B, C, Q, R, a=a, c=c)
            controller = make_controller(controller_name, A, B, C, M, a, c, task_spec["ref_series"][0])
            logs = run_closed_loop(run_plant, controller, observer, T=T_run, ref_fn=task_spec["ref_fn"])
            z = logs["y"] @ M.T
            metrics = compute_metrics(z, task_spec["ref_series"], settle_from=task_spec["settle_from"])
            task_results[controller_name] = {"logs": logs, "z": z, "metrics": metrics}
        suite["tasks"][task_name] = task_results
    return suite


def _task_table(task_results: Mapping[str, dict]):
    rows = []
    for controller_name in CONTROLLER_ORDER:
        if controller_name not in task_results:
            continue
        metrics = task_results[controller_name]["metrics"]
        rows.append(
            {
                "controller": controller_name,
                "settling_time": metrics["settling_time"],
                "steady_state_error": metrics["steady_state_error"],
                "rms_error": metrics["rms_error"],
            }
        )
    if pd is not None:
        return pd.DataFrame(rows)
    return rows


def _legend_handles(include_reference: bool = True):
    handles = []
    labels = []
    if include_reference:
        handles.append(Line2D([0], [0], color="black", lw=1.4, ls="--"))
        labels.append("desired")
    for controller_name in CONTROLLER_ORDER:
        handles.append(Line2D([0], [0], color=CONTROLLER_COLORS[controller_name], lw=2.0))
        labels.append(controller_name)
    return handles, labels


def plot_task_results(
    seed_label: str,
    task_name: str,
    task_spec: Mapping[str, object],
    task_results: Mapping[str, dict],
    *,
    M: np.ndarray,
    G: np.ndarray,
    z0: np.ndarray,
):
    ref_series = _as_2d(task_spec["ref_series"])
    t = np.arange(ref_series.shape[0])
    include_region = bool(task_spec.get("region", False))

    if include_region:
        fig = plt.figure(figsize=(14.5, 7.6))
        grid = GridSpec(2, 2, figure=fig, width_ratios=[1.9, 1.1], wspace=0.26, hspace=0.24)
        ax_pc1 = fig.add_subplot(grid[0, 0])
        ax_pc2 = fig.add_subplot(grid[1, 0], sharex=ax_pc1)
        ax_region = fig.add_subplot(grid[:, 1])
    else:
        fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.0), sharex=True)
        ax_pc1, ax_pc2 = axes
        ax_region = None

    for dim, axis in enumerate((ax_pc1, ax_pc2)):
        axis.plot(t, ref_series[:, dim], color="black", lw=1.4, ls="--", label="desired")
        for controller_name in CONTROLLER_ORDER:
            if controller_name not in task_results:
                continue
            z = task_results[controller_name]["z"]
            axis.plot(
                t,
                z[:, dim],
                color=CONTROLLER_COLORS[controller_name],
                lw=1.7,
                label=controller_name,
            )
        axis.set_ylabel(LABEL_PC1 if dim == 0 else LABEL_PC2)
        style_axis(axis)

    ax_pc2.set_xlabel(LABEL_TIME)
    question = f"{seed_label}: {task_spec['title']}"
    set_question_title(ax_pc1, question)

    handles, labels = _legend_handles(include_reference=True)
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))

    if ax_region is not None:
        verts = zonotope_vertices(G, z0)
        feasible_target = np.asarray(task_spec["feasible_target"], dtype=float)
        infeasible_target = np.asarray(task_spec["infeasible_target"], dtype=float)
        poly = verts[[0, 1, 3, 2, 0]]
        ax_region.fill(poly[:, 0], poly[:, 1], color="#a3c4dc", alpha=0.28, label="holdable region")
        ax_region.plot(poly[:, 0], poly[:, 1], color="#1f77b4", lw=1.5)
        ax_region.scatter(verts[:, 0], verts[:, 1], color="#1f77b4", s=28, zorder=5)
        ax_region.scatter(*feasible_target, marker="*", s=230, color="#2ca02c", edgecolor="black", linewidth=0.8, zorder=6)
        ax_region.scatter(*infeasible_target, marker="*", s=230, color="#d62728", edgecolor="black", linewidth=0.8, zorder=6)
        for controller_name in CONTROLLER_ORDER:
            if controller_name not in task_results:
                continue
            final_point = task_results[controller_name]["metrics"]["final_mean"]
            ax_region.scatter(
                *final_point,
                marker="P",
                s=105,
                color=CONTROLLER_COLORS[controller_name],
                edgecolor="black",
                linewidth=0.45,
                zorder=7,
            )
        ax_region.set_xlabel(LABEL_PC1)
        ax_region.set_ylabel(LABEL_PC2)
        ax_region.set_title("Can we hold the target?", loc="left")
        style_axis(ax_region)
        ax_region.set_aspect("equal", adjustable="datalim")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def display_task_results(suite: Mapping[str, object], seed_label: str) -> None:
    iface = suite["iface"]
    M = iface.readout_M
    G = suite["G"]
    z0 = suite["z0"]
    for task_name in ("hold_feasible", "hold_infeasible", "track_pulse", "suppress"):
        task_spec = suite["task_suite"][task_name]
        task_results = suite["tasks"][task_name]
        fig = plot_task_results(seed_label, task_name, task_spec, task_results, M=M, G=G, z0=z0)
        display(fig)
        plt.close(fig)
        table = _task_table(task_results)
        if pd is not None:
            display(table)
        else:
            print(f"\n{seed_label} | {task_name}")
            for row in table:
                print(
                    f"  {row['controller']:22s}  settling={row['settling_time']:4d}  "
                    f"steady_state_error={row['steady_state_error']:.4f}  rms_error={row['rms_error']:.4f}"
                )


def run_brain_study(*, seeds: list[int] | None = None, T_cal: int = 600, T_run: int = 200, n_latent: int = 4) -> dict:
    seeds = BRAIN_SEEDS if seeds is None else seeds
    all_results = {}
    for seed in seeds:
        suite = calibrate_and_run(make_brain_plant, seed, T_cal=T_cal, T_run=T_run, n_latent=n_latent)
        all_results[seed] = suite
        display_task_results(suite, seed_label=f"Brain seed {seed}")
    return all_results


def run_simulator_study(
    *,
    cases: Mapping[str, Callable[[int], object]] | None = None,
    seeds: list[int] | None = None,
    T_cal: int = 600,
    T_run: int = 200,
    n_latent: int = 4,
) -> dict:
    cases = SIMULATOR_CASES if cases is None else cases
    seeds = SIMULATOR_SEEDS if seeds is None else seeds
    all_results = {}
    for case_name, factory in cases.items():
        for seed in seeds:
            def plant_factory(seed_value: int, factory=factory):
                return make_simulator_plant(factory, seed_value)

            suite = calibrate_and_run(plant_factory, seed, T_cal=T_cal, T_run=T_run, n_latent=n_latent)
            all_results[(case_name, seed)] = suite
            display_task_results(suite, seed_label=f"Simulator {case_name} | seed {seed}")
    return all_results
