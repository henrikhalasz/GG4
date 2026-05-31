"""Shared plot styling for v3 E3 figures (spec §6 clarity rules).

Provides a consistent controller → colour map and a few primitives so every
v3 figure follows the same conventions:

    * plain-language title (one question per figure)
    * defined axes — never bare `z`, never "zonotope"
    * desired (dashed) / achieved (solid) / uncontrolled (faint grey) overlay
    * errors annotated as both fraction of target *and* multiples of the
      open-loop readout noise floor
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt


# Controller → colour map. Same across every E3 figure so a reader can scan.
CONTROLLER_COLORS = {
    "OpenLoop":      "#888888",  # neutral grey
    "PropFeedback":  "#d62728",  # red
    "LQG":           "#1f77b4",  # blue
    "PolePlace":     "#9467bd",  # purple
    "MPC":           "#2ca02c",  # green
    "PI":            "#e377c2",  # pink
    "LQGI":          "#17becf",  # cyan
    "OffsetFreeMPC": "#bcbd22",  # olive
}

# Plain-language axis labels. Use everywhere instead of "z[0]" / "z[1]".
LABEL_PC1 = "PC1 activity\n(1st dominant pattern of population activity)"
LABEL_PC2 = "PC2 activity\n(2nd dominant pattern of population activity)"
LABEL_TIME = "time step (one step = one measurement)"
LABEL_U = "input (push only, 0–1)"
LABEL_HOLD_REGION = "levels we can hold steady"


def style_axis(ax) -> None:
    """Apply the minimal v3 axis style."""
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def set_question_title(ax, question: str, subtitle: Optional[str] = None) -> None:
    """Title as a plain-language question (spec §6)."""
    if subtitle:
        ax.set_title(f"{question}\n{subtitle}", fontsize=11, loc="left")
    else:
        ax.set_title(question, fontsize=11, loc="left")


def plot_desired_vs_achieved(
    ax,
    t: np.ndarray,
    ref: np.ndarray,
    series: Iterable[tuple[str, np.ndarray]],
    *,
    uncontrolled: Optional[np.ndarray] = None,
    y_label: Optional[str] = None,
    x_label: Optional[str] = LABEL_TIME,
    target_annotation: Optional[float] = None,
) -> None:
    """Primitive panel: dashed desired + solid achieved curves + faint
    grey uncontrolled baseline. Each entry in ``series`` is
    ``(controller_name, achieved_array)``."""
    if uncontrolled is not None:
        ax.plot(t, uncontrolled, color="0.75", lw=0.9, alpha=0.7,
                label="uncontrolled (open loop)")
    ax.plot(t, ref, "--", color="black", lw=1.2, label="desired (target)")
    for name, achieved in series:
        color = CONTROLLER_COLORS.get(name, "tab:red")
        ax.plot(t, achieved, "-", color=color, lw=1.6, label=name)
    if target_annotation is not None:
        ax.axhline(target_annotation, color="black", lw=0.5, alpha=0.25)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    style_axis(ax)


def annotate_normalised_error(
    ax,
    rms_err: float,
    target_scale: float,
    noise_floor: float,
    *,
    loc: str = "upper right",
) -> None:
    """Corner box: error as % of target and × the noise floor (spec §6)."""
    if target_scale > 1e-9:
        frac_target_pct = rms_err / target_scale * 100.0
        line1 = f"rms err = {frac_target_pct:.0f}% of target"
    else:
        line1 = f"rms err = {rms_err:.2f}"
    if noise_floor > 1e-9:
        line2 = f"        = {rms_err / noise_floor:.1f}x noise floor"
    else:
        line2 = ""
    txt = "\n".join(filter(None, [line1, line2]))
    if loc == "upper right":
        x, y, ha, va = 0.98, 0.98, "right", "top"
    elif loc == "upper left":
        x, y, ha, va = 0.02, 0.98, "left", "top"
    elif loc == "lower right":
        x, y, ha, va = 0.98, 0.02, "right", "bottom"
    else:
        x, y, ha, va = 0.02, 0.02, "left", "bottom"
    ax.text(x, y, txt, transform=ax.transAxes, ha=ha, va=va,
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       alpha=0.92, edgecolor="0.7"))


def plot_holdable_region_2d(
    ax,
    verts: np.ndarray,
    *,
    feasible_target: Optional[np.ndarray] = None,
    infeasible_target: Optional[np.ndarray] = None,
    achieved: Optional[dict] = None,
) -> None:
    """2D plot of the (PC1, PC2) holdable region (the affine zonotope of
    `Gu + z0` over `u ∈ [0,1]^2`). Labels in plain language."""
    poly = verts[[0, 1, 3, 2, 0]]
    ax.fill(poly[:, 0], poly[:, 1], color="#a3c4dc", alpha=0.30,
            label=LABEL_HOLD_REGION)
    ax.plot(poly[:, 0], poly[:, 1], color="#1f77b4", lw=1.5)
    ax.scatter(verts[:, 0], verts[:, 1], color="#1f77b4", s=35, zorder=5)
    if feasible_target is not None:
        ax.scatter(*feasible_target, marker="*", s=260,
                    color="#2ca02c", edgecolor="black", linewidth=0.8,
                    label="feasible target (inside)", zorder=6)
    if infeasible_target is not None:
        ax.scatter(*infeasible_target, marker="*", s=260,
                    color="#d62728", edgecolor="black", linewidth=0.8,
                    label="infeasible target (outside)", zorder=6)
    if achieved is not None:
        for name, pt in achieved.items():
            color = CONTROLLER_COLORS.get(name, "tab:olive")
            ax.scatter(*pt, marker="P", s=120, color=color,
                        edgecolor="black", linewidth=0.5,
                        label=f"{name} achieved", zorder=7)
    ax.set_xlabel(LABEL_PC1)
    ax.set_ylabel(LABEL_PC2)
    ax.set_aspect("equal", adjustable="datalim")
    style_axis(ax)


def add_caption(fig, text: str) -> None:
    """One-line caption under the title (spec §6 — plain-language explanation)."""
    fig.text(0.5, 0.02, text, ha="center", va="bottom",
              fontsize=9.5, color="0.25",
              wrap=True)
