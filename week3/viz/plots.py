"""Multi-panel comparison figures + anchoring primitives.

Every control-error panel in M3 / M4 / M5 / the M6 deliverable expresses
steady-state error as **× noise floor** (single primary unit) with two
reference markers:

    * the noise floor at ``y = 1.0`` (the *best physically achievable*
      steady-state error against per-step measurement noise);
    * an open-loop reference (the no-control baseline) either as a bar
      in the comparison or a dashed horizontal line.

These helpers make the convention reusable and consistent across every
figure. Use them whenever you plot a control-error axis.

The colour map and plain-language axis labels live in :mod:`viz.style`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .style import CONTROLLER_COLORS


# ----------------------------------------------------------------------------
# Anchoring primitives — call exactly once per control-error axis.
# ----------------------------------------------------------------------------
def add_noise_floor_line(ax, *,
                         label: str = "noise floor (best achievable)",
                         y: float = 1.0,
                         color: str = "black",
                         ls: str = ":",
                         lw: float = 1.1) -> None:
    """Add the ``y = 1.0`` noise-floor reference line to a ``× NF`` axis.

    Pairs with :func:`add_openloop_reference` so every control-error
    panel answers "is this good?" without external context: bars near
    ``1.0`` are at the noise floor (best achievable), and the OpenLoop
    reference shows the gap closed-loop closes.
    """
    ax.axhline(y, color=color, ls=ls, lw=lw, label=label)


def add_openloop_reference(ax,
                           ol_value: Optional[float],
                           *,
                           label_prefix: str = "OpenLoop (no control)",
                           color: Optional[str] = None,
                           ls: str = "--",
                           lw: float = 1.2) -> None:
    """Add a dashed horizontal OpenLoop reference line.

    Skips silently when ``ol_value`` is ``None`` or non-finite (so calls
    from panels that may not have an OL number available are still safe).
    """
    if ol_value is None:
        return
    try:
        v = float(ol_value)
    except (TypeError, ValueError):
        return
    if not np.isfinite(v):
        return
    if color is None:
        color = CONTROLLER_COLORS.get("OpenLoop", "#888888")
    ax.axhline(v, color=color, ls=ls, lw=lw,
                label=f"{label_prefix} ≈ {v:.1f}×")


def error_axis(ax, *,
               ol_value: Optional[float] = None,
               ylabel: str = "steady-state error\n(× per-readout noise floor)",
               legend: bool = True,
               legend_fontsize: int = 7,
               legend_loc: str = "upper right") -> None:
    """Apply the full standard anchoring to a control-error axis::

        ax.set_ylabel( '× noise floor' )
        + noise-floor line at y = 1.0
        + OL reference line (if ``ol_value`` is finite)
        + legend.

    Call after plotting the bars / lines for the controllers.
    """
    ax.set_ylabel(ylabel)
    add_openloop_reference(ax, ol_value)
    add_noise_floor_line(ax)
    if legend:
        ax.legend(fontsize=legend_fontsize, loc=legend_loc)


def anchor_caption(panel_question: str, *,
                   ol_value: Optional[float] = None,
                   extra: Optional[str] = None) -> str:
    """Build the standard one-line caption stating the anchor.

    Example::

        anchor_caption(
            "Honest vs yardstick — hold task on real Brain",
            ol_value=11.5,
        )
        →  "Honest vs yardstick — hold task on real Brain
            × noise floor; 1.0 = best achievable; OpenLoop ≈ 11.5×"
    """
    parts = [panel_question,
             "× noise floor; 1.0 = best achievable"]
    if ol_value is not None and np.isfinite(ol_value):
        parts[-1] += f"; OpenLoop ≈ {float(ol_value):.1f}×"
    if extra:
        parts.append(extra)
    return "\n".join(parts)
