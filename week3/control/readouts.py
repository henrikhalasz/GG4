"""Readout projections — configurable ``M`` (shape ``(q, p)``).

A *readout* maps the ``p``-dimensional measurement ``y`` to a ``q``-dimensional
scalar / vector we want to hold or track::

    z = M y

This module owns the construction of two readouts that the rest of the
package consumes interchangeably:

* **2-D — top principal components.** ``compute_pca_readout(Y, k=2)`` returns
  the top-``k`` right singular vectors of mean-centred probe measurements.
  The two PCs are usually *not* equally controllable; PC2 is where the
  reachable-region / infeasible-setpoint story lives (spec §5).
* **1-D — dominant controllable direction.** (M4) The top left-singular
  vector of the steady-state DC gain ``G_full = C(I − A)^{-1} B``; this is
  the *sustainable* output direction and almost every target on it is
  feasible. Built in M4 — placeholder API below.

The readout's steady-state input → output gain is
``G_readout = M C (I − A)^{-1} B``; its condition number is the headline
diagnostic for "are PC1 and PC2 independently controllable by the input?"

Consumes only ``(A, B, C, M)``; no estimator imports here.
"""

from __future__ import annotations

import numpy as np


def compute_pca_readout(Y: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` principal components of measurements ``Y`` as a readout ``M``.

    Each PC is a weighted combination of the ``p`` measurements capturing a
    dominant pattern of population activity. Returned ``M`` has shape
    ``(k, p)`` (rows are unit-norm).
    """
    Y = np.asarray(Y, dtype=float)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    M = Vt[:k].copy()
    var_total = float((S ** 2).sum())
    explained = (S[:k] ** 2) / var_total if var_total > 0 else np.zeros(k)
    return M, explained


def readout_steady_state_gain(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, M: np.ndarray
) -> np.ndarray:
    """Steady-state input → readout gain ``G = M C (I − A)^{-1} B``.

    Used to verify the readout is well-controllable (small ``cond(G)`` means
    each row of the readout can be moved independently by the inputs).
    """
    n = A.shape[0]
    I = np.eye(n)
    return M @ C @ np.linalg.solve(I - A, B)


def dominant_controllable_direction(
    A: np.ndarray, B: np.ndarray, C: np.ndarray
) -> np.ndarray:
    """1-D readout along the top left-singular vector of the full DC gain.

    Returns ``M`` of shape ``(1, p)`` whose row is the top left-singular
    vector of the input → output DC gain ``G_full = C(I−A)^{-1}B``. This is
    the *sustainable* direction in output space: under a constant input
    ``u`` the steady-state shift in this direction is the largest possible
    for the given ``‖u‖``. Almost every target on this 1-D line is feasible
    (it's the dimension where the rank-1 nature of ``G`` is *helpful*).
    """
    n = A.shape[0]
    I = np.eye(n)
    G_full = C @ np.linalg.solve(I - A, B)  # (p, m)
    U, _, _ = np.linalg.svd(G_full, full_matrices=False)
    return U[:, :1].T  # (1, p) — unit-norm row


def auto_proportional_gain(
    G_readout: np.ndarray, alpha: float = 0.3, rcond: float = 0.1
) -> np.ndarray:
    """Model-aware default ``K_p`` for proportional feedback / PI.

    A common-mistake controller default — ``K_p = alpha * eye(m, q)`` —
    silently couples the FIRST input to the FIRST readout component
    regardless of the underlying input-output direction structure. On
    the Brain's 1-D dominant readout, `G_readout = [-42.2, 3.71]` means
    u[0] is the *wrong* channel to push for the controllable direction;
    that default produced 30× noise-floor divergence in M4. The fix:
    ``K_p = alpha * pinv_thresh(G_readout)`` so a positive error drives
    the input *direction* that moves the readout toward the target, with
    a singular-value threshold ``rcond`` that drops near-zero ``G``
    directions (otherwise pinv amplifies the poorly-controllable
    direction — e.g. the 2-D PCA readout on the Brain has
    ``σ₁/σ₀ = 8e-3``, so a non-thresholded pinv asks for ~125×
    amplification along PC2 and saturates the inputs).

    ``alpha`` is the per-step closed-loop fraction of the static-feedback
    law (rule of thumb 0.1–0.5; 0.3 by default).

    **Caveat — this makes proportional feedback model-aware.** There is
    no purely model-free baseline that survives on this system. The
    rank-1 input-output geometry means you must know ``G``'s direction
    (and exclude ``G``'s null space) to push the right way. State the
    finding rather than ship a "model-free" baseline that secretly uses
    ``G``: see RESULTS.md M6.
    """
    G = np.asarray(G_readout, dtype=float)
    return alpha * np.linalg.pinv(G, rcond=rcond)


def auto_integral_gain(
    G_readout: np.ndarray, alpha_i: float = 0.1, rcond: float = 0.1
) -> np.ndarray:
    """Model-aware default ``K_i`` for PI — see :func:`auto_proportional_gain`."""
    G = np.asarray(G_readout, dtype=float)
    return alpha_i * np.linalg.pinv(G, rcond=rcond)


def build_readout(
    kind: str,
    *,
    A: np.ndarray | None = None,
    B: np.ndarray | None = None,
    C: np.ndarray | None = None,
    Y_probe: np.ndarray | None = None,
    k: int = 2,
) -> np.ndarray:
    """Configurable readout builder — readouts are config, not code forks.

    Parameters
    ----------
    kind : {"pca", "dominant"}
        ``"pca"``    → 2-D (or ``k``-D) PCA of probe measurements ``Y_probe``;
                       used for the reachability / infeasible-setpoint story
                       where the 2nd PC may be poorly controllable (the
                       ``cond(G_readout)`` is the headline diagnostic).
        ``"dominant"`` → 1-D along the dominant left-singular vector of
                       ``C(I−A)^{-1}B``; used for the clean hold/track story
                       (almost every target on this line is feasible).

    Returns ``M`` of shape ``(q, p)`` where ``q`` is the readout dimension.
    """
    if kind == "pca":
        if Y_probe is None:
            raise ValueError("kind='pca' requires Y_probe")
        M, _ = compute_pca_readout(Y_probe, k=k)
        return M
    if kind == "dominant":
        if A is None or B is None or C is None:
            raise ValueError("kind='dominant' requires (A, B, C)")
        return dominant_controllable_direction(A, B, C)
    raise ValueError(f"unknown readout kind {kind!r}")
