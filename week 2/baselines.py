"""Comparison baseline for the Week 2 estimation task.

A simpler alternative to the primary subspace-ID + Kalman method
(``estimator.py``), used only for comparison in ``week2_analysis.ipynb``:

* **Approach A — PCA** (``estimate_pca``): a purely static linear decomposition.
  It keeps the top ``n`` principal components of the (centred) observations as the
  latent state.  It assumes nothing beyond linearity and so exposes the limits of
  a static decomposition on systems where the input becomes visible only after a
  dynamical delay.

It returns the same shapes as the primary method —
``(latent_states (T, n), inputs (T, m))`` — and inputs are estimated the same way
(AR(1) residual of the latent scores mapped through a pseudoinverse), so the two
methods are directly comparable.

The same similarity-transform / linear-map identifiability caveats as the primary
method apply (see ``estimator.py`` module docstring, ``week2.md`` §3 B.4): compare
through reconstructed observations and best-linear-fit input recovery, never by
direct subtraction.

Dependencies: ``numpy`` + the Python standard library only.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _ar1_residual_input(Z: np.ndarray, m: int) -> np.ndarray:
    """Estimate an ``m``-dimensional input from the residual of an AR(1) fit.

    Fits a first-order linear model ``Z_{t+1} ~ M Z_t`` to the latent scores,
    takes the one-step residual (the part the autonomous model cannot predict),
    and maps it down to ``m`` dimensions through the pseudoinverse of the residual
    sequence's dominant ``m`` directions.

    Parameters
    ----------
    Z : np.ndarray, shape (T, n)
        Latent score sequence.
    m : int
        Desired input dimension.

    Returns
    -------
    np.ndarray, shape (T, m)
        Estimated input (identifiable up to a linear map; final step padded).
    """
    T, n = Z.shape
    u = np.zeros((T, m))
    if T < 2:
        return u

    # AR(1) transition by least squares: Z[1:] ~ Z[:-1] @ M.
    M = np.linalg.lstsq(Z[:-1], Z[1:], rcond=None)[0]   # (n, n)
    resid = Z[1:] - Z[:-1] @ M                          # (T-1, n)

    # Dominant m directions of the residual play the role of the input matrix B.
    U, _, _ = np.linalg.svd(resid.T, full_matrices=False)
    cols = min(m, U.shape[1])
    B = np.zeros((n, m))
    B[:, :cols] = U[:, :cols]

    u[:-1] = resid @ np.linalg.pinv(B).T                # (T-1, m)
    u[-1] = u[-2]                                        # pad final step
    return u


def estimate_pca(Y: np.ndarray, n: int, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Approach A — PCA baseline.

    Centres ``Y``, takes its SVD ``Y = U S V^T`` and keeps the top ``n`` columns of
    ``U S`` as the latent states; inputs come from the AR(1) residual of those
    scores (see :func:`_ar1_residual_input`).

    Parameters
    ----------
    Y : np.ndarray, shape (T, p)
        Observation sequence.
    n : int
        Latent dimension.
    m : int
        Input dimension.

    Returns
    -------
    latent_states : np.ndarray, shape (T, n)
    inputs : np.ndarray, shape (T, m)
    """
    Y = np.asarray(Y, dtype=float)
    T = Y.shape[0]
    Yc = Y - Y.mean(axis=0)

    U, S, _ = np.linalg.svd(Yc, full_matrices=False)
    k = min(n, U.shape[1])
    scores = U[:, :k] * S[:k]                           # (T, k) = projection onto PCs
    latent = np.zeros((T, n))
    latent[:, :k] = scores

    inputs = _ar1_residual_input(latent, m)
    return latent, inputs


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week 1"))
    import Simulator as sim  # noqa: E402

    T = 500
    system = sim.default_neural_system(seed=0, obs_dim=16)
    u_seq = sim.mixed_input(T, system.input_dim, seed=0)
    y = system.simulate(T, U=u_seq)["y"]

    latent, inputs = estimate_pca(y, 4, 2)
    assert latent.shape == (T, 4), latent.shape
    assert inputs.shape == (T, 2), inputs.shape
    assert np.all(np.isfinite(latent)) and np.all(np.isfinite(inputs))
    print(f"PCA (A) latent {latent.shape}, inputs {inputs.shape} -- OK")
    print("baselines shape checks passed.")
