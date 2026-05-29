"""Week 2 comparison estimator — smoothness-penalised input back-out.

This module is a **drop-in alternative** to :mod:`estimator` that keeps the exact
same identification and Kalman-filtering pipeline but replaces only the final
input estimation step.  It is meant for comparison, not as a replacement; the
primary submission remains :mod:`estimator`.

Method
------
The primary estimator (``estimator.py`` §2(b)/§2(c) of ``interim_report.md``)
recovers the input from the filtered state with a *per-timestep* pseudoinverse,

    u_t = pinv(B_hat) (x_{t+1} - A_hat x_t),                                  (1)

solved independently at every ``t``.  Because ``pinv(B_hat)`` inverts the
dynamics one step at a time, it has no notion of temporal continuity: any
process noise ``w_t`` that leaks into the one-step increment
``x_{t+1} - A_hat x_t`` is attributed wholesale to a high-frequency wiggle in
``u_hat``.  When the true input is *autocorrelated* (a sinusoid, a slow ramp, a
pulse) this noise amplification dominates the recovered signal — exactly the
regime where ``interim_report.md`` §3 reports the worst input recovery
(``sinusoidal`` R^2 = 0.006, ``mixed`` R^2 = 0.186).

This module instead solves the **joint, smoothness-penalised** least-squares
problem over the whole trajectory,

    u_hat_{0:T-2} = argmin_u  sum_{t=0}^{T-2} || x_{t+1} - A_hat x_t - B_hat u_t ||^2
                              + lam * sum_{t=1}^{T-2} || u_t - u_{t-1} ||^2.    (2)

The first term is the same one-step dynamics residual minimised pointwise by
(1); the second is a first-difference (Tikhonov) penalty that pushes back
against spurious step-to-step jumps whenever the data permit a smoother ``u``.
Its closed form is the normal-equation linear system

    (H^T H + lam D^T D) u = H^T d,                                            (3)

where ``H`` is block-diagonal in ``B_hat``, ``d`` stacks the increments
``x_{t+1} - A_hat x_t`` and ``D`` is the first-difference operator.  The system
matrix ``H^T H + lam D^T D`` is **block-tridiagonal** (block size ``m``), so it
is assembled and solved sparsely (``scipy.sparse``) — never densely, which would
cost ``O(T^2 m^2)`` memory and OOM on long sequences.

Choice of ``lam``
-----------------
``lam`` is fixed once, at function entry, from the identified model — there is
**no call-time tuning**::

    lam = alpha * trace(Q_hat) / trace(B_hat B_hat^T),   alpha = 100.

``trace(Q_hat)`` is the total process-noise power and ``trace(B_hat B_hat^T)``
is the squared Frobenius norm of the input gain, so the *ratio* scales the
smoothness penalty with the noise-to-gain level that the per-timestep solution
would otherwise mistake for input (more process noise -> more smoothing; larger
input gain -> less).  The dimensionless ``alpha`` sets the overall strength.  It
is **not** 1: the data term in (2) is unweighted, so for the low-noise systems
here (``trace(Q_hat) ~ 1e-3``) a unit constant under-smooths to the point of
having no effect.  ``alpha = 100`` was chosen empirically on
``default_neural_system`` as the smallest round value that delivers a visible
gain on autocorrelated inputs while preserving white-input recovery (see the
``s4_lambda_sensitivity`` sweep in ``week2_analysis.ipynb``).  The chosen ``lam``
is recorded in the module-level :data:`LAST_LAMBDA` for notebook inspection.

Trade-off
---------
The penalty helps **autocorrelated** inputs (the common case) by suppressing the
noise-driven wiggles of (1); it can *slightly* hurt a genuinely **white**
(broadband) input, whose true step-to-step variation the penalty also shrinks.
The default ``lam`` is deliberately modest (noise-matched) so the white-input
cost stays small while the autocorrelated-input gain is realised — see the
``s4_lambda_sensitivity`` sweep in ``week2_analysis.ipynb``.

Public interface (identical to :mod:`estimator`)::

    estimate_latent_and_input(observation, LatentDim, InputDim)
        -> (latent_states (T, LatentDim), inputs (T, InputDim))

The same ``GL(n)`` similarity / linear-map identifiability caveats apply (see the
:mod:`estimator` module docstring): compare through reconstructed observations
``y_hat = C_hat @ x_hat`` and best-linear-fit input recovery, never by direct
subtraction.

Dependencies: ``numpy`` + ``scipy`` (sparse linear algebra for the banded solve).
The identification / filtering helpers are imported from :mod:`estimator` so that
code is reused, not duplicated.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# Reuse the existing identification + filtering machinery verbatim (do not
# duplicate it).  Only the input back-out below is new.
from estimator import (
    _N_FILTER_PASSES,
    _estimate_inputs,
    _fit_subspace_model,
    _kalman_filter,
)

# --- Fixed hyperparameters (never tuned at call time) ----------------------
_ALPHA = 100.0        # scales the noise-matched smoothness penalty (lam = alpha * tr Q / tr BB^T)
_SOLVE_JITTER = 1e-8  # added to the LHS diagonal to handle near-singular B^T B
_DENOM_FLOOR = 1e-12  # guards the lam trace-ratio against a vanishing input gain

# Module-level record of the most recently chosen lam (for notebook inspection).
LAST_LAMBDA: Optional[float] = None


# ---------------------------------------------------------------------------
# Smoothness-penalised input back-out
# ---------------------------------------------------------------------------
def _default_lambda(Q: np.ndarray, B: np.ndarray) -> float:
    """Noise-matched smoothness weight ``lam = alpha * tr(Q) / tr(B B^T)``.

    The ratio compares the total process-noise power ``trace(Q)`` (what the
    per-timestep pseudoinverse would otherwise mis-attribute to the input) with
    the squared Frobenius norm of the input gain ``trace(B B^T) = ||B||_F^2``,
    giving a penalty on the same scale as the noise the smoother is meant to
    reject.

    Parameters
    ----------
    Q : np.ndarray, shape (n, n)
        Estimated process-noise covariance.
    B : np.ndarray, shape (n, m)
        Estimated input matrix.

    Returns
    -------
    float
        Non-negative smoothness weight ``lam``.
    """
    tr_q = float(np.trace(Q))
    tr_bb = float(np.sum(B * B))               # trace(B B^T) = ||B||_F^2
    lam = _ALPHA * tr_q / max(tr_bb, _DENOM_FLOOR)
    return max(lam, 0.0)


def _build_smoothness_system(
    x_filt: np.ndarray, A: np.ndarray, B: np.ndarray, lam: float
) -> Tuple[sp.spmatrix, np.ndarray, sp.spmatrix]:
    """Assemble the sparse operators of the smoothness-penalised problem (2).

    Builds the three pieces of the least-squares objective
    ``||H u - d||^2 + lam ||D u||^2`` over the ``N = T - 1`` unknown input
    vectors ``u_0 .. u_{T-2}``:

    * ``H`` — block-diagonal data operator, ``N`` copies of ``B`` on the
      diagonal, so ``H @ vec(u)`` stacks ``B u_0 .. B u_{N-1}``;
    * ``d`` — stacked one-step increments ``x_{t+1} - A x_t``;
    * ``D`` — first-difference operator, ``D @ vec(u)`` stacks
      ``u_1 - u_0 .. u_{N-1} - u_{N-2}``.

    The penalty weight ``lam`` is applied later, in
    :func:`_solve_smoothed_inputs`; it is accepted here so the (build, solve)
    pair shares one signature and the assembled problem is testable in isolation.

    Parameters
    ----------
    x_filt : np.ndarray, shape (T, n)
        Filtered latent-state sequence.
    A : np.ndarray, shape (n, n)
        Estimated transition matrix.
    B : np.ndarray, shape (n, m)
        Estimated input matrix.
    lam : float
        Smoothness weight (unused here; see note above).

    Returns
    -------
    H : scipy.sparse.csr_matrix, shape (N*n, N*m)
        Block-diagonal data operator.
    d : np.ndarray, shape (N*n,)
        Stacked one-step increments.
    D : scipy.sparse.csr_matrix, shape ((N-1)*m, N*m)
        First-difference operator (empty 0-row matrix when ``N < 2``).
    """
    del lam  # applied in _solve_smoothed_inputs; kept for paired-API symmetry
    T, n = x_filt.shape
    m = B.shape[1]
    N = T - 1  # number of inputs u_0 .. u_{T-2} (one per state increment)
    if N < 1:
        raise ValueError(f"need T >= 2 to form increments; got T={T}")

    # Stacked increments d_t = x_{t+1} - A x_t, t = 0 .. N-1.
    incr = x_filt[1:] - x_filt[:-1] @ A.T       # (N, n)
    d = incr.reshape(-1)                        # (N*n,)

    # Data operator: H = I_N (x) B  (block-diagonal, N copies of B).
    H = sp.kron(sp.identity(N, format="csr"), sp.csr_matrix(B), format="csr")

    # First-difference operator on the stacked inputs: D = diff(N) (x) I_m.
    if N >= 2:
        diff = sp.diags(
            [-np.ones(N - 1), np.ones(N - 1)],
            offsets=[0, 1],
            shape=(N - 1, N),
            format="csr",
        )
        D = sp.kron(diff, sp.identity(m, format="csr"), format="csr")
    else:
        D = sp.csr_matrix((0, N * m))
    return H.tocsr(), d, D.tocsr()


def _solve_smoothed_inputs(
    H: sp.spmatrix, d: np.ndarray, D: sp.spmatrix, lam: float
) -> np.ndarray:
    """Solve the block-tridiagonal normal equations (3) for the stacked input.

    Forms ``M = H^T H + lam D^T D + jitter I`` and right-hand side
    ``rhs = H^T d`` and solves ``M u = rhs`` with a sparse direct solver.  ``M``
    is block-tridiagonal (block size ``m``); it is assembled and factored
    sparsely so memory stays ``O(T m^2)`` rather than ``O(T^2 m^2)``.  A tiny
    ``jitter I`` keeps ``M`` positive-definite even when ``B^T B`` is rank
    deficient (e.g. an all-zero or single-direction ``B_hat``).

    Parameters
    ----------
    H : scipy.sparse matrix, shape (N*n, N*m)
        Block-diagonal data operator from :func:`_build_smoothness_system`.
    d : np.ndarray, shape (N*n,)
        Stacked one-step increments.
    D : scipy.sparse matrix, shape ((N-1)*m, N*m)
        First-difference operator.
    lam : float
        Non-negative smoothness weight.

    Returns
    -------
    np.ndarray, shape (N*m,)
        Flattened solution ``[u_0; u_1; ...; u_{N-1}]``.
    """
    n_var = H.shape[1]                          # = N * m
    HtH = (H.T @ H).tocsr()
    M = HtH + _SOLVE_JITTER * sp.identity(n_var, format="csr")
    if lam > 0.0 and D.shape[0] > 0:
        M = M + lam * (D.T @ D)
    rhs = np.asarray(H.T @ d).ravel()
    u_flat = spsolve(M.tocsr(), rhs)
    return np.asarray(u_flat).ravel()


def _smoothed_inputs(
    x_filt: np.ndarray, A: np.ndarray, B: np.ndarray, lam: float
) -> np.ndarray:
    """Smoothness-penalised input estimate, padded to ``T`` rows.

    Builds (:func:`_build_smoothness_system`) and solves
    (:func:`_solve_smoothed_inputs`) problem (2), then reshapes the stacked
    solution to ``(T-1, m)`` and pads the final step ``u_{T-1} = u_{T-2}`` — the
    same end convention as :func:`estimator._estimate_inputs`.

    With ``lam = 0`` the blocks decouple and the solution reduces to the
    per-timestep ridge ``(B^T B + jitter I)^{-1} B^T (x_{t+1} - A x_t)``, which
    equals the pseudoinverse back-out (1) up to the ``1e-8`` jitter (exactly so
    when ``B`` has orthonormal columns, as the identified ``B_hat`` does).

    Parameters
    ----------
    x_filt : np.ndarray, shape (T, n)
        Filtered latent-state sequence.
    A : np.ndarray, shape (n, n)
        Estimated transition matrix.
    B : np.ndarray, shape (n, m)
        Estimated input matrix.
    lam : float
        Smoothness weight (>= 0).

    Returns
    -------
    np.ndarray, shape (T, m)
        Estimated input sequence (subspace-invariant up to a linear map).
    """
    T = x_filt.shape[0]
    m = B.shape[1]
    u = np.empty((T, m))
    if T < 2:
        u[:] = 0.0
        return u
    H, d, D = _build_smoothness_system(x_filt, A, B, lam)
    u_flat = _solve_smoothed_inputs(H, d, D, lam)
    u[:-1] = u_flat.reshape(T - 1, m)
    u[-1] = u[-2]                                # pad final step
    return u


# ---------------------------------------------------------------------------
# Identification + filtering (reused verbatim) and the new back-out
# ---------------------------------------------------------------------------
def _identify_and_filter(
    Y: np.ndarray, n: int, m: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the *same* two-pass identification + Kalman filtering as the primary.

    Mirrors the identification and filtering of :func:`estimator.fit_and_filter`
    exactly — centring, subspace ID via :func:`estimator._fit_subspace_model`,
    then ``_N_FILTER_PASSES`` Kalman passes (pass 1 with ``u = 0``, later passes
    re-filtering with the *pseudoinverse* input plugged into the prediction step,
    via :func:`estimator._estimate_inputs`).  The filtered state ``x_hat`` is
    therefore identical to the primary estimator's; only the back-out that
    follows differs.  No identification code is duplicated.

    Parameters
    ----------
    Y : np.ndarray, shape (T, p)
        Observation sequence.
    n, m : int
        Latent and input dimensions.

    Returns
    -------
    A, B, C, Q, R : np.ndarray
        Identified system matrices.
    x_hat : np.ndarray, shape (T, n)
        Filtered latent states (identical to the primary estimator's).
    y_mean : np.ndarray, shape (p,)
        Column mean removed before identification.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError(f"observation must be 2-D (T, p); got shape {Y.shape}")
    T = Y.shape[0]

    y_mean = Y.mean(axis=0)
    Yc = Y - y_mean

    A, B, C, Q, R = _fit_subspace_model(Yc, n, m)

    # Pass 1: filter with u = 0; later passes plug the pseudoinverse input into
    # the prediction step (identical to estimator.fit_and_filter).
    x_hat = _kalman_filter(Yc, A, B, C, Q, R, u=None)
    for _ in range(_N_FILTER_PASSES - 1):
        u_pinv = _estimate_inputs(x_hat, A, B)
        x_hat = _kalman_filter(Yc, A, B, C, Q, R, u=u_pinv)

    return A, B, C, Q, R, x_hat, y_mean


def fit_and_filter(observation: np.ndarray, LatentDim: int, InputDim: int) -> Dict[str, np.ndarray]:
    """Full pipeline with the smoothness-penalised back-out (analysis entry point).

    Identical identification and filtering to :func:`estimator.fit_and_filter`,
    but the returned ``inputs`` come from the joint smoothness-penalised solve
    (2) instead of the per-timestep pseudoinverse.  The chosen ``lam`` is stored
    in :data:`LAST_LAMBDA` and also returned under the ``"lam"`` key.

    Parameters
    ----------
    observation : np.ndarray, shape (T, p)
        Observation sequence.
    LatentDim : int
        Latent dimension ``n``.
    InputDim : int
        Input dimension ``m``.

    Returns
    -------
    dict
        Keys: ``A`` (n,n), ``B`` (n,m), ``C`` (p,n), ``Q`` (n,n), ``R`` (p,p),
        ``latent`` (T,n) filtered states, ``inputs`` (T,m) smoothed inputs,
        ``y_mean`` (p,), ``lam`` (float) the chosen smoothness weight.
    """
    global LAST_LAMBDA
    n, m = int(LatentDim), int(InputDim)

    A, B, C, Q, R, x_hat, y_mean = _identify_and_filter(observation, n, m)

    lam = _default_lambda(Q, B)
    LAST_LAMBDA = lam
    u_hat = _smoothed_inputs(x_hat, A, B, lam)

    return {
        "A": A, "B": B, "C": C, "Q": Q, "R": R,
        "latent": x_hat, "inputs": u_hat, "y_mean": y_mean, "lam": lam,
    }


def estimate_latent_and_input(
    observation: np.ndarray,
    LatentDim: int,
    InputDim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate latent states and inputs (smoothness-penalised back-out) — API.

    Same contract as :func:`estimator.estimate_latent_and_input`: identical
    identification and Kalman filtering, but the input is recovered by the joint
    smoothness-penalised solve (2).  ``lam`` is chosen automatically from the
    identified model at entry (no call-time hyperparameter; see
    :func:`_default_lambda`).

    Parameters
    ----------
    observation : np.ndarray, shape (T, p)
        Observed neural activity (time points x neurons).
    LatentDim : int
        Dimensionality of the latent state space to return.
    InputDim : int
        Dimensionality of the input space to return.

    Returns
    -------
    latent_states : np.ndarray, shape (T, LatentDim)
        Estimated latent states (Kalman-filtered, in the internal subspace basis).
    inputs : np.ndarray, shape (T, InputDim)
        Estimated inputs (identifiable up to a linear map; see module note).

    Notes
    -----
    Contractually guaranteed never to crash and to always return arrays of shape
    ``(T, LatentDim)`` / ``(T, InputDim)``.  On any failure (or a non-finite
    result) it emits a ``warnings.warn`` and falls back to zero-filled outputs of
    the correct shape, so the demonstrator's test harness never raises.
    """
    Y = np.asarray(observation, dtype=float)
    T = Y.shape[0] if Y.ndim >= 1 else 0
    n, m = int(LatentDim), int(InputDim)

    try:
        result = fit_and_filter(Y, n, m)
        latent = np.asarray(result["latent"], dtype=float)
        inputs = np.asarray(result["inputs"], dtype=float)

        if latent.shape != (T, n) or inputs.shape != (T, m):
            raise ValueError(
                f"internal shape mismatch: latent {latent.shape}, inputs "
                f"{inputs.shape}; expected {(T, n)}, {(T, m)}"
            )
        if not (np.all(np.isfinite(latent)) and np.all(np.isfinite(inputs))):
            raise FloatingPointError("estimate contained NaN or inf")
        return latent, inputs
    except Exception as exc:  # never crash the demonstrator's harness
        warnings.warn(
            f"estimate_latent_and_input (smooth): falling back to zeros "
            f"({type(exc).__name__}: {exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.zeros((T, n)), np.zeros((T, m))


# ---------------------------------------------------------------------------
# Self-test (run `python estimator_smooth.py`)
# ---------------------------------------------------------------------------
def _self_test() -> None:
    """Compare the smoothed back-out with the primary estimator on mixed_input."""
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week 1"))
    import Simulator as sim  # noqa: E402
    import estimator  # noqa: E402

    def best_linear_map(src, tgt):
        X = np.hstack([src, np.ones((len(src), 1))])
        W, *_ = np.linalg.lstsq(X, tgt, rcond=None)
        return X @ W

    def input_r2(u_hat, u_true):
        pred = best_linear_map(u_hat, u_true)
        ss_res = np.sum((u_true - pred) ** 2)
        ss_tot = np.sum((u_true - u_true.mean(0)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    def recon_rmse(y, model):
        y_hat = model["latent"] @ model["C"].T + model["y_mean"]
        return float(np.sqrt(np.mean((y - y_hat) ** 2)))

    T = 500
    system = sim.default_neural_system(seed=0, obs_dim=16)
    u_seq = sim.mixed_input(T, system.input_dim, seed=0)
    y = system.simulate(T, U=u_seq)["y"]
    u_true = system.simulate(T, U=u_seq)["u"]

    m_base = estimator.fit_and_filter(y, 4, 2)
    m_smooth = fit_and_filter(y, 4, 2)

    print("default_neural_system, mixed_input (T=500):")
    print(f"  lambda (auto) = {m_smooth['lam']:.6g}   "
          f"[= alpha * tr(Q)/tr(BB^T), alpha={_ALPHA}]")
    print(f"  recon RMSE : primary={recon_rmse(y, m_base):.4f}   "
          f"smooth={recon_rmse(y, m_smooth):.4f}")
    print(f"  input R^2  : primary={input_r2(m_base['inputs'], u_true):.4f}   "
          f"smooth={input_r2(m_smooth['inputs'], u_true):.4f}   "
          f"(primary baseline 0.186)")


if __name__ == "__main__":
    _self_test()
