"""Week 2 comparison estimator — iterative coordinate-ascent maximum likelihood.

This module is a third **drop-in alternative** to :mod:`estimator` (the subspace
ID + Kalman filter baseline) and :mod:`estimator_smooth` (the smoothness-penalised
back-out).  It refines *all* parameters and the latent trajectory jointly by
iterating a Kalman **smoother** with closed-form parameter re-estimation, instead
of the baseline's one-shot identification.

Motivation (``interim_report.md`` §3 / §4)
------------------------------------------
The baseline identifies ``(A, B, C, Q, R)`` once, then runs a one-pass filter and
a per-timestep input back-out.  This pipeline is biased when the true input is
*autocorrelated* (sinusoidal, mixed): an unidentifiable rank-``m`` component of
the input is absorbed into ``A_hat``, and the error compounds
(``A_hat`` -> states -> ``B_hat`` -> ``u_hat``).  Jointly refining the parameters
and the smoothed state corrects ``A_hat`` and ``B_hat`` and, in turn, the input.

Algorithm — NOT strict EM
-------------------------
We treat the input ``u`` as a **deterministic unknown** (not a latent random
variable with a prior that is marginalised out).  The scheme is therefore an
**iterative coordinate-ascent maximum-likelihood** procedure — closely related to
EM, of the Expectation/Conditional-Maximisation (ECM) family — that ascends the
data log-likelihood ``log p(y | u, theta)`` by alternating:

* **E-step.**  With the current ``(theta, u)`` run a Rauch-Tung-Striebel smoother
  (forward Kalman filter + backward pass) to get the posterior smoothed means
  ``x_hat_t``, covariances ``P_t`` and lag-one cross-covariances ``P_{t,t-1}``.
* **CM-step over u** (``theta`` fixed).  The exact maximiser of the expected
  complete-data log-likelihood along the ``u`` coordinate is the
  ``Q``-**weighted** back-out
  ``u_t = (B^T Q^-1 B + eps I)^-1 B^T Q^-1 (x_hat_{t+1} - A x_hat_t)``
  (see :func:`_estimate_inputs_q_weighted`).  The unweighted pseudoinverse
  ``pinv(B)`` coincides with it only when ``B`` has orthonormal columns or
  ``Q`` is a scalar multiple of the identity — neither of which need hold after
  the M-step updates ``B`` — so the weighted form is required for the
  log-likelihood to be monotone non-decreasing.
* **CM-step over theta** (``u`` fixed).  Closed-form ML re-estimation of
  ``(A, B, C, Q, R)`` from the smoothed moments (:func:`_m_step`), with the
  posterior-covariance corrections that the smoother provides.

Because each sub-step is the *exact* maximiser along its coordinate, the data
log-likelihood is monotone non-decreasing (up to ~1e-8 numerical noise per step).

Initialisation is **not** random — coordinate ascent has local optima and only
the subspace-ID start (:func:`estimator._fit_subspace_model`) reliably converges
to a good optimum.

Convergence and a known limitation
----------------------------------
Treating ``u`` as a *free deterministic unknown* gives it ``T x m`` parameters,
which is heavily over-parameterised: the input keeps absorbing observation
residuals, so the process covariance ``Q`` drifts slowly and the log-likelihood
climbs almost linearly **without** ever reaching a strict relative tolerance.
This drift is benign — the output-determining matrices ``(A, B, C)``, and hence
the recovered states and inputs, stabilise within a handful of iterations.  We
therefore declare convergence on the relative change of ``(A, B, C)`` (in
addition to the strict log-likelihood test), and document the ``Q``/log-
likelihood ridge rather than hiding it.  Empirically the joint refinement yields
only a **modest** input-recovery gain over the subspace-ID baseline in the
low-observation-noise regime (where the baseline is already near-optimal); the
gain grows as observation noise increases (see ``week2_em_analysis.ipynb``).  A
*state-augmented* variant that treats ``u`` as a stochastic latent with a
learnable AR prior (rather than a free deterministic parameter) would regularise
this over-parameterisation and is left as future work.

Initial-state convention
------------------------
The model has no initial-state prior in the simulator (``x_0`` is fixed), but the
innovation-form log-likelihood needs a prediction ``x_{0|-1}, P_{0|-1}`` at
``t = 0``.  We use a standard filter with a prior ``x_{0|-1} = pinv(C) y_0``,
``P_{0|-1} = I`` computed **once** from the initial subspace model and held
**fixed** across every iteration (it is *not* re-estimated).  A fixed prior makes
the ``t = 0`` term a proper Gaussian marginal whose contribution is constant under
the M-step, so coordinate-ascent monotonicity is exact.

Public interface (identical to :mod:`estimator`)::

    estimate_latent_and_input(observation, LatentDim, InputDim)
        -> (latent_states (T, LatentDim), inputs (T, InputDim))

The latent states returned are the **smoothed** states (offline recovery uses the
whole ``y_{1:T}``, so the smoother — not a causal filter — is the right tool).
The returned ``inputs`` use the *unweighted* per-timestep back-out
(:func:`estimator._estimate_inputs`), for consistency with the baseline and
smoothness estimators in the comparison; only the EM-internal updates use the
``Q``-weighted form.  The same ``GL(n)`` similarity / linear-map identifiability
caveats apply (see the :mod:`estimator` module docstring): compare through
reconstructed observations ``y_hat = C_hat x_hat`` and best-linear-fit input
recovery, never by direct subtraction.

Dependencies: ``numpy`` (and the std lib) only, matching :mod:`estimator`.  The
identification / filtering helpers are imported from :mod:`estimator` so that code
is reused, not duplicated.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np

# Reuse the existing identification + filtering machinery verbatim (do not
# duplicate it).  Only the smoother / coordinate-ascent loop below is new.
from estimator import (
    _COV_SHRINKAGE,
    _INV_JITTER,
    _estimate_inputs,
    _fit_subspace_model,
    _kalman_filter,
    _regularize_cov,
    _safe_inverse,
)
from estimator import estimate_latent_and_input as _baseline_estimate

# --- Fixed hyperparameters (never tuned at call time) ----------------------
_MAX_ITER = 200      # coordinate-ascent iteration cap
_TOL = 1e-7          # relative log-likelihood change for convergence (strict)
_PARAM_TOL = 1e-4    # relative change in (A, B, C) for convergence (see _em_fit)
_EPSILON = 1e-6      # covariance shrinkage / regulariser in the M-step


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------
def _logdet_and_inv(S: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return ``(log|S|, S^-1)`` for a symmetric PD matrix, computed consistently.

    Both quantities are taken from the *same* jitter-regularised, symmetrised
    matrix so the innovation log-likelihood is a coherent function of the
    parameters (essential for the monotonicity guarantee).  Falls back to the
    pseudoinverse if the regularised matrix is still numerically singular.
    """
    p = S.shape[0]
    S_reg = 0.5 * (S + S.T) + _INV_JITTER * np.eye(p)
    sign, logdet = np.linalg.slogdet(S_reg)
    if sign <= 0 or not np.isfinite(logdet):
        # Extremely ill-conditioned: fall back to an eigenvalue floor.
        w, V = np.linalg.eigh(S_reg)
        w = np.clip(w, _INV_JITTER, None)
        logdet = float(np.sum(np.log(w)))
        Sinv = (V * (1.0 / w)) @ V.T
        return logdet, Sinv
    try:
        Sinv = np.linalg.inv(S_reg)
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(S_reg)
    return float(logdet), Sinv


def _reg_inv(M: np.ndarray, epsilon: float) -> np.ndarray:
    """Invert ``M + epsilon I`` (symmetric/square), with a pseudoinverse fallback."""
    k = M.shape[0]
    M_reg = M + epsilon * np.eye(k)
    try:
        return np.linalg.inv(M_reg)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M_reg)


def _rel_change(new: np.ndarray, old: np.ndarray) -> float:
    """Relative Frobenius change ``||new - old|| / ||old||`` (||old||->1 if ~0)."""
    denom = np.linalg.norm(old)
    return float(np.linalg.norm(new - old) / denom) if denom > 1e-12 else float(
        np.linalg.norm(new - old)
    )


# ---------------------------------------------------------------------------
# Forward filter (shared by the smoother and the standalone log-likelihood)
# ---------------------------------------------------------------------------
def _forward_filter(
    y: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    u: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Standard Kalman filter with an explicit prior ``x_{0|-1}=x0, P_{0|-1}=P0``.

    Input convention matches :func:`estimator._kalman_filter`: ``u[t-1]`` drives
    the transition into ``x_t``.  Accumulates the innovation-form data
    log-likelihood so callers (smoother, convergence check) get it for free.

    Returns
    -------
    x_filt : (T, n)        filtered means ``x_{t|t}``
    P_filt : (T, n, n)     filtered covariances ``P_{t|t}``
    x_pred : (T, n)        one-step predicted means ``x_{t|t-1}`` (t=0 -> prior)
    P_pred : (T, n, n)     one-step predicted covariances ``P_{t|t-1}``
    gains  : (T, n, p)     Kalman gains ``K_t``
    loglik : float         sum_t -0.5(log|S_t| + e_t^T S_t^-1 e_t + p log 2pi)
    """
    T, p = y.shape
    n = A.shape[0]
    eye_n = np.eye(n)

    x_filt = np.empty((T, n))
    P_filt = np.empty((T, n, n))
    x_pred = np.empty((T, n))
    P_pred = np.empty((T, n, n))
    gains = np.empty((T, n, p))
    loglik = 0.0
    log2pi = np.log(2.0 * np.pi)

    for t in range(T):
        if t == 0:
            xp = np.asarray(x0, dtype=float)
            Pp = np.asarray(P0, dtype=float)
        else:
            xp = A @ x_filt[t - 1] + B @ u[t - 1]
            Pp = A @ P_filt[t - 1] @ A.T + Q
        x_pred[t] = xp
        P_pred[t] = Pp

        S = C @ Pp @ C.T + R
        logdet_S, Sinv = _logdet_and_inv(S)
        K = Pp @ C.T @ Sinv
        innovation = y[t] - C @ xp
        x_filt[t] = xp + K @ innovation
        P_filt[t] = (eye_n - K @ C) @ Pp
        gains[t] = K

        loglik += -0.5 * (logdet_S + innovation @ Sinv @ innovation + p * log2pi)

    return x_filt, P_filt, x_pred, P_pred, gains, float(loglik)


def _default_prior(
    y: np.ndarray, C: np.ndarray, x0: Optional[np.ndarray], P0: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve the initial prior ``(x0, P0)``; defaults to ``(pinv(C) y_0, I)``."""
    n = C.shape[1]
    if x0 is None:
        x0 = np.linalg.pinv(C) @ y[0]
    if P0 is None:
        P0 = np.eye(n)
    return np.asarray(x0, dtype=float), np.asarray(P0, dtype=float)


# ---------------------------------------------------------------------------
# Rauch-Tung-Striebel smoother (with lag-one cross-covariances)
# ---------------------------------------------------------------------------
def _kalman_smoother(
    y: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    u: np.ndarray,
    epsilon: float = _EPSILON,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """RTS smoother for the linear-Gaussian model with deterministic input ``u``.

    Runs :func:`_forward_filter`, then the backward recursion for
    ``t = T-2, ..., 0``::

        J_t      = P_{t|t} A^T (P_{t+1|t})^-1
        x_t^s    = x_{t|t} + J_t (x_{t+1}^s - x_{t+1|t})
        P_t^s    = P_{t|t} + J_t (P_{t+1}^s - P_{t+1|t}) J_t^T

    initialised at ``x_{T-1}^s = x_{T-1|T-1}``, ``P_{T-1}^s = P_{T-1|T-1}``, and the
    lag-one cross-covariance ``P_{t,t-1}^s = cov(x_t, x_{t-1} | y_{1:T})``::

        P_{T-1,T-2}^s = (I - K_{T-1} C) A P_{T-2|T-2}
        P_{t,t-1}^s   = P_{t|t} J_{t-1}^T + J_t (P_{t+1,t}^s - A P_{t|t}) J_{t-1}^T

    for ``t = T-2, ..., 1`` (``K_{T-1}`` is the final forward gain).

    Parameters
    ----------
    y : (T, p)
        Observation sequence (centred when called from the EM loop; raw for the
        true-model smoother tests).
    A, B, C, Q, R : np.ndarray
        System matrices.
    u : (T, m)
        Input sequence (``u[t-1]`` drives ``x_t``).
    epsilon : float
        Unused here directly (kept for signature symmetry / future jitter).
    x0, P0 : np.ndarray, optional
        Initial prior; defaults to ``(pinv(C) y_0, I)``.

    Returns
    -------
    x_smooth : (T, n)
    P_smooth : (T, n, n)
    P_lag    : (T, n, n)
        Lag-one cross-covariances; ``P_lag[t] = P_{t,t-1}^s`` for ``t = 1..T-1``
        and ``P_lag[0] = 0``.
    loglik   : float
        Data log-likelihood from the forward pass (so the EM loop tracks
        monotonicity without a second forward pass).
    """
    y = np.asarray(y, dtype=float)
    T, p = y.shape
    n = A.shape[0]
    x0v, P0v = _default_prior(y, C, x0, P0)

    x_filt, P_filt, x_pred, P_pred, gains, loglik = _forward_filter(
        y, A, B, C, Q, R, u, x0v, P0v
    )

    x_smooth = np.empty((T, n))
    P_smooth = np.empty((T, n, n))
    P_lag = np.zeros((T, n, n))
    J = np.zeros((T, n, n))  # smoother gains; J[t] used for t = 0..T-2

    x_smooth[T - 1] = x_filt[T - 1]
    P_smooth[T - 1] = P_filt[T - 1]

    for t in range(T - 2, -1, -1):
        Pp_next_inv = _safe_inverse(P_pred[t + 1])
        Jt = P_filt[t] @ A.T @ Pp_next_inv
        J[t] = Jt
        x_smooth[t] = x_filt[t] + Jt @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + Jt @ (P_smooth[t + 1] - P_pred[t + 1]) @ Jt.T

    # Lag-one cross-covariances (the most error-prone part of EM).
    if T >= 2:
        K_last = gains[T - 1]
        P_lag[T - 1] = (np.eye(n) - K_last @ C) @ A @ P_filt[T - 2]
        for t in range(T - 2, 0, -1):
            P_lag[t] = (
                P_filt[t] @ J[t - 1].T
                + J[t] @ (P_lag[t + 1] - A @ P_filt[t]) @ J[t - 1].T
            )

    return x_smooth, P_smooth, P_lag, loglik


def _log_likelihood(
    y: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    u: np.ndarray,
    epsilon: float = _EPSILON,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
) -> float:
    """Data log-likelihood via the innovations form (thin :func:`_forward_filter`).

    Kept as a standalone, individually-testable function (Test 3); the EM loop
    itself reuses the log-likelihood returned by :func:`_kalman_smoother`'s own
    forward pass, so it does not call this in the inner loop.
    """
    y = np.asarray(y, dtype=float)
    x0v, P0v = _default_prior(y, C, x0, P0)
    _, _, _, _, _, loglik = _forward_filter(y, A, B, C, Q, R, u, x0v, P0v)
    return loglik


# ---------------------------------------------------------------------------
# Q-weighted input back-out (the exact CM-step maximiser over u)
# ---------------------------------------------------------------------------
def _estimate_inputs_q_weighted(
    X: np.ndarray, A: np.ndarray, B: np.ndarray, Q: np.ndarray, epsilon: float = _EPSILON
) -> np.ndarray:
    """Exact along-``u`` maximiser of the expected complete-data log-likelihood.

    Solves the ``Q``-weighted least-squares back-out

        u_t = (B^T Q^-1 B + eps I)^-1 B^T Q^-1 (x_{t+1} - A x_t),  t = 0..T-2

    and pads the final step (``u_{T-1} = u_{T-2}``) to ``T`` rows.  This is the
    quantity that makes the coordinate-ascent monotone (see the module docstring);
    the unweighted :func:`estimator._estimate_inputs` is only its special case
    when ``B`` is orthonormal or ``Q`` is isotropic.

    Parameters
    ----------
    X : (T, n)
        Smoothed latent-state sequence.
    A : (n, n)
    B : (n, m)
    Q : (n, n)
        Process-noise covariance (the weighting matrix).
    epsilon : float
        Ridge added to ``B^T Q^-1 B`` for a rank-deficient / zero ``B``.

    Returns
    -------
    np.ndarray, shape (T, m)
        Estimated input sequence.
    """
    T, n = X.shape
    m = B.shape[1]
    u = np.empty((T, m))
    if T < 2:
        u[:] = 0.0
        return u
    Qinv = _safe_inverse(Q)
    BtQinv = B.T @ Qinv                       # (m, n)
    G = _reg_inv(BtQinv @ B, epsilon) @ BtQinv  # (m, n): weighted pseudoinverse
    increments = X[1:] - X[:-1] @ A.T         # (T-1, n): x_{t+1} - A x_t
    u[:-1] = increments @ G.T                 # (T-1, m)
    u[-1] = u[-2]                             # pad final step
    return u


# ---------------------------------------------------------------------------
# M-step (CM-step over theta): closed-form ML re-estimation
# ---------------------------------------------------------------------------
def _m_step(
    y: np.ndarray,
    x_smooth: np.ndarray,
    P_smooth: np.ndarray,
    P_lag: np.ndarray,
    u: np.ndarray,
    epsilon: float = _EPSILON,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One closed-form parameter update from the smoothed moments.

    Implements the standard linear-Gaussian M-step (Shumway-Stoffer / Ghahramani-
    Hinton) with the posterior-covariance corrections supplied by the smoother.
    Accumulated moments (``x_t = x_t^s``, ``P_t = P_t^s``)::

        Sxx   = sum_{0..T-1} (x_t x_t^T + P_t)
        Syx   = sum_{0..T-1} y_t x_t^T
        Sx_x_ = sum_{1..T-1} (x_{t-1} x_{t-1}^T + P_{t-1})
        Sxx_  = sum_{1..T-1} (x_t x_{t-1}^T + P_{t,t-1})
        Sx_u_ = sum_{1..T-1} x_{t-1} u_{t-1}^T
        Sxu_  = sum_{1..T-1} x_t u_{t-1}^T
        Suu   = sum_{0..T-2} u_t u_t^T

    Joint transition/input regression and emission update::

        [A B] = [Sxx_  Sxu_] ([[Sx_x_, Sx_u_], [Sx_u_^T, Suu]] + eps I)^-1
        C     = Syx (Sxx + eps I)^-1

    Process / observation covariances (with the essential ``P``-corrections)::

        Q = (1/(T-1)) sum_{1..T-1}[ r_t r_t^T + P_t - P_{t,t-1} A^T
                                    - A P_{t,t-1}^T + A P_{t-1} A^T ] + eps I
            with r_t = x_t - A x_{t-1} - B u_{t-1}    (A, B the NEW ones)
        R = (1/T) sum_{0..T-1}[ (y_t - C x_t)(y_t - C x_t)^T + C P_t C^T ] + eps I
            (C the NEW one)

    Returns ``(A_new, B_new, C_new, Q_new, R_new)``.
    """
    y = np.asarray(y, dtype=float)
    T, n = x_smooth.shape
    m = u.shape[1]

    P_all_sum = P_smooth.sum(axis=0)                      # sum_{0..T-1} P_t
    Sxx = x_smooth.T @ x_smooth + P_all_sum               # (n, n)
    Syx = y.T @ x_smooth                                  # (p, n)

    x_cur = x_smooth[1:]                                  # x_t,   t = 1..T-1
    x_prev = x_smooth[:-1]                                # x_{t-1}
    u_prev = u[:-1]                                       # u_{t-1}
    P_prev_sum = P_smooth[:-1].sum(axis=0)                # sum_{1..T-1} P_{t-1}
    P_cur_sum = P_smooth[1:].sum(axis=0)                  # sum_{1..T-1} P_t
    P_lag_sum = P_lag[1:].sum(axis=0)                     # sum_{1..T-1} P_{t,t-1}

    Sx_x_ = x_prev.T @ x_prev + P_prev_sum                # (n, n)
    Sxx_ = x_cur.T @ x_prev + P_lag_sum                   # (n, n)
    Sx_u_ = x_prev.T @ u_prev                             # (n, m)
    Sxu_ = x_cur.T @ u_prev                               # (n, m)
    Suu = u_prev.T @ u_prev                               # (m, m)

    # Joint [A B] update (regression of x_t onto [x_{t-1}; u_{t-1}]).
    top = np.hstack([Sxx_, Sxu_])                         # (n, n+m)
    block = np.block([[Sx_x_, Sx_u_], [Sx_u_.T, Suu]])    # (n+m, n+m)
    AB = top @ _reg_inv(block, epsilon)
    A_new = AB[:, :n]
    B_new = AB[:, n:]

    # Emission update.
    C_new = Syx @ _reg_inv(Sxx, epsilon)

    # Process-noise covariance with smoother corrections.
    resid = x_cur - x_prev @ A_new.T - u_prev @ B_new.T   # (T-1, n)
    Q_data = resid.T @ resid
    Q_corr = (
        P_cur_sum
        - P_lag_sum @ A_new.T
        - A_new @ P_lag_sum.T
        + A_new @ P_prev_sum @ A_new.T
    )
    denom_q = max(T - 1, 1)
    Q_new = _regularize_cov((Q_data + Q_corr) / denom_q, epsilon)

    # Observation-noise covariance with smoother corrections.
    resid_y = y - x_smooth @ C_new.T                      # (T, p)
    R_data = resid_y.T @ resid_y
    R_corr = C_new @ P_all_sum @ C_new.T
    R_new = _regularize_cov((R_data + R_corr) / T, epsilon)

    return A_new, B_new, C_new, Q_new, R_new


# ---------------------------------------------------------------------------
# Full coordinate-ascent loop
# ---------------------------------------------------------------------------
def _em_fit(
    y: np.ndarray, n: int, m: int, max_iter: int = _MAX_ITER, tol: float = _TOL,
    epsilon: float = _EPSILON, param_tol: float = _PARAM_TOL,
) -> Dict[str, object]:
    """Run the full coordinate-ascent maximum-likelihood loop on **centred** ``y``.

    Initialised from subspace ID (never randomly).  Each iteration: smoother
    (E-step, also yields the current log-likelihood) -> ``Q``-weighted input
    update (CM-step over ``u``) -> closed-form parameter update (CM-step over
    ``theta``).

    Convergence (dual criterion).  Stops when **either** the relative
    log-likelihood change drops below ``tol`` (1e-7; the strict criterion) **or**
    the relative Frobenius change of the *output-determining* matrices
    ``(A, B, C)`` drops below ``param_tol`` (1e-4).  The second criterion is the
    one that normally fires: under the deterministic-``u`` variant the free input
    (``T x m`` parameters) keeps reshaping the dynamics residual, so the process
    covariance ``Q`` exhibits a slow, non-decaying gauge drift and the
    log-likelihood climbs almost linearly without ever reaching the strict ``tol``
    (a documented property of the variant — see the module docstring / notebook).
    That ridge does **not** move the recovered states or inputs: ``A, B, C`` — and
    hence ``x_smooth`` and the input recovery — stabilise within a handful of
    iterations, which is exactly what ``param_tol`` detects.  ``Q``/``R`` are
    excluded from the convergence test for this reason.  Set ``param_tol = 0`` to
    disable the parameter criterion (used to expose the ridge in the notebook).

    A final smoother pass with the converged parameters produces the returned
    smoothed states; the returned input uses the unweighted public back-out for
    comparison-study consistency.

    Returns
    -------
    dict with keys ``A, B, C, Q, R`` (params), ``x_smooth`` (T, n),
    ``u`` (T, m, unweighted public back-out), ``ll_history`` (np.ndarray),
    ``u_history`` (list of (T, m), unweighted per-iteration estimates for the
    notebook diagnostic) and ``n_iter`` (int).
    """
    y = np.asarray(y, dtype=float)
    T = y.shape[0]

    # --- Subspace-ID initialisation (critical: never random) -------------
    A, B, C, Q, R = _fit_subspace_model(y, n, m)
    x_init = _kalman_filter(y, A, B, C, Q, R, u=None)
    u = _estimate_inputs(x_init, A, B)

    # Fixed initial prior (held constant across iterations -> monotonicity).
    m0 = np.linalg.pinv(C) @ y[0]
    P0 = np.eye(n)

    ll_history = []
    u_history = []
    ll_prev = -np.inf
    n_iter = 0

    for k in range(max_iter):
        x_smooth, P_smooth, P_lag, ll = _kalman_smoother(
            y, A, B, C, Q, R, u, epsilon=epsilon, x0=m0, P0=P0
        )
        ll_history.append(ll)
        # Report-style (unweighted) estimate at the current params, for the
        # notebook's per-iteration R^2 diagnostic only.
        u_history.append(_estimate_inputs(x_smooth, A, B))
        n_iter = k + 1

        ll_converged = (
            np.isfinite(ll) and np.isfinite(ll_prev)
            and abs(ll - ll_prev) / max(abs(ll_prev), 1.0) < tol
        )
        ll_prev = ll

        # CM-step over u (exact maximiser), then CM-step over theta.
        u = _estimate_inputs_q_weighted(x_smooth, A, B, Q, epsilon=epsilon)
        A_new, B_new, C_new, Q_new, R_new = _m_step(
            y, x_smooth, P_smooth, P_lag, u, epsilon=epsilon
        )
        param_change = max(
            _rel_change(A_new, A), _rel_change(B_new, B), _rel_change(C_new, C)
        )
        A, B, C, Q, R = A_new, B_new, C_new, Q_new, R_new

        if ll_converged or (param_tol > 0.0 and param_change < param_tol):
            break

    # Final smoother pass with the converged parameters (offline -> smoother).
    x_final, _, _, _ = _kalman_smoother(
        y, A, B, C, Q, R, u, epsilon=epsilon, x0=m0, P0=P0
    )
    u_final = _estimate_inputs(x_final, A, B)  # unweighted public back-out

    return {
        "A": A, "B": B, "C": C, "Q": Q, "R": R,
        "x_smooth": x_final, "u": u_final,
        "ll_history": np.asarray(ll_history, dtype=float),
        "u_history": u_history,
        "n_iter": n_iter,
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def fit_and_filter(observation: np.ndarray, LatentDim: int, InputDim: int) -> Dict[str, object]:
    """Run the full coordinate-ascent pipeline and return every quantity.

    Analysis-friendly entry point (mirrors :func:`estimator.fit_and_filter` /
    :func:`estimator_smooth.fit_and_filter`): centres ``y``, runs :func:`_em_fit`,
    and exposes the converged matrices, smoothed states, input estimate and the
    observation mean so reconstructions ``y_hat = latent @ C^T + y_mean`` can be
    formed.

    Returns
    -------
    dict
        Keys: ``A`` (n,n), ``B`` (n,m), ``C`` (p,n), ``Q`` (n,n), ``R`` (p,p),
        ``latent`` (T,n) smoothed states, ``inputs`` (T,m) estimated inputs,
        ``y_mean`` (p,), ``ll_history`` (np.ndarray), ``u_history`` (list),
        ``n_iter`` (int).
    """
    Y = np.asarray(observation, dtype=float)
    if Y.ndim != 2:
        raise ValueError(f"observation must be 2-D (T, p); got shape {Y.shape}")
    n, m = int(LatentDim), int(InputDim)

    y_mean = Y.mean(axis=0)
    Yc = Y - y_mean

    res = _em_fit(Yc, n, m)
    return {
        "A": res["A"], "B": res["B"], "C": res["C"], "Q": res["Q"], "R": res["R"],
        "latent": res["x_smooth"], "inputs": res["u"], "y_mean": y_mean,
        "ll_history": res["ll_history"], "u_history": res["u_history"],
        "n_iter": res["n_iter"],
    }


def estimate_latent_and_input(
    observation: np.ndarray,
    LatentDim: int,
    InputDim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate latent states and inputs (coordinate-ascent ML) — submission API.

    Same contract as :func:`estimator.estimate_latent_and_input`.  The latent
    states are the **smoothed** states (offline recovery from the full
    ``y_{1:T}``); the inputs use the unweighted per-timestep back-out for
    consistency with the baseline / smooth estimators.

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
        Smoothed latent states (in the internal subspace basis).
    inputs : np.ndarray, shape (T, InputDim)
        Estimated inputs (identifiable up to a linear map; see module note).

    Notes
    -----
    Contractually guaranteed never to crash and to always return arrays of shape
    ``(T, LatentDim)`` / ``(T, InputDim)``.  If the iteration diverges, fails to
    converge or produces a non-finite result, it emits a ``warnings.warn`` and
    **falls back to the subspace-ID baseline** (:func:`estimator.estimate_latent_and_input`),
    which itself never raises.
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
            f"estimate_latent_and_input (em): falling back to baseline "
            f"({type(exc).__name__}: {exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        return _baseline_estimate(Y, n, m)


# ---------------------------------------------------------------------------
# Self-test (run `python estimator_em.py`)
# ---------------------------------------------------------------------------
def _self_test() -> None:
    """Compare the coordinate-ascent estimator with the baseline on mixed_input."""
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
    m_em = fit_and_filter(y, 4, 2)

    print("default_neural_system, mixed_input (T=500):")
    print(f"  recon RMSE : baseline={recon_rmse(y, m_base):.4f}   "
          f"em={recon_rmse(y, m_em):.4f}")
    print(f"  input R^2  : baseline={input_r2(m_base['inputs'], u_true):.4f}   "
          f"em={input_r2(m_em['inputs'], u_true):.4f}   (baseline ref 0.186)")
    print(f"  iterations : {m_em['n_iter']}   "
          f"(final ll={m_em['ll_history'][-1]:.2f})")


if __name__ == "__main__":
    _self_test()
