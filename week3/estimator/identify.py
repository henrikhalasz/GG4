"""estimator/identify.py — Affine LGSSM identification (EM with known inputs).

Model fitted in TRUE input units (no input centring):

    x_{t+1} = A x_t + B u_t + a + w_t,   w ~ N(0, Q)        u_t ∈ [0, 1]^m, resting u = 0
    y_t     = C x_t + c + v_t,           v ~ N(0, R)

* ``a`` absorbs the constant drive from a non-negative probe.
* ``c`` absorbs the observation baseline.
* Resting input ``u = 0`` stays "off"; ``u = 1`` is max — no centring trick.

EM with known inputs:
    E-step: Kalman filter + RTS smoother give x^s, P^s, and the lag-one
            smoothed cov  E[x_{t+1} x_t^T | Y].
    M-step: closed-form regressions on [x_t; u_t; 1] and [x_t; 1].

Convention:
    Timing: ``u_t`` drives ``x_t -> x_{t+1}``; the filter predict at step ``t``
    consumes ``U[t-1]``. ``Y[0]`` is the very first observation.

This is the only file that needs to know about EM — controllers and the
observer consume only the fitted matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_JITTER = 1e-8
_COV_FLOOR = 1e-6
_RHO_CAP_INIT = 1.005  # warm-start spectral cap (refit-EM is free to move)


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------
def _sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _safe_inv(M: np.ndarray, jitter: float = _JITTER) -> np.ndarray:
    n = M.shape[0]
    return np.linalg.inv(M + jitter * np.eye(n))


def _regularize_cov(M: np.ndarray, eps: float = _COV_FLOOR) -> np.ndarray:
    return _sym(M) + eps * np.eye(M.shape[0])


def _logpdf_mvn_zero(x: np.ndarray, S: np.ndarray) -> float:
    """log N(0, S) at x — innovation log-pdf with explicit zero mean."""
    p = x.size
    Sj = S + _JITTER * np.eye(p)
    try:
        L = np.linalg.cholesky(Sj)
        z = np.linalg.solve(L, x)
        ld = 2.0 * float(np.sum(np.log(np.diag(L))))
        return -0.5 * (p * np.log(2.0 * np.pi) + ld + float(z @ z))
    except np.linalg.LinAlgError:
        sign, ld = np.linalg.slogdet(Sj)
        if sign <= 0:
            return float("-inf")
        return -0.5 * (p * np.log(2.0 * np.pi) + float(ld) + float(x @ np.linalg.solve(Sj, x)))


# ---------------------------------------------------------------------------
# Hankel/SVD warm start (reused idea from the Week-2 estimator)
# ---------------------------------------------------------------------------
def _block_hankel(X: np.ndarray, k: int, t0: int, j: int) -> np.ndarray:
    """Block-Hankel slice of ``X`` with ``k`` row-blocks starting at time ``t0``.

    Returns a ``(k * X.shape[1], j)`` matrix whose ``i``-th block row is
    ``X[t0+i : t0+i+j].T``.
    """
    T, d = X.shape
    if t0 + k + j - 1 > T:
        raise ValueError(f"block-Hankel out of range: t0={t0}, k={k}, j={j}, T={T}")
    H = np.empty((k * d, j))
    for i in range(k):
        H[i * d:(i + 1) * d, :] = X[t0 + i:t0 + i + j, :].T
    return H


def _n4sid_init(
    Y: np.ndarray, U: np.ndarray, n: int, k: Optional[int] = None
) -> Dict:
    """N4SID input-output subspace initialisation (Van Overschee & De Moor).

    Returns a warm-start dict matching :func:`_warm_start`'s contract
    (``A, B, C, Q, R, a, c, x0_mean, P0, y_mean``) but uses the **oblique
    projection of the future outputs onto the past inputs+outputs along the
    future inputs** to recover the extended observability matrix — the
    distinguishing feature of N4SID vs the output-only Hankel SVD.

    Steps (with window size ``k``, data columns ``j = T − 2k + 1``):

    1. Build block-Hankel matrices ``U_p, U_f, Y_p, Y_f`` (past/future).
    2. Combined past ``W_p = [U_p; Y_p]``.
    3. Oblique projection ``O_i = Y_f / U_f * W_p`` (project ``Y_f`` onto
       ``[W_p; U_f]`` via least squares, keep only the ``W_p`` part).
    4. SVD ``O_i = U Σ V^T``; truncate to rank ``n``. Then
       ``Γ = U_n √Σ_n``  (extended observability, shape ``(k p, n)``) and
       ``X̂_i = √Σ_n V_n^T`` (state estimate at time ``i = k``, ``(n, j)``).
    5. ``C = Γ[:p, :]`` and ``A = pinv(Γ[:-p, :]) @ Γ[p:, :]`` (shift-invariance).
    6. Build a smooth state trajectory by running a Kalman *prediction*
       pass with the (A, C) just recovered and tentative ``Q, R = I``;
       then regress ``[A B a]`` and ``[C c]`` from that state sequence and
       capture residual covariances ``Q, R`` for the EM seed.

    If any step fails numerically, falls back to :func:`_warm_start`.
    """
    T, p = Y.shape
    m = U.shape[1]
    if k is None:
        # Window length: needs k ≥ n + 1 for shift-invariance; the lstsq
        # in step (3) has ((m + p) k + m k) rows by j cols and wants
        # j > (m + p) k. Pick k around 2n with the constraint T ≥ 2k + 1 + j.
        k = max(2 * n, 3)
        # Keep j > (m + p) k by at least 2× safety.
        while T - 2 * k + 1 < 3 * (m + p) * k and k > n + 1:
            k -= 1
    j = T - 2 * k + 1
    if j < (m + p) * k + 1 or k < n + 1:
        # Not enough data — fall back to output-only init.
        return _warm_start(Y, U, n)

    y_mean = Y.mean(axis=0)
    Yc = Y - y_mean

    try:
        U_p = _block_hankel(U, k, 0, j)
        U_f = _block_hankel(U, k, k, j)
        Y_p = _block_hankel(Yc, k, 0, j)
        Y_f = _block_hankel(Yc, k, k, j)
        W_p = np.vstack([U_p, Y_p])
        n_w = W_p.shape[0]
        Z = np.vstack([W_p, U_f])
        # Oblique projection: regress each row of Y_f onto Z, keep only the
        # W_p part of the coefficient. coef.shape == (n_w + m k, k p).
        coef, *_ = np.linalg.lstsq(Z.T, Y_f.T, rcond=None)
        O_i = coef[:n_w, :].T @ W_p  # shape (k p, j)

        U_svd, S_svd, Vt_svd = np.linalg.svd(O_i, full_matrices=False)
        rank = max(1, min(n, S_svd.size))
        sqrt_s = np.sqrt(np.maximum(S_svd[:rank], 1e-12))
        Gamma = U_svd[:, :rank] * sqrt_s
        if rank < n:
            # Pad with tiny columns so dims line up; EM will refine.
            pad = np.zeros((Gamma.shape[0], n - rank))
            pad[:n - rank] = 1e-3 * np.eye(n - rank, Gamma.shape[0]).T
            Gamma = np.hstack([Gamma, pad])
        C_sub = Gamma[:p, :]
        # Shift-invariance: Γ[:-p] A = Γ[p:].
        A_sub, *_ = np.linalg.lstsq(Gamma[:-p, :], Gamma[p:, :], rcond=None)

        eig = np.linalg.eigvals(A_sub)
        rho = float(np.max(np.abs(eig))) if eig.size else 0.0
        if rho > _RHO_CAP_INIT:
            A_sub = A_sub * (0.99 / rho)
    except Exception:
        return _warm_start(Y, U, n)

    # Rough state trajectory: one-step Kalman predictor with placeholder Q, R.
    # We just need a state sequence good enough for the dynamics regression
    # below; EM will sharpen it.
    Q0 = np.eye(n)
    R0 = np.eye(p)
    a0 = np.zeros(n)
    c0 = y_mean.copy()
    x0 = np.zeros(n)
    P0 = np.eye(n)
    x_filt, _Pf, _xp, _Pp, _ll, _Kl = _kalman_filter_affine(
        Y, U, A_sub, np.zeros((n, m)), C_sub, Q0, R0, a0, c0, x0, P0
    )
    x_hat = x_filt

    # Same regressions as _warm_start, but on the N4SID state sequence.
    rt = np.hstack([x_hat[:-1], U[:-1], np.ones((T - 1, 1))])
    dyn_sol, *_ = np.linalg.lstsq(rt, x_hat[1:], rcond=None)
    dyn_sol = dyn_sol.T
    A_w = dyn_sol[:, :n]
    B_w = dyn_sol[:, n:n + m]
    a_w = dyn_sol[:, n + m]
    eig = np.linalg.eigvals(A_w)
    rho = float(np.max(np.abs(eig))) if eig.size else 0.0
    if rho > _RHO_CAP_INIT:
        A_w = A_w * (0.99 / rho)

    st = np.hstack([x_hat, np.ones((T, 1))])
    obs_sol, *_ = np.linalg.lstsq(st, Y, rcond=None)
    obs_sol = obs_sol.T
    C_w = obs_sol[:, :n]
    c_w = obs_sol[:, n]

    pred_x = rt @ dyn_sol.T
    Q_resid = x_hat[1:] - pred_x
    Q_w = _regularize_cov(np.cov(Q_resid.T, ddof=0)) if T > 2 else np.eye(n) * _COV_FLOOR

    pred_y = st @ obs_sol.T
    R_resid = Y - pred_y
    R_w = _regularize_cov(np.cov(R_resid.T, ddof=0)) if T > 1 else np.eye(p) * _COV_FLOOR

    x0_mean = x_hat[0].copy()
    P0_out = np.eye(n)

    return dict(A=A_w, B=B_w, C=C_w, Q=Q_w, R=R_w, a=a_w, c=c_w,
                x0_mean=x0_mean, P0=P0_out, y_mean=y_mean,
                init_method="n4sid",
                hankel_sigma=S_svd if 'S_svd' in dir() else None)


def estimate_order(
    Y: np.ndarray, U: np.ndarray, max_n: int = 12, k: Optional[int] = None,
    sigma_threshold: float = 1e-3,
) -> Tuple[int, np.ndarray]:
    """Order estimate from the N4SID Hankel singular spectrum.

    Returns ``(n_hat, sigma)``. ``n_hat`` is the number of singular values
    that exceed ``sigma_threshold × σ[0]`` (within the first ``max_n``) —
    the count of "non-trivial" modes against the noise floor.

    Notes
    -----
    Works well when the spectrum has a clean cliff (as on the Brain — a
    12-order-of-magnitude drop at the true ``n = 6``). For systems with a
    *continuous* spread of singular values (random small-n systems often
    look like this), automatic order selection is genuinely ambiguous and
    callers should set ``n`` manually. The threshold ``1e-3`` is the right
    bar for systems near the Brain's regime; tighten / loosen for cleaner
    or noisier data.
    """
    T, p = Y.shape
    m = U.shape[1]
    k_win = k
    if k_win is None:
        k_win = max(2 * max_n, 3)
        while T - 2 * k_win + 1 < (m + p) * k_win + 1 and k_win > max_n + 1:
            k_win -= 1
    j = T - 2 * k_win + 1
    if j < (m + p) * k_win + 1:
        raise ValueError(f"too few samples for order selection: T={T}, k={k_win}, max_n={max_n}")

    Yc = Y - Y.mean(axis=0)
    U_p = _block_hankel(U, k_win, 0, j)
    U_f = _block_hankel(U, k_win, k_win, j)
    Y_p = _block_hankel(Yc, k_win, 0, j)
    Y_f = _block_hankel(Yc, k_win, k_win, j)
    W_p = np.vstack([U_p, Y_p])
    Z = np.vstack([W_p, U_f])
    coef, *_ = np.linalg.lstsq(Z.T, Y_f.T, rcond=None)
    O_i = coef[:W_p.shape[0], :].T @ W_p
    sigma = np.linalg.svd(O_i, compute_uv=False)
    head = sigma[:max_n]
    if head.size == 0 or head[0] <= 0:
        return 1, sigma
    norm = head / head[0]

    # Two cooperating rules:
    #   (a) noise-floor: count SVs above ``sigma_threshold * σ[0]``;
    #   (b) knee:        last index where σ_k / σ_{k-1} ≤ 0.5 (≥ 2× drop).
    # We take the larger so that a clean cliff after a long shallow decay
    # (e.g. the 6th Brain mode at σ ≈ 1e-3 σ[0]) isn't truncated by an
    # over-eager threshold rule. Both rules independently land on n = 6
    # for the Brain.
    n_from_threshold = int((norm > sigma_threshold).sum())
    if head.size >= 2:
        ratios = head[1:] / np.maximum(head[:-1], 1e-300)
        knee_idx = np.where(ratios <= 0.5)[0]
        n_from_knee = int(knee_idx.max()) + 1 if knee_idx.size else 1
    else:
        n_from_knee = 1
    return max(n_from_threshold, n_from_knee, 1), sigma


def _hankel_past_future(Yc: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    T, p = Yc.shape
    j = T - 2 * k + 1
    if j < 1:
        raise ValueError(f"sequence too short for k={k} (T={T})")
    Yp = np.empty((p * k, j))
    Yf = np.empty((p * k, j))
    for i in range(k):
        Yp[i * p:(i + 1) * p, :] = Yc[i:i + j, :].T
        Yf[i * p:(i + 1) * p, :] = Yc[k + i:k + i + j, :].T
    return Yp, Yf


def _warm_start(Y: np.ndarray, U: np.ndarray, n: int) -> Dict:
    """Subspace + regression warm start of all affine LGSSM params.

    Steps:
      1. y_mean = mean(Y); Yc = Y - y_mean.
      2. Block-Hankel SVD on Yc -> rough Γ (extended obs) -> rough A, C.
      3. x_hat = Yc @ pinv(C).T  (very rough state estimate).
      4. Regress [A B a] from [x_t; u_t; 1] -> x_{t+1}.
      5. Regress [C c] from [x_t; 1] -> y_t.
      6. Q, R from regression residuals.
    """
    T, p = Y.shape
    m = U.shape[1]

    y_mean = Y.mean(axis=0)
    Yc = Y - y_mean

    # Hankel/SVD subspace fit
    k = max(2 * n, int(np.ceil(np.sqrt(T) / 2.0)))
    k = min(k, (T + 1) // 6)
    k = max(k, n + 1, 2)
    try:
        Yp, Yf = _hankel_past_future(Yc, k)
        projection = Yf @ np.linalg.pinv(Yp) @ Yp
        U_svd, S_svd, _ = np.linalg.svd(projection, full_matrices=False)
        n_eff = min(n, U_svd.shape[1], int(np.sum(S_svd > 1e-10 * (S_svd[0] if S_svd.size else 1.0))))
        n_eff = max(n_eff, 1)
        sqrt_s = np.sqrt(np.maximum(S_svd[:n_eff], 1e-12))
        Gamma = U_svd[:, :n_eff] * sqrt_s
        if n_eff < n:
            Gamma = np.hstack([Gamma, 1e-3 * np.eye(Gamma.shape[0], n - n_eff)])
        C_sub = Gamma[:p, :]
        A_sub = np.linalg.pinv(Gamma[:-p, :]) @ Gamma[p:, :]
    except Exception:
        C_sub = np.random.default_rng(0).standard_normal((p, n))
        A_sub = 0.9 * np.eye(n)

    # Stabilize A_sub if numerically unstable.
    eig = np.linalg.eigvals(A_sub)
    rho = float(np.max(np.abs(eig))) if eig.size else 0.0
    if rho > _RHO_CAP_INIT:
        A_sub = A_sub * (0.99 / rho)

    # Rough state estimate from subspace C.
    x_hat = Yc @ np.linalg.pinv(C_sub).T  # (T, n)

    # Dynamics regression: x_{t+1} ≈ [A B a] [x_t; u_t; 1]
    rt = np.hstack([x_hat[:-1], U[:-1], np.ones((T - 1, 1))])  # (T-1, n+m+1)
    dyn_sol, *_ = np.linalg.lstsq(rt, x_hat[1:], rcond=None)
    dyn_sol = dyn_sol.T  # (n, n+m+1)
    A_w = dyn_sol[:, :n]
    B_w = dyn_sol[:, n:n + m]
    a_w = dyn_sol[:, n + m]

    eig = np.linalg.eigvals(A_w)
    rho = float(np.max(np.abs(eig))) if eig.size else 0.0
    if rho > _RHO_CAP_INIT:
        A_w = A_w * (0.99 / rho)

    # Observation regression: y_t ≈ [C c] [x_t; 1]
    st = np.hstack([x_hat, np.ones((T, 1))])  # (T, n+1)
    obs_sol, *_ = np.linalg.lstsq(st, Y, rcond=None)
    obs_sol = obs_sol.T  # (p, n+1)
    C_w = obs_sol[:, :n]
    c_w = obs_sol[:, n]

    # Residual covariances
    pred_x = rt @ dyn_sol.T  # (T-1, n)
    Q_resid = x_hat[1:] - pred_x
    Q_w = _regularize_cov(np.cov(Q_resid.T, ddof=0)) if T > 2 else np.eye(n) * _COV_FLOOR

    pred_y = st @ obs_sol.T  # (T, p)
    R_resid = Y - pred_y
    R_w = _regularize_cov(np.cov(R_resid.T, ddof=0)) if T > 1 else np.eye(p) * _COV_FLOOR

    x0_mean = x_hat[0].copy()
    P0 = np.eye(n)

    return dict(A=A_w, B=B_w, C=C_w, Q=Q_w, R=R_w, a=a_w, c=c_w,
                x0_mean=x0_mean, P0=P0, y_mean=y_mean)


# ---------------------------------------------------------------------------
# Forward Kalman filter (affine)
# ---------------------------------------------------------------------------
def _kalman_filter_affine(
    Y: np.ndarray, U: np.ndarray,
    A: np.ndarray, B: np.ndarray, C: np.ndarray,
    Q: np.ndarray, R: np.ndarray, a: np.ndarray, c: np.ndarray,
    x0_mean: np.ndarray, P0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Return (x_filt, P_filt, x_pred, P_pred, loglik, K_last)."""
    T, p = Y.shape
    n = A.shape[0]
    x_filt = np.empty((T, n))
    P_filt = np.empty((T, n, n))
    x_pred = np.empty((T, n))
    P_pred = np.empty((T, n, n))
    eye_n = np.eye(n)
    loglik = 0.0
    K_last = np.zeros((n, p))

    for t in range(T):
        if t == 0:
            x_pred[0] = x0_mean
            P_pred[0] = P0
        else:
            x_pred[t] = A @ x_filt[t - 1] + B @ U[t - 1] + a
            P_pred[t] = _sym(A @ P_filt[t - 1] @ A.T + Q)

        innov = Y[t] - C @ x_pred[t] - c
        S = _sym(C @ P_pred[t] @ C.T + R)
        loglik += _logpdf_mvn_zero(innov, S)

        K = P_pred[t] @ C.T @ _safe_inv(S)
        x_filt[t] = x_pred[t] + K @ innov
        P_filt[t] = _sym((eye_n - K @ C) @ P_pred[t])
        if t == T - 1:
            K_last = K

    return x_filt, P_filt, x_pred, P_pred, loglik, K_last


# ---------------------------------------------------------------------------
# RTS smoother with lag-one cov (Shumway–Stoffer)
# ---------------------------------------------------------------------------
def _rts_smoother_affine(
    x_filt: np.ndarray, P_filt: np.ndarray,
    x_pred: np.ndarray, P_pred: np.ndarray,
    A: np.ndarray, C: np.ndarray, K_last: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_smooth, P_smooth, Plag).

    Plag[t] = Cov(x_{t+1}, x_t | Y) for t = 0..T-2.
    """
    T, n = x_filt.shape
    x_smooth = np.empty_like(x_filt)
    P_smooth = np.empty_like(P_filt)
    J = np.empty((max(T - 1, 0), n, n))

    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for t in range(T - 2, -1, -1):
        J[t] = P_filt[t] @ A.T @ _safe_inv(P_pred[t + 1])
        x_smooth[t] = x_filt[t] + J[t] @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = _sym(P_filt[t] + J[t] @ (P_smooth[t + 1] - P_pred[t + 1]) @ J[t].T)

    Plag = np.zeros((max(T - 1, 0), n, n))
    if T >= 2:
        eye_n = np.eye(n)
        Plag[T - 2] = (eye_n - K_last @ C) @ A @ P_filt[T - 2]
        for t in range(T - 3, -1, -1):
            Plag[t] = (
                P_filt[t + 1] @ J[t].T
                + J[t + 1] @ (Plag[t + 1] - A @ P_filt[t + 1]) @ J[t].T
            )

    return x_smooth, P_smooth, Plag


# ---------------------------------------------------------------------------
# Closed-form M-step
# ---------------------------------------------------------------------------
def _m_step(
    x_smooth: np.ndarray, P_smooth: np.ndarray, Plag: np.ndarray,
    Y: np.ndarray, U: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T, n = x_smooth.shape
    p = Y.shape[1]
    m = U.shape[1]
    Trr = T - 1

    # Dynamics sufficient stats (t = 0..T-2)
    Sxx = np.zeros((n, n))   # Σ E[x_t x_t^T]
    Sxnx = np.zeros((n, n))  # Σ E[x_{t+1} x_t^T]
    Sxnxn = np.zeros((n, n)) # Σ E[x_{t+1} x_{t+1}^T]
    Sxu = np.zeros((n, m))   # Σ x_t u_t^T
    Sxn_u = np.zeros((n, m)) # Σ x_{t+1} u_t^T
    Sxn = np.zeros(n)        # Σ x_{t+1}
    Sx = np.zeros(n)         # Σ x_t  (t=0..T-2)
    Suu = np.zeros((m, m))   # Σ u_t u_t^T
    Su = np.zeros(m)         # Σ u_t

    for t in range(Trr):
        xs_t = x_smooth[t]
        xs_tp1 = x_smooth[t + 1]
        Exx = P_smooth[t] + np.outer(xs_t, xs_t)
        Exnx = Plag[t] + np.outer(xs_tp1, xs_t)
        Exnxn = P_smooth[t + 1] + np.outer(xs_tp1, xs_tp1)
        Sxx += Exx
        Sxnx += Exnx
        Sxnxn += Exnxn
        Sxu += np.outer(xs_t, U[t])
        Sxn_u += np.outer(xs_tp1, U[t])
        Sxn += xs_tp1
        Sx += xs_t
        Suu += np.outer(U[t], U[t])
        Su += U[t]

    # Σ E[r_t r_t^T] where r_t = [x_t; u_t; 1]  (size n+m+1)
    Sigma_rr = np.zeros((n + m + 1, n + m + 1))
    Sigma_rr[:n, :n] = Sxx
    Sigma_rr[:n, n:n + m] = Sxu
    Sigma_rr[:n, n + m] = Sx
    Sigma_rr[n:n + m, :n] = Sxu.T
    Sigma_rr[n:n + m, n:n + m] = Suu
    Sigma_rr[n:n + m, n + m] = Su
    Sigma_rr[n + m, :n] = Sx
    Sigma_rr[n + m, n:n + m] = Su
    Sigma_rr[n + m, n + m] = float(Trr)

    Sigma_xnr = np.zeros((n, n + m + 1))
    Sigma_xnr[:, :n] = Sxnx
    Sigma_xnr[:, n:n + m] = Sxn_u
    Sigma_xnr[:, n + m] = Sxn

    Sigma_rr_inv = _safe_inv(Sigma_rr, jitter=1e-10)
    beta = Sigma_xnr @ Sigma_rr_inv  # (n, n+m+1)
    A_new = beta[:, :n]
    B_new = beta[:, n:n + m]
    a_new = beta[:, n + m]

    Q_unnorm = Sxnxn - beta @ Sigma_xnr.T
    Q_new = _regularize_cov(Q_unnorm / max(Trr, 1))

    # Observation sufficient stats (t = 0..T-1)
    Sxx_all = np.zeros((n, n))
    Sx_all = np.zeros(n)
    Syx = np.zeros((p, n))
    Sy = np.zeros(p)
    Syy = np.zeros((p, p))
    for t in range(T):
        xs_t = x_smooth[t]
        Exx = P_smooth[t] + np.outer(xs_t, xs_t)
        Sxx_all += Exx
        Sx_all += xs_t
        Syx += np.outer(Y[t], xs_t)
        Sy += Y[t]
        Syy += np.outer(Y[t], Y[t])

    Sigma_ss = np.zeros((n + 1, n + 1))
    Sigma_ss[:n, :n] = Sxx_all
    Sigma_ss[:n, n] = Sx_all
    Sigma_ss[n, :n] = Sx_all
    Sigma_ss[n, n] = float(T)

    Sigma_ys = np.zeros((p, n + 1))
    Sigma_ys[:, :n] = Syx
    Sigma_ys[:, n] = Sy

    Sigma_ss_inv = _safe_inv(Sigma_ss, jitter=1e-10)
    gamma = Sigma_ys @ Sigma_ss_inv  # (p, n+1)
    C_new = gamma[:, :n]
    c_new = gamma[:, n]

    R_unnorm = Syy - gamma @ Sigma_ys.T
    R_new = _regularize_cov(R_unnorm / T)

    return A_new, B_new, a_new, C_new, c_new, Q_new, R_new


# ---------------------------------------------------------------------------
# Public estimator class
# ---------------------------------------------------------------------------
@dataclass
class EstimatorNew:
    """Affine LGSSM with EM, known inputs and offsets ``a, c``.

    Conventions (controllers/observer/reachability rely on these):
        x_{t+1} = A x_t + B u_t + a + w,   w ~ N(0, Q)
        y_t     = C x_t + c + v,           v ~ N(0, R)
        u_t ∈ [0, 1]^m, resting input is 0.
    """

    A: Optional[np.ndarray] = None
    B: Optional[np.ndarray] = None
    C: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    a: Optional[np.ndarray] = None
    c: Optional[np.ndarray] = None
    x0_mean: Optional[np.ndarray] = None
    P0: Optional[np.ndarray] = None
    loglik_hist: List[float] = field(default_factory=list)
    x_filt: Optional[np.ndarray] = None
    x_smooth: Optional[np.ndarray] = None
    P_smooth: Optional[np.ndarray] = None
    warm_start_info: Dict = field(default_factory=dict)
    n: Optional[int] = None
    p: Optional[int] = None
    m: Optional[int] = None
    fit_iters: int = 0

    def fit(
        self,
        Y: np.ndarray,
        U: np.ndarray,
        n: Optional[int] = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        verbose: bool = False,
        init: str = "n4sid",
    ) -> "EstimatorNew":
        Y = np.asarray(Y, dtype=float)
        U = np.asarray(U, dtype=float)
        if Y.ndim != 2 or U.ndim != 2:
            raise ValueError("Y and U must be 2-D arrays.")
        T = Y.shape[0]
        if U.shape[0] != T:
            raise ValueError(f"Y/U length mismatch: T_Y={T}, T_U={U.shape[0]}")
        p = Y.shape[1]
        m = U.shape[1]
        if n is None:
            n_hat, _ = estimate_order(Y, U, max_n=min(12, max(2, (T // 12) - 1)))
            n = max(int(n_hat), 1)
        self.n, self.p, self.m = int(n), int(p), int(m)

        if init == "n4sid":
            warm = _n4sid_init(Y, U, n)
        elif init in ("output_ssi", "warm_start", "ssi"):
            warm = _warm_start(Y, U, n)
        elif init == "random":
            rng = np.random.default_rng(0)
            A0 = 0.8 * np.eye(n)
            B0 = 0.1 * rng.standard_normal((n, m))
            C0 = rng.standard_normal((p, n))
            warm = dict(
                A=A0, B=B0, C=C0,
                Q=np.eye(n), R=np.eye(p),
                a=np.zeros(n), c=Y.mean(axis=0),
                x0_mean=np.zeros(n), P0=np.eye(n),
                y_mean=Y.mean(axis=0),
            )
        else:
            raise ValueError(f"unknown init={init!r}; use 'n4sid', 'output_ssi', or 'random'")
        A, B, C = warm["A"], warm["B"], warm["C"]
        Q, R = warm["Q"], warm["R"]
        a, c = warm["a"], warm["c"]
        x0_mean = warm["x0_mean"]
        P0 = warm["P0"]
        self.warm_start_info = {
            "A_rho": float(np.max(np.abs(np.linalg.eigvals(A)))),
            "B_col_norms": np.linalg.norm(B, axis=0).tolist(),
            "y_mean": warm["y_mean"].tolist(),
        }

        ll_hist: List[float] = []
        prev_ll = -np.inf
        iters = 0
        for it in range(max_iter):
            iters = it + 1
            x_filt, P_filt, x_pred, P_pred, ll, K_last = _kalman_filter_affine(
                Y, U, A, B, C, Q, R, a, c, x0_mean, P0
            )
            x_smooth, P_smooth, Plag = _rts_smoother_affine(
                x_filt, P_filt, x_pred, P_pred, A, C, K_last
            )
            ll_hist.append(float(ll))
            if verbose:
                print(f"[EM] iter {it:3d}: ll = {ll:+.4f}")
            if it > 0 and (ll - prev_ll) < tol * max(abs(prev_ll), 1.0):
                break
            prev_ll = ll

            A, B, a, C, c, Q, R = _m_step(x_smooth, P_smooth, Plag, Y, U)

            # Update initial state from smoothed.
            x0_mean = x_smooth[0].copy()
            P0 = _regularize_cov(P_smooth[0])

        # Final E-step to capture final state estimates.
        x_filt, P_filt, x_pred, P_pred, ll, K_last = _kalman_filter_affine(
            Y, U, A, B, C, Q, R, a, c, x0_mean, P0
        )
        x_smooth, P_smooth, Plag = _rts_smoother_affine(
            x_filt, P_filt, x_pred, P_pred, A, C, K_last
        )
        ll_hist.append(float(ll))

        self.A, self.B, self.C, self.Q, self.R = A, B, C, Q, R
        self.a, self.c = a, c
        self.x0_mean = x0_mean
        self.P0 = P0
        self.loglik_hist = ll_hist
        self.x_filt = x_filt
        self.x_smooth = x_smooth
        self.P_smooth = P_smooth
        self.fit_iters = iters
        return self

    @property
    def params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.A, self.B, self.C, self.Q, self.R, self.a, self.c

    def loglik_history(self) -> List[float]:
        return list(self.loglik_hist)

    # ---- Validation (§5.4) -------------------------------------------------
    def validate(
        self,
        true_model: Optional[Dict] = None,
        Y_val: Optional[np.ndarray] = None,
        U_val: Optional[np.ndarray] = None,
        x_true: Optional[np.ndarray] = None,
    ) -> Dict:
        """Run the §5.4 validation checks.

        ``true_model`` is a dict with at least ``A, B, C`` (and optionally
        ``a, c``).  ``Y_val, U_val`` are an optional held-out trajectory for
        one-step-prediction error.  ``x_true`` (T_fit, n_true) enables the
        aligned state-tracking R² check on the fit data.
        """
        out: Dict = {}
        ll = self.loglik_hist
        # 4. EM sanity — log-lik (almost) monotone.
        out["em_monotone"] = bool(
            all(ll[i + 1] >= ll[i] - 1e-3 * max(abs(ll[i]), 1.0) for i in range(len(ll) - 1))
        )
        out["loglik_final"] = float(ll[-1]) if ll else float("nan")

        rho = float(np.max(np.abs(np.linalg.eigvals(self.A))))
        out["A_spectral_radius"] = rho
        out["A_unstable"] = rho >= 1.0
        out["params_finite"] = bool(
            all(np.all(np.isfinite(p)) for p in (self.A, self.B, self.C, self.Q, self.R, self.a, self.c))
        )

        # 3. G-matrix / zonotope check.
        n = self.A.shape[0]
        IminusA_inv_B = np.linalg.solve(np.eye(n) - self.A, self.B)
        IminusA_inv_a = np.linalg.solve(np.eye(n) - self.A, self.a)
        G_fit = self.C @ IminusA_inv_B      # (p, m)
        z0_fit = self.C @ IminusA_inv_a + self.c
        out["G_fit"] = G_fit
        out["z0_fit"] = z0_fit

        if true_model is not None:
            A_t = np.asarray(true_model["A"], dtype=float)
            B_t = np.asarray(true_model["B"], dtype=float)
            C_t = np.asarray(true_model["C"], dtype=float)
            n_t = A_t.shape[0]
            a_t = np.asarray(true_model.get("a", np.zeros(n_t)), dtype=float)
            c_t = np.asarray(true_model.get("c", np.zeros(C_t.shape[0])), dtype=float)
            G_true = C_t @ np.linalg.solve(np.eye(n_t) - A_t, B_t)
            z0_true = C_t @ np.linalg.solve(np.eye(n_t) - A_t, a_t) + c_t
            out["G_true"] = G_true
            out["z0_true"] = z0_true
            out["G_err_frob"] = float(np.linalg.norm(G_fit - G_true))
            out["G_rel_err"] = float(np.linalg.norm(G_fit - G_true) / max(np.linalg.norm(G_true), 1e-12))
            out["z0_err"] = float(np.linalg.norm(z0_fit - z0_true))
            out["z0_rel_err"] = float(np.linalg.norm(z0_fit - z0_true) / max(np.linalg.norm(z0_true), 1e-12))

        # 2. State-tracking R² (affine alignment: latent has gauge + translation).
        if x_true is not None and self.x_smooth is not None:
            X_hat = self.x_smooth
            T_a = min(X_hat.shape[0], x_true.shape[0])
            Xh = X_hat[:T_a]
            Xt = x_true[:T_a]
            Xh_c = Xh - Xh.mean(axis=0)
            Xt_c = Xt - Xt.mean(axis=0)
            T_align, *_ = np.linalg.lstsq(Xh_c, Xt_c, rcond=None)
            pred = Xh_c @ T_align
            ss_res = float(np.sum((Xt_c - pred) ** 2))
            ss_tot = float(np.sum(Xt_c ** 2))
            out["state_r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # 1. One-step prediction RMS on held-out data.
        if Y_val is not None and U_val is not None:
            xf, Pf, xp, Pp, _, _ = _kalman_filter_affine(
                Y_val, U_val,
                self.A, self.B, self.C, self.Q, self.R, self.a, self.c,
                self.x0_mean, self.P0,
            )
            y_pred = (self.C @ xp.T).T + self.c
            resid = Y_val - y_pred
            out["one_step_rms"] = float(np.sqrt(np.mean(resid ** 2)))

        return out


# ---------------------------------------------------------------------------
# Module self-test (`python estimator_new.py`) — synthetic affine LGSSM
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(0)
    n, m, p, T = 3, 2, 8, 600
    # Stable A
    A = np.array([[0.95, 0.0, 0.0],
                  [0.0, 0.85, 0.0],
                  [0.05, 0.05, 0.7]])
    B = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [0.5, 0.5]])
    C = rng.standard_normal((p, n))
    a = np.array([0.1, -0.05, 0.02])
    c = rng.standard_normal(p)
    Q = 1e-3 * np.eye(n)
    R = 1e-2 * np.eye(p)

    # Simulate
    U = rng.uniform(0.0, 1.0, size=(T, m))
    x = np.zeros((T + 1, n))
    Y = np.zeros((T, p))
    for t in range(T):
        Y[t] = C @ x[t] + c + rng.multivariate_normal(np.zeros(p), R)
        x[t + 1] = A @ x[t] + B @ U[t] + a + rng.multivariate_normal(np.zeros(n), Q)

    est = EstimatorNew().fit(Y, U, n=n, max_iter=60, verbose=False)
    ll = est.loglik_history()
    print(f"EM iters: {est.fit_iters}, ll path: {ll[0]:.2f} -> {ll[-1]:.2f}")
    diffs = np.diff(ll)
    print(f"min ll increment across iters: {diffs.min():+.4f} (should be >= ~0)")
    rep = est.validate(true_model={"A": A, "B": B, "C": C, "a": a, "c": c}, x_true=x[:T])
    for k, v in rep.items():
        if isinstance(v, np.ndarray):
            continue
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _self_test()
