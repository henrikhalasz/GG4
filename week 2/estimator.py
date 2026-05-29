from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np

# --- Fixed hyperparameters  -----------------------------------------------
_COV_SHRINKAGE = 1e-6   # added to identified Q_hat / R_hat (keeps them PD)
_INV_JITTER = 1e-8      # added before every explicit inverse / on covariances
_N_FILTER_PASSES = 2    # forward Kalman passes (pass 1: u=0; pass 2: with u_hat)
_SPECTRAL_CAP = 1.005   # only rescale A_hat if its spectral radius exceeds this
_SPECTRAL_TARGET = 0.999  # value the spectral radius is pulled back to when capped


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------
def _regularize_cov(matrix: np.ndarray, jitter: float) -> np.ndarray:
    """Symmetrise and add jitter * I so the result stays positive-definite.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix, intended to be a covariance estimate.
    jitter : float
        Non-negative value added to the diagonal.

    Returns
    -------
    np.ndarray
        ``0.5 (M + M^T) + jitter * I``.
    """
    n = matrix.shape[0]
    sym = 0.5 * (matrix + matrix.T)
    return sym + jitter * np.eye(n)


def _safe_inverse(matrix: np.ndarray) -> np.ndarray:
    """Invert a (small) symmetric PD matrix, falling back to the pseudoinverse.

    A tiny jitter is added first so that a numerically singular innovation
    covariance never raises ``LinAlgError``.
    """
    n = matrix.shape[0]
    reg = matrix + _INV_JITTER * np.eye(n)
    try:
        return np.linalg.inv(reg)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(reg)


def _select_block_size(T: int, n: int) -> int:
    """Choose the Hankel block size ``k``

    Robust default ``k = max(2n, ceil(sqrt(T) / 2))`` clipped so the Hankel
    matrix keeps at least ``4k`` columns (``j = T - 2k + 1 >= 4k``) and so that
    ``k > n`` (shift block non-empty and ``k >= LatentDim``).

    Parameters
    ----------
    T : int
        Number of observation time points.
    n : int
        Requested latent dimension.

    Returns
    -------
    int
        Block size ``k`` (>= 2).
    """
    k = max(2 * n, int(np.ceil(np.sqrt(T) / 2.0)))
    # j = T - 2k + 1 >= 4k  <=>  k <= (T + 1) / 6
    k = min(k, (T + 1) // 6)
    k = max(k, n + 1)   # need k > n for the shift-invariance regression
    return max(k, 2)


def _block_hankel_past_future(Yc: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build stacked past / future block-Hankel matrices from centred data.

    Parameters
    ----------
    Yc : np.ndarray, shape (T, p)
        Centred observation sequence.
    k : int
        Block size (number of stacked time steps per block).

    Returns
    -------
    Yp : np.ndarray, shape (p * k, j)
        Past block-Hankel matrix; column ``c`` stacks ``y_c .. y_{c+k-1}``.
    Yf : np.ndarray, shape (p * k, j)
        Future block-Hankel matrix; column ``c`` stacks ``y_{c+k} .. y_{c+2k-1}``.

    Here ``j = T - 2k + 1`` is the number of usable columns.
    """
    T, p = Yc.shape
    j = T - 2 * k + 1
    if j < 1:
        raise ValueError(f"sequence too short for block size k={k} (T={T})")
    Yp = np.empty((p * k, j))
    Yf = np.empty((p * k, j))
    for i in range(k):
        Yp[i * p:(i + 1) * p, :] = Yc[i:i + j, :].T
        Yf[i * p:(i + 1) * p, :] = Yc[k + i:k + i + j, :].T
    return Yp, Yf


def _stabilise(A: np.ndarray) -> np.ndarray:
    """Gently rescale ``A`` if subspace ID returned an unstable transition.

    Identification noise can push the estimated spectral radius slightly above
    1, which would let the long-horizon Kalman prediction covariance grow without
    bound.  When the spectral radius exceeds ``_SPECTRAL_CAP`` we scale the whole
    matrix so its radius becomes ``_SPECTRAL_TARGET``; genuinely near-unit modes
    (e.g. the slow-drift system at 0.999) are left untouched.
    """
    eigvals = np.linalg.eigvals(A)
    rho = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
    if rho > _SPECTRAL_CAP:
        A = A * (_SPECTRAL_TARGET / rho)
    return A


# ---------------------------------------------------------------------------
# Core estimation steps
# ---------------------------------------------------------------------------
def _kalman_filter(
    Y: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    u: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Run a standard linear Kalman filter and return the filtered states.

    Parameters
    ----------
    Y : np.ndarray, shape (T, p)
        Observation sequence (already centred when called internally).
    A, B, C : np.ndarray
        System matrices of shapes ``(n, n)``, ``(n, m)``, ``(p, n)``.
    Q, R : np.ndarray
        Process / observation noise covariances, shapes ``(n, n)`` / ``(p, p)``.
    u : np.ndarray, shape (T, m), optional
        Input sequence.  ``None`` (default) is treated as the zero input.

    Returns
    -------
    np.ndarray, shape (T, n)
        Filtered state estimates ``x_hat_{t|t}``.
    """
    T, p = Y.shape
    n = A.shape[0]
    m = B.shape[1]
    if u is None:
        u = np.zeros((T, m))

    x = np.empty((T, n))
    Cpinv = np.linalg.pinv(C)
    eye_n = np.eye(n)

    # Initialise x_{0|0} = pinv(C) y_0, P_{0|0} = I (week2.md §3 B.2).
    x[0] = Cpinv @ Y[0]
    P = eye_n.copy()

    for t in range(1, T):
        # Predict.
        x_pred = A @ x[t - 1] + B @ u[t - 1]
        P_pred = A @ P @ A.T + Q
        # Update.
        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T @ _safe_inverse(S)
        innovation = Y[t] - C @ x_pred
        x[t] = x_pred + K @ innovation
        P = (eye_n - K @ C) @ P_pred
    return x


def _estimate_inputs(X: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Estimate the input that best explains the latent dynamics.

    Solves ``u_t = pinv(B) (x_{t+1} - A x_t)`` for ``t = 0 .. T-2`` and pads the
    final step (``u_{T-1} = u_{T-2}``) so the output has exactly ``T`` rows.

    Parameters
    ----------
    X : np.ndarray, shape (T, n)
        (Filtered) latent state sequence.
    A : np.ndarray, shape (n, n)
        Estimated transition matrix.
    B : np.ndarray, shape (n, m)
        Estimated input matrix.

    Returns
    -------
    np.ndarray, shape (T, m)
        Estimated input sequence (subspace-invariant up to a linear map; see the
        module identifiability note).
    """
    T = X.shape[0]
    m = B.shape[1]
    Bpinv = np.linalg.pinv(B)                       # (m, n)
    increments = X[1:] - X[:-1] @ A.T               # (T-1, n): x_{t+1} - A x_t
    u = np.empty((T, m))
    if T >= 2:
        u[:-1] = increments @ Bpinv.T               # (T-1, m)
        u[-1] = u[-2]                               # pad final step
    else:
        u[:] = 0.0
    return u


def _fit_subspace_model(
    Y: np.ndarray, n: int, m: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identify ``(A_hat, B_hat, C_hat, Q_hat, R_hat)`` from centred observations.

    Implements N4SID-style subspace identification for
    ``A_hat`` / ``C_hat``, a residual-SVD construction (Option 1) for ``B_hat``,
    and residual covariances (with shrinkage) for ``Q_hat`` / ``R_hat``.

    Parameters
    ----------
    Y : np.ndarray, shape (T, p)
        **Centred** observation sequence.
    n : int
        Latent dimension.
    m : int
        Input dimension.

    Returns
    -------
    A_hat : np.ndarray, shape (n, n)
    B_hat : np.ndarray, shape (n, m)
    C_hat : np.ndarray, shape (p, n)
    Q_hat : np.ndarray, shape (n, n)
    R_hat : np.ndarray, shape (p, p)

    Notes
    -----
    The matrices are expressed in the arbitrary similarity-transform basis of the
    subspace decomposition (module identifiability note).
    """
    T, p = Y.shape
    k = _select_block_size(T, n)

    # --- Subspace ID: A_hat, C_hat ---------------------------------------
    Yp, Yf = _block_hankel_past_future(Y, k)
    # Orthogonal projection of the future row space onto the past row space.
    projection = Yf @ np.linalg.pinv(Yp) @ Yp        # (p*k, j)
    U, S, _ = np.linalg.svd(projection, full_matrices=False)

    n_eff = min(n, U.shape[1], int(np.sum(S > 1e-10 * (S[0] if S.size else 1.0))))
    n_eff = max(n_eff, 1)
    sqrt_s = np.sqrt(S[:n_eff])
    Gamma = U[:, :n_eff] * sqrt_s                    # extended observability (p*k, n_eff)

    # Pad to the requested n if the projection was rank-deficient, so downstream
    # shapes are exactly (n, n) / (p, n) as the interface promises.
    if n_eff < n:
        Gamma = np.hstack([Gamma, np.zeros((Gamma.shape[0], n - n_eff))])

    C_hat = Gamma[:p, :]                             # first block row (p, n)
    # Initial transition from the shift-invariance of Gamma.
    A_hat = _stabilise(np.linalg.pinv(Gamma[:-p, :]) @ Gamma[p:, :])

    # --- Preliminary Q, R for the autonomous filtering pass --------------
    Cpinv = np.linalg.pinv(C_hat)                    # (n, p)
    x_ls = Y @ Cpinv.T                               # crude least-squares states (T, n)
    R0 = _regularize_cov(np.cov((Y - x_ls @ C_hat.T).T), _INV_JITTER) if T > 1 else np.eye(p)
    Q0 = _regularize_cov(np.cov((x_ls[1:] - x_ls[:-1] @ A_hat.T).T), _INV_JITTER) if T > 2 else np.eye(n)
    R0, Q0 = np.atleast_2d(R0), np.atleast_2d(Q0)

    # --- Refine A by regression on the filtered state --------------------
    B_zero = np.zeros((n, m))
    x_auto = _kalman_filter(Y, A_hat, B_zero, C_hat, Q0, R0, u=None)
    if x_auto.shape[0] > n + 1:
        A_hat = _stabilise(np.linalg.lstsq(x_auto[:-1], x_auto[1:], rcond=None)[0].T)
        x_auto = _kalman_filter(Y, A_hat, B_zero, C_hat, Q0, R0, u=None)

    # --- B_hat from the autonomous-filter residual (Option 1) ------------
    dyn_resid = x_auto[1:] - x_auto[:-1] @ A_hat.T   # (T-1, n) ~ B u + w
    if dyn_resid.shape[0] >= 1:
        Ud, _, _ = np.linalg.svd(dyn_resid.T, full_matrices=False)  # (n, .)
        cols = min(m, Ud.shape[1])
        B_hat = np.zeros((n, m))
        B_hat[:, :cols] = Ud[:, :cols]               # top-m left singular vectors
    else:
        B_hat = np.zeros((n, m))

    # --- Refined Q_hat, R_hat with the identified input ------------------
    u_auto = _estimate_inputs(x_auto, A_hat, B_hat)
    dyn_resid_in = x_auto[1:] - x_auto[:-1] @ A_hat.T - u_auto[:-1] @ B_hat.T
    obs_resid = Y - x_auto @ C_hat.T
    Q_hat = _regularize_cov(np.cov(dyn_resid_in.T), _COV_SHRINKAGE) if T > 2 else np.eye(n)
    R_hat = _regularize_cov(np.cov(obs_resid.T), _COV_SHRINKAGE) if T > 1 else np.eye(p)
    Q_hat = np.atleast_2d(Q_hat)
    R_hat = np.atleast_2d(R_hat)

    return A_hat, B_hat, C_hat, Q_hat, R_hat


def fit_and_filter(observation: np.ndarray, LatentDim: int, InputDim: int) -> Dict[str, np.ndarray]:
    """Run the full pipeline and return every intermediate quantity.

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
        ``latent`` (T,n) filtered states, ``inputs`` (T,m) estimated inputs,
        ``y_mean`` (p,) column mean removed before identification.
    """
    Y = np.asarray(observation, dtype=float)
    if Y.ndim != 2:
        raise ValueError(f"observation must be 2-D (T, p); got shape {Y.shape}")
    n, m = int(LatentDim), int(InputDim)

    y_mean = Y.mean(axis=0)
    Yc = Y - y_mean

    A, B, C, Q, R = _fit_subspace_model(Yc, n, m)

    # Pass 1: filter with u = 0.
    x_hat = _kalman_filter(Yc, A, B, C, Q, R, u=None)
    # Subsequent passes: estimate the input then refilter with it plugged in.
    u_hat = np.zeros((Y.shape[0], m))
    for _ in range(_N_FILTER_PASSES - 1):
        u_hat = _estimate_inputs(x_hat, A, B)
        x_hat = _kalman_filter(Yc, A, B, C, Q, R, u=u_hat)
    u_hat = _estimate_inputs(x_hat, A, B)

    return {
        "A": A, "B": B, "C": C, "Q": Q, "R": R,
        "latent": x_hat, "inputs": u_hat, "y_mean": y_mean,
    }


def estimate_latent_and_input(
    observation: np.ndarray,
    LatentDim: int,
    InputDim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate latent states and inputs from observed activity.

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
        Estimated latent states
    inputs : np.ndarray, shape (T, InputDim)
        Estimated inputs
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
    except Exception as exc:
        warnings.warn(
            f"estimate_latent_and_input: falling back to zeros ({type(exc).__name__}: {exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.zeros((T, n)), np.zeros((T, m))


# ---------------------------------------------------------------------------
# Self-test (run `python estimator.py`)
# ---------------------------------------------------------------------------
def _self_test() -> None:
    """Smoke-test the estimator on a few simulator scenarios."""
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week 1"))
    import Simulator as sim  # noqa: E402

    def recon_rmse(y, model):
        y_hat = model["latent"] @ model["C"].T + model["y_mean"]
        return float(np.sqrt(np.mean((y - y_hat) ** 2)))

    scenarios = [
        ("default_neural_system", sim.default_neural_system, 4, 16),
        ("hidden_input_system", sim.hidden_input_system, 5, 10),
        ("slow_drift_system", sim.slow_drift_system, 4, 16),
    ]
    T = 500
    print(f"Self-test (T={T}, mixed_input):")
    for name, factory, n, p in scenarios:
        system = factory(seed=0, obs_dim=p)
        u_seq = sim.mixed_input(T, system.input_dim, seed=0)
        data = system.simulate(T, U=u_seq)
        y = data["y"]
        latent, inputs = estimate_latent_and_input(y, n, 2)
        assert latent.shape == (T, n), latent.shape
        assert inputs.shape == (T, 2), inputs.shape
        model = fit_and_filter(y, n, 2)
        rmse = recon_rmse(y, model)
        y_rms = float(np.sqrt(np.mean(y ** 2)))
        print(f"  {name:24s} shapes OK  recon RMSE={rmse:.4f}  "
              f"(y RMS={y_rms:.3f}, ratio={rmse / y_rms:.3f})")
    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
