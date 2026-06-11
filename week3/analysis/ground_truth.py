"""Determinism-differencing ground-truth model — BENCHMARK YARDSTICK ONLY.

⚠ Quarantined per spec rule #2. This module **exploits the simulator's
determinism** to cancel noise from impulse responses; the deployed honest
pipeline never imports it. The resulting model is used only as a *yardstick*
against which :mod:`estimator.identify` is graded in M3.

How it works
------------
For a stationary LGSSM driven by ``u(t)`` with seeded process noise ``w(t)``
and observation noise ``v(t)``::

    x(t+1) = A x(t) + B u(t) + a + w(t)
    y(t)   = C x(t) + c + v(t)

If we run the system twice with the same seed (so the noise sequences are
*identical*) but different inputs ``u^A(t)`` and ``u^B(t)``, the difference
``Δy(t) = y^A(t) − y^B(t)`` is **noise-free**::

    Δx(t+1) = A Δx(t) + B (u^A − u^B)(t)
    Δy(t)   = C Δx(t)

Take ``u^B ≡ 0`` and ``u^A(t) = δ_{t=0} e_i`` (unit impulse on channel ``i``
at time 0). Then ``Δy(t) = C A^{t−1} B e_i`` for ``t ≥ 1``, i.e. the ``i``-th
column of the ``t``-th Markov parameter. Repeat for every input channel and
we have the noise-free Markov sequence ``{H_1, H_2, …}``.

ERA / Ho-Kalman on the block-Hankel of these Markov parameters returns
``(A, B, C)`` exactly (in *some* basis — only basis-free invariants matter).

``R`` is read off directly: ``measure()`` redraws noise without advancing, so
repeated calls at a frozen state give a clean sample covariance.

``Q`` is fit by matching the stationary output covariance under zero input
(spec §2): ``Σ_y = C Σ_x C^T + R`` with ``Σ_x = A Σ_x A^T + Q``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

# The yardstick (this module) is allowed to reuse the estimator's internal
# filter/smoother helpers — it is quarantined and not part of the deployed
# pipeline. The honest pipeline in `estimator/identify.py` still estimates
# Q from its own single noisy probe via the full EM (it never sees the
# yardstick Q).
from estimator.identify import _kalman_filter_affine, _rts_smoother_affine


# ----------------------------------------------------------------------------
# Plant adapter — anything with `measure()`, `next_state(u)`, `input_dim`,
# `obs_dim` works (the Brain via BrainPlant, a SimulatorPlant, etc.)
# ----------------------------------------------------------------------------
PlantFactory = Callable[[int], object]
"""Callable that takes ``seed`` and returns a fresh plant instance."""


# ----------------------------------------------------------------------------
# 1. Noise-free Markov parameters via determinism differencing.
# ----------------------------------------------------------------------------
def noise_free_markov_parameters(
    plant_factory: PlantFactory,
    seed: int,
    T: int = 300,
) -> np.ndarray:
    """Return ``H`` of shape ``(T, p, m)`` with ``H[k] = C A^{k-1} B``.

    ``H[0]`` is identically zero (no direct feedthrough); ``H[1] = CB``.

    The protocol runs the plant ``m + 1`` times, each from a fresh instance
    with the same ``seed`` so the noise streams align: one zero-input
    baseline plus one unit-impulse run per input channel.
    """
    # Probe a fresh plant to learn dimensions.
    probe = plant_factory(seed)
    m = int(probe.input_dim)
    y0 = np.asarray(probe.measure(), dtype=float)
    p = y0.size

    # Zero-input baseline.
    Y_zero = _collect_outputs(plant_factory, seed, T, _zero_input_fn(m))

    # Impulse on each input channel.
    H = np.zeros((T, p, m))
    for ch in range(m):
        Y_imp = _collect_outputs(plant_factory, seed, T,
                                 _impulse_fn(m, ch))
        H[:, :, ch] = Y_imp - Y_zero  # noise-free
    return H


def _zero_input_fn(m: int):
    z = np.zeros(m)
    return lambda t: z


def _impulse_fn(m: int, channel: int):
    z = np.zeros(m)
    e = np.zeros(m); e[channel] = 1.0

    def u(t):
        return e if t == 0 else z
    return u


def _collect_outputs(plant_factory, seed, T, u_fn) -> np.ndarray:
    """Run a fresh plant for ``T`` steps with input ``u_fn(t)``; return ``Y``."""
    plant = plant_factory(seed)
    y0 = np.asarray(plant.measure(), dtype=float)
    p = y0.size
    Y = np.empty((T, p))
    Y[0] = y0
    plant.next_state(u_fn(0))
    for t in range(1, T):
        Y[t] = np.asarray(plant.measure(), dtype=float)
        plant.next_state(u_fn(t))
    return Y


# ----------------------------------------------------------------------------
# 2. ERA / Ho-Kalman on the block-Hankel of Markov parameters.
# ----------------------------------------------------------------------------
def era_ho_kalman(
    H: np.ndarray,
    n: int,
    alpha: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct ``(A, B, C)`` at order ``n`` from Markov params ``H``.

    Parameters
    ----------
    H : ndarray, shape ``(L+1, p, m)``
        Markov parameters with ``H[0]`` ignored (assumed zero) and
        ``H[k] = C A^{k-1} B`` for ``k = 1, …, L``.
    n : int
        Truncation order.
    alpha : int, optional
        Rows of the block Hankel matrix. Defaults to ``L // 2``.

    Returns
    -------
    A, B, C : ndarray
        Reconstructed system matrices (in some basis; basis-free invariants —
        eigenvalues, Markov parameters, DC gain — are exactly recovered).
    sigma : ndarray, shape ``(min(αp, βm),)``
        Singular values of the Hankel matrix. The knee location is the order.
    """
    H = np.asarray(H, dtype=float)
    L_plus_1, p, m = H.shape
    L = L_plus_1 - 1
    if alpha is None:
        alpha = L // 2
    beta = L - alpha  # we need H[1..alpha+beta] = H[1..L], plus shift
                     # H[2..alpha+beta+1] = H[2..L+1] — so the last shifted
                     # block needs H[L+1]. We construct the block Hankel up
                     # to alpha+beta-1 = L-1 and the shifted up to L.
    # Re-budget so we can include the shift without indexing past H[L]:
    if alpha + beta > L:
        beta = L - alpha
    if beta < 1:
        raise ValueError(
            f"too few Markov params (L={L}) to form a block Hankel with "
            f"alpha={alpha}"
        )

    Hkl = np.zeros((alpha * p, beta * m))
    Hsh = np.zeros((alpha * p, beta * m))
    for i in range(alpha):
        for j in range(beta):
            Hkl[i * p:(i + 1) * p, j * m:(j + 1) * m] = H[i + j + 1]
            Hsh[i * p:(i + 1) * p, j * m:(j + 1) * m] = H[i + j + 2]

    U, sigma, Vt = np.linalg.svd(Hkl, full_matrices=False)
    rank = int(min(n, sigma.size))
    Un = U[:, :rank]
    Sn = sigma[:rank]
    Vtn = Vt[:rank, :]
    sqrt_S = np.sqrt(Sn)
    inv_sqrt_S = 1.0 / sqrt_S

    O = Un * sqrt_S[None, :]       # extended observability (alpha*p, n)
    R_ext = sqrt_S[:, None] * Vtn   # extended controllability (n, beta*m)

    A = (Un.T @ Hsh @ Vtn.T) * (inv_sqrt_S[:, None] * inv_sqrt_S[None, :])
    C = O[:p, :]
    B = R_ext[:, :m]
    return A, B, C, sigma


# ----------------------------------------------------------------------------
# 3. Observation-noise covariance R from repeated frozen-state measurements.
# ----------------------------------------------------------------------------
def measurement_noise_covariance(
    plant_factory: PlantFactory,
    seed: int,
    n_samples: int = 2000,
) -> np.ndarray:
    """``R`` from ``n_samples`` repeated ``measure()`` calls at a frozen state.

    ``measure()`` redraws ``v(t)`` without advancing the state, so the sample
    covariance is exactly ``R`` (plus 1/√N empirical noise).
    """
    plant = plant_factory(seed)
    y0 = np.asarray(plant.measure(), dtype=float)
    p = y0.size
    Y = np.empty((n_samples, p))
    Y[0] = y0
    for k in range(1, n_samples):
        Y[k] = np.asarray(plant.measure(), dtype=float)
    return np.cov(Y, rowvar=False, ddof=1)


# ----------------------------------------------------------------------------
# 4. Process-noise covariance Q by EM with (A, B, C, R, a, c) frozen.
# ----------------------------------------------------------------------------
def _q_only_m_step(
    x_smooth: np.ndarray, P_smooth: np.ndarray, Plag: np.ndarray,
    U: np.ndarray,
    A: np.ndarray, B: np.ndarray, a: np.ndarray,
) -> np.ndarray:
    """Closed-form M-step for Q with everything else held fixed.

    Q = (1/(T−1)) Σ_{t=0}^{T−2} E[(x_{t+1} − A x_t − B u_t − a)
                                  (x_{t+1} − A x_t − B u_t − a)^T | Y]

    Expanding in smoother sufficient statistics, with ``b_t = B u_t + a``::

        E[e_t e_t^T] = (P^s_{t+1} + xs_{t+1} xs_{t+1}^T)
                     − (L_t + xs_{t+1} xs_t^T) A^T
                     − A (L_t + xs_{t+1} xs_t^T)^T
                     − xs_{t+1} b_t^T − b_t xs_{t+1}^T
                     + A (P^s_t + xs_t xs_t^T) A^T
                     + A xs_t b_t^T + b_t xs_t^T A^T
                     + b_t b_t^T

    where ``L_t = Plag[t] = Cov(x_{t+1}, x_t | Y)``.
    """
    T, n = x_smooth.shape
    Q = np.zeros((n, n))
    for t in range(T - 1):
        xs_t = x_smooth[t]
        xs_tp1 = x_smooth[t + 1]
        b_t = B @ U[t] + a

        Exx_tp1 = P_smooth[t + 1] + np.outer(xs_tp1, xs_tp1)
        Exx_t = P_smooth[t] + np.outer(xs_t, xs_t)
        Exnx = Plag[t] + np.outer(xs_tp1, xs_t)

        Q_t = (
            Exx_tp1
            - Exnx @ A.T - A @ Exnx.T
            - np.outer(xs_tp1, b_t) - np.outer(b_t, xs_tp1)
            + A @ Exx_t @ A.T
            + np.outer(A @ xs_t, b_t) + np.outer(b_t, A @ xs_t)
            + np.outer(b_t, b_t)
        )
        Q += Q_t
    Q /= max(T - 1, 1)
    Q = 0.5 * (Q + Q.T)
    # Numerical hygiene: clip any tiny negative eigenvalues (sample noise).
    w, V = np.linalg.eigh(Q)
    w = np.clip(w, 1e-10, None)
    return (V * w) @ V.T


def process_noise_covariance(
    plant_factory: PlantFactory,
    A: np.ndarray, B: np.ndarray, C: np.ndarray, R: np.ndarray,
    seed: int,
    a: Optional[np.ndarray] = None,
    c: Optional[np.ndarray] = None,
    T: int = 4000,
    burn_in: int = 200,
    n_iter: int = 20,
    tol: float = 1e-4,
    Q_init: Optional[np.ndarray] = None,
    probe_seed: Optional[int] = None,
) -> np.ndarray:
    """``Q`` via EM with ``(A, B, C, R, a, c)`` frozen at the yardstick values.

    Spec §2 option (a): "Get Q by EM with (A,B,C,R) fixed". We run a single
    noisy probe (``u ~ Uniform[0,1]^m``) for ``T`` steps, then iterate

        E-step:  Kalman filter + RTS smoother with current Q;
        M-step:  closed-form Q from smoothed sufficient statistics;

    holding everything except Q constant. Convergence in 5–20 iterations is
    typical because Q is small relative to R (the Kalman gain barely moves
    between iterations). The probe is post-burn-in only — the first
    ``burn_in`` samples are discarded so the chain has reached stationarity.

    The previous Σ_y back-substitution (``Σ_x = C⁺ (Σ_y − R) C⁺ᵀ``) was
    high-variance because it inverts an ill-conditioned C; this M-step
    formulation absorbs that ill-conditioning into the optimal smoother
    (which is well-posed) instead.
    """
    n = A.shape[0]
    p = C.shape[0]
    m = B.shape[1]
    if a is None:
        a = np.zeros(n)
    if c is None:
        c = np.zeros(p)
    if Q_init is None:
        Q_init = 0.1 * np.eye(n)
    rng = np.random.default_rng(seed if probe_seed is None else probe_seed)

    # One long noisy probe, post-burn-in.
    U_full = rng.uniform(0.0, 1.0, size=(T + burn_in, m))

    plant_seed = seed
    plant = plant_factory(plant_seed)
    Y_full = np.empty((T + burn_in, p))
    Y_full[0] = np.asarray(plant.measure(), dtype=float)
    plant.next_state(U_full[0])
    for t in range(1, T + burn_in):
        Y_full[t] = np.asarray(plant.measure(), dtype=float)
        plant.next_state(U_full[t])
    Y = Y_full[burn_in:]
    U = U_full[burn_in:]

    # Stationary-mean init for the filter.
    try:
        x0_mean = np.linalg.solve(np.eye(n) - A, a)
    except np.linalg.LinAlgError:
        x0_mean = np.zeros(n)

    Q = Q_init.copy()
    prev_tr = None
    for it in range(n_iter):
        # Update P0 to the steady-state Lyapunov solution under current Q —
        # spurious initial transient in the filter then cancels out.
        try:
            from scipy.linalg import solve_discrete_lyapunov
            P0 = solve_discrete_lyapunov(A, Q)
            P0 = 0.5 * (P0 + P0.T)
        except Exception:
            P0 = 100.0 * np.eye(n)
        x_filt, P_filt, x_pred, P_pred, _ll, K_last = _kalman_filter_affine(
            Y, U, A, B, C, Q, R, a, c, x0_mean, P0
        )
        x_smooth, P_smooth, Plag = _rts_smoother_affine(
            x_filt, P_filt, x_pred, P_pred, A, C, K_last
        )
        Q_new = _q_only_m_step(x_smooth, P_smooth, Plag, U, A, B, a)
        tr_new = float(np.trace(Q_new))
        if prev_tr is not None and abs(tr_new - prev_tr) / max(abs(prev_tr), 1e-12) < tol:
            Q = Q_new
            break
        Q = Q_new
        prev_tr = tr_new
    return Q


# ----------------------------------------------------------------------------
# 5. Resting-state baseline ``c`` (output mean under zero input).
# ----------------------------------------------------------------------------
def resting_output(
    plant_factory: PlantFactory,
    seed: int,
    T: int = 1000,
    burn_in: int = 200,
) -> np.ndarray:
    """Mean of ``y`` under ``u ≡ 0`` after burn-in — defines the ``c`` gauge.

    We adopt the gauge ``a = 0`` (so ``x_ss = 0`` under ``u = 0``) and
    ``c = y_ss``. Together with the recovered ``(A, B, C)`` this gives a
    complete affine model in the ERA basis.
    """
    m = int(plant_factory(seed).input_dim)
    Y = _collect_outputs(plant_factory, seed, T, _zero_input_fn(m))
    return Y[burn_in:].mean(axis=0)


# ----------------------------------------------------------------------------
# 6. Headline orchestrator — assembles the full ground-truth model.
# ----------------------------------------------------------------------------
@dataclass
class GroundTruthModel:
    """Yardstick model recovered by determinism differencing + ERA.

    Stored in the *ERA basis*; only basis-free invariants are meaningful for
    comparison (eigenvalues, Markov parameters, DC gain, held-out output
    prediction). Loaded by experiments in M3 — never by the deployed pipeline.
    """
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    a: np.ndarray
    c: np.ndarray
    hankel_sigma: np.ndarray  # singular values of the block Hankel (for the knee)
    n: int                    # order used
    seed: int                 # the seed this model was identified from
    T_impulse: int            # length of each impulse-response run

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            A=self.A, B=self.B, C=self.C, Q=self.Q, R=self.R,
            a=self.a, c=self.c,
            hankel_sigma=self.hankel_sigma,
            n=self.n, seed=self.seed, T_impulse=self.T_impulse,
        )

    @classmethod
    def load(cls, path: Path) -> "GroundTruthModel":
        d = np.load(Path(path))
        return cls(
            A=d["A"], B=d["B"], C=d["C"], Q=d["Q"], R=d["R"],
            a=d["a"], c=d["c"],
            hankel_sigma=d["hankel_sigma"],
            n=int(d["n"]), seed=int(d["seed"]),
            T_impulse=int(d["T_impulse"]),
        )


def build_ground_truth_model(
    plant_factory: PlantFactory,
    seed: int,
    n: int = 6,
    T_impulse: int = 300,
    n_noise_samples: int = 2000,
    T_q: int = 4000,
) -> GroundTruthModel:
    """Run the full ⚠ yardstick recipe (spec §2) on ``plant_factory(seed)``.

    Returns a :class:`GroundTruthModel` in the ERA basis with the gauge
    ``a = 0`` and ``c = y_ss``. Only basis-free invariants are meaningful.
    """
    H = noise_free_markov_parameters(plant_factory, seed=seed, T=T_impulse)
    A, B, C, sigma = era_ho_kalman(H, n=n)
    R = measurement_noise_covariance(plant_factory, seed=seed,
                                     n_samples=n_noise_samples)
    c = resting_output(plant_factory, seed=seed)
    a = np.zeros(A.shape[0])
    Q = process_noise_covariance(plant_factory,
                                  A=A, B=B, C=C, R=R, a=a, c=c,
                                  seed=seed, T=T_q)
    return GroundTruthModel(
        A=A, B=B, C=C, Q=Q, R=R, a=a, c=c,
        hankel_sigma=sigma, n=n, seed=seed, T_impulse=T_impulse,
    )
