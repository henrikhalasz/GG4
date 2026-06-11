"""identify.py -- offline system identification of the brain (Week 4, Step 1).

Pipeline (week4_step1_estimator.md S6):
  1. Generate richly-excited data by driving the live Brain over the multi-probe
     schedule (excitation.py), using the Week-3 timing convention.
  2. De-bias every observation by the resting sensor bias (S5): bias = mean(y) over the
     zero-input rest segment, subtracted everywhere. We fit a ZERO-OFFSET LGSSM
         x_{t+1} = A x_t + B u_t + w,   w ~ N(0, Q)
         y_t     = C x_t + v,           v ~ N(0, R)
     -- no state offset a (process-noise mean is zero, so a ~ 0; we confirm with a
     free-a EM check), no output offset c (removed in preprocessing).
  3. Subspace (N4SID oblique projection) warm start at the fixed order n = 6.
  4. EM refine (Ghahramani-Hinton, known inputs, zero offsets) to convergence; the
     log-likelihood must be monotone non-decreasing.
  5. Save identified_model.npz.

Also: the uniform-only vs full-schedule ablation (did rich excitation earn its place?),
the free-a confirmation, and the excitation_inputs.pdf figure.

This file is self-contained: it mirrors the *method* of week3/estimator/identify.py but
imports nothing from Week 3, and never imports true_brain_params.py.

Timing convention (matches week3/control/control_interface.py and the Week-4 notebook's
run_closed_loop): at step t we measure() -> y[t], then next_state(u[t]); u[t] drives
x_t -> x_{t+1}; the filter predict at step t consumes u[t-1]; y[0] is the first observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from GG4 import Brain  # the live brain (the only "system" we get to see)

import excitation as exc

HERE = Path(__file__).resolve().parent

# --- Fixed configuration -----------------------------------------------------
ORDER = 6                 # the true order; we do NOT sweep (S3)
OBS_DIM = 16
INPUT_DIM = 2

# Step budget (S4) -- named constants, printed at runtime.
T_REST = 3000             # zero-input recording for the sensor bias (S5)
T_PRBS = 4000             # per channel x2  => 8000 (input-0-only, then input-1-only)
T_CHIRP = 4000            # per channel x2  => 8000
T_MSINE = 3000            # both channels, decorrelated
T_LPN = 3000              # both channels, decorrelated
T_TRAIN = T_REST + 2 * T_PRBS + 2 * T_CHIRP + T_MSINE + T_LPN   # ~25,000
T_VAL = 4000              # held-out, both-channel decorrelated, different seed

SEED_TRAIN = 0
SEED_VAL = 12345          # different seed -> different noise realisation, same system

EM_MAX_ITER = 120
EM_TOL = 1e-5

_JITTER = 1e-8
_COV_FLOOR = 1e-6
_RHO_CAP = 1.005


# ---------------------------------------------------------------------------
# Numerical helpers (mirrors week3/estimator/identify.py)
# ---------------------------------------------------------------------------
def _sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _safe_inv(M: np.ndarray, jitter: float = _JITTER) -> np.ndarray:
    return np.linalg.inv(M + jitter * np.eye(M.shape[0]))


def _regularize_cov(M: np.ndarray, eps: float = _COV_FLOOR) -> np.ndarray:
    return _sym(M) + eps * np.eye(M.shape[0])


def _logpdf_mvn_zero(x: np.ndarray, S: np.ndarray) -> float:
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


def _spectral_cap(A: np.ndarray, cap: float = _RHO_CAP) -> np.ndarray:
    eig = np.linalg.eigvals(A)
    rho = float(np.max(np.abs(eig))) if eig.size else 0.0
    return A * (0.99 / rho) if rho > cap else A


# ---------------------------------------------------------------------------
# Subspace (N4SID) warm start -- zero-offset version
# ---------------------------------------------------------------------------
def _block_hankel(X: np.ndarray, k: int, t0: int, j: int) -> np.ndarray:
    T, d = X.shape
    if t0 + k + j - 1 > T:
        raise ValueError(f"block-Hankel out of range: t0={t0}, k={k}, j={j}, T={T}")
    H = np.empty((k * d, j))
    for i in range(k):
        H[i * d:(i + 1) * d, :] = X[t0 + i:t0 + i + j, :].T
    return H


def _n4sid_init(Y: np.ndarray, U: np.ndarray, n: int, k: Optional[int] = None) -> Dict:
    """N4SID input-output subspace init (oblique projection of future outputs onto
    past inputs+outputs along future inputs). Returns a zero-offset warm start.

    Y is already de-biased; we still mean-centre for the SVD conditioning, but the
    final regressions carry NO constant term (a = 0, c = 0). ``k`` is the past/future
    block horizon (larger k resolves closely-spaced / weak modes better).
    """
    T, p = Y.shape
    m = U.shape[1]
    if k is None:
        k = max(2 * n, 3)
        while T - 2 * k + 1 < 3 * (m + p) * k and k > n + 1:
            k -= 1
    j = T - 2 * k + 1

    Yc = Y - Y.mean(axis=0)
    U_p = _block_hankel(U, k, 0, j)
    U_f = _block_hankel(U, k, k, j)
    Y_p = _block_hankel(Yc, k, 0, j)
    Y_f = _block_hankel(Yc, k, k, j)
    W_p = np.vstack([U_p, Y_p])
    n_w = W_p.shape[0]
    Z = np.vstack([W_p, U_f])
    coef, *_ = np.linalg.lstsq(Z.T, Y_f.T, rcond=None)
    O_i = coef[:n_w, :].T @ W_p

    U_svd, S_svd, _ = np.linalg.svd(O_i, full_matrices=False)
    sqrt_s = np.sqrt(np.maximum(S_svd[:n], 1e-12))
    Gamma = U_svd[:, :n] * sqrt_s
    C_sub = Gamma[:p, :]
    A_sub, *_ = np.linalg.lstsq(Gamma[:-p, :], Gamma[p:, :], rcond=None)
    A_sub = _spectral_cap(A_sub)

    # Rough state trajectory: zero-offset Kalman predictor with placeholder Q, R.
    Q0, R0 = np.eye(n), np.eye(p)
    x_filt, *_ = _kalman_filter(Y, U, A_sub, np.zeros((n, m)), C_sub, Q0, R0,
                                np.zeros(n), np.eye(n))

    # Zero-offset regressions on the N4SID state sequence.
    rt = np.hstack([x_filt[:-1], U[:-1]])           # [x_t; u_t]  (no constant)
    dyn, *_ = np.linalg.lstsq(rt, x_filt[1:], rcond=None)
    dyn = dyn.T
    A_w = _spectral_cap(dyn[:, :n])
    B_w = dyn[:, n:n + m]

    C_w, *_ = np.linalg.lstsq(x_filt, Y, rcond=None)  # y = C x  (no constant)
    C_w = C_w.T

    Q_resid = x_filt[1:] - rt @ dyn.T
    Q_w = _regularize_cov(np.cov(Q_resid.T, ddof=0))
    R_resid = Y - x_filt @ C_w.T
    R_w = _regularize_cov(np.cov(R_resid.T, ddof=0))

    return dict(A=A_w, B=B_w, C=C_w, Q=Q_w, R=R_w,
                x0_mean=x_filt[0].copy(), P0=np.eye(n), hankel_sigma=S_svd)


# ---------------------------------------------------------------------------
# Kalman filter / RTS smoother (zero offset, optional free state-offset a)
# ---------------------------------------------------------------------------
def _kalman_filter(
    Y, U, A, B, C, Q, R, x0_mean, P0, a: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    T, p = Y.shape
    n = A.shape[0]
    if a is None:
        a = np.zeros(n)
    x_filt = np.empty((T, n)); P_filt = np.empty((T, n, n))
    x_pred = np.empty((T, n)); P_pred = np.empty((T, n, n))
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
        innov = Y[t] - C @ x_pred[t]
        S = _sym(C @ P_pred[t] @ C.T + R)
        loglik += _logpdf_mvn_zero(innov, S)
        K = P_pred[t] @ C.T @ _safe_inv(S)
        x_filt[t] = x_pred[t] + K @ innov
        P_filt[t] = _sym((eye_n - K @ C) @ P_pred[t])
        if t == T - 1:
            K_last = K
    return x_filt, P_filt, x_pred, P_pred, loglik, K_last


def _rts_smoother(x_filt, P_filt, x_pred, P_pred, A, C, K_last):
    T, n = x_filt.shape
    x_smooth = np.empty_like(x_filt); P_smooth = np.empty_like(P_filt)
    J = np.empty((max(T - 1, 0), n, n))
    x_smooth[-1] = x_filt[-1]; P_smooth[-1] = P_filt[-1]
    for t in range(T - 2, -1, -1):
        J[t] = P_filt[t] @ A.T @ _safe_inv(P_pred[t + 1])
        x_smooth[t] = x_filt[t] + J[t] @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = _sym(P_filt[t] + J[t] @ (P_smooth[t + 1] - P_pred[t + 1]) @ J[t].T)
    Plag = np.zeros((max(T - 1, 0), n, n))
    if T >= 2:
        eye_n = np.eye(n)
        Plag[T - 2] = (eye_n - K_last @ C) @ A @ P_filt[T - 2]
        for t in range(T - 3, -1, -1):
            Plag[t] = P_filt[t + 1] @ J[t].T + J[t + 1] @ (Plag[t + 1] - A @ P_filt[t + 1]) @ J[t].T
    return x_smooth, P_smooth, Plag


def _m_step(x_smooth, P_smooth, Plag, Y, U, free_a: bool = False):
    """Closed-form M-step. c = 0 always (de-biased y); a = 0 unless free_a."""
    T, n = x_smooth.shape
    p = Y.shape[1]; m = U.shape[1]
    Trr = T - 1

    Sxx = np.zeros((n, n)); Sxnx = np.zeros((n, n)); Sxnxn = np.zeros((n, n))
    Sxu = np.zeros((n, m)); Sxn_u = np.zeros((n, m))
    Sxn = np.zeros(n); Sx = np.zeros(n)
    Suu = np.zeros((m, m)); Su = np.zeros(m)
    for t in range(Trr):
        xs_t, xs_tp1 = x_smooth[t], x_smooth[t + 1]
        Sxx += P_smooth[t] + np.outer(xs_t, xs_t)
        Sxnx += Plag[t] + np.outer(xs_tp1, xs_t)
        Sxnxn += P_smooth[t + 1] + np.outer(xs_tp1, xs_tp1)
        Sxu += np.outer(xs_t, U[t])
        Sxn_u += np.outer(xs_tp1, U[t])
        Sxn += xs_tp1; Sx += xs_t
        Suu += np.outer(U[t], U[t]); Su += U[t]

    if free_a:
        d = n + m + 1
        Srr = np.zeros((d, d)); Sxnr = np.zeros((n, d))
        Srr[:n, :n] = Sxx; Srr[:n, n:n + m] = Sxu; Srr[:n, n + m] = Sx
        Srr[n:n + m, :n] = Sxu.T; Srr[n:n + m, n:n + m] = Suu; Srr[n:n + m, n + m] = Su
        Srr[n + m, :n] = Sx; Srr[n + m, n:n + m] = Su; Srr[n + m, n + m] = float(Trr)
        Sxnr[:, :n] = Sxnx; Sxnr[:, n:n + m] = Sxn_u; Sxnr[:, n + m] = Sxn
    else:
        d = n + m
        Srr = np.zeros((d, d)); Sxnr = np.zeros((n, d))
        Srr[:n, :n] = Sxx; Srr[:n, n:] = Sxu
        Srr[n:, :n] = Sxu.T; Srr[n:, n:] = Suu
        Sxnr[:, :n] = Sxnx; Sxnr[:, n:] = Sxn_u

    beta = Sxnr @ _safe_inv(Srr, jitter=1e-10)
    A_new = beta[:, :n]
    B_new = beta[:, n:n + m]
    a_new = beta[:, n + m] if free_a else np.zeros(n)
    Q_new = _regularize_cov((Sxnxn - beta @ Sxnr.T) / max(Trr, 1))

    # Observation: y = C x (no constant).
    Sxx_all = np.zeros((n, n)); Syx = np.zeros((p, n)); Syy = np.zeros((p, p))
    for t in range(T):
        xs_t = x_smooth[t]
        Sxx_all += P_smooth[t] + np.outer(xs_t, xs_t)
        Syx += np.outer(Y[t], xs_t)
        Syy += np.outer(Y[t], Y[t])
    C_new = Syx @ _safe_inv(Sxx_all)
    R_new = _regularize_cov((Syy - C_new @ Syx.T) / T)

    return A_new, B_new, a_new, C_new, Q_new, R_new


# ---------------------------------------------------------------------------
# Fit driver
# ---------------------------------------------------------------------------
@dataclass
class FitResult:
    A: np.ndarray; B: np.ndarray; C: np.ndarray; Q: np.ndarray; R: np.ndarray
    a: np.ndarray; x0_mean: np.ndarray; P0: np.ndarray
    loglik_hist: List[float] = field(default_factory=list)
    iters: int = 0
    hankel_sigma: Optional[np.ndarray] = None
    x_smooth: Optional[np.ndarray] = None

    @property
    def eig(self) -> np.ndarray:
        return np.linalg.eigvals(self.A)

    @property
    def rho(self) -> float:
        return float(np.max(np.abs(self.eig)))


def fit(Y, U, n: int = ORDER, max_iter: int = EM_MAX_ITER, tol: float = EM_TOL,
        free_a: bool = False, verbose: bool = False) -> FitResult:
    Y = np.asarray(Y, float); U = np.asarray(U, float)
    warm = _n4sid_init(Y, U, n)
    A, B, C = warm["A"], warm["B"], warm["C"]
    Q, R = warm["Q"], warm["R"]
    a = np.zeros(n)
    x0_mean, P0 = warm["x0_mean"], warm["P0"]

    ll_hist: List[float] = []
    prev_ll = -np.inf
    iters = 0
    for it in range(max_iter):
        iters = it + 1
        xf, Pf, xp, Pp, ll, Kl = _kalman_filter(Y, U, A, B, C, Q, R, x0_mean, P0, a)
        xs, Ps, Plag = _rts_smoother(xf, Pf, xp, Pp, A, C, Kl)
        ll_hist.append(float(ll))
        if verbose:
            print(f"    [EM] iter {it:3d}: ll = {ll:+.2f}")
        if it > 0 and (ll - prev_ll) < tol * max(abs(prev_ll), 1.0):
            break
        prev_ll = ll
        A, B, a, C, Q, R = _m_step(xs, Ps, Plag, Y, U, free_a=free_a)
        x0_mean = xs[0].copy(); P0 = _regularize_cov(Ps[0])

    xf, Pf, xp, Pp, ll, Kl = _kalman_filter(Y, U, A, B, C, Q, R, x0_mean, P0, a)
    xs, Ps, Plag = _rts_smoother(xf, Pf, xp, Pp, A, C, Kl)
    ll_hist.append(float(ll))
    return FitResult(A=A, B=B, C=C, Q=Q, R=R, a=a, x0_mean=x0_mean, P0=P0,
                     loglik_hist=ll_hist, iters=iters,
                     hankel_sigma=warm["hankel_sigma"], x_smooth=xs)


# ---------------------------------------------------------------------------
# Data generation against the live Brain
# ---------------------------------------------------------------------------
def generate_data(seed: int, U: np.ndarray) -> np.ndarray:
    """Drive Brain(seed) over input sequence U with the Week-3 timing convention.

    y[0] = measure(); next_state(u[0]); for t>=1: y[t] = measure(); next_state(u[t]).
    Returns Y of shape (T, 16).
    """
    brain = Brain(random_seed=seed)
    T = U.shape[0]
    Y = np.empty((T, OBS_DIM))
    Y[0] = brain.measure()
    brain.next_state(U[0])
    for t in range(1, T):
        Y[t] = brain.measure()
        brain.next_state(U[t])
    return Y


def one_step_prediction(model: FitResult, Yd: np.ndarray, U: np.ndarray) -> Tuple[float, float]:
    """Held-out one-step prediction on de-biased Yd: returns (VAF, RMS)."""
    xf, Pf, xp, Pp, _, _ = _kalman_filter(Yd, U, model.A, model.B, model.C,
                                          model.Q, model.R, model.x0_mean, model.P0, model.a)
    y_pred = xp @ model.C.T
    resid = Yd - y_pred
    rms = float(np.sqrt(np.mean(resid ** 2)))
    vaf = 1.0 - float(np.sum(resid ** 2) / np.sum((Yd - Yd.mean(0)) ** 2))
    return vaf, rms


# ---------------------------------------------------------------------------
# Excitation figure (S4 deliverable)
# ---------------------------------------------------------------------------
def make_excitation_figure(sch: exc.Schedule, U_val: np.ndarray, path: Path) -> None:
    U_full = np.vstack([sch.U, U_val])
    segs = list(sch.segments) + [("validation", sch.T, sch.T + U_val.shape[0])]
    Ttot = U_full.shape[0]

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 0.8, 1.0], hspace=0.45)
    ax0, ax1, axf, axz = (fig.add_subplot(gs[i]) for i in range(4))

    for ax, ch, lab in ((ax0, 0, "u0(t)  (input 0)"), (ax1, 1, "u1(t)  (input 1)")):
        ax.plot(U_full[:, ch], lw=0.4, color="C0")
        ax.set_ylabel(lab); ax.set_ylim(-0.05, 1.05); ax.set_xlim(0, Ttot)
        for name, s, e in segs:
            ax.axvline(s, color="0.6", lw=0.6, ls="--")
            ax.text(0.5 * (s + e), 1.06, name, ha="center", va="bottom",
                    fontsize=7, rotation=0, transform=ax.get_xaxis_transform())
    ax0.set_title("Excitation schedule -- full input time series across all segments")

    # Chirp instantaneous frequency placed at the segments' global times.
    axf.set_ylabel("chirp freq\n(cyc/step)"); axf.set_xlim(0, Ttot)
    for name, s, e in segs:
        if name in sch.chirp_freq:
            axf.semilogy(np.arange(s, e), sch.chirp_freq[name], color="C3", lw=1.0)
            axf.text(0.5 * (s + e), sch.chirp_freq[name].max(), name,
                     ha="center", va="bottom", fontsize=7)
    axf.axhline(1.0 / 36.5, color="0.5", ls=":", lw=0.8)
    axf.text(Ttot * 0.01, 1.0 / 36.5, " resonance ~1/36.5", fontsize=7, va="bottom", color="0.4")
    axf.set_title("Chirp instantaneous frequency (log scale)")

    # Zoom: a few hundred steps inside the chirp-1 segment (clear swept waveform).
    name_z = "chirp-1"
    s_z = next(s for nm, s, _ in segs if nm == name_z)
    z0, z1 = s_z + 200, s_z + 700
    axz.plot(np.arange(z0, z1), U_full[z0:z1, 0], lw=0.7, label="u0")
    axz.plot(np.arange(z0, z1), U_full[z0:z1, 1], lw=0.7, label="u1")
    axz.set_xlim(z0, z1); axz.set_ylim(-0.05, 1.05)
    axz.set_xlabel("step"); axz.set_ylabel("u")
    axz.legend(loc="upper right", fontsize=8)
    axz.set_title(f"Zoom: steps {z0}-{z1} (within {name_z})")

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    print("=" * 72)
    print("STEP BUDGET (S4)")
    print(f"  T_REST  = {T_REST:5d}  (bias)")
    print(f"  T_PRBS  = {T_PRBS:5d}  x2 channels = {2 * T_PRBS}")
    print(f"  T_CHIRP = {T_CHIRP:5d}  x2 channels = {2 * T_CHIRP}")
    print(f"  T_MSINE = {T_MSINE:5d}")
    print(f"  T_LPN   = {T_LPN:5d}")
    print(f"  => training total       = {T_TRAIN}")
    print(f"  T_VAL   = {T_VAL:5d}  (held-out, seed {SEED_VAL})")
    print("=" * 72)

    # --- Excitation + data ---------------------------------------------------
    sch = exc.build_schedule(seed=SEED_TRAIN, t_rest=T_REST, t_prbs=T_PRBS,
                             t_chirp=T_CHIRP, t_msine=T_MSINE, t_lpn=T_LPN)
    assert sch.T == T_TRAIN, f"schedule length {sch.T} != budget {T_TRAIN}"
    U_val = exc.validation_inputs(seed=SEED_VAL, T=T_VAL)
    print(f"clipping fraction (training): {sch.clip_fraction():.2e}  (must be ~0)")

    print("driving the live Brain ...")
    Y_raw = generate_data(SEED_TRAIN, sch.U)
    Y_val_raw = generate_data(SEED_VAL, U_val)

    # --- De-bias (S5) --------------------------------------------------------
    rest_s, rest_e = sch.segments[0][1], sch.segments[0][2]
    bias = Y_raw[rest_s:rest_e].mean(axis=0)
    Yd = Y_raw - bias
    Y_val_d = Y_val_raw - bias
    print("\nDE-BIAS (S5)")
    print(f"  bias = mean(y) over rest [{rest_s}:{rest_e}]:")
    print(f"    {bias}")
    print(f"  residual mean of de-biased y over rest segment: "
          f"{np.abs(Yd[rest_s:rest_e].mean(0)).max():.2e}  (max |.|, ~0 by construction)")
    print(f"  residual mean of de-biased y over WHOLE training run: "
          f"{Yd.mean(0)}")
    print(f"    (non-zero: the input-driven DC, which C x explains -- not a sensor offset)")

    # --- Subspace + EM (full schedule) --------------------------------------
    print("\nFITTING full schedule (N4SID init -> EM) ...")
    full = fit(Yd, sch.U, n=ORDER, verbose=False)
    sig = full.hankel_sigma
    print(f"  Hankel singular values (first {ORDER + 3}): {sig[:ORDER + 3]}")
    print(f"    sigma ratio across the n=6 cliff: sigma6/sigma7 = {sig[ORDER - 1] / sig[ORDER]:.1f}")
    print(f"  EM iterations: {full.iters},  final log-lik: {full.loglik_hist[-1]:.1f}")
    dll = np.diff(full.loglik_hist)
    print(f"  min EM log-lik increment: {dll.min():+.3e}  (must be >= ~0; a drop is a bug)")
    print(f"  spectral radius rho(A_id) = {full.rho:.4f}")
    eig_full = full.eig
    print(f"  recovered eigenvalues:\n    {eig_full}")
    print(f"  eigenvalue magnitudes (sorted): {np.sort(np.abs(eig_full))}")
    print(f"    expected (truth, approx):     [0.017 0.824 0.943 0.951 0.951 0.961]")

    # --- Free-a confirmation (S5) -------------------------------------------
    print("\nFREE-a CHECK (S5): re-fit with the state offset a free; expect ||a|| ~ 0")
    free = fit(Yd, sch.U, n=ORDER, free_a=True)
    print(f"  ||a|| = {np.linalg.norm(free.a):.3e}   a = {free.a}")

    # --- Ablation: uniform-only vs full schedule (S4) ------------------------
    print("\nABLATION: uniform-only vs full schedule (did rich excitation earn its place?)")
    U_uni = exc.uniform_schedule(seed=SEED_TRAIN, T=T_TRAIN)
    Y_uni_raw = generate_data(SEED_TRAIN, U_uni)
    Y_uni_d = Y_uni_raw - bias                    # same system -> reuse the measured bias
    uni = fit(Y_uni_d, U_uni, n=ORDER)
    print(f"  uniform-only: EM iters {uni.iters}, rho {uni.rho:.4f}")
    print(f"    eigenvalue magnitudes: {np.sort(np.abs(uni.eig))}")
    vaf_full, rms_full = one_step_prediction(full, Y_val_d, U_val)
    vaf_uni, rms_uni = one_step_prediction(uni, Y_val_d, U_val)
    print(f"  held-out one-step prediction on the SAME validation set (seed {SEED_VAL}):")
    print(f"    full schedule : VAF = {vaf_full:.4f}   RMS = {rms_full:.4f}")
    print(f"    uniform only  : VAF = {vaf_uni:.4f}   RMS = {rms_uni:.4f}")

    # --- Save ----------------------------------------------------------------
    out = HERE / "identified_model.npz"
    np.savez(
        out,
        A=full.A, B=full.B, C=full.C, Q=full.Q, R=full.R,
        x0=full.x0_mean, P0=full.P0, bias=bias,
        order=ORDER, seed_train=SEED_TRAIN, seed_val=SEED_VAL,
        T_train=T_TRAIN, T_val=T_VAL,
        loglik_hist=np.array(full.loglik_hist), hankel_sigma=sig,
        segment_names=np.array([s[0] for s in sch.segments]),
        eig_uniform=uni.eig,
    )
    print(f"\nsaved -> {out}")

    # --- Figures -------------------------------------------------------------
    fig_exc = HERE / "excitation_inputs.pdf"
    make_excitation_figure(sch, U_val, fig_exc)
    print(f"saved -> {fig_exc}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(full.loglik_hist, marker=".", ms=3)
    ax.set_xlabel("EM iteration"); ax.set_ylabel("log-likelihood")
    ax.set_title("EM log-likelihood (full schedule) -- monotone non-decreasing")
    fig.savefig(HERE / "em_loglik.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {HERE / 'em_loglik.pdf'}")


if __name__ == "__main__":
    main()
