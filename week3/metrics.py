"""Evaluation metrics used by M2/M3 experiment scripts.

Pure-numpy. No imports of the rest of the package — these are used both by
white-box and Brain experiments.
"""

from __future__ import annotations

import numpy as np


def rms(arr) -> float:
    a = np.asarray(arr, dtype=float)
    return float(np.sqrt(np.mean(a ** 2)))


def saturation_fraction(U, lo: float = 0.0, hi: float = 1.0, tol: float = 1e-6) -> float:
    U = np.asarray(U, dtype=float)
    return float(np.mean((U <= lo + tol) | (U >= hi - tol)))


def control_effort(U) -> float:
    """Sum_t ||u_t||_2 — common LQR-style effort measure."""
    U = np.asarray(U, dtype=float)
    return float(np.sum(np.linalg.norm(U, axis=1)))


def settling_time(z_t, target, tol_frac: float = 0.05, reference_scale=None) -> int:
    """First step ``t`` after which ``|z(τ) - target| < tol_frac * scale`` for all τ ≥ t.

    .. deprecated::
       This "first step that stays inside the band forever" form saturates
       at ``len(z_t)`` whenever measurement noise exceeds the band width
       (see M3 — typical for noise-limited Brain regimes). Prefer
       :func:`band_occupancy_settling` for an occupancy-based settling
       metric that decouples from steady-state error.
    """
    z = np.asarray(z_t, dtype=float)
    target = np.asarray(target, dtype=float)
    err = np.linalg.norm(z - target, axis=-1)
    scale = reference_scale if reference_scale is not None else max(float(np.linalg.norm(target)), 1.0)
    threshold = tol_frac * scale
    for t in range(len(err)):
        if np.all(err[t:] < threshold):
            return t
    return len(err)


def band_occupancy_settling(
    z_t,
    target,
    tol_frac: float = 0.10,
    reference_scale=None,
    tail_frac: float = 0.5,
) -> dict:
    """Settling time as **first step the trajectory** *enters* the band,
    decoupled from steady-state error via a separate **occupancy** metric.

    Returns ``{"settle": int, "band_occupancy": float, "band": float}``::

        settle         — first ``t`` with ``||z(t) − target|| < band``;
                         ``len(z_t)`` if never entered.
        band_occupancy — fraction of the final ``tail_frac`` of steps with
                         ``||z(τ) − target|| < band``. Reads as "how often
                         we sat inside the band" — 1.0 = held perfectly,
                         0.0 = never inside, intermediate = noisy hold.
        band           — the numerical band threshold used.

    Spec §6 / M3 reframe: do NOT report a single settling number when the
    trajectory drifts in and out of the band due to noise. Report the
    "first entry" time AND the occupancy fraction separately, with the
    band size explicit.
    """
    z = np.asarray(z_t, dtype=float)
    target = np.asarray(target, dtype=float)
    err = np.linalg.norm(z - target, axis=-1)
    scale = reference_scale if reference_scale is not None else max(
        float(np.linalg.norm(target)), 1.0
    )
    band = tol_frac * scale
    # First entry into the band.
    inside = err < band
    if inside.any():
        settle = int(np.argmax(inside))
    else:
        settle = int(len(err))
    # Occupancy over the final tail.
    T = len(err)
    tail = max(1, int(round(T * tail_frac)))
    occ = float(np.mean(inside[-tail:]))
    return dict(settle=settle, band_occupancy=occ, band=float(band))


def steady_state_error(z_t, target, tail_frac: float = 0.5) -> float:
    """Mean ``||z(τ) − target||`` over the final ``tail_frac`` of the trajectory.

    The spec rule (§6): whole-trajectory RMS is dominated by the ramp-up
    transient and mislabels holding error. Report steady-state error
    *separately* from settling time, computed over a **post-settling**
    window (default: last 50 % of the trajectory). Pair this with a
    noise-floor anchor when reporting ("holds to within 1.2× per-step
    jitter").
    """
    z = np.asarray(z_t, dtype=float)
    target = np.asarray(target, dtype=float)
    T = z.shape[0]
    tail = max(1, int(round(T * tail_frac)))
    z_tail = z[-tail:]
    err = np.linalg.norm(z_tail - target, axis=-1)
    return float(np.mean(err))


def prediction_error_one_step(
    y_pred: np.ndarray, y_true: np.ndarray, noise_floor: float | None = None,
) -> dict:
    """One-step output-prediction error with an optional noise-floor anchor.

    Returns ``{"rms": rms, "rms_frac_y": rms/y_rms,
              "rms_x_noise_floor": rms/noise_floor}`` (the last is ``nan``
    if ``noise_floor`` is not given).
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    resid = y_true - y_pred
    r = float(np.sqrt(np.mean(resid ** 2)))
    y_rms = float(np.sqrt(np.mean(y_true ** 2)))
    out = {
        "rms": r,
        "rms_frac_y": r / max(y_rms, 1e-12),
        "rms_x_noise_floor": (r / max(float(noise_floor), 1e-12)
                              if noise_floor is not None else float("nan")),
    }
    return out


def spectral_radius(A) -> float:
    A = np.asarray(A, dtype=float)
    if A.size == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def state_r2_aligned(X_true, X_hat) -> float:
    """R² of ``X_hat`` against ``X_true`` after a least-squares linear alignment.

    The identified basis has a similarity-transform gauge; we factor it out by
    finding the best ``T`` with ``X_hat @ T ≈ X_true``.
    """
    X_true = np.asarray(X_true, dtype=float)
    X_hat = np.asarray(X_hat, dtype=float)
    T_align, *_ = np.linalg.lstsq(X_hat, X_true, rcond=None)
    pred = X_hat @ T_align
    ss_res = float(np.sum((X_true - pred) ** 2))
    ss_tot = float(np.sum((X_true - X_true.mean(axis=0)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def diverged(*arrays, threshold: float = 1e6) -> bool:
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            continue
        if not np.all(np.isfinite(a)):
            return True
        if np.any(np.abs(a) > threshold):
            return True
    return False
