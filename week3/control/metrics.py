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

    ``scale`` defaults to ``max(||target||, 1)``. Returns ``len(z_t)`` if never settled.
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
