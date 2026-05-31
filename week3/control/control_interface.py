"""IdentifiedSystem — the ONLY importer of ``estimator_new``.

Spec rule #2: controllers / observer / reachability / driver consume only
``(A, B, C, Q, R, a, c)``. This file is the lone bridge to the EM identifier.

Calibration:
    1. Drive ``plant`` with a Uniform[0,1] probe for ``T_cal`` steps.
    2. Fit ``EstimatorNew`` (affine LGSSM, EM with known inputs).
    3. Record diagnostics + degenerate-identification flags.

No dither, no online re-identification (spec rule #3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add week3/ to sys.path so we can import the sibling estimator_new module.
_WEEK3_DIR = Path(__file__).resolve().parent.parent
_week3_str = str(_WEEK3_DIR)
if _week3_str not in sys.path:
    sys.path.insert(0, _week3_str)

import estimator_new as _estimator_new  # noqa: E402 (only importer)

from .observer import SteadyStateKalman
from .probes import uniform_probe


def compute_pca_readout(Y: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Top-k principal components of measurements ``Y`` as a readout matrix.

    Each PC is one weighted combination of the 16 measurements capturing a
    dominant pattern of population activity. Returned ``M`` has shape ``(k, p)``
    and rows are the top-k right singular vectors of mean-centred ``Y``.

    Parameters
    ----------
    Y : ndarray, shape (T, p)
        Measurement matrix from a probe or open-loop run.
    k : int
        Number of leading principal components (default 2 — matches the
        number of inputs, so we can independently hold k quantities).

    Returns
    -------
    M : ndarray, shape (k, p)
        Readout projection (rows are unit-norm).
    explained_var_ratio : ndarray, shape (k,)
        Fraction of total variance captured by each PC.
    """
    Y = np.asarray(Y, dtype=float)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    # SVD of T×p matrix: U(T×r) S(r) Vt(r×p). PCs of Y are rows of Vt.
    _, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    M = Vt[:k].copy()
    var_total = float((S ** 2).sum())
    explained = (S[:k] ** 2) / var_total if var_total > 0 else np.zeros(k)
    return M, explained


def readout_steady_state_gain(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, M: np.ndarray
) -> np.ndarray:
    """Steady-state input → readout gain ``G = M C (I − A)^{-1} B``.

    Used to verify the readout is well-controllable (cond(G) small means
    each PC can be moved independently by the inputs).
    """
    n = A.shape[0]
    I = np.eye(n)
    return M @ C @ np.linalg.solve(I - A, B)


class IdentifiedSystem:
    """Container for ``(A, B, C, Q, R, a, c)`` after Phase-A probe + EM fit.

    Attributes
    ----------
    A, B, C, Q, R, a, c : ndarray
        Fitted affine LGSSM parameters.
    diagnostics : dict
        Numeric summary (T_cal, n, m, spectral radius, ||CB||_F, traces).
    degenerate_flags : list of str
        Non-empty if the fit looks suspicious (non-finite, near-zero B,
        unstable A, near-zero CB).
    validation : dict
        Output of ``estimator_new.validate`` against the truth (when given).
    """

    def __init__(self):
        self.A = self.B = self.C = self.Q = self.R = None
        self.a = self.c = None
        self.Y_cal = None
        self.U_cal = None
        self.estimator: _estimator_new.EstimatorNew | None = None
        self.diagnostics: dict = {}
        self.degenerate_flags: list[str] = []
        self.validation: dict = {}
        # PCA-based readout (set in calibrate); see compute_pca_readout.
        self.readout_M: np.ndarray | None = None
        self.readout_explained_var: np.ndarray | None = None
        self.readout_G: np.ndarray | None = None  # 2x2 input→readout DC gain
        self.readout_cond_G: float = float("nan")

    def calibrate(
        self,
        plant,
        T_cal: int,
        n: int = 4,
        m: int | None = None,
        probe=None,
        seed: int = 0,
        em_max_iter: int = 80,
        em_tol: float = 1e-5,
        true_model: dict | None = None,
        x_true: np.ndarray | None = None,
    ) -> "IdentifiedSystem":
        if m is None:
            m = plant.input_dim

        if probe is None:
            probe = uniform_probe(T_cal, m, seed=seed)
        else:
            probe = np.asarray(probe, dtype=float)
            if probe.shape != (T_cal, m):
                raise ValueError(f"probe shape {probe.shape} != ({T_cal}, {m})")

        # Phase A: drive plant, record observations.  Timing convention
        # matches run_closed_loop: at step t we measure y[t], then apply
        # probe[t] (which drives the state from t to t+1).
        y0 = np.asarray(plant.measure(), dtype=float)
        p = y0.size
        Y = np.empty((T_cal, p))
        Y[0] = y0
        plant.next_state(probe[0])
        for t in range(1, T_cal):
            Y[t] = plant.measure()
            plant.next_state(probe[t])

        est = _estimator_new.EstimatorNew().fit(
            Y, probe, n=n, max_iter=em_max_iter, tol=em_tol
        )

        self.A = est.A
        self.B = est.B
        self.C = est.C
        self.Q = est.Q
        self.R = est.R
        self.a = est.a
        self.c = est.c
        self.Y_cal = Y
        self.U_cal = probe
        self.estimator = est

        # Diagnostics
        eig = np.linalg.eigvals(self.A)
        rho = float(np.max(np.abs(eig))) if eig.size else 0.0
        CB = self.C @ self.B
        self.diagnostics = {
            "T_cal": int(T_cal),
            "n": int(n),
            "m": int(m),
            "p": int(p),
            "fit_iters": int(est.fit_iters),
            "loglik_final": float(est.loglik_hist[-1]) if est.loglik_hist else float("nan"),
            "A_spectral_radius": rho,
            "B_col_norms": np.linalg.norm(self.B, axis=0).tolist(),
            "CB_frob": float(np.linalg.norm(CB)),
            "a_norm": float(np.linalg.norm(self.a)),
            "c_norm": float(np.linalg.norm(self.c)),
            "Q_trace": float(np.trace(self.Q)),
            "R_trace": float(np.trace(self.R)),
        }
        self.degenerate_flags = []
        self.controller_warnings: list[str] = []
        if not all(np.all(np.isfinite(p)) for p in (self.A, self.B, self.C, self.Q, self.R, self.a, self.c)):
            self.degenerate_flags.append("non-finite params")
        if np.linalg.norm(self.B) < 1e-8:
            self.degenerate_flags.append("B near zero")
        if rho > 1.01:
            self.degenerate_flags.append(f"A unstable (rho={rho:.3f})")

        # PCA-based readout: 2 leading principal components of probe y.
        # Each PC is a weighted combination of the p measurements capturing
        # a dominant pattern of population activity (spec §3).
        k_readout = min(2, m)
        self.readout_M, self.readout_explained_var = compute_pca_readout(Y, k=k_readout)
        self.readout_G = readout_steady_state_gain(
            self.A, self.B, self.C, self.readout_M
        )
        try:
            self.readout_cond_G = float(np.linalg.cond(self.readout_G))
        except np.linalg.LinAlgError:
            self.readout_cond_G = float("inf")
        self.diagnostics["readout_explained_var"] = self.readout_explained_var.tolist()
        self.diagnostics["readout_cond_G"] = self.readout_cond_G
        self.diagnostics["readout_G"] = self.readout_G.tolist()

        # Validation (only meaningful when true_model is available).
        if true_model is not None or x_true is not None:
            self.validation = est.validate(true_model=true_model, x_true=x_true)
        else:
            self.validation = {"em_monotone": all(
                est.loglik_hist[i + 1] >= est.loglik_hist[i] - 1e-3 * max(abs(est.loglik_hist[i]), 1.0)
                for i in range(len(est.loglik_hist) - 1))}

        # v3 E4 fix (spec §8): the "degenerate identification" flag now
        # gates on **fit quality** (one-step prediction RMS, state R²,
        # G relative error) — not on ‖CB‖_F. The previous gating
        # false-flagged `slow_drift` (state R² ≈ 1.0, fit is fine) and
        # missed `input_blind` (state R² ≈ 0.68, fit is mediocre).
        # ‖CB‖_F is kept as a separate **controller-authority warning**
        # in ``controller_warnings`` — it reflects "the input can barely
        # be seen in the measurements", which limits any controller's
        # leverage independently of identification quality.
        in_sample_val = est.validate(Y_val=Y, U_val=probe)
        one_step_rms = float(in_sample_val.get("one_step_rms", float("nan")))
        y_rms_val = float(np.sqrt(np.mean(Y ** 2)))
        self.diagnostics["one_step_rms"] = one_step_rms
        self.diagnostics["y_rms"] = y_rms_val
        self.diagnostics["one_step_rms_frac"] = (
            one_step_rms / max(y_rms_val, 1e-12)
            if np.isfinite(one_step_rms) else float("nan")
        )

        # Fit-quality degenerate flags.
        if np.isfinite(one_step_rms) and one_step_rms > 0.5 * y_rms_val:
            self.degenerate_flags.append(
                f"poor one-step prediction (RMS {one_step_rms:.2f} > "
                f"0.5 × y-RMS {y_rms_val:.2f}) — fit is unreliable"
            )
        if "state_r2" in self.validation and self.validation["state_r2"] < 0.7:
            self.degenerate_flags.append(
                f"low state R² ({self.validation['state_r2']:.2f} < 0.7) "
                f"— latent state poorly tracked"
            )
        if "G_rel_err" in self.validation and self.validation["G_rel_err"] > 0.5:
            self.degenerate_flags.append(
                f"high G relative error ({self.validation['G_rel_err']:.2f} > 0.5) "
                f"— input→readout gain mis-identified"
            )

        # ‖CB‖_F controller-authority warning (kept as a SEPARATE list,
        # not in degenerate_flags — it does not imply estimation failure).
        CB_threshold = 0.05 * float(np.linalg.norm(self.C) * np.linalg.norm(self.B))
        self.diagnostics["CB_visibility_threshold"] = CB_threshold
        if self.diagnostics["CB_frob"] < max(CB_threshold, 1e-4):
            self.controller_warnings.append(
                f"low input visibility (‖CB‖={self.diagnostics['CB_frob']:.3f} "
                f"< threshold {CB_threshold:.3f}) — hidden-input regime; the "
                f"input is barely seen in the measurements, so any controller "
                f"will have limited authority (independent of fit quality)"
            )

        return self

    def params(self):
        return self.A, self.B, self.C, self.Q, self.R, self.a, self.c

    def make_observer(self) -> SteadyStateKalman:
        return SteadyStateKalman(self.A, self.B, self.C, self.Q, self.R, self.a, self.c)
