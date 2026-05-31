"""Steady-state Kalman observer (affine form).

Solves the predict-covariance DARE once at construction, runs a fixed-gain
predict / update online with the spec's affine offsets:

    predict:  x⁻ = A x̂ + B u_prev + a
    innov:    e  = y - C x⁻ - c
    update:   x̂  = x⁻ + L e

Consumes only `(A, B, C, Q, R, a, c)` — no estimator import here.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are


class SteadyStateKalman:
    def __init__(self, A, B, C, Q, R, a=None, c=None):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.n = self.A.shape[0]
        self.p = self.C.shape[0]
        self.m = self.B.shape[1]
        self.a = np.zeros(self.n) if a is None else np.asarray(a, dtype=float).reshape(self.n)
        self.c = np.zeros(self.p) if c is None else np.asarray(c, dtype=float).reshape(self.p)

        try:
            P = solve_discrete_are(self.A.T, self.C.T, self.Q, self.R)
        except Exception:
            P = np.eye(self.n)
        S = self.C @ P @ self.C.T + self.R
        S_reg = S + 1e-10 * np.eye(S.shape[0])
        self.L = P @ self.C.T @ np.linalg.inv(S_reg)

        # Start at the model's natural equilibrium under u = 0 (resting):
        # x⁻₀ = (I - A)⁻¹ a, so the filter doesn't see a huge transient before
        # the first measurement.
        try:
            self.x0 = np.linalg.solve(np.eye(self.n) - self.A, self.a)
        except np.linalg.LinAlgError:
            self.x0 = np.zeros(self.n)
        self.x = self.x0.copy()

    def reset(self, x0=None) -> None:
        if x0 is None:
            self.x = self.x0.copy()
        else:
            self.x = np.asarray(x0, dtype=float).copy()

    def filter_step(self, y, u_prev) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        u_prev = np.asarray(u_prev, dtype=float)
        x_pred = self.A @ self.x + self.B @ u_prev + self.a
        innov = y - self.C @ x_pred - self.c
        self.x = x_pred + self.L @ innov
        return self.x.copy()
