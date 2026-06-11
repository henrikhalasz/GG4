"""kalman.py -- steady-state Kalman filter on the identified model (Week 4, Step 1, S8).

The runtime estimator the Step-2 controller may call. We solve the predict-covariance
DARE once (Week-3 SteadyStateKalman form), then run a fixed-gain predict/update online,
de-biasing every observation with the S5 sensor bias. The model is zero-offset, so the
filter is the plain affine-free form:

    predict:  x- = A x_hat + B u_prev
    innov:    e  = (y - bias) - C x-
    update:   x_hat = x- + L e

Deployment is single-trajectory: one fresh noisy measurement per step, no noise-freezing
or seed tricks. This file imports nothing from Week 3 and never imports true_brain_params.

Timing (matches identify.py / the Week-4 notebook): observations[t] = y[t] = measure()
before applying inputs[t]; inputs[t] drives x_t -> x_{t+1}; the predict at step t consumes
inputs[t-1] (zeros for t = 0).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are

from GG4 import Brain
import excitation as exc

HERE = Path(__file__).resolve().parent


class SteadyStateKalman:
    """Fixed-gain Kalman filter on a zero-offset LGSSM (A, B, C, Q, R) with sensor bias."""

    def __init__(self, A, B, C, Q, R, bias=None):
        self.A = np.asarray(A, float)
        self.B = np.asarray(B, float)
        self.C = np.asarray(C, float)
        self.Q = np.asarray(Q, float)
        self.R = np.asarray(R, float)
        self.n, self.m, self.p = self.A.shape[0], self.B.shape[1], self.C.shape[0]
        self.bias = np.zeros(self.p) if bias is None else np.asarray(bias, float).reshape(self.p)

        # Predict-covariance DARE, solved once: P = A P A^T + Q - A P C^T (C P C^T + R)^-1 C P A^T
        try:
            P = solve_discrete_are(self.A.T, self.C.T, self.Q, self.R)
        except Exception:
            P = np.eye(self.n)
        self.P_pred = P
        S = self.C @ P @ self.C.T + self.R
        self.S = S
        self.L = P @ self.C.T @ np.linalg.inv(S + 1e-10 * np.eye(self.p))   # predict-form gain
        self.P_filt = (np.eye(self.n) - self.L @ self.C) @ P                # filtered cov

        # Resting equilibrium under u = 0 is the origin (zero-offset model).
        self.x0 = np.zeros(self.n)
        self.x = self.x0.copy()

    def reset(self, x0=None) -> None:
        self.x = self.x0.copy() if x0 is None else np.asarray(x0, float).copy()

    def filter_step(self, y, u_prev):
        """One predict/update from a fresh measurement y and the previously applied input."""
        y = np.asarray(y, float) - self.bias
        x_pred = self.A @ self.x + self.B @ np.asarray(u_prev, float)
        innov = y - self.C @ x_pred
        self.x = x_pred + self.L @ innov
        return self.x.copy(), innov

    def estimate(self, observations, inputs) -> np.ndarray:
        """Latent-state estimates for a whole trajectory. Returns (T, n).

        observations[t] = y[t]; inputs[t] drives x_t -> x_{t+1}. The predict at step t
        uses inputs[t-1] (zeros for t = 0), matching the identification timing.
        """
        Y = np.asarray(observations, float)
        U = np.asarray(inputs, float)
        T = Y.shape[0]
        self.reset()
        X = np.empty((T, self.n))
        for t in range(T):
            u_prev = U[t - 1] if t > 0 else np.zeros(self.m)
            X[t], _ = self.filter_step(Y[t], u_prev)
        return X


def load_model(path: Path | None = None) -> dict:
    d = np.load(path or (HERE / "identified_model.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def main() -> None:
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    mdl = load_model()
    kf = SteadyStateKalman(mdl["A"], mdl["B"], mdl["C"], mdl["Q"], mdl["R"], mdl["bias"])
    print("Steady-state Kalman on the identified model")
    print(f"  n={kf.n}, p={kf.p}, m={kf.m}")
    print(f"  steady-state filtered-covariance trace tr(P_filt) = {np.trace(kf.P_filt):.4f}")
    print(f"  predict-covariance trace          tr(P_pred) = {np.trace(kf.P_pred):.4f}")

    # Deployment: a single fresh noisy trajectory (no seed tricks). Use a held-out seed
    # and a both-channel decorrelated drive.
    seed = 2024
    T = 4000
    U = exc.validation_inputs(seed=seed, T=T)
    brain = Brain(random_seed=seed)
    Y = np.empty((T, kf.p))
    Y[0] = brain.measure(); brain.next_state(U[0])
    for t in range(1, T):
        Y[t] = brain.measure(); brain.next_state(U[t])

    # Run the filter, collecting innovations for whiteness / NIS sanity checks.
    kf.reset()
    innov = np.empty((T, kf.p))
    X = np.empty((T, kf.n))
    for t in range(T):
        u_prev = U[t - 1] if t > 0 else np.zeros(kf.m)
        X[t], innov[t] = kf.filter_step(Y[t], u_prev)

    burn = 200
    e = innov[burn:]
    Sinv = np.linalg.inv(kf.S + 1e-10 * np.eye(kf.p))
    nis = np.einsum("ti,ij,tj->t", e, Sinv, e)
    print(f"\nInnovation diagnostics (after {burn}-step burn-in, T={T - burn}):")
    print(f"  mean NIS = {nis.mean():.2f}   (expected ~ p = {kf.p} if S is consistent)")
    print(f"  NIS 5-95 pct = [{np.percentile(nis, 5):.1f}, {np.percentile(nis, 95):.1f}]")

    # Whiteness: lag-1 autocorrelation of the normalised innovations per channel.
    L = np.linalg.cholesky(kf.S + 1e-10 * np.eye(kf.p))
    z = np.linalg.solve(L, e.T).T          # whitened innovations, unit cov if consistent
    ac1 = np.array([np.corrcoef(z[:-1, i], z[1:, i])[0, 1] for i in range(kf.p)])
    print(f"  whitened-innovation lag-1 autocorr: mean |.| = {np.abs(ac1).mean():.3f} "
          f"(near 0 => white)")
    print(f"  whitened-innovation covariance trace/p = {np.trace(np.cov(z.T)) / kf.p:.3f} "
          f"(near 1 => S consistent)")
    print(f"\n  estimate() output shape: {kf.estimate(Y, U).shape}")


if __name__ == "__main__":
    main()
