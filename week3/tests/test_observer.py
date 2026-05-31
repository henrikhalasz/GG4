"""Steady-state Kalman observer tracks the true latent state on a known LDS."""

from __future__ import annotations

import unittest

import numpy as np

from . import _path_helper  # noqa: F401  (extends sys.path)
import Simulator as sim  # noqa: E402

from control.observer import SteadyStateKalman  # noqa: E402


def _r_squared_aligned(X_true: np.ndarray, X_hat: np.ndarray) -> float:
    """Linear-alignment R² (week3.md §5: identified basis up to similarity)."""
    T_align, *_ = np.linalg.lstsq(X_hat, X_true, rcond=None)
    pred = X_hat @ T_align
    ss_res = float(np.sum((X_true - pred) ** 2))
    ss_tot = float(np.sum((X_true - X_true.mean(axis=0)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


class ObserverTracksTrueState(unittest.TestCase):
    def test_default_neural_system(self):
        system = sim.default_neural_system(seed=0, obs_dim=16)
        T = 500
        # Random non-negative drive — keeps system excited beyond the IC transient.
        rng = np.random.default_rng(1)
        U = rng.uniform(0.0, 1.0, size=(T, system.input_dim))
        data = system.simulate(T, U=U)
        y = data["y"]
        x_true = data["x"][:T]

        # Simulator scenarios have a = 0 and c = 0 (no affine offsets in truth).
        n = system.A.shape[0]
        obs = SteadyStateKalman(system.A, system.B, system.C, system.Q, system.R,
                                 a=np.zeros(n), c=np.zeros(system.obs_dim))
        u_prev = np.zeros(system.input_dim)
        x_hat = np.empty_like(x_true)
        for t in range(T):
            x_hat[t] = obs.filter_step(y[t], u_prev)
            u_prev = U[t]

        # Skip the first few steps (init transient) when scoring.
        burn = 20
        r2 = _r_squared_aligned(x_true[burn:], x_hat[burn:])
        self.assertGreater(r2, 0.95, msg=f"observer R² too low: {r2:.3f}")


if __name__ == "__main__":
    unittest.main()
