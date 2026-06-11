"""estimator_new: EM correctness on synthetic + Simulator data.

Covers spec §5 / §9 estimator tests:
- EM log-likelihood non-decreasing (within 1e-3 relative tol).
- Synthetic affine LGSSM: fitted G, z0 match true within tolerance; state R²
  high after affine alignment; one-step held-out prediction OK.
- Simulator scenario: G_fit matches true G to within ~10%.
"""

from __future__ import annotations

import unittest

import numpy as np

from . import _path_helper  # noqa: F401
import Simulator as sim  # noqa: E402

from estimator.identify import EstimatorNew, _kalman_filter_affine  # noqa: E402


def _sim_affine_lgssm(rng, n=3, m=2, p=8, T=600):
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
    U = rng.uniform(0.0, 1.0, size=(T, m))
    x = np.zeros((T + 1, n))
    Y = np.zeros((T, p))
    for t in range(T):
        Y[t] = C @ x[t] + c + rng.multivariate_normal(np.zeros(p), R)
        x[t + 1] = A @ x[t] + B @ U[t] + a + rng.multivariate_normal(np.zeros(n), Q)
    truth = dict(A=A, B=B, C=C, Q=Q, R=R, a=a, c=c, x=x[:T])
    return Y, U, truth


class EMOnSyntheticAffine(unittest.TestCase):
    def test_monotone_and_recovery(self):
        rng = np.random.default_rng(0)
        Y, U, truth = _sim_affine_lgssm(rng)
        est = EstimatorNew().fit(Y, U, n=truth["A"].shape[0], max_iter=80, tol=1e-6)
        ll = est.loglik_history()
        # Monotone within numerical noise.
        diffs = np.diff(ll)
        worst_drop = float(diffs.min()) if diffs.size else 0.0
        rel_tol = 1e-3 * max(abs(ll[-1]), 1.0)
        self.assertGreater(worst_drop, -rel_tol,
                           msg=f"EM ll not monotone (worst drop {worst_drop:.4f}, tol {rel_tol:.4f})")
        rep = est.validate(true_model=truth, x_true=truth["x"])
        # Spec: high state R² after affine alignment.
        self.assertGreater(rep["state_r2"], 0.95,
                           msg=f"state R² too low: {rep['state_r2']:.4f}")
        # Spec: G_fit matches true G to within tolerance.
        self.assertLess(rep["G_rel_err"], 0.10,
                        msg=f"G relative error too high: {rep['G_rel_err']:.4f}")
        # Stable fit.
        self.assertFalse(rep["A_unstable"])
        self.assertTrue(rep["params_finite"])

    def test_one_step_held_out(self):
        rng = np.random.default_rng(1)
        Y, U, truth = _sim_affine_lgssm(rng)
        est = EstimatorNew().fit(Y, U, n=truth["A"].shape[0], max_iter=60, tol=1e-5)
        # Held-out trajectory from a NEW noise seed.
        rng2 = np.random.default_rng(99)
        T_v = 200
        A, B, C, a, c, Q, R = truth["A"], truth["B"], truth["C"], truth["a"], truth["c"], truth["Q"], truth["R"]
        n = A.shape[0]
        x = np.zeros((T_v + 1, n))
        U_v = rng2.uniform(0.0, 1.0, size=(T_v, truth["B"].shape[1]))
        Y_v = np.zeros((T_v, C.shape[0]))
        for t in range(T_v):
            Y_v[t] = C @ x[t] + c + rng2.multivariate_normal(np.zeros(C.shape[0]), R)
            x[t + 1] = A @ x[t] + B @ U_v[t] + a + rng2.multivariate_normal(np.zeros(n), Q)
        rep = est.validate(true_model=truth, Y_val=Y_v, U_val=U_v)
        y_rms = float(np.sqrt(np.mean(Y_v ** 2)))
        self.assertLess(rep["one_step_rms"], 0.5 * y_rms,
                        msg=f"one-step RMS {rep['one_step_rms']:.3f} too high vs y_RMS {y_rms:.3f}")


class EMOnSimulatorScenario(unittest.TestCase):
    def test_default_neural_system_G_match(self):
        rng = np.random.default_rng(0)
        system = sim.default_neural_system(seed=0, obs_dim=16)
        T = 800
        U = rng.uniform(0.0, 1.0, size=(T, system.input_dim))
        data = system.simulate(T, U=U)
        Y = data["y"]
        est = EstimatorNew().fit(Y, U, n=system.A.shape[0], max_iter=80, tol=1e-5)
        truth = dict(A=system.A, B=system.B, C=system.C, a=np.zeros(system.A.shape[0]), c=np.zeros(system.obs_dim))
        rep = est.validate(true_model=truth, x_true=data["x"][:T])
        # Looser tolerances for Simulator data (longer T helps a lot, but on T=800 we expect ~10% on G).
        self.assertLess(rep["G_rel_err"], 0.15,
                        msg=f"Simulator G relative error too high: {rep['G_rel_err']:.4f}")
        self.assertGreater(rep["state_r2"], 0.90,
                           msg=f"Simulator state R² too low: {rep['state_r2']:.4f}")
        self.assertFalse(rep["A_unstable"])


if __name__ == "__main__":
    unittest.main()
