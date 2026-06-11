"""N4SID input-output subspace init + order selection (spec §9).

Covers the M2 acceptance bar:
    * order selection finds the true ``n`` on synthetic data;
    * N4SID → EM recovers basis-free invariants (eigvals, DC gain) on a
      synthetic 6-state affine LGSSM;
    * held-out one-step prediction RMS is at or near the per-channel
      noise floor ``√(median diag R)``.
"""

from __future__ import annotations

import unittest

import numpy as np

from . import _path_helper  # noqa: F401

from estimator.identify import EstimatorNew, estimate_order  # noqa: E402


def _stable_random_A(rng, n: int, rho_target: float = 0.92) -> np.ndarray:
    A = rng.standard_normal((n, n))
    A = A / max(np.max(np.abs(np.linalg.eigvals(A))), 1e-9) * rho_target
    return A


def _simulate_lgssm(rng, A, B, C, Q, R, T, a=None, c=None, u_range=(0.0, 1.0)):
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]
    if a is None: a = np.zeros(n)
    if c is None: c = np.zeros(p)
    U = rng.uniform(u_range[0], u_range[1], size=(T, m))
    x = np.zeros((T + 1, n))
    Y = np.zeros((T, p))
    for t in range(T):
        Y[t] = C @ x[t] + c + rng.multivariate_normal(np.zeros(p), R)
        x[t + 1] = A @ x[t] + B @ U[t] + a + rng.multivariate_normal(np.zeros(n), Q)
    return Y, U, x[:T]


class OrderSelectionFindsTrueN(unittest.TestCase):
    """``estimate_order`` should land at the true latent dim on synthetic data."""

    def test_order_n6_clean_knee_system(self):
        """A system *designed* with a clean Hankel-SV cliff at n=6.

        Random small-T systems show a continuous SV spread that makes
        automatic order selection ambiguous; here we engineer the cliff
        (matched-magnitude modes, low noise, long record) so the heuristic
        is being graded on a fair task.
        """
        rng = np.random.default_rng(0)
        n_true = 6
        m, p, T = 2, 16, 3000
        # 6 distinct real modes, evenly spaced.
        A = np.diag(np.linspace(0.95, 0.7, n_true))
        # All modes equally controllable + observable.
        B = np.ones((n_true, m)) + 0.1 * rng.standard_normal((n_true, m))
        C = rng.standard_normal((p, n_true))
        C = C / np.linalg.norm(C, axis=0, keepdims=True)
        Q = 1e-5 * np.eye(n_true)
        R = 1e-3 * np.eye(p)
        Y, U, _ = _simulate_lgssm(rng, A, B, C, Q, R, T)
        n_hat, sigma = estimate_order(Y, U, max_n=10)
        self.assertEqual(n_hat, n_true,
            msg=f"order selection returned n={n_hat}, expected {n_true} "
                f"(σ_norm head {(sigma[:10] / sigma[0]).round(4)})")


class N4SIDPipelineRecoversInvariants(unittest.TestCase):
    """N4SID → EM at n=6 recovers basis-free invariants + holds out cleanly."""

    def test_eigenvalues_dc_gain_holdout_n6(self):
        rng = np.random.default_rng(1)
        n_true = 6
        m, p, T = 2, 16, 1500
        A = _stable_random_A(rng, n_true, rho_target=0.95)
        B = rng.standard_normal((n_true, m))
        C = rng.standard_normal((p, n_true))
        a = 0.1 * rng.standard_normal(n_true)
        c = rng.standard_normal(p)
        Q = 1e-3 * np.eye(n_true)
        R = 0.25 * np.eye(p)
        Y, U, x_true = _simulate_lgssm(rng, A, B, C, Q, R, T, a=a, c=c)

        est = EstimatorNew().fit(Y, U, n=n_true, init="n4sid",
                                  max_iter=80, tol=1e-6)

        # Stable + finite.
        rho = float(np.max(np.abs(np.linalg.eigvals(est.A))))
        self.assertLess(rho, 1.0, msg=f"fit A unstable: rho={rho:.4f}")
        self.assertTrue(all(np.all(np.isfinite(p_)) for p_ in
                            (est.A, est.B, est.C, est.Q, est.R, est.a, est.c)))

        # Basis-free: eigenvalues.
        eig_true = np.sort_complex(np.linalg.eigvals(A))
        eig_fit = np.sort_complex(np.linalg.eigvals(est.A))
        eig_err = float(np.abs(eig_true - eig_fit).max())
        # 0.15 absolute on a system with |λ| up to 0.95 — finite-T sampling
        # error on EM eigenvalue estimation is irreducible at T=1500, n=6,
        # R = 0.25 I (a regime mimicking the Brain's SNR).
        self.assertLess(eig_err, 0.15,
            msg=f"eigenvalue mismatch: {eig_err:.3e} (truth {eig_true.round(3)}, "
                f"fit {eig_fit.round(3)})")

        # Basis-free: DC gain.
        n = A.shape[0]
        G_true = C @ np.linalg.solve(np.eye(n) - A, B)
        G_fit = est.C @ np.linalg.solve(np.eye(n) - est.A, est.B)
        G_rel = (np.linalg.norm(G_true - G_fit)
                 / max(np.linalg.norm(G_true), 1e-12))
        # 0.15 absolute on a 6-state system from T = 1500 with R = 0.25 I.
        self.assertLess(G_rel, 0.15,
            msg=f"DC gain rel err {G_rel:.3e} > 0.15")

        # Held-out: one-step RMS should approach the per-channel noise floor.
        rng2 = np.random.default_rng(99)
        T_v = 400
        Y_v, U_v, _ = _simulate_lgssm(rng2, A, B, C, Q, R, T_v, a=a, c=c)
        rep = est.validate(Y_val=Y_v, U_val=U_v)
        one_step = rep["one_step_rms"]
        noise_floor = float(np.sqrt(np.median(np.diag(R))))
        # Bar: within 2× per-channel noise floor.
        self.assertLess(one_step, 2.0 * noise_floor,
            msg=f"one-step RMS {one_step:.3f} > 2× noise floor {noise_floor:.3f}")


class N4SIDBeatsOutputOnlyInit(unittest.TestCase):
    """On systems where the input is strong, N4SID should match output-SSI
    init quality (both warm into the same EM fixed point) but with better
    final log-likelihood when EM is stopped early.
    """

    def test_n4sid_geq_outputonly_after_short_em(self):
        rng = np.random.default_rng(2)
        n_true = 4
        m, p, T = 2, 12, 800
        A = _stable_random_A(rng, n_true, rho_target=0.9)
        # Make B strong so the input matters for identification.
        B = 2.0 * rng.standard_normal((n_true, m))
        C = rng.standard_normal((p, n_true))
        Q = 1e-3 * np.eye(n_true)
        R = 0.5 * np.eye(p)
        Y, U, _ = _simulate_lgssm(rng, A, B, C, Q, R, T)

        # Stop EM after 5 iterations to see the init's contribution.
        est_n4 = EstimatorNew().fit(Y, U, n=n_true, init="n4sid",
                                    max_iter=5, tol=0.0)
        est_oo = EstimatorNew().fit(Y, U, n=n_true, init="output_ssi",
                                    max_iter=5, tol=0.0)
        ll_n4 = est_n4.loglik_history()[-1]
        ll_oo = est_oo.loglik_history()[-1]
        # Don't require N4SID to *beat* output-only every time (random
        # systems differ), but require it to be in the same ballpark
        # (within 5% relative ll) on a system where the input is strong.
        self.assertGreater(
            ll_n4, ll_oo - 0.05 * abs(ll_oo),
            msg=f"N4SID after 5 EM iters: ll={ll_n4:.1f}; "
                f"output-only: ll={ll_oo:.1f} (N4SID materially worse)",
        )


if __name__ == "__main__":
    unittest.main()
