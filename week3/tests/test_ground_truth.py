"""Ground-truth harness validation (spec §9).

Validates that determinism-differencing + ERA recover a known SimulatorPlant's
basis-free invariants (eigenvalues + Markov parameters + DC gain) to high
precision. This protects the M1 yardstick from silent bugs — if ERA gets
``(A, B, C)`` wrong on a system we *do* know the truth of, we cannot trust
its result on the Brain.

The test does NOT check ``Q`` or ``R`` precision (those come from sample
covariances and have ``1/√N`` statistical error) — they get a weaker sanity
check.
"""

from __future__ import annotations

import unittest

import numpy as np

from . import _path_helper  # noqa: F401  (puts week3/ and week 1/ on sys.path)
import Simulator as sim  # noqa: E402

from control.plant import SimulatorPlant  # noqa: E402

from analysis.ground_truth import (  # noqa: E402
    noise_free_markov_parameters,
    era_ho_kalman,
    measurement_noise_covariance,
    process_noise_covariance,
)


def _markov_chain(A, B, C, K):
    """Reference Markov params {C A^{k-1} B} for k = 1..K."""
    H = np.zeros((K, C.shape[0], B.shape[1]))
    AB = B.copy()
    for k in range(K):
        H[k] = C @ AB
        AB = A @ AB
    return H


def _dc_gain(A, B, C):
    n = A.shape[0]
    return C @ np.linalg.solve(np.eye(n) - A, B)


class DifferencingCancelsNoise(unittest.TestCase):
    """The differenced Markov params should equal the true Markov params
    *exactly* (to floating-point) on a SimulatorPlant — both runs use the
    same seed, so process and observation noise streams cancel."""

    def test_markov_match_to_floating_point(self):
        system = sim.default_neural_system(seed=0, obs_dim=16)
        # Match SimulatorPlant's signature: it constructs its own RNG from
        # the seed passed at __init__ time.
        plant_factory = lambda seed: SimulatorPlant(system, seed=seed)

        T = 60
        H_diff = noise_free_markov_parameters(plant_factory, seed=11, T=T)
        H_true = _markov_chain(system.A, system.B, system.C, T)
        # H_diff[0] is identically zero by construction; H_true[0] = CB.
        # Compare from index 1 (i.e. Markov param H_1 = CB), but note our
        # convention shifted: H_diff[1] should equal H_true[0] = CB
        # (the first response after the impulse propagates one step).
        err = np.abs(H_diff[1:T] - H_true[:T - 1]).max()
        self.assertLess(err, 1e-9,
            msg=f"differencing did NOT cancel noise — max Markov err = {err:.2e}")


class ERARecoversInvariants(unittest.TestCase):
    """ERA on the differenced Markov params recovers eigenvalues + DC gain
    to numerical precision (basis-free; we do NOT compare matrix entries)."""

    def test_eigenvalues_and_dc_gain(self):
        system = sim.default_neural_system(seed=0, obs_dim=16)
        n_true = int(system.A.shape[0])
        plant_factory = lambda seed: SimulatorPlant(system, seed=seed)

        T = 80  # plenty of Markov params for a 4-state system
        H = noise_free_markov_parameters(plant_factory, seed=11, T=T)
        A_hat, B_hat, C_hat, sigma = era_ho_kalman(H, n=n_true)

        # 1) Hankel rank knee should sit at n_true: a clear drop after sigma[n-1].
        ratio = sigma[n_true] / sigma[n_true - 1] if sigma.size > n_true else 0.0
        self.assertLess(ratio, 1e-6,
            msg=f"Hankel singular-value knee not clean: sigma[{n_true}]/sigma[{n_true-1}] = {ratio:.2e}")

        # 2) Eigenvalues match the truth (sorted by magnitude).
        eig_true = np.sort_complex(np.linalg.eigvals(system.A))
        eig_hat = np.sort_complex(np.linalg.eigvals(A_hat))
        eig_err = np.abs(eig_true - eig_hat).max()
        self.assertLess(eig_err, 1e-8,
            msg=f"eigenvalue mismatch — max |λ_true − λ_hat| = {eig_err:.2e}")

        # 3) DC gain matches (basis-free invariant).
        G_true = _dc_gain(system.A, system.B, system.C)
        G_hat = _dc_gain(A_hat, B_hat, C_hat)
        rel = np.linalg.norm(G_true - G_hat) / max(np.linalg.norm(G_true), 1e-12)
        self.assertLess(rel, 1e-8,
            msg=f"DC gain mismatch — relative error {rel:.2e}")

        # 4) Markov parameters {C A^k B} round-trip (the strongest invariant).
        H_true = _markov_chain(system.A, system.B, system.C, T - 1)
        H_hat = _markov_chain(A_hat, B_hat, C_hat, T - 1)
        markov_err = np.abs(H_true - H_hat).max()
        self.assertLess(markov_err, 1e-7,
            msg=f"reconstructed Markov chain mismatch — max err {markov_err:.2e}")


class ROughlyMatchesObservationCovariance(unittest.TestCase):
    """``measurement_noise_covariance`` should approximate the Simulator's R."""

    def test_default_neural_system(self):
        system = sim.default_neural_system(seed=0, obs_dim=16)
        plant_factory = lambda seed: SimulatorPlant(system, seed=seed)
        R_hat = measurement_noise_covariance(plant_factory, seed=11,
                                             n_samples=4000)
        # Frobenius relative error on a 16x16 R from 4k samples should be
        # well under 10% — Wishart standard deviation is ~ p/√N = 16/63 ≈ 0.25
        # entry-wise, but most of R's mass is on the diagonal so the
        # *relative* Frobenius error is much smaller.
        rel = (np.linalg.norm(R_hat - system.R, "fro")
               / max(np.linalg.norm(system.R, "fro"), 1e-12))
        self.assertLess(rel, 0.10,
            msg=f"R relative Frobenius error too high: {rel:.3f}")


class QRecoveredByEMOnSimulatorPlant(unittest.TestCase):
    """The frozen-(A,B,C,R) EM Q-update should recover the Simulator's Q.

    Uses the *true* (A, B, C, R) (not the ERA reconstruction) so the test
    isolates the Q estimator from any ERA basis ambiguity. Also checks that
    repeating across seeds gives stable tr(Q) — the v1 stationary-cov
    estimator failed this badly (tr(Q) swung 2.3× across seeds).
    """

    def test_q_recovery_simulator_plant(self):
        system = sim.default_neural_system(seed=0, obs_dim=16)
        plant_factory = lambda seed: SimulatorPlant(system, seed=seed)
        n = int(system.A.shape[0])
        a0 = np.zeros(n)
        c0 = np.zeros(system.obs_dim)

        Q_hats = []
        for seed in (3, 11, 23):
            Q_hat = process_noise_covariance(
                plant_factory,
                A=system.A, B=system.B, C=system.C, R=system.R,
                a=a0, c=c0, seed=seed, T=3000, burn_in=200, n_iter=15,
            )
            Q_hats.append(Q_hat)

        Q_true = system.Q
        rel_errs = [
            np.linalg.norm(Q - Q_true, "fro") / max(np.linalg.norm(Q_true, "fro"), 1e-12)
            for Q in Q_hats
        ]
        traces = [float(np.trace(Q)) for Q in Q_hats]
        tr_true = float(np.trace(Q_true))
        tr_spread = (max(traces) - min(traces)) / max(abs(tr_true), 1e-12)

        # Per-seed Frobenius accuracy: tight, since EM with known (A,B,C,R)
        # has only Q as a free parameter and the smoother is well-posed.
        for seed, rel in zip((3, 11, 23), rel_errs):
            self.assertLess(rel, 0.20,
                msg=f"seed {seed}: Q Frobenius rel err {rel:.3f} > 0.20")

        # Trace consistency across seeds: the true Q is fixed, so the
        # estimate's spread should be small (this is the property the
        # v1 estimator failed — it had a 2.3x spread on the Brain).
        self.assertLess(tr_spread, 0.20,
            msg=f"tr(Q) spread {tr_spread:.3f} across seeds "
                f"(traces {traces}, true {tr_true:.3f})")


if __name__ == "__main__":
    unittest.main()
