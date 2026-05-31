"""Diagnostic: B sign-flipped → controller is inert (u ≡ 0).

Spec §9 diagnostic — feed the controller TRUE params but with B sign-flipped.
On a one-sided actuator (u ∈ [0, 1]) the resulting feedforward `u_ff` for any
non-zero target lands in the negative quadrant; clip(·, 0, 1) collapses it to
zero. This reproduces the v1 "inert controller" failure mode mechanically.
"""

from __future__ import annotations

import unittest

import numpy as np

from . import _path_helper  # noqa: F401
import Simulator as sim  # noqa: E402

from control.plant import SimulatorPlant  # noqa: E402
from control.observer import SteadyStateKalman  # noqa: E402
from control.controllers import LQG  # noqa: E402
from control.closed_loop import run_closed_loop  # noqa: E402
from control.reachability import steady_state_gain  # noqa: E402


class SignFlippedBProducesInertController(unittest.TestCase):
    def test_default_neural_system(self):
        system = sim.default_neural_system(seed=0, obs_dim=16)
        n = system.A.shape[0]
        M = np.zeros((2, system.obs_dim))
        M[0, 0] = 1.0
        M[1, 1] = 1.0
        a0 = np.zeros(n)
        c0 = np.zeros(system.obs_dim)
        # Feasible target with TRUE B (positive corner).
        G_true, z0_true = steady_state_gain(system.A, system.B, system.C, M, a=a0, c=c0)
        ref = G_true @ np.array([0.5, 0.5]) + z0_true

        plant = SimulatorPlant(system, seed=7)
        obs = SteadyStateKalman(system.A, system.B, system.C, system.Q, system.R, a=a0, c=c0)
        # Same matrices the plant uses for everything EXCEPT we hand the controller -B.
        lqg = LQG(system.A, -system.B, system.C, M, a=a0, c=c0, rho=0.1, ref=ref)
        logs = run_closed_loop(plant, lqg, obs, T=100, ref_fn=lambda t: ref)

        # With B sign-flipped the requested u_ff is negative -> clip(0, 1) = 0.
        # The closed-loop u should be ≡ 0 (the inert state).
        u = logs["u"]
        max_u = float(np.max(np.abs(u)))
        self.assertLess(max_u, 1e-6,
                        msg=f"sign-flipped B should produce u ≡ 0; got max|u| = {max_u}")


if __name__ == "__main__":
    unittest.main()
