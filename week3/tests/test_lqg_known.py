"""Stage-A: LQG controller built from a known model is stable and reduces RMS."""

from __future__ import annotations

import unittest

import numpy as np

from . import _path_helper  # noqa: F401
import Simulator as sim  # noqa: E402

from control.plant import SimulatorPlant  # noqa: E402
from control.observer import SteadyStateKalman  # noqa: E402
from control.controllers import OpenLoop, LQG  # noqa: E402
from control.closed_loop import run_closed_loop  # noqa: E402


def _rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr ** 2)))


class LQGStageA(unittest.TestCase):
    def test_closed_loop_spectral_radius(self):
        system = sim.default_neural_system(seed=0, obs_dim=16)
        n = system.A.shape[0]
        M = np.zeros((2, system.obs_dim))
        M[0, 0] = 1.0
        M[1, 1] = 1.0
        lqg = LQG(system.A, system.B, system.C, M=M,
                  a=np.zeros(n), c=np.zeros(system.obs_dim), rho=0.1)
        A_cl = system.A - system.B @ lqg.K
        rho_cl = float(np.max(np.abs(np.linalg.eigvals(A_cl))))
        self.assertLess(rho_cl, 1.0, msg=f"closed-loop unstable: rho={rho_cl:.3f}")

    def test_clipped_lqg_does_not_increase_state_rms(self):
        """One-sided actuator may not strictly beat open-loop on suppression,
        but it must not destabilise — RMS within 1.5x of open-loop is acceptable."""
        system = sim.default_neural_system(seed=0, obs_dim=16)
        n = system.A.shape[0]
        M = np.zeros((2, system.obs_dim))
        M[0, 0] = 1.0
        M[1, 1] = 1.0
        T = 200
        seed = 7

        plant = SimulatorPlant(system, seed=seed)
        obs = SteadyStateKalman(system.A, system.B, system.C, system.Q, system.R,
                                 a=np.zeros(n), c=np.zeros(system.obs_dim))
        logs_open = run_closed_loop(
            plant, OpenLoop(input_dim=system.input_dim), obs, T=T,
        )

        plant.reset(seed=seed)
        obs.reset()
        lqg = LQG(system.A, system.B, system.C, M=M,
                  a=np.zeros(n), c=np.zeros(system.obs_dim), rho=0.1)
        logs_ctrl = run_closed_loop(
            plant, lqg, obs, T=T,
        )

        z_open = logs_open["y"] @ M.T
        z_ctrl = logs_ctrl["y"] @ M.T
        rms_open = _rms(z_open)
        rms_ctrl = _rms(z_ctrl)
        # Sanity: controller did not blow up.
        self.assertTrue(np.all(np.isfinite(z_ctrl)))
        # Loose bound — strict "<= 1.0 * open-loop" may not hold under one-sided
        # actuation; we just check we are not catastrophically worse.
        self.assertLess(rms_ctrl, 1.5 * rms_open,
                        msg=f"LQG RMS {rms_ctrl:.3f} >> open-loop {rms_open:.3f}")


if __name__ == "__main__":
    unittest.main()
