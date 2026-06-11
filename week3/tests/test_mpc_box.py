"""MPC respects the box constraint and is competitive with clipped LQR."""

from __future__ import annotations

import unittest

import numpy as np

from . import _path_helper  # noqa: F401
import Simulator as sim  # noqa: E402

from control.plant import SimulatorPlant  # noqa: E402
from control.observer import SteadyStateKalman  # noqa: E402
from control.controllers import LQG, MPC  # noqa: E402
from control.closed_loop import run_closed_loop  # noqa: E402
from metrics import rms, control_effort  # noqa: E402


class MPCBoxConstraintAndRMS(unittest.TestCase):
    def test_default_neural_system(self):
        system = sim.default_neural_system(seed=0, obs_dim=16)
        n = system.A.shape[0]
        M = np.zeros((2, system.obs_dim))
        M[0, 0] = 1.0
        M[1, 1] = 1.0
        T = 150
        seed = 5
        a0 = np.zeros(n)
        c0 = np.zeros(system.obs_dim)

        # MPC
        plant = SimulatorPlant(system, seed=seed)
        obs = SteadyStateKalman(system.A, system.B, system.C, system.Q, system.R, a=a0, c=c0)
        mpc = MPC(system.A, system.B, system.C, M=M, a=a0, c=c0, horizon=15, rho=0.1)
        logs_mpc = run_closed_loop(plant, mpc, obs, T=T)

        # Constraint compliance: no violations.
        u = logs_mpc["u"]
        self.assertTrue(np.all(np.isfinite(u)), "MPC produced non-finite u")
        self.assertTrue(np.all(u >= 0.0 - 1e-9), f"MPC u below 0: min={u.min():.4g}")
        self.assertTrue(np.all(u <= 1.0 + 1e-9), f"MPC u above 1: max={u.max():.4g}")

        # Clipped LQR baseline
        plant.reset(seed=seed)
        obs.reset()
        lqg = LQG(system.A, system.B, system.C, M=M, a=a0, c=c0, rho=0.1)
        logs_lqg = run_closed_loop(plant, lqg, obs, T=T)

        z_mpc = logs_mpc["y"] @ M.T
        z_lqg = logs_lqg["y"] @ M.T
        rms_mpc = rms(z_mpc)
        rms_lqg = rms(z_lqg)
        eff_mpc = control_effort(logs_mpc["u"])
        eff_lqg = control_effort(logs_lqg["u"])

        # Sanity: MPC didn't blow up.
        self.assertLess(rms_mpc, 2.0 * rms_lqg,
                        msg=f"MPC RMS {rms_mpc:.3f} >> LQR RMS {rms_lqg:.3f}")
        # Diagnostic — these will be reported by the experiment script too.
        print(f"\n  MPC: RMS={rms_mpc:.4f}  effort={eff_mpc:.3f}")
        print(f"  LQR: RMS={rms_lqg:.4f}  effort={eff_lqg:.3f}")


if __name__ == "__main__":
    unittest.main()
