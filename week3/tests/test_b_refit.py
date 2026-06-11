"""IdentifiedSystem.calibrate (Stage B wiring through estimator/identify).

Verifies the calibration round-trip on default_neural_system:
  - shapes correct,
  - degenerate flags clean,
  - fitted G matches true G within tolerance.
"""

from __future__ import annotations

import unittest

import numpy as np

from . import _path_helper  # noqa: F401
import Simulator as sim  # noqa: E402

from control.control_interface import IdentifiedSystem  # noqa: E402
from control.plant import SimulatorPlant  # noqa: E402
from control.reachability import steady_state_gain  # noqa: E402


class CalibrationProducesUsableModel(unittest.TestCase):
    def test_default_neural_system(self):
        system = sim.default_neural_system(seed=0, obs_dim=16)
        plant = SimulatorPlant(system, seed=11)
        n = system.A.shape[0]

        iface = IdentifiedSystem().calibrate(plant, T_cal=800, n=n, m=2, seed=42)

        # Shapes
        self.assertEqual(iface.A.shape, (n, n))
        self.assertEqual(iface.B.shape, (n, system.input_dim))
        self.assertEqual(iface.C.shape, (system.obs_dim, n))
        self.assertEqual(iface.a.shape, (n,))
        self.assertEqual(iface.c.shape, (system.obs_dim,))

        # Finite, no degenerate flags
        self.assertEqual(iface.degenerate_flags, [],
                         msg=f"unexpected degenerate flags: {iface.degenerate_flags}")
        self.assertLess(iface.diagnostics["A_spectral_radius"], 1.0,
                        msg=f"A unstable: {iface.diagnostics['A_spectral_radius']}")

        # B column norms in true input units should be non-trivial.
        col_norms = np.linalg.norm(iface.B, axis=0)
        self.assertTrue(np.all(col_norms > 1e-3),
                        msg=f"B columns near zero: {col_norms}")

        # G match: project to M-readout subspace (we use M = I in the
        # 2 readout dims used by figure 1 — top-2 components).
        M = np.zeros((2, system.obs_dim))
        M[0, 0] = 1.0
        M[1, 1] = 1.0
        G_fit, z0_fit = steady_state_gain(iface.A, iface.B, iface.C, M,
                                          a=iface.a, c=iface.c)
        G_true, z0_true = steady_state_gain(system.A, system.B, system.C, M,
                                            a=np.zeros(n), c=np.zeros(system.obs_dim))
        G_rel = np.linalg.norm(G_fit - G_true) / max(np.linalg.norm(G_true), 1e-12)
        self.assertLess(G_rel, 0.15,
                        msg=f"M-projected G relative error {G_rel:.3f} too high")


if __name__ == "__main__":
    unittest.main()
