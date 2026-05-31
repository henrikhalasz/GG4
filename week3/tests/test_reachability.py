"""Reachability zonotope correctly classifies feasible vs infeasible setpoints."""

from __future__ import annotations

import unittest

import numpy as np

from . import _path_helper  # noqa: F401
import Simulator as sim  # noqa: E402

from control.reachability import steady_state_gain, zonotope_vertices, feasibility  # noqa: E402


class ZonotopeFeasibility(unittest.TestCase):
    def test_default_neural_system(self):
        system = sim.default_neural_system(seed=0, obs_dim=16)
        M = np.zeros((2, system.obs_dim))
        M[0, 0] = 1.0
        M[1, 1] = 1.0

        n = system.A.shape[0]
        a0 = np.zeros(n)
        c0 = np.zeros(system.obs_dim)
        G, z0 = steady_state_gain(system.A, system.B, system.C, M, a=a0, c=c0)
        self.assertEqual(G.shape, (2, 2))
        verts = zonotope_vertices(G, z0)
        self.assertEqual(verts.shape, (4, 2))
        # z0 = 0 because a = 0 and c = 0 -> the floor vertex sits at the origin.
        self.assertTrue(np.allclose(z0, 0.0))

        # Origin is the affine floor vertex.
        inside_origin, u_origin = feasibility(np.zeros(2), G, z0)
        self.assertTrue(inside_origin)
        self.assertTrue(np.allclose(u_origin, 0.0, atol=1e-6))

        # Midpoint of zonotope: u = (0.5, 0.5).
        z_mid = G @ np.array([0.5, 0.5]) + z0
        inside_mid, u_mid = feasibility(z_mid, G, z0)
        self.assertTrue(inside_mid)
        self.assertTrue(np.allclose(u_mid, 0.5, atol=1e-6))

        # A point clearly outside: scale a vertex by 3 (in offset-free coords).
        z_far = 3.0 * (G @ np.array([1.0, 1.0])) + z0
        inside_far, _ = feasibility(z_far, G, z0)
        self.assertFalse(inside_far)

        # Negative of a vertex (relative to z0) — non-negative actuator can't reach it.
        z_neg = -1.0 * (G @ np.array([1.0, 1.0])) + z0
        inside_neg, _ = feasibility(z_neg, G, z0)
        self.assertFalse(inside_neg)


if __name__ == "__main__":
    unittest.main()
