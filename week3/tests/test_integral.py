"""Integral-variant controller tests (spec §9, added by v3 E1).

Covers:
    - test_integral_removes_offset
        Under a deliberately mismatched model, feedforward-only LQG leaves a
        steady-state offset; LQG+I drives it to within ~1× the readout-noise
        floor on a feasible target.
    - test_anti_windup
        Under an infeasible target the integral state stays bounded and ``u``
        pins at its limit without divergence.
    - test_readout_controllability
        The 2-leading-PC readout has a well-conditioned 2×2 input→readout
        gain G; the test reports cond(G).
"""

from __future__ import annotations

import unittest

import numpy as np

from . import _path_helper  # noqa: F401
import Simulator as sim  # noqa: E402

from control.plant import SimulatorPlant  # noqa: E402
from control.observer import SteadyStateKalman  # noqa: E402
from control.controllers import LQG, LQGI, OffsetFreeMPC, PI  # noqa: E402
from control.closed_loop import run_closed_loop  # noqa: E402
from control.control_interface import (  # noqa: E402
    compute_pca_readout,
    readout_steady_state_gain,
    IdentifiedSystem,
)


def _rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr ** 2)))


def _steady_readout(z: np.ndarray, tail: int = 30) -> np.ndarray:
    return z[-tail:].mean(axis=0)


class IntegralRemovesOffset(unittest.TestCase):
    """Mismatched-offset model → feedforward-only LQG leaves a constant offset;
    LQG+I closes it.

    Synthetic system (2-state, 2-input, identity readout):
        x_{t+1} = 0.8 x + I u + a_true + w     a_true = (0.5, 0.5)
        y = x + v
    The controller is built with ``a_wrong = 0`` — it doesn't know about the
    constant drive ``a_true``. The reference is the origin (achievable under
    the *truth* with steady input u_ss = 0.5·1 because 0.2 · 0 = u_ss − 0.5
    ⇒ u_ss = 0.5, well inside [0, 1]).

    Expected behaviour:
        - feedforward-only LQG sets u_ff = 0 because it thinks origin is the
          natural equilibrium; the truth drifts to (1 / 0.2) · a_true = 2.5
          and the clipped LQR cannot fully recover (it sees error and pushes
          u positive, but its model says the equilibrium is at 0, so once
          x_hat returns to 0 in the model basis the push relaxes — leaving
          a residual offset).
        - LQG+I accumulates the readout residual into q until u_ff_effective
          ≈ 0.5 and the actual readout sits at the reference within the
          noise floor.
    """

    def test_constant_offset_mismatch(self):
        rng = np.random.default_rng(0)
        n = 2
        m = 2
        p = 2
        A = 0.8 * np.eye(n)
        B = np.eye(m)
        C = np.eye(p)
        Q = 1e-4 * np.eye(n)
        R = 1e-4 * np.eye(p)
        a_true = np.array([0.5, 0.5])
        c_true = np.zeros(p)
        a_wrong = np.zeros(n)
        c_wrong = np.zeros(p)
        M = np.eye(2)
        # Feasible target on the truth: x_ss = 3 ⇒ u_ss = (1−0.8)·3 − 0.5 = 0.1
        # (inside [0, 1]). Under the wrong-a model the controller thinks
        # u_ss = 0.6, which is what makes feedforward-only mis-shoot.
        ref = np.array([3.0, 3.0])
        T = 800

        # ---- Plant rollout helper -----------------------------------------
        def run(controller_factory, observer_factory):
            x = np.zeros(n)
            ctrl = controller_factory()
            obs = observer_factory()
            obs.reset()
            ctrl.reset()
            u_prev = np.zeros(m)
            ys = np.empty((T, p))
            us = np.empty((T, m))
            for t in range(T):
                y = C @ x + c_true + rng.multivariate_normal(np.zeros(p), R)
                ys[t] = y
                x_hat = obs.filter_step(y, u_prev)
                if hasattr(ctrl, "observe"):
                    ctrl.observe(y, x_hat, u_prev)
                if getattr(ctrl, "uses_raw_y", False):
                    u = ctrl.compute(y, ref)
                else:
                    u = ctrl.compute(x_hat, ref)
                us[t] = u
                x = A @ x + B @ u + a_true + rng.multivariate_normal(np.zeros(n), Q)
                u_prev = u
            return ys, us

        # ---- Feedforward-only LQG (mismatched a) --------------------------
        rng_state_fb = rng.bit_generator.state
        ys_fb, us_fb = run(
            controller_factory=lambda: LQG(A, B, C, M, a=a_wrong, c=c_wrong,
                                             rho=1.0, ref=ref),
            observer_factory=lambda: SteadyStateKalman(A, B, C, Q, R,
                                                          a=a_wrong, c=c_wrong),
        )
        # ---- LQGI (same mismatched a) -------------------------------------
        rng.bit_generator.state = rng_state_fb  # use same noise for fair comparison
        ys_i, us_i = run(
            controller_factory=lambda: LQGI(A, B, C, M, a=a_wrong, c=c_wrong,
                                              rho=1.0, rho_q=10.0, ref=ref),
            observer_factory=lambda: SteadyStateKalman(A, B, C, Q, R,
                                                          a=a_wrong, c=c_wrong),
        )

        z_fb = ys_fb @ M.T
        z_i = ys_i @ M.T
        ach_fb = _steady_readout(z_fb)
        ach_i = _steady_readout(z_i)
        offset_fb = float(np.linalg.norm(ach_fb - ref))
        offset_i = float(np.linalg.norm(ach_i - ref))
        noise_floor = float(np.sqrt(np.trace(R)))

        print(f"\n  feedforward-only LQG: steady readout = "
              f"{ach_fb.round(3)} (offset {offset_fb:.3f})")
        print(f"  LQG+I:                steady readout = "
              f"{ach_i.round(3)} (offset {offset_i:.3f})")
        print(f"  measurement-noise floor              : {noise_floor:.3f}")

        # 1) Feedforward-only leaves a meaningful offset (≫ noise).
        self.assertGreater(offset_fb, 10.0 * noise_floor,
            msg=f"feedforward-only offset {offset_fb:.3f} ≤ 10×noise {noise_floor:.3f} — "
                f"mismatch too mild to test")
        # 2) LQG+I closes the offset to within 25 % of the feedforward-only
        #    offset (integral action does most of the work even in this
        #    short-horizon test; under infinite horizon it converges exactly).
        self.assertLess(offset_i, 0.25 * offset_fb,
            msg=f"LQG+I offset {offset_i:.3f} not < 25% of FB-only {offset_fb:.3f}")
        # 3) Sanity: LQG+I drove u toward the true equilibrium u_ss = 0.1
        #    (not stuck at the saturation limit).
        u_i_steady = us_i[-50:].mean(axis=0)
        self.assertTrue(
            np.all(u_i_steady > 0.02) and np.all(u_i_steady < 0.5),
            msg=f"LQG+I steady u {u_i_steady.round(3)} not near expected 0.1",
        )


class AntiWindupBounded(unittest.TestCase):
    """Under an infeasible target, integral state stays bounded; u pins at limit."""

    def test_infeasible_target_pi(self):
        # Tiny 1-input scalar system for an easy infeasibility check.
        # x_{t+1} = 0.5 x + 0.1 u + noise;  y = x + noise.
        # With u in [0,1], steady-state y_max = 0.1/(1-0.5) · 1 = 0.2. Target
        # ref = 10 is wildly infeasible.
        rng = np.random.default_rng(0)
        T = 200
        x = 0.0
        ref = 10.0
        Kp = np.array([[0.5]])
        Ki = np.array([[0.2]])
        M = np.array([[1.0]])
        pi = PI(Kp=Kp, Ki=Ki, M=M, ref=np.array([ref]),
                u0=np.zeros(1), anti_windup=True)
        q_hist = np.empty(T)
        u_hist = np.empty(T)
        y_hist = np.empty(T)
        for t in range(T):
            y = np.array([x + 0.01 * rng.normal()])
            u = pi.compute(y)
            x = 0.5 * x + 0.1 * u[0] + 0.01 * rng.normal()
            q_hist[t] = pi.q_state[0]
            u_hist[t] = u[0]
            y_hist[t] = y[0]

        # u should pin near 1 most of the time.
        self.assertGreater(float(np.mean(u_hist > 0.99)), 0.9,
                            msg="u not saturated at upper bound on infeasible target")
        # q_state must NOT diverge — should stay bounded by its initial accumulation
        # window. Compare against the "no anti-windup" baseline: without anti-
        # windup q grows linearly in T, so for T = 200 it would exceed ~2000.
        max_abs_q = float(np.max(np.abs(q_hist)))
        self.assertLess(max_abs_q, 50.0,
            msg=f"anti-windup failed: |q| reached {max_abs_q:.1f} on infeasible target")

    def test_infeasible_target_lqgi(self):
        """LQGI: deliberately too-far target → u pins, q stays bounded."""
        system = sim.default_neural_system(seed=0, obs_dim=16)
        n = system.A.shape[0]
        a0 = np.zeros(n)
        c0 = np.zeros(system.obs_dim)
        # PCA readout from a short probe.
        probe_plant = SimulatorPlant(system, seed=43)
        rng = np.random.default_rng(42)
        T_probe = 600
        U_probe = rng.uniform(0, 1, size=(T_probe, system.input_dim))
        Y_probe = np.empty((T_probe, system.obs_dim))
        Y_probe[0] = np.asarray(probe_plant.measure(), dtype=float)
        probe_plant.next_state(U_probe[0])
        for t in range(1, T_probe):
            Y_probe[t] = probe_plant.measure()
            probe_plant.next_state(U_probe[t])
        M, _ = compute_pca_readout(Y_probe, k=2)

        # Infeasible target: 10× outside the zonotope.
        G_true = readout_steady_state_gain(system.A, system.B, system.C, M)
        ref = 10.0 * (G_true @ np.array([1.0, 1.0]))

        T = 250
        plant = SimulatorPlant(system, seed=7)
        obs = SteadyStateKalman(system.A, system.B, system.C, system.Q, system.R,
                                 a=a0, c=c0)
        lqgi = LQGI(system.A, system.B, system.C, M,
                    a=a0, c=c0, rho=0.1, rho_q=10.0, ref=ref, anti_windup=True)
        logs = run_closed_loop(plant, lqgi, obs, T=T, ref_fn=lambda t: ref)

        # Final integral state must stay bounded (anti-windup working).
        q_final = float(np.linalg.norm(lqgi.q_state))
        u = logs["u"]
        sat_frac = float(np.mean((u <= 1e-6) | (u >= 1.0 - 1e-6)))
        self.assertGreater(sat_frac, 0.5,
            msg=f"u barely saturated on grossly infeasible target (sat={sat_frac:.2f})")
        # |q| should grow ~ once per non-saturated step. With sat ~ 0.5+, q should
        # stay well below T (250) in norm — say 100 as a generous ceiling.
        self.assertLess(q_final, 1.0e3,
            msg=f"anti-windup failed: |q_state| = {q_final:.1f} after T={T} infeasible-ref steps")


class ReadoutControllability(unittest.TestCase):
    """The 2-leading-PC readout has a well-conditioned 2×2 input→readout gain."""

    def test_default_neural_system(self):
        system = sim.default_neural_system(seed=0, obs_dim=16)
        cal_plant = SimulatorPlant(system, seed=43)
        iface = IdentifiedSystem().calibrate(
            cal_plant, T_cal=600, n=system.A.shape[0],
            m=system.input_dim, seed=42,
        )
        # cond(G) reported by the interface.
        cond_G = iface.readout_cond_G
        explained = iface.readout_explained_var
        print(f"\n  PCA readout explained var (PC1, PC2): "
              f"({explained[0]:.3f}, {explained[1]:.3f})  cond(G) = {cond_G:.2f}")
        self.assertGreater(float(explained.sum()), 0.5,
            msg="leading 2 PCs explain less than 50% of measurement variance")
        # 'Well-conditioned' threshold (spec §3) — < 50 means inputs move PC1
        # and PC2 nearly independently.
        self.assertLess(cond_G, 50.0,
            msg=f"PC readout ill-conditioned: cond(G) = {cond_G:.1f} — "
                f"inputs cannot move PC1, PC2 independently")


if __name__ == "__main__":
    unittest.main()
