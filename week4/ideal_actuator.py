"""ideal_actuator.py -- idealised actuator stand-in (Week 4, Step 2A).

The reacher (reacher.py) decides *which joint(s) to move and how fast*. To prove
that steering logic before any brain noise, we run it against an idealised
actuator that takes a **desired joint-velocity command directly** -- no muscles,
no rhythm decoding, deterministic. It is, however, not instantaneous: the real
cascade has a slow, accumulating onset (a single command moves nothing; a
sustained command ramps the hand over many steps -- measured in the regime-A
probe). We capture that with a tunable **onset lag** and, when the command
returns to zero, a **coast**. Both time-constants are *placeholders* here, to be
replaced by the 2B primitive calibration.

Common plant surface (mirrors week3/control/plant.py, re-implemented cleanly):
    hand_pos     -> clean (2,) hand position
    command(c)   -> advance one step under control command c
    reset()      -> back to the home pose

In 2A, ``c`` is a desired joint-velocity vector ``v_des = (omega_s, omega_e)``.
From 2C the same surface carries the cascade input ``u`` instead (CascadePlant).
"""

from __future__ import annotations

import numpy as np

import kinematics as kin


class IdealActuator:
    """Deterministic joint-velocity integrator with onset lag + coast.

    Each ``command(v_des)`` step:
      1. relax the realised joint velocity ``v`` toward ``v_des`` by a first-order
         lag (time-constant ``tau_on`` when driving, ``tau_coast`` when the
         command is ~zero, i.e. coasting to a stop);
      2. clamp ``|v|`` to the per-joint saturation speed (the genuine _Arm caps);
      3. integrate ``theta += v``.

    ``tau_on`` / ``tau_coast`` are PLACEHOLDERS (calibrated in 2B). With
    ``tau_on = tau_coast -> 0`` the actuator becomes the instantaneous limit
    ``theta += clip(v_des)``.
    """

    # Placeholder lag/coast time-constants, in steps (replaced by 2B calibration).
    TAU_ON = 3.0
    TAU_COAST = 4.0
    ZERO_CMD = 1e-9  # |v_des| below this counts as "commanding a stop" -> coast

    def __init__(self, theta0=(0.0, 0.0), *, tau_on: float = TAU_ON,
                 tau_coast: float = TAU_COAST):
        self.theta0 = np.asarray(theta0, dtype=float).copy()
        self.tau_on = float(tau_on)
        self.tau_coast = float(tau_coast)
        self._alpha_on = 1.0 - np.exp(-1.0 / self.tau_on)
        self._decay_coast = np.exp(-1.0 / self.tau_coast)
        self.reset()

    # --- common plant surface ------------------------------------------------
    def reset(self) -> None:
        self.theta = self.theta0.copy()
        self.v = np.zeros(2)  # realised joint velocity

    @property
    def hand_pos(self) -> np.ndarray:
        return kin.forward(self.theta)

    def command(self, c) -> np.ndarray:
        """Advance one step under desired joint velocity ``c = v_des``."""
        v_des = np.asarray(c, dtype=float)
        if np.linalg.norm(v_des) < self.ZERO_CMD:
            # Coast: first-order decay of the realised velocity toward zero.
            self.v = self.v * self._decay_coast
        else:
            # Onset lag: first-order relax toward the commanded velocity.
            self.v = self.v + self._alpha_on * (v_des - self.v)
        self.v = np.clip(self.v, -kin.SAT_SPEED, kin.SAT_SPEED)
        self.theta = self.theta + self.v
        return self.hand_pos

    # --- coast model (feeds the reacher's stopping lead) ---------------------
    def coast_travel(self, v=None) -> np.ndarray:
        """Remaining joint travel if we command a stop from velocity ``v`` now.

        With first-order coast ``v_{k+1} = decay * v_k``, the future angular
        travel is ``sum_{k>=1} decay^k * v = v * decay/(1 - decay)``.
        """
        v = self.v if v is None else np.asarray(v, dtype=float)
        return v * self._decay_coast / (1.0 - self._decay_coast)

    def coast_distance(self, v=None) -> float:
        """Cartesian hand distance covered while coasting to a stop from ``v``."""
        dtheta = self.coast_travel(v)
        return float(np.linalg.norm(kin.jacobian(self.theta) @ dtheta))


def _main():
    print("=== ideal_actuator.py self-check ===")
    act = IdealActuator()
    print(f"tau_on = {act.tau_on}, tau_coast = {act.tau_coast}")
    print(f"alpha_on = {act._alpha_on:.4f}, decay_coast = {act._decay_coast:.4f}")
    print(f"saturation = {kin.SAT_SPEED.round(6)} rad/step")

    # (1) Onset ramp: command full elbow velocity; watch v_elbow ramp to sat.
    act.reset()
    v_des = np.array([0.0, kin.SAT_ELBOW])  # drive the elbow at saturation
    print("\nOnset ramp (command v_des =", v_des.round(4), "):")
    for k in range(1, 11):
        act.command(v_des)
        frac = act.v[1] / kin.SAT_ELBOW
        print(f"  step {k:2d}: v_elbow = {act.v[1]:.5f}  ({100*frac:5.1f}% of sat)")
    # (2) Coast: command stop, measure how far the elbow still travels.
    v_now = act.v.copy()
    travel = act.coast_travel(v_now)
    dist = act.coast_distance(v_now)
    print(f"\nFrom v = {v_now.round(5)}:")
    print(f"  predicted coast joint-travel = {travel.round(5)} rad")
    print(f"  predicted coast hand-distance = {dist:.4f} units")
    # Verify the prediction by actually coasting.
    theta_before = act.theta.copy()
    for _ in range(200):
        act.command(np.zeros(2))
    realised = act.theta - theta_before
    print(f"  realised coast joint-travel  = {realised.round(5)} rad "
          f"(err {np.max(np.abs(realised - travel)):.2e})")
    assert np.allclose(realised, travel, atol=1e-6), "coast model must match integration"

    # (3) Saturation clamp holds even for an over-large command.
    act.reset()
    for _ in range(50):
        act.command(np.array([10.0, -10.0]))
    assert np.all(np.abs(act.v) <= kin.SAT_SPEED + 1e-12), "must clamp to saturation"
    print(f"\nOver-large command clamps to sat: v = {act.v.round(6)}")
    print("All assertions passed.")


if __name__ == "__main__":
    _main()
