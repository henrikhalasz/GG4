"""control_policy.py -- the stateful Week-4 control wrapper (Step 2A).

The Week-4 control interface is a callable ``control_policy(observations, target)
-> u`` (see week4.ipynb): ``observations`` is the array of *past* outputs (empty
on the first call -- the current measurement is appended only *after* the call,
so the policy is always one step behind), and ``u`` is the command applied next.

``ReachingPolicy`` wraps the feedback ``Reacher`` as exactly that callable. It
owns all controller state: the reacher (which tracks joint angles by IK
continuity), the current index into a target sequence, and the one-step lag /
empty-first-call handling. It advances to the next target once the current one is
touched, and rests (commands zero) when the sequence is exhausted.

The seam to 2C is ``_emit``: in 2A the actuator command *is* the desired
joint-velocity ``v_des`` from the reacher; in 2C the same wrapper will translate
``v_des`` (via the calibrated primitive library) into the cascade input ``u``.
Nothing else in the wrapper changes.

Observation convention: each observation is the hand position (2,). When this
policy is later run against the real cascade, the harness supplies the live
``hand_pos`` as the observation (regime-A assumption, see cascade_plant.py).
"""

from __future__ import annotations

import numpy as np

from reacher import Reacher


class ReachingPolicy:
    def __init__(self, targets=None, *, mode: str = "one_joint", tol: float = 2.0,
                 settle_win: int = 8, settle_eps: float = 0.05, cascade: bool = False,
                 **driver_kw):
        self.cascade = bool(cascade)
        if self.cascade:
            # 2C: drive the real cascade through the calibrated primitives.
            from reach_controller import ReachController
            self.controller = ReachController(tol=tol, **driver_kw)
            self.reacher = self.controller.reacher   # exposed for the demo's arm drawing
            self.rest_u = np.array([0.5, 0.5])       # DC on (power stays latched), no AC
        else:
            # 2A: drive the IdealActuator with a joint-velocity command.
            self.controller = None
            self.reacher = Reacher(mode=mode, tol=tol, **driver_kw)
            self.rest_u = np.zeros(2)
        self.tol = float(tol)
        self.settle_win = int(settle_win)
        self.settle_eps = float(settle_eps)
        self._init_targets = None if targets is None else self._as_list(targets)
        self.reset()

    # --- single swappable hand_pos accessor (regime A) -----------------------
    def _hand_from_obs(self, obs) -> np.ndarray:
        """Extract the live hand position from the observation array.

        Regime A (our harness): the observation IS the clean ``CascadePlant.hand_pos``,
        so the latest observation is the hand. This is the ONE place to change if the
        scored harness delivers the hand by another channel; if it turns out to be
        regime B (16-D brain outputs only), this is where the Step-1 Kalman +
        cascade-replica dead-reckoning would plug in (not built -- see the brief)."""
        return np.asarray(obs, dtype=float).reshape(-1, 2)[-1]

    def _new_target_driver(self) -> None:
        (self.controller or self.reacher).new_target()

    def _drive(self, hand, tgt) -> np.ndarray:
        if self.cascade:
            return self.controller.step(hand, tgt)
        return self._emit(self.reacher.command(hand, tgt))

    @staticmethod
    def _as_list(targets) -> list:
        arr = np.asarray(targets, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, 2)
        return [np.asarray(t, dtype=float) for t in arr]

    def set_targets(self, targets) -> None:
        """Replace the target sequence (and reset state) -- used by the harness."""
        self._init_targets = None if targets is None else self._as_list(targets)
        self.reset()

    # --- state ---------------------------------------------------------------
    def reset(self) -> None:
        if self.cascade:
            self.controller.reset()
        else:
            self.reacher.reset()
            self.reacher.new_target()
        self.targets = list(self._init_targets) if self._init_targets else None
        self.idx = 0
        self.steps_to_touch: list = []
        self._step = 0          # internal step counter (for steps-to-touch)
        self._target_start = 0  # step index at which the current target began
        self._touched_step = None  # first step within tol of the current target
        self._recent: list = []  # recent hands (for settle detection)

    @property
    def done(self) -> bool:
        return self.targets is not None and self.idx >= len(self.targets)

    def current_target(self):
        if self.targets is None or self.done:
            return None
        return self.targets[self.idx]

    def _advance(self) -> None:
        """Move to the next target in the sequence (current one settled)."""
        self.idx += 1
        self._target_start = self._step
        self._touched_step = None
        self._recent = []
        self._new_target_driver()

    def _settled(self, hand) -> bool:
        """True once the hand has been within tol and nearly stationary."""
        self._recent.append(np.asarray(hand, dtype=float))
        if len(self._recent) > self.settle_win:
            self._recent.pop(0)
        if len(self._recent) < self.settle_win:
            return False
        rec = np.asarray(self._recent)
        return float(np.max(np.linalg.norm(np.diff(rec, axis=0), axis=1))) < self.settle_eps

    # --- the Week-4 callable -------------------------------------------------
    def __call__(self, observations, target=None) -> np.ndarray:
        # Adopt the target sequence supplied at call time (the Week-4 signature),
        # unless one was fixed at construction.
        if self.targets is None and target is not None:
            self.targets = self._as_list(target)

        obs = np.asarray(observations, dtype=float)
        if obs.size == 0:
            return self.rest_u.copy()       # empty first call -> rest (DC on for cascade)
        hand = self._hand_from_obs(obs)     # latest hand (one step behind), regime A
        self._step += 1

        tgt = self.current_target()
        if tgt is None:
            return self.rest_u.copy()       # sequence exhausted -> rest (hold DC)

        within = np.linalg.norm(tgt - hand) < self.tol
        if within and self._touched_step is None:
            # First touch: record steps-to-touch (but keep steering until settled).
            self._touched_step = self._step
            self.steps_to_touch.append(self._step - self._target_start)
        if within and self._settled(hand):
            # Settled within tolerance -> advance to the next target.
            self._advance()
            tgt = self.current_target()
            if tgt is None:
                return self.rest_u.copy()

        # Closed-loop steering: ideal -> joint-velocity command; cascade -> the
        # power-on-once / select-switch-stop ReachController (graded approach + lead).
        return self._drive(hand, tgt)

    def _emit(self, v_des) -> np.ndarray:
        """2A: the actuator command is the desired joint velocity itself.

        2C overrides this to translate ``v_des`` into the cascade input ``u``
        through the calibrated primitive library -- the only change at the seam.
        """
        return np.asarray(v_des, dtype=float)


def _main():
    """Drive an IdealActuator through a short target list via the wrapper."""
    from ideal_actuator import IdealActuator

    print("=== control_policy.py self-check ===")
    targets = np.array([[30.0, 30.0], [0.0, 40.0], [-35.0, 10.0], [45.0, -20.0]])
    policy = ReachingPolicy(targets, mode="one_joint", tol=2.0)
    plant = IdealActuator()

    # Mirror week4.ipynb run_closed_loop: measure (append AFTER the call) so the
    # policy is one step behind.
    observations = []
    T = 400
    for t in range(T):
        hand = plant.hand_pos
        u = policy(np.array(observations), targets)
        plant.command(u)
        observations.append(hand)
        if policy.done and np.linalg.norm(plant.v) < 1e-4:
            break

    hands = np.asarray(observations)
    print(f"ran {len(observations)} steps; targets reached: {policy.idx}/{len(targets)}")
    for i, tgt in enumerate(targets):
        d = np.min(np.linalg.norm(tgt - hands, axis=1))
        reached = "yes" if d < policy.tol else "NO"
        print(f"  target {i} {tgt.round(1)}: min dist = {d:.3f}  reached={reached}")
    print(f"steps-to-touch per target: {policy.steps_to_touch}")
    assert policy.idx == len(targets), "wrapper must reach every target in the list"
    # empty first call returns rest
    assert np.allclose(ReachingPolicy(targets)(np.array([]), targets), 0.0)
    print("All assertions passed.")


if __name__ == "__main__":
    _main()
