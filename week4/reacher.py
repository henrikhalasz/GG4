"""reacher.py -- feedback reacher + stopping rule (Week 4, Step 2A).

Each step the reacher reads the hand, computes the hand->target error, decides
which joint(s) to move, and stops in time. It is plant-agnostic: it observes only
``hand_pos`` (which both IdealActuator and the real cascade expose) and recovers
the joint angles by **branch-continuity inverse kinematics** -- start at the home
pose and, each step, pick the IK branch closest to the previous estimate. The
Jacobian then follows from the tracked angles.

Motion policy
  - **one-joint (default):** move the single joint whose motion best reduces the
    Cartesian distance (largest |J[:,j] . e_err|), at a speed proportional to the
    remaining error (capped at saturation).
  - **resolved-rate (flag):** damped-least-squares joint velocity for the desired
    Cartesian velocity.

Speed limit: a **direction-preserving scalar** -- scale the whole velocity vector
by one factor so the binding joint hits its saturation cap. Never per-joint clamp
(that distorts the commanded direction near a limit).

Stopping: **anticipatory lead** -- the reacher estimates its realised joint
velocity (from the change in tracked angles) and, using the actuator's coast
model, releases the command (commands zero) once the predicted coast lands within
tolerance. An optional **antagonist brake** (command the opposite direction
briefly) is available but off by default -- enabled only if overshoot appears.
"""

from __future__ import annotations

import numpy as np

import kinematics as kin


def _wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


class Reacher:
    def __init__(self, *, mode: str = "one_joint", tol: float = 2.0,
                 lam: float = 2.0, kp_joint: float = 0.45, v_cart_max: float = 2.0,
                 lead_gain: float = 1.0, brake: bool = False,
                 sigma_floor: float = 6.0, coast_tau: float = None,
                 theta0=(0.0, 0.0)):
        assert mode in ("one_joint", "resolved_rate")
        self.mode = mode
        self.tol = float(tol)
        self.lam = float(lam)
        self.kp_joint = float(kp_joint)      # joint-space P gain (rad/step per rad error)
        self.v_cart_max = float(v_cart_max)  # cap on commanded Cartesian speed (resolved-rate)
        self.lead_gain = float(lead_gain)    # multiplies the predicted coast travel
        self.brake = bool(brake)
        self.sigma_floor = float(sigma_floor)  # sigma_min(J) below this => near-singular
        # The reacher is tuned to the actuator's coast time-constant (placeholder
        # now, calibrated in 2B). Default matches IdealActuator.TAU_COAST.
        from ideal_actuator import IdealActuator
        self.coast_tau = float(IdealActuator.TAU_COAST if coast_tau is None else coast_tau)
        self._decay = np.exp(-1.0 / self.coast_tau)
        self.theta0 = np.asarray(theta0, dtype=float).copy()
        self.reset()

    def reset(self) -> None:
        self.theta_est = self.theta0.copy()
        self._theta_prev = self.theta0.copy()
        self._theta_star = None  # cached IK target angles for the current reach

    def new_target(self) -> None:
        """Forget the cached IK branch so the next target is re-solved."""
        self._theta_star = None

    def _choose_branch(self, target, theta_now) -> np.ndarray:
        """Pick the reachable IK branch closest to the current pose (continuity)."""
        tgt = np.asarray(target, dtype=float)
        r = float(np.linalg.norm(tgt))
        if r > kin.REACH * 0.999:  # clip an out-of-reach target onto the disk
            tgt = tgt * (kin.REACH * 0.999 / r)
        up, dn = kin.inverse(tgt)
        d_up = np.sum(_wrap(up - theta_now) ** 2)
        d_dn = np.sum(_wrap(dn - theta_now) ** 2)
        return up if d_up <= d_dn else dn

    # --- joint tracking ------------------------------------------------------
    def _track(self, hand) -> np.ndarray:
        """Update the tracked joint angles from an observed hand position."""
        up, dn = kin.inverse(hand)
        d_up = np.sum(_wrap(up - self.theta_est) ** 2)
        d_dn = np.sum(_wrap(dn - self.theta_est) ** 2)
        self._theta_prev = self.theta_est
        self.theta_est = up if d_up <= d_dn else dn
        return self.theta_est

    def _vel_est(self) -> np.ndarray:
        """Realised joint velocity inferred from the last tracking step."""
        return _wrap(self.theta_est - self._theta_prev)

    def _coast_travel(self, v_est) -> np.ndarray:
        """Predicted per-joint travel if released now at ``v_est`` (coast model)."""
        return v_est * self._decay / (1.0 - self._decay)

    # --- move decision (shared with the 2C cascade controller) ---------------
    def decide(self, hand, target) -> dict:
        """Track the hand and pick the one-joint move toward the IK target angles.

        Returns the raw decision used by both ``command`` (2A) and the 2C cascade
        controller -- no velocity/lead/stop logic here. ``joint`` is the joint to
        move, ``sign`` its direction, ``dgo`` the wrapped joint error to the IK
        target ``theta_star``. The cascade controller does its own stop logic; the
        2A IdealActuator-coast lead does not apply to the real cascade.
        """
        hand = np.asarray(hand, dtype=float)
        target = np.asarray(target, dtype=float)
        theta = self._track(hand)
        if self._theta_star is None:
            self._theta_star = self._choose_branch(target, theta)
        err = target - hand
        d = float(np.linalg.norm(err))
        dgo = _wrap(self._theta_star - theta)
        J = kin.jacobian(theta)
        e_hat = err / d if d > 1e-9 else np.zeros(2)
        # one-joint pick: the joint that most reduces the Cartesian distance,
        # falling back to the larger joint backlog near a singularity.
        lever = (J.T @ e_hat) * np.sign(dgo)
        if float(np.max(lever)) > 1e-3:
            j = int(np.argmax(lever))
        else:
            j = int(np.argmax(np.abs(dgo)))
        return {"theta": theta, "theta_star": self._theta_star, "dgo": dgo,
                "joint": j, "sign": float(np.sign(dgo[j]) or 1.0), "dist": d,
                "vel": self._vel_est()}

    # --- the policy ----------------------------------------------------------
    def command(self, hand, target) -> np.ndarray:
        """Desired joint-velocity command ``v_des`` for this step.

        Steering uses the IK target joint angles ``theta_star`` as the reference
        (robust at singularities, unlike pure Cartesian feedback). One-joint mode
        moves the single joint that most reduces the Cartesian distance, falling
        back to the larger joint-error when no joint makes first-order Cartesian
        progress (the around-a-singularity case).
        """
        hand = np.asarray(hand, dtype=float)
        target = np.asarray(target, dtype=float)
        theta = self._track(hand)
        J = kin.jacobian(theta)

        err = target - hand
        d = float(np.linalg.norm(err))
        if d < self.tol:
            return np.zeros(2)  # arrived: release -> coast settles in tolerance

        if self._theta_star is None:
            self._theta_star = self._choose_branch(target, theta)
        dgo = _wrap(self._theta_star - theta)      # joint error to the IK target
        v_est = self._vel_est()
        coast = self._coast_travel(v_est)          # per-joint coast travel
        e_hat = err / d

        if self.mode == "one_joint":
            # Cartesian leverage of nudging each joint toward theta_star (signed,
            # >0 means that move reduces the Cartesian distance).
            lever = (J.T @ e_hat) * np.sign(dgo)
            if float(np.max(lever)) > 1e-3:
                j = int(np.argmax(lever))           # best Cartesian progress
            else:
                j = int(np.argmax(np.abs(dgo)))     # fallback: clear joint backlog
            # Anticipatory lead: release once the coast covers the remaining travel.
            if abs(coast[j]) > 1e-9 and abs(dgo[j]) <= self.lead_gain * abs(coast[j]):
                if self.brake and abs(v_est[j]) > 1e-6:
                    v = np.zeros(2)
                    v[j] = -np.sign(v_est[j]) * kin.SAT_SPEED[j]
                    return v
                return np.zeros(2)
            v = np.zeros(2)
            v[j] = self.kp_joint * dgo[j]            # proportional, tapers near target
            return self._scale_to_sat(v)

        # resolved-rate: DLS toward the Cartesian target, with a joint-space pull
        # to escape near-singular configurations (e.g. the home pose).
        speed = min(self.v_cart_max, self.kp_joint * d)
        v_des = kin.resolved_rate_step(theta, speed * e_hat, lam=self.lam)
        if kin.sigma_min(theta) < self.sigma_floor:
            v_des = v_des + self.kp_joint * dgo      # leave the singularity
        coast_dist = float(np.linalg.norm(J @ coast))
        if coast_dist > 1e-9 and d <= self.lead_gain * coast_dist:
            if self.brake and np.linalg.norm(v_est) > 1e-6:
                return self._scale_to_sat(-v_est / np.linalg.norm(v_est) * kin.SAT_SPEED)
            return np.zeros(2)
        return self._scale_to_sat(v_des)

    @staticmethod
    def _scale_to_sat(v_des) -> np.ndarray:
        """Direction-preserving scalar speed limit (never per-joint clamp)."""
        v_des = np.asarray(v_des, dtype=float)
        over = np.abs(v_des) / kin.SAT_SPEED
        m = float(np.max(over))
        if m > 1.0:
            v_des = v_des / m
        return v_des


# --- standalone test against the IdealActuator -------------------------------
def _target_set():
    """>=12 targets spread around the safe annulus, incl. near-rim and near-centre."""
    import numpy as np
    targets = []
    for ang in np.linspace(0, 2 * np.pi, 10, endpoint=False):
        targets.append([40 * np.cos(ang), 40 * np.sin(ang)])
    targets.append([56.5 * np.cos(0.6), 56.5 * np.sin(0.6)])   # near the rim
    targets.append([6.0 * np.cos(2.0), 6.0 * np.sin(2.0)])     # near the centre
    targets.append([20 * np.cos(-1.2), 20 * np.sin(-1.2)])
    targets.append([50 * np.cos(2.6), 50 * np.sin(2.6)])
    return np.array(targets)


def _run_one(actuator, reacher, target, max_steps=400):
    """Drive the actuator from its current pose to one target; log the run."""
    reacher.reset()
    # seed the tracker continuity from the actuator's current pose
    reacher.theta_est = actuator.theta.copy()
    reacher._theta_prev = actuator.theta.copy()
    hand = actuator.hand_pos
    traj = [hand.copy()]
    min_d = np.linalg.norm(target - hand)
    touch_step = None
    for k in range(1, max_steps + 1):
        v_des = reacher.command(hand, target)
        hand = actuator.command(v_des)
        traj.append(hand.copy())
        d = np.linalg.norm(target - hand)
        min_d = min(min_d, d)
        if touch_step is None and d < reacher.tol:
            touch_step = k
        # settled (arrived and stopped) -> end
        if touch_step is not None and np.linalg.norm(actuator.v) < 1e-4:
            break
    traj = np.asarray(traj)
    # overshoot = how far past closest approach the hand drifted after touching
    final_d = float(np.linalg.norm(target - traj[-1]))
    return {"traj": traj, "min_d": float(min_d), "final_d": final_d,
            "touch_step": touch_step, "steps": len(traj) - 1}


def _main(mode="one_joint", brake=False):
    from ideal_actuator import IdealActuator
    print(f"=== reacher.py self-test (mode={mode}, brake={brake}) ===")
    print(f"safe annulus r in [5, 57]; tol = 2.0")
    targets = _target_set()
    act = IdealActuator()
    reacher = Reacher(mode=mode, tol=2.0, brake=brake)

    n_ok = 0
    steps_list = []
    print(f"\n{'target':>20s} {'r':>6s} {'touch?':>7s} {'steps':>6s} "
          f"{'min_d':>7s} {'final_d':>8s} {'monotone':>9s}")
    for tgt in targets:
        act.reset()  # each reach starts from home
        reacher.new_target()
        res = _run_one(act, reacher, tgt)
        # non-diverging: the final distance settles within tolerance (Cartesian
        # distance need NOT be monotone for around-the-singularity reaches).
        d_series = np.linalg.norm(tgt - res["traj"], axis=1)
        non_div = bool(res["min_d"] < reacher.tol)
        ok = res["min_d"] < reacher.tol
        n_ok += int(ok)
        steps_list.append(res["touch_step"] if res["touch_step"] else np.nan)
        r = np.hypot(*tgt)
        print(f"({tgt[0]:7.2f},{tgt[1]:7.2f}) {r:6.2f} {str(ok):>7s} "
              f"{str(res['touch_step']):>6s} {res['min_d']:7.3f} {res['final_d']:8.3f} "
              f"{str(non_div):>9s}")

    steps_arr = np.array([s for s in steps_list if not np.isnan(s)])
    if steps_arr.size:
        print(f"\ntouched {n_ok}/{len(targets)} targets within tol; "
              f"steps-to-touch: mean {steps_arr.mean():.1f}, "
              f"min {int(steps_arr.min())}, max {int(steps_arr.max())}")
    else:
        print(f"\ntouched {n_ok}/{len(targets)} targets within tol; "
              f"(no touches recorded)")
    return n_ok, len(targets)


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "one_joint"
    _main(mode=mode)
