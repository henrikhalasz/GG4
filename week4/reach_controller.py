"""reach_controller.py -- close the loop on the real cascade (Week 4, Step 2C).

The 2A reacher emits a per-step joint *velocity*; the cascade cannot follow that.
To move a joint we pick a muscle (joint+sign) and oscillate input 0 at its band
frequency -- it then moves at a roughly fixed steady speed (graded by amplitude),
after a slow ~94-step cold onset (paid once) and a ~50-step warm switch to
re-select, and it coasts after the AC is cut. ``ReachController`` is the state
machine that drives the real arm under hand feedback with that alphabet:

  - **Power on once:** hold input-0 DC at 0.5 the entire run (idle/stop included),
    so power latches once (~94 steps) and never re-ramps.
  - **Select / switch / stop by AC frequency.** Drive the chosen muscle's band at
    amplitude A; switch muscles by changing the band; stop by dropping the AC.
  - **One joint at a time:** commit to the joint with the larger remaining IK
    joint-error, drive it to (within its coast of) its target angle, then switch.
  - **Graded approach + live-coast lead:** reduce A as the joint nears its target
    so the coast shrinks, and cut the AC when the remaining angle is within the
    *live* coast estimate (= measured joint speed x coast_steps); the coast lands
    it on target. Feedback corrects any residual on the next move.
  - **el- watchdog:** el- is slow and misfires ~25%; if a move makes no motion
    within its onset/warm window, re-trigger (drop+reapply the AC).

It reuses the 2A ``Reacher`` only for IK tracking + target-joint-angle selection
(``Reacher.decide``); all cascade logic lives here. Recipe numbers are loaded from
``primitive_library.npz`` (2B) -- never hard-coded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import kinematics as kin
from reacher import Reacher, _wrap

HERE = Path(__file__).resolve().parent
LIB_PATH = HERE / "primitive_library.npz"


def load_library(path=LIB_PATH) -> dict:
    """Per-muscle recipe dict keyed by muscle index 0..3 (sh+,sh-,el+,el-)."""
    d = np.load(path, allow_pickle=True)
    lib = {}
    for k in range(4):
        v = float(d[f"steady_v_{k}"][0])
        coast = float(d[f"coast_{k}"][0])
        lib[k] = {
            "name": str(d["muscle_names"][k]),
            "f": float(d["band_f"][k]),
            "A_star": float(d[f"A_star_{k}"]),
            "steady_v": v,
            "coast": coast,
            "coast_steps": coast / abs(v) if abs(v) > 1e-6 else 27.0,
            "warm": float(d[f"warm_{k}"][0]),
            "onset": float(d[f"onset_{k}"][0]),
        }
    return lib


def muscle_index(joint: int, sign: float) -> int:
    """(joint 0=shoulder/1=elbow, sign +/-) -> muscle 0 sh+ /1 sh- /2 el+ /3 el-."""
    return 2 * int(joint) + (0 if sign > 0 else 1)


# Warm speed-vs-amplitude model per muscle (measured live, seed 0):
#   steady |v| ~= slope * (A - A0),  valid above the firing floor A_min.
# Used to pick the amplitude whose coast matches the remaining joint angle.
SPEED_MODEL = {
    0: (0.198, 0.081, 0.10),   # sh+ (fires down to A~0.10, coast ~3 units)
    1: (0.144, 0.136, 0.13),   # sh-
    2: (0.130, 0.330, 0.35),   # el+ (sharp threshold near 0.33)
    3: (0.066, 0.371, 0.40),   # el- (bottleneck; barely gradable)
}


class ReachController:
    REST = np.array([0.5, 0.5])  # DC on, no AC -> power stays latched, arm freezes

    def __init__(self, *, tol: float = 2.0, lib=None, lib_path=LIB_PATH,
                 sh_A_min: float = 0.16, sh_grade_zone: float = 1.0,
                 vbar_decay: float = 0.6, settle_vel: float = 0.004,
                 lead_margin: float = 0.05, switch_margin: float = 0.06,
                 speed_gain: float = 0.75, capture_mult: float = 3.0,
                 sticky_capture: bool = False,
                 cart_slow_zone: float = 14.0, cart_min_factor: float = 0.16,
                 cold_window: int = 120, warm_pad: int = 35, retrig_steps: int = 5):
        self.lib = lib if lib is not None else load_library(lib_path)
        self.reacher = Reacher(mode="one_joint", tol=tol)
        self.tol = float(tol)
        self.sh_A_min = float(sh_A_min)            # shoulder grades down to this (fires ~0.01/step)
        self.sh_grade_zone = float(sh_grade_zone)  # rad; start slowing the shoulder within this
        self.vbar_decay = float(vbar_decay)        # EMA factor for the velocity estimate
        self.settle_vel = float(settle_vel)        # ||joint vel|| below this = coast settled
        self.lead_margin = float(lead_margin)      # rad, extra lead before the coast
        self.switch_margin = float(switch_margin)  # rad, a joint below this needs no drive
        self.speed_gain = float(speed_gain)        # <1 biases the approach to undershoot (creep in)
        self.capture_mult = float(capture_mult)    # hold band = capture_mult * tol
        self.sticky_capture = bool(sticky_capture)  # once reached, hold for the rest of the target
        self.cart_slow_zone = float(cart_slow_zone)    # slow all drives as the HAND nears target
        self.cart_min_factor = float(cart_min_factor)
        self.cold_window = int(cold_window)
        self.warm_pad = float(warm_pad)
        self.retrig_steps = int(retrig_steps)
        self.reset()

    def reset(self) -> None:
        self.reacher.reset()
        self.reacher.new_target()
        self.t = 0                 # global phase / step counter (DC held throughout)
        self.move = None           # committed (joint, sign)
        self.captured = False      # within-tol hysteresis (hold once reached)
        self.move_age = 0
        self.vbar = 0.0            # EMA |velocity| of the active joint
        self.retrig = 0
        self.n_retrig = 0          # diagnostics: total re-triggers this reach
        self.last = {}
        self.draw_theta = self.reacher.theta_est.copy()

    # --- matched-coast graded amplitude --------------------------------------
    def _amplitude(self, m: int, dgo_abs: float, cart_factor: float = 1.0) -> float:
        """Pick A so the coast (~speed x coast_steps) matches the remaining angle,
        so the joint decelerates and lands on target instead of overshooting.
        ``cart_factor`` (<=1 as the HAND nears the target) slows every joint so a
        fast move that reaches Cartesian tolerance has little coast left. Clamped to
        the muscle's firing floor .. A* (the elbow only grades a little)."""
        slope, a0, a_min = SPEED_MODEL[m]
        a_star = self.lib[m]["A_star"]
        desired_v = cart_factor * self.speed_gain * dgo_abs / self.lib[m]["coast_steps"]
        return float(np.clip(a0 + desired_v / slope, a_min, a_star))

    def new_target(self) -> None:
        """Start a new target: clear the move state but KEEP power warm (phase t)."""
        self.reacher.new_target()
        self.move = None
        self.captured = False
        self.move_age = 0
        self.vbar = 0.0
        self.retrig = 0

    def _idle(self) -> np.ndarray:
        self.move = None
        return self.REST.copy()

    # --- the control step ----------------------------------------------------
    def step(self, hand, target) -> np.ndarray:
        self.t += 1
        if target is None:
            return self._idle()

        dec = self.reacher.decide(hand, target)
        self.draw_theta = dec["theta"].copy()
        dgo = dec["dgo"]
        speed = float(np.linalg.norm(dec["vel"]))

        if dec["dist"] < self.tol:                 # reached: hold (feedback advances)
            self.captured = True
            return self._idle()
        if self.captured:                          # hysteresis: hold once reached
            if self.sticky_capture or dec["dist"] < self.capture_mult * self.tol:
                return self._idle()                # sticky: hold for the rest of this target
            self.captured = False                  # drifted out -> resume steering
        if self.retrig > 0:                        # re-trigger: keep AC dropped a few steps
            self.retrig -= 1
            return self.REST.copy()

        # Between moves: wait for the previous coast to settle before re-picking,
        # so we never re-drive a joint that is still coasting.
        if self.move is None:
            if speed > self.settle_vel:
                return self.REST.copy()
            need = np.abs(dgo) > self.switch_margin
            if not need.any():
                return self.REST.copy()            # both joints within tolerance band
            j = int(np.argmax(np.abs(dgo) * need))
            self.move = (j, 1.0 if dgo[j] > 0 else -1.0)
            self.move_age = 0
            self.vbar = 0.0

        j, s = self.move
        self.move_age += 1
        m = muscle_index(j, s)
        rec = self.lib[m]
        dgo_j = abs(float(dgo[j]))
        self.vbar = self.vbar_decay * self.vbar + (1.0 - self.vbar_decay) * abs(float(dec["vel"][j]))

        # el- / cold-start watchdog: no motion within the window -> re-trigger
        window = self.cold_window if self.t < self.cold_window else (rec["warm"] + self.warm_pad)
        if self.move_age > window and self.vbar < 5e-4:
            self.retrig = self.retrig_steps
            self.n_retrig += 1
            return self._idle()

        # live-coast lead (smoothed velocity): cut when the coast lands it on target
        live_coast = self.vbar * rec["coast_steps"]
        if dgo_j <= live_coast + self.lead_margin:
            return self._idle()                    # stop; coast lands it, then settle-wait

        cart_factor = float(np.clip(dec["dist"] / self.cart_slow_zone,
                                    self.cart_min_factor, 1.0))
        A = self._amplitude(m, dgo_j, cart_factor)
        u0 = 0.5 + A * np.cos(2.0 * np.pi * rec["f"] * self.t)
        self.last = {"joint": j, "muscle": rec["name"], "dgo_j": dgo_j,
                     "vbar": self.vbar, "live_coast": live_coast, "A": A}
        return np.array([u0, 0.5])


# --- standalone: one reach on the real cascade + stop-timing figure ----------
def _run_one(controller, plant, target, max_steps=900, tol=2.0):
    plant.reset()
    controller.reset()
    hands, cmds, active = [], [], []
    touch = None
    for k in range(max_steps):
        hand = np.asarray(plant.hand_pos, dtype=float)
        u = controller.step(hand, target)
        plant.command(u)
        hands.append(hand.copy())
        cmds.append(u.copy())
        active.append(controller.move[0] if controller.move else -1)
        d = np.linalg.norm(target - hand)
        if touch is None and d < tol:
            touch = k
        if touch is not None and k > touch + 8:
            recent = np.asarray(hands[-8:])
            if np.max(np.linalg.norm(np.diff(recent, axis=0), axis=1)) < 1e-3:
                break
    hands = np.asarray(hands)
    d = np.linalg.norm(target - hands, axis=1)
    tc = int(np.argmin(d))
    return {"hands": hands, "cmds": np.asarray(cmds), "active": np.asarray(active),
            "target": np.asarray(target), "min_dist": float(d[tc]), "t_closest": tc,
            "touch": touch, "steps": len(hands),
            "overshoot": float(np.max(d[tc:]) - d[tc]), "n_retrig": controller.n_retrig}


def _main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cascade_plant import CascadePlant

    print("=== reach_controller.py: one reach on the real cascade ===")
    lib = load_library()
    for k in range(4):
        r = lib[k]
        print(f"  {r['name']:>9s}: f={r['f']:.3f} A*={r['A_star']:.2f} "
              f"v={r['steady_v']:+.4f} coast={r['coast']:.2f} "
              f"coast_steps={r['coast_steps']:.0f} warm={r['warm']:.0f}")

    target = np.array([12.0, 26.0])  # mid-disk, needs shoulder + elbow
    ctrl = ReachController(tol=2.0)
    plant = CascadePlant(seed=0)
    res = _run_one(ctrl, plant, target)
    print(f"\ntarget {target} (r={np.hypot(*target):.1f}):")
    print(f"  touched={res['min_dist'] < 2.0}  min_dist={res['min_dist']:.2f}  "
          f"steps_to_touch={res['touch']}  total_steps={res['steps']}  "
          f"overshoot={res['overshoot']:.2f}  retriggers={res['n_retrig']}")

    # stop-timing figure: hand distance + per-joint angle vs its IK target
    hands = res["hands"]
    d = np.linalg.norm(target - hands, axis=1)
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(d, color="tab:blue"); ax[0].axhline(2.0, color="0.6", ls="--", label="tol")
    ax[0].set_ylabel("hand-target dist"); ax[0].legend(); ax[0].set_title(
        f"Cascade reach to {tuple(target)} (one joint at a time)")
    ax[1].plot(res["cmds"][:, 0], color="tab:orange", lw=0.8)
    ax[1].set_ylabel("u0 (DC+AC)"); ax[1].set_xlabel("step")
    ax[1].axhline(0.5, color="0.7", lw=0.8)
    fig.tight_layout()
    fig.savefig(HERE / "stop_timing.pdf"); fig.savefig(HERE / "stop_timing.png", dpi=110)
    plt.close(fig)
    print("saved stop_timing.pdf/png")


if __name__ == "__main__":
    _main()
