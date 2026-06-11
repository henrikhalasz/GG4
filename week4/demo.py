"""demo.py -- interactive reaching demo + headless record mode (Step 2A).

Interactive window: the reachable circle and the two-link arm, animated as the
controller reaches. **Click inside the circle** to set a target; the **Random
target** button drops a new reachable point; once a target is reached you can set
another. A readout shows the target, reached?, and steps-to-touch.

    python demo.py                      # interactive window (IdealActuator)
    python demo.py --record demo.mp4    # headless: scripted random reaches -> MP4
    python demo.py --record demo.mp4 --n 5 --seed 1
    python demo.py --mode resolved_rate # coordinated two-joint variant

The demo talks only to the plant surface (hand_pos / command / reset) and the
``ReachingPolicy``; the arm is drawn from the reacher's tracked joint angles. It
runs against the ``IdealActuator`` now and is swappable to ``CascadePlant`` in 2C
with no demo changes (``--plant cascade`` wires the real cascade; it will only
reach once the 2B primitives drive ``u``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation, FFMpegWriter

import kinematics as kin
from control_policy import ReachingPolicy

HERE = Path(__file__).resolve().parent
R_INNER, R_OUTER = 5.0, 57.0  # safe annulus for random / clamped targets


def make_plant(kind: str = "ideal", seed: int = 0):
    if kind == "ideal":
        from ideal_actuator import IdealActuator
        return IdealActuator()
    if kind == "cascade":
        from cascade_plant import CascadePlant
        return CascadePlant(seed=seed)
    raise ValueError(kind)


def random_target(rng) -> np.ndarray:
    """A reachable point uniformly in the safe annulus."""
    r = np.sqrt(rng.uniform(R_INNER ** 2, R_OUTER ** 2))
    a = rng.uniform(0, 2 * np.pi)
    return np.array([r * np.cos(a), r * np.sin(a)])


class DemoController:
    """Drives one closed-loop reach per target, on the common plant surface."""

    def __init__(self, plant, policy):
        self.plant = plant
        self.policy = policy
        self.plant.reset()
        self.policy.reset()
        self.obs: list = []
        self.target = None
        self.reached = False
        self.steps_to_touch = None
        self.frames = 0
        self.draw_theta = self.policy.reacher.theta_est.copy()

    def set_target(self, xy) -> None:
        xy = np.asarray(xy, dtype=float)
        r = float(np.linalg.norm(xy))
        if r > kin.REACH * 0.99:           # clamp a click outside the disk
            xy = xy * (kin.REACH * 0.99 / r)
        self.target = xy
        self.policy.set_targets(xy.reshape(1, 2))
        self.obs = []
        self.reached = False
        self.steps_to_touch = None
        self.frames = 0

    def step(self) -> None:
        """Advance one control step toward the current target (if any)."""
        if self.target is None or self.reached:
            return
        hand = np.asarray(self.plant.hand_pos, dtype=float)
        u = self.policy(np.array(self.obs), self.target)
        self.plant.command(u)
        self.obs.append(hand)
        self.frames += 1
        self.draw_theta = self.policy.reacher.theta_est.copy()
        if self.steps_to_touch is None and self.policy.steps_to_touch:
            self.steps_to_touch = self.policy.steps_to_touch[-1]
        if self.policy.done:
            self.reached = True

    @property
    def settled(self) -> bool:
        """True once the target has been touched and the arm has stopped moving
        (so the record mode waits for a genuine settle, not just a fly-through)."""
        if self.steps_to_touch is None or len(self.obs) < 8:
            return False
        rec = np.asarray(self.obs[-8:])
        return float(np.max(np.linalg.norm(np.diff(rec, axis=0), axis=1))) < 0.05


class ArmView:
    """Matplotlib artists for the disk, the arm, the target and the readout."""

    def __init__(self, ax, ctrl: DemoController, tol: float):
        self.ax = ax
        self.ctrl = ctrl
        self.tol = tol
        ax.add_patch(plt.Circle((0, 0), kin.REACH, fill=False, ls="--", ec="0.6"))
        ax.add_patch(plt.Circle((0, 0), R_OUTER, fill=False, ls=":", ec="0.8"))
        ax.add_patch(plt.Circle((0, 0), R_INNER, fill=False, ls=":", ec="0.8"))
        (self.arm_line,) = ax.plot([], [], "-o", lw=4, color="tab:blue", ms=7,
                                   mfc="tab:blue", zorder=5)
        (self.hand_dot,) = ax.plot([], [], "o", color="tab:blue", ms=10, zorder=6)
        (self.tgt_mark,) = ax.plot([], [], "x", color="tab:red", ms=14, mew=3, zorder=6)
        self.tol_circle = plt.Circle((0, 0), tol, fill=False, ec="tab:red",
                                     lw=1, visible=False)
        ax.add_patch(self.tol_circle)
        self.readout = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                               fontsize=10, family="monospace",
                               bbox=dict(fc="w", ec="0.7", alpha=0.85))
        ax.set_aspect("equal")
        ax.set_xlim(-65, 65)
        ax.set_ylim(-65, 65)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Week-4 reaching demo (click to set a target)")

    def update(self):
        th = self.ctrl.draw_theta
        elbow = kin.elbow_pos(th)
        hand = kin.forward(th)
        self.arm_line.set_data([0, elbow[0], hand[0]], [0, elbow[1], hand[1]])
        self.hand_dot.set_data([hand[0]], [hand[1]])
        if self.ctrl.target is not None:
            t = self.ctrl.target
            self.tgt_mark.set_data([t[0]], [t[1]])
            self.tol_circle.center = (t[0], t[1])
            self.tol_circle.set_visible(True)
            d = float(np.linalg.norm(t - hand))
            status = "REACHED" if self.ctrl.reached else "reaching..."
            stt = self.ctrl.steps_to_touch
            self.readout.set_text(
                f"target ({t[0]:+5.1f},{t[1]:+5.1f})\n"
                f"dist   {d:5.2f}  (tol {self.tol})\n"
                f"status {status}\n"
                f"steps-to-touch {stt if stt is not None else '--'}")
        else:
            self.tgt_mark.set_data([], [])
            self.tol_circle.set_visible(False)
            self.readout.set_text("click inside the circle\nor press 'Random target'")
        return (self.arm_line, self.hand_dot, self.tgt_mark, self.readout)


def run_interactive(plant, policy, tol: float, *, steps_per_frame: int = 3,
                    interval: int = 15, seed: int = 0):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(7.5, 8))
    fig.subplots_adjust(bottom=0.13)
    ctrl = DemoController(plant, policy)
    view = ArmView(ax, ctrl, tol)

    def on_click(event):
        if event.inaxes is ax and event.xdata is not None:
            ctrl.set_target([event.xdata, event.ydata])

    def on_random(_event):
        ctrl.set_target(random_target(rng))

    fig.canvas.mpl_connect("button_press_event", on_click)
    btn_ax = fig.add_axes([0.7, 0.02, 0.25, 0.06])
    button = Button(btn_ax, "Random target")
    button.on_clicked(on_random)

    def update(_frame):
        # Advance several cascade steps per rendered frame: identical simulation,
        # fewer redraws -> faster to watch. --speed 1 is the original smooth pacing.
        for _ in range(max(1, steps_per_frame)):
            ctrl.step()
        return view.update()

    anim = FuncAnimation(fig, update, interval=interval, blit=False,
                         cache_frame_data=False)
    fig._demo_anim = anim  # keep a reference alive
    plt.show()


def run_record(plant, policy, tol: float, out_path, n_targets: int = 4,
               seed: int = 0, fps: int = 30, max_frames_per: int = 240,
               steps_per_frame: int = 2):
    """Headless: scripted random reaches saved to MP4 (FFMpegWriter)."""
    rng = np.random.default_rng(seed)
    queue = [random_target(rng) for _ in range(n_targets)]

    fig, ax = plt.subplots(figsize=(7.5, 8))
    ctrl = DemoController(plant, policy)
    view = ArmView(ax, ctrl, tol)
    ax.set_title(f"Week-4 reaching demo ({type(plant).__name__})")
    ctrl.set_target(queue.pop(0))

    state = {"hold": 0}

    def frames():
        done_all = False
        f = 0
        while not done_all:
            yield f
            f += 1
            if ctrl.reached or ctrl.settled:  # reached within tol, or arm stopped near it
                state["hold"] += 1
                if state["hold"] > 15:        # dwell on the settled pose, then next target
                    state["hold"] = 0
                    if queue:
                        ctrl.set_target(queue.pop(0))
                    else:
                        done_all = True
            elif ctrl.frames > max_frames_per:  # safety: give up on a stuck target
                if queue:
                    ctrl.set_target(queue.pop(0))
                else:
                    done_all = True

    def update(_frame):
        for _ in range(max(1, steps_per_frame)):
            ctrl.step()
        return view.update()

    writer = FFMpegWriter(fps=fps, bitrate=1800)
    out_path = Path(out_path)
    anim = FuncAnimation(fig, update, frames=frames, blit=False,
                         cache_frame_data=False, save_count=n_targets * max_frames_per + 100)
    anim.save(str(out_path), writer=writer, dpi=110)
    plt.close(fig)
    return out_path


def _build(args):
    cascade = (args.plant == "cascade")
    plant = make_plant(args.plant, seed=args.seed)
    # For the demo we trade a little reach speed for a clean SETTLE: a slow final
    # Cartesian approach (small coast) plus a firm capture band, so the arm stops on
    # the target rather than drifting past it. (The 2C harness metrics use the
    # controller defaults and are unaffected -- this only styles the demo.)
    extra = ({"sticky_capture": True, "speed_gain": 0.5, "cart_slow_zone": 22.0,
              "cart_min_factor": 0.10, "switch_margin": 0.04} if cascade else {})
    policy = ReachingPolicy(mode=args.mode, tol=args.tol, cascade=cascade, **extra)
    return plant, policy, cascade


def main():
    ap = argparse.ArgumentParser(description="Week-4 reaching demo")
    ap.add_argument("--record", metavar="OUT.mp4", default=None,
                    help="headless record mode -> save an MP4")
    ap.add_argument("--n", type=int, default=4, help="number of scripted targets (record)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["one_joint", "resolved_rate"], default="one_joint")
    ap.add_argument("--plant", choices=["ideal", "cascade"], default="ideal")
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--speed", type=int, default=None,
                    help="cascade steps advanced per drawn frame (higher = faster to "
                         "watch; the simulation is identical). Default 3 interactive / 2 record.")
    ap.add_argument("--interval", type=int, default=15,
                    help="ms between frames (interactive only; lower = faster)")
    args = ap.parse_args()

    plant, policy, cascade = _build(args)
    if args.record:
        out = run_record(plant, policy, args.tol, args.record, n_targets=args.n,
                         seed=args.seed, max_frames_per=2000 if cascade else 240,
                         steps_per_frame=args.speed if args.speed else 2)
        print(f"saved demo clip -> {out}")
    else:
        run_interactive(plant, policy, args.tol, seed=args.seed,
                        steps_per_frame=args.speed if args.speed else 3,
                        interval=args.interval)


if __name__ == "__main__":
    main()
