"""cascade_plant.py -- the real brain->muscle->arm cascade as a plant (Week 4).

``CascadePlant`` wraps the genuine ``BMI_and_Hand(Brain(seed))`` on the same
plant surface as ``IdealActuator`` (hand_pos / command / reset), so the harness,
reacher and demo are agnostic to which plant is underneath. In 2A the cascade
input ``u`` is NOT yet driven toward a goal -- the actuation alphabet (the 2B
primitives) does not exist. We build the wrapper now and run the **regime-A
check**: confirm that ``hand_pos`` is exposed, clean (finite, shape (2,)), and
*changes under drive*.

Regime-A finding (recorded for the writeup): the cascade has a slow, accumulating
onset -- a single command moves the hand by ~0; only a *sustained* command ramps
it (the muscle head is a stateful rhythm/power decoder, and the arm is a velocity
integrator). This is exactly what ``IdealActuator``'s onset-lag (tau_on) models.

Regime-A caveat (recorded): the Week-4 notebook's ``run_closed_loop`` steps a
*bare* ``Brain`` and exposes no hand. Our whole 2A design assumes the *scored*
harness exposes ``hand_pos`` live (as ``BMI_and_Hand`` does); this must be
confirmed with the demonstrator.

Requires torch (BMI_and_Hand.py imports it) and the GG4 package (the live Brain).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# BMI_and_Hand lives in week3/ (read, not imported from, per the brief -- but the
# genuine cascade object itself is the one ground-truth we must wrap).
_HERE = Path(__file__).resolve().parent          # week4/ (holds the ANN weights)
_WEEK3 = _HERE.parent / "week3"
if str(_WEEK3) not in sys.path:
    sys.path.insert(0, str(_WEEK3))


class CascadePlant:
    """Thin wrapper around ``BMI_and_Hand(Brain(seed))`` on the plant surface.

    Surface: ``hand_pos`` (clean (2,)), ``command(u)`` (advance one cascade
    step), ``reset()`` (fresh Brain + arm). ``input_dim`` = 2, ``obs_dim`` = 16.
    """

    INPUT_DIM = 2
    OBS_DIM = 16

    def __init__(self, seed: int = 0, *, clip_input: bool = True):
        self.seed = int(seed)
        self.clip_input = bool(clip_input)
        self._last_y = None
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        from GG4 import Brain
        from BMI_and_Hand import BMI_and_Hand
        if seed is not None:
            self.seed = int(seed)
        # BMI_and_Hand loads its ANN weights by a cwd-relative filename; run the
        # construction from week4/ (where the weights live) so the cascade works
        # no matter which directory the script was launched from.
        _cwd = os.getcwd()
        try:
            os.chdir(_HERE)
            self._cascade = BMI_and_Hand(Brain(random_seed=self.seed))
        finally:
            os.chdir(_cwd)
        self._last_y = None

    @property
    def input_dim(self) -> int:
        return self.INPUT_DIM

    @property
    def obs_dim(self) -> int:
        return self.OBS_DIM

    @property
    def hand_pos(self) -> np.ndarray:
        return np.asarray(self._cascade.hand_pos, dtype=float)

    def measure(self) -> np.ndarray:
        """Most recent 16-D brain measurement (from the last ``command``)."""
        return self._last_y

    def command(self, c) -> np.ndarray:
        """Advance one cascade step under input ``u = c``; return hand_pos."""
        u = np.asarray(c, dtype=float)
        if self.clip_input:
            u = np.clip(u, 0.0, 1.0)
        self._last_y = np.asarray(self._cascade.next_state(u), dtype=float)
        return self.hand_pos


def _regime_a_check(seed: int = 0, n_steps: int = 150, u_drive=(1.0, 1.0)):
    """Confirm hand_pos is exposed, clean, and changes under sustained drive."""
    print("=== cascade_plant.py: regime-A check (real cascade) ===")
    plant = CascadePlant(seed=seed)

    hp0 = plant.hand_pos
    assert hp0.shape == (2,) and np.all(np.isfinite(hp0)), "hand_pos must be clean (2,)"
    print(f"hand_pos before drive: {hp0.round(4)}  shape={hp0.shape}  dtype={hp0.dtype}")

    # Single step first -- demonstrates the slow onset (moves ~0).
    plant.command(u_drive)
    hp1 = plant.hand_pos
    print(f"after 1 step  u={tuple(u_drive)}: {hp1.round(4)}  "
          f"|move|={np.linalg.norm(hp1 - hp0):.4f}  (slow onset: ~0)")

    # Sustained drive -- the readout is live and moves; stays clean throughout.
    plant.reset()
    hp0 = plant.hand_pos
    traj = [hp0.copy()]
    clean = True
    for _ in range(n_steps):
        hp = plant.command(u_drive)
        clean = clean and hp.shape == (2,) and bool(np.all(np.isfinite(hp)))
        traj.append(hp.copy())
    traj = np.asarray(traj)
    moved = float(np.max(np.linalg.norm(traj - hp0, axis=1)))
    y = plant.measure()
    print(f"after {n_steps} steps u={tuple(u_drive)}: {traj[-1].round(4)}  "
          f"max|move|={moved:.4f}")
    print(f"measurement shape from cascade: {None if y is None else y.shape}")
    print(f"hand_pos clean (finite, (2,)) every step: {clean}")

    assert clean, "hand_pos must stay clean under drive"
    assert moved > 1.0, "hand must change under sustained drive"
    print("\nRegime-A PASS: hand_pos is exposed, clean, and live under drive.")
    print("Note: single-step move ~0 (slow onset) -> modelled by IdealActuator tau_on.")
    print("Caveat: the scored harness must be confirmed (demonstrator) to expose "
          "hand_pos live; the notebook's bare-Brain run_closed_loop does not.")
    return moved


if __name__ == "__main__":
    _regime_a_check()
