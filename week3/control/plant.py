"""Plant interface for closed-loop experiments.

Two plants with the same surface:

- ``SimulatorPlant`` wraps a white-box ``Simulator`` from ``week 1/Simulator.py``;
  it maintains its own internal state, clips ``u`` to ``[0,1]``, and exposes
  ``true_state()`` for white-box verification.
- ``BrainPlant`` wraps the black-box ``GG4.Brain``; ``true_state()`` returns ``None``.

Both clip control inputs to ``[0,1]`` at the plant boundary, matching the Brain
contract documented in ``week3.md``.
"""

from __future__ import annotations

import numpy as np


class SimulatorPlant:
    """White-box plant wrapping a ``Simulator`` instance.

    Maintains internal state ``x`` and an independent RNG so we can re-run the
    same plant deterministically across uncontrolled / controlled comparisons.
    """

    def __init__(self, simulator, seed: int | None = None):
        self.sim = simulator
        self.x = np.asarray(simulator.x0, dtype=float).copy()
        self.rng = np.random.default_rng(seed)

    @property
    def input_dim(self) -> int:
        return self.sim.input_dim

    @property
    def obs_dim(self) -> int:
        return self.sim.obs_dim

    def reset(self, seed: int | None = None, x0=None) -> None:
        self.x = np.asarray(self.sim.x0 if x0 is None else x0, dtype=float).copy()
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def measure(self) -> np.ndarray:
        v = self.rng.multivariate_normal(np.zeros(self.sim.obs_dim), self.sim.R)
        return self.sim.C @ self.x + v

    def next_state(self, u) -> None:
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        w = self.rng.multivariate_normal(np.zeros(self.sim.state_dim), self.sim.Q)
        self.x = self.sim.A @ self.x + self.sim.B @ u + w

    def true_state(self) -> np.ndarray:
        return self.x.copy()


class BrainPlant:
    """Black-box plant wrapping a ``GG4.Brain`` instance.

    The Brain owns its own RNG and time stamp. ``true_state()`` is unobservable.
    """

    OBS_DIM = 16  # known from Brain spec

    def __init__(self, brain):
        self.brain = brain

    @property
    def input_dim(self) -> int:
        return int(self.brain.input_dim)

    @property
    def obs_dim(self) -> int:
        return self.OBS_DIM

    def measure(self) -> np.ndarray:
        return np.asarray(self.brain.measure(), dtype=float)

    def next_state(self, u) -> None:
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        self.brain.next_state(u)

    def true_state(self):  # noqa: D401
        return None
