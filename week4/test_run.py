"""test_run.py -- plant factory for evaluate_controller.py.

Provides ``make_plant(use_mock, seed, el_weak)`` returning a plant exposing the
two members the eval uses: ``hand_pos`` (property, (2,)) and
``next_state([u0, u1])``. The real plant wraps our live ``CascadePlant`` (which
itself chdirs into week4/ so the brain ANN weights load from any working
directory). A mock plant is not wired in this repo -- run the eval with ``--real``.

Importing this module (without ``--live``) is also where evaluate_controller pins
matplotlib to the headless Agg backend.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")          # headless: the eval only saves figures

import numpy as np

from cascade_plant import CascadePlant


def make_plant(use_mock, seed, el_weak=0.6):
    """Real cascade plant for the eval. ``el_weak`` is a mock-only knob (ignored)."""
    if use_mock:
        raise NotImplementedError(
            "No mock plant in this repo -- run evaluate_controller.py with --real "
            "(the live cascade).")
    return _RealPlant(int(seed))


class _RealPlant:
    """Thin adapter: next_state([u0,u1]) -> CascadePlant.command; hand_pos passthrough."""

    def __init__(self, seed: int):
        self._c = CascadePlant(seed=int(seed))

    @property
    def hand_pos(self) -> np.ndarray:
        return np.asarray(self._c.hand_pos, dtype=float)

    def next_state(self, u):
        self._c.command(np.asarray(u, dtype=float))
