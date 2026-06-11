"""arm_controller.py -- adapter that exposes our 2C ``reach_controller.ReachController``
in the interface ``evaluate_controller.py`` expects, so the eval suite can run
against the controller we built without changing either file.

The eval drives one reach per (seed, target): it pre-latches power with ``WARMUP``
steps of ``[DC, 0.5]``, builds ``ReachController(target, **kw)``, then calls
``command(hand) -> u0`` each step until ``.done``, reducing ``.log`` and ``.deadband``
to metrics. We hold the target and delegate to ``reach_controller.ReachController``
(default 2C params; ``**kwargs`` from ``--ctrl-kw`` pass straight through for A/B).

NOTE on the metric: ``landed`` in the eval means the FINAL pose settled within the
deadband (tol = 2.0) -- stricter than the 2C report's "touch-success" (closest
approach < tol). So the landing rate here reads lower than the 93% touch-success;
that stricter bar (settle, not just fly through) is the point of this suite.
"""

from __future__ import annotations

import numpy as np

import reach_controller as _rc
from reach_controller import muscle_index

# Module constants the eval imports.
DC = float(_rc.ReachController.REST[0])   # 0.5 -- input-0 DC level (power held on)
WARMUP = 100                              # >= the ~94-step cold power onset (paid once)
ENGAGE = 50                               # ~ warm switch; segment-length threshold in metrics

_SETTLE_STEPS = 8                         # within deadband this many steps -> done


class ReachController:
    """One-reach wrapper around ``reach_controller.ReachController``."""

    def __init__(self, target, **kwargs):
        self.target = np.asarray(target, dtype=float).reshape(2)
        self._c = _rc.ReachController(**kwargs)   # default 2C controller
        self.deadband = float(self._c.tol)
        self.log: list = []
        self.done = False
        self._still = 0

    def command(self, hand):
        """Advance one step toward the fixed target; return the input-0 command u0."""
        hand = np.asarray(hand, dtype=float).reshape(2)
        u = np.asarray(self._c.step(hand, self.target), dtype=float)
        err = float(np.linalg.norm(self.target - hand))
        mv = self._c.move                         # (joint, sign) while driving, else None
        band = muscle_index(*mv) if mv is not None else None
        self.log.append({"err": err, "band": band})

        # done = settled within the deadband (held there, not just passing through)
        if err <= self.deadband:
            self._still += 1
            if self._still >= _SETTLE_STEPS:
                self.done = True
        else:
            self._still = 0
        return float(u[0])
