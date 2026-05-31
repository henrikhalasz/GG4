"""Probe generators for identification.

Uniform[0,1] independent per channel is broadband, non-negative, persistently
exciting — the default the spec recommends.
"""

from __future__ import annotations

import numpy as np


def uniform_probe(T: int, input_dim: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(T, input_dim))
