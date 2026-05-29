"""Interface-contract tests for the smoothness-penalised estimator.

A verbatim copy of ``test_interface.py`` retargeted at ``estimator_smooth`` (the
only change is the import below).  Verifies the same hard rules from
``week2.md`` §2 / §4.5:

* output shapes are exactly ``(T, LatentDim)`` and ``(T, InputDim)`` for several
  ``(T, p, LatentDim, InputDim)`` combinations;
* outputs contain no NaN / inf;
* ``T = 1000, p = 16`` runs in well under 30 seconds;
* the estimator never crashes on any listed simulator scenario (or on degenerate
  inputs), always returning finite arrays of the right shape.

``pytest`` is not required: running ``python test_interface_smooth.py`` executes
every check and prints a PASS/FAIL summary (exit code 1 on any failure).  The
same ``test_*`` functions are collected normally if ``pytest`` is available.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "week 1")))

import Simulator as sim          # noqa: E402
from estimator_smooth import estimate_latent_and_input  # noqa: E402

# (name, factory, true latent dim, obs dim) for every scenario in §1.
SCENARIOS = [
    ("default_neural_system", sim.default_neural_system, 4, 16),
    ("input_aligned_system", sim.input_aligned_system, 4, 16),
    ("input_blind_system", sim.input_blind_system, 4, 16),
    ("hidden_input_system", sim.hidden_input_system, 5, 10),
    ("slow_drift_system", sim.slow_drift_system, 4, 16),
    ("non_normal_system", sim.non_normal_system, 4, 8),
    ("ill_conditioned_system", sim.ill_conditioned_system, 4, 8),
]


def _simulate(factory, T, n, obs_dim, seed=0):
    """Simulate a trajectory driven by ``mixed_input`` and return ``y``."""
    system = factory(seed=seed, obs_dim=obs_dim)
    u = sim.mixed_input(T, system.input_dim, seed=seed)
    return system.simulate(T, U=u)["y"]


def test_output_shapes():
    """Shapes are exactly (T, LatentDim) / (T, InputDim) across combinations."""
    combos = [
        (200, 8, 3, 2),
        (500, 16, 4, 2),
        (750, 10, 5, 2),
        (1000, 16, 4, 1),
        (300, 12, 2, 3),
    ]
    rng = np.random.default_rng(0)
    for T, p, n, m in combos:
        y = rng.standard_normal((T, p))
        latent, inputs = estimate_latent_and_input(y, n, m)
        assert latent.shape == (T, n), f"latent {latent.shape} != {(T, n)}"
        assert inputs.shape == (T, m), f"inputs {inputs.shape} != {(T, m)}"


def test_no_nan_inf():
    """Outputs are finite on real simulator data."""
    for name, factory, n, obs_dim in SCENARIOS:
        y = _simulate(factory, 400, n, obs_dim)
        latent, inputs = estimate_latent_and_input(y, n, 2)
        assert np.all(np.isfinite(latent)), f"non-finite latent for {name}"
        assert np.all(np.isfinite(inputs)), f"non-finite inputs for {name}"


def test_runs_under_30s():
    """T = 1000, p = 16 completes comfortably within the 30 s budget."""
    y = _simulate(sim.default_neural_system, 1000, 4, 16)
    start = time.perf_counter()
    latent, inputs = estimate_latent_and_input(y, 4, 2)
    elapsed = time.perf_counter() - start
    assert latent.shape == (1000, 4) and inputs.shape == (1000, 2)
    assert elapsed < 30.0, f"took {elapsed:.2f}s (>= 30s)"


def test_no_crash_on_scenarios():
    """Estimator returns correct finite shapes on every listed scenario."""
    for name, factory, n, obs_dim in SCENARIOS:
        y = _simulate(factory, 500, n, obs_dim)
        latent, inputs = estimate_latent_and_input(y, n, 2)
        assert latent.shape == (500, n), f"{name}: latent {latent.shape}"
        assert inputs.shape == (500, 2), f"{name}: inputs {inputs.shape}"
        assert np.all(np.isfinite(latent)) and np.all(np.isfinite(inputs)), name


def test_no_crash_on_degenerate_inputs():
    """Degenerate / pathological inputs fall back gracefully, never crash."""
    cases = [
        ("tiny T", np.random.default_rng(0).standard_normal((3, 16)), 4, 2),
        ("p < n", np.random.default_rng(0).standard_normal((200, 2)), 4, 2),
        ("all zeros", np.zeros((100, 16)), 4, 2),
        ("single channel", np.random.default_rng(0).standard_normal((50, 1)), 4, 2),
        ("constant rows", np.ones((120, 8)), 4, 2),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # fallback warnings are expected here
        for name, y, n, m in cases:
            latent, inputs = estimate_latent_and_input(y, n, m)
            T = y.shape[0]
            assert latent.shape == (T, n), f"{name}: latent {latent.shape}"
            assert inputs.shape == (T, m), f"{name}: inputs {inputs.shape}"
            assert np.all(np.isfinite(latent)) and np.all(np.isfinite(inputs)), name


def _run_standalone() -> int:
    """Run all ``test_*`` functions, print a summary, return an exit code."""
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
