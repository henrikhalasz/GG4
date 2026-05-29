"""Tests for the smoothness-penalised estimator (``estimator_smooth.py``).

Standalone (no ``pytest`` needed; patterned after ``test_interface.py``): running
``python test_smooth.py`` executes every check and prints a PASS/FAIL summary
(exit code 1 on any failure).  The same ``test_*`` functions are collected
normally if ``pytest`` is available.

Coverage
--------
* **Interface contract** — output shapes ``(T, LatentDim)`` / ``(T, InputDim)``
  across combinations, no NaN / inf, graceful handling of degenerate inputs, and
  a ``T = 1000, p = 16`` run well under 30 s (mirrors ``test_interface.py``).
* **Synthetic ground truth** — on a known LGSSM (``n=2, m=1, p=4``) driven by a
  *smooth* (autocorrelated) input, the smoothness-penalised back-out recovers the
  input with lower MSE than the per-timestep pseudoinverse.
* **No regression** — on ``default_neural_system`` + ``mixed_input`` the
  reconstruction RMSE is within 5 % of ``estimator.py`` (the smoothing changes
  ``u_hat`` only, not the filtered state ``x_hat``).
* **Limiting case** — with ``lam = 0`` the smoothed solve reduces to the original
  pseudoinverse solution (agreement < 1e-6).  This is the key linear-algebra
  sanity check: if it fails, the joint solve is wrong.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "week 1")))

import Simulator as sim                # noqa: E402
import estimator                       # noqa: E402
import estimator_smooth                # noqa: E402
from estimator_smooth import estimate_latent_and_input, fit_and_filter  # noqa: E402

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _simulate(factory, T, obs_dim, seed=0):
    """Simulate a ``mixed_input``-driven trajectory and return ``y``."""
    system = factory(seed=seed, obs_dim=obs_dim)
    u = sim.mixed_input(T, system.input_dim, seed=seed)
    return system.simulate(T, U=u)["y"]


def _best_linear_map(src, tgt):
    """Least-squares affine map ``src -> tgt`` (handles the GL identifiability)."""
    X = np.hstack([src, np.ones((len(src), 1))])
    W, *_ = np.linalg.lstsq(X, tgt, rcond=None)
    return X @ W


def _aligned_mse(u_hat, u_true):
    """MSE of the best-linear-fit alignment of ``u_hat`` onto ``u_true``."""
    return float(np.mean((u_true - _best_linear_map(u_hat, u_true)) ** 2))


def _ar1_input(T, seed, phi=0.8):
    """A *smooth* (autocorrelated) broadband scalar input: a unit-variance AR(1).

    A single sinusoid is degenerate for blind identification — it is absorbed
    into the estimated transition ``A_hat`` (the autocorrelation confound of
    ``interim_report.md`` §3/§4), so neither estimator recovers it and the test
    would be uninformative.  An AR(1) process is temporally smooth (so the
    first-difference penalty is the correct prior) yet broadband (so it is not
    swallowed by a single eigenvalue of ``A_hat``), giving a clean test.
    """
    rng = np.random.default_rng(seed)
    u = np.zeros(T)
    for t in range(1, T):
        u[t] = phi * u[t - 1] + rng.standard_normal()
    u = (u - u.mean()) / u.std()
    return u[:, None]


def _make_synthetic_lgssm():
    """Known LGSSM (n=2, m=1, p=4): a slow stable oscillator with a scalar input."""
    radius, period = 0.9, 50.0
    theta = 2 * np.pi / period
    c, s = np.cos(theta), np.sin(theta)
    A = np.array([[radius * c, -radius * s], [radius * s, radius * c]])
    B = np.array([[1.0], [0.5]])
    C = np.random.default_rng(100).standard_normal((4, 2))
    Q = 1e-2 * np.eye(2)
    R = 1e-2 * np.eye(4)
    return sim.Simulator(A, B, C, Q, R, x0=np.array([1.0, 0.0]), seed=0)


# ---------------------------------------------------------------------------
# Interface contract (mirrors test_interface.py)
# ---------------------------------------------------------------------------
def test_output_shapes():
    """Shapes are exactly (T, LatentDim) / (T, InputDim) across combinations."""
    combos = [(200, 8, 3, 2), (500, 16, 4, 2), (750, 10, 5, 2),
              (1000, 16, 4, 1), (300, 12, 2, 3)]
    rng = np.random.default_rng(0)
    for T, p, n, m in combos:
        y = rng.standard_normal((T, p))
        latent, inputs = estimate_latent_and_input(y, n, m)
        assert latent.shape == (T, n), f"latent {latent.shape} != {(T, n)}"
        assert inputs.shape == (T, m), f"inputs {inputs.shape} != {(T, m)}"


def test_no_nan_inf_on_scenarios():
    """Outputs are finite, correctly shaped on every listed simulator scenario."""
    for name, factory, n, obs_dim in SCENARIOS:
        y = _simulate(factory, 500, obs_dim)
        latent, inputs = estimate_latent_and_input(y, n, 2)
        assert latent.shape == (500, n), f"{name}: latent {latent.shape}"
        assert inputs.shape == (500, 2), f"{name}: inputs {inputs.shape}"
        assert np.all(np.isfinite(latent)) and np.all(np.isfinite(inputs)), name


def test_runs_under_30s():
    """T = 1000, p = 16 completes comfortably within the 30 s budget."""
    y = _simulate(sim.default_neural_system, 1000, 16)
    start = time.perf_counter()
    latent, inputs = estimate_latent_and_input(y, 4, 2)
    elapsed = time.perf_counter() - start
    assert latent.shape == (1000, 4) and inputs.shape == (1000, 2)
    assert elapsed < 30.0, f"took {elapsed:.2f}s (>= 30s)"
    print(f"      (T=1000, p=16 ran in {elapsed:.2f}s)")


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


# ---------------------------------------------------------------------------
# Synthetic ground truth
# ---------------------------------------------------------------------------
def test_synthetic_smooth_beats_pinv():
    """Smoothed back-out beats the pseudoinverse on a smooth ground-truth input.

    Both estimators share the same identification and filtering, so this isolates
    the back-out step: the smoothness penalty should recover the (smooth) input
    with strictly lower MSE.
    """
    T = 800
    system = _make_synthetic_lgssm()
    u_true = _ar1_input(T, seed=0, phi=0.8)
    y = system.simulate(T, U=u_true)["y"]
    u_truth = system.simulate(T, U=u_true)["u"]

    smooth = fit_and_filter(y, 2, 1)["inputs"]
    pinv = estimator.fit_and_filter(y, 2, 1)["inputs"]  # identical x_hat, A, B

    mse_smooth = _aligned_mse(smooth, u_truth)
    mse_pinv = _aligned_mse(pinv, u_truth)
    print(f"      synthetic input MSE: pseudoinverse={mse_pinv:.5f}  "
          f"smoothed={mse_smooth:.5f}  (lam={estimator_smooth.LAST_LAMBDA:.4f})")
    assert mse_smooth < mse_pinv, (
        f"smoothed MSE {mse_smooth:.5f} not < pseudoinverse MSE {mse_pinv:.5f}")


# ---------------------------------------------------------------------------
# No regression
# ---------------------------------------------------------------------------
def test_no_regression_recon_vs_primary():
    """Reconstruction RMSE stays within 5 % of the primary estimator.

    The smoothing changes only how ``u_hat`` is computed, not how ``x_hat`` is
    filtered, so the basis-invariant reconstruction ``C_hat x_hat`` should be
    essentially unchanged.
    """
    T = 500
    system = sim.default_neural_system(seed=0, obs_dim=16)
    u = sim.mixed_input(T, system.input_dim, seed=0)
    y = system.simulate(T, U=u)["y"]

    def recon_rmse(model):
        y_hat = model["latent"] @ model["C"].T + model["y_mean"]
        return float(np.sqrt(np.mean((y - y_hat) ** 2)))

    rmse_base = recon_rmse(estimator.fit_and_filter(y, 4, 2))
    rmse_smooth = recon_rmse(fit_and_filter(y, 4, 2))
    rel = abs(rmse_smooth - rmse_base) / rmse_base
    print(f"      recon RMSE: primary={rmse_base:.4f}  smooth={rmse_smooth:.4f}  "
          f"(rel diff {rel:.2e})")
    assert rel < 0.05, f"reconstruction RMSE drifted {rel:.1%} (>= 5%)"


# ---------------------------------------------------------------------------
# Limiting case (the critical linear-algebra sanity check)
# ---------------------------------------------------------------------------
def test_limiting_case_reduces_to_pinv():
    """With lam = 0 the smoothed solve equals the pseudoinverse back-out.

    Decoupled (lam = 0) the joint solve is a per-timestep ridge with a 1e-8
    jitter; since the identified ``B_hat`` has orthonormal columns this matches
    ``pinv(B_hat)`` to within the jitter.  Agreement to 1e-6 confirms the normal
    equations / sparse assembly are correct.
    """
    T = 500
    system = sim.default_neural_system(seed=0, obs_dim=16)
    u = sim.mixed_input(T, system.input_dim, seed=0)
    y = system.simulate(T, U=u)["y"]

    A, B, C, Q, R, x_hat, y_mean = estimator_smooth._identify_and_filter(y, 4, 2)
    u_smoothed0 = estimator_smooth._smoothed_inputs(x_hat, A, B, 0.0)
    u_pinv = estimator._estimate_inputs(x_hat, A, B)

    max_abs = float(np.max(np.abs(u_smoothed0 - u_pinv)))
    print(f"      lam=0 vs pseudoinverse: max abs diff = {max_abs:.2e}")
    assert max_abs < 1e-6, f"lam=0 solution differs from pseudoinverse by {max_abs:.2e}"


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
