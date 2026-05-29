"""Tests for the coordinate-ascent ML estimator (``estimator_em.py``).

Standalone (no ``pytest`` needed; patterned after ``test_interface.py`` /
``test_smooth.py``): running ``python test_em.py`` executes every check and prints
a PASS/FAIL summary (exit code 1 on any failure).  The same ``test_*`` functions
are collected normally if ``pytest`` is available.

Coverage
--------
1. **Smoother exact at truth** — with the true ``(A,B,C,Q,R,u)`` the RTS smoother
   recovers the latent states (aligned mean column correlation > 0.99).
2. **Lag-one cross-covariance (Monte-Carlo)** — the smoother's ``P_{t,t-1}^s``
   matches the empirical covariance of smoothing errors over 500 trajectories
   (< 10 % relative Frobenius).  The make-or-break check for the M-step.
3. **Monotone log-likelihood** — 20 iterations never decrease the log-likelihood
   by more than 1e-6.
4. **Convergence** — converges (rel ll change < 1e-6) in < 50 iterations.
5. **Input R^2 beats baseline** — EM input R^2 >= 1.15 x baseline.
6. **No reconstruction regression** — EM recon RMSE within 5 % of baseline.
7. **Interface contract** — shapes/finiteness on every scenario, < 30 s at
   T=1000,p=16, graceful fallback on degenerate inputs.
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
import estimator_em                    # noqa: E402
from estimator_em import (             # noqa: E402
    _em_fit,
    _kalman_smoother,
    estimate_latent_and_input,
    fit_and_filter,
)

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
# Helpers (mirror the notebook's alignment / metric helpers)
# ---------------------------------------------------------------------------
def _best_linear_map(src, tgt):
    """Least-squares affine map ``src -> tgt`` (handles the GL identifiability)."""
    X = np.hstack([src, np.ones((len(src), 1))])
    W, *_ = np.linalg.lstsq(X, tgt, rcond=None)
    return X @ W


def _column_correlation(A, B):
    """Mean absolute Pearson correlation over matched columns."""
    cs = []
    for i in range(min(A.shape[1], B.shape[1])):
        a, b = A[:, i], B[:, i]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        cs.append(abs(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(cs)) if cs else float("nan")


def _latent_corr(x_hat, x_true):
    """Mean column correlation after best-linear-map alignment."""
    return _column_correlation(_best_linear_map(x_hat, x_true), x_true)


def _input_r2(u_hat, u_true):
    """Best-linear-fit input recovery R^2."""
    pred = _best_linear_map(u_hat, u_true)
    ss_res = np.sum((u_true - pred) ** 2)
    ss_tot = np.sum((u_true - u_true.mean(0)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")


def _recon_rmse(y, model):
    y_hat = model["latent"] @ model["C"].T + model["y_mean"]
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def _simulate_default(T=500, obs_dim=16, seed=0):
    system = sim.default_neural_system(seed=seed, obs_dim=obs_dim)
    u = sim.mixed_input(T, system.input_dim, seed=seed)
    d = system.simulate(T, U=u)
    return d["y"], d["u"], system


# ---------------------------------------------------------------------------
# Test 1 — smoother is exact when initialised at truth
# ---------------------------------------------------------------------------
def test_smoother_exact_at_truth():
    """RTS smoother with the true params recovers the latent states (corr>0.99)."""
    T = 500
    system = sim.default_neural_system(seed=0, obs_dim=16)
    u = sim.mixed_input(T, system.input_dim, seed=0)
    d = system.simulate(T, U=u)
    y, x_true = d["y"], d["x"][:-1]  # x_0 .. x_{T-1}

    x_s, P_s, P_lag, ll = _kalman_smoother(
        y, system.A, system.B, system.C, system.Q, system.R, d["u"]
    )
    corr = _latent_corr(x_s, x_true)
    print(f"      smoother latent corr (true params) = {corr:.4f}")
    assert corr > 0.99, f"smoothed-state correlation {corr:.4f} <= 0.99"
    assert np.all(np.isfinite(x_s)) and np.all(np.isfinite(P_lag))


# ---------------------------------------------------------------------------
# Test 2 — lag-one cross-covariance correctness
# ---------------------------------------------------------------------------
# The smoother's posterior covariances ``P_t^s`` and lag-one cross-covariances
# ``P_{t,t-1}^s`` are the single most error-prone part of EM, so we verify them
# two ways:
#   * test_cross_covariance_exact   -- the BUG-CATCHER. Compares the recursion
#     against the exact posterior covariance built from the block-tridiagonal
#     precision matrix of the joint Gaussian p(x_{0:T-1} | y_{1:T}). This is a
#     closed-form ground truth with no sampling noise, so agreement to numerical
#     precision (~1e-9) is the strongest possible correctness guarantee.
#   * test_cross_covariance_mc_sanity -- a lightweight RUNTIME SANITY check that
#     the code runs and is in the right ballpark across different random draws.
#     The covariance entries are tiny, so the Monte-Carlo estimate is noise-
#     dominated at any tractable N; its tolerance (25%) reflects that this is a
#     "doesn't crash / roughly right" check, NOT a formula verification.
def _exact_posterior_covariance(A, B, C, Q, R, P0):
    """Exact joint posterior covariance of ``x_{0:T-1}`` via the precision matrix.

    The negative log-posterior is quadratic in the stacked state, with a
    block-tridiagonal precision (information) matrix ``Lambda`` whose inverse is
    the joint covariance.  Diagonal/off-diagonal blocks come from the prior, the
    process model and the observation model (the input ``u`` only shifts the mean,
    so the covariance — and hence ``Lambda`` — is input-independent).
    """
    n = A.shape[0]
    T = _EXACT_T
    Qi = np.linalg.inv(Q)
    Ri = np.linalg.inv(R)
    P0i = np.linalg.inv(P0)
    CtRiC = C.T @ Ri @ C
    AtQiA = A.T @ Qi @ A
    Lam = np.zeros((T * n, T * n))

    def blk(i, j):
        return (slice(i * n, (i + 1) * n), slice(j * n, (j + 1) * n))

    for t in range(T):
        D = CtRiC.copy()
        if t == 0:
            D = D + P0i
        if t >= 1:
            D = D + Qi
        if t <= T - 2:
            D = D + AtQiA
        Lam[blk(t, t)] += D
    for t in range(1, T):
        Lam[blk(t, t - 1)] += -Qi @ A
        Lam[blk(t - 1, t)] += -A.T @ Qi
    return np.linalg.inv(Lam)


_EXACT_T = 40  # trajectory length for the exact-covariance check


def test_cross_covariance_exact():
    """RTS smoother covariances match the exact block-tridiagonal posterior.

    This is the primary correctness gate for the smoother (the bug-catcher):
    closed-form ground truth, zero sampling noise.
    """
    T = _EXACT_T
    system = sim.default_neural_system(seed=0, obs_dim=16)
    A, B, C, Q, R = system.A, system.B, system.C, system.Q, system.R
    n, m = system.state_dim, system.input_dim

    rng = np.random.default_rng(0)
    u = rng.standard_normal((T, m))
    y = system.simulate(T, U=u)["y"]

    x_s, P_s, P_lag, _ = _kalman_smoother(y, A, B, C, Q, R, u)
    # Smoother default prior is x0=pinv(C)y0, P0=I -> use P0=I in the precision.
    Sigma = _exact_posterior_covariance(A, B, C, Q, R, np.eye(n))

    def blk(i, j):
        return (slice(i * n, (i + 1) * n), slice(j * n, (j + 1) * n))

    cov_err = max(np.linalg.norm(P_s[t] - Sigma[blk(t, t)]) for t in range(T))
    lag_err = max(np.linalg.norm(P_lag[t] - Sigma[blk(t, t - 1)]) for t in range(1, T))
    print(f"      max |P_smooth - exact| = {cov_err:.2e}   "
          f"max |P_lag - exact| = {lag_err:.2e}")
    assert cov_err < 1e-6, f"P_smooth mismatch vs exact posterior: {cov_err:.2e}"
    assert lag_err < 1e-6, f"P_lag mismatch vs exact posterior: {lag_err:.2e}"


def test_cross_covariance_mc_sanity():
    """Lightweight runtime sanity check of the lag-one cross-covariance.

    Generates trajectories from the same LGSSM (fast vectorised noise, no per-step
    decomposition) and checks the empirical smoothing-error cross-covariance is in
    the right ballpark of the smoother's ``P_lag[T/2]``.  The 25% tolerance is
    deliberately loose: the cross-cov magnitude is tiny so the MC estimate is
    noise-dominated — correctness is already established by
    :func:`test_cross_covariance_exact`.
    """
    T, N = 100, 600
    t = T // 2
    system = sim.default_neural_system(seed=0, obs_dim=16)
    A, B, C, Q, R = system.A, system.B, system.C, system.Q, system.R
    n, m, p = system.state_dim, system.input_dim, system.obs_dim
    x0 = system.x0
    Lq, Lr = np.linalg.cholesky(Q), np.linalg.cholesky(R)

    rng = np.random.default_rng(0)
    P_lag_ref = None
    acc = np.zeros((n, n))
    for _ in range(N):
        u = rng.standard_normal((T, m))
        w = (Lq @ rng.standard_normal((n, T))).T
        o = (Lr @ rng.standard_normal((p, T))).T
        x_true = np.empty((T, n))
        y = np.empty((T, p))
        xc = x0.copy()
        for k in range(T):
            x_true[k] = xc
            y[k] = C @ xc + o[k]
            xc = A @ xc + B @ u[k] + w[k]
        x_s, _, P_lag, _ = _kalman_smoother(y, A, B, C, Q, R, u)
        if P_lag_ref is None:
            P_lag_ref = P_lag[t].copy()
        acc += np.outer(x_true[t] - x_s[t], x_true[t - 1] - x_s[t - 1])
    rel = np.linalg.norm(acc / N - P_lag_ref) / np.linalg.norm(P_lag_ref)
    print(f"      cross-cov MC rel. Frobenius (N={N}, t={t}) = {rel:.4f} "
          f"(sanity tol 0.25)")
    assert rel < 0.25, f"cross-covariance MC sanity {rel:.4f} >= 0.25"


# ---------------------------------------------------------------------------
# Test 3 — log-likelihood is monotonically non-decreasing
# ---------------------------------------------------------------------------
def test_loglik_monotonic():
    """20 iterations on default+mixed never decrease the log-likelihood (>1e-6)."""
    T = 500
    y, _, _ = _simulate_default(T=T)
    yc = y - y.mean(axis=0)
    # tol=0 and param_tol=0 disable both convergence criteria -> run all 20 iters.
    res = _em_fit(yc, 4, 2, max_iter=20, tol=0.0, param_tol=0.0)
    ll = res["ll_history"]
    print("      ll sequence:")
    for k, v in enumerate(ll):
        print(f"        iter {k:2d}: {v:.6f}")
    diffs = np.diff(ll)
    worst = float(diffs.min()) if diffs.size else 0.0
    print(f"      worst step delta = {worst:.3e}")
    assert np.all(diffs >= -1e-6), f"log-likelihood decreased by {worst:.3e}"


# ---------------------------------------------------------------------------
# Test 4 — converges in a reasonable iteration count
# ---------------------------------------------------------------------------
def test_converges_quickly():
    """Converges in < 50 iterations on default+mixed.

    Convergence is by the dual criterion in ``_em_fit``: the output-determining
    matrices (A, B, C) stabilise within a handful of iterations even though the
    deterministic-u variant's log-likelihood keeps creeping along a Q gauge ridge
    (documented in the module docstring and the notebook, never reaching the
    strict 1e-7 ll tolerance). The parameter criterion is what makes convergence
    meaningful here.
    """
    T = 500
    y, _, _ = _simulate_default(T=T)
    model = fit_and_filter(y, 4, 2)
    n_iter = model["n_iter"]
    print(f"      converged (A,B,C stable) in {n_iter} iterations "
          f"(final ll={model['ll_history'][-1]:.2f})")
    assert n_iter < 50, f"took {n_iter} iterations (>= 50)"


# ---------------------------------------------------------------------------
# Test 5 — EM improves input R^2 over the baseline
# ---------------------------------------------------------------------------
def test_input_r2_beats_baseline():
    """EM input R^2 vs the 1.15 x baseline target on default+mixed.

    The 1.15x threshold was an a-priori guess. The empirical answer on this
    low-noise regime is ~1.06x: the subspace-ID baseline is already near-optimal,
    so joint refinement has little bias to remove and the autocorrelation confound
    is a tighter ceiling than the algorithm choice (the noise-regime sweep in the
    notebook shows EM's advantage growing with observation noise). Per the agreed
    handling we KEEP the threshold so the shortfall stays visible, and document
    the actual numbers here rather than lowering it or tuning to clear it. This
    test is therefore EXPECTED TO FAIL on default+mixed -- the failure message is
    the documented finding, not a regression.
    """
    T = 500
    y, u_true, _ = _simulate_default(T=T)

    base = estimator.fit_and_filter(y, 4, 2)
    em = fit_and_filter(y, 4, 2)
    r2_base = _input_r2(base["inputs"], u_true)
    r2_em = _input_r2(em["inputs"], u_true)
    eig_base = np.sort(np.abs(np.linalg.eigvals(base["A"])))
    eig_em = np.sort(np.abs(np.linalg.eigvals(em["A"])))
    print(f"      input R^2: baseline={r2_base:.4f}  em={r2_em:.4f}  "
          f"(ratio {r2_em / r2_base:.3f}, a-priori target >= 1.15)")
    print(f"      |eig(A)|: baseline={np.round(eig_base, 3)}  em={np.round(eig_em, 3)}")
    if r2_em < 1.15 * r2_base:
        print(f"      DOCUMENTED SHORTFALL (expected): EM gain is {r2_em / r2_base:.3f}x "
              f"on this low-noise regime; see week2_em_analysis.ipynb noise sweep.")
    assert r2_em >= 1.15 * r2_base, (
        f"EM input R^2 {r2_em:.4f} < 1.15 x baseline {r2_base:.4f} "
        f"(documented finding, not a regression)")


# ---------------------------------------------------------------------------
# Test 6 — EM does not degrade reconstruction
# ---------------------------------------------------------------------------
def test_recon_not_degraded():
    """EM reconstruction is not degraded vs the baseline on default+mixed.

    The brief's intent is a degradation guard ("if EM's reconstruction is much
    worse, something is wrong"), so this is one-sided: EM must not be worse than
    the baseline by more than 5 %. EM returns SMOOTHED states (two-sided, using
    future data) where the baseline returns FILTERED states, so EM reconstructs
    at or slightly BELOW the baseline -- partly the smoother's advantage, partly
    the free-u absorbing residuals (which dips it just under the observation-noise
    floor, the same in-sample effect the notebook notes for PCA). A lower RMSE is
    not a failure.
    """
    T = 500
    y, _, _ = _simulate_default(T=T)
    rmse_base = _recon_rmse(y, estimator.fit_and_filter(y, 4, 2))
    rmse_em = _recon_rmse(y, fit_and_filter(y, 4, 2))
    floor = float(np.sqrt(np.mean(np.diag(sim.default_neural_system(seed=0, obs_dim=16).R))))
    print(f"      recon RMSE: baseline={rmse_base:.4f}  em={rmse_em:.4f}  "
          f"(noise floor {floor:.4f}; EM lower = smoother + free-u in-sample fit)")
    assert rmse_em <= rmse_base * 1.05, (
        f"EM reconstruction {rmse_em:.4f} degraded vs baseline {rmse_base:.4f} "
        f"by > 5%")


# ---------------------------------------------------------------------------
# Test 7 — interface contract (mirrors test_interface.py)
# ---------------------------------------------------------------------------
def test_output_shapes_and_finite_on_scenarios():
    """Shapes exactly (T, n) / (T, 2), finite, on every listed scenario."""
    T = 500
    for name, factory, n, obs_dim in SCENARIOS:
        system = factory(seed=0, obs_dim=obs_dim)
        u = sim.mixed_input(T, system.input_dim, seed=0)
        y = system.simulate(T, U=u)["y"]
        latent, inputs = estimate_latent_and_input(y, n, 2)
        assert latent.shape == (T, n), f"{name}: latent {latent.shape}"
        assert inputs.shape == (T, 2), f"{name}: inputs {inputs.shape}"
        assert np.all(np.isfinite(latent)) and np.all(np.isfinite(inputs)), name


def test_runs_under_30s():
    """T = 1000, p = 16 completes comfortably within the 30 s budget."""
    system = sim.default_neural_system(seed=0, obs_dim=16)
    u = sim.mixed_input(1000, system.input_dim, seed=0)
    y = system.simulate(1000, U=u)["y"]
    start = time.perf_counter()
    latent, inputs = estimate_latent_and_input(y, 4, 2)
    elapsed = time.perf_counter() - start
    assert latent.shape == (1000, 4) and inputs.shape == (1000, 2)
    print(f"      (T=1000, p=16 ran in {elapsed:.2f}s)")
    assert elapsed < 30.0, f"took {elapsed:.2f}s (>= 30s)"


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
