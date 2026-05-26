# Week 2 — Estimation: Build Specification

This document is a complete brief for an autonomous coding agent (Claude Code) to
implement the Week 2 deliverable of the GG4 project. It contains:

1. The system model and the scenarios we must handle.
2. The fixed interface contract the demonstrator will test against.
3. The theory behind the chosen estimation approach.
4. The exact deliverables (code files, figures, report).
5. A prompt to give Claude Code at the end.

The agent should treat this file as the single source of truth for the task.

---

## 1. Project context

We are working with a **partially-observed linear Gaussian dynamical system**:

```
x_{t+1} = A x_t + B u_t + w_t,    w_t ~ N(0, Q)
y_t     = C x_t + o_t,            o_t ~ N(0, R)
```

- `x_t ∈ R^n` is the hidden latent state.
- `u_t ∈ R^m` is the input (stimulation).
- `y_t ∈ R^p` is the noisy observation (neural-style measurement).
- `A, B, C, Q, R` and the true `u_t` are **unknown** to the estimator at test time.

The Week 1 group code (in `Simulator.py`) provides a `Simulator` class and a set of
factory functions that produce concrete scenarios. We must build an estimator that
works well across these scenarios without scenario-specific tuning.

Typical scenario parameters (from `Simulator.py`):

- Latent dim `n` ∈ {4, 5}
- Input dim `m` = 2
- Observation dim `p` ∈ {8, 10, 16}
- `Q ≈ 1e-3 · I`, `R ≈ 1e-2 · I`

### Scenarios the estimator should be tested on

All defined in `Simulator.py`:

| Factory | What makes it hard |
|---|---|
| `default_neural_system` | Baseline: stable oscillator + decays, random `C`. |
| `input_aligned_system` | Two rows of `C` equal `B.T` — input is partially directly observed. |
| `input_blind_system` | `C` columns 0 and 1 zeroed — input effect only reaches `y` via slow coupling. |
| `hidden_input_system` | `CB = 0` exactly — input is invisible at the moment it arrives. |
| `slow_drift_system` | `A[2,2] = 0.999`, near-random-walk slow mode. |
| `nonnormal_chain_system` | Non-normal `A`; large transient growth of `‖A^t‖`. |
| `ill_conditioned_system` | `cond(V) ≈ 20`; uneven mode mixing. |

We should test on **at least** `default_neural_system`, plus 2–3 of the
challenging ones — the report will discuss which scenarios the method handles
well and where it breaks.

---

## 2. The estimation interface (fixed, do not change)

The demonstrator calls this exact function. The signature is non-negotiable.

```python
from typing import Tuple
import numpy as np

def estimate_latent_and_input(
    observation: np.ndarray,    # shape (T, p)
    LatentDim: int,             # n  (the n the demonstrator expects)
    InputDim: int,              # m  (the m the demonstrator expects)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        latent_states: shape (T, LatentDim)
        inputs:        shape (T, InputDim)
    """
```

**Hard rules:**

- Do **not** change the signature.
- Output shapes **must** be exactly `(T, LatentDim)` and `(T, InputDim)` where
  `T = observation.shape[0]`. No off-by-one.
- The function must not crash on any of the scenarios listed above.
- No hyperparameter tweaking at test time. If we want tunable knobs, fix them
  with `functools.partial` *inside the submitted file*. Automatic data-driven
  tuning inside the function is fine and encouraged.
- The function receives only `observation`. It does **not** receive `u`, and
  does **not** receive the true `A, B, C, Q, R`. Everything must be learnt
  from `observation` plus the dimensions `LatentDim`, `InputDim`.

---

## 3. Approach

We will implement and compare three approaches, then submit one primary method.

### Approach A — PCA baseline (must include for comparison)

Centre `Y ∈ R^{T×p}`. Take SVD: `Y = U Σ V^T`. Keep top `LatentDim` columns of
`U Σ` as latent states. Estimate inputs by fitting a linear AR(1) model on the
latent scores and treating the residual, mapped through a pseudoinverse, as the
input.

**Why include it.** It is a transparent baseline that requires zero assumptions
beyond linearity. It exposes the limits of a purely static decomposition on
systems where input visibility is delayed (e.g. `hidden_input_system`).

### Approach B — Subspace identification + Kalman filter (primary)

This is the **primary submitted method**.

#### B.1 — Subspace identification (N4SID-style)

Build past/future block-Hankel matrices from `y`:

```
Y_p = [ y_0 y_1 ... y_{T-2k}   ]    (p·k rows, T-2k+1 columns; rows stack y_{j-1}, y_{j-2}, ...)
Y_f = [ y_k y_{k+1} ... y_{T-k}]    (p·k rows, future block)
```

with block size `k` chosen so that `p·k ≥ 2·LatentDim` and `k ≥ LatentDim`. A
robust default: `k = max(2·LatentDim, ceil(sqrt(T)/2))`, clipped so the Hankel
matrix has at least `4·k` columns.

Compute the oblique projection of future onto past (in the unforced case
`u = 0` we can just use a plain projection; we'll do the simpler version since
we don't have access to `u` at test time):

```
O = Y_f @ pinv(Y_p) @ Y_p          # row-space projection
```

Truncated SVD of `O`:

```
U_n, S_n, V_n^T = SVD(O), keep top n = LatentDim
extended observability  Γ = U_n @ diag(sqrt(S_n))
state sequence          X = diag(sqrt(S_n)) @ V_n^T            # shape (n, ·)
```

Recover system matrices from `Γ` and `X`:

```
C_hat = Γ[:p, :]                   # first p rows of Γ
A_hat = pinv(Γ[:-p, :]) @ Γ[p:, :] # shift-invariance: Γ_underline · A = Γ_overline
```

For `B_hat`, estimate it from the residual of the dynamics regression. Stack:

```
X_next = A_hat @ X_curr + B_hat @ U_curr + noise
```

Since we don't observe `U_curr` directly, we treat `B_hat` and `u_t` as a joint
unknown. We can either:

- **Option 1 (recommended):** estimate `B_hat` by least squares using a
  *self-supervised* construction: fit the **innovations form** of the model
  and then identify `B` from the part of the latent-state residual that is
  *not* explained by the past observation innovations. Specifically, run a
  Kalman filter assuming `B = 0`, look at the unexplained latent residual
  `r_t = x̂_{t} - A_hat x̂_{t-1}`, and regress this onto a learned dictionary
  of `m` input directions. Concretely: do an SVD of the residual sequence
  `R = [r_1 ... r_{T-1}]`, keep the top `m` left singular vectors as columns
  of `B_hat`.

- **Option 2 (fallback if Option 1 is unstable):** set `B_hat` to the top
  `m` left singular vectors of the *innovation-driven* state increments,
  scaled by their singular values.

In both cases `B_hat ∈ R^{n × m}`.

For noise covariances:

```
Q_hat = cov of latent residual r_t = x_t - A_hat x_{t-1} - B_hat u_hat_t
R_hat = cov of observation residual y_t - C_hat x_t
```

Use shrinkage (e.g. add `1e-6 · I`) to keep covariances strictly positive
definite.

#### B.2 — Kalman filtering

With `(A_hat, B_hat, C_hat, Q_hat, R_hat)` in hand, run the standard linear
Kalman filter on `y`:

```
predict:
    x̂_{t|t-1} = A_hat · x̂_{t-1|t-1} + B_hat · û_{t}           # or B·0 in pass 1
    P_{t|t-1} = A_hat · P_{t-1|t-1} · A_hat.T + Q_hat

update:
    S_t = C_hat · P_{t|t-1} · C_hat.T + R_hat
    K_t = P_{t|t-1} · C_hat.T · inv(S_t)
    x̂_{t|t} = x̂_{t|t-1} + K_t · (y_t - C_hat · x̂_{t|t-1})
    P_{t|t} = (I - K_t · C_hat) · P_{t|t-1}
```

Initialise with `x̂_{0|0} = pinv(C_hat) · y_0` and `P_{0|0} = I`.

For the **first forward pass**, set `û_t = 0`. For the **second pass**,
estimate `û_t` from the filtered states (see B.3 below), then re-run the
filter with the estimated `û_t` plugged into the predict step. Two passes are
usually enough.

#### B.3 — Input estimation

After the first filter pass, estimate the input that best explains the
observed latent dynamics:

```
û_t = pinv(B_hat) @ (x̂_{t+1|t+1} - A_hat @ x̂_{t|t})
```

Pad the final timestep so the output has exactly `T` rows (we use `û_{T-1} = û_{T-2}`
or zero, whichever the implementation finds cleaner).

#### B.4 — Identifiability note (must be in the report)

Subspace ID identifies `(A, B, C)` only up to a similarity transform `T ∈ GL(n)`:
if `(A, B, C)` fits the data, so does `(T⁻¹AT, T⁻¹B, CT)`. Therefore:

- Our `latent_states` won't numerically match the simulator's true `x`.
  Evaluate via **predicted observations** `ŷ = C_hat · x̂` instead.
- Our `û` is identifiable only up to a linear transformation of input space.
  Evaluate by **canonical correlation** or by **best linear regression**
  from `û` onto the true `u`, then report the residual.

### Approach C — Time-delay (Hankel) embedding + PCA

For each `t`, build an augmented vector

```
z_t = [y_t; y_{t-1}; ... ; y_{t-L+1}]   ∈ R^{p·L}
```

with `L = max(2·LatentDim, 5)`. PCA `z` to `LatentDim` dimensions. This
captures dynamical structure without explicit state-space ID. Estimate inputs
the same way as Approach B (linear AR(1) residual through `pinv(B_hat)`).

**Why include it.** It's a middle ground: simpler than full N4SID, much
stronger than plain PCA on systems with delayed input visibility.

### Submission choice

Approach B is the primary submitted method, with A and C as comparison
baselines documented in the notebook.

---

## 4. Deliverables

The agent must produce the following files in the project root.

### 4.1 `estimator.py` (submission file)

Self-contained module exposing exactly `estimate_latent_and_input(observation,
LatentDim, InputDim)`. Internally implements Approach B. May import only
`numpy` and standard library. **No scipy, no sklearn** — keep dependencies
minimal so the demonstrator's environment definitely runs it.

The file should also define:

- `_fit_subspace_model(Y, n, m)` → returns `A_hat, B_hat, C_hat, Q_hat, R_hat`.
- `_kalman_filter(Y, A, B, C, Q, R, u=None)` → returns filtered state sequence.
- `_estimate_inputs(X, A, B)` → returns `û` of shape `(T, m)`.

All hyperparameters (block size `k`, regularisation values, number of passes)
are fixed inside the file; nothing tuned at call time.

### 4.2 `baselines.py`

Implements `estimate_pca(Y, n, m)` (Approach A) and `estimate_hankel(Y, n, m)`
(Approach C). Same return shape as the primary method. Used only for
comparison in the notebook.

### 4.3 `week2_analysis.ipynb`

A Jupyter notebook that:

1. **Imports** `Simulator.py`, `Illustrator.py`, `estimator.py`, `baselines.py`.

2. **Section 1 — Sanity check on `default_neural_system`.**
   Generate a 500-step trajectory with `mixed_input`. Run all three estimators.
   Plot (a) true latent states vs estimated (after Procrustes alignment for B
   and C), (b) reconstructed observations `ŷ` vs `y`, (c) estimated input `û`
   vs true `u` (after linear alignment).

3. **Section 2 — Scenario sweep.**
   For each of the 6 challenging scenarios listed in §1, run Approach B and
   report:
   - Reconstruction RMSE of `y` (in-sample).
   - Linear-regression `R²` between `û` and the true `u`.
   - Time series plots side-by-side for the first 200 steps.

4. **Section 3 — Noise sensitivity.**
   On `default_neural_system`, scale `R` by `{0.1, 1, 10, 100}` and report
   how observation reconstruction RMSE and input recovery `R²` change.

5. **Section 4 — Input-pattern generalisation.**
   On `default_neural_system`, run the estimator on trajectories driven by
   each of `zero_input`, `pulse_input`, `sinusoidal_input`, `random_input`,
   `channel_sweep_input`, `mixed_input`. Tabulate reconstruction RMSE and
   input `R²`.

6. **Section 5 — Long-horizon stability.**
   On `default_neural_system` with `mixed_input`, run for `T = 2000`. Plot
   running reconstruction RMSE in 100-step windows. The estimator should not
   diverge.

7. **Section 6 — Limitations.**
   Brief notes on what fails and why. Specifically discuss the `hidden_input_system`
   result (expect degraded input recovery because `CB = 0`) and `slow_drift_system`
   (expect issues with the slow mode because the steady-state Kalman gain for a
   near-random-walk is high).

All figures should be saved into a `figures/` subdirectory as `.png` at 150 dpi.

### 4.4 `interim_report.md`

A markdown report, ~4 pages when rendered, **strictly no more than 5 pages**
including figures. Structure:

1. **System description and Week 1 summary** (~0.5 pages).
2. **Estimation approach** (~1.5 pages): describe Approach B with the maths,
   mention A and C as comparisons.
3. **Results** (~1.5 pages): selected figures from the notebook with
   interpretation. Focus on depth: pick the 3–4 most informative plots
   rather than dumping everything.
4. **Limitations and next steps** (~0.5 pages): identifiability note,
   scenarios where the method fails, plan for Week 3 (control).

Cite figures by filename. Equations in LaTeX. Keep it tight.

### 4.5 `test_interface.py`

A short pytest-style script that verifies the interface contract:

- Output shapes are `(T, LatentDim)` and `(T, InputDim)` for several
  `(T, p, LatentDim, InputDim)` combinations.
- No NaN, no inf in the output.
- Runs in under 30 seconds on `T = 1000, p = 16`.
- Doesn't crash on any of the listed scenarios.

Used both by the agent for self-checking and by us before submission.

---

## 5. Evaluation metrics (used in the notebook)

For every comparison plot the notebook produces, use these metrics:

- **Observation reconstruction RMSE:**
  `sqrt(mean((y - C_hat · x̂)^2))`
- **Latent recovery (subspace-invariant):**
  After Procrustes alignment of `x̂` to true `x`, report mean column-wise
  correlation.
- **Input recovery (subspace-invariant):**
  Fit `M ∈ R^{m×m}` by least squares so `M · û` best matches `u`. Report
  `R²` of the fit and the residual RMSE.
- **Noise sensitivity curve:**
  Plot reconstruction RMSE and input `R²` versus log-scaled `R` multiplier.

---

## 6. Style and quality bar

- Code: type-annotated, docstrings on every public function, NumPy-style.
- Numerical safety: regularise every pseudoinverse/inverse with at least
  `1e-8 · I` on the relevant covariance. Don't crash on rank-deficient
  Hankel matrices.
- Reproducibility: every notebook cell that uses randomness sets a fixed
  seed.
- No silent failures: if a scenario produces NaN, raise with a clear
  message during development; in the submitted `estimator.py`, fall back
  to a zero-filled output of the right shape with a `warnings.warn` so
  the demonstrator's test never crashes.

---

## 7. What to do *before* writing code

1. Read `Simulator.py` end-to-end so the agent understands the exact model
   and the scenario factories.
2. Read `Illustrator.py` so plotting helpers are reused rather than
   reinvented.
3. Skim `introduction_filtering.ipynb` — it defines the Kalman filter
   recursion the project assumes.
4. Skim `week1.ipynb and system_analysis.ipynb` to see what plotting/exploration conventions the
   group already uses, so the Week 2 notebook reads as a natural
   continuation.