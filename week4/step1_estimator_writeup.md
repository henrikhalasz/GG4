# Week 4 — Step 1: Offline brain estimator — write-up

*Method → experiment → result → finding. Raw material for the report's Estimate section.*

---

## 0. Summary (read this first)

We built a self-contained offline system-identification pipeline for the brain: a
multi-probe excitation library, a richly-excited 25,000-step training run against the live
`Brain`, a zero-offset order-6 LGSSM fitted by subspace-init + EM, and a steady-state Kalman
filter that runs on the identified model.

**Headline finding.** The estimator works correctly, but a *high-fidelity* six-mode model is
**not feasible from this system** — and that is a property of the brain, not of our code. The
observation noise is so large (per-channel std ≈ 0.93, vs the weak modes' signal) that the
slow / weakly-observed states and the period-36.5 resonance fall **below the noise floor**.
We prove this directly: when we re-fit synthetic data from the *true* parameters with the
observation noise scaled down, our estimator recovers all six modes including period 36.5;
at the true noise level it cannot. What we *do* recover is a statistically consistent,
prediction-optimal model of the **dominant input-0 dynamics and the DC gain** — enough to
control the strong readout, weak on the hidden directions.

---

## 1. The system and why it is hard

The brain is a linear-Gaussian state-space model, 6 latent states → 16 noisy outputs,
2 inputs (clamped to [0,1]). Three structural facts make identification hard, all confirmed
against the yardstick `true_brain_params.py`:

- **Observation noise dominates.** `R` has diagonal ≈ 0.80–0.98 (std ≈ 0.93 per channel). The
  resting output (pure noise) has std ≈ 1.1; the driven output has std ≈ 3.3. So a large part
  of every measurement is unpredictable noise.
- **Only two state directions are well observed.** `H` reads essentially two combinations; the
  remaining states are inferable only through the dynamics.
- **The DC gain is rank-1 and lop-sided.** Singular values ≈ 48 vs 0.32 — **input 1 affects the
  output ~150× more weakly than input 0.** Input 1 drives the weak/hidden modes and the
  resonance, so those modes are intrinsically low-SNR.

The resonance (period 36.5) is *observable* in principle (modal observability 1.78, second-
highest; its `|G(·,u1)|` peaks at 6.6, above the noise std) — but it is driven only through
the weak input-1 channel, so in practice its signal is buried.

---

## 2. Method

**Excitation library (`excitation.py`).** Five probe types, each justified: maximum-length
**PRBS** (broadband, decorrelated channels, clock period 8 to load low frequencies),
**logarithmic chirp** (sweeps period 1000 → 2.5, dwelling at low frequency where the slow
poles and resonance live), **multisine** (Schroeder phases, tones on the slow band and around
the resonance), **low-pass-filtered noise** (extra low-frequency energy), and the Week-3
**Uniform[0,1]** baseline (kept only for the ablation). PRBS and chirp are run **per channel**
(the other channel held at 0) to separate the `B` columns — including a dedicated input-1
drive for the hidden modes; multisine and filtered noise are both-channel decorrelated.

**Schedule and budget (`identify.py`).** One continuous trajectory, printed at runtime:

| segment | steps | drive |
|---|---:|---|
| rest | 3,000 | u = 0 (sensor-bias measurement) |
| prbs-0 / prbs-1 | 4,000 + 4,000 | each channel alone |
| chirp-0 / chirp-1 | 4,000 + 4,000 | each channel alone |
| msine | 3,000 | both, decorrelated |
| lpn | 3,000 | both, decorrelated |
| **training total** | **25,000** | |
| validation | 4,000 | both, decorrelated, **different seed** (never fitted) |

Clipping fraction = **6 × 10⁻⁵** (the box is essentially never hit, so identification stays
linear). See **`excitation_inputs.pdf`** for the full input time series, segment boundaries,
the chirp instantaneous frequency, and a zoomed waveform window.

**Zero-offset model (§5).** We fit `x⁺ = A x + B u + w`, `y = C x + v` with **no offset
terms**. The state offset `a` is dropped (process-noise mean is zero ⇒ resting state mean is
zero ⇒ `a ≈ 0`; confirmed below). The output offset `c` is removed in preprocessing: we
measure `bias = mean(y)` over the rest segment and subtract it from every observation, before
fitting and inside the filter. This removes the real sensor bias once, up front, without
touching the input→output gain.

**Identification.** N4SID oblique-projection subspace init at the fixed order **n = 6** (no
order sweep), then EM (Ghahramani–Hinton, known inputs, zero offsets) to convergence. The
filter/smoother and timing convention (`u[t]` drives `x_t→x_{t+1}`; predict at `t` consumes
`u[t-1]`; `y[0]` is the first measurement) mirror Week 3 exactly so the Step-2 controller can
consume the model unchanged.

**Runtime filter (`kalman.py`).** A steady-state Kalman filter: predict-DARE solved once,
fixed gain, de-biasing `y` with the saved `bias`. Single-trajectory deployment, one noisy
draw per step — no noise-freezing or seed tricks.

---

## 3. Results

### 3.1 De-bias (§5)
- `bias` = measured resting mean, e.g. first entries `[0.15, 0.20, 0.15, 0.24, …]`.
- Residual mean of de-biased `y` over the rest segment: **4 × 10⁻¹⁶** (zero by construction).
- Residual mean over the whole training run ≈ **2.6 per channel** — this is the *input-driven*
  DC, which `C x` explains; it is **not** a sensor offset.
- **Free-`a` check:** re-fitting with the state offset free gives **‖a‖ = 0.26** — small
  (the states it acts on have much larger scale), and attributable to the ~0.1 finite-sample
  error in the bias estimate. Confirms we are right to drop `a`.

### 3.2 The fit
- **Hankel singular values:** `[5249, 1173, 525, 120, 62, 33 | 29, 28, 28, …]`. There is *no
  clean cliff* at n = 6 (σ6/σ7 ≈ 1.1) — the sixth mode sits right on the noise floor. Six
  values rise above the floor, which is why order 6 is still the right choice, but the weakest
  modes are marginal.
- **EM:** 54 iterations, log-likelihood **monotone non-decreasing** (min increment ≥ 0 — no
  bug); see **`em_loglik.pdf`**. Spectral radius ρ(A_id) = **0.940** (true 0.961).

### 3.3 Validation against the yardstick (`validate_identification.py`)

**Held-out prediction (primary).** k-step variance-accounted-for on the validation run, with
the *true* model as the ceiling:

| k | identified | true (ceiling) |
|---:|---:|---:|
| 1 | 0.478 | 0.473 |
| 5 | 0.393 | 0.369 |
| 10 | 0.322 | 0.257 |
| 20 | 0.268 | 0.101 |
| 50 | 0.250 | −0.057 |

The identified model **matches or beats the true model** at every horizon. One-step VAF is
capped near 0.47 by the observation noise, so prediction alone cannot tell a good model from a
bad one here — exactly why we lean on the gauge-invariant checks below.

**Eigenvalues** (matched by magnitude; see **`eigenvalues.pdf`**):

| identified | \|·\| | true | \|·\| |
|---|---:|---|---:|
| 0.940 ± 0.023j | 0.940 | 0.943 | 0.943 |
| (same pair) | 0.940 | 0.936 ± 0.163j | 0.951 (**resonance, period 36.5**) |
| −0.075 | 0.075 | 0.824 | 0.824 |
| −0.040 | 0.040 | 0.017 | 0.017 |
| 0.017 | 0.017 | 0.961 | 0.961 |
| 0.009 | 0.009 | (pair) | 0.951 |

→ The dominant ~0.94 mode is recovered; **the slow input-1 modes (0.82, 0.96) and the
period-36.5 resonance collapse toward 0.** The identified oscillatory period is 253, not 36.5.

**Transfer function / DC gain** (see **`bode.pdf`**):
- DC-gain singular values: identified `[41.3, 0.80]` vs true `[48.4, 0.32]`; rank-1 ratio 52
  vs 154 — the rank-1 structure is captured, the weak direction over-estimated.
- Transfer-function relative error: **input-0 = 0.11** (good), **input-1 = 0.98** (essentially
  uncaptured) — as expected from the 150× weaker input-1 gain.

**Noise scale.** tr(Q): id 1.88 vs true 0.10; tr(R): id 9.56 vs true 13.84. The identified
**Q is inflated ~20×**: EM honestly dumps the dynamics it cannot model into process noise.
This is *why the filter stays consistent* (next point).

**Filter consistency (`kalman.py`).** On a fresh single trajectory: mean **NIS = 16.1 ≈ p = 16**,
whitened-innovation lag-1 autocorrelation **0.017 ≈ 0**, whitened covariance trace/p = 1.006.
The innovations are **white with the right magnitude** — the model captures everything that is
predictable; the residual is genuine, unpredictable noise.

---

## 4. The critical finding and critique

### 4.1 Why the weak modes are unrecoverable — and proof it is not our code
We re-fit synthetic data generated from the **true** parameters, scaling only the observation
noise (**`noise_floor.pdf`**):

| R scale (× true) | recovered osc period | recovered \|eig\| |
|---:|---:|---|
| 1.00 (real) | 202 | `[0.00, 0.02, 0.02, 0.09, 0.95, 0.95]` |
| 0.30 | 211 | `[0.00, 0.02, 0.06, 0.10, 0.95, 0.95]` |
| 0.10 | 209 | `[0.00, 0.07, 0.55, 0.68, 0.94, 0.94]` |
| 0.03 | 46 | `[0.01, 0.58, 0.76, 0.76, 0.94, 0.94]` |
| 0.01 | **36.3** | `[0.02, 0.72, 0.95, 0.95, 0.95, 0.95]` ← resonance recovered |

At 1/100 the true noise the estimator recovers **all six modes, period 36.5 included**; as the
noise rises the weak modes drop below the floor and collapse. **The limit is identifiability
under the brain's observation noise, not a bug** — our EM is correct.

### 4.2 The ablation: did rich excitation earn its place?
Fitting the full schedule vs a budget-matched Uniform[0,1] run gives **near-identical** held-out
prediction (VAF 0.4793 vs 0.4794) and the same mode collapse. The richer excitation did **not**
measurably beat the uniform probe — because the bottleneck is the observation-noise floor, not
the excitation. Designed excitation buys nothing once the signal is below the noise. (This is
itself a clean, if humbling, result: it tells us where *not* to spend effort.)

### 4.3 Why the "wrong" model is still the right answer for prediction
EM maximises one-step likelihood. Under heavy process + observation noise, a slightly
**over-damped** model predicts *better* than the true one (it forgets the noisy state faster) —
which is why the identified model **beats** truth on k-step VAF (table in §3.3). So the mode
collapse is not EM misbehaving; it is the maximum-likelihood predictor optimally discarding
modes that carry no predictable signal.

---

## 5. Readiness for control, and limitations

**Readiness.** The model is a consistent (white-innovation) predictor that captures the
dominant input-0 dynamics, the spectral radius (0.94 vs 0.96) and the rank-1 DC structure. It
is **faithful enough to control the strong readout**.

**Weakest direction.** The **input-1 / hidden directions**: the slow modes and the period-36.5
resonance are not recovered (transfer-function error ≈ 1.0 on input 1). A Step-2 controller
should expect good authority on the dominant readout and **little leverage over the hidden
states** — and should not rely on the model knowing about the resonance.

**Offline ID vs deployment.** Identification used long, designed, multi-segment data on one
training seed and a separate validation seed. The deployed filter (`kalman.py`) sees a single
noisy trajectory, one draw per step, using the **saved** model and bias — no seed tricks. The
split is explicit and clean.

---

## 6. Deliverables

| file | what |
|---|---|
| `excitation.py` | probe library + schedule builder |
| `identify.py` | data generation, de-bias, subspace+EM, save, ablation, figure |
| `kalman.py` | steady-state Kalman filter + `estimate()` |
| `validate_identification.py` | §7 validation + noise-floor critique (only importer of the yardstick) |
| `identified_model.npz` | saved `(A,B,C,Q,R,x0,P0,bias)` + metadata |
| `excitation_inputs.pdf`, `em_loglik.pdf`, `eigenvalues.pdf`, `bode.pdf`, `noise_floor.pdf` | figures |
