# Week 4 — Step 1: Offline brain estimator (Claude Code handoff)

*Self-contained brief for the `week4/` build. Read it fully before writing code. This is **Step 1 only** — the brain estimator. The hand controller is Step 2 and is out of scope here; stop at the checkpoint in §10.*

---

## 0. Your task

Build, from scratch and self-contained in `week4/`, an **offline system-identification pipeline** for the brain, plus the **steady-state Kalman filter** that runs on the identified model. This replaces the Week-3 estimator with an offline version that can use long, richly-excited data. Everything is new code in `week4/`; **do not `import` anything from the Week-3 folder.** You *should* read the Week-3 scripts first (see §2) so the method and conventions carry over, then re-implement cleanly.

Work in our usual style: terse, "we" voice, British spelling; deliver runnable standalone scripts (never code described in prose); every load-bearing claim tied to a printed number or a figure; iterate in small steps and **stop for review at the checkpoint**.

---

## 1. The system we are identifying

A linear-Gaussian state-space model, accessed live through the GG4 wheel as `Brain`:

```
x <- A x + B clamp(u,0,1) + w,   w ~ N(0, Q)        # 6 latent states
y  = H x + v,                    v ~ N(obs_mean, R)  # 16 noisy outputs
```

- Interface: `Brain(random_seed=int)`, `.next_state(u)` (advance one step; `u` length 2, omitted ⇒ zeros), `.measure()` (one fresh noisy draw, no advance), `.current_time_stamp`, `.input_dim` (=2).
- **Inputs are clamped to [0,1] before `B`.** This is the only nonlinearity. Keep excitation inside the box and verify the clipping fraction is negligible, so identification stays in the linear regime.
- `random_seed` changes only the **noise realisation**; the parameters are compiled-in constants, so every seed is the *same* system. Use several seeds for more data, hold seeds out for validation.

**Known structure to expect and to validate against** (from prior weeks + the held-out yardstick; the estimator must *recover* these from data, not assume them):
- **Order 6** (fixed — see §3). One lightly-damped **oscillatory mode at period ≈ 36.5**; the others are slow real decays (eigenvalue magnitudes ≈ {0.017, 0.824, 0.943, 0.951, 0.951, 0.961}; spectral radius 0.961).
- **`H` essentially reads only two state directions** — states 2–5 are weakly observable, only inferable through the dynamics. This is the ill-conditioned part, and the reason rich excitation matters.
- **Input 0 drives one strong direction; input 1 drives the weakly-observed states.** The steady-state (DC) gain is **rank-1** (singular values ≈ 48 vs 0.3). The weak/hidden modes show up mainly under **AC excitation of input 1** — sweep it deliberately or those modes stay invisible.
- **Lopsided noise:** tiny process noise (`Q` diag ≈ 0.04/0.006/0.001), large observation noise (`R` diag ≈ 0.80–0.98) **with a constant nonzero mean** — see the offsets note in §5.

---

## 2. What carries over from Week 3 (read, don't import)

Read these files in the Week-3 control package (ask if the folder path is unclear) and mirror their method and conventions, then **re-implement fresh** in `week4/` with no cross-folder imports:
- `estimator/` — the `EstimatorNew.fit` EM identifier (affine LGSSM, known inputs). Reuse the *method* (subspace/Hankel init → EM refine), not the affine offsets — see §5.
- `control/control_interface.py` — the `calibrate()` Phase-A timing convention. **Preserve it exactly:** at step `t` we `measure()` → `y[t]`, then `next_state(probe[t])` (the input that drives the state from `t` to `t+1`). The identified model's input/output alignment must match this, since the Step-2 controller and the `run_closed_loop` driver assume it.
- `control/observer.py` — `SteadyStateKalman` (solve the predict DARE once, fixed-gain predict/update). Keep this steady-state form.
- `control/probes.py` — `uniform_probe` (the Week-3 broadband probe). **Replaced** here by the frequency-sweep excitation in §4.

**What changes vs Week 3:** Week 3 was constrained to whatever data the online setting gave us and used a Uniform[0,1] probe; Week 4 Step 1 is **fully offline**, so we use *designed* swept-frequency excitation, longer horizons, and several seeds. The identification *method* stays the same family; the excitation and data budget get richer, and we drop the offset parameters (§5).

---

## 3. Order — fixed at 6

Use **n = 6** (the true order). Do **not** sweep orders. In validation, report honestly whether all six modes are recovered or whether the two weakest (smallest-magnitude / weakly-observed) modes sit at the noise floor — that is a finding, not a reason to change the order.

---

## 4. Excitation — a multi-probe library (PRBS + sweep + more) — *experiment*

The decision this gates: does designed, rich excitation recover the slow and weakly-observed modes better than the Week-3 uniform probe? Build an excitation **library** in `excitation.py` and concatenate a schedule of segments for the fit. All kept inside [0,1]; verify near-zero clipping.

**The library — each type justified:**
- **PRBS** (maximum-length pseudo-random binary), independent per channel. The canonical SysID probe: broadband and persistently exciting; decorrelated channels help separate the `B` columns. Use a **clock/bit period of ~5–15 steps** (not every step) so low frequencies — the slow poles — are loaded; toggle between levels inside the box (e.g. 0.1/0.9).
- **Logarithmic chirp (frequency sweep)**, per channel. Sweeps period ≈ 1000 → near-Nyquist (≈ 0.4 cyc/step); the log schedule dwells at low frequency where the period-36.5 resonance and slow poles live, and it is the easiest segment to read in the input plot. Swept channel `u(t) = 0.5 + 0.4·chirp(t)` (range [0.1, 0.9]).
- **Multisine** (sum of sinusoids, Schroeder phases for low crest factor). Precise spectral content at high SNR within the box; place tones across the band with **emphasis near the resonance and the slow region**.
- **Low-pass filtered white noise.** Extra low-frequency energy for the slow / weakly-observed modes that plain white noise under-excites.
- **Uniform[0,1] baseline** (the Week-3 probe). Keep one segment so we can show whether the richer designs actually helped — this is the **ablation** the experiment gates.

**Per-channel vs both-channel:** run **PRBS and chirp per channel** (the other channel held at 0) to isolate each `B` column — including the dedicated **input-1** sweep/PRBS that excites the hidden states; do not skip it. Use **both-channel decorrelated** excitation for multisine, filtered noise, and the held-out validation run.

### Step budget — state it exactly
Named constants at the top of `identify.py`; print totals at runtime; trim if you want it leaner:
- `T_REST  = 3000`  — zero-input recording for the sensor-bias estimate (§5).
- `T_PRBS  = 4000`  per channel ×2  ⇒ 8000  (input-0-only, then input-1-only).
- `T_CHIRP = 4000`  per channel ×2  ⇒ 8000  (input-0-only, then input-1-only).
- `T_MSINE = 3000`  (both channels, decorrelated).
- `T_LPN   = 3000`  (both channels, decorrelated).
- ⇒ **training ≈ 25,000 steps.**
- `T_VAL   = 4000`  — held-out both-channel decorrelated run, **different seed** (never fitted; §7).

Rationale: the slowest pole has τ ≈ 25 steps and the resonance period is 36.5, so each excited channel needs many time-constants/cycles, and the per-channel separation needs each channel driven alone for several thousand steps; ~25k training is rich but bounded — a few× the Week-3 probe, not excessive.

### Required figure — plot exactly what we put in
Produce `excitation_inputs.pdf`: the full input time series `u0(t)` and `u1(t)` across **all** segments with boundaries labelled (rest | prbs-0 | prbs-1 | chirp-0 | chirp-1 | msine | lpn | validation), a panel showing the **chirp instantaneous frequency vs time**, and a zoomed window (a few hundred steps) showing the waveforms. Required deliverable.

**Must-capture:** the exact step counts and segment layout; the clipping fraction (≈ 0); the ablation — fit on **uniform-only vs the full schedule** and compare recovered eigenvalues / held-out prediction to show whether the rich excitation helped (and which segments most improve the weak modes); `excitation_inputs.pdf`.

---

## 5. Offsets — we do **not** carry `a` or `c` (justified)

Week 3 fitted two affine offsets: a state offset `a` and an output offset `c`. We avoid both, with justification:
- **State offset `a`: not needed.** The process-noise mean is zero, so the resting state mean is zero and `a ≈ 0`. We set it to zero (and, as a check, confirm an EM run with `a` free returns `‖a‖ ≈ 0`).
- **Output offset `c`: removed in preprocessing, not fitted.** The observation noise has a constant nonzero mean (a fixed sensor bias). Because the resting state mean is zero, a **zero-input recording** measures this bias directly: `bias = mean of y over the T_REST run at u = 0`. Subtract `bias` from **all** observations before identification and inside the filter. This removes the real bias once, up front, so neither the identified model nor the Step-2 controller carries an offset term — the DC input→output gain is untouched (we removed only the constant, not the input-driven part).

So the identified model is a **zero-offset** LGSSM `(A, B, C, Q, R)` and the observer is the plain affine-free form `x⁻ = A x̂ + B u_prev`, `innov = y_debiased − C x⁻`. **Must-capture:** the measured `bias` vector and the residual output mean after subtraction (should be ≈ 0); confirmation that a free-`a` fit gives `‖a‖ ≈ 0`.

---

## 6. Identification pipeline — *experiment*

1. **De-bias** every observation by the §5 `bias`.
2. **Subspace / Hankel init:** build the block-Hankel matrices from the de-biased input/output data, SVD, realise initial `(A, B, C)` at order 6 (mirror the Week-3 init).
3. **EM refine** (Ghahramani–Hinton, known inputs, zero offsets): refine `(A, B, C, Q, R, x0, P0)` to convergence; plot the log-likelihood (must be monotone non-decreasing — a drop is a bug).
4. **Save** `identified_model.npz` (`A, B, C, Q, R, x0, P0, bias`, plus metadata: order, seeds, step counts).

**Must-capture:** the Hankel singular-value spectrum; EM log-likelihood curve and iteration count; the saved parameters.

## 7. Validation against the yardstick — *experiment*

Ground truth is in `week4/true_brain_params.py` (`A, B, H, Q, R, means`, plus `dc_gain()`, `transfer_function(z)`, `impulse_markov(n)`). **Yardstick only — never imported by the deployed estimator or controller.** We are always a similarity transform away, so **do not compare `A`/`B`/`H` entrywise.** Use gauge-invariant and downstream measures:

- **Held-out prediction (primary):** one-step and k-step-ahead error on the `T_VAL` run, as variance-explained / normalised RMS (purely input→output, so gauge-free).
- **Eigenvalues:** `eig(A_id)` vs the six true eigenvalues, matched by magnitude; check the complex pair recovers period ≈ 36.5 and the slow real poles; report any weak mode that lands at the noise floor.
- **Transfer function / DC gain:** `G_id(z)` vs `G_true(z)` per input channel (Bode + relative error) and the rank-1 DC structure. Invariant to state coordinates.
- **Noise scale:** `Q_id`, `R_id` magnitudes vs truth (scale only — exact entries are gauge-dependent).
- **Readiness statement:** one paragraph on whether the model is faithful enough not to bottleneck control, and where it is weakest (expect: the input-1 / hidden directions).

Forward-note: the definitive "estimator-not-the-limiting-factor" test is **controller performance with `identified` vs `true` model** — that belongs to Step 2; here, just confirm the model is plausibly good enough and flag risks.

**Must-capture:** prediction-error numbers; eigenvalue and transfer-function comparison figures; the explicit readiness statement and weakest direction.

## 8. Kalman filter — *experiment*

Implement a steady-state Kalman filter on `(A_id, B_id, C_id, Q_id, R_id)` in the Week-3 `SteadyStateKalman` form (DARE once, fixed gain), **de-biasing `y` with the §5 `bias`**, exposing a clean `estimate(observations, inputs) -> latent_states` of shape `(T, 6)`. Report the innovation whiteness / NIS sanity check and the steady-state estimation covariance. This is the runtime estimator the Step-2 controller may call. Deployment is single-trajectory, one noisy draw per step — no noise-freezing or seed tricks anywhere.

## 9. Files to produce (all in `week4/`)

- `excitation.py` — the multi-probe library: PRBS, log chirp, multisine, low-pass noise, uniform baseline (§4).
- `identify.py` — builds the segment schedule + data generation + de-bias + subspace init + EM + save `identified_model.npz`; runs the uniform-only vs full-schedule ablation; prints the step budget; writes `excitation_inputs.pdf`.
- `kalman.py` — steady-state Kalman filter on the identified model + `estimate(...)` (§8).
- `validate_identification.py` — all of §7; prints numbers, writes `.pdf` figures.
- `identified_model.npz` — saved model (output of `identify.py`).
- `excitation_inputs.pdf` — the required input time plot (§4).
- `true_brain_params.py` — **already provided**; drop it in, do not modify, import it only in validation.
- `step1_estimator_writeup.md` — method → experiment → result → finding, numbers and figures embedded; raw material for the report's Estimate section.

## 10. Checkpoint — stop here

After §7/§8 validate and the writeup is drafted, **stop and report**: the recovered eigenvalues vs truth (and any weak mode at the floor), the held-out prediction error, the transfer-function fidelity, the measured bias and residual mean, and the readiness statement. Do **not** start the hand controller — that is Step 2, planned separately.

## 11. Constraints recap
- Self-contained `week4/`; no imports from Week-3 (reading it is encouraged).
- Standalone runnable scripts, each with `__main__`; print the must-capture numbers; save figures as files.
- Order fixed at 6; **no** offset parameters (bias removed in preprocessing, §5).
- Offline ID may use long data and many seeds; the deployed filter sees a single noisy trajectory — keep the split explicit in the writeup.
- `true_brain_params.py` appears only in `validate_identification.py`; nothing that runs at control time may read it.
- Build incrementally and pause after each script runs cleanly.