# Week 3 — Control: Benchmarking, Cleanup & Clarity (spec v4, for Claude Code)

This **consolidates and extends** the existing Week-3 build. It supersedes v2/v3 as the current
plan. Major additions this round: (1) **clean up the messy folder** into a clear modular layout;
(2) probe the **real Brain** (via the wheel) to get a ground-truth model and use it to **benchmark
the honest identification pipeline** (full study); (3) fix the **error metrics**; (4) keep **both
1-D and 2-D readouts**; (5) test **integral action under poor identification**; (6) produce
**clean, consistent comparison graphs**. Keep working code where correct; reorganise, don't
rewrite for its own sake.

Two facts from a direct probe of the real Brain reshape the targets (confirm them in M1, don't
assume): the Brain is a **fixed 6-state** linear-Gaussian system (the latent dim is **6**, not the
proxy's 4), **identical across seeds** (the seed only changes the noise realisation), slow and only
mildly oscillatory, with **heavy observation noise** and **essentially one sustainable output
control direction** (the input→output DC gain is near rank-1).

---

## 0. PASTE-IN PROMPT

> Read `week3_spec_v4.md` in full. It consolidates and extends the existing Week-3 build. Work
> milestone by milestone (M0→M6); after each, run its tests, save its figures to `results/`, append
> to `RESULTS.md`, and stop to report numbers + what worked/failed before continuing. Hard rules:
> (1) **M0 is a pure cleanup/reorg** — get the folder modular and documented before adding features;
> (2) the **honest control pipeline never exploits the simulator's determinism** — identify from ONE
> noisy input-output probe run via N4SID→EM with known inputs; the determinism trick is used **only**
> to compute a ground-truth model as a *benchmark yardstick*, clearly quarantined in `analysis/`;
> (3) compare models by **basis-free invariants and held-out prediction error**, never by matrix
> entries; (4) keep every method in its own module behind a common interface, with **consistent
> per-method colours** and **desired-vs-achieved** comparison graphs; (5) report **settling time and
> steady-state error separately** (never whole-trajectory RMS as "the error"), with normalised
> anchors; (6) honest negative results are findings, not failures. Ask before installing anything
> beyond the GG4 wheel + numpy/scipy/matplotlib (QP solver: scipy/osqp/cvxpy).

---

## 1. Target folder layout (M0)

Reorganise the mess into clear, single-responsibility modules; delete dead files (stale dither
artefacts, duplicate figures); add a `README.md` mapping each module to its job.

```
week3/
  estimator/
    identify.py        # N4SID init + affine EM (known inputs); order selection; common Identifier API
  control/
    observer.py        # steady-state Kalman filter (affine, known u)
    controllers.py     # OpenLoop, ProportionalFeedback, PI, LQG, LQGI, PolePlace, MPC, OffsetFreeMPC
    reachability.py     # readout reachable set + feasibility (per readout)
    readouts.py        # build 1-D (dominant controllable dir) and 2-D (PCs) projections M
    plant.py           # BrainPlant, SimulatorPlant (common measure/next_state(clip)/true_state)
    closed_loop.py     # run_closed_loop(plant, controller, model, readout, T, ref) -> logs
    control_interface.py  # IdentifiedSystem — the ONLY importer of estimator/
  analysis/
    brain_probe.py     # wheel install check, probe, catalogue (initial investigation)
    ground_truth.py    # determinism-differencing -> exact (A,B,C,Q,R)  [BENCHMARK ONLY]
  experiments/         # one script per study, named for what it produces
  viz/
    style.py           # shared colour map (method->colour), fonts, helpers
    plots.py           # the comparison-figure functions
  metrics.py           # settling time, steady-state error, effort, saturation, prediction error
  tests/
  results/             # figures + RESULTS.md
  run_week3.py  week3_solution.ipynb  README.md
```
**Common interfaces (enforce):** every controller → `reset()`, `compute(x_hat, ref) -> u`; every
identifier → `identify(Y, U, n) -> model`; readout is a configurable projection `M` (1×p or 2×p),
never duplicated logic. Only `control_interface.py` imports `estimator/`.

---

## 2. Probe the real Brain & build the ground-truth yardstick (M1, `analysis/`)

Install the cp312 manylinux wheel; `from GG4 import Brain; brain = Brain(random_seed=SEED)`;
interface: `input_dim` (2), `current_time_stamp`, `measure()` (16 floats, reads current state,
redraws noise, does **not** advance), `next_state(u)` (apply `u∈[0,1]²`, advance).

**Catalogue (the "initial investigation"):** dimensions; observation-noise scale; autonomous
behaviour; **order via Hankel singular-value knee (expect 6)**; **seed-invariance** (compare
eigenvalues/DC gain across ≥3 seeds — expect identical); **control authority** (singular values of
the DC gain `C(I−A)^{-1}B` — expect near rank-1; report the dominant output direction and the
reachable range along it).

**Ground-truth model (BENCHMARK ONLY — never used by the deployed controller):** exploit
determinism to cancel noise. For each input channel, run an impulse with seed `S` and the same with
zero input, **difference** the outputs → noise-free Markov parameters `H_k = C A^{k-1} B`; run
ERA / Ho-Kalman on their block-Hankel → exact `(A,B,C)` at order 6. Get `R` by repeated `measure()`
at a frozen state; get `Q` by EM with `(A,B,C,R)` fixed (or by matching the stationary output
covariance). Save these as `ground_truth/*` and label them loudly as a yardstick, quarantined in
`analysis/`.

---

## 3. Honest identification pipeline (M2, `estimator/`)

The deployed identifier uses **only** one realistic noisy run with **known** inputs (what a real
brain would give): persistently-exciting probe `u_t ~ Uniform[0,1]²`, record `(U, Y)`. Then:
**N4SID** (input-output subspace) for initialisation → **affine EM** (joint `A,B,C,Q,R` + offsets
`a,c`, known `u` in the predict/M-step), as in the existing `estimator_new`. Set/confirm **`n=6`**
(order selection should find it). Keep `n` configurable. This is the model the controller uses.

---

## 4. The full benchmarking study (M3, `experiments/`)

Grade the honest pipeline against the ground-truth yardstick — the report's rigour centrepiece.

**Comparison must be basis-free** (both models are only defined up to a coordinate change):
- eigenvalues of `A` (sorted); impulse response / Markov parameters `C A^{k} B`; DC gain
  `G = C(I−A)^{-1}B`; and the headline metric, **held-out output-prediction error** on a fresh
  input sequence (one-step and rolled-out). **Never** compare `A,B,C` entry-by-entry.

**Three studies:**
1. **Headline:** honest model's held-out prediction error vs the ground-truth model's — is the
   honest model "good enough" (close to the noise floor)?
2. **Data-efficiency:** sweep calibration length `T_cal ∈ {100,200,500,1000,2000}`; plot
   identification error (held-out prediction) **and** a downstream closed-loop control metric vs
   `T_cal`. The "how much probing do we need" curve.
3. **Local minima:** run EM from several inits (N4SID, output-only SSI, random); compare final
   log-likelihood and held-out prediction error — do they agree, or does EM get stuck?

---

## 5. Readouts — keep BOTH 1-D and 2-D (M4, `control/readouts.py`)

A readout is a configurable projection `M` (`readout = M y`). Build two:
- **1-D — dominant controllable direction:** the top left-singular vector of the DC gain
  `G_full = C(I−A)^{-1}B`. This is the *sustainable* direction; almost every target on it is
  feasible. Use it for the clean "hold/track works" story.
- **2-D — top-2 observation PCs** (PC1, PC2). Keep precisely because PC2 is poorly controllable, so
  this is where the **reachable-region / infeasible-setpoint** story lives. Report the readout's
  `2×2` gain condition number so the limitation is explicit.

Both are config, not code forks. Run the tasks (§7) on each and discuss the contrast in the report:
1-D shows clean control; 2-D shows the physical rank limit and how MPC copes (closest feasible point).

---

## 6. Fix the error metrics (M4, `metrics.py`)

Whole-trajectory RMS is dominated by the ramp-up transient and mislabels holding error — drop it as
"the error". Report **separately**:
- **Settling time** — steps to reach and stay within a band (e.g. ±5% of target).
- **Steady-state error** — mean |achieved − target| over a **post-settling** window (e.g. last 50%),
  anchored to the noise floor ("holds to within 1.2× the per-step jitter").
- **Control effort** (`Σ‖u‖`) and **saturation fraction**.
Pick the sensible anchor per context (don't print "11% of target" and "12.7× noise floor" together
when they disagree wildly — that reads as a contradiction). For the model fit, report relative
prediction error with a clear anchor.

---

## 7. Controllers, tasks, and the integral-under-poor-ID test (M4–M5)

**Controllers** (each its own module, common interface): OpenLoop, ProportionalFeedback, PI, LQG,
LQGI, PolePlace, MPC, OffsetFreeMPC. Integral variants kept **alongside** their non-integral
counterparts; anti-windup required (freeze integral on saturation).

**Tasks** (same controller, different reference), on each readout: **hold** (constant), **track**
(range: slow sine, fast sine, staircase, one deliberately-too-fast), **suppress** (`r=0`, expected
near-floor — a finding, with the one-sided-actuator mechanism).

**The integral test (resolve the open question):** integral action can't help when the model is
accurate (feedforward is already right) — so test it where the model is **biased**: reuse the
data-efficiency sweep (§4). At small `T_cal` the EM model is biased ⇒ feedforward leaves a
steady-state offset ⇒ integral should close it; at large `T_cal` the gap should vanish. Plot
steady-state error vs `T_cal` for feedforward-only vs +integral. State the conclusion the evidence
gives (likely: integral matters only under model error; harmful on tracking via phase lag). Also
keep the accurate-model comparison and report it honestly.

**On infeasible setpoints:** there is no way around them — an out-of-region target is unreachable by
*any* controller because the input is bounded and one-sided; that's physics. The resolution is the
1-D readout (feasibility largely moot) plus showing MPC's graceful closest-feasible-point behaviour
on the 2-D readout. Don't hunt for a trick; explain the limit.

---

## 8. Clean comparison graphs (M4, `viz/`)

A shared `viz/style.py` (one method→colour map used everywhere) + `viz/plots.py`. Rules: one
question per figure in a plain-language title; desired (dashed) vs achieved (solid) vs uncontrolled
(faint grey) for anything target-related; normalised, correctly-anchored errors; consistent colours;
side-by-side bars/small-multiples so "which method wins where" is eyeballable in seconds. Minimal
panels, big labels, no undefined symbols, no jargon ("zonotope" → "levels we can hold"). Also fix
the degenerate-flag to gate on **fit quality** (held-out error / state R²), not just `‖CB‖`.

---

## 9. Tests (extend; keep all existing green)
- Control math on a known synthetic model (observer R², LQR stability, MPC box-feasibility, feedforward
  holds a feasible setpoint).
- **Identification:** EM log-likelihood non-decreasing; on synthetic data the honest pipeline recovers
  invariants (eigenvalues, DC gain) and low held-out prediction error; order selection finds the true `n`.
- **Ground-truth harness:** differencing recovers the known `(A,B,C)` of a SimulatorPlant to high
  precision (validates the benchmark method itself).
- **Integral:** removes steady-state offset under a deliberately biased model; anti-windup bounded under
  infeasible target.
- **Readouts:** 1-D direction matches the top DC-gain singular vector; 2-D condition number reported.

---

## 10. Milestones & acceptance
- **M0 — Cleanup.** Reorg to §1; delete dead files; `README` maps modules; common interfaces; all
  existing tests still pass. *Accept:* clean tree, nothing broken, one importer of `estimator/`.
- **M1 — Brain probe + ground truth.** Catalogue (§2); confirm `n=6`, seed-invariance, rank-1 authority;
  save quarantined ground-truth model. *Accept:* initial-investigation figure + numbers in `RESULTS.md`;
  ground-truth differencing validated on a SimulatorPlant.
- **M2 — Honest pipeline.** N4SID→EM, `n=6`, from one noisy probe run; wired through `control_interface`.
  *Accept:* identifies the Brain; finite, stable model; held-out prediction near noise floor.
- **M3 — Benchmark study.** Headline + data-efficiency + local-minima (§4), basis-free. *Accept:* the
  three figures + a written verdict on "is the honest model good enough / how much data / does EM stick".
- **M4 — Readouts, metrics, controllers, graphs.** Both readouts; fixed metrics; full controller
  comparison incl. integral; clean `viz` graphs. *Accept:* per-task comparison figures (consistent
  colours, settling + steady-state separated) for both readouts; degenerate-flag fixed; the
  integral-under-poor-ID figure with its conclusion.
- **M5 — Real-Brain closed loop.** All tasks on the Brain across ≥5 seeds; robustness/sensitivity sweeps.
  *Accept:* Brain results confirm seed-invariance end-to-end; each comparison has a written "why".
- **M6 — Deliverable.** `week3_solution.ipynb`/`run_week3.py` telling the arc: probe → objective →
  honest ID + benchmark → readouts → control → evaluation → limitations. *Accept:* a newcomer can follow
  it; every plot self-evident.

---

## 11. Definition of done
Clean modular tree (one job per module, one importer of `estimator/`, `README`); real-Brain probe +
quarantined ground-truth yardstick; honest N4SID→EM pipeline at `n=6`; full benchmark study
(held-out prediction, data-efficiency, local-minima, all basis-free); both 1-D and 2-D readouts;
fixed settling/steady-state metrics; controllers incl. integral with the under-poor-ID conclusion;
consistent, self-evident comparison graphs; tests green; a step-by-step notebook. No determinism
exploited in the deployed pipeline; Week-2 `estimator.py` untouched.
