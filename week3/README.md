# Week 3 — Control

Build a closed-loop controller around the `GG4.Brain`. The current organisation
follows spec v4 (`week3_spec_v4.md`). Open that document first if you want the
"why"; this README is the "what lives where".

## Layout

```
week3/
  estimator/
    identify.py        # N4SID warm-start + affine EM (known inputs); the
                       # deployed identifier. Common Identifier API.
  control/
    plant.py           # SimulatorPlant + BrainPlant (clip u, common surface)
    probes.py          # Uniform[0,1] persistently-exciting probe
    observer.py        # Steady-state affine Kalman filter
    controllers.py     # OpenLoop, ProportionalFeedback, PI, LQG, LQGI,
                       # PolePlacement, MPC, OffsetFreeMPC — common API:
                       #   reset(); compute(x_hat, ref) -> u
    reachability.py    # Affine steady-state zonotope + feasibility
    readouts.py        # 2-D PCs (built); 1-D dominant controllable direction
                       # (filled in M4). Each readout is a projection M.
    closed_loop.py     # run_closed_loop(plant, controller, observer, T, ref)
    control_interface.py  # **only** importer of estimator/. The bridge between
                          # the honest identifier and the control stack.
  analysis/
    brain_probe.py     # Initial-investigation catalogue (M1)
    ground_truth.py    # Determinism-differencing → ground-truth model.
                       # QUARANTINED yardstick — NEVER used by the deployed
                       # controller (spec rule #2).
  experiments/         # One script per study, named for what it produces
  viz/
    style.py           # Shared per-method colour map + axis/title primitives
    plots.py           # Comparison-figure helpers (filled in M4)
  metrics.py           # Settling time, steady-state error, effort, saturation,
                       # prediction error — separated, with normalised anchors
  tests/               # All tests; `python -m unittest discover -s tests -t .`
  results/             # Figures + RESULTS.md (findings log per milestone)
  run_week3.py         # End-to-end script telling the spec arc on the Brain
  week3_solution.ipynb # Narrative wrapper around run_week3
  week3.ipynb          # The original assignment notebook (DO NOT EDIT)
  wheels/              # The GG4 wheel artefacts
```

## Hard rules (spec v4)

1. **M0 was a pure cleanup/reorg.** No new features were added during M0 — only
   moves, splits, and a stub for `analysis/` and `viz.plots`.
2. **Honest identification never exploits determinism.** The pipeline through
   `estimator/identify.py` only ever sees one noisy probe run with known inputs.
   The determinism trick lives in `analysis/ground_truth.py` and is used only
   to compute a benchmark yardstick.
3. **Compare models by basis-free invariants and held-out prediction error.**
   Never compare `A, B, C` entry-by-entry — both models are only defined up to
   a coordinate change (eigenvalues, Markov parameters, DC gain, held-out
   prediction error are the right comparisons).
4. **Common interfaces.** Every controller exposes `reset()` and
   `compute(x_hat, ref) -> u`; every identifier exposes
   `fit(Y, U, n) -> model`; readouts are projections `M`.
5. **Report settling time and steady-state error separately.** Never use
   whole-trajectory RMS as "the error" — it is dominated by the ramp-up
   transient (see `metrics.py`).
6. **Honest negative results are findings, not failures.**

## Import boundary (the lone gate)

```
$ grep -rn "^\s*from\s\+estimator" week3
estimator/__init__.py           : from . import identify
control/control_interface.py    : from estimator import identify  # the only consumer
tests/test_estimator_new.py     : from estimator.identify import EstimatorNew  # tests the identifier itself
experiments/m1_estimator_em.py  : from estimator.identify import EstimatorNew  # validates EM directly
```

Controllers, observer, reachability, closed-loop driver — none of them touch
`estimator/`. They consume only `(A, B, C, Q, R, a, c)`.

## Running things

```bash
# Tests (21 cases, all green)
python -m unittest discover -s week3/tests -t week3

# End-to-end Week-3 arc on the Brain (the M6 deliverable)
python week3/run_week3.py --seed 0 --multi-seed 5

# Or open the notebook wrapper:
jupyter notebook week3/week3_solution.ipynb

# Individual milestone experiments
python week3/experiments/m1_initial_investigation.py
python week3/experiments/m2_honest_pipeline.py
python week3/experiments/m3_benchmark_study.py
python week3/experiments/m4_controller_comparison.py
python week3/experiments/m5_real_brain.py
```

## The arc, in one paragraph

`run_week3.py` walks the seven steps of the v4 spec end-to-end on the
real Brain: **PROBE** with a single noisy `Uniform[0,1]²` input run;
**IDENTIFY** with N4SID → affine EM at `n = 6`; **BENCHMARK** against
the quarantined yardstick (basis-free invariants — eigenvalues, DC
gain, Markov parameters); build both **READOUTS** (1-D
dominant-controllable and 2-D PCA) as config; run closed-loop
**CONTROL** on hold / track / suppress tasks; **EVALUATE** with
band-occupancy settling and steady-state error separated, anchored to
the per-readout noise floor; and document **LIMITATIONS** as predicted
findings (suppress = OpenLoop by physics; tracking buried at the noise
floor; 1-D readout ≈ noise scale; etc.).

The deliverable figure is `results/week3_solution.png`. The full
milestone-by-milestone log is `results/RESULTS.md`.

## Headline result

> Honest N4SID → EM identification (one noisy probe, `T_cal = 1000`,
> `n = 6`) gives a model that PREDICTS at the noise floor and CONTROLS
> within ~10 % of the structurally-correct yardstick — achieving a
> **~9× closed-loop win over open-loop** on the 2-D PCA readout across
> 5 Brain seeds. The structural fit error is large (eigenvalue Δ up to
> 0.94, DC-gain rel err 18–63 %), but the control-relevant invariant
> σ₀(G) ratio is recovered to within ~25 %, and that's what control
> needs.

## Findings

See `results/RESULTS.md`. Each milestone appends one section (M0, M1,
… M6) recording the numbers, the figures saved, and what worked /
didn't. The Definition-of-Done checklist at the bottom of the M6
section ticks off every item in spec §11.
