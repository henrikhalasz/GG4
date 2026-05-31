# Week 3 — Control: Exploration & Clarity Pass (spec v3, for Claude Code)

This **extends the existing v2 build** (the affine `estimator_new.py` + `control/` package +
Stage A/B harness). Do **not** rebuild working parts. This round is about (a) making the whole
thing **obvious to a reader who doesn't know the project**, (b) reframing from "one objective"
to **exploration of several control tasks**, and (c) adding **integral-action variants alongside**
the existing controllers and comparing them. The single most important success criterion this
round is **plot clarity**.

---

## 0. PASTE-IN PROMPT

> Read `week3_spec_v3.md` in full. It extends the existing Week-3 build — keep the working
> `estimator_new.py`, `control/`, and Stage A/B harness; do not rebuild them. Implement the
> changes milestone by milestone (E0→E4), and after each one regenerate the figures it names,
> append to `RESULTS.md`, and stop to report. The priorities, in order: (1) **clarity** — every
> plot must be obvious to someone who doesn't know the project: define `x`/`y`/readout once,
> control **the 2 leading principal components** (explained in words as the dominant population
> activity modes), use **desired-vs-achieved** plots everywhere a target is involved, **report
> errors normalised** (as a fraction of the target and relative to the noise/open-loop floor —
> never bare absolute numbers), and use plain-language titles and axis labels (no undefined
> symbols, no "zonotope"); (2) **exploration** — exercise ONE controller family across three
> tasks (hold a target, follow a trajectory, suppress) and compare; (3) **integral action** —
> add integral variants *alongside* the existing controllers (don't replace them) and compare
> with-vs-without; (4) **analysis** — for every comparison, write down *why* the ordering came
> out as it did, not just the numbers. Keep the modular boundary (only `control_interface.py`
> imports the estimator). Ask before installing anything new.

---

## 1. What changes vs the v2 build

Keep: `estimator_new.py`, the observer, the existing controllers, plant wrappers, driver, the
Stage-A-vs-B harness, reachability math. Change/add:

1. **Readout = the 2 leading principal components (keep PCs), explained clearly.** The 2 leading
   PCs of the measurements are the directions of largest population activity — the dominant
   collective patterns the system actually expresses — so "control the top-2 PCs" means "control
   the dominant modes of population activity," a more meaningful target than two arbitrary
   electrodes and standard in neural analysis (§3). The only clarity cost is the axis *label*; fix
   that by stating in words what a PC is, not by changing the readout. Verify the 2 PCs are
   well-controllable (inputs actually move them) and say so.
2. **Reframe to exploration of three tasks** (§5): hold / track / suppress — same controller,
   different reference. No "primary objective."
3. **Add integral-action variants** (§4) alongside the existing controllers and compare.
4. **Redesign all figures into a step-by-step story** (§6) with strict clarity rules.
5. **Small fixes** (§8): delete the stale dither figure; fix the degenerate-flag; note the
   spurious `z0`.

---

## 2. Concepts the figures and `RESULTS.md` must make explicit

State these in words, once, before any result (in the notebook narrative and `RESULTS.md`):

- **`x` — hidden state** (the ~4 latent variables; never observed).
- **`y` — the 16 measurements** (what we see each step).
- **readout — the 2 quantities we choose to control**, here the **2 leading principal components
  (PCs) of the measurements**: each is one weighted combination of the 16 measurements capturing a
  dominant pattern of population activity. Explain a PC in one plain sentence on first use; call
  them "the dominant activity modes" / "PC1, PC2", never a bare `z`.
- **Two senses of "control" (this is the key clarification):**
  - *Influence over time* — the 2 inputs, applied repeatedly, CAN stir the entire latent state
    (the systems are controllable). We are **not** limited to 2 latents.
  - *Hold at a steady value* — to pin quantities at fixed targets needs steady inputs, and with
    **2 input knobs you can independently hold exactly 2 quantities** (like 2 taps setting flow
    and temperature). The "2" comes from having 2 inputs — nothing to do with the 4 latents, and
    it's why we control a 2-D readout.
- **Feasible vs infeasible target** — because each input is one-sided and bounded (`0≤u≤1`,
  push-only), the set of *(PC1, PC2)* levels you can **hold** is a bounded region.
  **Feasible** = inside it (some allowed steady input holds it); **infeasible** = outside it
  (no allowed input reaches it). This is the reachable region; label its axes "PC1 activity" /
  "PC2 activity", not "zonotope".

---

## 3. Readout selection (PCs, documented)

The readout is the **2 leading principal components** of the measurements: run PCA on the observed
`y` (from a probe or open-loop run), take the top-2 components → a `2×16` projection `M`, readout
`= M y`. Justify in one line (dominant population activity modes). Then **verify controllability of
the readout**: compute the steady-state input→readout gain `G = M C (I − A)^{-1} B` (`2×2`) and
check it is well-conditioned (the inputs can move PC1 and PC2 independently); report its condition
number in `RESULTS.md`. If the leading PCs happen to be poorly controllable, say so and note which
PCs *are* controllable — don't silently swap to a different readout. Keep the choice configurable.

---

## 4. Controllers & integral variants to compare

Existing (keep): **OpenLoop**, **ProportionalFeedback** (model-free), **LQG** (feedforward + state
feedback), **MPC**, **PolePlacement**. Add integral variants *alongside*:

- **PI (model-free)** = ProportionalFeedback + integral term. Clean, interpretable, no model.
- **LQG + I** = LQG with an integral state. Augment with the running error
  `q_{t+1} = q_t + (r − readout_t)` and design the gain on the augmented state `[x_hat; q]`
  (LQR on the augmented system); law `u = clip(u_ff − K x_dev + K_i q, 0, 1)`.
- **Offset-free MPC** = standard MPC augmented with a constant disturbance state estimated online
  (the MPC analogue of integral action; corrects steady-state offset from model error).

**Anti-windup (required):** when `u` saturates at `0` or `1`, stop accumulating the integral (freeze
or clamp `q`). Without it, an infeasible target makes `q` grow unbounded ("windup").

**Where integral is appropriate:** for **hold** and **track** (driving a readout to a reference),
where it removes steady-state offset due to model error. **Not** for suppression (`r=0` targets the
*mean*; suppression is about *variance* — say so). Compare each integral variant against its
non-integral counterpart and explain the difference.

---

## 5. The three tasks (exploration; same controller, different reference)

Run every controller on each, and compare:

1. **Hold a target steady** (reference = constant). Use one **feasible** target (should land on it)
   and one **infeasible** target (should get as close as the reachable region allows and stop).
2. **Follow a trajectory** (reference moves). Try a **range** and compare: slow sine, faster sine,
   a staircase of step changes, and at least one **deliberately too-aggressive** reference (too
   large → leaves the region, or too fast → system can't keep up). Explain which succeed and *why*.
3. **Suppress** (reference = 0). Show achieved vs uncontrolled baseline; expect it to barely work
   and explain the mechanism (one-sided input cannot cancel a symmetric oscillation).

---

## 6. Figures: the step-by-step story (clarity is the deliverable)

**Global plot rules (enforce on every figure):**
- One question per figure; put it in the title in plain language (e.g. "Can we hold PC1 at a
  target?"). No undefined symbols anywhere on the plot. No jargon ("zonotope" → "levels we can hold").
- For anything about reaching a target: **desired (dashed) vs achieved (solid), with the
  uncontrolled run as faint grey**, target value annotated, y-axis labelled "PC1 activity" etc.
- **Report errors normalised, never as bare absolute numbers.** Give error as a fraction of the
  target (e.g. "settles within 3% of the target") **and** relative to the achievable floor (the
  readout's noise jitter / open-loop variability — i.e. "hits the target to within the noise").
  The same applies to the model fit: report `G` error as relative error `‖G_fit−G_true‖/‖G_true‖`
  with that anchor, not a bare percentage. A raw "0.2" or "6%" with no reference is not acceptable.
- Comparisons: one consistent controller→colour map across all figures; clear legend; bars/lines
  with the metric named in words.
- Keep it minimal — fewer panels, bigger labels.

**Sequence (the narrative):**
1. **Setup.** A small schematic + words: hidden state `x`, 16 measurements `y`, and the readout =
   the 2 leading PCs (one plain sentence on what a PC is). Plus the **uncontrolled baseline**: the
   2 PCs drifting/wobbling on their own.
2. **What can we hold?** The reachable region in (PC1, PC2) axes, shaded = "levels we can
   hold steady", with a feasible target (inside, green) and an infeasible one (outside, red), and a
   one-line caption: bounded because inputs only push (0→1).
3. **Task 1 — hold a target.** Desired vs achieved for the feasible target (lands) and the infeasible
   target (stops at the region edge). Controllers overlaid (Open, LQG, LQG+I, MPC).
4. **What integral adds.** Zoom on the held value: feedforward-only parks slightly off-target;
   +integral closes the gap to zero (feasible); show windup vs anti-windup on the infeasible target.
5. **Task 2 — follow a trajectory.** Desired vs achieved for the range of references; a small-multiple
   grid (one panel per reference), with a one-line verdict per panel (tracks / lags / leaves region).
6. **Task 3 — suppress.** Achieved vs uncontrolled baseline; show the small effect and state why.
7. **Does the learned model suffice?** Stage A (true model) vs Stage B (fitted) overlaid on the hold
   and track tasks — same result ⇒ estimator good enough. (Reuse the working harness.)
8. **Controller comparison.** Error-vs-effort (and a simple error bar chart) per task; which wins
   where, with the mechanism explained.
9. **Is the learned model trustworthy?** EM validation (convergence, fitted-vs-true `G`, state R²) —
   the existing fig, retitled in plain language.
10. **Limitations & findings.** Suppression floor; `slow_drift` can't settle in the horizon;
    `hidden_input` authority loss; integral windup. Each as a short, honest finding.

Supporting (keep, lightly relabelled): robustness vs noise; sensitivity sweeps (`T_cal`, `n`, `ρ`, `H`);
Brain multi-seed performance on the hold task.

---

## 7. Analysis requirement (not just numbers)

For every comparison figure, `RESULTS.md` must include a short **"why"**: the mechanism behind the
ordering. Examples of the kind of reasoning expected: "MPC beats clipped-LQG near saturation because
it plans within `[0,1]` instead of clipping after the fact"; "PI removes the steady-state offset that
feedforward-LQG leaves because the integral keeps pushing until error is zero"; "tracking the fast
sine fails because the reference moves faster than the closed-loop bandwidth / leaves the reachable
region." Also state results against the right anchor (fraction of target, and vs the noise floor),
so "good" and "bad" are unambiguous. This is the exploration deliverable.

---

## 8. Small fixes
- Delete the leftover `fig8_dither_tradeoff.png` (dither was cut); the real fig 8/9 is EM validation.
- Fix the degenerate-flag: gate on **fit quality** (state R² / one-step error / `G` error), not just
  `‖CB‖` — it currently false-flags `slow_drift` (R²≈1) and misses `input_blind` (R²≈0.68).
- Note the spurious fitted `z0` offset (true `z0≈0` on the Simulator, EM invents a small one) —
  harmless (doesn't affect `G` or setpoint), but mention it; the offset capacity is still correct
  for the Brain.

---

## 9. Tests (extend the existing suite)
- **Integral removes offset:** on a known model with a deliberately *mismatched* model given to the
  controller, feedforward-only leaves a steady-state offset and the integral variant drives it to ~0
  (feasible target).
- **Anti-windup:** under an infeasible target, the integral state stays bounded and `u` pins at the
  limit without divergence.
- **Readout controllability:** the 2-leading-PC readout has a well-conditioned `2×2` input→readout
  gain `G` (PC1 and PC2 independently controllable); report the condition number.
- Keep all existing tests (control math, EM correctness, Stage-A-vs-B) green.

---

## 10. Milestones & acceptance

- **E0 — Readout + concepts.** Keep the 2-leading-PC readout but verify and document its
  controllability (§3); add the concept text to the notebook/`RESULTS.md` (§2). *Accept:* readout
  is the 2 leading PCs with a reported (well-conditioned) `G`; notebook states `x`/`y`/readout,
  what a PC is, and the two senses of "control" plainly.
- **E1 — Integral variants.** Add PI, LQG+I, offset-free MPC with anti-windup (§4); new tests pass.
  *Accept:* with-vs-without integral comparison runs; integral closes steady-state offset on a
  mismatched-model test; anti-windup bounded under infeasible target.
- **E2 — Three-task exploration.** Run all controllers on hold/track/suppress with the reference
  range (§5). *Accept:* results table per task; tracking includes a deliberately-failing reference.
- **E3 — Figure redesign.** Regenerate all figures to the §6 sequence under the §6 clarity rules.
  *Accept:* every figure has a plain-language title, defined axes, desired-vs-achieved where relevant,
  consistent colours; stale dither figure gone.
- **E4 — Story + analysis.** Update `week3_solution.ipynb`/`run_week3.py` to tell the step-by-step
  story; add the "why" analysis (§7) for each comparison. *Accept:* a reader new to the project can
  follow setup → what's holdable → hold → integral → track → suppress → model-good-enough →
  comparison → limitations, with each plot self-evident.

---

## 11. Definition of done
The existing build, extended: 2-leading-PC readout (controllability verified and documented);
integral variants (PI, LQG+I, offset-free MPC) compared against their non-integral counterparts;
all three tasks explored with a range of tracking references; every figure redesigned to be
self-evident (defined notation, desired-vs-achieved, **normalised errors with clear anchors**,
plain titles); a step-by-step notebook a newcomer can follow; and a "why" mechanism written for
every comparison. Modular boundary intact; `estimator.py` (Week 2) untouched.
