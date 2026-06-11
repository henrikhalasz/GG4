# Week 4 — Step 2B: Calibrate the primitive library

*Self-contained brief for the `week4/` build. Read it fully before writing code. This is **Step 2B only** — measuring the four "wiggle recipes" on the real cascade. **No closed-loop reaching yet** (that is 2C). Stop at the checkpoint in §8.*

---

## 0. Your task

Build the **primitive library**: for each of the four muscles, find the input drive that fires it and measure, on the *live* cascade, what it does to the arm. Output is a small table of recipes the 2C reacher will consume. New, self-contained code in `week4/`; read the Week-3 folder and Step-1/2A files for conventions but do not import across folders unless noted. Same style: terse, "we" voice, British spelling; standalone runnable scripts; every number tied to a print or figure; build incrementally and **stop for review**.

This needs both prior tracks: the **Step-1 identified model** (to predict feasibility) and the **2A kinematics + harness** (to recover joint angles from `hand_pos` and to log/score). 2B is where they merge.

---

## 1. The mechanism that shapes everything — read first

Movement needs `x1` (selection) to *oscillate* at a muscle's band frequency, while `x2` (power) is held high. The decoder gives `x1 = M[0]·y + b[0]`, `x2 = M[1]·y + b[1]`; both are driven by **input 0** (input 1 is a 20–34× spectator). So one channel does both jobs, separated by frequency: **DC level of input 0 → power; AC oscillation of input 0 → selection.**

**The decisive structural fact (confirm it live):** the muscle head has **one shared `power` scalar** multiplying **four per-band `selector` values** — `activation = power · selector`. Consequences the controller will exploit, so measure them as *separate* timescales:
- **Power onset is slow and paid once.** Power is a leaky evidence accumulator on `x2` (enter-rate ≈ 0.055, exit-rate ≈ 0.5): it ramps up over tens of steps from cold, then *latches* as long as `x2` stays above threshold (a sustained DC ≈ 0.5 on input 0 holds it; raising DC further saturates power **and** shrinks the AC swing, so 0.5 is the sweet spot).
- **Switching muscles is faster.** With power already latched, firing muscle k means oscillating `x1` at `f_k`; the per-band selector rises over the band-filter window (~tens of steps, but no fresh power ramp). Switching to muscle j just changes the frequency — power stays up.
- **Stopping = stop oscillating.** Hold `x1` steady (drop the AC, keep DC) → all selectors decay → `activation → 0` → the arm freezes (no inertia). Power can stay latched harmlessly.

This is why the controller should **turn power on once at the start of a reach and keep input-0 DC up throughout, modulating only the oscillation frequency** to select/switch/stop. 2B measures the numbers that make that work.

---

## 2. Read / reuse
- **`BMI_and_Hand` + the muscle-head weights** — the live cascade; wrap a `Brain(seed)` and read `hand_pos` (the 2A `CascadePlant`).
- **2A `kinematics.py`** — `inverse()` (branch-continuity) to recover joint angles `θ = (θs, θe)` from each clean `hand_pos`, and the constants (sat speeds shoulder 0.127 / elbow 0.19, viscous 7.5 / 5). A single-muscle drive moves mostly one joint, so branch tracking is clean.
- **Step-1 `identified_model.npz` + the decoder `M`** — to predict feasibility before measuring (§6).

---

## 3. The drive form and the band frequencies
Per muscle k: `u0(t) = 0.5 + A·cos(2π·f_k·t)`, input 1 held at a constant (default 0.5; check 0 changes little — it is a spectator). DC 0.5 powers the muscle; the cosine selects it. Phase is irrelevant (we do not care about it). Effort is free, so drive `A` hard — but `A` near 0.5 hits the box edges and clips, leaking energy into neighbouring bands, so **sweep `A` (≈0.2–0.49) and pick the value that maximises selectivity (intended band dominates) and joint speed without clipping.** The selector is the graded knob (more AC → higher selector → faster joint); power is effectively on/off at ≈0.65.

Muscle order and bands (from the weights): **0 shoulder+ (period 14.3, f≈0.070), 1 shoulder− (8.0, 0.125), 2 elbow+ (4.9, 0.204), 3 elbow− (3.2, 0.312).**

**Feasibility prediction already computed from the true brain + decoder** (confirm against your identified model, then live): the per-unit-AC selection gain `|x1←u0|` is **sh+ 1.88, sh− 1.08, el+ 0.69, el− 0.50** — monotonically weaker with frequency, so **elbow− is the bottleneck** (~3.8× weaker than shoulder+). Expect el− to clear the selector threshold (0.15) only marginally.

---

## 4. Calibration protocol — *experiment* (the core of 2B)

For each muscle, from a **well-conditioned mid-disk pose** (r ≈ 30, the best-conditioned region) and staying inside the safe annulus (r ∈ [5, 57]) during the drive, recover `θ(t)` from `hand_pos` via IK and measure:

1. **Steady joint angular velocity** — the slope of the driven joint's angle once motion settles. (Pose-independent in principle — the brain/head don't see the arm — so **verify** by driving from 2–3 poses and confirming the same joint velocity.)
2. **Cold power onset** — from power-off, steps until the joint reaches ~90% of steady velocity. This is the one-time startup latency (power-ramp dominated). **The single most important number.**
3. **Warm selector switch** — with power already latched (after a cold start on one muscle), change the frequency to another band and measure steps until the new joint reaches ~90% steady velocity. Confirm this is **shorter** than the cold onset and that **power stays latched** across the switch (the shared-power claim, §1).
4. **Coast / stop** — from steady motion, cut the oscillation (hold `u0` at constant 0.5) and measure the extra joint travel before the velocity reaches ~0. This is the selector-decay coast the 2C stopping rule must lead by.
5. **Active antagonist brake** — instead of holding steady, switch the frequency to the *antagonist* band for a short burst; measure how much faster the joint stops vs passive coast. (Decides lead-only vs lead+brake in 2C.)
6. **Leakage / selectivity** — during a single-muscle drive, measure the *other* joint's velocity (one tone → one joint?) and the runner-up band's selector energy.
7. **Spread** — repeat across ~8 `Brain` seeds; report mean and range for each number above (this is where single-trial honesty bites — the deployed controller sees one trajectory).

**Must-capture:** the per-muscle recipe `{f_k, A*, steady joint velocity, cold onset, warm switch, coast, brake gain, leakage, spread}`; explicit confirmation of the shared-power mechanism (warm switch ≪ cold onset, power latched across switches); the el− bottleneck (lowest reachable activation/velocity); a figure of one cold-start drive showing onset → steady → coast in joint angle.

---

## 5. Forks to resolve with data — *experiment*
- **Fork 1 — coordinated two-joint motion?** Drive `x1` with a **two-tone** sum (e.g. a shoulder band + an elbow band) at power. Do *both* selectors rise and *both* joints move, and does it **superpose** (each joint ≈ its single-tone velocity) or **interfere** (tones split energy / cross-fire the wrong muscle)? Clean superposition ⇒ enable the 2A coordinated reacher in 2C; otherwise stay one-joint-at-a-time. (Default remains one-joint unless this is clean.)
- **Fork 2 — open-loop reliability.** Across the 8 seeds, how often does a primitive fail to fire / fire the wrong muscle within the onset window? In regime A the hand feedback covers occasional misfires, but record the misfire rate so 2C knows whether an inner `x1` check is ever needed (expected: not needed).

**Must-capture:** the two-tone verdict (superpose vs interfere, with numbers) and the open-loop misfire rate.

## 6. Feasibility: identified vs measured — *forward-note for 2D*
Predict `|x1←u0|` at the four bands from the **identified** model and the decoder, and check it reproduces the ordering/magnitudes in §3 (the true-brain numbers). Then compare to the live-measured activations. Record the agreement — this is the start of the "estimator is not the limiting factor" argument that 2D completes. Do **not** build the full identified-vs-true controller comparison here.

## 7. Files to produce (all in `week4/`)
- `primitives.py` — the drive generator (`u0 = 0.5 + A·cos(2π f_k t)`, input 1 constant) and the per-muscle band table.
- `calibrate.py` — the live-cascade calibration (§4) over muscles × seeds via `hand_pos`+IK; prints all must-capture numbers; saves `primitive_library.npz` and the onset/coast figure.
- `forks.py` — the two-tone and open-loop-reliability experiments (§5).
- `primitive_library.npz` — the saved recipe table (per muscle: drive params + measured numbers + spread).
- `primitive_onset.pdf` (cold onset → steady → coast) and a per-muscle summary figure.
- `step2b_writeup.md` — method → experiment → result → finding, numbers/figures embedded; the report's "primitive library" material.

## 8. Checkpoint — stop here
Report: the per-muscle recipe table with spread; the two separated timescales (cold onset vs warm switch) and confirmation of the shared-power mechanism; the coast and brake numbers; the el− bottleneck confirmed live; the two-tone and misfire verdicts; and the identified-vs-true feasibility agreement. **Do not** close the loop, wire the reacher to the cascade, or build the stopping controller — that is 2C.

## 9. Constraints recap
- Drive input 0 only (input 1 a constant spectator); DC 0.5; sweep `A` for selectivity without clipping.
- Recover joint angles from `hand_pos` via 2A IK; never reconstruct from noisy observations.
- Measure cold onset and warm switch as **separate** timescales; confirm the single shared power.
- Characterise across ~8 seeds; report spread, not just means (single-trial honesty).
- The model predicts feasibility; the **live cascade is the source of truth** for every recipe number.
- Standalone scripts with `__main__`; pause after each runs cleanly.
