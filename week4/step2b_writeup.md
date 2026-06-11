# Week 4 — Step 2B: the primitive library

*Self-contained build in `week4/`. We calibrate the four "wiggle recipes" on the
**live** cascade `BMI_and_Hand(Brain(seed))` — the drive that fires each muscle and
what it does to the arm — and resolve the two 2C design forks. **No closed-loop
reaching yet** (that is 2C). Joint angles are recovered from the clean `hand_pos`
by 2A branch-continuity IK; the live cascade is the source of truth for every
number. Run on the GG4 venv python.*

---

## 1. The mechanism (read first, confirmed live)

Input 0 does both jobs, separated by frequency: its **DC** powers the muscle
(`x2`), its **AC** cosine selects the band (`x1`); input 1 is a constant spectator.
We confirmed directly on the cascade:

- **Power is DC-driven and band-independent.** Pure DC=0.5 (no AC) latches the
  shared `power` state to ≈0.65; all four bands cross power 0.5 at the *same* step.
- **One shared power × four selectors.** Reading the muscle head's `power_state`
  and per-band `selector` live: `activation = power · selector`. Power latches once
  and **stays latched when the band frequency changes** (power floor ≥ 0.03–0.11
  across every switch). The selector is a fast, per-band function of the AC tone.
- **Spectator confirmed (identified model + decoder):** input 0 dominates input 1
  by **99×** for power (x2 at DC) and **10–16×** for selection (x1 at the bands);
  held constant, input 1 adds no AC selection.

This is why the controller (2C) pays for power **once** at the start of a reach,
holds input-0 DC up, and selects/switches/stops by changing only the AC frequency.

## 2. The drive and the bands (`primitives.py`)

`u0(t) = 0.5 + A·cos(2π f_k t)`, input 1 = 0.5. Band centres recovered from the
muscle-head filters (zero-padded FFT): **sh+ 0.070, sh− 0.125, el+ 0.205, el− 0.315**
(periods 14.3 / 8.0 / 4.9 / 3.2). The amplitude sweep (`A* ∈ [0.30,0.55]`, fresh
cold latch per A) shows velocity rising with A, **no clipping below 0.5**, and
**clipping at A=0.55 (27% of samples hit the box)** which leaks energy — so
**A* ≈ 0.49** (max clean AC). The strong muscles' selectors saturate (sh+ is flat
from A=0.40, so its A* of 0.45 vs 0.49 is immaterial); the weak el− uses every bit
of AC, so A* = 0.49 there.

## 3. The recipe table (`calibrate.py`, 8 Brain seeds, mean [min,max])

| muscle | f | steady \|v\| (rad/step) | cold onset (steps) | warm switch (steps) | own selector | coast (rad) | brake (rad) |
|---|---|---|---|---|---|---|---|
| sh+ | 0.070 | **0.065** [0.044,0.080] | 93 [68,125] | 36 | 0.98 | 1.89 | 1.53 |
| sh− | 0.125 | 0.038 [0.022,0.049] | 93 [69,123] | 56 | 0.56 | 0.71 | 0.48 |
| el+ | 0.205 | 0.027 [0.019,0.038] | 95 [79,120] | 54 | 0.32 | 0.58 | 0.24 |
| el− | 0.315 | **0.0085** [0.003,0.015] | 94 [31,120] | 58 | **0.198** | 0.22 | 0.15 |

*(numbers from the 8-seed run; see `primitive_library.npz`.)*

**Findings:**
- **Two separated timescales (the shared-power claim):** mean **cold onset ≈ 93
  steps** (the slow, band-independent power ramp — the single most important
  number) versus **warm switch ≈ 51 steps** (the selector rise with power already
  latched), ~1.8× faster, with power staying latched across the switch. Paying for
  power once and switching cheaply is real.
- **el− is the bottleneck**, ~7.7× slower than sh+ (0.0085 vs 0.065 rad/step). Its
  own selector sits at **0.198**, barely above the 0.15 threshold (below it on some
  seeds → near-misfires), confirming the predicted ordering and that el− only just
  clears the threshold.
- **Stopping:** cutting the AC (hold DC) coasts the joint a further **0.22–1.89 rad**
  (selector decay) — the lead the 2C stopping rule must anticipate; the shoulder
  coasts furthest. An **antagonist brake** cuts the stop distance by ~**1.2–2.4×**
  (brake travel < coast travel for every muscle), so 2C can add a brake if lead-only
  overshoots.
- **Selectivity:** the other joint barely moves under a single tone (≤ 0.0013
  rad/step, vs 0.0085–0.065 for the driven joint) and the runner-up selector stays
  ≤ 0.06 — one tone → one joint.
- **Spread (single-trial honesty):** seeds vary — notably **seeds 3 and 6 are
  "slow-power"** (cold onset ~120 vs ~70), and el− occasionally fails to clear the
  selector threshold. The deployed 2C controller sees one trajectory, so this spread
  matters; the hand feedback in regime A covers occasional misfires.

![cold onset → steady → coast](primitive_onset.png)
![per-muscle summary](primitive_summary.png)

## 4. The two forks (`forks.py`)

- **Fork 1 — coordinated two-joint motion: SUPERPOSES.** A two-tone drive (sh+ band
  + el+ band, A=0.24 each) moves **both** joints, each at ~**1.00×** its single-tone
  speed (no interference, no wrong-muscle cross-fire) across 4 seeds. So 2C *may*
  enable the coordinated reacher. **Caveat:** the amplitude budget must split
  (2A < 0.5 ⇒ A≈0.24 per tone), so each joint runs at the weaker two-tone drive —
  coordinated motion is feasible but slower per joint than dedicated one-joint at
  A=0.49, so **one-joint stays the default** unless the speed-up from moving two
  joints at once is worth it.
- **Fork 2 — open-loop misfire rate: 6% overall.** sh+, sh−, el+ never misfire
  (0/8 seeds); **el− misfires 25% (2/8)** — it only marginally clears the selector
  threshold. In regime A the hand feedback covers occasional misfires, so **no inner
  x1 check is needed** in 2C (as expected); el− is the one to watch.

## 5. Feasibility: identified vs true vs live (§6, forward note for 2D)

Per-band selection gain `|x1←u0|` predicted from the **identified** model + decoder
reproduces the true-brain numbers in ordering *and* magnitude, and matches the live
ordering:

| muscle | identified | true-brain | live steady \|v\| (norm) |
|---|---|---|---|
| sh+ | 1.92 | 1.88 | 1.92 |
| sh− | 1.10 | 1.08 | 1.12 |
| el+ | 0.71 | 0.69 | 0.79 |
| el− | 0.51 | 0.50 | 0.25 |

The model is an excellent predictor of feasibility (the estimator reproduces the
true gains; el− under-performs live because it only marginally clears the
selector threshold). This is the start of the 2D "the estimator is not the limiting
factor" argument — the full identified-vs-true controller comparison is **not** built
here.

## Files
`primitives.py`, `calibrate.py`, `forks.py`; artefacts `primitive_library.npz`,
`primitive_onset.pdf`, `primitive_summary.pdf`.

## What 2B leaves to 2C
Closing the loop: wiring the reacher to the cascade, turning power on once per reach
and modulating the AC frequency to select/switch/stop, and the stopping controller
(lead from the measured coast, optional antagonist brake). The recipe table above is
exactly what that controller consumes.
