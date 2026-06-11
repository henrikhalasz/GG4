# Week 4 — Step 2C: Close the loop on the real cascade

*Self-contained brief for the `week4/` build. Read it fully before writing code. This is **Step 2C** — wire the 2A reacher to the real cascade through the 2B primitives and reach targets end-to-end. Stop at the checkpoint in §10.*

---

## 0. Your task

Make the whole thing run on the real arm: the 2A steering, driving the 2B-calibrated primitives, through `BMI_and_Hand`, reaching a sequence of targets under hand feedback — and the live demo now on the real cascade. New/edited code in `week4/`, reusing the 2A modules in place. Same style: terse, "we" voice, British spelling; standalone runnable scripts; numbers/figures tied to claims; build incrementally and **stop for review**.

The strategy is fixed by 2B: **turn power on once via DC, then select / switch / stop muscles by changing the AC frequency, leading each stop by the measured coast (plus brake where it helps), one joint at a time.**

---

## 1. Precondition — `hand_pos` access (confirm before relying on it)

The controller steers by reading the clean `hand_pos` each step (regime A). In our dev harness that is the `CascadePlant.hand_pos` property, so 2C builds and tests against that with no issue. **What must be confirmed separately:** that the *scored* harness gives the controller the same live access (the notebook's `control_policy(observations, target)` signature does not pass the hand, and its bare-brain loop exposes none). Build 2C to obtain `hand_pos` through a single, swappable accessor so that, if scoring provides it by a different channel, only that accessor changes. If scoring turns out to be regime B (observations only), the fallback is to dead-reckon the hand through the Step-1 Kalman filter + a cascade replica (drifting, conservative) — keep that seam in mind but do not build it unless needed.

---

## 2. The calibrated recipes (from 2B — `primitive_library.npz`)

| muscle | joint, sign | f | steady \|v\| (rad/step) | coast (rad) | brake factor | notes |
|---|---|---|---|---|---|---|
| sh+ | shoulder + | 0.070 | 0.065 | 1.89 | 1.5–2.5× | strong, reliable; largest coast |
| sh− | shoulder − | 0.125 | 0.038 | 0.71 | | reliable |
| el+ | elbow + | 0.205 | 0.027 | 0.58 | | reliable |
| el− | elbow − | 0.315 | 0.0085 | 0.22 | | **bottleneck: 8× slow, selector ~0.198 (just clears 0.15), 25% misfire** |

Global: cold power onset ≈ 94 steps (paid once); warm switch ≈ 50 steps; `A* ≈ 0.49`; DC = 0.5; input 1 = 0.5 (spectator). Load these from `primitive_library.npz` rather than hard-coding. Note the structural fact for the stopping design: **coast ≈ steady |v| × ~27 steps**, i.e. the joint keeps moving at roughly its current speed for ~27 steps after the AC is cut, so coast *scales with approach speed*.

---

## 3. Runtime strategy
- **Power on once, keep it hot.** Hold input-0 DC at 0.5 for the entire run — including while "stopped" (drop the AC, keep DC) — so power latches once (~94 steps at the start) and never has to re-ramp between moves. Subsequent moves pay only the ~50-step warm switch.
- **Select / switch / stop by AC frequency.** To move a joint, oscillate input 0 at the chosen muscle's `f_k` at amplitude `A*`. To switch muscles, change `f_k`. To stop, drop the AC (hold `u0 = 0.5` constant) → selectors decay → arm freezes.
- **One joint at a time** (default). Coordinated two-tone is available (2B showed clean superposition at half amplitude) but stays off unless the evaluation shows it is needed.

## 4. Reacher → input translation (the `_emit` seam from 2A)
The 2A reacher already reads the hand, recovers θ by branch-continuity IK, computes target joint angles, and picks one joint + sign (one-joint default). 2C fills its `_emit` seam:
- map (joint, sign) → muscle index → `(f_k, A*)`;
- emit `u = [0.5 + A*·cos(2π f_k·t), 0.5]` (continuous phase in the global step `t`);
- on "stop", emit `u = [0.5, 0.5]`.
Keep the wrapper emitting the *same* command type the demo/harness already pass to the plant.

## 5. Stopping — anticipatory lead, the shoulder problem, the el− watchdog — *the hard part*
- **Anticipatory lead.** Cut the AC when the moving joint is within its **coast** of the target angle, so the coast lands it on target. Lead is per muscle (the §2 coast column).
- **The shoulder big-coast problem.** sh+ coast (1.89 rad) exceeds many moves — driving at full `A*` then cutting would overshoot badly. Because **coast scales with speed**, the robust fix is a **graded approach**: drive at full `A*` while far, then *reduce the amplitude* as the joint nears its target (lower selector → lower speed → proportionally smaller coast), so the final approach is slow and lands tightly. Add the **antagonist brake** (a short burst of the opposite muscle's frequency) where coasting alone overshoots — but account for its ~warm-switch engagement lag (the brake is not instant). Tune lead/brake/graded-approach against the real arm.
- **The el− watchdog.** el− is slow and misfires ~25% of the time. When commanding el−, watch the hand: if no usable motion within the onset/switch window, **re-trigger** (drop and re-apply the AC to restart the selector build-up) rather than waiting indefinitely. Expect el− moves to take longer and occasionally need a retry; the hand feedback covers it.
- **Feedback is the safety net.** Read the hand every step; any residual over/undershoot is corrected on the next move. The lead just makes it land cleanly rather than hunting.

## 6. Decision cadence
Sense every step, but **decide per move, not per step**: pick a joint+sign, commit, drive until the stop condition, then re-evaluate from the hand and pick the next move. Do not switch muscles faster than the warm switch allows (no benefit — the selector cannot keep up). The very first move must wait out the ~94-step cold onset without thrashing; after that, moves cost ~50 steps to engage.

## 7. The policy wrapper
Finish `control_policy.py` as the stateful Week-4 object: empty first call, one-step observation lag, target-list advancement, the single swappable `hand_pos` accessor (§1), holding the current move and its phase, and `reset()`. All controller state lives inside the object.

## 8. Demo + harness on the real cascade
Swap the demo/harness plant from `IdealActuator` to `CascadePlant` — no other demo changes (2A built it swappable). The interactive click/random-target demo now drives the real arm; produce a fresh `demo_reach_cascade.mp4` of several sequential reaches. Run the harness over the same target set used in 2A (safe annulus r ∈ [5, 57]) across several `Brain` seeds; report touch-success, steps-to-touch, and overshoot, with the spread.

## 9. Files to produce (all in `week4/`)
- `control_policy.py` — completed stateful wrapper (the `_emit` seam filled, `hand_pos` accessor).
- `reach_controller.py` (or extend `reacher.py`) — the power-on-once / select-switch-stop logic, graded approach + brake, el− watchdog, per-move cadence.
- updated `demo.py` / `harness.py` — on `CascadePlant`.
- `demo_reach_cascade.mp4`, `reach_trajectories_cascade.pdf`, and a stop-timing figure (one reach showing lead/coast landing on target).
- `step2c_writeup.md` — method → result → finding: end-to-end reaches on the real cascade, the stopping design and its numbers, el− behaviour, and any failure cases.

## 10. Checkpoint — stop here
Report: touch-success / steps / overshoot across seeds on the real cascade; how the stopping (lead, graded approach, brake) performed and where it over/undershot; el− behaviour and retry rate in closed loop; whether one-joint sufficed or coordination looked needed; and the demo clip on the real arm. **Do not** start the evaluation/ablation sweep or the report — that is 2D.

## 11. Constraints recap
- Confirm/route the `hand_pos` accessor (§1); regime-B Kalman fallback only if forced.
- Power on once via DC=0.5, held all run; select/switch/stop by AC; one joint default.
- Lead each stop by the measured coast; graded approach for the big-coast shoulder; brake accounts for its engagement lag; el− watchdog re-triggers on misfire.
- Decide per move, not per step; first move waits out the cold onset.
- Steer in joint-angle space (2A); recover θ from clean `hand_pos`, never from noisy observations.
- Load recipe numbers from `primitive_library.npz`; the live arm is the source of truth.
- Standalone scripts with `__main__`; pause after each runs cleanly.
