# Week 4 — Step 2C: closing the loop on the real cascade

*Self-contained build in `week4/`. We wire the 2A reacher to the real
`BMI_and_Hand` cascade through the 2B-calibrated primitives and reach a sequence
of targets end-to-end under hand feedback, with the live demo now on the real arm.
The controller steers by the clean `hand_pos` each step (regime A), recovered to
joint angles by 2A branch-continuity IK. **No evaluation/ablation sweep or report
yet** (that is 2D). Run on the GG4 venv python.*

---

## 1. Strategy (fixed by 2B)

**Turn power on once, then select / switch / stop muscles by changing the AC
frequency, one joint at a time.** Input-0 DC is held at 0.5 the entire run
(idle/stop included), so the shared power latches once (~94 steps) and never
re-ramps; a joint is moved by oscillating input 0 at its band `f_k` at amplitude
`A`, switched by changing `f_k`, and stopped by dropping the AC (`u=[0.5,0.5]`).
Recipe numbers are loaded from `primitive_library.npz` — never hard-coded.

A new layer, **`ReachController`**, sits between the hand and the cascade input.
The cascade cannot follow a per-step velocity command (the 2A path), so the
controller reuses the 2A `Reacher` only for **IK tracking + target-joint-angle
selection** (`Reacher.decide`) and owns the power-on-once / select-switch-stop
state machine. `control_policy.py` routes to it in cascade mode through a single
swappable `hand_pos` accessor (the one place to change if scoring delivers the
hand differently; the regime-B Kalman fallback is noted, not built).

## 2. The stopping design (the hard part)

The shoulder is fast (0.065 rad/step) with a **1.89 rad coast** — at r=30 that is
~57 units, far more than the 2-unit tolerance, so cutting at full speed overshoots
massively. We measured the live **speed-vs-amplitude** curve of each muscle and
built the stop from it:

- **Matched-coast graded approach.** Coast ≈ speed × ~24 steps, so we pick the
  amplitude whose coast equals the remaining joint angle (`desired_v =
  dgo/coast_steps`): the joint decelerates as it approaches and its coast always
  matches what is left. The shoulder grades down to A≈0.10 (coast ~3 units); the
  **elbow muscles have a sharp firing threshold** (el+ dies below A≈0.33), so they
  grade only within [0.35, 0.49] and otherwise run at `A*`.
- **Cartesian slow-down.** A fast move can reach Cartesian tolerance while a joint
  is still mid-stroke (the other joint compensating); we additionally scale every
  drive down as the *hand* nears the target, so little coast is left at the touch.
- **Live-coast lead + capture hysteresis.** Cut the AC when the remaining angle is
  within the smoothed live-coast estimate; once inside tolerance, hold (rest)
  within a capture band so the controller does not hunt.
- **el− watchdog.** el− is the bottleneck (selector ~0.198, ~25% misfire); if a
  move makes no motion within its onset/warm window, re-trigger (drop+reapply AC).
- **Decide per move, not per step.** Commit to the joint with the larger remaining
  IK joint-error, drive it (graded), then wait for its coast to settle before
  picking the next joint — never re-driving mid-coast or switching faster than the
  ~50-step warm switch allows.

![one cascade reach: distance + drive](stop_timing.png)

## 3. Results — end-to-end reaches on the real cascade

`harness.py --cascade --seeds 4`: isolated reaches from home over the 2A
safe-annulus target set (14 targets), per Brain seed. Each reach rebuilds a fresh
cold-power Brain, so it pays the ~94-step cold onset once.

| metric | value |
|---|---|
| **touch-success** | **52/56 = 93 %** (4 seeds × 14 targets) |
| steps-to-touch | mean **411** [197, 876] (includes the ~94-step cold onset) |
| overshoot (drift after closest approach) | mean **10.2**, max 33.7 units |

![cascade reach trajectories](reach_trajectories_cascade.png)

**Findings:**
- **One joint at a time sufficed** — 93 % of reaches touch within tolerance on the
  real arm under hand feedback. Coordinated two-tone (2B showed clean superposition)
  was left off and not needed.
- **Where it over/undershot.** Touch-success is high, but the **shoulder's large
  coast** still produces a moderate post-touch drift (overshoot mean ~10 units):
  on a fast shoulder-dominated reach the hand sweeps through the tolerance disk and
  coasts past before the graded approach + lead can fully arrest it; feedback then
  pulls it back. The graded approach and Cartesian slow-down cut this from ~25–40
  units (ungraded) to ~10, but it does not vanish — landing a 0.065 rad/step joint
  with a ~24-step coast inside a 0.07-rad window is at the edge of what the
  fixed-speed muscle allows. This is the honest limitation of 2C's stopping.
- **el− behaviour.** el− is slow and misfires; the one consistently hard target,
  **(7, −19) at r=20 (1/4 seeds)**, needs an elbow-extension component that el−
  often fails to deliver within the window. The watchdog re-triggers help but
  cannot make el− reliable; reaches dominated by el− are the slowest and least
  reliable — exactly the 2B bottleneck, now seen closed-loop.
- **Cold onset is paid once.** The ~94-step first-move latency dominates the
  steps-to-touch on short reaches; in a *sequential* session (the demo) only the
  first reach pays it, the rest engage on the ~50-step warm switch.

## 4. The live demo on the real arm

`demo.py --plant cascade` swaps `IdealActuator → CascadePlant` with no other demo
changes (2A built it swappable); the interactive click/random-target demo now
drives the real cascade. **`demo_reach_cascade.mp4`** records four sequential
random reaches on the real arm (power held hot across targets — only the first
pays the cold onset).

## Files
`reach_controller.py` (new), `control_policy.py` (cascade routing + hand accessor),
`harness.py` / `demo.py` (on `CascadePlant`); artefacts `reach_trajectories_cascade.pdf`,
`stop_timing.pdf`, `demo_reach_cascade.mp4`.

## What 2C leaves to 2D
The evaluation/ablation sweep and the report: robustness across more seeds/noise,
the identified-vs-true controller comparison (begun in 2B's feasibility note),
and tuning trade-offs (e.g. whether a faster but looser stop, or coordinated
two-tone, is worth it). 2C delivers the working closed loop; 2D evaluates it.
