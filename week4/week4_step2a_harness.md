# Week 4 — Step 2A: Reaching harness, kinematics, idealised actuator, live demo

*Self-contained brief for the `week4/` build. Read it fully before writing code. This is **Step 2A only** — geometry, the test harness, an idealised actuator stand-in, and the live demo. **No brain and no primitives yet** (those are 2B). Stop at the checkpoint in §10.*

---

## 0. Your task

Build the **reaching skeleton**: the arm kinematics, an evaluation harness to the Week-4 control interface, an **idealised actuator** that stands in for the real cascade so the steering logic can be proven before any brain noise, and the **live demo** front-end. Everything new and self-contained in `week4/`; do not import from the Week-3 folder, but read it for conventions (§2). Same style as before: terse, "we" voice, British spelling; runnable standalone scripts; numbers/figures tied to claims; build incrementally and **stop for review**.

Why this first: the geometry is fully known and deterministic, so it is cheap to de-risk, and it *is* the evaluation harness and the demo. We replace the idealised actuator with the calibrated primitives + real cascade in 2C.

---

## 1. What we are building (and the control idea it serves)

The controller will be a **feedback reacher**: each step, read the hand, compute the hand→target error, decide which joint(s) to move, and stop in time. 2A builds and tests that logic against an idealised actuator. The actuation alphabet (the "wiggle recipes") arrives in 2B; here the idealised actuator takes a **desired joint-velocity command directly**, with a tunable onset lag and coast so the stopping logic is exercised realistically.

**Default motion policy: one joint at a time** (pick the single joint that most reduces the Cartesian error, move it, re-evaluate). Also implement a **resolved-rate (coordinated two-joint)** option behind a flag, for the later comparison — but the default path is one-joint.

---

## 2. Read from source — do not guess

- **Arm constants — read from `BMI_and_Hand.py` (class `_Arm`) and hard-confirm:** link lengths 30 + 30 ⇒ reach is the disk of radius 60; FK `x = 30·cosθs + 30·cos(θs+θe)`, `y = 30·sinθs + 30·sin(θs+θe)` (θs = absolute shoulder angle, θe = elbow angle relative to the upper arm); per-joint update `net = pos − neg`, dead-zone 0.05, `speed = (net − sign(net)·0.05)/viscous` with **viscous = 7.5 (shoulder), 5 (elbow)**, then `angle += speed`. No inertia, no joint limits — a pure velocity integrator. Saturation speed at activation 1: shoulder `(1−0.05)/7.5 = 0.1267`, elbow `0.95/5 = 0.19` rad/step. **Start pose θ = (0,0) ⇒ hand (60,0) = the outer singularity.**
- **Control interface — read `week4.ipynb`:** the policy is `control_policy(observations, target) -> u` of shape `(2,)`; `observations` is the array of *past* brain outputs (empty on the first call; the current `y` is appended *after* the call, so the policy is one measurement behind). The scored run is replayed through the cascade to produce the hand trajectory.
- **Week-3 conventions — read, don't import:** `control/closed_loop.py` (the measure → filter → compute → apply loop and timing) and `control/plant.py` (the `BrainPlant`/`SimulatorPlant` surface). Mirror the plant surface so our plants are swappable.

---

## 3. Plant interface — one surface, two plants

Define a common surface so the harness, reacher, and demo are agnostic to which plant is underneath:
- `hand_pos` → clean `(2,)` hand position.
- `command(c)` → advance one step under control command `c`, where in 2A `c` is a desired joint-velocity vector (the idealised actuator) and from 2C it becomes the cascade input `u`.
- `reset()`.

Two implementations:
1. **`IdealActuator`** (this step): holds `θ`; on `command(v_des)` it ramps the realised joint velocity toward `v_des` with a **tunable onset time-constant** and, when `v_des → 0`, **coasts** down with a tunable time-constant, then integrates `θ`. Clamp realised speeds to the saturation values above. These lag/coast constants are **placeholders** to be replaced by the 2B calibration. Deterministic — no noise.
2. **`CascadePlant`** (interface only, for later): a thin wrapper around `BMI_and_Hand(Brain(seed))` exposing `hand_pos` and `command(u) → next_state(u)`. Build it now and run a small script to **confirm `hand_pos` is exposed and clean on the real cascade** (read it before/after a `next_state`, check shape `(2,)` and that it changes under drive). This is the regime-A check — it does not yet drive muscles (no primitives), just verifies the readout.

> **Regime-A caveat to record:** the Week-4 `run_closed_loop` in the notebook steps a *bare* `Brain` and exposes no hand. Note in the writeup that the *scored* harness must be confirmed (with the demonstrator) to expose `hand_pos` live; our whole design assumes it does.

---

## 4. Kinematics — *build + verify*

Implement in `kinematics.py`:
- `forward(theta) -> hand` (FK above).
- `jacobian(theta)` — `J = [[−30 s1 − 30 s12, −30 s12], [30 c1 + 30 c12, 30 c12]]` (s1=sinθs, s12=sin(θs+θe), etc.).
- `det J = 900·sin θe` — print/confirm it symbolically and numerically; **singular at θe = 0 (rim, r=60) and θe = π (centre, r=0)**.
- `inverse(hand) -> (theta_up, theta_down)` — the two analytic elbow branches; round-trip FK(IK(·)) must match to ~1e-9 everywhere reachable.
- `resolved_rate_step(theta, hand_err, ...)` — damped-least-squares Jacobian inverse, `dθ = Jᵀ(JJᵀ + λ²I)⁻¹ · v`. **Damping is mandatory** (λ≈2): at the singular home `λ=0` raises `LinAlgError`. Verify the DLS step is well-defined at θ=(0,0).

**Verify against the genuine `_Arm`:** drive `_Arm.move` with known activations and confirm our FK and our forward integration reproduce its `hand_pos` to ~0 — our kinematics must match the real arm exactly (it is deterministic).

**Singularity map (figure):** plot `det J`, `σ_min(J)`, and `cond(J)` across the reachable disk; mark the inner/outer singular loci and a safe working annulus (e.g. r ∈ [5, 57]). **Must-capture:** the singular loci, the home pose sitting on the outer one, and the chosen margins.

---

## 5. Reacher + stopping — *build + test on the idealised actuator*

`reacher.py` — outer loop (default one-joint, resolved-rate optional):
- **One-joint policy:** from the hand error and `J`, pick the single joint whose motion most reduces the Cartesian distance; command that joint's velocity (sign from the error projection).
- **Resolved-rate option:** DLS step to a desired joint-velocity vector.
- **Direction-preserving speed limit (important):** scale the *whole* velocity vector by one scalar to respect the saturation limit — do **not** clamp joints independently. (Per-joint clamping distorts the commanded direction and can push the hand *away* from the target near limits; a single scalar limit keeps every step pointing down-error.)
- **Stopping rule:** anticipatory lead plus an optional antagonist brake — stop driving (and optionally command the opposite joint briefly) early enough that the actuator's coast lands on, or controlled-passes through, the target tolerance. The lead distance comes from the actuator's coast model (placeholder now; calibrated in 2B). Start with **lead only**; add the brake only if you see overshoot.

Test against `IdealActuator`: a fixed set of targets in the safe annulus (≥12, spread around the disk, including one near the rim and one near the centre). **Must-capture:** touch-success (min hand-target distance < tolerance) and steps-to-touch per target; confirm reaches are monotone / non-diverging under the direction-preserving limit; show the lead/coast lands within tolerance without overshoot.

---

## 6. The policy wrapper — *build*

`control_policy.py` — wrap the reacher as the stateful Week-4 object: `control_policy(observations, target) -> u`. Handle the **empty first call**, the **one-step lag**, advancing through a target list, and holding the current move. In 2A its output is the idealised actuator command; the same wrapper will emit `u` in 2C (keep the seam clean). Keep all controller state inside the object; expose `reset()`.

---

## 7. Harness + metrics — *build*

`harness.py` — run a controller against a plant over a target sequence; log hand trajectory, commands, and per-target outcome; compute the two metrics that matter (**touched within tolerance?**, **steps-to-touch**) plus overshoot and path length. Produce the **by-eye trajectory plot** (hand path over the disk with targets marked). Tolerance is a parameter (default ~2.0 units out of radius 60); state it.

## 8. Live demo — *build* (the deliverable you asked for)

`demo.py` — an interactive matplotlib window:
- Draws the reachable circle and the two-link arm; animates the arm as the controller runs (`FuncAnimation`, one cascade/actuator step per frame).
- **Set a target by clicking** inside the circle (`mpl_connect`), and a **"random target" button** (`matplotlib.widgets.Button`) that drops a new reachable point; when one target is reached, you can set another and it moves there. A small text readout shows reached? and steps-to-touch.
- Runs against `IdealActuator` now (so it is fully testable without the brain); the plant is swappable to `CascadePlant` in 2C with no demo changes.
- Provide a **headless record mode** that runs a scripted sequence of random targets and saves an **MP4/GIF** for the report and presentation (standard matplotlib only — no extra dependencies).

**Must-capture:** a saved demo clip reaching a few random targets in sequence against the idealised actuator.

## 9. Files to produce (all in `week4/`)
- `kinematics.py` — FK, Jacobian, det J, analytic IK (both branches), DLS step, singularity helpers.
- `ideal_actuator.py` — `IdealActuator` (lag + coast + integrate) on the common plant surface.
- `cascade_plant.py` — `CascadePlant` wrapping `BMI_and_Hand`; + the small `hand_pos` regime-A check script.
- `reacher.py` — one-joint (default) and resolved-rate reacher + the stopping rule.
- `control_policy.py` — the stateful Week-4 wrapper.
- `harness.py` — target-sequence runner, metrics, by-eye trajectory plot.
- `demo.py` — interactive live demo + headless record-to-clip mode.
- `singularity_map.pdf`, `reach_trajectories.pdf`, and the demo clip.
- `step2a_writeup.md` — method → result → finding, numbers and figures embedded.

## 10. Checkpoint — stop here
Report: the kinematics verified against `_Arm` (round-trip + FK match), the singularity map and chosen annulus/margins, the reacher clearing the target set on the idealised actuator (success and steps, monotone reaches), the confirmation that `hand_pos` reads clean on the real cascade, and the working demo clip. **Do not** start the primitive library or touch `u`/the brain — that is 2B.

## 11. Constraints recap
- Self-contained `week4/`; no imports from Week-3 (reading it is encouraged).
- Standalone runnable scripts with `__main__`; print must-capture numbers; save figures/clips as files.
- One-joint motion is the default; resolved-rate is an option behind a flag.
- Direction-preserving scalar speed limit (never per-joint clamp); DLS damping mandatory.
- Idealised-actuator lag/coast are placeholders, replaced by 2B calibration; keep the plant surface swappable.
- Build incrementally and pause after each script runs cleanly.
