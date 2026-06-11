# Week 4 — Step 2A: reaching skeleton, idealised actuator, live demo

*Self-contained build in `week4/`. This step delivers the geometry, the evaluation
harness, an idealised actuator stand-in, and the live demo — **no brain control and
no primitives yet** (that is 2B). Everything is re-implemented cleanly; we read the
Week-3 control package for conventions but import nothing from it.*

All scripts are standalone (`python <file>.py`) and run on the project venv
(`venv/Scripts/python.exe`), the only environment with the `GG4` brain installed.

---

## 1. Kinematics — built and verified against the genuine arm

**Method.** We re-implement the planar two-link arm in `kinematics.py`: forward
kinematics, the hand Jacobian, the analytic two-branch inverse, and a
damped-least-squares (DLS) resolved-rate step. Link lengths and the velocity-
integrator rule are read from `_Arm` in `BMI_and_Hand.py` (links 30+30; per-joint
`speed = (net − sign(net)·0.05)/viscous`, viscous 7.5 shoulder / 5 elbow).

**Result.**

| check | value |
|---|---|
| `det J` vs `900·sin θe` (home / mid / centre) | exact: 0.0 / 900.0 / 0.0 |
| home pose θ=(0,0) → hand | (60, 0) — the outer singular rim |
| IK round-trip `max‖forward(inverse(·)) − hand‖`, both branches | **1.0 × 10⁻¹³** |
| FK vs genuine `_Arm` (2000 random poses) | **0.0** |
| forward integration vs `_Arm.move` (random drives) | **0.0** |
| DLS step at the singular home (λ=2) | finite `[0, 0]`; λ=0 raises `LinAlgError` |

**Finding.** Our kinematics reproduce the real arm *exactly* (deterministic, no
inertia), so the geometry is not a source of error downstream. The home pose sits
on the outer singularity, and DLS damping (λ≈2) is mandatory — without it the home
Jacobian is rank-deficient and the plain inverse throws.

---

## 2. Singularity map and the safe working annulus

**Method.** `singularity_map.py`. Because changing the absolute shoulder angle
merely rotates the arm, the singular values of `J` are rotation-invariant — `det J`,
`σ_min(J)` and `cond(J)` depend only on the elbow angle, i.e. only on the hand
radius `r = 60·|cos(θe/2)|`. The conditioning map is therefore radially symmetric.

![singularity map](singularity_map.png)

**Result.**

| r | cond(J) | σ_min(J) |
|---:|---:|---:|
| 0.5 (near centre) | 60.0 | 0.50 |
| **5 (inner margin)** | **6.0** | **4.98** |
| 20–30 (best) | ≈1.7 | ≈18–21 |
| **57 (outer margin)** | **7.6** | **8.36** |
| 60 (home / rim) | ∞ | 0 |

**Finding.** The arm is singular at the rim (`r=60`, elbow straight) and the centre
(`r=0`, elbow folded). We confine targets to a **safe annulus `r ∈ [5, 57]`**, where
`cond(J) ≤ ~7.6` throughout — comfortably away from both singular loci while still
covering most of the workspace.

---

## 3. The idealised actuator

**Method.** `ideal_actuator.py` implements the common plant surface
(`hand_pos` / `command(c)` / `reset()`). It takes a **desired joint-velocity**
command directly (no muscles), relaxes the realised velocity toward it with a
first-order **onset lag** (`tau_on`), **coasts** to zero when released
(`tau_coast`), clamps to the saturation speeds, and integrates the angle. The
lag/coast constants are **placeholders**, to be replaced by the 2B calibration.

**Result.** Saturation speeds `[0.1267, 0.19]` rad/step (shoulder, elbow). With the
placeholder `tau_on=3`, the velocity reaches 95 % of saturation in ~9 steps. With
`tau_coast=4`, releasing from full elbow speed coasts a further 0.645 rad ≈
**19.4 units** of hand travel — the coast model matches the integrated coast to
2 × 10⁻¹⁵.

**Finding.** The actuator is non-instantaneous by design: its onset lag and coast
are what the reacher's stopping logic must anticipate, and they stand in for the
real cascade's slow onset (§5).

---

## 4. The reacher, the wrapper, and the harness

**Method.** `reacher.py` is the feedback reacher; `control_policy.py` wraps it as
the Week-4 callable `control_policy(observations, target) → u`; `harness.py` runs
it against a plant over a target set and scores it.

Design choices (per the brief):
- **Plant-agnostic sensing.** The reacher observes only `hand_pos` and recovers the
  joint angles by **branch-continuity inverse kinematics** (start at home, pick the
  IK branch nearest the previous estimate). The same code therefore drives the
  IdealActuator now and the real cascade later.
- **Steering toward the IK target.** Pure Cartesian feedback **stalls at the singular
  home**: there `J = [[0,0],[60,30]]`, so any joint motion moves the hand only
  tangentially — a radially-inward target lies in the null space and the hand never
  moves. We therefore steer toward the **IK target joint angles** (joint-space
  control has no singularity), selecting in one-joint mode the joint that most
  reduces the Cartesian distance and falling back to the larger joint backlog when
  no joint makes first-order Cartesian progress (the around-a-singularity case).
- **One joint at a time** is the default; **resolved-rate** (DLS, coordinated
  two-joint) is available behind `--mode resolved_rate`.
- **Direction-preserving scalar speed limit** — the whole velocity vector is scaled
  by one factor so the binding joint hits saturation; we never clamp joints
  independently (that distorts the commanded direction near a limit).
- **Stopping: anticipatory lead only.** The reacher estimates its realised joint
  velocity (from the tracked-angle change) and, using the actuator's coast model,
  releases the command once the predicted coast covers the remaining travel. The
  wrapper then keeps closed-loop hold within tolerance, pulling back any coast that
  drifts out. **No antagonist brake was needed** — overshoot stayed within tolerance.

**Result (isolated reaches from home, 14 targets in the safe annulus, tol = 2.0).**

![reach trajectories](reach_trajectories.png)

| policy | touched | steps-to-touch (mean / min / max) | max overshoot |
|---|---|---|---|
| one-joint (default) | **14 / 14** | 29.4 / 13 / 45 | **1.68 < tol** |
| resolved-rate | 14 / 14 | 33.0 / 10 / 51 | within tol |

Every target — including one near the rim (`r=56.5`) and one near the centre
(`r=6`) — is touched, with the closest approach ≤ 1.0 unit for all but two targets.
Reaches are **non-diverging** and settle within tolerance. Cartesian distance is *not*
monotone for around-the-singularity reaches (the hand must swing out before coming
in), which is expected and visible in the trajectory plot.

A short lead release with a closed-loop hold was sufficient: tuning showed a
*shorter* lead actually *worsens* overshoot (higher velocity at release → longer
coast), so `lead_gain = 1.0` with a gentle approach gain (`kp_joint = 0.45`) is the
clean operating point.

**Finding.** The steering logic clears the workspace reliably against a realistic
(lagged, coasting) actuator, with overshoot held inside the tolerance band by
anticipatory release plus closed-loop hold — no brake required.

---

## 5. The real cascade reads clean (regime-A check)

**Method.** `cascade_plant.py` wraps the genuine `BMI_and_Hand(Brain(seed))` on the
same plant surface and runs the regime-A check: confirm `hand_pos` is exposed,
clean, and changes under drive. (We do **not** yet drive muscles toward a goal — the
actuation alphabet is 2B.)

**Result.** `hand_pos` is a clean `(2,)` float64, finite at every step. A **single**
command moves the hand by ~0; a **sustained** 150-step drive (`u=[1,1]`) moves it
**21.5 units**, staying clean throughout.

**Finding.** The cascade has a slow, accumulating onset (the muscle head is a
stateful rhythm/power decoder, and the arm is a velocity integrator): one command
does nothing, sustained commands ramp the hand. This is exactly the behaviour the
IdealActuator's onset lag (`tau_on`) stands in for, and it tells us 2B's primitives
must be *sustained, shaped* drives rather than single pushes.

> **Regime-A caveat (recorded).** The Week-4 notebook's `run_closed_loop` steps a
> *bare* `Brain` and exposes no hand. Our whole design assumes the *scored* harness
> exposes `hand_pos` live (as `BMI_and_Hand` does) — **to be confirmed with the
> demonstrator.**

---

## 6. The live demo

**Method.** `demo.py` — an interactive matplotlib window drawing the reachable
circle and the two-link arm, animated by `FuncAnimation` (one actuator step per
frame). **Click inside the circle** to set a target; a **"Random target"** button
drops a new reachable point; a readout shows the target, reached?, and
steps-to-touch. A **headless record mode** (`--record out.mp4`) runs a scripted
sequence of random reaches and saves an MP4 (`FFMpegWriter`). The demo talks only to
the plant surface and the policy wrapper, and draws the arm from the reacher's
tracked angles — so it is **swappable to the cascade in 2C with no demo changes**.

**Result.** `demo.py --record demo_reach.mp4 --n 4` produces a 16 s clip
(`demo_reach.mp4`, 485 frames) of the arm reaching four random targets in sequence
against the IdealActuator. Interactive mode shares the same verified step/draw core.

**Finding.** The reaching skeleton is demonstrable end-to-end before any brain is in
the loop — the deliverable the report and presentation need.

---

## Files
`kinematics.py`, `ideal_actuator.py`, `cascade_plant.py`, `reacher.py`,
`control_policy.py`, `harness.py`, `singularity_map.py`, `demo.py`; artefacts
`singularity_map.pdf`, `reach_trajectories.pdf`, `demo_reach.mp4`.

## What 2A deliberately leaves to 2B / 2C
The primitive library (the actuation alphabet / "wiggle recipes"), the calibration
that replaces `tau_on` / `tau_coast`, and driving the cascade input `u`. The seam is
kept clean: `ReachingPolicy._emit` returns the joint-velocity command in 2A and will
translate it into `u` in 2C with no other change to the wrapper.
