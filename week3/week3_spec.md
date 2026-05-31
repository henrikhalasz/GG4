# Week 3 — Control: Build & Test Specification v2 (for Claude Code)

This supersedes any earlier `week3.md`. Read it **in full** before writing code, then build
milestone by milestone (M0→M5), testing at each step. Two design decisions changed since the
first build and are now **binding**:

- The estimator is **rebuilt from scratch** as `estimator_new.py` (affine model, EM with known
  inputs). The Week-2 `estimator.py` is left untouched as a separate, graded artifact.
- **No dither and no online matrix updating.** The system is time-invariant: identify once,
  offline, then control. (If you later detect drift you'd revisit this — but check first, don't
  assume.)

---

## 0. PASTE-IN PROMPT

> Read `week3_spec.md` in full, plus `Simulator.py`, `Illustrator.py`, `week3.ipynb` (for the
> Brain interface and the assessed deliverables), and the Week-2 `estimator.py` (reference only —
> do not edit or import it). Then build and test the Week-3 control system exactly as specified,
> milestone by milestone (M0→M5). After **each** milestone: run its tests, save the figures it
> names to `results/`, append a findings entry to `RESULTS.md`, and stop to report what worked,
> what didn't, and the numbers — then continue. Hard rules: (1) build a **fresh** `estimator_new.py`
> (affine LGSSM, EM, known inputs); do not edit or import the Week-2 `estimator.py`; (2) only
> `control_interface.py` imports the estimator; controllers, observer, reachability, and driver
> consume only the model params; (3) **no dither, no online re-identification**; (4) verify the
> control math against **known** ground-truth models before trusting the estimator (Stage A vs B);
> (5) report honest negative results — saturation, near-floor suppression, instability are findings
> to document, not to hide or fake. Ask me before installing anything beyond
> numpy/scipy/matplotlib (QP solver: prefer `scipy.optimize`, optionally `osqp`/`cvxpy`).

---

## 1. Mission & mapping to the assessed notebook (`week3.ipynb`)

Design, implement, and rigorously evaluate a closed-loop controller for the black-box Brain,
with a freshly built known-input estimator. Every notebook task must be satisfied:

| `week3.ipynb` task | Satisfied by |
|---|---|
| 1. Define a control objective & justify | **Primary: drive a neural-population readout to a target level and hold it** (closed-loop targeted neuromodulation). Justified by reachability (§7). |
| 2. Design strategy; compare approaches; justify final | 5 controllers compared (§6); MPC justified as final via the study (§8). |
| 3. Implement, integrating Week-1 sim & Week-2 estimation | `SimulatorPlant` wraps the Week-1 `Simulator`; `estimator_new.py` is the adapted Week-2 pipeline (§5). |
| 4. Test in closed loop (continuous interaction) | `closed_loop.py` driver on Simulator scenarios and the Brain (§6, §8). |
| 5. Evaluate: stability, responsiveness, effort, robustness, sensitivity | Metrics + sweeps + figures (§8). |
| 6. Show working closed loop, perf plots (uncontrolled vs controlled, stability), identified issues, an attempted improvement | Figures 1–10 + documented failure modes + MPC-over-LQG and the estimator rebuild as the improvement (§8, §11). |

**Final deliverable is a runnable notebook/script** (`week3_solution.ipynb` or
`run_week3.py`, callable from the notebook) that, on the Brain, does the whole story end to end:
install/load Brain → probe → fit `estimator_new` → validate → closed-loop control → figures +
narrative.

---

## 2. Operating rules (non-negotiable)

1. **Fresh estimator.** Build `estimator_new.py` from scratch. Do **not** edit or import the
   Week-2 `estimator.py`; you may read it for reference (e.g. its Hankel/SVD warm start).
2. **Modularity.** Only `control_interface.py` imports the estimator. Controllers, observer,
   reachability, driver depend solely on the model params + the plant interface. Any controller
   swappable without touching identification, and vice versa.
3. **No dither, no online updating.** One offline probe-and-fit; fixed model during control.
4. **Ground-truth first.** Validate control math on **known** models (a Simulator scenario's
   *true* matrices) before plugging in the estimator (Stage A vs B, §8).
5. **Test as you go.** A milestone is done only when its tests pass and figures are saved.
6. **Honesty.** Document saturation, near-floor suppression, instability, degenerate fits.
   Never fabricate or tune away a real negative result.

---

## 3. Environment & files

- **`week3.ipynb`** — Brain interface + assessed deliverables + wheel install (`wheels/`, Py 3.11–3.13).
- **`Simulator.py`** — white-box systems with ground truth (verification bench).
- **`Illustrator.py`** — plotting helpers; reuse for observation plots.
- **`estimator.py`** (Week-2) — reference only.

### Brain interface (confirmed)
```python
from GG4 import Brain
brain = Brain(random_seed=SEED)         # each seed = a different instance (multi-seed eval)
brain.input_dim                          # == 2
brain.current_time_stamp                 # int, read-only
brain.measure()                          # -> list[float], shape (16,), obs of CURRENT state
brain.next_state(u)                      # apply u (shape (2,)), advance; None => no input
```
Inputs are clipped internally to `[0,1]` per channel. **Resting input is 0**, not 0.5.

### Simulator (confirmed)
Stateless/batch: `Simulator(A,B,C,Q,R,x0,seed)`, `.step(x,u)`, `.simulate(T,U)->dict`. Factories
(use the `obs_dim=16` ones to match the Brain): `default_neural_system`, `input_aligned_system`,
`input_blind_system`, `slow_drift_system`, `closed_loop_system`; plus `non_normal_system`(8),
`hidden_input_system`(10). Helpers: `controllability_matrix`, `observability_matrix`,
`matrix_rank`. **The Simulator does not clip** — your plant wrapper clips to `[0,1]`.

---

## 4. The model & conventions

**Affine controlled LGSSM, fit and used in TRUE input units** (no input centering):
```
x_{t+1} = A x_t + B u_t + a + w_t,   w ~ N(0,Q)      # u_t in [0,1]^2, resting u = 0
y_t     = C x_t + c + v_t,           v ~ N(0,R)
```
- `a` absorbs the constant drive from a non-negative probe; `c` absorbs the observation baseline.
- **Do not center the input.** Centering secretly redefines the resting input to ~0.5, which
  reintroduces the bias-point bug (a constant 0.5 push just injects energy). The offsets `a,c`
  represent the baseline honestly while keeping `u=0` = "off", `u=1` = "max".
- **Timing:** input at `t` affects state at `t+1`; feed `u_prev` into the predict step.
- **Single basis:** params, `x_hat`, gains, feedforward all from the same fit (similarity-transform
  ambiguity cancels). Never mix fits.
- **Observation offset in the filter:** innovation is `y_t - C x⁻ - c`.

---

## 5. `estimator_new.py` — probe-and-fit (the rebuild)

### 5.1 Probe (Phase A data collection)
Drive the plant and record `{(u_t, y_t)}`:
```
for t in range(T_cal):
    y[t] = plant.measure(); u[t] = Uniform(0,1, size=2); plant.next_state(u[t])
```
Probe = i.i.d. `Uniform[0,1]` per channel (broadband, persistently exciting; both channels
independent so `B` is identifiable). Choose `T_cal` by the held-out one-step-error knee
(§5.4); expect a few hundred to ~1000.

### 5.2 Warm start
Rough init from a Hankel/SVD output subspace fit (reuse the idea from the Week-2 estimator):
singular spectrum → latent dim `n`; leading vectors → rough `A,C`; init `B,a,c` by regression on
the warm-start states. EM is forgiving of init, so identity-ish fallbacks are acceptable.

### 5.3 EM with known inputs (the fit)
Alternate until the log-likelihood increment `< tol`:

**E-step** — Kalman filter + RTS smoother over the batch, **using known `u_t` and the offsets**:
```
predict:  x⁻ = A x̂ + B u_prev + a ;  P⁻ = A P Aᵀ + Q
update:   innov = y - C x⁻ - c ;  S = C P⁻ Cᵀ + R ;  Kf = P⁻ Cᵀ S⁻¹
          x̂ = x⁻ + Kf innov ;  P = (I - Kf C) P⁻
smoother: RTS backward pass -> smoothed mean x_t^s, cov P_t^s, lag-one cov P_{t,t-1}^s
```
Sufficient statistics: `E[x_t]=x_t^s`, `E[x_t x_tᵀ]=P_t^s + x_t^s x_t^sᵀ`,
`E[x_t x_{t-1}ᵀ]=P_{t,t-1}^s + x_t^s x_{t-1}^sᵀ`.

**M-step** — closed-form regressions; the input and the constant are **known** regressors, which
is what pins `B` to true scale/sign:
```
dynamics:  regress x_{t+1} on r_t = [x_t ; u_t ; 1]  ->  [A B a] = (Σ E[x_{t+1} r_tᵀ]) (Σ E[r_t r_tᵀ])^{-1}
           Q = (1/(T-1)) Σ ( E[x_{t+1}x_{t+1}ᵀ] - [A B a] E[r_t x_{t+1}ᵀ] ),   symmetrise
obs:       regress y_t on s_t = [x_t ; 1]            ->  [C c] = (Σ y_t E[s_tᵀ]) (Σ E[s_t s_tᵀ])^{-1}
           R = (1/T) Σ ( y_t y_tᵀ - [C c] E[s_t] y_tᵀ ),                        symmetrise
```
(In `r_t r_tᵀ` and `s_t s_tᵀ`, the `u` and `1` blocks are deterministic; only `x` blocks use the
smoothed second moments.)

### 5.4 Validation checks (run on Simulator scenarios with ground truth)
In increasing order of what they catch:
1. **One-step prediction RMS** on held-out data (basic predictive check; also the `T_cal`/`n` knee).
2. **State-tracking R²** after aligning estimated states to true states by a least-squares linear
   map (states are basis-ambiguous → align, then score).
3. **`G`-matrix / zonotope check (the decisive one):** compute `G = C(I-A)^{-1}B` and the offset
   `z0 = C(I-A)^{-1}a + c` from the fit; compare to the true `G, z0`. `G` is exactly the
   held-input→steady-output map the controller relies on, so a matching `G` certifies the
   controller will behave on the identified model as it did on the true model. This is the check
   that catches the inert-controller bug.
4. **EM sanity:** log-likelihood is non-decreasing across iterations; flag non-finite or unstable
   `A` (spectral radius ≥ 1 when the truth is stable).

### 5.5 Interface (the contract `control_interface.py` consumes)
```python
class EstimatorNew:
    def fit(self, Y, U, n=None) -> None          # warm start + EM; selects n if None
    @property
    def params(self): return A, B, C, Q, R, a, c
    def loglik_history(self) -> list[float]
    def validate(self, true_model=None) -> dict   # the §5.4 checks
```

---

## 6. Control layer

`control/` package; only `control_interface.py` imports the estimator.
```
control_interface.py  IdentifiedSystem: run probe, fit EstimatorNew, expose params + observer
observer.py           steady-state Kalman filter (affine): filter_step(y, u_prev) -> x_hat
controllers.py        OpenLoop, ProportionalFeedback, LQG, MPC, PolePlacement
reachability.py        affine zonotope, feasibility test, G-check, finite-horizon set
plant.py              BrainPlant, SimulatorPlant (common measure()/next_state()(clips)/true_state())
closed_loop.py        run_closed_loop(plant, controller, iface, T, ref_fn) -> logs
probes.py  metrics.py
```

**Affine forms the controllers must use:**
- Equilibrium under constant `u`: `x_ss = (I-A)^{-1}(B u + a)`; readout `z_ss = M(C x_ss + c)`.
- Reachability map: `z_ss(u) = G u + z0`, `G = MC(I-A)^{-1}B`, `z0 = M(C(I-A)^{-1}a + c)`.
- Feedforward for target `r`: solve `[[I-A, -B],[MC, 0]] [x_ss; u_ff] = [a ; r - M c]`, check `u_ff∈[0,1]`.
- **LQR:** `K` from DARE on `(A,B,Q_x,R_u)` (offsets don't affect `K`); law
  `u = clip(u_ff - K(x_hat - x_ss), 0, 1)`. `Q_x=(MC)ᵀ(MC)`, `R_u=ρI`.
- **MPC:** condensed box-QP with dynamics `x_{k+1}=Ax_k+Bu_k+a`, `z_k=M(Cx_k+c)`, `0≤u_k≤1`,
  terminal cost = LQR `P`, horizon `H∈[10,30]`; apply `u_0`, recede.
- **ProportionalFeedback:** `u = clip(u0 - Kp·M(y - r), 0, 1)`, model-free baseline.
- **PolePlacement:** place `eig(A-BK)`; note it needs the `u_ff` feedforward for setpoints
  (it has no inherent steady-state-offset handling).

---

## 7. Objectives (primary + comparison)

Readout `z = M y` with `dim z ≤ m = 2` (default: top-2 observation principal directions; document
the choice). Three references on one controller:

- **Primary — setpoint regulation** (`r = r*`): drive `z` to a target level and hold it. Motivated
  as targeted neuromodulation. This is the objective the one-sided actuator best supports.
- **Comparison — suppression** (`r=0`) and **tracking** (`r=r(t)`).

**Stabilisation is not an objective here:** the systems are already stable (eigenvalues inside the
unit circle), so there is nothing to stabilise — state this explicitly.

The reachability analysis (§6) is itself a headline result: it predicts which targets are feasible
(inside the affine zonotope) before any closed-loop run, and the controller is shown to hit
feasible targets and land on the nearest reachable point for infeasible ones. This is the rigorous
form of "which objective the actuator supports."

---

## 8. Evaluation

**Two tiers.** *Verify* on white-box Simulator scenarios (ground truth → state R², true-`G`
comparison); *evaluate* on the **Brain across ≥5 seeds** (real target; multi-seed = robustness).

**Stage A vs Stage B (the key regression test).** For each scenario: Stage A feeds the controller
the scenario's *true* params; Stage B feeds `estimator_new`'s fitted params. With the rebuild,
**Stage B must now match Stage A on supported objectives** — concretely, Stage-B setpoint error
within ~1.5× of Stage A, `u` not identically zero, and not worse than open-loop. (This is exactly
what failed before; it is the test that proves the rebuild fixed it.)

**Metrics:** setpoint/tracking RMS error; settling time; suppression RMS; control effort `Σ‖u‖`;
saturation fraction; closed-loop spectral radius; divergence/NaN checks; white-box state R².

**Experiment matrix:** all 5 controllers on the three objectives (suppression, feasible setpoint,
infeasible setpoint, tracking); reachability per objective; sweeps over noise scale (×Q,×R),
`T_cal`, `n`, `ρ`, `H`.

**Figures → `results/`:**
1. Uncontrolled vs controlled `y(t)` + readout vs reference — **Stage A and Stage B both working** (headline; proves the fix).
2. `u(t)` with `[0,1]` band + saturation shading.
3. Affine reachable zonotope with feasible & infeasible targets (predicted vs achieved).
4. Per-objective achieved error across controllers.
5. Suppression error-vs-effort.
6. Robustness vs noise scale (LQG vs MPC).
7. Sensitivity sweeps (`T_cal`, `n`, `ρ`, `H`).
8. **EM validation:** log-likelihood convergence + fitted-vs-true `G` + state R² (replaces the old dither figure).
9. State R² per scenario.
10. Convergence/settling on the primary objective.
Plus a Brain multi-seed setpoint-performance figure.

---

## 9. Tests (rigorous; `tests/`)

**Control math on a known synthetic affine LGSSM (you set all params):**
- observer recovers state (high R²); steady-state gain finite & stable.
- LQR `K` gives closed-loop spectral radius `< 1`; suppresses on an oscillatory known system.
- MPC: zero `[0,1]` violations; equals LQR when constraints inactive.
- reachability: fitted `G,z0` equal analytic; membership test correct; feasible setpoint reached,
  infeasible not; feedforward holds a feasible setpoint at steady state.

**Estimator (`estimator_new`) on synthetic + Simulator data:**
- EM log-likelihood non-decreasing every iteration.
- On data from a known affine LGSSM: state R² high after alignment; fitted `G,z0` match true within
  tolerance; offsets `a,c` recovered; `B` scale/sign correct (via `G`).
- one-step prediction error low on held-out data; warm start improves under EM.
- degenerate detection fires on non-finite/unstable fits.

**Integration:**
- Stage A control works on each scenario (setpoint reached; suppression at predicted floor).
- **Stage B matches Stage A** on supported objectives (the regression test above) — `u` not all-zero.
- Brain multi-seed: closed loop runs without NaN; setpoint control beats open-loop on a majority of seeds.

**Diagnostic (documents the old failure signature):** feed true params but with `B` sign-flipped →
controller goes inert (`u≡0`) → confirms one-sided clipping + wrong-`B` is the inert-controller
mechanism.

---

## 10. Milestones & acceptance

- **M0 — Discovery & scaffold.** Read files; confirm interfaces; install Brain; create package +
  venv. *Accept:* interface summary in `RESULTS.md`; skeleton imports; Brain `measure/next_state` work.
- **M1 — `estimator_new.py`.** Probe, warm start, EM, validation. *Accept:* on `default`, EM
  log-lik non-decreasing; state R² high; fitted `G` matches true `G` within tolerance; figure 8 saved.
- **M2 — Control math, Stage A.** plant/observer/controllers/reachability/driver; verify on TRUE
  params. *Accept:* control-math unit tests pass; Stage-A suppression beats no-control; feasible
  setpoint reached; MPC obeys `[0,1]`.
- **M3 — Stage B + the fix.** Wire `estimator_new` through `control_interface`; run Stage B.
  *Accept:* Stage B matches Stage A on setpoint (within ~1.5×), `u` not all-zero, not worse than
  open-loop; figures 1–3 saved.
- **M4 — Study.** Full matrix across scenarios + ≥5 Brain seeds; sweeps. *Accept:* figures 4–7,9,10
  + Brain figure saved; `RESULTS.md` states the per-objective feasibility verdict and the method
  comparison; failure modes documented (suppression floor, hidden-input authority loss).
- **M5 — Notebook deliverable.** `week3_solution.ipynb` / `run_week3.py` runs the full story on the
  Brain end to end with narrative mapping to the §1 notebook tasks. *Accept:* runs clean; produces
  the figures; reads as a self-contained submission.

---

## 11. Definition of done
A modular `control/` package + `estimator_new.py`; `tests/` covering control math, EM correctness,
and the Stage-A-vs-B regression; populated `results/` (all figures) and a `RESULTS.md` with:
discovered interfaces, per-milestone findings, the method comparison, the per-objective feasibility
verdict with the reachability justification, documented failure modes, and the stated improvement
(estimator rebuild fixing Stage-B inertia; MPC over LQG). A runnable notebook/script reproducing it
on the Brain. The Week-2 `estimator.py` is unchanged and unimported; only `control_interface.py`
imports `estimator_new.py`.
