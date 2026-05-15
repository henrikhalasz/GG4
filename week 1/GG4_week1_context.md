# GG4 Week 1 — Claude Code Context

## Project Overview

**GG4: Neural Control with Adaptive State Estimation** (CUED IIA, 2026)

You are working with a **partially observed linear dynamical system** inspired by neural interfaces. The hidden state is not directly observable — you can only apply inputs `u(t)` and observe noisy measurements `y(t)`. The project is about modelling, inference, and control under uncertainty.

---

## The System Model

The system follows a **linear state-space model (LDS)**:

```
x_{t+1} = A x_t + B u_t + w_t,    w_t ~ N(0, Q)   # state transition
y_t     = C x_t + o_t,             o_t ~ N(0, R)   # observation model
```

| Variable | Meaning |
|----------|---------|
| `x_t`   | Hidden latent state (not directly observed) |
| `y_t`   | Noisy observations (what you measure) |
| `u_t`   | Input / stimulation command |
| `A`     | State transition matrix (system dynamics) |
| `B`     | Input matrix (how inputs affect state) |
| `C`     | Observation matrix (how state maps to observations) |
| `Q`     | Process noise covariance |
| `R`     | Observation noise covariance |

---

## Week 1 Goal

**Group task.** Build a shared simulation and exploration workflow that all group members will use as the foundation for individual work in Weeks 2–4.

**Deliverable:** Group simulation code + brief documentation  
**Marks:** 10 (group)  
**Due:** Friday 22 May, 11am–1pm (compulsory check-in session)

At the check-in, the demonstrator will use your code and ask every group member questions about it. Everyone must understand and be able to explain the full codebase.

---

## Concrete Tasks

### Task 1 — Complete `Illustrator.py`

**File:** `Illustrator.py`  
**Class:** `Illustrator`

The skeleton is already provided. The `__init__` accepts a 3D numpy array of shape `(Trials, Timepoints, Neurons)` and stores:
- `self.observation`
- `self.trial_cnt`, `self.timestep_cnt`, `self.neuron_cnt`

You must **implement methods** that provide:
- Summary statistics (mean, variance, etc. across trials/neurons/time)
- Standard visualisation functions — e.g. time series plots, heatmaps, trial-averaged activity, per-neuron plots
- Any other analyses that help understand what's in the data

The demonstrator will import and use `Illustrator` directly in `week1.ipynb` **without seeing your notebook code**, so the class must be self-contained and well-documented with clear docstrings.

Example usage in `week1.ipynb`:
```python
from Illustrator import Illustrator
illustrator = Illustrator(data)
# demonstrator calls your methods here
```

### Task 2 — Build `Simulator.py`

**File:** `Simulator.py` (does not exist yet — create it)

Implement a simulator for the linear dynamical system above. It must:
- Accept parameters: `A`, `B`, `C`, `Q`, `R`, initial state `x0`, and a sequence of inputs `u`
- Simulate observations `y` of any length
- Be well-documented so the demonstrator can use it without seeing your notebook code

Example interface (design your own, but make it clean):
```python
import Simulator as sim
# e.g. sim.run(A, B, C, Q, R, x0, u_sequence) -> (x_trajectory, y_observations)
```

### Task 3 — Explore Input Patterns and System Dynamics in `week1.ipynb`

Using your `Illustrator` and `Simulator` together, run experiments to answer:

> *"What are the effects of different input patterns and system dynamics?"*

Experiments to run and document:

| Input type | What to vary |
|------------|-------------|
| Random inputs | Different noise levels / distributions |
| Pulse inputs | Single spikes in one or multiple channels |
| Oscillatory inputs | Sinusoidal / periodic drives at different frequencies |
| Single-channel activation | Activate one input channel at a time, observe which neurons respond |
| Multi-channel, varying amplitudes | How does amplitude ratio affect the observed signals? |
| Coupling exploration | How does changing `A` (transition matrix) change system behaviour — stable vs. oscillatory vs. divergent? |
| Input–dynamics coupling | How does `B` structure interact with `A` to shape trajectories? |

For each experiment: plot the observations, note what you find, and write a brief interpretation.

### Task 4 — Load and Explore the Example Dataset

**File:** `ExampleDataset.npy` (shape: `5 Trials × 60 Timepoints × 16 Neurons`)

```python
import numpy as np
data = np.load("ExampleDataset.npy")  # shape (5, 60, 16)
```

Use your `Illustrator` to characterise this real dataset:
- What does the neural activity look like across trials?
- Are some neurons more active / variable than others?
- Is there structure across time?
- How does it compare to your simulated data?

### Task 5 — Brief Documentation

Write a short documentation section (can be markdown cells in `week1.ipynb` or a separate README) covering:
- What your `Illustrator` methods do
- What your `Simulator` interface is and what parameters it expects
- Key findings from your exploration
- Any modelling decisions you made (e.g. how you chose dimensionality, noise levels)

---

## Files Overview

| File | Status | What to do |
|------|--------|------------|
| `introduction.ipynb` | Read-only | Background and project overview |
| `week1.ipynb` | **Edit** | Main notebook — run experiments here |
| `Illustrator.py` | **Complete** | Add visualisation/stats methods |
| `Simulator.py` | **Create** | Build LDS simulator from scratch |
| `ExampleDataset.npy` | Read-only | Load and explore with Illustrator |
| `introduction_filtering.ipynb` | Reference | Background on Kalman filtering (useful for Week 2) |
| `introduction_control.ipynb` | Reference | Background on control (useful for Week 3) |

---

## Modelling Decisions You Need to Make

These are design choices your group must justify:

- **Latent state dimensionality** — how many hidden dimensions does `x_t` have?
- **Structure of `A`** — diagonal? fully coupled? stable (eigenvalues inside unit circle)?
- **Structure of `C`** — how does the hidden state project to observations?
- **Noise levels** — `Q` and `R` — how noisy is the system vs. the observations?
- **Input structure** — how many input channels? how does `B` connect inputs to states?

Different choices lead to very different system behaviours and estimation challenges. Explore several and compare.

---

## What Makes a Good Week 1 Submission

- `Illustrator` class with multiple useful, well-documented methods
- `Simulator` that cleanly implements the LDS equations and handles different parameter sets
- Experiments in `week1.ipynb` covering multiple input types and dynamics regimes
- Clear plots with labels, titles, and interpretation
- Every group member can explain every part of the code

---

## Key Equations Reference

**Stable system:** eigenvalues of `A` inside the unit circle (`|λ| < 1`)  
**Oscillatory system:** complex eigenvalues of `A` with `|λ| ≈ 1`  
**Divergent system:** eigenvalues of `A` outside the unit circle (`|λ| > 1`) — avoid for long simulations

**Simulation loop:**
```python
x = x0
for t in range(T):
    w = np.random.multivariate_normal(np.zeros(n), Q)
    o = np.random.multivariate_normal(np.zeros(p), R)
    x_next = A @ x + B @ u[t] + w
    y[t]   = C @ x + o
    x = x_next
```
