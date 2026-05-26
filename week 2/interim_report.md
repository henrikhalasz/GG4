# Week 2 Interim Report — Estimation

**Subspace identification + Kalman filtering for a partially-observed linear-Gaussian system**

---

## 1. System and Week 1 summary

We study a partially-observed linear-Gaussian state-space ("neural") system

$$
x_{t+1} = A x_t + B u_t + w_t,\quad w_t\sim\mathcal N(0,Q),\qquad
y_t = C x_t + o_t,\quad o_t\sim\mathcal N(0,R),
$$

with hidden latent state $x_t\in\mathbb R^n$, input $u_t\in\mathbb R^m$, and noisy
observation $y_t\in\mathbb R^p$. Typical parameters are $n\in\{4,5\}$, $m=2$,
$p\in\{8,10,16\}$, $Q\approx10^{-3}I$, $R\approx10^{-2}I$. At test time the
estimator sees **only** $y$ and the dimensions $(n,m)$; $A,B,C,Q,R$ and the true
$u$ are all unknown.

Week 1 built the `Simulator` (the model above plus seven scenario factories) and
the `Illustrator` exploration toolkit. The exploration showed that the
observations are a low-rank linear mixture ($p$ neurons driven by $n\ll p$ latent
modes) buried in white observation noise of standard deviation
$\sqrt{\operatorname{tr}R/p}\approx0.1$, with strong temporal autocorrelation set
by the eigenvalues of $A$ (a stable oscillator plus decay modes). This motivates a
**dynamical** estimator that exploits both the spatial low-rank structure and the
temporal dynamics, rather than a purely static decomposition.

## 2. Estimation approach

**Objective.** Recover (i) a denoised latent state that reconstructs the
observations, and (ii) the unknown input that drives the dynamics — a
representation directly usable for Week 3 control. The primary method is
**Approach B: subspace identification + Kalman filtering**; PCA (Approach A) and
time-delay embedding + PCA (Approach C) are kept as baselines.

**(a) Subspace identification.** From the centred observations we form past/future
block-Hankel matrices $Y_p,Y_f$ with block size
$k=\max(2n,\lceil\sqrt T/2\rceil)$ (clipped so the matrices keep $\ge4k$ columns).
The orthogonal projection of the future onto the past row space,
$\mathcal O = Y_f\,Y_p^{+}Y_p$, has a truncated SVD
$\mathcal O \approx U_n\Sigma_n V_n^{\top}$ giving the extended observability
matrix $\Gamma=U_n\Sigma_n^{1/2}$. We read off $\hat C=\Gamma_{[:p]}$.

The textbook transition estimate uses shift-invariance,
$\hat A=\Gamma_{[:-p]}^{+}\Gamma_{[p:]}$. We found this estimate is **badly biased
when the data are input-driven**: the unknown input correlates with the state and
contaminates the unforced projection, returning wrong eigenvalues. Because the
Kalman filter's *state* estimate is corrected by the observations and stays
accurate regardless of small errors in $A$, we instead **refine $\hat A$ by
least-squares regression on the autonomously-filtered state**,
$\hat A=\arg\min_A\sum_t\lVert\hat x_{t+1}-A\hat x_t\rVert^2$. This is the
maximum-likelihood transition under the assumption that the input behaves like
extra process noise — exact for broadband inputs (see §3). A light spectral cap
($\rho(\hat A)\le0.999$ only if it exceeds $1.005$) guards long-horizon stability
while leaving genuine near-unit modes intact.

**(b) Input matrix and noise.** With $\hat A,\hat C$ fixed we run a Kalman filter
assuming $B=0$ and take the one-step latent residual
$r_t=\hat x_t-\hat A\hat x_{t-1}\approx Bu_{t-1}+w_{t-1}$. Its dominant $m$ left
singular vectors form $\hat B\in\mathbb R^{n\times m}$. The covariances
$\hat Q,\hat R$ are the residual covariances of the dynamics and observation
equations, with $10^{-6}I$ shrinkage to stay positive-definite.

**(c) Kalman filtering and input estimation.** With
$(\hat A,\hat B,\hat C,\hat Q,\hat R)$ we run the standard linear filter,

$$
\hat x_{t|t-1}=\hat A\hat x_{t-1}+\hat B\hat u_{t-1},\quad
P^{-}=\hat A P\hat A^{\top}+\hat Q,\quad
K=P^{-}\hat C^{\top}(\hat C P^{-}\hat C^{\top}+\hat R)^{-1},
$$
$$
\hat x_{t}= \hat x_{t|t-1}+K\,(y_t-\hat C\hat x_{t|t-1}),\quad
P=(I-K\hat C)P^{-}.
$$

The input that best explains the filtered dynamics is
$\hat u_t=\hat B^{+}(\hat x_{t+1}-\hat A\hat x_t)$ (final step padded). We use
**two passes**: pass 1 with $\hat u=0$, then re-filter with $\hat u$ plugged into
the prediction step. Every inverse/pseudoinverse is regularised; the public
`estimate_latent_and_input` is wrapped so it *never crashes* — on any failure it
warns and returns zero-filled arrays of the correct shape.

**Baselines.** *PCA* keeps the top-$n$ principal components of $y$ as latent
states; *Hankel+PCA* first augments each $y_t$ with its recent history. Both
estimate inputs from the AR(1) residual of the latent scores through a
pseudoinverse, giving identical output shapes for a fair comparison.

**Identifiability (§3 B.4).** Subspace ID recovers $(A,B,C)$ only up to a
similarity transform $T\in GL(n)$: $(T^{-1}AT,T^{-1}B,CT)$ fits the data
identically. Hence the latent states live in an arbitrary internal basis and the
input is defined only up to a linear map. We therefore evaluate the basis-
invariant reconstruction $\hat y=\hat C\hat x$ directly, compare latent states
only after a best-linear-map alignment (generalised Procrustes), and score input
recovery by the $R^2$ of a best linear fit $M\hat u\approx u$ — never by direct
subtraction.

## 3. Results

All figures are produced by `week2_analysis.ipynb` (figures saved at 150 dpi).

**Reconstruction is excellent and basis-invariant.** On `default_neural_system`
(500 steps, `mixed_input`) the reconstruction RMSE is **0.108**, essentially equal
to the observation-noise floor of $0.10$; across all seven scenarios the relative
error $\text{RMSE}/\text{RMS}(y)$ stays between $0.5\%$ and $3.5\%$
(`figures/s2_scenario_metrics.png`, left). Figure `s1_reconstruction.png` shows
$\hat C\hat x$ overlaying $y$ almost exactly. Notably, PCA attains a *lower*
in-sample RMSE ($0.086$) — but that is **below the noise floor**, i.e. PCA fits
observation noise, whereas the Kalman reconstruction sits exactly at the floor
because it correctly separates signal from noise. Hankel+PCA reconstructs worst
($0.52$): its delay-embedding components are not optimised for instantaneous
reconstruction. Latent states are recovered with mean column correlation
$\approx0.999$ after linear alignment, confirming the filter tracks the true state
up to the expected $GL(n)$ transform.

![Observation reconstruction on default_neural_system](figures/s1_reconstruction.png)

**Input recovery depends critically on the input spectrum.** Figure
`s4_input_generalisation.png` is the most informative result: driving
`default_neural_system` with each Week-1 input pattern, input-recovery $R^2$ is
**0.76 for broadband `random` (white) input** but collapses for strongly
autocorrelated inputs — `sinusoidal` $0.006$, `channel_sweep` $0.035$, `pulse`
$0.306$, `mixed` $0.186$. This is not an implementation defect but a *fundamental*
limitation: when the input is white it is uncorrelated with the state and the
regression for $A$ is unbiased, so $\hat u=\hat B^{+}(\hat x_{t+1}-\hat A\hat x_t)$
recovers $u$ up to a linear map; when the input is autocorrelated it is
confounded with the autonomous dynamics, and an unidentifiable rank-$m$ component
of $A$ is absorbed into the input channel (compounding the $GL(n)$ ambiguity).
Reconstruction, by contrast, is unaffected — confirming the two tasks have very
different identifiability.

![Input-pattern generalisation](figures/s4_input_generalisation.png)

**Scenario sweep.** Reconstruction is uniformly strong; input recovery
(`mixed_input`) is modest and similar across the well-observed scenarios —
`slow_drift` ($R^2=0.246$) narrowly edges out `default` ($0.186$) and
`input_aligned` ($0.184$) — and is **near zero for `input_blind` and `hidden_input`**
(`figures/s2_scenario_metrics.png`, right), the two scenarios where the input is
structurally hidden from the observations.

![Scenario sweep: reconstruction (left) and input recovery (right)](figures/s2_scenario_metrics.png)

**Noise sensitivity and stability.** Scaling $R$ by $\{0.1,1,10,100\}$
(`figures/s3_noise_sensitivity.png`) degrades reconstruction RMSE smoothly from
$0.065$ to $0.89$ and input $R^2$ from $0.19$ to $0.05$, with no instability. Over
a long $T=2000$ run the 100-step-windowed reconstruction RMSE stays flat at
$\approx0.087$ (`figures/s5_long_horizon.png`): the filter does not diverge.

## 4. Limitations and next steps

**Hidden input ($CB=0$).** In `hidden_input_system` the Markov parameter $CB=0$,
so a stimulus is invisible the instant it arrives and only leaks into $y$ later
through $A$ (`figures/s6_limitations.png`, left: $\lVert CA^kB\rVert$ starts at
zero and grows for $k\ge1$, unlike `default`). A causal filter cannot attribute
the current observation to the current input, so input recovery collapses
($R^2\approx0$) even though reconstruction remains excellent. `input_blind`
(columns of $C$ zeroed) fails for the same reason.

![Limitations: input-visibility delay (left); the latent slow-drift mode is tracked accurately despite degraded input recovery (right)](figures/s6_limitations.png)

**Slow drift.** We expected the near-random-walk mode ($A_{22}=0.999$) to be
problematic because its steady-state Kalman gain is large. In practice, at the
default low noise and with many sensors the filter does not lose the slow mode:
`figures/s6_limitations.png` (right) shows the *latent* slow-drift state tracked
accurately after alignment — included to confirm that state estimation survives
even though input recovery is degraded by the autocorrelation confound. The input
$R^2=0.246$ only narrowly edges out `default` ($0.186$) and `input_aligned`
($0.184$), not a dramatic margin; the anticipated gain-driven noise sensitivity
emerges only at elevated $R$ (§3 noise sweep).

**Autocorrelated-input confound.** As shown in §3, single-trajectory blind input
estimation cannot separate a structured input from the autonomous dynamics. This
is the dominant limitation and the $GL(n)$ similarity ambiguity is irremovable
without side information.

**Next steps (Week 3, control).** The identified $(\hat A,\hat B,\hat C)$ and the
Kalman state are exactly the ingredients for LQG/feedback control. The confound
also points to a remedy now under our control: in closed loop we *choose* the
input, so injecting a broadband probe (or known dither) makes $A$ identifiable and
$\hat B$ well-scaled, turning the Week-2 weakness into a controllable design
choice. We will (i) validate one-step-ahead prediction, (ii) design an LQR on the
identified model, and (iii) close the loop through the Kalman filter, using
broadband excitation to keep the model identifiable during operation.
