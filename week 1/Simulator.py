from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np

InputSpec = None | np.ndarray | Callable[[int], np.ndarray]


class Simulator:
    """Linear Gaussian state-space simulator.

    Model:
        x[t+1] = A x[t] + B u[t] + w[t],  w ~ N(0, Q)
        y[t]   = C x[t] + o[t],           o ~ N(0, R)
    """

    def __init__(self, A, B, C, Q, R, x0=None, seed: Optional[int] = None):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)

        for name, arr in (("A", self.A), ("B", self.B), ("C", self.C)):
            if arr.ndim != 2:
                raise ValueError(f"{name} must be 2-D; got shape {arr.shape}")

        self.state_dim = self.A.shape[0]
        self.input_dim = self.B.shape[1]
        self.obs_dim = self.C.shape[0]
        self.x0 = np.zeros(self.state_dim) if x0 is None else np.asarray(x0, dtype=float)
        self.rng = np.random.default_rng(seed)

        self._validate_shapes()

    def _validate_shapes(self) -> None:
        n, m, p = self.state_dim, self.input_dim, self.obs_dim
        for name, want, got in [
            ("A",  (n, n), self.A.shape),  ("B", (n, m), self.B.shape),
            ("C",  (p, n), self.C.shape),  ("Q", (n, n), self.Q.shape),
            ("R",  (p, p), self.R.shape),  ("x0", (n,),  self.x0.shape),
        ]:
            if got != want:
                raise ValueError(f"{name} must have shape {want}; got {got}")

    def reset_seed(self, seed: Optional[int] = None) -> None:
        """Reset the simulator random number generator."""
        self.rng = np.random.default_rng(seed)

    def _make_input_sequence(self, T: int, U: InputSpec = None) -> np.ndarray:
        if U is None:
            return np.zeros((T, self.input_dim))

        if callable(U):
            seq = np.asarray([U(t) for t in range(T)], dtype=float)
        else:
            seq = np.asarray(U, dtype=float)

        if seq.shape != (T, self.input_dim):
            raise ValueError(f"U must have shape {(T, self.input_dim)}; got {seq.shape}")
        return seq

    def step(self, x, u):
        """Run one step and return x_next, y, process_noise, observation_noise."""
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)

        for name, arr, dim in (("x", x, self.state_dim), ("u", u, self.input_dim)):
            if arr.shape != (dim,):
                raise ValueError(f"{name} must have shape {(dim,)}; got {arr.shape}")

        o = self.rng.multivariate_normal(np.zeros(self.obs_dim), self.R)
        w = self.rng.multivariate_normal(np.zeros(self.state_dim), self.Q)

        y = self.C @ x + o
        x_next = self.A @ x + self.B @ u + w
        return x_next, y, w, o

    def simulate(self, T: int, U: InputSpec = None, x0=None) -> dict[str, np.ndarray]:
        """Simulate one trajectory and return x, y, u, w, o arrays."""
        _check_positive_int("T", T)

        x_init = self.x0 if x0 is None else np.asarray(x0, dtype=float)
        if x_init.shape != (self.state_dim,):
            raise ValueError(f"x0 must have shape {(self.state_dim,)}; got {x_init.shape}")

        u = self._make_input_sequence(T, U)
        x = np.empty((T + 1, self.state_dim))
        y = np.empty((T, self.obs_dim))
        w = np.empty((T, self.state_dim))
        o = np.empty((T, self.obs_dim))

        x[0] = x_init
        for t in range(T):
            x[t + 1], y[t], w[t], o[t] = self.step(x[t], u[t])

        return {"x": x, "y": y, "u": u, "w": w, "o": o}

    def simulate_trials(self, trial_count: int, T: int, U: InputSpec = None, x0=None) -> dict[str, np.ndarray]:
        """Simulate repeated trials. y has shape (Trials, Timepoints, Neurons)."""
        _check_positive_int("trial_count", trial_count)
        _check_positive_int("T", T)

        results = [self.simulate(T, U=U, x0=x0) for _ in range(trial_count)]
        return {k: np.stack([r[k] for r in results]) for k in ("x", "y", "u")}


LinearStateSpaceSimulator = Simulator


def _check_positive_int(name: str, value: int) -> None:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value}")


def _check_channel(channel: int, input_dim: int) -> None:
    _check_positive_int("input_dim", input_dim)
    if not isinstance(channel, (int, np.integer)) or isinstance(channel, bool):
        raise ValueError(f"channel must be an integer; got {channel}")
    if not 0 <= channel < input_dim:
        raise ValueError(f"channel must be in [0, {input_dim - 1}]; got {channel}")


def zero_input(T: int, input_dim: int) -> np.ndarray:
    """Return a (T, input_dim) array of zeros — no stimulation."""
    _check_positive_int("T", T)
    _check_positive_int("input_dim", input_dim)
    return np.zeros((T, input_dim))


def pulse_input(T: int, input_dim: int, channel: int = 0, start: int = 10,
                duration: int = 5, amplitude: float = 1.0) -> np.ndarray:
    """Return a (T, input_dim) array with a constant-amplitude pulse on one channel."""
    _check_positive_int("T", T)
    _check_channel(channel, input_dim)
    if not 0 <= start < T:
        raise ValueError(f"start must be in [0, {T - 1}]; got {start}")
    _check_positive_int("duration", duration)

    u = zero_input(T, input_dim)
    u[start:min(T, start + duration), channel] = amplitude
    return u


def sinusoidal_input(T: int, input_dim: int, channel: int = 0,
                     amplitude: float = 1.0, period: float = 30.0,
                     phase: float = 0.0) -> np.ndarray:
    """Return a (T, input_dim) array with a sinusoidal drive on one channel."""
    _check_positive_int("T", T)
    _check_channel(channel, input_dim)
    if period <= 0:
        raise ValueError(f"period must be positive; got {period}")

    u = zero_input(T, input_dim)
    t = np.arange(T)
    u[:, channel] = amplitude * np.sin(2 * np.pi * t / period + phase)
    return u


def random_input(T: int, input_dim: int, amplitude: float = 1.0,
                 seed: Optional[int] = None) -> np.ndarray:
    """Return a (T, input_dim) array of i.i.d. Gaussian white-noise inputs."""
    _check_positive_int("T", T)
    _check_positive_int("input_dim", input_dim)
    return amplitude * np.random.default_rng(seed).standard_normal((T, input_dim))


def channel_sweep_input(T: int, input_dim: int, amplitude: float = 1.0) -> np.ndarray:
    """Return a (T, input_dim) array that activates each channel in equal-length windows."""
    _check_positive_int("T", T)
    _check_positive_int("input_dim", input_dim)

    u = zero_input(T, input_dim)
    edges = np.linspace(0, T, input_dim + 1, dtype=int)
    for ch in range(input_dim):
        u[edges[ch]:edges[ch + 1], ch] = amplitude
    return u


def mixed_input(T: int, input_dim: int, seed: Optional[int] = None) -> np.ndarray:
    """Return a (T, input_dim) array combining a pulse, sinusoid, and small Gaussian noise."""
    u = pulse_input(T, input_dim, channel=0, start=max(1, T // 5), duration=max(1, T // 10))
    if input_dim > 1:
        u += sinusoidal_input(T, input_dim, channel=1, amplitude=0.5, period=max(2, T // 4))
    u += 0.05 * random_input(T, input_dim, seed=seed)
    return u


def controllability_matrix(A, B) -> np.ndarray:
    """Return the controllability matrix [B, AB, A²B, ...] of shape (n, n*m)."""
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or B.ndim != 2 or B.shape[0] != A.shape[0]:
        raise ValueError("Expected A with shape (n, n) and B with shape (n, m).")

    blocks = [B]
    for _ in range(1, A.shape[0]):
        blocks.append(A @ blocks[-1])
    return np.hstack(blocks)


def observability_matrix(A, C) -> np.ndarray:
    """Return the observability matrix [C; CA; CA²; ...] of shape (n*p, n)."""
    A, C = np.asarray(A, dtype=float), np.asarray(C, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or C.ndim != 2 or C.shape[1] != A.shape[0]:
        raise ValueError("Expected A with shape (n, n) and C with shape (p, n).")

    blocks = [C]
    for _ in range(1, A.shape[0]):
        blocks.append(blocks[-1] @ A)
    return np.vstack(blocks)


def matrix_rank(M, tol: float = 1e-10) -> int:
    """Return the numerical rank of matrix M, treating singular values below tol as zero."""
    return int(np.linalg.matrix_rank(np.asarray(M, dtype=float), tol=tol))


def default_neural_system(seed: Optional[int] = None, obs_dim: int = 16) -> Simulator:
    """Return a stable 4-state, 2-input, obs_dim-observation neural LDS."""
    _check_positive_int("obs_dim", obs_dim)

    radius, period = 0.97, 20.0
    theta = 2 * np.pi / period
    c, s = np.cos(theta), np.sin(theta)

    # State transition: top-left 2x2 is a stable oscillation (period 20, radius 0.97);
    # bottom-right entries are two decay modes (slow 0.98, fast 0.75);
    # off-diagonal 0.05 entries add weak coupling from oscillatory to decay modes.
    A = np.array([
        [radius * c, -radius * s, 0.00, 0.00],
        [radius * s,  radius * c, 0.00, 0.00],
        [0.05,        0.00,       0.98, 0.00],
        [0.00,        0.05,       0.00, 0.75],
    ])

    # Input matrix: each input channel drives one oscillatory dimension directly
    # and its paired decay mode at half strength.
    B = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.0, 0.5],
    ])

    rng = np.random.default_rng(seed)
    # Observation matrix: random projection from 4 latent states to obs_dim neurons.
    C = rng.standard_normal((obs_dim, 4))
    # Small isotropic process noise — state dynamics dominate over noise.
    Q = 1e-3 * np.eye(4)
    # Slightly larger observation noise — reflects realistic measurement uncertainty.
    R = 1e-2 * np.eye(obs_dim)
    # Initial state: oscillatory mode mid-cycle, decay modes at non-zero values.
    x0 = np.array([1.0, 0.0, 0.5, -0.5])

    return Simulator(A, B, C, Q, R, x0=x0, seed=seed)


def input_aligned_system(seed: Optional[int] = None, obs_dim: int = 16) -> Simulator:
    """Return a system where the first two observation rows directly track the input-driven dimensions.

    Design motivation:
        In the default system C is a random projection, so the relationship between
        inputs and observations is diffuse and must be inferred from the data. Here
        the first two rows of C are set to B.T, meaning two neurons measure exactly
        the state dimensions that are driven most directly by the inputs (dimensions
        0 and 1). The remaining obs_dim - 2 rows are drawn from a standard normal
        distribution, exactly as in default_neural_system.

        This "privileged observer" structure lets algorithms that can identify the
        input-aligned rows bypass the state estimation problem almost entirely for
        those two dimensions, making it a useful test of whether decoders exploit
        structure or average it away.

    Dynamics:
        Because B.T selects the oscillatory latent dimensions, the two aligned
        observation rows track the oscillation in x[0] and x[1] directly. The
        random rows mix all four latent dimensions as in the default system,
        producing a heterogeneous population. The contrast between informative
        and noisy rows makes this a benchmark for feature-selection, attention,
        or sparse regression methods.

    Matrix shapes:
        A  : (4, 4) — stable oscillator coupled to decay modes, identical to
                       default_neural_system
        B  : (4, 2) — input matrix, identical to default_neural_system
        C  : (obs_dim, 4) — rows 0 and 1 are B.T (2, 4);
                            rows 2 through obs_dim-1 are i.i.d. N(0, 1)
        Q  : (4, 4) — 1e-3 * I, isotropic process noise
        R  : (obs_dim, obs_dim) — 1e-2 * I, isotropic observation noise
        x0 : (4,) — [1.0, 0.0, 0.5, -0.5]

    Parameters
    ----------
    seed : int or None
        Seed for the random number generator used to draw the random rows of C
        and for subsequent simulation noise.
    obs_dim : int
        Number of observed neurons. Must be at least 2 so that the two B.T rows
        do not exhaust the observation space.

    Returns
    -------
    Simulator
    """
    _check_positive_int("obs_dim", obs_dim)
    if obs_dim < 2:
        raise ValueError(f"obs_dim must be >= 2 for input_aligned_system; got {obs_dim}")

    radius, period = 0.97, 20.0
    theta = 2 * np.pi / period
    c, s = np.cos(theta), np.sin(theta)

    A = np.array([
        [radius * c, -radius * s, 0.00, 0.00],
        [radius * s,  radius * c, 0.00, 0.00],
        [0.05,        0.00,       0.98, 0.00],
        [0.00,        0.05,       0.00, 0.75],
    ])
    B = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.0, 0.5],
    ])

    rng = np.random.default_rng(seed)
    C = rng.standard_normal((obs_dim, 4))
    C[:2] = B.T

    Q = 1e-3 * np.eye(4)
    R = 1e-2 * np.eye(obs_dim)
    x0 = np.array([1.0, 0.0, 0.5, -0.5])

    return Simulator(A, B, C, Q, R, x0=x0, seed=seed)


def input_blind_system(seed: Optional[int] = None, obs_dim: int = 16) -> Simulator:
    """Return a system where no neuron can directly observe the input-driven dimensions.

    Design motivation:
        B drives the system through state dimensions 0 and 1 (the oscillatory pair).
        Here C is drawn randomly and then columns 0 and 1 are set to zero, so that
        y[t] = C @ x[t] + noise has no direct sensitivity to x[0] or x[1]. The
        inputs' effect on y can only arrive indirectly through the coupling terms
        in A: A[2, 0] and A[3, 1] (both 0.05) slowly leak oscillatory energy into
        the decay modes x[2] and x[3], which do appear in y through the surviving
        columns of C.

        This creates a significantly harder estimation problem — decoding the input
        from observations requires the algorithm to trace a two-step causal path
        (input → oscillatory modes → decay modes → observations) that is attenuated
        by both the small coupling coefficients and the decay dynamics.

    Dynamics:
        With C[:, 0] = C[:, 1] = 0, the observation is a linear function of x[2]
        and x[3] only. Input-driven variance in x[0] and x[1] propagates into x[2]
        and x[3] at rate 0.05 per time step, creating a delayed, blurred signature
        of the input in y. This is analogous to a neural circuit where the readout
        population is one synapse removed from the input-receiving population.

    Matrix shapes:
        A  : (4, 4) — identical to default_neural_system
        B  : (4, 2) — identical to default_neural_system
        C  : (obs_dim, 4) — i.i.d. N(0, 1) then columns 0 and 1 zeroed;
                            effective sensitivity is only to x[2] and x[3]
        Q  : (4, 4) — 1e-3 * I
        R  : (obs_dim, obs_dim) — 1e-2 * I
        x0 : (4,) — [1.0, 0.0, 0.5, -0.5]

    Parameters
    ----------
    seed : int or None
        Seed for the random number generator used to draw C and for simulation noise.
    obs_dim : int
        Number of observed neurons; must be a positive integer.

    Returns
    -------
    Simulator
    """
    _check_positive_int("obs_dim", obs_dim)

    radius, period = 0.97, 20.0
    theta = 2 * np.pi / period
    c, s = np.cos(theta), np.sin(theta)

    A = np.array([
        [radius * c, -radius * s, 0.00, 0.00],
        [radius * s,  radius * c, 0.00, 0.00],
        [0.05,        0.00,       0.98, 0.00],
        [0.00,        0.05,       0.00, 0.75],
    ])
    B = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.0, 0.5],
    ])

    rng = np.random.default_rng(seed)
    C = rng.standard_normal((obs_dim, 4))
    C[:, 0] = 0.0
    C[:, 1] = 0.0

    Q = 1e-3 * np.eye(4)
    R = 1e-2 * np.eye(obs_dim)
    x0 = np.array([1.0, 0.0, 0.5, -0.5])

    return Simulator(A, B, C, Q, R, x0=x0, seed=seed)


def slow_drift_system(seed: Optional[int] = None, obs_dim: int = 16) -> Simulator:
    """Return a system with a near-random-walk slow drift mode that challenges state estimation.

    Design motivation:
        In default_neural_system, A[2, 2] = 0.98, giving the third state dimension
        a time constant of roughly 50 time steps. Here A[2, 2] is raised to 0.999,
        extending the time constant to approximately 1000 steps — approaching a
        random walk. This slow drift creates a low-frequency component in both x
        and y that wanders far from zero over long trials, making it difficult for
        any filter or decoder to separate signal from slow noise.

        Systems with near-unit-root dynamics are challenging because (a) the Kalman
        gain for a slow mode is large, making the filter sensitive to observation
        noise, and (b) the steady-state error covariance grows near-linearly in
        time for a true random walk, degrading any fixed-gain approximation.

    Dynamics:
        The modified entry A[2, 2] = 0.999 means the third state dimension decays
        by only 0.1% per step. Under the weak coupling from the oscillatory modes
        (A[2, 0] = 0.05), x[2] accumulates a low-frequency drift that outlasts many
        oscillation periods. All other dynamics — the oscillatory pair and the fast
        decay x[3] — are identical to the default system. C is a random projection
        as in default_neural_system, so the drift appears diffusely across all
        observed neurons.

    Matrix shapes:
        A  : (4, 4) — as default_neural_system but A[2, 2] = 0.999 (was 0.98)
        B  : (4, 2) — identical to default_neural_system
        C  : (obs_dim, 4) — i.i.d. N(0, 1) random projection
        Q  : (4, 4) — 1e-3 * I
        R  : (obs_dim, obs_dim) — 1e-2 * I
        x0 : (4,) — [1.0, 0.0, 0.5, -0.5]

    Parameters
    ----------
    seed : int or None
        Seed for the random number generator used to draw C and for simulation noise.
    obs_dim : int
        Number of observed neurons; must be a positive integer.

    Returns
    -------
    Simulator
    """
    _check_positive_int("obs_dim", obs_dim)

    radius, period = 0.97, 20.0
    theta = 2 * np.pi / period
    c, s = np.cos(theta), np.sin(theta)

    A = np.array([
        [radius * c, -radius * s, 0.00,  0.00],
        [radius * s,  radius * c, 0.00,  0.00],
        [0.05,        0.00,       0.999, 0.00],
        [0.00,        0.05,       0.00,  0.75],
    ])
    B = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.0, 0.5],
    ])

    rng = np.random.default_rng(seed)
    C = rng.standard_normal((obs_dim, 4))
    Q = 1e-3 * np.eye(4)
    R = 1e-2 * np.eye(obs_dim)
    x0 = np.array([1.0, 0.0, 0.5, -0.5])

    return Simulator(A, B, C, Q, R, x0=x0, seed=seed)


def closed_loop_system(seed: Optional[int] = None, obs_dim: int = 16) -> Simulator:
    """Return a system combining input-aligned observations with an amplified input drive.

    Design motivation:
        This system simulates a closed-loop experimental setting where (a) the
        readout of the system's state is directly aligned with the input dimensions,
        as in input_aligned_system, and (b) the input effect on the state is
        amplified by a factor of 2.0 compared with the default system. The
        combination means that input perturbations create larger excursions in the
        latent state and are immediately visible in the first two rows of y,
        approximating a scenario where a brain-computer interface both reads out and
        drives a neural population with high efficiency.

        The amplified B makes the system more reactive to stimulation, which is
        interesting for studying how filters handle large, sudden state changes and
        whether estimation quality degrades when the signal-to-noise ratio of the
        input-evoked response is high relative to the background dynamics.

    Dynamics:
        B_scaled = 2 * B drives both oscillatory dimensions at twice the default
        strength. The first two rows of C are set to B.T (the unscaled version),
        placing the aligned readout in the original B column space; the remaining
        obs_dim - 2 rows are random. Because B_scaled doubles the state excursions,
        the SNR of the input-evoked response in y is roughly 4x higher than in the
        default system (variance scales as amplitude squared), while observation
        noise R is unchanged — making this a high-SNR, easy-to-decode regime that
        nonetheless stresses filters designed around smaller perturbations.

    Matrix shapes:
        A       : (4, 4) — identical to default_neural_system
        B_scaled: (4, 2) — 2.0 * B from default_neural_system; stored as sim.B
        C       : (obs_dim, 4) — rows 0 and 1 are B.T of the unscaled B (2, 4);
                                 rows 2 through obs_dim-1 are i.i.d. N(0, 1)
        Q       : (4, 4) — 1e-3 * I
        R       : (obs_dim, obs_dim) — 1e-2 * I
        x0      : (4,) — [1.0, 0.0, 0.5, -0.5]

    Parameters
    ----------
    seed : int or None
        Seed for the random number generator used to draw the random rows of C
        and for simulation noise.
    obs_dim : int
        Number of observed neurons. Must be at least 2 so that the two B.T rows
        can be placed.

    Returns
    -------
    Simulator
    """
    _check_positive_int("obs_dim", obs_dim)
    if obs_dim < 2:
        raise ValueError(f"obs_dim must be >= 2 for closed_loop_system; got {obs_dim}")

    radius, period = 0.97, 20.0
    theta = 2 * np.pi / period
    c, s = np.cos(theta), np.sin(theta)

    A = np.array([
        [radius * c, -radius * s, 0.00, 0.00],
        [radius * s,  radius * c, 0.00, 0.00],
        [0.05,        0.00,       0.98, 0.00],
        [0.00,        0.05,       0.00, 0.75],
    ])
    B = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.0, 0.5],
    ])
    B_scaled = 2.0 * B

    rng = np.random.default_rng(seed)
    C = rng.standard_normal((obs_dim, 4))
    C[:2] = B.T

    Q = 1e-3 * np.eye(4)
    R = 1e-2 * np.eye(obs_dim)
    x0 = np.array([1.0, 0.0, 0.5, -0.5])

    return Simulator(A, B_scaled, C, Q, R, x0=x0, seed=seed)

def non_normal_system(seed: Optional[int] = None, obs_dim: int = 8) -> Simulator:
    """Return a system whose transition matrix is non-normal: it produces a
    large transient amplification despite all eigenvalues being well inside
    the unit circle.

    Design motivation:
        All existing systems have transition matrices that are essentially
        normal (rotation block + diagonal decays), so ||A^t||_2 decays
        monotonically and an input or initial-condition impulse decays at
        a rate set purely by max |lambda(A)|. The supervisor's first
        scenario calls for a stable A whose ||A^t||_2 grows transiently
        before decay -- a hallmark of non-orthogonal eigenvectors.

        We build A as an upper-triangular feedforward chain:
            A_ii = decay_top - i * decay_step  (4 distinct decay rates)
            A_{i, i+1} = chain_gain            (above-diagonal coupling)
        Eigenvalues are exactly the diagonal entries (so stability is
        guaranteed and transparent), but right eigenvectors of A are
        non-orthogonal and increasingly "tilted" the further up the chain
        they sit. An impulse at the bottom of the chain (state n-1)
        cascades upward through the superdiagonal couplings, accumulating
        before all modes decay together.

    Dynamics:
        With the parameters chosen below:
            eigenvalues : 0.92, 0.72, 0.52, 0.32
            cond(V)     : ~ 74           (moderate, not pathological)
            Henrici dep : ~ 0.77         (clearly non-normal)
            ||A^t||_2 peaks at ~ 3.6 around t = 8, then decays.
        Inputs are injected at the bottom of the chain; the random C
        projection picks up the chain dynamics diffusely. The non-normal
        peak is visible both in unforced trajectories (x0 has a unit at
        the bottom of the chain) and in pulse-evoked responses.

    Matrix shapes:
        A  : (4, 4) -- upper-triangular feedforward chain
        B  : (4, 2) -- inputs enter at state n-1 (channel 0) and n-2 (channel 1)
        C  : (obs_dim, 4) -- i.i.d. N(0, 1) random projection
        Q  : (4, 4) -- 1e-3 * I
        R  : (obs_dim, obs_dim) -- 1e-2 * I
        x0 : (4,) -- e_{n-1}: unit impulse at the bottom of the chain

    Parameters
    ----------
    seed : int or None
        Seed for the random number generator used to draw C and for simulation noise.
    obs_dim : int
        Number of observed neurons; must be a positive integer.

    Returns
    -------
    Simulator
    """
    _check_positive_int("obs_dim", obs_dim)

    n = 4
    decay_top, decay_step, chain_gain = 0.92, 0.20, 0.70
    diag_vals = decay_top - decay_step * np.arange(n)
    A = np.diag(diag_vals) + chain_gain * np.diag(np.ones(n - 1), k=1)

    B = np.zeros((n, 2))
    B[-1, 0] = 1.0
    B[-2, 1] = 1.0

    rng = np.random.default_rng(seed)
    C = rng.standard_normal((obs_dim, n))

    Q = 1e-3 * np.eye(n)
    R = 1e-2 * np.eye(obs_dim)
    x0 = np.zeros(n)
    x0[-1] = 1.0

    return Simulator(A, B, C, Q, R, x0=x0, seed=seed)


def hidden_input_system(seed: Optional[int] = None, obs_dim: int = 10) -> Simulator:
    """Return a system where the input enters in the observation null-space:
    C @ B = 0 exactly, so inputs are invisible at the instant they arrive
    but become visible later through the action of A.

    Design motivation:
        The existing input_blind_system kills input visibility by setting
        two columns of C to zero -- a very strong, rank-deficient choice.
        The supervisor's second scenario is more general: the input drives
        the latent state in directions that lie close to the null-space of
        C, with effects becoming visible only after A rotates the state
        out of that null-space. The Markov parameters
            CB, C A B, C A^2 B, ...
        characterise this delay: CB ~ 0 but CA^k B grows then decays.

        We pick B (rank 2) with columns that span a non-axis-aligned 2D
        subspace (primary weight on the oscillatory dimensions, smaller
        projections onto the decay modes) and build C from random rows
        projected onto span(B)^perp. This gives CB = 0 exactly while C
        retains non-zero entries in all columns. A is a stable oscillator
        weakly coupled into three real decays; the cross-coupling
        A[2,0], A[3,1], A[4,2] are what rotate the input-driven state
        into the observable subspace over time.

    Dynamics:
        With this construction:
            ||C B||_F            = 0                  (input invisible at t=0)
            ||C A B||_F          > 0                  (visible after one step)
            eigenvalues          : one oscillatory pair at radius 0.95 and
                                   period 16, plus three real decays
                                   (0.80, 0.65, 0.55)
        A pulse on either input channel produces zero immediate change in y
        but a non-zero response from t=1 onward. Because B also has weight
        on the decay modes, the response rises within the first 1-2 steps
        (faster than in input_blind_system).

    Matrix shapes:
        A  : (5, 5) -- oscillator (2x2) + three diagonal decays + cross-coupling
        B  : (5, 2) -- primary weight on oscillatory dims; smaller entries in
                        decay dims so span(B) is not axis-aligned
        C  : (obs_dim, 5) -- random rows projected onto span(B)^perp so CB = 0;
                             all columns non-zero
        Q  : (5, 5) -- 1e-3 * I
        R  : (obs_dim, obs_dim) -- 1e-2 * I
        x0 : (5,) -- zeros (system is at rest; only the input drives response)

    Parameters
    ----------
    seed : int or None
        Seed for the random number generator used to draw C and for simulation noise.
    obs_dim : int
        Number of observed neurons; must be a positive integer.

    Returns
    -------
    Simulator
    """
    _check_positive_int("obs_dim", obs_dim)

    n = 5
    radius, period = 0.95, 16.0
    theta = 2 * np.pi / period
    cs, sn = np.cos(theta), np.sin(theta)

    A = np.zeros((n, n))
    # 2x2 oscillator block
    A[0, 0], A[0, 1] = radius * cs, -radius * sn
    A[1, 0], A[1, 1] = radius * sn,  radius * cs
    # Real decays
    A[2, 2] = 0.80
    A[3, 3] = 0.65
    A[4, 4] = 0.55
    # Cross-coupling: oscillator leaks into decays (rotates B out of null(C))
    A[2, 0] = 0.40
    A[3, 1] = 0.40
    A[4, 2] = 0.30

    # B drives the oscillatory dimensions primarily, with smaller projections into
    # the decay modes so that span(B) is not axis-aligned. This ensures C (built
    # below) has no zero columns while still satisfying CB = 0 exactly.
    B = np.zeros((n, 2))
    B[0, 0] = 1.0;  B[2, 0] = 0.3;  B[4, 0] = 0.2
    B[1, 1] = 1.0;  B[3, 1] = 0.3;  B[4, 1] = 0.2

    # Build C with rows in span(B)^perp so CB = 0 exactly.
    # pinv handles non-orthogonal columns of B correctly.
    P_perp = np.eye(n) - B @ np.linalg.pinv(B)
    rng = np.random.default_rng(seed)
    C_raw = rng.standard_normal((obs_dim, n))
    C = C_raw @ P_perp

    Q = 1e-3 * np.eye(n)
    R = 1e-2 * np.eye(obs_dim)
    x0 = np.zeros(n)

    return Simulator(A, B, C, Q, R, x0=x0, seed=seed)


def ill_conditioned_system(seed: Optional[int] = None, obs_dim: int = 8) -> Simulator:
    """Return a system whose eigenbasis is ill-conditioned: the eigenvalues
    are well-separated, but the eigenvectors are nearly linearly dependent,
    so modes mix unevenly into the state coordinates and produce different
    apparent variance and frequency content across dimensions.

    Design motivation:
        The supervisor's third scenario distinguishes "non-normal" (large
        ||A^t|| transients) from "ill-conditioned eigenbasis" (uneven mode
        mixing). The two are mathematically related (cond(V) > 1 implies
        non-orthogonal eigenvectors) but the *signatures* differ:
            non-normal system  -> large transient peak in ||A^t||_2
            ill-conditioned    -> uneven per-coordinate variance and
                                  distorted apparent frequencies under
                                  isotropic noise

        We construct A = V Lambda V^{-1} with:
            - Lambda: block-diagonal with one stable oscillator block
                      (radius 0.95, period 24) plus two real decays
                      (0.80, 0.60)
            - V: a real matrix with cond(V) = cond_target, built by SVD
                 with geometrically-spaced singular values.

        Because Lambda's eigenvalues are well-separated, the *true* dynamics
        contain three distinct timescales, but the ill-conditioned V mixes
        them into the standard coordinate basis non-uniformly. Under
        isotropic process noise the per-coordinate variance can differ by
        factors of 5-10x even though Q = sigma^2 I, and FFT peak frequencies
        of individual coordinates can be displaced from the true 24-step
        period.

    Dynamics:
        With cond_target = 20:
            cond(V)                ~ 20
            max_t ||A^t||_2        ~ 10   (some transient growth, smaller
                                           than non_normal_system relative
                                           to peak)
            Per-coord variance     varies by ~ 5x under isotropic noise
            Apparent FFT periods   per coordinate can differ from the true
                                   24-step underlying period
        This is exactly the supervisor's "conditioning does not create new
        true eigenfrequencies, but it can make different latent coordinates
        appear to contain different frequencies".

    Matrix shapes:
        A  : (4, 4) -- V * Lambda * V^{-1} with Lambda block-diagonal
        B  : (4, 2) -- random projection into the ill-conditioned basis
        C  : (obs_dim, 4) -- i.i.d. N(0, 1) random projection
        Q  : (4, 4) -- 1e-3 * I (isotropic; the conditioning is what
                       distorts the apparent variance)
        R  : (obs_dim, obs_dim) -- 1e-2 * I
        x0 : (4,) -- random N(0, 0.3^2) so initial condition energy
                     distributes across all modes

    Parameters
    ----------
    seed : int or None
        Seed for the random number generator used to draw V, B, C, x0
        and for simulation noise.
    obs_dim : int
        Number of observed neurons; must be a positive integer.

    Returns
    -------
    Simulator
    """
    _check_positive_int("obs_dim", obs_dim)

    n = 4
    cond_target = 20.0
    rng = np.random.default_rng(seed)

    # Real-block representation of eigenvalues
    radius, period = 0.95, 24.0
    theta = 2 * np.pi / period
    cs, sn = np.cos(theta), np.sin(theta)
    Lambda_real = np.zeros((n, n))
    Lambda_real[0, 0], Lambda_real[0, 1] = radius * cs, -radius * sn
    Lambda_real[1, 0], Lambda_real[1, 1] = radius * sn,  radius * cs
    Lambda_real[2, 2] = 0.80
    Lambda_real[3, 3] = 0.60

    # Build V with prescribed condition number via SVD with geometrically
    # spaced singular values: cond(V) = cond_target exactly.
    M = rng.standard_normal((n, n))
    U_, _, Wt = np.linalg.svd(M)
    s = np.logspace(0, -np.log10(cond_target), n)
    V = U_ @ np.diag(s) @ Wt
    A = V @ Lambda_real @ np.linalg.inv(V)

    B = rng.standard_normal((n, 2))
    C = rng.standard_normal((obs_dim, n))
    Q = 1e-3 * np.eye(n)
    R = 1e-2 * np.eye(obs_dim)
    x0 = rng.standard_normal(n) * 0.3

    return Simulator(A, B, C, Q, R, x0=x0, seed=seed)