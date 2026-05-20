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
    _check_positive_int("T", T)
    _check_positive_int("input_dim", input_dim)
    return np.zeros((T, input_dim))


def pulse_input(T: int, input_dim: int, channel: int = 0, start: int = 10,
                duration: int = 5, amplitude: float = 1.0) -> np.ndarray:
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
    _check_positive_int("T", T)
    _check_positive_int("input_dim", input_dim)
    return amplitude * np.random.default_rng(seed).standard_normal((T, input_dim))


def channel_sweep_input(T: int, input_dim: int, amplitude: float = 1.0) -> np.ndarray:
    _check_positive_int("T", T)
    _check_positive_int("input_dim", input_dim)

    u = zero_input(T, input_dim)
    edges = np.linspace(0, T, input_dim + 1, dtype=int)
    for ch in range(input_dim):
        u[edges[ch]:edges[ch + 1], ch] = amplitude
    return u


def mixed_input(T: int, input_dim: int, seed: Optional[int] = None) -> np.ndarray:
    u = pulse_input(T, input_dim, channel=0, start=max(1, T // 5), duration=max(1, T // 10))
    if input_dim > 1:
        u += sinusoidal_input(T, input_dim, channel=1, amplitude=0.5, period=max(2, T // 4))
    u += 0.05 * random_input(T, input_dim, seed=seed)
    return u


def controllability_matrix(A, B) -> np.ndarray:
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or B.ndim != 2 or B.shape[0] != A.shape[0]:
        raise ValueError("Expected A with shape (n, n) and B with shape (n, m).")

    blocks = [B]
    for _ in range(1, A.shape[0]):
        blocks.append(A @ blocks[-1])
    return np.hstack(blocks)


def observability_matrix(A, C) -> np.ndarray:
    A, C = np.asarray(A, dtype=float), np.asarray(C, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or C.ndim != 2 or C.shape[1] != A.shape[0]:
        raise ValueError("Expected A with shape (n, n) and C with shape (p, n).")

    blocks = [C]
    for _ in range(1, A.shape[0]):
        blocks.append(blocks[-1] @ A)
    return np.vstack(blocks)


def matrix_rank(M, tol: float = 1e-10) -> int:
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
