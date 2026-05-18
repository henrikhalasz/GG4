"""
Simulator: a linear dynamical system (LDS) simulator for GG4 Week 1.

This module implements the partially observed linear state-space model
used throughout the project:

    x[t+1] = A x[t] + B u[t] + w[t],    w[t] ~ N(0, Q)   (state transition)
    y[t]   = C x[t] + o[t],             o[t] ~ N(0, R)    (observation model)

where

    x[t]  hidden latent state          shape (state_dim,)
    u[t]  input / stimulation command  shape (input_dim,)
    y[t]  noisy observation            shape (obs_dim,)
    A     state transition matrix      shape (state_dim, state_dim)
    B     input matrix                 shape (state_dim, input_dim)
    C     observation matrix           shape (obs_dim, state_dim)
    Q     process noise covariance     shape (state_dim, state_dim)
    R     observation noise covariance shape (obs_dim, obs_dim)

The module is intentionally self-contained: it depends only on NumPy and
the standard library. It is designed to be used directly by a demonstrator
without reading the notebook.

Typical usage
-------------
    import Simulator as sim

    system = sim.default_neural_system(seed=0)
    result = system.simulate(T=100)
    y = result["y"]                 # (100, obs_dim) neural observations

    # Multiple trials, ready for Illustrator (Trials, Timepoints, Neurons):
    trials = system.simulate_trials(trial_count=5, T=60)
    from Illustrator import Illustrator
    illustrator = Illustrator(trials["y"])

Both import styles are supported::

    from Simulator import Simulator
    from Simulator import LinearStateSpaceSimulator   # alias of Simulator
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Union

import numpy as np

# An input specification may be: nothing (zero input), an explicit array,
# or a callable mapping a timestep index to an input vector.
InputSpec = Union[None, np.ndarray, Callable[[int], np.ndarray]]


class Simulator:
    """
    Simulate a linear dynamical system with Gaussian process and
    observation noise.

    The system follows

        x[t+1] = A x[t] + B u[t] + w[t],    w[t] ~ N(0, Q)
        y[t]   = C x[t] + o[t],             o[t] ~ N(0, R)

    Axis conventions match the rest of the project: trajectories are
    stored time-major, and multi-trial output uses the
    (Trials, Timepoints, Neurons) layout expected by ``Illustrator``,
    where the number of "neurons" equals ``obs_dim``.

    Parameters
    ----------
    A : array-like, shape (state_dim, state_dim)
        State transition matrix.
    B : array-like, shape (state_dim, input_dim)
        Input matrix.
    C : array-like, shape (obs_dim, state_dim)
        Observation matrix.
    Q : array-like, shape (state_dim, state_dim)
        Process noise covariance (should be symmetric positive
        semi-definite).
    R : array-like, shape (obs_dim, obs_dim)
        Observation noise covariance (should be symmetric positive
        semi-definite).
    x0 : array-like, shape (state_dim,), optional
        Initial latent state. Defaults to a zero vector.
    seed : int or None, optional
        Seed for the internal ``numpy.random.default_rng`` generator.

    Attributes
    ----------
    A, B, C, Q, R : np.ndarray
        The system matrices as float64 arrays.
    x0 : np.ndarray, shape (state_dim,)
        The default initial state.
    state_dim, input_dim, obs_dim : int
        Dimensions inferred from the matrices.
    rng : numpy.random.Generator
        The random number generator used for all noise draws.

    Raises
    ------
    ValueError
        If any matrix or ``x0`` has an inconsistent shape.
    """

    def __init__(
        self,
        A,
        B,
        C,
        Q,
        R,
        x0=None,
        seed: Optional[int] = None,
    ):
        # Convert everything to float64 NumPy arrays up front so all
        # downstream maths is well-defined and avoids integer surprises.
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)

        # Dimensions are taken from A (state) and C (observation). B's
        # second axis defines the input dimension. These are validated
        # against every other matrix in _validate_shapes.
        if self.A.ndim != 2:
            raise ValueError(
                f"A must be a 2D matrix; got array with shape {self.A.shape}"
            )
        self.state_dim = self.A.shape[0]

        if self.B.ndim != 2:
            raise ValueError(
                f"B must be a 2D matrix; got array with shape {self.B.shape}"
            )
        self.input_dim = self.B.shape[1]

        if self.C.ndim != 2:
            raise ValueError(
                f"C must be a 2D matrix; got array with shape {self.C.shape}"
            )
        self.obs_dim = self.C.shape[0]

        # Default the initial state to the origin of the latent space.
        if x0 is None:
            self.x0 = np.zeros(self.state_dim, dtype=np.float64)
        else:
            self.x0 = np.asarray(x0, dtype=np.float64)

        self._validate_shapes()

        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _validate_shapes(self) -> None:
        """
        Validate the shapes of A, B, C, Q, R and x0 against each other.

        Raises
        ------
        ValueError
            With a clear message identifying the offending matrix and
            the shape that was expected.
        """
        n, m, p = self.state_dim, self.input_dim, self.obs_dim

        if self.A.shape != (n, n):
            raise ValueError(
                f"A must have shape (state_dim, state_dim) = ({n}, {n}); "
                f"got {self.A.shape}"
            )
        if self.B.shape != (n, m):
            raise ValueError(
                f"B must have shape (state_dim, input_dim) = ({n}, {m}); "
                f"got {self.B.shape}"
            )
        if self.C.shape != (p, n):
            raise ValueError(
                f"C must have shape (obs_dim, state_dim) = ({p}, {n}); "
                f"got {self.C.shape}"
            )
        if self.Q.shape != (n, n):
            raise ValueError(
                f"Q must have shape (state_dim, state_dim) = ({n}, {n}); "
                f"got {self.Q.shape}"
            )
        if self.R.shape != (p, p):
            raise ValueError(
                f"R must have shape (obs_dim, obs_dim) = ({p}, {p}); "
                f"got {self.R.shape}"
            )
        if self.x0.shape != (n,):
            raise ValueError(
                f"x0 must have shape (state_dim,) = ({n},); "
                f"got {self.x0.shape}"
            )

    # ------------------------------------------------------------------ #
    # Randomness control
    # ------------------------------------------------------------------ #
    def reset_seed(self, seed: Optional[int] = None) -> None:
        """
        Reset the internal random number generator.

        Parameters
        ----------
        seed : int or None, optional
            New seed. ``None`` reseeds from fresh OS entropy, i.e. the
            next simulation will be non-reproducible.
        """
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Input handling
    # ------------------------------------------------------------------ #
    def _make_input_sequence(
        self,
        T: int,
        U: InputSpec = None,
    ) -> np.ndarray:
        """
        Turn an input specification into a dense (T, input_dim) array.

        Parameters
        ----------
        T : int
            Number of timesteps.
        U : None, np.ndarray, or callable, optional
            - ``None``: zero input for every timestep.
            - array-like of shape ``(T, input_dim)``: used directly.
            - callable ``U(t)``: called for ``t = 0 .. T-1`` and must
              return an array-like of shape ``(input_dim,)``.

        Returns
        -------
        np.ndarray, shape (T, input_dim)
            The dense input sequence as float64.

        Raises
        ------
        ValueError
            If the resulting sequence does not have shape
            ``(T, input_dim)``.
        """
        m = self.input_dim

        if U is None:
            return np.zeros((T, m), dtype=np.float64)

        if callable(U):
            seq = np.empty((T, m), dtype=np.float64)
            for t in range(T):
                u_t = np.asarray(U(t), dtype=np.float64)
                if u_t.shape != (m,):
                    raise ValueError(
                        f"Callable U returned shape {u_t.shape} at t={t}; "
                        f"expected (input_dim,) = ({m},)"
                    )
                seq[t] = u_t
            return seq

        # Otherwise treat U as an explicit array.
        arr = np.asarray(U, dtype=np.float64)
        if arr.shape != (T, m):
            raise ValueError(
                f"U array must have shape (T, input_dim) = ({T}, {m}); "
                f"got {arr.shape}"
            )
        return arr

    # ------------------------------------------------------------------ #
    # Core dynamics
    # ------------------------------------------------------------------ #
    def step(self, x, u):
        """
        Advance the system by a single timestep.

        The update order is fixed so that the observation ``y`` is the
        measurement of the *current* state ``x`` (before it transitions):

            o      = observation noise   ~ N(0, R)
            y      = C @ x + o
            w      = process noise       ~ N(0, Q)
            x_next = A @ x + B @ u + w

        Parameters
        ----------
        x : array-like, shape (state_dim,)
            Current latent state.
        u : array-like, shape (input_dim,)
            Input applied at this timestep.

        Returns
        -------
        x_next : np.ndarray, shape (state_dim,)
            Next latent state.
        y : np.ndarray, shape (obs_dim,)
            Noisy observation of the current state.
        w : np.ndarray, shape (state_dim,)
            Process noise sample used in the transition.
        o : np.ndarray, shape (obs_dim,)
            Observation noise sample used in the measurement.

        Raises
        ------
        ValueError
            If ``x`` or ``u`` has the wrong shape.
        """
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)

        if x.shape != (self.state_dim,):
            raise ValueError(
                f"x must have shape (state_dim,) = ({self.state_dim},); "
                f"got {x.shape}"
            )
        if u.shape != (self.input_dim,):
            raise ValueError(
                f"u must have shape (input_dim,) = ({self.input_dim},); "
                f"got {u.shape}"
            )

        o = self.rng.multivariate_normal(np.zeros(self.obs_dim), self.R)
        y = self.C @ x + o

        w = self.rng.multivariate_normal(np.zeros(self.state_dim), self.Q)
        x_next = self.A @ x + self.B @ u + w

        return x_next, y, w, o

    def simulate(
        self,
        T: int,
        U: InputSpec = None,
        x0=None,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate a single trajectory of length ``T``.

        Parameters
        ----------
        T : int
            Number of timesteps. Must be a positive integer. The latent
            trajectory has ``T + 1`` entries (the initial state plus one
            per step); observations have ``T`` entries.
        U : None, np.ndarray, or callable, optional
            Input specification, see :meth:`_make_input_sequence`.
        x0 : array-like, shape (state_dim,), optional
            Initial state for this trajectory. Defaults to ``self.x0``.

        Returns
        -------
        dict
            Dictionary with keys::

                "x" : np.ndarray, shape (T + 1, state_dim)  latent states
                "y" : np.ndarray, shape (T,     obs_dim)    observations
                "u" : np.ndarray, shape (T,     input_dim)  inputs used
                "w" : np.ndarray, shape (T,     state_dim)  process noise
                "o" : np.ndarray, shape (T,     obs_dim)    obs. noise

        Raises
        ------
        ValueError
            If ``T`` is not a positive integer, or if ``x0`` / ``U``
            have inconsistent shapes.
        """
        if not isinstance(T, (int, np.integer)) or isinstance(T, bool):
            raise ValueError(f"T must be an integer; got {type(T).__name__}")
        if T <= 0:
            raise ValueError(f"T must be a positive integer; got {T}")

        if x0 is None:
            x = self.x0.copy()
        else:
            x = np.asarray(x0, dtype=np.float64)
            if x.shape != (self.state_dim,):
                raise ValueError(
                    f"x0 must have shape (state_dim,) = "
                    f"({self.state_dim},); got {x.shape}"
                )

        u_seq = self._make_input_sequence(T, U)

        x_traj = np.empty((T + 1, self.state_dim), dtype=np.float64)
        y_traj = np.empty((T, self.obs_dim), dtype=np.float64)
        w_traj = np.empty((T, self.state_dim), dtype=np.float64)
        o_traj = np.empty((T, self.obs_dim), dtype=np.float64)

        x_traj[0] = x
        for t in range(T):
            x_next, y_t, w_t, o_t = self.step(x_traj[t], u_seq[t])
            x_traj[t + 1] = x_next
            y_traj[t] = y_t
            w_traj[t] = w_t
            o_traj[t] = o_t

        return {
            "x": x_traj,
            "y": y_traj,
            "u": u_seq,
            "w": w_traj,
            "o": o_traj,
        }

    def simulate_trials(
        self,
        trial_count: int,
        T: int,
        U: InputSpec = None,
        x0=None,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate several independent trials sharing the same parameters.

        Each trial uses the same input specification but draws fresh,
        independent noise from the shared RNG, so trials differ only by
        noise (and, for a callable ``U``, by whatever the callable
        returns — it is re-evaluated per trial).

        Parameters
        ----------
        trial_count : int
            Number of independent trials. Must be a positive integer.
        T : int
            Timesteps per trial. Must be a positive integer.
        U : None, np.ndarray, or callable, optional
            Input specification, see :meth:`_make_input_sequence`.
        x0 : array-like, shape (state_dim,), optional
            Initial state shared by every trial. Defaults to
            ``self.x0``.

        Returns
        -------
        dict
            Dictionary with keys::

                "x" : np.ndarray, (trial_count, T + 1, state_dim)
                "y" : np.ndarray, (trial_count, T,     obs_dim)
                "u" : np.ndarray, (trial_count, T,     input_dim)

        Notes
        -----
        The ``"y"`` array uses the (Trials, Timepoints, Neurons) layout
        expected by ``Illustrator``; here ``obs_dim`` is the number of
        neurons. ``"u"`` is stacked per trial for convenience even
        though it is identical across trials for a fixed array ``U``.

        Raises
        ------
        ValueError
            If ``trial_count`` or ``T`` is not a positive integer, or if
            ``U`` / ``x0`` have inconsistent shapes.
        """
        if (
            not isinstance(trial_count, (int, np.integer))
            or isinstance(trial_count, bool)
        ):
            raise ValueError(
                f"trial_count must be an integer; "
                f"got {type(trial_count).__name__}"
            )
        if trial_count <= 0:
            raise ValueError(
                f"trial_count must be a positive integer; got {trial_count}"
            )

        x_all = np.empty(
            (trial_count, T + 1, self.state_dim), dtype=np.float64
        )
        y_all = np.empty((trial_count, T, self.obs_dim), dtype=np.float64)
        u_all = np.empty((trial_count, T, self.input_dim), dtype=np.float64)

        for r in range(trial_count):
            result = self.simulate(T, U=U, x0=x0)
            x_all[r] = result["x"]
            y_all[r] = result["y"]
            u_all[r] = result["u"]

        return {"x": x_all, "y": y_all, "u": u_all}


# Both names refer to the same class so either import style works.
LinearStateSpaceSimulator = Simulator


# ====================================================================== #
# Module-level input-pattern helpers
# ====================================================================== #
def _check_T(T: int) -> None:
    """Raise ValueError unless T is a positive integer."""
    if not isinstance(T, (int, np.integer)) or isinstance(T, bool):
        raise ValueError(f"T must be an integer; got {type(T).__name__}")
    if T <= 0:
        raise ValueError(f"T must be a positive integer; got {T}")


def _check_input_dim(input_dim: int) -> None:
    """Raise ValueError unless input_dim is a positive integer."""
    if (
        not isinstance(input_dim, (int, np.integer))
        or isinstance(input_dim, bool)
    ):
        raise ValueError(
            f"input_dim must be an integer; got {type(input_dim).__name__}"
        )
    if input_dim <= 0:
        raise ValueError(
            f"input_dim must be a positive integer; got {input_dim}"
        )


def _check_channel(channel: int, input_dim: int) -> None:
    """Raise ValueError unless 0 <= channel < input_dim."""
    if not isinstance(channel, (int, np.integer)) or isinstance(channel, bool):
        raise ValueError(
            f"channel must be an integer; got {type(channel).__name__}"
        )
    if not (0 <= channel < input_dim):
        raise ValueError(
            f"channel must be in [0, input_dim - 1] = "
            f"[0, {input_dim - 1}]; got {channel}"
        )


def zero_input(T: int, input_dim: int) -> np.ndarray:
    """
    Return an all-zero input sequence.

    Parameters
    ----------
    T : int
        Number of timesteps (positive integer).
    input_dim : int
        Number of input channels (positive integer).

    Returns
    -------
    np.ndarray, shape (T, input_dim)
        Array of zeros.
    """
    _check_T(T)
    _check_input_dim(input_dim)
    return np.zeros((T, input_dim), dtype=np.float64)


def pulse_input(
    T: int,
    input_dim: int,
    channel: int = 0,
    start: int = 10,
    duration: int = 5,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    Return an input sequence with a single rectangular pulse.

    The pulse is placed on one channel: it is ``amplitude`` for the
    ``duration`` timesteps starting at ``start`` and zero everywhere
    else.

    Parameters
    ----------
    T : int
        Number of timesteps (positive integer).
    input_dim : int
        Number of input channels (positive integer).
    channel : int, default 0
        Channel the pulse is applied to; must satisfy
        ``0 <= channel < input_dim``.
    start : int, default 10
        Timestep at which the pulse begins; must satisfy
        ``0 <= start < T``.
    duration : int, default 5
        Number of timesteps the pulse lasts; must be ``>= 1``. The
        pulse is clipped at ``T`` if ``start + duration`` exceeds it.
    amplitude : float, default 1.0
        Pulse height.

    Returns
    -------
    np.ndarray, shape (T, input_dim)
        The input sequence.

    Raises
    ------
    ValueError
        If any argument is out of range.
    """
    _check_T(T)
    _check_input_dim(input_dim)
    _check_channel(channel, input_dim)
    if not isinstance(start, (int, np.integer)) or isinstance(start, bool):
        raise ValueError(
            f"start must be an integer; got {type(start).__name__}"
        )
    if not (0 <= start < T):
        raise ValueError(
            f"start must be in [0, T - 1] = [0, {T - 1}]; got {start}"
        )
    if (
        not isinstance(duration, (int, np.integer))
        or isinstance(duration, bool)
    ):
        raise ValueError(
            f"duration must be an integer; got {type(duration).__name__}"
        )
    if duration < 1:
        raise ValueError(f"duration must be >= 1; got {duration}")

    seq = np.zeros((T, input_dim), dtype=np.float64)
    end = min(start + duration, T)
    seq[start:end, channel] = float(amplitude)
    return seq


def sinusoidal_input(
    T: int,
    input_dim: int,
    channel: int = 0,
    amplitude: float = 1.0,
    period: float = 30,
    phase: float = 0.0,
) -> np.ndarray:
    """
    Return a sinusoidal drive on one channel.

    The signal on ``channel`` is

        amplitude * sin(2*pi * t / period + phase),   t = 0 .. T-1

    and zero on all other channels.

    Parameters
    ----------
    T : int
        Number of timesteps (positive integer).
    input_dim : int
        Number of input channels (positive integer).
    channel : int, default 0
        Driven channel; must satisfy ``0 <= channel < input_dim``.
    amplitude : float, default 1.0
        Peak amplitude of the sinusoid.
    period : float, default 30
        Oscillation period in timesteps; must be ``> 0``.
    phase : float, default 0.0
        Phase offset in radians.

    Returns
    -------
    np.ndarray, shape (T, input_dim)
        The input sequence.

    Raises
    ------
    ValueError
        If ``period <= 0`` or ``channel`` is invalid.
    """
    _check_T(T)
    _check_input_dim(input_dim)
    _check_channel(channel, input_dim)
    if period <= 0:
        raise ValueError(f"period must be > 0; got {period}")

    t = np.arange(T, dtype=np.float64)
    seq = np.zeros((T, input_dim), dtype=np.float64)
    seq[:, channel] = amplitude * np.sin(2.0 * np.pi * t / period + phase)
    return seq


def random_input(
    T: int,
    input_dim: int,
    amplitude: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Return a Gaussian random input sequence.

    Every entry is drawn independently from ``N(0, amplitude**2)``.

    Parameters
    ----------
    T : int
        Number of timesteps (positive integer).
    input_dim : int
        Number of input channels (positive integer).
    amplitude : float, default 1.0
        Standard deviation of the Gaussian noise.
    seed : int or None, optional
        Seed for a local ``numpy.random.default_rng``; pass an integer
        for a reproducible sequence.

    Returns
    -------
    np.ndarray, shape (T, input_dim)
        The input sequence.
    """
    _check_T(T)
    _check_input_dim(input_dim)
    rng = np.random.default_rng(seed)
    return amplitude * rng.standard_normal((T, input_dim))


def channel_sweep_input(
    T: int,
    input_dim: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    Activate each input channel in turn, one at a time.

    The timeline is split into ``input_dim`` consecutive equal blocks;
    during block ``k`` only channel ``k`` is held at ``amplitude``. This
    makes it easy to read off, from the observations, which states /
    neurons each input channel drives.

    Parameters
    ----------
    T : int
        Number of timesteps (positive integer).
    input_dim : int
        Number of input channels (positive integer).
    amplitude : float, default 1.0
        Activation level for the currently active channel.

    Returns
    -------
    np.ndarray, shape (T, input_dim)
        The input sequence. If ``T`` is not divisible by ``input_dim``
        the final channel's block absorbs the remainder.
    """
    _check_T(T)
    _check_input_dim(input_dim)
    seq = np.zeros((T, input_dim), dtype=np.float64)
    block = T // input_dim
    for k in range(input_dim):
        lo = k * block
        hi = T if k == input_dim - 1 else (k + 1) * block
        seq[lo:hi, k] = float(amplitude)
    return seq


def mixed_input(
    T: int,
    input_dim: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Return a demonstration input combining several patterns.

    The sequence superimposes:

    - a rectangular pulse on channel 0,
    - a sinusoidal drive on channel 1 (only if ``input_dim >= 2``),
    - a small Gaussian background on every channel.

    This is a convenient single call for showing off how the system
    responds to a structured, multi-component stimulus.

    Parameters
    ----------
    T : int
        Number of timesteps (positive integer).
    input_dim : int
        Number of input channels (positive integer).
    seed : int or None, optional
        Seed for the background noise component.

    Returns
    -------
    np.ndarray, shape (T, input_dim)
        The combined input sequence.
    """
    _check_T(T)
    _check_input_dim(input_dim)

    seq = np.zeros((T, input_dim), dtype=np.float64)

    # Pulse on channel 0, sized relative to the trajectory length.
    pulse_start = max(1, T // 5)
    pulse_dur = max(1, T // 10)
    seq += pulse_input(
        T,
        input_dim,
        channel=0,
        start=min(pulse_start, T - 1),
        duration=pulse_dur,
        amplitude=1.0,
    )

    # Sinusoid on channel 1, if it exists.
    if input_dim >= 2:
        seq += sinusoidal_input(
            T, input_dim, channel=1, amplitude=0.5, period=max(2, T // 4)
        )

    # Small random background on every channel.
    rng = np.random.default_rng(seed)
    seq += 0.05 * rng.standard_normal((T, input_dim))
    return seq


# ====================================================================== #
# Linear-systems analysis helpers
# ====================================================================== #
def controllability_matrix(A, B) -> np.ndarray:
    """
    Build the controllability matrix of an LDS.

    The controllability matrix is the horizontal stack

        [ B , A B , A^2 B , ... , A^(n-1) B ]

    where ``n`` is the state dimension. The pair ``(A, B)`` is
    controllable iff this matrix has full row rank ``n``.

    Parameters
    ----------
    A : array-like, shape (n, n)
        State transition matrix.
    B : array-like, shape (n, m)
        Input matrix.

    Returns
    -------
    np.ndarray, shape (n, n * m)
        The controllability matrix.

    Raises
    ------
    ValueError
        If ``A`` is not square or ``B``'s first axis does not match.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be a square matrix; got shape {A.shape}")
    n = A.shape[0]
    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError(
            f"B must have shape (n, m) with n = {n}; got {B.shape}"
        )

    blocks = [B]
    for _ in range(1, n):
        blocks.append(A @ blocks[-1])
    return np.hstack(blocks)


def observability_matrix(A, C) -> np.ndarray:
    """
    Build the observability matrix of an LDS.

    The observability matrix is the vertical stack

        [ C ; C A ; C A^2 ; ... ; C A^(n-1) ]

    where ``n`` is the state dimension. The pair ``(A, C)`` is
    observable iff this matrix has full column rank ``n``.

    Parameters
    ----------
    A : array-like, shape (n, n)
        State transition matrix.
    C : array-like, shape (p, n)
        Observation matrix.

    Returns
    -------
    np.ndarray, shape (n * p, n)
        The observability matrix.

    Raises
    ------
    ValueError
        If ``A`` is not square or ``C``'s second axis does not match.
    """
    A = np.asarray(A, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be a square matrix; got shape {A.shape}")
    n = A.shape[0]
    if C.ndim != 2 or C.shape[1] != n:
        raise ValueError(
            f"C must have shape (p, n) with n = {n}; got {C.shape}"
        )

    blocks = [C]
    for _ in range(1, n):
        blocks.append(blocks[-1] @ A)
    return np.vstack(blocks)


def matrix_rank(M, tol: float = 1e-10) -> int:
    """
    Numerical rank of a matrix.

    Thin wrapper around :func:`numpy.linalg.matrix_rank` with an
    explicit absolute singular-value tolerance, so that controllability
    / observability rank tests behave predictably for nearly-singular
    systems.

    Parameters
    ----------
    M : array-like
        The matrix whose rank is wanted.
    tol : float, default 1e-10
        Absolute threshold below which singular values are treated as
        zero.

    Returns
    -------
    int
        The numerical rank.
    """
    M = np.asarray(M, dtype=np.float64)
    return int(np.linalg.matrix_rank(M, tol=tol))


# ====================================================================== #
# Ready-made default system
# ====================================================================== #
def default_neural_system(
    seed: Optional[int] = None,
    obs_dim: int = 16,
) -> Simulator:
    """
    Construct a ready-to-use, interpretable neural LDS.

    The returned :class:`Simulator` has four latent states with
    deliberately distinct, interpretable dynamics:

    - **states 0 & 1** — a damped oscillatory pair (a rotation of
      radius ~0.97, period ~20 steps): produces decaying oscillations.
    - **state 2** — a slow persistent mode (eigenvalue ~0.98): decays
      very slowly, so it carries long-lived structure.
    - **state 3** — a fast decaying mode (eigenvalue ~0.75): forgets
      its input quickly.

    Mild one-way coupling feeds the oscillatory pair into the slow and
    fast modes, so the modes are not perfectly independent. The system
    is stable: every eigenvalue of ``A`` lies strictly inside the unit
    circle.

    Two input channels are wired so that each channel primarily drives a
    different part of the latent space (channel 0 → oscillatory state 0
    and the slow mode; channel 1 → oscillatory state 1 and the fast
    mode). The observation matrix ``C`` is drawn once from a standard
    normal using the seeded RNG, so each "neuron" is a random linear
    mixture of the latent state. ``Q`` and ``R`` are small diagonal
    covariances, and the initial state ``x0`` is non-zero so there is
    visible transient behaviour even with no input.

    Parameters
    ----------
    seed : int or None, optional
        Seed controlling both the random ``C`` matrix and the
        simulator's noise stream, for full reproducibility.
    obs_dim : int, default 16
        Number of observation channels ("neurons"). Must be a positive
        integer. The default of 16 matches the example dataset.

    Returns
    -------
    Simulator
        A configured simulator instance, ready for ``simulate`` or
        ``simulate_trials``.

    Raises
    ------
    ValueError
        If ``obs_dim`` is not a positive integer.

    Notes / assumptions
    -------------------
    - State dimension is fixed at 4 and input dimension at 2.
    - Oscillatory block radius 0.97, period 20 timesteps.
    - Slow-mode eigenvalue 0.98, fast-mode eigenvalue 0.75.
    - Coupling gain 0.05 from each oscillatory state into the
      slow / fast modes (lower-triangular, so eigenvalues are
      unchanged and stability is preserved).
    - ``Q = 1e-3 * I_4``, ``R = 1e-2 * I_obs_dim``.
    - ``x0 = [1.0, 0.0, 0.5, -0.5]``.
    """
    if not isinstance(obs_dim, (int, np.integer)) or isinstance(obs_dim, bool):
        raise ValueError(
            f"obs_dim must be an integer; got {type(obs_dim).__name__}"
        )
    if obs_dim <= 0:
        raise ValueError(f"obs_dim must be a positive integer; got {obs_dim}")

    state_dim = 4
    input_dim = 2

    # --- A: block-structured, interpretable, stable -------------------
    radius = 0.97
    period = 20.0
    theta = 2.0 * np.pi / period
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    A = np.zeros((state_dim, state_dim), dtype=np.float64)
    # Damped oscillatory 2x2 rotation block on states 0 and 1.
    A[0, 0] = radius * cos_t
    A[0, 1] = -radius * sin_t
    A[1, 0] = radius * sin_t
    A[1, 1] = radius * cos_t
    # Slow persistent mode and fast decaying mode on the diagonal.
    A[2, 2] = 0.98
    A[3, 3] = 0.75
    # Mild one-way coupling: oscillatory states feed the slow / fast
    # modes. Placing these strictly below the (block-)diagonal keeps A
    # block-lower-triangular, so the eigenvalues - and hence stability -
    # are exactly those of the diagonal blocks.
    A[2, 0] = 0.05
    A[3, 1] = 0.05

    # --- B: each input channel targets different latent states --------
    B = np.array(
        [
            [1.0, 0.0],   # channel 0 -> oscillatory state 0
            [0.0, 1.0],   # channel 1 -> oscillatory state 1
            [0.5, 0.0],   # channel 0 -> slow mode
            [0.0, 0.5],   # channel 1 -> fast mode
        ],
        dtype=np.float64,
    )

    # --- C: reproducible random projection to observations ------------
    rng = np.random.default_rng(seed)
    C = rng.standard_normal((obs_dim, state_dim))

    # --- Small diagonal noise covariances -----------------------------
    Q = 1e-3 * np.eye(state_dim, dtype=np.float64)
    R = 1e-2 * np.eye(obs_dim, dtype=np.float64)

    # --- Non-zero initial state for a visible transient ---------------
    x0 = np.array([1.0, 0.0, 0.5, -0.5], dtype=np.float64)

    return Simulator(A, B, C, Q, R, x0=x0, seed=seed)
