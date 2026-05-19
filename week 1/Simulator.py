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

        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError(f"A must be square; got {self.A.shape}")
        if self.B.ndim != 2:
            raise ValueError(f"B must be 2D; got {self.B.shape}")
        if self.C.ndim != 2:
            raise ValueError(f"C must be 2D; got {self.C.shape}")

        self.state_dim = self.A.shape[0]
        self.input_dim = self.B.shape[1]
        self.obs_dim = self.C.shape[0]
        self.x0 = np.zeros(self.state_dim) if x0 is None else np.asarray(x0, dtype=float)
        self.rng = np.random.default_rng(seed)

        self._validate_shapes()

    def _validate_shapes(self) -> None:
        n, m, p = self.state_dim, self.input_dim, self.obs_dim
        expected = {
            "A": ((n, n), self.A.shape),
            "B": ((n, m), self.B.shape),
            "C": ((p, n), self.C.shape),
            "Q": ((n, n), self.Q.shape),
            "R": ((p, p), self.R.shape),
            "x0": ((n,), self.x0.shape),
        }
        for name, (want, got) in expected.items():
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

        if x.shape != (self.state_dim,):
            raise ValueError(f"x must have shape {(self.state_dim,)}; got {x.shape}")
        if u.shape != (self.input_dim,):
            raise ValueError(f"u must have shape {(self.input_dim,)}; got {u.shape}")

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
        return {
            "x": np.stack([r["x"] for r in results]),
            "y": np.stack([r["y"] for r in results]),
            "u": np.stack([r["u"] for r in results]),
        }


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
    Q = 1e-3 * np.eye(4)
    R = 1e-2 * np.eye(obs_dim)
    x0 = np.array([1.0, 0.0, 0.5, -0.5])

    return Simulator(A, B, C, Q, R, x0=x0, seed=seed)
