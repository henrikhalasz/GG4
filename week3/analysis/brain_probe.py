"""Brain catalogue helpers (spec §2 — the initial investigation).

Each function answers one question about the real Brain (or any plant with
the same ``measure() / next_state(u) / input_dim`` surface) and is composed
into the figure / numbers by ``experiments/m1_initial_investigation.py``.

Lives in :mod:`analysis` because the catalogue is exploratory and informs
the *yardstick* in :mod:`analysis.ground_truth` — it is **not** part of the
deployed honest pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence

import numpy as np

from .ground_truth import (
    build_ground_truth_model,
    measurement_noise_covariance,
    noise_free_markov_parameters,
    _collect_outputs,
    _zero_input_fn,
    era_ho_kalman,
    GroundTruthModel,
)


PlantFactory = Callable[[int], object]


# ----------------------------------------------------------------------------
# 1. Dimensions and basic contract.
# ----------------------------------------------------------------------------
@dataclass
class PlantContract:
    input_dim: int
    obs_dim: int
    first_y: np.ndarray   # the very first measure() under no perturbation


def discover_contract(plant_factory: PlantFactory, seed: int = 0) -> PlantContract:
    """Probe a fresh plant to record its dimensions and a sample first y."""
    plant = plant_factory(seed)
    y = np.asarray(plant.measure(), dtype=float)
    return PlantContract(input_dim=int(plant.input_dim),
                         obs_dim=int(y.size),
                         first_y=y)


# ----------------------------------------------------------------------------
# 2. Observation-noise scale.
# ----------------------------------------------------------------------------
def observation_noise_scale(
    plant_factory: PlantFactory, seed: int = 0, n_samples: int = 2000
) -> dict:
    """Diagonal and Frobenius scale of the observation-noise covariance ``R``.

    Repeats ``measure()`` at the frozen initial state — exactly the
    quantity needed for the steady-state Kalman gain.
    """
    R = measurement_noise_covariance(plant_factory, seed=seed,
                                     n_samples=n_samples)
    diag = np.sqrt(np.diag(R))
    return dict(
        R=R,
        per_channel_std=diag,
        median_std=float(np.median(diag)),
        frobenius=float(np.linalg.norm(R, "fro")),
    )


# ----------------------------------------------------------------------------
# 3. Autonomous behaviour (u ≡ 0).
# ----------------------------------------------------------------------------
def autonomous_trajectory(
    plant_factory: PlantFactory, seed: int = 0, T: int = 500
) -> np.ndarray:
    """Return ``Y`` of shape ``(T, p)`` under zero input from initialisation."""
    plant = plant_factory(seed)
    y0 = np.asarray(plant.measure(), dtype=float)
    m = int(plant.input_dim)
    return _collect_outputs(plant_factory, seed=seed, T=T,
                            u_fn=_zero_input_fn(m))


# ----------------------------------------------------------------------------
# 4. Order via Hankel singular-value knee.
# ----------------------------------------------------------------------------
@dataclass
class HankelKnee:
    sigma: np.ndarray            # all singular values of the Hankel matrix
    knee: int                    # heuristic order estimate (sharpest drop)
    knee_ratio: float            # σ[knee] / σ[knee−1]
    H: np.ndarray                # the underlying Markov params (T+1, p, m)


def hankel_singular_values(
    plant_factory: PlantFactory,
    seed: int = 0,
    T: int = 300,
    n_probe: int = 12,
) -> HankelKnee:
    """Run the differencing protocol and report the Hankel singular spectrum.

    ``n_probe`` is the maximum order considered; the routine returns enough
    singular values to see the knee. The "knee" is the largest ratio drop
    ``σ_k / σ_{k−1}`` in the first ``n_probe`` values (the spec expects this
    to land at ``k = 6``).
    """
    H = noise_free_markov_parameters(plant_factory, seed=seed, T=T)
    A, B, C, sigma = era_ho_kalman(H, n=n_probe)  # we discard A,B,C — only σ used
    sigma_head = sigma[:n_probe + 1]
    ratios = sigma_head[1:] / np.maximum(sigma_head[:-1], 1e-300)
    knee = int(np.argmin(ratios)) + 1  # +1 because ratio index k = σ[k+1]/σ[k]
    return HankelKnee(sigma=sigma, knee=knee,
                      knee_ratio=float(ratios[knee - 1]), H=H)


# ----------------------------------------------------------------------------
# 5. Seed-invariance — eigenvalues + DC gain across multiple seeds.
# ----------------------------------------------------------------------------
@dataclass
class SeedInvarianceReport:
    seeds: List[int]
    models: List[GroundTruthModel]
    eig_sorted: np.ndarray             # (n_seeds, n) complex
    eig_pairwise_max_err: float        # max over (i, j) of ||sort(λ_i) − sort(λ_j)||∞
    G: np.ndarray                      # (n_seeds, p, m) DC gains
    G_pairwise_max_relerr: float       # max over (i, j) of ||G_i − G_j|| / ||G_i||


def seed_invariance(
    plant_factory: PlantFactory,
    seeds: Sequence[int],
    n: int = 6,
    T_impulse: int = 300,
) -> SeedInvarianceReport:
    """Build ground-truth models at each seed and report basis-free agreement.

    The spec predicts the Brain has a **fixed** linear-Gaussian structure;
    the seed only changes the noise realisation. Eigenvalues and DC gain
    should match to numerical precision across seeds.
    """
    models = [build_ground_truth_model(plant_factory, seed=s,
                                       n=n, T_impulse=T_impulse)
              for s in seeds]
    eig = np.array([np.sort_complex(np.linalg.eigvals(m.A)) for m in models])

    # Pairwise max eigenvalue error (handles small re-ordering robustly via sort).
    n_seeds = len(seeds)
    eig_err = 0.0
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            eig_err = max(eig_err, float(np.abs(eig[i] - eig[j]).max()))

    G = np.array([m.C @ np.linalg.solve(np.eye(n) - m.A, m.B) for m in models])
    G_err = 0.0
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            denom = max(np.linalg.norm(G[i]), 1e-12)
            G_err = max(G_err, float(np.linalg.norm(G[i] - G[j]) / denom))

    return SeedInvarianceReport(
        seeds=list(map(int, seeds)),
        models=models,
        eig_sorted=eig,
        eig_pairwise_max_err=eig_err,
        G=G,
        G_pairwise_max_relerr=G_err,
    )


# ----------------------------------------------------------------------------
# 6. Control authority — singular values of the DC gain.
# ----------------------------------------------------------------------------
@dataclass
class ControlAuthorityReport:
    G: np.ndarray                # (p, m) full DC gain in observation space
    sigma: np.ndarray            # (min(p, m),) singular values
    rank_ratio: float            # σ[1] / σ[0] — closer to 0 ⇒ more rank-1
    dominant_output: np.ndarray  # top left-singular vector of G (p,)
    reachable_extent: float      # max ||G u|| over u ∈ [0, 1]^m along dominant dir
    reachable_floor: float       # min ||G u|| at u = 0 along dominant dir


def control_authority(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> ControlAuthorityReport:
    """Catalogue the input→output DC-gain rank structure.

    The spec expects the Brain's ``G`` to be **near rank-1** — only one
    sustainable output direction is freely controllable. This routine
    reports the singular spectrum and the reachable range along the
    dominant output direction.
    """
    n = A.shape[0]
    G = C @ np.linalg.solve(np.eye(n) - A, B)
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    rank_ratio = float(s[1] / s[0]) if s.size >= 2 else 0.0
    dominant = U[:, 0]

    # Reachable extent along the dominant output direction. Since u ∈ [0,1]^m,
    # the projection on the dominant direction is u_dir = dominant.T @ G @ u =
    # (dominant.T @ G) @ u, maximised by u = (signs of the row) clipped to [0,1].
    g_dir = dominant @ G  # shape (m,)
    u_max = np.clip(np.sign(g_dir), 0.0, 1.0)
    extent = float(g_dir @ u_max)
    floor = 0.0  # u = 0 is the reference

    return ControlAuthorityReport(
        G=G, sigma=s, rank_ratio=rank_ratio, dominant_output=dominant,
        reachable_extent=extent, reachable_floor=floor,
    )
