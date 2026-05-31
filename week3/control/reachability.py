"""Affine steady-state reachable set under ``u ∈ [0, 1]^m``.

Spec §6 / §7:
    z_ss(u) = G u + z0
    G  = M C (I - A)^{-1} B            (slope of the readout vs constant input)
    z0 = M (C (I - A)^{-1} a + c)      (readout at the resting input u = 0)

The reachable set in readout space is the affine zonotope
    R = { G u + z0 : u ∈ [0, 1]^m }.
For m = 2 this is a parallelogram with the four vertices
    z0, z0 + G·e_1, z0 + G·e_2, z0 + G·(e_1+e_2).

Consumes only ``(A, B, C, M, a, c)``; no estimator imports here.
"""

from __future__ import annotations

import itertools

import numpy as np


def steady_state_gain(A, B, C, M, a=None, c=None):
    """Return ``(G, z0)`` for the affine reachable map ``z_ss = G u + z0``."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    M = np.asarray(M, dtype=float)
    n = A.shape[0]
    a = np.zeros(n) if a is None else np.asarray(a, dtype=float).reshape(n)
    c = np.zeros(C.shape[0]) if c is None else np.asarray(c, dtype=float).reshape(C.shape[0])
    IminusA_inv_B = np.linalg.solve(np.eye(n) - A, B)
    IminusA_inv_a = np.linalg.solve(np.eye(n) - A, a)
    G = M @ C @ IminusA_inv_B
    z0 = M @ (C @ IminusA_inv_a + c)
    return G, z0


def zonotope_vertices(G, z0=None):
    """Vertices of the affine zonotope ``{ G u + z0 : u ∈ [0,1]^m }``.

    Returns a ``(2^m, q)`` array.
    """
    G = np.asarray(G, dtype=float)
    q, m = G.shape
    z0 = np.zeros(q) if z0 is None else np.asarray(z0, dtype=float).reshape(q)
    corners = np.array(list(itertools.product([0.0, 1.0], repeat=m)))
    return corners @ G.T + z0


def feasibility(z, G, z0=None, tol: float = 1e-9):
    """Is ``z`` in the affine zonotope ``{ G u + z0 : u ∈ [0,1]^m }``?

    Returns ``(inside, u_ss)``. ``u_ss`` is the lstsq solution to
    ``G u + z0 = z``; when ``inside`` is True it satisfies ``0 ≤ u_ss ≤ 1``.
    """
    z = np.asarray(z, dtype=float)
    G = np.asarray(G, dtype=float)
    if z0 is not None:
        z = z - np.asarray(z0, dtype=float).reshape(z.shape)
    u, *_ = np.linalg.lstsq(G, z, rcond=None)
    inside = bool(np.all(u >= -tol) and np.all(u <= 1.0 + tol))
    return inside, u
