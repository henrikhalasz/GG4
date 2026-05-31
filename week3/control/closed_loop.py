"""Closed-loop driver (affine LGSSM form).

Convention (spec §6):
    - measure y at time t
    - predict observer with u_prev (input applied at t-1, drove state into t)
    - update observer with y (the observer subtracts its own offset c)
    - compute u = controller(x_hat, ref)   (or controller(y, ref) for raw-y)
    - apply u to plant (advances state to t+1)
    - u_prev := u

At t = 0, ``u_prev`` starts as zeros (resting input).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def run_closed_loop(
    plant,
    controller,
    observer,
    T: int,
    *,
    ref_fn: Optional[Callable[[int], np.ndarray]] = None,
) -> dict:
    n_u = plant.input_dim
    n_x = observer.n if hasattr(observer, "n") else 0
    n_y = observer.p if hasattr(observer, "p") else getattr(plant, "obs_dim", None)
    if n_y is None:
        raise RuntimeError("Cannot determine observation dim — pass an observer with .p or a plant with .obs_dim.")

    has_true = plant.true_state() is not None
    Xtrue = None

    Y = np.empty((T, n_y))
    U = np.empty((T, n_u))
    Xhat = np.empty((T, n_x)) if n_x else None
    if has_true:
        Xtrue = np.empty((T, plant.true_state().size))

    Ref = None
    if ref_fn is not None:
        ref0 = ref_fn(0)
        if ref0 is not None:
            ref0 = np.asarray(ref0, dtype=float)
            Ref = np.empty((T, ref0.size))

    observer.reset()
    controller.reset()
    u_prev = np.zeros(n_u)

    for t in range(T):
        y = np.asarray(plant.measure(), dtype=float)
        Y[t] = y
        if Xtrue is not None:
            Xtrue[t] = plant.true_state()

        x_hat = observer.filter_step(y, u_prev)
        if Xhat is not None:
            Xhat[t] = x_hat

        ref = ref_fn(t) if ref_fn is not None else None
        if ref is not None and Ref is not None:
            Ref[t] = np.asarray(ref, dtype=float)

        # Optional observation hook (e.g. OffsetFreeMPC's d_hat update). Called
        # AFTER the observer's x_hat update and BEFORE compute, so the
        # controller can fuse y and the up-to-date x_hat (spec §4).
        if hasattr(controller, "observe"):
            controller.observe(y, x_hat, u_prev)

        if getattr(controller, "uses_raw_y", False):
            u = controller.compute(y, ref)
        else:
            u = controller.compute(x_hat, ref)
        U[t] = u

        plant.next_state(u)
        u_prev = u

    return {"y": Y, "u": U, "x_hat": Xhat, "x_true": Xtrue, "ref": Ref}
