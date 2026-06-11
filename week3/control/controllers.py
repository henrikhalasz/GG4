"""Controllers — all in affine LGSSM form.

Every controller has ``reset()`` and ``compute(x_hat, ref) -> u``.

Spec §6 affine forms used here:
    Equilibrium under constant u:    x_ss = (I - A)⁻¹ (B u + a)
    Readout at equilibrium:          z_ss = M (C x_ss + c)
    Feedforward for target r ∈ ℝ^q:  solve
        [[I - A,  -B], [M C,  0]]  [x_ss; u_ff]  =  [a; r - M c]
    LQR gain:                        K via DARE on (A, B, Qx, Ru)
    Control law:                     u = clip(u_ff - K (x_hat - x_ss), 0, 1)

``ProportionalFeedback`` is the model-free baseline (consumes raw y).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize
from scipy.signal import place_poles


def _solve_feedforward(A, B, C, M, a, c, ref):
    """Solve the spec-§6 KKT block for (x_ss, u_ff). Falls back to lstsq.

    Returns
    -------
    x_ss : (n,) array
    u_ff : (m,) array  (NOT pre-clipped — caller decides)
    """
    n = A.shape[0]
    m = B.shape[1]
    q = M.shape[0]
    top = np.hstack([np.eye(n) - A, -B])
    bot = np.hstack([M @ C, np.zeros((q, m))])
    Mff = np.vstack([top, bot])
    rhs = np.concatenate([a, ref - M @ c])
    sol, *_ = np.linalg.lstsq(Mff, rhs, rcond=None)
    return sol[:n], sol[n:]


class OpenLoop:
    """Constant input ``u_const`` every step (default zero — resting input)."""

    uses_raw_y = False

    def __init__(self, u_const=None, input_dim: int = 2):
        if u_const is None:
            u_const = np.zeros(input_dim)
        self.u = np.clip(np.asarray(u_const, dtype=float), 0.0, 1.0)

    def reset(self) -> None:
        pass

    def compute(self, x_hat=None, ref=None) -> np.ndarray:
        return self.u.copy()


class ProportionalFeedback:
    """Model-free proportional feedback on raw observation.

    Spec §6: ``u = clip(u0 - Kp · M (y - r), 0, 1)``.
    ``u0`` defaults to zero (resting input).
    """

    uses_raw_y = True

    def __init__(self, Kp, M, ref=None, u0=None):
        self.Kp = np.asarray(Kp, dtype=float)
        self.M = np.asarray(M, dtype=float)
        q = self.M.shape[0]
        self.ref = np.zeros(q) if ref is None else np.asarray(ref, dtype=float)
        m = self.Kp.shape[0]
        self.u0 = np.zeros(m) if u0 is None else np.asarray(u0, dtype=float)

    def reset(self) -> None:
        pass

    def compute(self, y, ref=None) -> np.ndarray:
        if ref is not None:
            self.ref = np.asarray(ref, dtype=float)
        z = self.M @ np.asarray(y, dtype=float)
        u = self.u0 - self.Kp @ (z - self.ref)
        return np.clip(u, 0.0, 1.0)


class LQG:
    """Clipped LQR with affine feedforward.

    K from DARE on (A, B, Qx=(MC)ᵀ(MC), Ru=ρI) — offsets `a, c` do not enter K.
    Feedforward (x_ss, u_ff) from the spec KKT block.
    """

    uses_raw_y = False

    def __init__(self, A, B, C, M, a=None, c=None, rho: float = 1.0, ref=None):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.M = np.asarray(M, dtype=float)
        self.rho = float(rho)
        self.n = self.A.shape[0]
        self.m = self.B.shape[1]
        self.q = self.M.shape[0]
        self.a = np.zeros(self.n) if a is None else np.asarray(a, dtype=float).reshape(self.n)
        self.c = np.zeros(self.C.shape[0]) if c is None else np.asarray(c, dtype=float).reshape(self.C.shape[0])

        MC = self.M @ self.C
        Qx = MC.T @ MC + 1e-8 * np.eye(self.n)
        Ru = self.rho * np.eye(self.m)
        P = solve_discrete_are(self.A, self.B, Qx, Ru)
        self.P = P
        self.K = np.linalg.solve(Ru + self.B.T @ P @ self.B, self.B.T @ P @ self.A)

        ref0 = np.zeros(self.q) if ref is None else np.asarray(ref, dtype=float)
        self._ref_cache = None
        self.x_ss = np.zeros(self.n)
        self.u_ff = np.zeros(self.m)
        self.set_reference(ref0)

    def set_reference(self, ref) -> None:
        ref = np.asarray(ref, dtype=float)
        if (self._ref_cache is not None
                and ref.shape == self._ref_cache.shape
                and np.allclose(ref, self._ref_cache)):
            return
        self.x_ss, self.u_ff = _solve_feedforward(
            self.A, self.B, self.C, self.M, self.a, self.c, ref
        )
        self._ref_cache = ref.copy()

    def reset(self) -> None:
        pass

    def compute(self, x_hat, ref=None) -> np.ndarray:
        if ref is not None:
            self.set_reference(ref)
        x_hat = np.asarray(x_hat, dtype=float)
        u = self.u_ff - self.K @ (x_hat - self.x_ss)
        return np.clip(u, 0.0, 1.0)


class PolePlacement:
    """State-feedback by direct eigenvalue assignment.

    Needs a feedforward to handle the affine offset for setpoints:
        u = clip(u_ff - K (x_hat - x_ss), 0, 1)
    where (x_ss, u_ff) come from the spec KKT block.

    Falls back to a moderate LQR if ``place_poles`` fails.
    """

    uses_raw_y = False

    def __init__(self, A, B, C, M, target_poles, a=None, c=None, ref=None):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.M = np.asarray(M, dtype=float)
        target_poles = np.asarray(target_poles, dtype=float)
        self.n = self.A.shape[0]
        self.m = self.B.shape[1]
        self.q = self.M.shape[0]
        self.a = np.zeros(self.n) if a is None else np.asarray(a, dtype=float).reshape(self.n)
        self.c = np.zeros(self.C.shape[0]) if c is None else np.asarray(c, dtype=float).reshape(self.C.shape[0])

        try:
            res = place_poles(self.A, self.B, target_poles)
            self.K = np.asarray(res.gain_matrix, dtype=float)
            self.method = "place_poles"
        except Exception as exc:
            rho = float(np.max(np.abs(target_poles)))
            Qx = np.eye(self.n)
            Ru = (1.0 / max(rho, 1e-3)) * np.eye(self.m)
            P = solve_discrete_are(self.A, self.B, Qx, Ru)
            self.K = np.linalg.solve(Ru + self.B.T @ P @ self.B, self.B.T @ P @ self.A)
            self.method = f"lqr-fallback ({type(exc).__name__})"

        ref0 = np.zeros(self.q) if ref is None else np.asarray(ref, dtype=float)
        self._ref_cache = None
        self.x_ss = np.zeros(self.n)
        self.u_ff = np.zeros(self.m)
        self.set_reference(ref0)

    def set_reference(self, ref) -> None:
        ref = np.asarray(ref, dtype=float)
        if (self._ref_cache is not None
                and ref.shape == self._ref_cache.shape
                and np.allclose(ref, self._ref_cache)):
            return
        self.x_ss, self.u_ff = _solve_feedforward(
            self.A, self.B, self.C, self.M, self.a, self.c, ref
        )
        self._ref_cache = ref.copy()

    def reset(self) -> None:
        pass

    def compute(self, x_hat, ref=None) -> np.ndarray:
        if ref is not None:
            self.set_reference(ref)
        x_hat = np.asarray(x_hat, dtype=float)
        u = self.u_ff - self.K @ (x_hat - self.x_ss)
        return np.clip(u, 0.0, 1.0)


class MPC:
    """Condensed box-QP MPC with affine dynamics.

    Cost  Σ_{k=0..H-1} (z_k - r)ᵀ Q_z (z_k - r) + (u_k - u_ff)ᵀ Ru (u_k - u_ff)
    plus terminal LQR-P penalty on (x_H - x_ss). Dynamics
        x_{k+1} = A x_k + B u_k + a
        z_k     = M (C x_k + c)
    Subject to 0 ≤ u_k ≤ 1 elementwise.

    Solved by L-BFGS-B with analytic gradient + warm-start receding shift.
    """

    uses_raw_y = False

    def __init__(self, A, B, C, M, a=None, c=None, horizon: int = 10,
                 rho: float = 0.1, Q_z=None, ref=None):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.M = np.asarray(M, dtype=float)
        self.H = int(horizon)
        self.rho = float(rho)
        self.n = self.A.shape[0]
        self.m = self.B.shape[1]
        self.q = self.M.shape[0]
        self.Q_z = np.eye(self.q) if Q_z is None else np.asarray(Q_z, dtype=float)
        self.a = np.zeros(self.n) if a is None else np.asarray(a, dtype=float).reshape(self.n)
        self.c = np.zeros(self.C.shape[0]) if c is None else np.asarray(c, dtype=float).reshape(self.C.shape[0])

        MC = self.M @ self.C
        Qx = MC.T @ self.Q_z @ MC + 1e-8 * np.eye(self.n)
        Ru = self.rho * np.eye(self.m)
        P_term = solve_discrete_are(self.A, self.B, Qx, Ru)
        self.P_term = P_term
        self.Qx = Qx
        self.Ru = Ru
        self.MC = MC

        # Stack: X = Phi x0 + Gamma U + Psi  (Psi from constant offset a)
        Phi = np.empty((self.H * self.n, self.n))
        Gamma = np.zeros((self.H * self.n, self.H * self.m))
        Psi = np.empty((self.H * self.n,))
        # x_{k+1} = A x_k + B u_k + a; so x_k = A^k x_0 + sum_{j<k} A^{k-1-j} B u_j + sum_{j<k} A^{k-1-j} a
        # We stack states x_1..x_H.
        A_pow = [np.eye(self.n)]
        for _ in range(self.H):
            A_pow.append(A_pow[-1] @ self.A)
        # Pre-compute Σ_{j=0}^{k-1} A^j (= partial sums for the constant-offset block).
        S = [np.zeros((self.n, self.n))]
        for k in range(self.H):
            S.append(S[-1] + A_pow[k])
        for k in range(self.H):
            Phi[k * self.n:(k + 1) * self.n, :] = A_pow[k + 1]
            for j in range(k + 1):
                Gamma[k * self.n:(k + 1) * self.n, j * self.m:(j + 1) * self.m] = A_pow[k - j] @ self.B
            Psi[k * self.n:(k + 1) * self.n] = S[k + 1] @ self.a

        Q_bar = np.zeros((self.H * self.n, self.H * self.n))
        for k in range(self.H - 1):
            Q_bar[k * self.n:(k + 1) * self.n, k * self.n:(k + 1) * self.n] = Qx
        Q_bar[(self.H - 1) * self.n:, (self.H - 1) * self.n:] = Qx + P_term

        self.Phi = Phi
        self.Gamma = Gamma
        self.Psi = Psi
        self.Q_bar = Q_bar
        self._GtQ = Gamma.T @ Q_bar
        R_bar = np.kron(np.eye(self.H), Ru)
        H_qp = 2.0 * (self._GtQ @ Gamma + R_bar)
        self.H_qp = 0.5 * (H_qp + H_qp.T)

        ref0 = np.zeros(self.q) if ref is None else np.asarray(ref, dtype=float)
        self._ref_cache = None
        self.X_ref = np.zeros(self.H * self.n)
        self.set_reference(ref0)

        self._U_warm = np.zeros(self.H * self.m)
        self._bounds = [(0.0, 1.0)] * (self.H * self.m)

    def set_reference(self, ref) -> None:
        ref = np.asarray(ref, dtype=float)
        if (self._ref_cache is not None
                and ref.shape == self._ref_cache.shape
                and np.allclose(ref, self._ref_cache)):
            return
        # Pick x_ss from the spec KKT block — corresponds to the constant input
        # the affine MPC will try to settle at; X_ref is x_ss replicated.
        x_ss, _ = _solve_feedforward(self.A, self.B, self.C, self.M,
                                     self.a, self.c, ref)
        self.X_ref = np.tile(x_ss, self.H)
        self._ref_cache = ref.copy()

    def reset(self) -> None:
        self._U_warm = np.zeros(self.H * self.m)

    def compute(self, x_hat, ref=None) -> np.ndarray:
        if ref is not None:
            self.set_reference(ref)
        x_hat = np.asarray(x_hat, dtype=float)
        # f = ∂/∂U [(Phi x + Gamma U + Psi - X_ref)ᵀ Q_bar (...)] = 2 Gᵀ Q_bar (Phi x + Psi - X_ref)
        f = 2.0 * self._GtQ @ (self.Phi @ x_hat + self.Psi - self.X_ref)

        H_qp = self.H_qp
        def obj(U):
            return 0.5 * U @ (H_qp @ U) + f @ U
        def grad(U):
            return H_qp @ U + f

        U0 = np.clip(self._U_warm, 0.0, 1.0)
        result = minimize(
            obj, U0, jac=grad, method="L-BFGS-B", bounds=self._bounds,
            options={"maxiter": 100, "gtol": 1e-6, "ftol": 1e-9},
        )
        U_opt = result.x
        u0 = U_opt[:self.m]
        # Receding shift for warm start.
        self._U_warm = np.concatenate([U_opt[self.m:], np.zeros(self.m)])
        return np.clip(u0, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Integral variants (v3 — spec §4). Added alongside the controllers above;
# the existing five are untouched. Each integral variant has anti-windup so
# the integral state stays bounded when the target is infeasible.
# -----------------------------------------------------------------------------


class PI(ProportionalFeedback):
    """Model-free proportional + integral feedback on the readout error.

        u = clip(u0 + Kp (ref − M y) + Ki q, 0, 1)
        q_{t+1} = q_t + (ref − M y_t)         [frozen when u saturates]

    Anti-windup: when any component of the unclipped law lands outside
    [0, 1], freeze the integral update for that step (spec §4 — keeps q
    bounded under an infeasible target).
    """

    uses_raw_y = True

    def __init__(self, Kp, Ki, M, ref=None, u0=None, anti_windup: bool = True):
        super().__init__(Kp, M, ref=ref, u0=u0)
        self.Ki = np.asarray(Ki, dtype=float)
        self.anti_windup = bool(anti_windup)
        readout_dim = self.M.shape[0]
        self.q_state = np.zeros(readout_dim)

    def reset(self) -> None:
        self.q_state = np.zeros(self.M.shape[0])

    def compute(self, y, ref=None) -> np.ndarray:
        if ref is not None:
            self.ref = np.asarray(ref, dtype=float)
        z = self.M @ np.asarray(y, dtype=float)
        err = self.ref - z
        u_unclipped = self.u0 + self.Kp @ err + self.Ki @ self.q_state
        u = np.clip(u_unclipped, 0.0, 1.0)
        saturated = bool(np.any(u != u_unclipped)) if self.anti_windup else False
        if not saturated:
            self.q_state = self.q_state + err
        return u


class LQGI:
    """LQG + integral state on the **measured** readout error (spec §4).

    Augmented state: x_aug = [x; q] with q_{t+1} = q_t + (ref − M y_t).
    The LQR gain ``K_aug = [K_x, K_q]`` is designed on the augmented system
    (A_aug, B_aug). The control law (deviation form) is

        u = clip(u_ff − K_x (x_hat − x_ss) − K_q q, 0, 1)

    where (x_ss, u_ff) come from the spec KKT block (same as LQG). With
    K_i ≡ −K_q this matches the spec's "u_ff − K x_dev + K_i q" form.

    The integral is driven by the **measured** readout ``M y``, not the
    model-frame proxy ``M (C x_hat + c)`` — otherwise, under model mismatch
    the observer's biased ``x_hat`` masks the very offset the integral is
    meant to close. The driver's ``observe(y, x_hat)`` hook supplies y.

    Anti-windup: when the previous step's ``u`` saturated, the q-update for
    this step is frozen.
    """

    uses_raw_y = False

    def __init__(self, A, B, C, M, a=None, c=None,
                 rho: float = 1.0, rho_q: float = 1.0,
                 ref=None, anti_windup: bool = True):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.M = np.asarray(M, dtype=float)
        self.rho = float(rho)
        self.rho_q = float(rho_q)
        self.anti_windup = bool(anti_windup)
        self.n = self.A.shape[0]
        self.m = self.B.shape[1]
        self.q_dim = self.M.shape[0]
        self.a = np.zeros(self.n) if a is None else np.asarray(a, dtype=float).reshape(self.n)
        self.c = (np.zeros(self.C.shape[0]) if c is None
                  else np.asarray(c, dtype=float).reshape(self.C.shape[0]))

        MC = self.M @ self.C
        # Augmented dynamics (LQR-relevant homogeneous form):
        #   x_{t+1} = A x + B u
        #   q_{t+1} = -MC x + I q   (the constant ref-M c drops out for K design)
        A_aug = np.block([
            [self.A,                                  np.zeros((self.n, self.q_dim))],
            [-MC,                                     np.eye(self.q_dim)],
        ])
        B_aug = np.vstack([self.B, np.zeros((self.q_dim, self.m))])
        Q_aug = np.block([
            [MC.T @ MC + 1e-8 * np.eye(self.n),       np.zeros((self.n, self.q_dim))],
            [np.zeros((self.q_dim, self.n)),          self.rho_q * np.eye(self.q_dim)],
        ])
        Ru = self.rho * np.eye(self.m)
        P = solve_discrete_are(A_aug, B_aug, Q_aug, Ru)
        K_aug = np.linalg.solve(Ru + B_aug.T @ P @ B_aug, B_aug.T @ P @ A_aug)
        self.K_x = K_aug[:, :self.n]
        self.K_q = K_aug[:, self.n:]

        ref0 = np.zeros(self.q_dim) if ref is None else np.asarray(ref, dtype=float)
        self._ref_cache = None
        self.x_ss = np.zeros(self.n)
        self.u_ff = np.zeros(self.m)
        self.ref = ref0.copy()
        self.set_reference(ref0)
        self.q_state = np.zeros(self.q_dim)
        self._last_sat = False

    def set_reference(self, ref) -> None:
        ref = np.asarray(ref, dtype=float)
        if (self._ref_cache is not None
                and ref.shape == self._ref_cache.shape
                and np.allclose(ref, self._ref_cache)):
            return
        self.x_ss, self.u_ff = _solve_feedforward(
            self.A, self.B, self.C, self.M, self.a, self.c, ref
        )
        self._ref_cache = ref.copy()
        self.ref = ref.copy()

    def reset(self) -> None:
        self.q_state = np.zeros(self.q_dim)
        self._last_y = None

    def observe(self, y, x_hat=None, u_prev=None) -> None:
        """Driver hook: cache the latest measurement for use in compute().

        The integral state is actually updated in ``compute`` after the
        saturation decision (so anti-windup applies to the SAME step's
        control, not the previous step's — otherwise the very first step's
        large reference−measurement gap leaks into q before saturation
        gating can engage).
        """
        self._last_y = np.asarray(y, dtype=float)

    def compute(self, x_hat, ref=None) -> np.ndarray:
        if ref is not None:
            self.set_reference(ref)
        x_hat = np.asarray(x_hat, dtype=float)
        u_unclipped = (
            self.u_ff
            - self.K_x @ (x_hat - self.x_ss)
            - self.K_q @ self.q_state
        )
        u = np.clip(u_unclipped, 0.0, 1.0)
        # Integral update: q_{t+1} = q_t + (ref − M y_t), frozen if the
        # control just saturated (spec §4 anti-windup).
        saturated = bool(np.any(u != u_unclipped)) if self.anti_windup else False
        if (not saturated) and (self._last_y is not None):
            residual = self.ref - self.M @ self._last_y
            self.q_state = self.q_state + residual
        return u


class OffsetFreeMPC(MPC):
    """MPC with an online-estimated constant output disturbance (spec §4).

    Standard "offset-free MPC" pattern: maintain a disturbance estimate
    ``d_hat`` of shape (p,) (observation dim). The nominal MPC model says
    ``y_t = C x_t + c``; the augmented model says ``y_t = C x_t + c + d_hat``.
    Each step we shift the MPC's internal target so the *actual* readout
    hits the user's reference:

        effective_ref = user_ref − M d_hat

    ``d_hat`` is updated from the latest measurement residual via the
    ``observe(y, x_hat)`` hook (called by the closed-loop driver before
    ``compute``):

        innov = y − (C x_hat + c + d_hat)
        d_hat := d_hat + dist_gain · innov     [frozen when u saturated last step]

    This is the MPC analogue of integral action: under steady-state model
    error the disturbance estimate accumulates and the shifted target
    closes the offset. Anti-windup freezes the ``d_hat`` update whenever
    the previous step's input saturated.
    """

    uses_raw_y = False  # consume x_hat as the driver normally provides

    def __init__(self, A, B, C, M, a=None, c=None, horizon: int = 10,
                 rho: float = 0.1, Q_z=None, ref=None,
                 dist_gain: float = 0.1, anti_windup: bool = True):
        super().__init__(A, B, C, M, a=a, c=c, horizon=horizon,
                         rho=rho, Q_z=Q_z, ref=ref)
        self.dist_gain = float(dist_gain)
        self.anti_windup = bool(anti_windup)
        ref0 = np.zeros(self.q) if ref is None else np.asarray(ref, dtype=float)
        self._user_ref = ref0.copy()
        self.d_hat = np.zeros(self.C.shape[0])
        self._last_sat = False

    def reset(self) -> None:
        super().reset()
        self.d_hat = np.zeros(self.C.shape[0])
        self._last_y = None
        self._last_x_hat = None

    def observe(self, y, x_hat=None, u_prev=None) -> None:
        """Driver hook: cache (y, x_hat) for the d_hat update in compute().

        Like LQGI, the disturbance update happens after the control's
        saturation status is known — otherwise the initial transient leaks
        into d_hat before anti-windup can engage.
        """
        self._last_y = np.asarray(y, dtype=float)
        self._last_x_hat = (None if x_hat is None
                             else np.asarray(x_hat, dtype=float))

    def compute(self, x_hat, ref=None) -> np.ndarray:
        if ref is not None:
            self._user_ref = np.asarray(ref, dtype=float)
        # Shift the MPC's internal target by M d_hat so the actual readout
        # (nominal model + d_hat) hits the user's reference.
        eff_ref = self._user_ref - self.M @ self.d_hat
        super().set_reference(eff_ref)
        u = super().compute(x_hat, ref=None)
        # d_hat update with anti-windup on the SAME step's saturation.
        saturated = bool(np.any(u <= 1e-9) or np.any(u >= 1.0 - 1e-9))
        if (not saturated) and self._last_y is not None and self._last_x_hat is not None:
            innov = self._last_y - (self.C @ self._last_x_hat
                                       + self.c + self.d_hat)
            self.d_hat = self.d_hat + self.dist_gain * innov
        return u
