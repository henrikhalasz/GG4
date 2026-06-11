"""kinematics.py -- planar two-link arm kinematics (Week 4, Step 2A).

The arm is the genuine ``_Arm`` from ``BMI_and_Hand.py``: two rigid links of
length 30, no inertia, no joint limits. The shoulder angle ``theta_s`` is
absolute (measured from the +x axis); the elbow angle ``theta_e`` is relative
to the upper arm. We re-implement the geometry cleanly here (we import nothing
from Week 3) and verify it reproduces ``_Arm`` exactly.

Forward kinematics
    x = L1*cos(theta_s) + L2*cos(theta_s + theta_e)
    y = L1*sin(theta_s) + L2*sin(theta_s + theta_e)

Jacobian (d hand / d theta)
    J = [[-L1 s1 - L2 s12,  -L2 s12],
         [ L1 c1 + L2 c12,   L2 c12]]
    det J = L1*L2*sin(theta_e) = 900*sin(theta_e)

so the arm is singular at theta_e = 0 (elbow straight, the outer rim r = 60)
and theta_e = pi (elbow folded, the centre r = 0). The start pose theta = (0, 0)
puts the hand at (60, 0) -- on the outer singular locus.

All angles in radians. Hand positions are float ``np.ndarray`` of shape (2,).
"""

from __future__ import annotations

import numpy as np

# --- Arm constants (read from BMI_and_Hand.py, class _Arm) -------------------
L1 = 30.0  # upper arm (upperarm_length)
L2 = 30.0  # forearm   (forearm_length)
REACH = L1 + L2  # 60 -- radius of the reachable disk

# Per-joint velocity-integrator parameters (class _Arm.move). The arm has no
# inertia: each step it converts a net muscle drive into a joint speed via a
# Coulomb dead-zone (dry friction) and a viscous divisor (wet friction), then
# integrates angle += speed. Encoded here so both the actuator and the _Arm
# verification share one definition.
DEAD_ZONE = 0.05            # _shoulder/_elbow_friction_dry
VISCOUS_SHOULDER = 7.5      # _shoulder_friction_wet
VISCOUS_ELBOW = 5.0         # _elbow_friction_wet

# Saturation joint speed at full muscle drive (net = 1): (1 - dead_zone)/viscous.
SAT_SHOULDER = (1.0 - DEAD_ZONE) / VISCOUS_SHOULDER  # 0.126667 rad/step
SAT_ELBOW = (1.0 - DEAD_ZONE) / VISCOUS_ELBOW        # 0.190000 rad/step
SAT_SPEED = np.array([SAT_SHOULDER, SAT_ELBOW])


def joint_speed(net: np.ndarray | float, viscous: np.ndarray | float,
                dead_zone: float = DEAD_ZONE) -> np.ndarray:
    """The genuine ``_Arm`` net-drive -> joint-speed map (Coulomb + viscous).

    ``speed = (net - sign(net)*dead_zone)/viscous`` when ``|net| > dead_zone``,
    else ``0``. Vectorised over the two joints.
    """
    net = np.asarray(net, dtype=float)
    viscous = np.asarray(viscous, dtype=float)
    active = np.abs(net) > dead_zone
    speed = (net - np.sign(net) * dead_zone) / viscous
    return np.where(active, speed, 0.0)


# --- Forward kinematics ------------------------------------------------------
def forward(theta) -> np.ndarray:
    """Hand position for joint angles ``theta = (theta_s, theta_e)``."""
    ts, te = float(theta[0]), float(theta[1])
    x = L1 * np.cos(ts) + L2 * np.cos(ts + te)
    y = L1 * np.sin(ts) + L2 * np.sin(ts + te)
    return np.array([x, y])


def elbow_pos(theta) -> np.ndarray:
    """Elbow joint position (for drawing the arm)."""
    ts = float(theta[0])
    return np.array([L1 * np.cos(ts), L1 * np.sin(ts)])


def jacobian(theta) -> np.ndarray:
    """2x2 hand Jacobian ``d hand / d theta`` at ``theta``."""
    ts, te = float(theta[0]), float(theta[1])
    s1, c1 = np.sin(ts), np.cos(ts)
    s12, c12 = np.sin(ts + te), np.cos(ts + te)
    return np.array([
        [-L1 * s1 - L2 * s12, -L2 * s12],
        [L1 * c1 + L2 * c12,  L2 * c12],
    ])


def det_jacobian(theta) -> float:
    """Analytic determinant ``900*sin(theta_e)``."""
    return L1 * L2 * np.sin(float(theta[1]))


# --- Inverse kinematics (both analytic elbow branches) -----------------------
def inverse(hand):
    """Return the two IK solutions ``(theta_up, theta_down)`` for ``hand``.

    ``theta_up`` has ``theta_e >= 0`` (elbow-up branch); ``theta_down`` has
    ``theta_e <= 0``. Both reproduce ``hand`` under ``forward`` to machine
    precision. At the rim (r = 60) the branches coincide (theta_e = 0); at the
    centre (r = 0) the shoulder angle is undefined (we return theta_s = 0).
    """
    x, y = float(hand[0]), float(hand[1])
    r2 = x * x + y * y
    cos_e = (r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    cos_e = float(np.clip(cos_e, -1.0, 1.0))
    te = np.arccos(cos_e)  # in [0, pi]

    def branch(theta_e):
        k1 = L1 + L2 * np.cos(theta_e)
        k2 = L2 * np.sin(theta_e)
        theta_s = np.arctan2(y, x) - np.arctan2(k2, k1)
        # wrap shoulder to (-pi, pi]
        theta_s = (theta_s + np.pi) % (2.0 * np.pi) - np.pi
        return np.array([theta_s, theta_e])

    return branch(te), branch(-te)


# --- Damped-least-squares (resolved-rate) step -------------------------------
def resolved_rate_step(theta, hand_err, lam: float = 2.0) -> np.ndarray:
    """DLS joint step ``dtheta = J^T (J J^T + lam^2 I)^-1 v`` for a hand error.

    Damping ``lam`` is mandatory: at the singular home theta = (0, 0) the plain
    pseudo-inverse (``lam = 0``) is rank-deficient and raises ``LinAlgError``.
    With ``lam ~ 2`` the step stays finite and points down-error.
    """
    J = jacobian(theta)
    v = np.asarray(hand_err, dtype=float)
    JJt = J @ J.T
    A = JJt + (lam ** 2) * np.eye(2)
    return J.T @ np.linalg.solve(A, v)


# --- Singularity helpers (for the map) ---------------------------------------
def sigma_min(theta) -> float:
    """Smallest singular value of J (distance to singularity)."""
    return float(np.linalg.svd(jacobian(theta), compute_uv=False)[-1])


def cond(theta) -> float:
    """Condition number of J (1 = isotropic, inf = singular)."""
    s = np.linalg.svd(jacobian(theta), compute_uv=False)
    return float(s[0] / s[-1]) if s[-1] > 0 else np.inf


# --- Verification against the genuine _Arm -----------------------------------
def _verify_against_arm(n_poses: int = 2000, n_drive_steps: int = 50,
                        seed: int = 0):
    """Confirm our FK and our forward integration reproduce ``_Arm`` exactly.

    Needs torch (``BMI_and_Hand.py`` imports it). Returns ``(fk_err, dyn_err)``
    or raises ``ImportError`` if torch/_Arm is unavailable.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "week3"))
    from BMI_and_Hand import _Arm  # noqa: E402  (genuine arm)

    rng = np.random.default_rng(seed)

    # (1) FK match: random joint angles, compare _Arm.hand_pos to forward().
    fk_err = 0.0
    for _ in range(n_poses):
        ts = rng.uniform(-np.pi, np.pi)
        te = rng.uniform(-np.pi, np.pi)
        arm = _Arm()
        arm._shoulder_angle = ts
        arm._elbow_angle = te
        fk_err = max(fk_err, float(np.max(np.abs(arm.hand_pos - forward([ts, te])))))

    # (2) Forward-integration match: drive _Arm.move with a random activation
    #     sequence; replicate the net->speed->angle integration ourselves.
    arm = _Arm()
    theta = np.array([0.0, 0.0])
    dyn_err = 0.0
    for _ in range(n_drive_steps):
        a = rng.uniform(0.0, 1.0, size=4)  # (sh_pos, sh_neg, el_pos, el_neg)
        arm.move(*a)
        net = np.array([a[0] - a[1], a[2] - a[3]])
        theta = theta + joint_speed(net, np.array([VISCOUS_SHOULDER, VISCOUS_ELBOW]))
        dyn_err = max(dyn_err, float(np.max(np.abs(arm.hand_pos - forward(theta)))))

    return fk_err, dyn_err


# --- Standalone checks -------------------------------------------------------
def _main():
    print("=== kinematics.py self-check ===")
    print(f"L1 = {L1}, L2 = {L2}, reach = {REACH}")
    print(f"saturation speed: shoulder = {SAT_SHOULDER:.6f}, elbow = {SAT_ELBOW:.6f} rad/step")

    # det J at key poses.
    for name, th in [("home (0,0)", [0.0, 0.0]),
                     ("mid (0, pi/2)", [0.0, np.pi / 2]),
                     ("centre (0, pi)", [0.0, np.pi])]:
        dJ_analytic = det_jacobian(th)
        dJ_numeric = float(np.linalg.det(jacobian(th)))
        hand = forward(th)
        r = float(np.hypot(*hand))
        print(f"  {name:16s}: hand={hand.round(4)}, r={r:6.3f}, "
              f"detJ={dJ_analytic:+9.4f} (numeric {dJ_numeric:+9.4f})")
    assert abs(det_jacobian([0.0, 0.0])) < 1e-12, "home must be singular"
    assert np.allclose(forward([0.0, 0.0]), [60.0, 0.0]), "home hand must be (60,0)"

    # IK round-trip over a reachable grid (avoid the exact rim/centre).
    rng = np.random.default_rng(0)
    max_rt = 0.0
    for _ in range(20000):
        r = rng.uniform(1.0, 59.0)
        a = rng.uniform(-np.pi, np.pi)
        hand = np.array([r * np.cos(a), r * np.sin(a)])
        up, dn = inverse(hand)
        max_rt = max(max_rt,
                     float(np.max(np.abs(forward(up) - hand))),
                     float(np.max(np.abs(forward(dn) - hand))))
    print(f"IK round-trip max |forward(inverse(.)) - hand| = {max_rt:.2e} (both branches)")
    assert max_rt < 1e-9, "round-trip must match to ~1e-9"

    # DLS well-defined at the singular home; lam=0 must blow up there.
    dtheta = resolved_rate_step([0.0, 0.0], [-1.0, 0.0], lam=2.0)
    print(f"DLS step at home for hand_err (-1,0), lam=2: dtheta = {dtheta.round(6)}")
    assert np.all(np.isfinite(dtheta)), "DLS at home must be finite"
    try:
        np.linalg.solve(jacobian([0.0, 0.0]), np.array([-1.0, 0.0]))
        raised = False
    except np.linalg.LinAlgError:
        raised = True
    print(f"lam=0 plain inverse at home raises LinAlgError: {raised}")
    assert raised, "home Jacobian must be singular for lam=0"

    # Verify against the genuine _Arm (needs torch).
    try:
        fk_err, dyn_err = _verify_against_arm()
        print(f"vs genuine _Arm:  FK max err = {fk_err:.2e},  "
              f"forward-integration max err = {dyn_err:.2e}")
        assert fk_err < 1e-9 and dyn_err < 1e-9, "must match _Arm to ~0"
        print("  -> our kinematics reproduce _Arm exactly.")
    except ImportError as e:
        print(f"vs genuine _Arm:  SKIPPED ({e}) -- install torch to run this check.")

    print("All assertions passed.")


if __name__ == "__main__":
    _main()
