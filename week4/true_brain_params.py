"""
true_brain_params.py  --  VALIDATION YARDSTICK ONLY.

The compiled-in ground-truth parameters of the Week-4 Brain (a linear-Gaussian
state-space model). These are used ONLY to validate the identified model offline
(eigenvalues, transfer function, noise scale) and to confirm that the estimator is
not the limiting factor for the controller.

NEVER import this module from the deployed estimator or controller. The deployed
pipeline must use the *identified* model only. Identification is always a similarity
transform away from these matrices, so do NOT compare A/B/H entrywise -- compare
gauge-invariant quantities (eigenvalues, impulse response / transfer function, DC
gain, prediction error).

State x in R^6, input u in R^2 (clamped to [0,1] before B), output y in R^16.
    x <- A x + B clamp(u,0,1) + w,   w ~ N(transition_noise_mean, Q)
    y  = H x + v,                    v ~ N(observation_noise_mean, R)
"""

import numpy as np

state_dim = 6
obs_dim = 16
input_dim = 2

A = np.array([
    [ 0.9360, -0.0210,  0.0150,  0.0190, -0.0390, -0.0260],
    [ 0.0030,  0.9290, -0.0000,  0.2330,  0.0180,  0.0160],
    [ 0.0010,  0.0230,  0.9720, -0.1700, -0.9930, -0.0190],
    [ 0.0180, -0.0010,  0.1490,  0.9490,  0.0240, -0.0030],
    [-0.0090, -0.0070,  0.0110,  0.0070,  0.0080,  0.0090],
    [ 0.0000, -0.0080, -0.0100, -0.0160,  0.0120,  0.8230],
])

B = np.array([
    [ 0.8040,  0.0000],
    [ 0.0010,  0.0030],
    [ 0.0020,  1.0020],
    [-0.0010, -0.0030],
    [ 0.0010,  0.9980],
    [ 0.0020,  0.2010],
])

H = np.array([
    [0.8520, 0.5230, 0.0020, 0.0030, 0.0060, 0.0030],
    [0.5960, 0.8030, 0.0020, 0.0030, 0.0020, 0.0000],
    [0.9990, 0.0470, 0.0030, 0.0000, 0.0020, 0.0020],
    [0.3000, 0.9540, 0.0060, 0.0030, 0.0070, 0.0020],
    [0.9820, 0.1860, 0.0070, 0.0010, 0.0000, 0.0150],
    [0.7290, 0.6850, 0.0060, 0.0090, 0.0060, 0.0080],
    [0.9550, 0.2970, 0.0020, 0.0020, 0.0020, 0.0040],
    [0.3350, 0.9420, 0.0050, 0.0060, 0.0010, 0.0020],
    [0.6440, 0.7650, 0.0050, 0.0060, 0.0060, 0.0030],
    [0.9810, 0.1920, 0.0040, 0.0060, 0.0010, 0.0000],
    [0.6850, 0.7290, 0.0050, 0.0010, 0.0020, 0.0090],
    [0.9860, 0.1650, 0.0020, 0.0040, 0.0030, 0.0000],
    [0.9930, 0.1150, 0.0020, 0.0000, 0.0010, 0.0010],
    [0.8540, 0.5190, 0.0140, 0.0000, 0.0020, 0.0060],
    [0.9600, 0.2770, 0.0200, 0.0110, 0.0290, 0.0040],
    [0.5660, 0.8250, 0.0030, 0.0020, 0.0010, 0.0010],
])

Q = np.array([
    [0.0405, 0.0006, 0.0004, 0.0005, 0.0008, 0.0005],
    [0.0006, 0.0407, 0.0006, 0.0006, 0.0001, 0.0007],
    [0.0004, 0.0006, 0.0060, 0.0009, 0.0009, 0.0006],
    [0.0005, 0.0006, 0.0009, 0.0060, 0.0008, 0.0013],
    [0.0008, 0.0001, 0.0009, 0.0008, 0.0014, 0.0008],
    [0.0005, 0.0007, 0.0006, 0.0013, 0.0008, 0.0013],
])

transition_noise_mean = np.zeros(6)

observation_noise_mean = np.array([
    0.0240, 0.0680, 0.0590, 0.0910, 0.1990, 0.0970, 0.0020, 0.0210,
    0.0780, 0.1230, 0.0940, 0.0120, 0.0560, 0.0360, 0.0800, 0.0030,
])

R = np.array([
    [0.850, 0.061, 0.035, 0.114, 0.138, 0.087, 0.009, 0.068, 0.020, 0.073, 0.096, 0.158, 0.048, 0.083, 0.080, 0.154],
    [0.061, 0.919, 0.015, 0.143, 0.111, 0.076, 0.044, 0.064, 0.046, 0.113, 0.068, 0.056, 0.139, 0.133, 0.122, 0.123],
    [0.035, 0.015, 0.833, 0.017, 0.099, 0.107, 0.072, 0.058, 0.066, 0.160, 0.072, 0.086, 0.077, 0.113, 0.160, 0.051],
    [0.114, 0.143, 0.017, 0.848, 0.070, 0.166, 0.103, 0.064, 0.061, 0.059, 0.158, 0.156, 0.078, 0.130, 0.023, 0.008],
    [0.138, 0.111, 0.099, 0.070, 0.862, 0.064, 0.047, 0.060, 0.100, 0.020, 0.166, 0.033, 0.069, 0.111, 0.057, 0.047],
    [0.087, 0.076, 0.107, 0.166, 0.064, 0.814, 0.050, 0.071, 0.063, 0.042, 0.101, 0.152, 0.060, 0.170, 0.087, 0.074],
    [0.009, 0.044, 0.072, 0.103, 0.047, 0.050, 0.844, 0.071, 0.036, 0.122, 0.102, 0.014, 0.006, 0.088, 0.043, 0.044],
    [0.068, 0.064, 0.058, 0.064, 0.060, 0.071, 0.071, 0.908, 0.063, 0.072, 0.111, 0.034, 0.027, 0.016, 0.069, 0.045],
    [0.020, 0.046, 0.066, 0.061, 0.100, 0.063, 0.036, 0.063, 0.969, 0.098, 0.100, 0.059, 0.104, 0.063, 0.157, 0.034],
    [0.073, 0.113, 0.160, 0.059, 0.020, 0.042, 0.122, 0.072, 0.098, 0.901, 0.045, 0.145, 0.101, 0.109, 0.148, 0.034],
    [0.096, 0.068, 0.072, 0.158, 0.166, 0.101, 0.102, 0.111, 0.100, 0.045, 0.879, 0.035, 0.111, 0.070, 0.052, 0.064],
    [0.158, 0.056, 0.086, 0.156, 0.033, 0.152, 0.014, 0.034, 0.059, 0.145, 0.035, 0.802, 0.113, 0.131, 0.060, 0.082],
    [0.048, 0.139, 0.077, 0.078, 0.069, 0.060, 0.006, 0.027, 0.104, 0.101, 0.111, 0.113, 0.816, 0.031, 0.158, 0.033],
    [0.083, 0.133, 0.113, 0.130, 0.111, 0.170, 0.088, 0.016, 0.063, 0.109, 0.070, 0.131, 0.031, 0.805, 0.079, 0.067],
    [0.080, 0.122, 0.160, 0.023, 0.057, 0.087, 0.043, 0.069, 0.157, 0.148, 0.052, 0.060, 0.158, 0.079, 0.983, 0.145],
    [0.154, 0.123, 0.051, 0.008, 0.047, 0.074, 0.044, 0.045, 0.034, 0.034, 0.064, 0.082, 0.033, 0.067, 0.145, 0.811],
])


def dc_gain():
    """Steady-state input->output gain  H (I - A)^{-1} B,  shape (16, 2)."""
    return H @ np.linalg.solve(np.eye(state_dim) - A, B)


def transfer_function(z):
    """G(z) = H (zI - A)^{-1} B for a complex scalar z, shape (16, 2)."""
    return H @ np.linalg.solve(z * np.eye(state_dim) - A, B)


def impulse_markov(n):
    """First n Markov parameters G_k = H A^{k} B (k=0..n-1), shape (n, 16, 2). G_0 = D = 0 here; first true term is H B at k corresponding to one-step."""
    out = []
    Ak = np.eye(state_dim)
    for _ in range(n):
        out.append(H @ Ak @ B)
        Ak = A @ Ak
    return np.array(out)


if __name__ == "__main__":
    # Self-checks against the stated structural facts.
    ev = np.linalg.eigvals(A)
    mags = np.sort(np.abs(ev))
    print("eigenvalue magnitudes :", np.round(mags, 3))
    print("  expected (approx)   : [0.017 0.824 0.943 0.951 0.951 0.961]")
    print("spectral radius       :", round(mags.max(), 3), "(expected 0.961)")

    # oscillatory mode period
    osc = ev[np.argmax(np.abs(ev.imag))]
    if abs(osc.imag) > 1e-9:
        period = 2 * np.pi / abs(np.angle(osc))
        print("oscillatory pole period:", round(period, 1), "steps")

    s = np.linalg.svd(dc_gain(), compute_uv=False)
    print("DC-gain singular values:", np.round(s, 3), " ratio:", round(s[0] / s[1], 1))

    for name, M in [("Q", Q), ("R", R)]:
        sym = np.allclose(M, M.T)
        spd = np.all(np.linalg.eigvalsh(M) > 0)
        print(f"{name}: symmetric={sym}  positive-definite={spd}")
    print("R diagonal range      :", round(np.diag(R).min(), 3), "to", round(np.diag(R).max(), 3))
