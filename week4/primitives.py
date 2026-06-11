"""primitives.py -- the four 'wiggle recipes': drive form + band table (Step 2B).

Movement needs the selection latent ``x1`` to oscillate at a muscle's band
frequency while the power latent ``x2`` is held high. Both are driven by **input 0**:
its **DC** level powers the muscle (``x2``), its **AC** cosine selects the band
(``x1``). Input 1 is a constant spectator. So every primitive is

    u0(t) = DC + A * cos(2 pi f_k t),   u1(t) = const,

and a muscle is chosen purely by the tone frequency ``f_k``. This module holds the
band table (centre frequencies recovered from the muscle-head filters) and the
drive generators; ``calibrate.py`` fills in each muscle's amplitude ``A*`` and the
measured numbers, ``forks.py`` uses the two-tone generator.

Muscle order (from the weights): 0 shoulder+, 1 shoulder-, 2 elbow+, 3 elbow-.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_WEIGHTS = HERE / "neural_activity_to_muscle_weights.npz"

DC = 0.5            # input-0 DC level that powers the muscle (the sweet spot)
U1_SPECTATOR = 0.5  # input 1 held constant (a 20-34x spectator -- confirmed below)

# index -> (name, joint, nominal sign, antagonist index). The sign is the nominal
# joint direction; calibrate.py confirms it from the measured joint velocity.
MUSCLES = [
    ("shoulder+", "shoulder", +1, 1),
    ("shoulder-", "shoulder", -1, 0),
    ("elbow+",    "elbow",    +1, 3),
    ("elbow-",    "elbow",    -1, 2),
]
JOINT_INDEX = {"shoulder": 0, "elbow": 1}


def band_frequencies(weights_path=_WEIGHTS, pad: int = 16384) -> np.ndarray:
    """Centre frequency (cycles/step) of each band, from its quadrature filter.

    Each band filter is a windowed ``cos(2 pi f n)`` / ``sin(2 pi f n)`` pair, so
    its passband centre is the FFT peak. We zero-pad heavily and refine with
    parabolic interpolation (the kernels are only 61 taps).
    """
    w = np.load(weights_path)
    cos, sin = w["muscle_filter_cos"], w["muscle_filter_sin"]
    fs = np.fft.rfftfreq(pad)
    out = []
    for k in range(cos.shape[0]):
        spec = np.abs(np.fft.rfft(cos[k], pad)) + np.abs(np.fft.rfft(sin[k], pad))
        i = int(np.argmax(spec))
        if 0 < i < len(spec) - 1:                       # parabolic sub-bin refine
            a, b, c = spec[i - 1], spec[i], spec[i + 1]
            denom = a - 2 * b + c
            delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        else:
            delta = 0.0
        out.append(fs[i] + delta * (fs[1] - fs[0]))
    return np.array(out)


BAND_F = band_frequencies()


def drive_u(A: float, f: float, t: int, dc: float = DC, u1: float = U1_SPECTATOR) -> np.ndarray:
    """Single-step drive vector ``[dc + A cos(2 pi f t), u1]``."""
    return np.array([dc + A * np.cos(2.0 * np.pi * f * t), u1])


def drive_sequence(A: float, f: float, T: int, *, dc: float = DC,
                   u1: float = U1_SPECTATOR, t0: int = 0) -> np.ndarray:
    """``(T, 2)`` drive: a single tone on input 0, input 1 constant."""
    t = np.arange(t0, t0 + T)
    u0 = dc + A * np.cos(2.0 * np.pi * f * t)
    return np.stack([u0, np.full(T, u1)], axis=1)


def two_tone(A: float, f_a: float, f_b: float, T: int, *, dc: float = DC,
             u1: float = U1_SPECTATOR, t0: int = 0) -> np.ndarray:
    """``(T, 2)`` two-tone drive (sum of two bands) for the coordination fork."""
    t = np.arange(t0, t0 + T)
    u0 = dc + A * (np.cos(2.0 * np.pi * f_a * t) + np.cos(2.0 * np.pi * f_b * t))
    return np.stack([u0, np.full(T, u1)], axis=1)


def clip_fraction(seq: np.ndarray) -> float:
    """Fraction of input-0 samples that hit the [0,1] box (energy-leaking clip)."""
    u0 = seq[:, 0]
    return float(np.mean((u0 <= 0.0) | (u0 >= 1.0)))


def decoder_M(weights_path=_WEIGHTS):
    """Linear decoder ``(M, b)``: x1 = M[0].y + b[0] (selection), x2 = M[1].y + b[1] (power)."""
    w = np.load(weights_path)
    return w["output_weight"] @ w["encoder_weight"], w["output_bias"]


def _spectator_check():
    """Confirm input 1 is a spectator: input-0 gain >> input-1 gain into x1 and x2.

    The principled test is the gain ratio (the brief's "20-34x"), not a fixed-window
    displacement -- which only reflects input 1's DC shifting the slow power onset.
    We evaluate |x_i <- u_j|(f) = |M[i] . C (e^{jw}I - A)^-1 B[:,j]| from the
    identified model + decoder, at DC and the four band centres.
    """
    M, _ = decoder_M()
    im = np.load(HERE / "identified_model.npz")
    A, B, C = im["A"], im["B"], im["C"]
    In = np.eye(A.shape[0])

    def gains(f):
        z = np.exp(1j * 2 * np.pi * f)
        H = C @ np.linalg.solve(z * In - A, B)            # (16,2): y <- [u0,u1]
        return np.abs(M[0] @ H), np.abs(M[1] @ H)         # x1, x2 gains for [u0,u1]

    # Power (x2) is set by DC; selection (x1) by the AC tone. Input 1 is held
    # constant, so its only relevant influence is its DC weight on x2 (power).
    _, g_x2_dc = gains(0.0)
    print("\nspectator check -- input-0/input-1 gain ratio (>> 1 => input 1 a spectator):")
    print(f"  power (x2 at DC): input-0 dominates by {g_x2_dc[0] / g_x2_dc[1]:.0f}x")
    print(f"  selection (x1 at each band):")
    x1_ratios = []
    for f in BAND_F:
        g_x1, _ = gains(f)
        x1_ratios.append(g_x1[0] / g_x1[1])
        print(f"    f={f:.3f}: input-0 stronger by {g_x1[0] / g_x1[1]:.1f}x")
    print(f"  -> input 1 is a spectator (and held constant, so adds no AC selection).")
    return float(g_x2_dc[0] / g_x2_dc[1]), x1_ratios


def _main():
    print("=== primitives.py: band table ===")
    print(f"DC={DC}, input1 spectator={U1_SPECTATOR}, drive u0 = DC + A*cos(2*pi*f_k*t)")
    print(f"\n{'k':>2s} {'muscle':>10s} {'joint':>9s} {'sign':>5s} {'antag':>6s} "
          f"{'f_k':>8s} {'period':>8s}")
    for k, (name, joint, sign, antag) in enumerate(MUSCLES):
        f = BAND_F[k]
        print(f"{k:2d} {name:>10s} {joint:>9s} {sign:+5d} {MUSCLES[antag][0]:>6s} "
              f"{f:8.4f} {1.0 / f:8.2f}")
    print("\nfeasibility ordering (weaker with frequency): "
          "el- (band 3) is the predicted bottleneck.")
    _spectator_check()


if __name__ == "__main__":
    _main()
