"""excitation.py -- multi-probe excitation library for offline brain ID (Week 4, Step 1).

Week 3 identified the brain online with a single Uniform[0,1] probe. Offline we can
*design* the excitation, so we build a small library and concatenate a schedule of
segments rich enough to load every mode -- in particular the slow real poles and the
weakly-observed states that input 1 drives only under AC excitation.

Every probe stays inside the input box [0, 1] (the Brain clamps there; outside it the
map is nonlinear and identification leaves the linear regime). We verify the clipping
fraction is ~0.

Library (each justified in week4_step1_estimator.md S4):
  - prbs            maximum-length pseudo-random binary; broadband, decorrelated channels
  - log_chirp       logarithmic frequency sweep; dwells at low frequency (slow poles, resonance)
  - multisine       sum of sinusoids with Schroeder phases; precise spectral content, low crest
  - lowpass_noise   low-pass-filtered white noise; extra low-frequency energy
  - uniform         Week-3 baseline (kept only for the ablation)

build_schedule() lays out the training schedule (one continuous trajectory):
  rest | prbs-0 | prbs-1 | chirp-0 | chirp-1 | msine | lpn
with the per-channel segments holding the other channel at 0 to separate the B columns,
and msine / lpn both-channel decorrelated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt

# Box and per-segment default budget (the canonical constants live in identify.py and
# are passed in; these defaults let this module self-test standalone).
BOX_LO, BOX_HI = 0.0, 1.0
_LO, _HI = 0.1, 0.9            # binary / swept levels stay clear of the box edges
_CENTRE = 0.5 * (_LO + _HI)    # 0.5
_AMP = 0.5 * (_HI - _LO)       # 0.4

# Brain structure we are deliberately exciting (from prior weeks; used only to *place*
# tones, never to fit). Resonance period ~36.5 -> f ~ 0.0274 cyc/step; slowest tau ~25.
_F_RESONANCE = 1.0 / 36.5


# ---------------------------------------------------------------------------
# Maximum-length binary sequence (Fibonacci LFSR)
# ---------------------------------------------------------------------------
# Maximal tap sets (1-indexed stages, standard tables) keyed by register length.
_MLS_TAPS = {
    7: [7, 6], 8: [8, 6, 5, 4], 9: [9, 5], 10: [10, 7], 11: [11, 9],
    12: [12, 11, 10, 4], 13: [13, 12, 11, 8], 14: [14, 13, 12, 2],
    15: [15, 14], 16: [16, 15, 13, 4],
}


def _mls_bits(n_bits: int, seed: int) -> np.ndarray:
    """``n_bits`` of a maximum-length sequence; register sized so one period covers it."""
    r = 7
    while (2 ** r - 1) < n_bits and r < 16:
        r += 1
    taps = _MLS_TAPS[r]
    rng = np.random.default_rng(seed)
    state = rng.integers(0, 2, size=r).tolist()
    if not any(state):
        state[0] = 1  # all-zero is a fixed point; avoid it
    out = np.empty(n_bits, dtype=int)
    for i in range(n_bits):
        out[i] = state[-1]
        fb = 0
        for t in taps:
            fb ^= state[t - 1]
        state = [fb] + state[:-1]
    return out


def prbs(T: int, period: int = 8, seed: int = 0, lo: float = _LO, hi: float = _HI) -> np.ndarray:
    """Maximum-length PRBS held for ``period`` steps per bit, toggling between ``lo`` / ``hi``.

    A clock period > 1 loads low frequencies (the slow poles), which a per-step binary
    flip would not. Returns shape ``(T,)``.
    """
    n_bits = int(np.ceil(T / period))
    bits = _mls_bits(n_bits, seed)
    sig = np.repeat(bits, period)[:T]
    return np.where(sig > 0, hi, lo).astype(float)


def log_chirp(
    T: int, p_start: float = 1000.0, p_end: float = 2.5,
    lo: float = _LO, hi: float = _HI,
) -> Tuple[np.ndarray, np.ndarray]:
    """Logarithmic chirp sweeping instantaneous *period* ``p_start`` -> ``p_end``.

    Frequency f(t) = f0 (f1/f0)^(t/T), f0 = 1/p_start, f1 = 1/p_end (cyc/step). The log
    schedule dwells at low frequency where the period-36.5 resonance and the slow poles
    live. Output = centre + amp*sin(phase), mapped into ``[lo, hi]``.

    Returns ``(u, f_inst)`` both shape ``(T,)``; ``f_inst`` is the instantaneous frequency.
    """
    t = np.arange(T, dtype=float)
    f0, f1 = 1.0 / p_start, 1.0 / p_end
    k = (f1 / f0) ** (t / max(T, 1))
    f_inst = f0 * k
    # phase(t) = 2*pi * integral_0^t f(tau) dtau  for the exponential sweep
    ratio = f1 / f0
    phase = 2.0 * np.pi * f0 * T / np.log(ratio) * (k - 1.0)
    centre = 0.5 * (lo + hi)
    amp = 0.5 * (hi - lo)
    return centre + amp * np.sin(phase), f_inst


def _schroeder_phases(n: int) -> np.ndarray:
    """Schroeder phases for ``n`` equal-amplitude tones -> low crest factor."""
    k = np.arange(1, n + 1)
    return -np.pi * k * (k - 1) / n


def _multisine_band(T: int) -> np.ndarray:
    """Tone grid across the band, emphasising the slow region and the resonance."""
    # Slow / low-frequency cluster (loads the slow poles), tones around the resonance,
    # and a spread up towards Nyquist so the fast dynamics are excited too.
    slow = np.array([1, 2, 3, 5, 8]) / T            # a few cycles over the segment
    res = _F_RESONANCE * np.array([0.5, 0.8, 1.0, 1.25, 1.6])
    spread = np.array([0.06, 0.10, 0.16, 0.24, 0.34])
    freqs = np.unique(np.concatenate([slow, res, spread]))
    return freqs[freqs < 0.45]                       # keep clear of Nyquist (0.5)


def multisine(T: int, seed: int = 0) -> np.ndarray:
    """Equal-amplitude multisine with Schroeder phases, scaled into ``[_LO, _HI]``.

    ``seed`` selects which interleaved half of the tone grid is used, so the two
    channels can be driven by disjoint (hence decorrelated) tone sets.
    """
    freqs = _multisine_band(T)
    # Disjoint tone subsets per channel: even / odd grid points (seed parity picks one).
    freqs = freqs[seed % 2::2]
    phases = _schroeder_phases(len(freqs))
    t = np.arange(T, dtype=float)[:, None]
    sig = (np.sin(2.0 * np.pi * freqs[None, :] * t + phases[None, :])).sum(axis=1)
    sig = sig / np.max(np.abs(sig))                  # normalise to unit peak
    return _CENTRE + _AMP * sig


def lowpass_noise(T: int, cutoff: float = 0.05, seed: int = 0, sigma_k: float = 3.5) -> np.ndarray:
    """Low-pass-filtered white noise mapped towards ``[_LO, _HI]`` (unclipped).

    ``cutoff`` is in cyc/step (Nyquist = 0.5); a low cutoff puts extra energy on the
    slow / weakly-observed modes that plain white noise under-excites. Scaled so that
    ``sigma_k`` standard deviations reach the box edge, so the rare excursions the box
    later clips are negligible. Returned *unclipped* so the schedule can measure the
    true clip fraction; the box clamp is applied once at schedule assembly.
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(T)
    b, a = butter(4, cutoff / 0.5)                   # 4th-order Butterworth, Wn vs Nyquist
    x = filtfilt(b, a, w)
    x = x / (sigma_k * np.std(x))
    return _CENTRE + _AMP * x


def uniform(T: int, m: int = 2, seed: int = 0) -> np.ndarray:
    """Week-3 Uniform[0,1] baseline probe (kept only for the ablation)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(T, m))


# ---------------------------------------------------------------------------
# Schedule assembly
# ---------------------------------------------------------------------------
def _clip_with_fraction(U_raw: np.ndarray) -> Tuple[np.ndarray, float]:
    """Clamp to the box and report the fraction of samples the box actually clipped."""
    outside = (U_raw < BOX_LO - 1e-12) | (U_raw > BOX_HI + 1e-12)
    return np.clip(U_raw, BOX_LO, BOX_HI), float(np.mean(outside))


@dataclass
class Schedule:
    U: np.ndarray                       # (T_total, 2) input time series (box-clamped)
    segments: List[Tuple[str, int, int]]  # (name, start, end) half-open
    chirp_freq: Dict[str, np.ndarray]   # name -> instantaneous frequency over that segment
    clip_frac: float = 0.0              # fraction of samples the box clipped (true, pre-clip)

    @property
    def T(self) -> int:
        return self.U.shape[0]

    def clip_fraction(self) -> float:
        return self.clip_frac


def build_schedule(
    seed: int = 0, t_rest: int = 3000, t_prbs: int = 4000, t_chirp: int = 4000,
    t_msine: int = 3000, t_lpn: int = 3000, prbs_period: int = 8,
) -> Schedule:
    """Training schedule as one continuous trajectory (state carries across boundaries).

    rest | prbs-0 | prbs-1 | chirp-0 | chirp-1 | msine | lpn.
    Per-channel segments hold the other channel at 0 to separate the B columns; msine /
    lpn are both-channel decorrelated.
    """
    rng = np.random.default_rng(seed)
    s_prbs0, s_prbs1 = rng.integers(0, 1 << 30, size=2)
    s_lpn0, s_lpn1 = rng.integers(0, 1 << 30, size=2)

    blocks: List[Tuple[str, np.ndarray, np.ndarray]] = []  # name, u0, u1
    chirp_freq: Dict[str, np.ndarray] = {}

    # rest: zero input (measures the sensor bias, S5)
    blocks.append(("rest", np.zeros(t_rest), np.zeros(t_rest)))

    # prbs per channel (other at 0)
    blocks.append(("prbs-0", prbs(t_prbs, prbs_period, int(s_prbs0)), np.zeros(t_prbs)))
    blocks.append(("prbs-1", np.zeros(t_prbs), prbs(t_prbs, prbs_period, int(s_prbs1))))

    # chirp per channel (other at 0) -- the dedicated input-1 sweep excites the hidden modes
    c0, f0 = log_chirp(t_chirp)
    c1, f1 = log_chirp(t_chirp)
    blocks.append(("chirp-0", c0, np.zeros(t_chirp)))
    blocks.append(("chirp-1", np.zeros(t_chirp), c1))

    # multisine: both channels, decorrelated (disjoint tone sets)
    blocks.append(("msine", multisine(t_msine, seed=0), multisine(t_msine, seed=1)))

    # low-pass noise: both channels, decorrelated (independent seeds)
    blocks.append(("lpn", lowpass_noise(t_lpn, seed=int(s_lpn0)),
                   lowpass_noise(t_lpn, seed=int(s_lpn1))))

    U_parts, segments, cursor = [], [], 0
    for name, u0, u1 in blocks:
        n = u0.shape[0]
        U_parts.append(np.column_stack([u0, u1]))
        segments.append((name, cursor, cursor + n))
        if name == "chirp-0":
            chirp_freq[name] = f0
        elif name == "chirp-1":
            chirp_freq[name] = f1
        cursor += n

    U, clip_frac = _clip_with_fraction(np.vstack(U_parts))
    return Schedule(U=U, segments=segments, chirp_freq=chirp_freq, clip_frac=clip_frac)


def validation_inputs(seed: int, T: int = 4000) -> np.ndarray:
    """Held-out both-channel decorrelated excitation (different seed, never fitted).

    We use independent low-pass-filtered noise per channel -- a different signal class
    from the training multisine, so the held-out error is a fair generalisation test.
    """
    rng = np.random.default_rng(seed)
    s0, s1 = rng.integers(0, 1 << 30, size=2)
    U_raw = np.column_stack([lowpass_noise(T, seed=int(s0)), lowpass_noise(T, seed=int(s1))])
    U, _ = _clip_with_fraction(U_raw)
    return U


def uniform_schedule(seed: int, T: int) -> np.ndarray:
    """Budget-matched Uniform[0,1] schedule for the ablation baseline."""
    return uniform(T, 2, seed)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _main() -> None:
    sch = build_schedule(seed=0)
    print(f"training schedule: T = {sch.T} steps, {len(sch.segments)} segments")
    for name, s, e in sch.segments:
        u = sch.U[s:e]
        print(f"  {name:8s}  [{s:6d}:{e:6d})  len {e - s:5d}  "
              f"u0 in [{u[:, 0].min():.3f},{u[:, 0].max():.3f}]  "
              f"u1 in [{u[:, 1].min():.3f},{u[:, 1].max():.3f}]")
    print(f"overall clipping fraction: {sch.clip_fraction():.2e}  (must be ~0)")
    assert sch.U.min() >= BOX_LO - 1e-9 and sch.U.max() <= BOX_HI + 1e-9, "input left the box"

    val = validation_inputs(seed=999)
    print(f"validation inputs: T = {val.shape[0]}, range "
          f"[{val.min():.3f}, {val.max():.3f}]")
    cf = sch.chirp_freq["chirp-0"]
    print(f"chirp instantaneous frequency: {cf[0]:.4f} -> {cf[-1]:.4f} cyc/step "
          f"(period {1 / cf[0]:.0f} -> {1 / cf[-1]:.1f})")


if __name__ == "__main__":
    _main()
