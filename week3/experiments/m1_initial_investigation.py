"""M1 — Initial investigation of the real Brain (spec §2).

Runs the analysis-package catalogue against the GG4 wheel and saves:

    - results/m1_initial_investigation.png       — single multi-panel figure
    - results/ground_truth/seed_{S}.npz          — yardstick model per seed
    - results/m1_initial_investigation.txt       — printable numbers

Usage::

    python -m week3.experiments.m1_initial_investigation
    python week3/experiments/m1_initial_investigation.py

The script never imports anything from ``control/`` or ``estimator/`` — it
lives entirely on the *analysis* side. The ground-truth model it saves is
read by M3 experiments to grade the honest pipeline; the deployed
controller never sees it (spec rule #2).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Force UTF-8 stdout on Windows so the plain-language status prints (with
# Greek letters, ≡, etc.) survive the cp1252 console default.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent  # week3/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "week 1"))

from analysis.brain_probe import (  # noqa: E402
    discover_contract,
    observation_noise_scale,
    autonomous_trajectory,
    hankel_singular_values,
    seed_invariance,
    control_authority,
)
from analysis.ground_truth import build_ground_truth_model  # noqa: E402


# ----------------------------------------------------------------------------
# Brain factory — only place that touches ``GG4``.
# ----------------------------------------------------------------------------
def brain_plant_factory(seed: int):
    """Return a fresh ``BrainPlant(Brain(seed))`` instance.

    Imports are lazy so this file can be imported without ``GG4`` installed
    (e.g. for docs or syntax checks).
    """
    from GG4 import Brain
    from control.plant import BrainPlant
    return BrainPlant(Brain(random_seed=int(seed)))


# ----------------------------------------------------------------------------
# Main routine.
# ----------------------------------------------------------------------------
def main(
    seeds: List[int] = (0, 1, 2),
    T_impulse: int = 300,
    T_autonomous: int = 400,
    n: int = 6,
    n_probe_orders: int = 12,
    n_noise_samples: int = 2000,
    out_dir: Path = None,
) -> dict:
    if out_dir is None:
        out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = out_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    log_lines: List[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        log_lines.append(msg)

    log("=" * 78)
    log("  M1 — Initial investigation of the real Brain")
    log("=" * 78)

    # ------ 1. Contract ----------------------------------------------------
    t0 = time.time()
    contract = discover_contract(brain_plant_factory, seed=seeds[0])
    log(f"\n[1] Contract (seed {seeds[0]})")
    log(f"    input_dim      = {contract.input_dim}")
    log(f"    obs_dim        = {contract.obs_dim}")
    log(f"    first measure: range [{contract.first_y.min():.2f}, "
        f"{contract.first_y.max():.2f}]")

    # ------ 2. Observation-noise scale ------------------------------------
    log(f"\n[2] Observation noise (R from {n_noise_samples} frozen-state samples)")
    R_info = observation_noise_scale(brain_plant_factory, seed=seeds[0],
                                     n_samples=n_noise_samples)
    log(f"    per-channel std (16 outputs):")
    log(f"    {np.round(R_info['per_channel_std'], 3).tolist()}")
    log(f"    median per-channel std = {R_info['median_std']:.3f}")
    log(f"    ||R||_F              = {R_info['frobenius']:.3f}")

    # ------ 3. Autonomous behaviour ---------------------------------------
    log(f"\n[3] Autonomous behaviour (u ≡ 0 for T={T_autonomous} steps)")
    Y_auto = autonomous_trajectory(brain_plant_factory, seed=seeds[0],
                                   T=T_autonomous)
    auto_std = Y_auto.std(axis=0)
    log(f"    per-channel std over T: median {np.median(auto_std):.2f}, "
        f"max {auto_std.max():.2f}")
    log(f"    autonomous mean (y_ss) sample = {np.round(Y_auto[-50:].mean(0)[:4], 2).tolist()} …")

    # ------ 4. Hankel singular-value knee → order estimate ----------------
    log(f"\n[4] Order via Hankel SV knee (T_impulse={T_impulse})")
    knee = hankel_singular_values(brain_plant_factory, seed=seeds[0],
                                  T=T_impulse, n_probe=n_probe_orders)
    log(f"    σ[0..{n_probe_orders}] = "
        f"{np.round(knee.sigma[:n_probe_orders + 1], 3).tolist()}")
    log(f"    largest drop at k = {knee.knee}  "
        f"(σ[{knee.knee}]/σ[{knee.knee-1}] = {knee.knee_ratio:.2e})")
    log(f"    spec prediction n = 6   → "
        f"{'CONFIRMED' if knee.knee == 6 else f'OBSERVED {knee.knee}'}")

    # ------ 5. Seed-invariance --------------------------------------------
    log(f"\n[5] Seed-invariance (eigvals + DC gain across seeds {list(seeds)})")
    si = seed_invariance(brain_plant_factory, seeds=seeds,
                         n=n, T_impulse=T_impulse)
    log(f"    eigenvalues (sorted, |·|) per seed:")
    for s, ev in zip(si.seeds, si.eig_sorted):
        log(f"      seed {s}: {[f'{abs(x):.4f}' for x in ev]}")
    log(f"    pairwise max |λ_i − λ_j|         = {si.eig_pairwise_max_err:.2e}")
    log(f"    pairwise max ||G_i − G_j||/||G_i|| = {si.G_pairwise_max_relerr:.2e}")
    if si.eig_pairwise_max_err < 1e-6 and si.G_pairwise_max_relerr < 1e-6:
        log("    → seed-invariance CONFIRMED (model is fixed across seeds)")
    else:
        log("    → seeds disagree at finite precision; "
            "structure may still be the same up to noise-induced fit drift")

    # ------ 6. Control authority -----------------------------------------
    log(f"\n[6] Control authority — singular values of full DC gain")
    auth = control_authority(si.models[0].A, si.models[0].B, si.models[0].C)
    log(f"    σ(G) = {[f'{x:.4f}' for x in auth.sigma]}")
    log(f"    σ[1]/σ[0] = {auth.rank_ratio:.4e}  "
        f"({'near rank-1 (one sustainable direction)' if auth.rank_ratio < 0.1 else 'multi-directional'})")
    log(f"    dominant output direction (first 4 comps) = "
        f"{np.round(auth.dominant_output[:4], 3).tolist()} …")
    log(f"    reachable extent on dominant direction in u ∈ [0,1]^m = "
        f"{auth.reachable_extent:.2f}")

    # ------ 7. Save ground-truth models -----------------------------------
    log(f"\n[7] Saving ground-truth models")
    for model in si.models:
        path = gt_dir / f"seed_{model.seed}.npz"
        # Q is fit by stationary covariance match (a slower call than the
        # impulse differencing); the seed_invariance models above only
        # computed (A, B, C). Build a full model for each so the .npz has
        # everything M3 needs.
        full = build_ground_truth_model(brain_plant_factory, seed=model.seed,
                                        n=n, T_impulse=T_impulse)
        full.save(path)
        log(f"    saved {path.relative_to(out_dir.parent)} — "
            f"ρ(A)={np.max(np.abs(np.linalg.eigvals(full.A))):.3f}, "
            f"tr(Q)={np.trace(full.Q):.3f}, tr(R)={np.trace(full.R):.3f}")

    # ------ 8. Figure -----------------------------------------------------
    log(f"\n[8] Figure → {out_dir / 'm1_initial_investigation.png'}")
    fig = _plot_initial_investigation(
        Y_auto=Y_auto, knee=knee, si=si, auth=auth,
        R_info=R_info, seeds=seeds,
    )
    fig_path = out_dir / "m1_initial_investigation.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    elapsed = time.time() - t0
    log(f"\n[done]  total wall-clock = {elapsed:.1f}s")

    text_path = out_dir / "m1_initial_investigation.txt"
    text_path.write_text("\n".join(log_lines), encoding="utf-8")
    return dict(contract=contract, R=R_info, knee=knee,
                seed_invariance=si, auth=auth,
                figure=fig_path, text=text_path)


# ----------------------------------------------------------------------------
# Figure — one panel per question (spec §8 clarity rule).
# ----------------------------------------------------------------------------
def _plot_initial_investigation(Y_auto, knee, si, auth, R_info, seeds):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # (1) Autonomous trajectory — first few channels.
    ax = axes[0, 0]
    t = np.arange(Y_auto.shape[0])
    for ch in range(min(4, Y_auto.shape[1])):
        ax.plot(t, Y_auto[:, ch], lw=0.8, label=f"y[{ch}]")
    ax.set_xlabel("time step")
    ax.set_ylabel("measurement")
    ax.set_title("Autonomous (u ≡ 0) trajectory — 4 channels")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # (2) Observation-noise scale per channel.
    ax = axes[0, 1]
    p = R_info["per_channel_std"].size
    ax.bar(range(p), R_info["per_channel_std"], color="0.5")
    ax.axhline(R_info["median_std"], color="tab:red", ls=":",
               label=f"median = {R_info['median_std']:.2f}")
    ax.set_xlabel("output channel")
    ax.set_ylabel("noise std (√diag R)")
    ax.set_title("Observation-noise scale per channel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (3) Hankel singular-value spectrum (the order knee).
    ax = axes[0, 2]
    s_head = knee.sigma[: min(15, knee.sigma.size)]
    idx = np.arange(s_head.size)
    ax.semilogy(idx, s_head, "o-", color="tab:blue")
    ax.axvline(knee.knee - 0.5, color="tab:red", ls=":",
               label=f"knee at n = {knee.knee}")
    ax.axvline(5.5, color="0.5", ls="--", label="spec predicts n = 6")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("σ (log)")
    ax.set_title("Hankel singular spectrum — order knee")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, which="both")

    # (4) Seed-invariance — eigenvalue magnitudes per seed.
    ax = axes[1, 0]
    width = 0.8 / len(si.seeds)
    n = si.eig_sorted.shape[1]
    x_base = np.arange(n)
    for i, (s, ev) in enumerate(zip(si.seeds, si.eig_sorted)):
        ax.bar(x_base + i * width, np.abs(ev), width=width,
               label=f"seed {s}")
    ax.set_xlabel("eigenvalue index (sorted)")
    ax.set_ylabel("|λ|")
    ax.set_title(f"Seed-invariance of eigenvalues  "
                 f"(max pairwise err = {si.eig_pairwise_max_err:.1e})")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    # (5) Control-authority — singular values of DC gain.
    ax = axes[1, 1]
    ax.bar(range(auth.sigma.size), auth.sigma, color="tab:purple")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("σ(G)")
    ax.set_title(f"Control authority — input→output DC gain singular values\n"
                 f"σ[1]/σ[0] = {auth.rank_ratio:.2e}  "
                 f"({'NEAR RANK-1' if auth.rank_ratio < 0.1 else 'multi-directional'})")
    ax.grid(True, alpha=0.3, axis="y")

    # (6) Dominant output direction (first 8 components for legibility).
    ax = axes[1, 2]
    p = auth.dominant_output.size
    ax.bar(range(p), auth.dominant_output, color="tab:green")
    ax.set_xlabel("output channel")
    ax.set_ylabel("loading")
    ax.set_title("Dominant controllable output direction\n"
                 f"reachable extent in u ∈ [0,1]^m = {auth.reachable_extent:.2f}")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("M1 — Initial investigation of the real Brain "
                 f"(seeds {list(si.seeds)})",
                 fontsize=14, y=1.00)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
