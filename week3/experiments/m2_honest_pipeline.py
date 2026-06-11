"""M2 — Honest N4SID → EM pipeline at n = 6 on the Brain (spec §3).

Runs the *deployed* identification pipeline against the real Brain and the
M1 yardstick saved in ``results/ground_truth/``:

    1. Probe (one noisy run, ``u ~ U[0,1]^2``, ``T_cal = 1000``).
    2. N4SID input-output subspace init at ``n = 6``.
    3. Affine EM with known inputs to refinement.
    4. Hold-out prediction on a fresh probe; report against the noise floor.
    5. Basis-free agreement with the yardstick (eigenvalues, DC gain,
       Markov parameters).

The honest pipeline NEVER reads the yardstick — only this experiment compares
the two after the fact.

Outputs::

    results/m2_honest_pipeline.png
    results/m2_honest_pipeline.txt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent  # week3/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "week 1"))

from analysis.ground_truth import GroundTruthModel  # noqa: E402
from estimator.identify import EstimatorNew, estimate_order  # noqa: E402
from control.plant import BrainPlant  # noqa: E402


def brain_plant_factory(seed: int):
    from GG4 import Brain
    return BrainPlant(Brain(random_seed=int(seed)))


def _probe(plant_factory, seed: int, T_cal: int, m: int,
           probe_seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Single Uniform[0,1]^m noisy probe; returns (Y, U)."""
    rng = np.random.default_rng(probe_seed)
    U = rng.uniform(0.0, 1.0, size=(T_cal, m))
    plant = plant_factory(seed)
    p = len(plant.measure())
    Y = np.empty((T_cal, p))
    # reset so the first measure() pairs with U[0] driving the next step
    plant = plant_factory(seed)
    Y[0] = np.asarray(plant.measure(), dtype=float)
    plant.next_state(U[0])
    for t in range(1, T_cal):
        Y[t] = np.asarray(plant.measure(), dtype=float)
        plant.next_state(U[t])
    return Y, U


def _markov_chain(A, B, C, K):
    H = np.zeros((K, C.shape[0], B.shape[1]))
    AB = B.copy()
    for k in range(K):
        H[k] = C @ AB
        AB = A @ AB
    return H


def _dc_gain(A, B, C):
    n = A.shape[0]
    return C @ np.linalg.solve(np.eye(n) - A, B)


def main(
    seeds: List[int] = (0, 1, 2),
    T_cal: int = 1000,
    T_holdout: int = 500,
    n: int = 6,
    out_dir: Path = None,
) -> Dict:
    if out_dir is None:
        out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = out_dir / "ground_truth"

    log_lines: List[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        log_lines.append(msg)

    log("=" * 78)
    log("  M2 — Honest N4SID → EM pipeline at n = 6")
    log("=" * 78)

    per_seed: List[Dict] = []
    t0 = time.time()
    for s in seeds:
        log(f"\n[seed {s}]")

        # 1. Calibration probe.
        Y_cal, U_cal = _probe(brain_plant_factory, seed=s, T_cal=T_cal,
                              m=2, probe_seed=42 + s)

        # 2. Order selection (sanity — should land at 6).
        n_hat, sigma_n4 = estimate_order(Y_cal, U_cal, max_n=10)
        log(f"  order selection (N4SID Hankel knee): n_hat = {n_hat}  "
            f"({'matches spec' if n_hat == 6 else 'differs'})")

        # 3. N4SID + EM.
        est = EstimatorNew().fit(Y_cal, U_cal, n=n, init="n4sid",
                                  max_iter=80, tol=1e-5)
        rho = float(np.max(np.abs(np.linalg.eigvals(est.A))))
        ll_path = est.loglik_history()
        log(f"  EM iters = {est.fit_iters},  ll: "
            f"{ll_path[0]:.1f} → {ll_path[-1]:.1f},  ρ(A_fit) = {rho:.4f}")

        # 4. Hold-out one-step prediction on a fresh probe (different seed).
        Y_h, U_h = _probe(brain_plant_factory, seed=s, T_cal=T_holdout,
                          m=2, probe_seed=9999 + s)
        rep = est.validate(Y_val=Y_h, U_val=U_h)
        y_rms = float(np.sqrt(np.mean(Y_h ** 2)))
        one_step = rep["one_step_rms"]
        # Noise floor: trace(R) per channel → median per-channel std.
        noise_floor_per_channel = float(np.sqrt(np.median(np.diag(est.R))))
        log(f"  held-out one-step RMS = {one_step:.3f}  "
            f"(y-RMS {y_rms:.3f}; ratio {one_step / y_rms:.3f})")
        log(f"  fit noise floor √median(diag R) = "
            f"{noise_floor_per_channel:.3f}  "
            f"⇒ one-step / noise-floor = {one_step / noise_floor_per_channel:.2f}")

        # 5. Basis-free comparison against the yardstick saved in M1.
        gt = GroundTruthModel.load(gt_dir / f"seed_{s}.npz")
        eig_fit = np.sort_complex(np.linalg.eigvals(est.A))
        eig_gt = np.sort_complex(np.linalg.eigvals(gt.A))
        eig_err = float(np.abs(eig_fit - eig_gt).max())
        G_fit = _dc_gain(est.A, est.B, est.C)
        G_gt = _dc_gain(gt.A, gt.B, gt.C)
        G_rel = float(np.linalg.norm(G_fit - G_gt)
                      / max(np.linalg.norm(G_gt), 1e-12))
        K = 60
        H_fit = _markov_chain(est.A, est.B, est.C, K)
        H_gt = _markov_chain(gt.A, gt.B, gt.C, K)
        H_rel = float(np.linalg.norm(H_fit - H_gt)
                      / max(np.linalg.norm(H_gt), 1e-12))
        log(f"  vs yardstick — max |Δeig| = {eig_err:.3e}, "
            f"DC gain rel err = {G_rel:.3e}, Markov rel err = {H_rel:.3e}")

        per_seed.append(dict(
            seed=s, n_hat=int(n_hat), rho=rho,
            ll0=ll_path[0], ll_final=ll_path[-1], iters=est.fit_iters,
            one_step=one_step, y_rms=y_rms,
            noise_floor=noise_floor_per_channel,
            eig_err=eig_err, G_rel=G_rel, H_rel=H_rel,
            eig_fit=eig_fit, eig_gt=eig_gt, sigma_n4=sigma_n4,
            H_fit=H_fit, H_gt=H_gt,
        ))

    log(f"\n[done]  total wall-clock = {time.time() - t0:.1f}s")

    # ------ Figure --------------------------------------------------------
    fig = _plot_m2_summary(per_seed)
    fig_path = out_dir / "m2_honest_pipeline.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"\nfigure: {fig_path}")

    text_path = out_dir / "m2_honest_pipeline.txt"
    text_path.write_text("\n".join(log_lines), encoding="utf-8")
    return dict(per_seed=per_seed, figure=fig_path, text=text_path)


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def _plot_m2_summary(per_seed):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. N4SID order-selection spectrum across seeds.
    ax = axes[0, 0]
    for r in per_seed:
        sig = r["sigma_n4"][:12]
        ax.semilogy(sig / max(sig[0], 1e-12), "o-",
                    label=f"seed {r['seed']}", alpha=0.85)
    ax.axvline(5.5, color="0.5", ls="--", label="spec n = 6")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("σ / σ[0] (log)")
    ax.set_title("N4SID Hankel singular spectrum (probe data)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # 2. Eigenvalues fit vs yardstick (mag).
    ax = axes[0, 1]
    seeds_plot = [r["seed"] for r in per_seed]
    width = 0.35
    n = per_seed[0]["eig_fit"].size
    base = np.arange(n)
    for i, r in enumerate(per_seed):
        ax.bar(base + (2 * i - len(per_seed) + 1) * width / len(per_seed),
               np.abs(r["eig_fit"]), width=width / len(per_seed),
               label=f"seed {r['seed']} (fit)", alpha=0.7)
    # Single yardstick bar set (seeds match anyway)
    ax.plot(base, np.abs(per_seed[0]["eig_gt"]),
            "k_", ms=20, label="yardstick |λ|")
    ax.set_xlabel("eigenvalue index (sorted)")
    ax.set_ylabel("|λ|")
    ax.set_title("Eigenvalues — honest fit vs yardstick")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Markov param decay (one channel) — fit vs yardstick.
    ax = axes[1, 0]
    K = per_seed[0]["H_fit"].shape[0]
    t = np.arange(K)
    out_ch, in_ch = 0, 0
    for r in per_seed:
        ax.plot(t, r["H_fit"][:, out_ch, in_ch], "-",
                label=f"fit seed {r['seed']}", alpha=0.75)
    ax.plot(t, per_seed[0]["H_gt"][:, out_ch, in_ch], "k--",
            lw=1.5, label="yardstick")
    ax.set_xlabel("k (lag)")
    ax.set_ylabel(f"H_k[{out_ch}, {in_ch}] = C A^k B")
    ax.set_title("Markov-param decay (channel 0 ← input 0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Held-out prediction error and basis-free agreement bars.
    ax = axes[1, 1]
    labels = ["one-step / y-RMS", "one-step / noise floor",
              "eig err", "DC-gain rel err", "Markov rel err"]
    x = np.arange(len(labels))
    width = 0.8 / len(per_seed)
    for i, r in enumerate(per_seed):
        bars = [
            r["one_step"] / r["y_rms"],
            r["one_step"] / r["noise_floor"],
            r["eig_err"],
            r["G_rel"],
            r["H_rel"],
        ]
        ax.bar(x + (i - 1) * width, bars, width=width,
               label=f"seed {r['seed']}")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_title("Honest pipeline — held-out + basis-free agreement")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both", axis="y")

    fig.suptitle("M2 — Honest N4SID → EM at n = 6 on the Brain",
                 fontsize=14, y=1.00)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
