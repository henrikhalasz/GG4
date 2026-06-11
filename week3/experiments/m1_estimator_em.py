"""M1 experiment: EM validation on the white-box Simulator (Fig 8).

What we check (spec §5.4 / §8 / §10 M1 acceptance):
  1. Run a Uniform[0,1] probe through ``default_neural_system`` to collect (Y, U).
  2. Fit ``EstimatorNew`` (affine LGSSM with EM).
  3. Validate against the ground truth:
       - EM log-likelihood non-decreasing.
       - Fitted G = C(I-A)⁻¹B matches true G within ~10%.
       - State R² (affine alignment) >= 0.95.
       - Stable fit (rho(A) < 1).

Produces:
  - ``results/fig8_em_validation.png`` (4-panel: ll curve, G scatter, z0 scatter,
    state R² text + diagnostics).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent  # week3/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "week 1"))

import Simulator as sim  # noqa: E402
from estimator.identify import EstimatorNew  # noqa: E402


def collect_probe(system, T_cal, seed=42):
    """Drive system with Uniform[0,1] inputs, return (Y, U)."""
    rng = np.random.default_rng(seed)
    U = rng.uniform(0.0, 1.0, size=(T_cal, system.input_dim))
    data = system.simulate(T_cal, U=U)
    return data["y"], U, data["x"][:T_cal]


def main():
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Fit on default_neural_system --------------------------------------
    system = sim.default_neural_system(seed=0, obs_dim=16)
    n = system.A.shape[0]
    T_cal = 800
    Y, U, x_true = collect_probe(system, T_cal, seed=42)

    est = EstimatorNew().fit(Y, U, n=n, max_iter=60, tol=1e-5, verbose=False)
    truth = dict(A=system.A, B=system.B, C=system.C,
                 a=np.zeros(n), c=np.zeros(system.obs_dim))
    rep = est.validate(true_model=truth, x_true=x_true)

    # Held-out one-step prediction error — SAME system instance, fresh probe seed.
    system_held = sim.default_neural_system(seed=0, obs_dim=16)
    system_held.reset_seed(seed=999)
    Y_v, U_v, _ = collect_probe(system_held, 300, seed=999)
    rep_held = est.validate(true_model=truth, Y_val=Y_v, U_val=U_v)
    rep["one_step_rms"] = rep_held.get("one_step_rms", float("nan"))
    y_rms_held = float(np.sqrt(np.mean(Y_v ** 2)))
    rep["one_step_rel"] = rep["one_step_rms"] / y_rms_held if y_rms_held > 0 else float("nan")

    print("=" * 64)
    print("M1 — EM validation on default_neural_system")
    print("=" * 64)
    print(f"  T_cal           = {T_cal}")
    print(f"  n (latent)      = {n}")
    print(f"  EM iterations   = {est.fit_iters}")
    print(f"  loglik final    = {rep['loglik_final']:.2f}")
    print(f"  EM monotone     = {rep['em_monotone']}")
    print(f"  rho(A_fit)      = {rep['A_spectral_radius']:.4f}")
    print(f"  state R² (aligned) = {rep['state_r2']:.4f}")
    print(f"  ||G_fit - G_true|| / ||G_true|| = {rep['G_rel_err']:.4f}")
    print(f"  ||z0_fit - z0_true|| = {rep['z0_err']:.4f}")
    print(f"  one-step RMS (held-out) = {rep['one_step_rms']:.4f}  (y_RMS-relative: {rep['one_step_rel']:.4f})")
    print(f"  params finite   = {rep['params_finite']}")

    # ---- Plot fig 8 ---------------------------------------------------------
    ll = est.loglik_history()
    G_fit = rep["G_fit"]
    G_true = rep["G_true"]
    z0_fit = rep["z0_fit"]
    z0_true = rep["z0_true"]

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # (1) Log-lik convergence
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(np.arange(len(ll)), ll, "-o", color="tab:blue", markersize=4)
    ax1.set_xlabel("EM iteration")
    ax1.set_ylabel("log-likelihood")
    ax1.set_title(f"EM convergence — monotone: {rep['em_monotone']}")
    ax1.grid(True, alpha=0.3)

    # (2) Fitted G entries vs true G entries
    ax2 = fig.add_subplot(gs[0, 1])
    gt = G_true.flatten()
    gf = G_fit.flatten()
    ax2.scatter(gt, gf, s=20, alpha=0.7, color="tab:orange")
    lo, hi = float(min(gt.min(), gf.min())), float(max(gt.max(), gf.max()))
    pad = 0.05 * (hi - lo + 1e-9)
    ax2.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="0.5", lw=1)
    ax2.set_xlim(lo - pad, hi + pad)
    ax2.set_ylim(lo - pad, hi + pad)
    ax2.set_xlabel("G_true entry")
    ax2.set_ylabel("G_fit entry")
    ax2.set_title(f"G = C(I-A)⁻¹B  —  rel.err = {rep['G_rel_err']:.3f}")
    ax2.grid(True, alpha=0.3)

    # (3) z0 entries vs true z0 entries
    ax3 = fig.add_subplot(gs[1, 0])
    zt = z0_true.flatten()
    zf = z0_fit.flatten()
    ax3.scatter(zt, zf, s=24, alpha=0.7, color="tab:green")
    lo, hi = float(min(zt.min(), zf.min())), float(max(zt.max(), zf.max()))
    pad = 0.05 * (hi - lo + 1e-9)
    ax3.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="0.5", lw=1)
    ax3.set_xlim(lo - pad, hi + pad)
    ax3.set_ylim(lo - pad, hi + pad)
    ax3.set_xlabel("z0_true entry")
    ax3.set_ylabel("z0_fit entry")
    ax3.set_title(f"z0 = C(I-A)⁻¹a + c  —  ||err|| = {rep['z0_err']:.3f}")
    ax3.grid(True, alpha=0.3)

    # (4) Summary text panel
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    txt = (
        "EM diagnostics — default_neural_system\n"
        "-----------------------------------------\n"
        f"T_cal = {T_cal}, n = {n}, p = {system.obs_dim}, m = {system.input_dim}\n"
        f"EM iters  : {est.fit_iters}\n"
        f"log-lik   : {ll[0]:.2f} → {ll[-1]:.2f}\n"
        f"monotone  : {rep['em_monotone']}\n"
        f"ρ(A_fit)  : {rep['A_spectral_radius']:.4f}\n\n"
        f"state R² (affine-aligned) : {rep['state_r2']:.4f}\n"
        f"G rel.err                  : {rep['G_rel_err']:.4f}\n"
        f"z0 err                     : {rep['z0_err']:.4f}\n"
        f"1-step held-out RMS        : {rep['one_step_rms']:.4f}\n"
        f"  (y-RMS relative)         : {rep['one_step_rel']:.4f}\n"
        f"params finite              : {rep['params_finite']}\n"
    )
    ax4.text(0.0, 1.0, txt, family="monospace", fontsize=10,
             va="top", transform=ax4.transAxes)

    fig.suptitle("Fig 8 — EM validation (estimator_new on default_neural_system)",
                 fontsize=12)
    out_path = out_dir / "fig8_em_validation.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")
    print(f"  (warm start info: {est.warm_start_info})")
    return rep


if __name__ == "__main__":
    main()
