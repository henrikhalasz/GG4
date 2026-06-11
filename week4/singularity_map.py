"""singularity_map.py -- where the arm Jacobian is well-conditioned (Step 2A).

The 2x2 hand Jacobian has ``det J = 900 sin(theta_e)``, so the arm is singular at
``theta_e = 0`` (elbow straight -> the outer rim r = 60) and ``theta_e = pi``
(elbow folded -> the centre r = 0). Because a change of the absolute shoulder
angle merely rotates the whole arm, the singular values of J are rotation
invariant: ``det J``, ``sigma_min(J)`` and ``cond(J)`` depend ONLY on the elbow
angle, i.e. only on the hand radius ``r = 60*|cos(theta_e/2)|``. The conditioning
map is therefore radially symmetric -- concentric rings between the two singular
loci.

We pick a **safe working annulus r in [5, 57]**: inside r = 5 the arm is near the
folded centre singularity, outside r = 57 it is near the straight rim
singularity (where the home pose (60,0) sits). The reacher confines its targets
to this annulus.

Produces ``singularity_map.pdf``: a sigma_min heatmap over the disk (left) and the
three conditioning measures vs radius (right), with the loci, annulus and home
marked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kinematics as kin

HERE = Path(__file__).resolve().parent
R_INNER, R_OUTER = 5.0, 57.0  # safe annulus


def theta_e_for_radius(r):
    """Elbow angle (elbow-up branch) that places the hand at radius ``r``."""
    cos_e = np.clip((np.asarray(r) ** 2 - 1800.0) / 1800.0, -1.0, 1.0)
    return np.arccos(cos_e)


def conditioning_at_radius(r):
    """(|det J|, sigma_min, cond) at hand radius ``r`` (shoulder angle is irrelevant)."""
    te = float(theta_e_for_radius(r))
    J = kin.jacobian([0.0, te])
    s = np.linalg.svd(J, compute_uv=False)
    det = abs(kin.L1 * kin.L2 * np.sin(te))
    cond = s[0] / s[-1] if s[-1] > 1e-12 else np.inf
    return det, float(s[-1]), float(cond)


def make_figure(path: str | Path = HERE / "singularity_map.pdf") -> Path:
    fig, (axd, axr) = plt.subplots(1, 2, figsize=(13, 6))

    # --- left: sigma_min heatmap over the reachable disk --------------------
    g = np.linspace(-60, 60, 401)
    X, Y = np.meshgrid(g, g)
    R = np.hypot(X, Y)
    TE = theta_e_for_radius(R)
    # sigma_min depends only on theta_e; build it via the closed-form svd per r.
    Smin = np.full_like(R, np.nan)
    inside = R <= kin.REACH
    # vectorise: sigma_min is a 1-D function of r -> sample on a fine r grid, interp.
    r_lut = np.linspace(0, kin.REACH, 2000)
    smin_lut = np.array([conditioning_at_radius(max(r, 1e-6))[1] for r in r_lut])
    Smin[inside] = np.interp(R[inside], r_lut, smin_lut)

    pcm = axd.pcolormesh(X, Y, Smin, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=axd, label=r"$\sigma_{\min}(J)$")
    # singular loci + annulus + home
    axd.add_patch(plt.Circle((0, 0), kin.REACH, fill=False, ec="red", lw=2,
                             ls="--", label="outer singular rim (r=60)"))
    axd.add_patch(plt.Circle((0, 0), 0.7, fc="red", ec="red"))
    axd.add_patch(plt.Circle((0, 0), R_INNER, fill=False, ec="w", lw=1.3, ls=":"))
    axd.add_patch(plt.Circle((0, 0), R_OUTER, fill=False, ec="w", lw=1.3, ls=":",
                             label=f"safe annulus [{R_INNER:.0f},{R_OUTER:.0f}]"))
    axd.plot(60, 0, "*", color="yellow", ms=18, mec="k",
             label="home (60,0) on rim")
    axd.plot(0, 0, "o", color="red", ms=6)
    axd.annotate("centre singular (r=0)", (0, 0), color="red", fontsize=8,
                 textcoords="offset points", xytext=(6, 6))
    axd.set_aspect("equal")
    axd.set_xlim(-63, 63)
    axd.set_ylim(-63, 63)
    axd.set_xlabel("x")
    axd.set_ylabel("y")
    axd.set_title(r"Conditioning over the disk ($\sigma_{\min}$, radially symmetric)")
    axd.legend(loc="upper left", fontsize=7)

    # --- right: det, sigma_min, cond vs radius ------------------------------
    r = np.linspace(0.05, kin.REACH - 0.05, 600)
    det = np.array([conditioning_at_radius(ri)[0] for ri in r])
    smin = np.array([conditioning_at_radius(ri)[1] for ri in r])
    cond = np.array([conditioning_at_radius(ri)[2] for ri in r])

    axr.plot(r, det, label=r"$|\det J| = 900\,\sin\theta_e$", color="tab:blue")
    axr.plot(r, smin * 30, label=r"$30\,\sigma_{\min}(J)$", color="tab:green")
    axr.set_xlabel("hand radius r")
    axr.set_ylabel("magnitude")
    axr.axvspan(R_INNER, R_OUTER, color="0.85", label=f"safe annulus [{R_INNER:.0f},{R_OUTER:.0f}]")
    axr.axvline(0, color="red", ls="--", lw=1)
    axr.axvline(60, color="red", ls="--", lw=1)
    axr.plot(60, 0, "*", color="orange", ms=14, mec="k", clip_on=False,
             label="home (r=60)")

    axc = axr.twinx()
    axc.semilogy(r, cond, color="tab:red", ls="-.", label=r"$\mathrm{cond}(J)$")
    axc.set_ylabel(r"$\mathrm{cond}(J)$ (log)", color="tab:red")
    axc.tick_params(axis="y", labelcolor="tab:red")

    l1, lab1 = axr.get_legend_handles_labels()
    l2, lab2 = axc.get_legend_handles_labels()
    axr.legend(l1 + l2, lab1 + lab2, loc="upper center", fontsize=8)
    axr.set_title("Conditioning vs radius (singular at r=0 and r=60)")
    fig.tight_layout()

    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path


def _main():
    print("=== singularity_map.py ===")
    print("det J = 900 sin(theta_e); singular at theta_e=0 (r=60 rim) and "
          "theta_e=pi (r=0 centre).")
    print(f"\n{'r':>6s} {'theta_e':>8s} {'|detJ|':>9s} {'sigma_min':>10s} {'cond':>9s}")
    for r in [0.5, R_INNER, 10, 20, 30, 40, R_OUTER, 59.5]:
        te = float(theta_e_for_radius(r))
        det, smin, cond = conditioning_at_radius(r)
        print(f"{r:6.1f} {te:8.4f} {det:9.2f} {smin:10.4f} {cond:9.2f}")

    di, si, ci = conditioning_at_radius(R_INNER)
    do, so, co = conditioning_at_radius(R_OUTER)
    print(f"\nSafe annulus margins:")
    print(f"  inner r={R_INNER}: sigma_min={si:.3f}, cond={ci:.2f}")
    print(f"  outer r={R_OUTER}: sigma_min={so:.3f}, cond={co:.2f}")
    print(f"  home  r=60.0: cond -> inf (on the outer singular rim)")

    out = make_figure()
    print(f"\nsaved figure -> {out}")


if __name__ == "__main__":
    _main()
