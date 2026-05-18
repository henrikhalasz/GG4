"""
demonstrate_illustrate.py
-------------------------
End-to-end demo of every public Illustrator capability on the example
dataset, plus the `compare` method against synthetically generated
datasets with deliberately different dynamics.

Every figure produced is saved to ./figures/ as a PNG. The script can
be run as

    python demonstrate_illustrate.py

from the directory that contains Illustrator.py and ExampleDataset.npy.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from Illustrator import Illustrator


# ---------------------------------------------------------------------
# Setup: output directory and a small save helper
# ---------------------------------------------------------------------
FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)


def save(fig, name):
    """Save a Matplotlib figure to FIGDIR/<name> and close it."""
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def banner(text):
    """Print a section banner to stdout."""
    print(f"\n{'=' * 64}\n{text}\n{'=' * 64}")


# ---------------------------------------------------------------------
# Load the real dataset and build the main Illustrator
# ---------------------------------------------------------------------
data = np.load("ExampleDataset.npy")
ill = Illustrator(data)
print(f"Loaded ExampleDataset.npy: shape={data.shape}")
print(
    f"  trial_cnt={ill.trial_cnt}, "
    f"timestep_cnt={ill.timestep_cnt}, "
    f"neuron_cnt={ill.neuron_cnt}"
)

# A handful of evenly-spaced neuron indices reused across sections, so
# the subset-style demos are comparable across methods.
n_pick = min(4, ill.neuron_cnt)
selected_neurons = np.linspace(0, ill.neuron_cnt - 1, n_pick).astype(int)
print(f"  reusable neuron subset for demos: {selected_neurons.tolist()}")


# ---------------------------------------------------------------------
# 1. plot_timeseries — every meaningful permutation
# ---------------------------------------------------------------------
banner("1. plot_timeseries")

# 1a. Pure defaults: all neurons, all trials, auto-decided trial overlay.
fig = ill.plot_timeseries()
save(fig, "01a_timeseries_default.png")

# 1b. Single neuron — useful when you want to inspect one carefully.
fig = ill.plot_timeseries(neuron_indices=[0])
save(fig, "01b_timeseries_single_neuron.png")

# 1c. A handful of neurons spread across the population.
fig = ill.plot_timeseries(neuron_indices=selected_neurons)
save(fig, "01c_timeseries_subset_neurons.png")

# 1d. A single trial — band is automatically suppressed (undefined).
fig = ill.plot_timeseries(trial_indices=[0])
save(fig, "01d_timeseries_single_trial.png")

# 1e. A small subset of trials — per-trial overlay shown.
n_few_trials = min(5, ill.trial_cnt)
fig = ill.plot_timeseries(trial_indices=np.arange(n_few_trials))
save(fig, "01e_timeseries_few_trials.png")

# 1f. All trials but with per-trial overlay off (clean mean + band only).
fig = ill.plot_timeseries(show_trials=False)
save(fig, "01f_timeseries_mean_band_only.png")

# 1g. Mean only — no band, no per-trial lines.
fig = ill.plot_timeseries(show_trials=False, show_band=False)
save(fig, "01g_timeseries_mean_only.png")

# 1h. Band only — no mean line on top.
fig = ill.plot_timeseries(show_trials=False, show_mean=False)
save(fig, "01h_timeseries_band_only.png")

# 1i. Force per-trial overlay on, even with many trials (spaghetti view).
fig = ill.plot_timeseries(show_trials=True)
save(fig, "01i_timeseries_force_overlay.png")

# 1j. Custom grid layout: two columns wide instead of auto.
fig = ill.plot_timeseries(n_cols=2)
save(fig, "01j_timeseries_ncols2.png")

# 1k. Combined: a neuron subset on a single trial — minimal view.
fig = ill.plot_timeseries(
    neuron_indices=selected_neurons,
    trial_indices=[0],
)
save(fig, "01k_timeseries_subset_neurons_single_trial.png")


# ---------------------------------------------------------------------
# 2. plot_heatmap
# ---------------------------------------------------------------------
banner("2. plot_heatmap")

# 2a. Default: trial-mean population activity, viridis colormap.
fig = ill.plot_heatmap()
save(fig, "02a_heatmap_trial_mean.png")

# 2b. A single trial, raw values.
fig = ill.plot_heatmap(trial_index=0)
save(fig, "02b_heatmap_trial0.png")

# 2c. Z-scored per neuron — diverging colormap, comparable dynamics.
fig = ill.plot_heatmap(zscore=True)
save(fig, "02c_heatmap_zscored.png")

# 2d. Single trial AND z-scored — emphasises within-trial fluctuations.
fig = ill.plot_heatmap(trial_index=0, zscore=True)
save(fig, "02d_heatmap_trial0_zscored.png")


# ---------------------------------------------------------------------
# 3. plot_autocorrelation — every mode and every meaningful subset
# ---------------------------------------------------------------------
banner("3. plot_autocorrelation")

# 3a. Default mode (overlay), default max_lag (T // 4).
fig = ill.plot_autocorrelation()
save(fig, "03a_acf_overlay_default.png")

# 3b. Subplots mode — one neuron per panel, shared y-axis.
fig = ill.plot_autocorrelation(mode="subplots")
save(fig, "03b_acf_subplots.png")

# 3c. Heatmap mode — population-wide ACF view.
fig = ill.plot_autocorrelation(mode="heatmap")
save(fig, "03c_acf_heatmap.png")

# 3d. A subset of neurons in overlay mode — uncluttered comparison.
fig = ill.plot_autocorrelation(
    neuron_indices=selected_neurons, mode="overlay",
)
save(fig, "03d_acf_subset_overlay.png")

# 3e. First-half-of-trials ACF — split-half stationarity diagnostic.
half = max(1, ill.trial_cnt // 2)
fig = ill.plot_autocorrelation(
    trial_indices=np.arange(half), mode="overlay",
)
save(fig, "03e_acf_first_half_trials.png")

# 3f. Second-half-of-trials ACF — compare with 3e by eye for stationarity.
fig = ill.plot_autocorrelation(
    trial_indices=np.arange(half, ill.trial_cnt), mode="overlay",
)
save(fig, "03f_acf_second_half_trials.png")

# 3g. Custom shorter max_lag — focus on near-lag decay.
short_lag = max(2, ill.timestep_cnt // 8)
fig = ill.plot_autocorrelation(max_lag=short_lag)
save(fig, "03g_acf_short_lag.png")


# ---------------------------------------------------------------------
# 4. compute_snr
# ---------------------------------------------------------------------
banner("4. compute_snr")

# 4a. With diagnostic plot — `compute_snr` returns (snr_array, fig).
snr, fig = ill.compute_snr(plot=True)
save(fig, "04a_snr_diagnostic.png")

print("\n  Bias-corrected SNR per neuron:")
for n, s in enumerate(snr):
    pretty = "inf (zero-noise neuron)" if np.isnan(s) else f"{s:.4f}"
    print(f"    neuron {n:2d}: {pretty}")

# 4b. Plot suppressed — the function still returns the SNR array.
snr_silent, _ = ill.compute_snr(plot=False)
identical = np.allclose(snr, snr_silent, equal_nan=True)
print(
    f"\n  compute_snr(plot=False) returns shape={snr_silent.shape}; "
    f"identical to plotted call: {identical}"
)


# ---------------------------------------------------------------------
# 5. plot_correlation_matrix
# ---------------------------------------------------------------------
banner("5. plot_correlation_matrix")

# 5a. Default: clustered, trial-mean.
C, fig = ill.plot_correlation_matrix()
save(fig, "05a_corr_default_clustered.png")
print(f"  Returned correlation matrix shape: {C.shape}")
N_sel = C.shape[0]
print(
    f"  Off-diagonal correlation range: "
    f"[{np.nanmin(C - np.eye(N_sel)):.3f}, "
    f"{np.nanmax(C - np.eye(N_sel)):.3f}]"
)

# 5b. Unclustered (original neuron order) — see raw spatial structure.
_, fig = ill.plot_correlation_matrix(cluster=False)
save(fig, "05b_corr_unclustered.png")

# 5c. From a single trial instead of the trial mean — noisier, includes
#     trial-specific fluctuations.
_, fig = ill.plot_correlation_matrix(trial_index=0)
save(fig, "05c_corr_trial0.png")

# 5e. Single trial, unclustered — shows raw per-trial structure.
_, fig = ill.plot_correlation_matrix(trial_index=0, cluster=False)
save(fig, "05e_corr_trial0_unclustered.png")

# ---------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------
n_saved = len([f for f in os.listdir(FIGDIR) if f.endswith(".png")])
print(
    f"\n{'=' * 64}\n"
    f"DONE — {n_saved} figures saved to ./{FIGDIR}/\n"
    f"{'=' * 64}"
)

