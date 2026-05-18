"""
Illustrator: exploration and visualisation of neural activity data
with shape (Trials, Timepoints, Neurons).

Works on both real experimental data and Simulator output.
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


class Illustrator:
    """
    Inspect, summarise, and visualise a neural-activity dataset.

    The dataset is a 3D numpy array of shape
    (trial_cnt, timestep_cnt, neuron_cnt). Every method operates on
    this convention; axis 0 is always trials, axis 1 is always time,
    axis 2 is always neurons.
    """

    # Beyond this many trials, per-trial lines turn into spaghetti and
    # we default to mean + band only. Caller can override via show_trials.
    MAX_TRIALS_OVERLAY = 10

    def __init__(self, observation: np.ndarray):
        """
        Initialise from a 3D array of shape (Trials, Timepoints, Neurons).

        Parameters
        ----------
        observation : np.ndarray
            3D array of neural activity.

        Raises
        ------
        TypeError
            If `observation` is not a numpy array.
        ValueError
            If `observation` is not 3D, has a zero-length axis, or
            contains NaN / Inf.
        """
        if not isinstance(observation, np.ndarray):
            raise TypeError(
                f"observation must be a numpy array, got "
                f"{type(observation).__name__}"
            )
        if observation.ndim != 3:
            raise ValueError(
                f"observation must be 3D (Trials, Timepoints, Neurons); "
                f"got shape {observation.shape}"
            )
        if 0 in observation.shape:
            raise ValueError(
                f"observation has an empty axis: shape {observation.shape}"
            )
        if not np.all(np.isfinite(observation)):
            raise ValueError("observation contains NaN or Inf values")

        # Store as float64 for numerical stability in later reductions
        self.observation = np.asarray(observation, dtype=np.float64)
        self.trial_cnt, self.timestep_cnt, self.neuron_cnt = self.observation.shape

        # Cache once: trial-averaged signal and trial-to-trial std.
        # ddof is clamped so single-trial input gives an exact-zero std
        # (ddof=0 → divide by 1) instead of NaN (ddof=1 → divide by 0).
        # With trial_cnt >= 2 this is the usual sample std.
        self._trial_mean = self.observation.mean(axis=0)              # (T_steps, N)
        self._trial_std  = self.observation.std(
            axis=0, ddof=min(1, self.trial_cnt - 1),
        )                                                              # (T_steps, N)

    def plot_timeseries(
        self,
        neuron_indices=None,
        trial_indices=None,
        show_trials: bool | None = None,
        show_mean: bool = True,
        show_band: bool = True,
        ncols: int | None = None,
    ):
        """
        Plot neural activity over time, one subplot per neuron.

        For each selected neuron, individual trials are drawn as thin
        semi-transparent lines; the trial-averaged mean is a bold line;
        a shaded band shows ±1 standard deviation across trials.

        Parameters
        ----------
        neuron_indices : array-like of int, optional
            Which neurons to plot. Defaults to all neurons.
        trial_indices : array-like of int, optional
            Which trials to include. The mean and band are computed over
            this subset only. Defaults to all trials.
        show_trials : bool, optional
            Whether to draw the thin per-trial lines. If None (default),
            they are shown when n_trials_plot <= MAX_TRIALS_OVERLAY and
            hidden otherwise to avoid spaghetti.
        show_mean : bool, default True
            Whether to overlay the trial-mean as a bold line.
        show_band : bool, default True
            Whether to shade ±1 SD across trials. Automatically suppressed
            when fewer than 2 trials are selected.
        ncols : int, optional
            Number of columns in the subplot grid. Defaults to an
            approximately-square layout, capped at 4 columns.

        Returns
        -------
        matplotlib.figure.Figure
        The figure, so the caller can save or further customise it.

        Notes
        -----
        Uses sample standard deviation (ddof=1) for the band, which is
        undefined for a single trial.
        """
        # --- 1. Resolve and validate neuron / trial selections --------------
        if neuron_indices is None:
            neuron_idx = np.arange(self.neuron_cnt)
        else:
            neuron_idx = np.asarray(neuron_indices, dtype=int)
            if neuron_idx.ndim != 1:
                raise ValueError("neuron_indices must be 1D")
            if neuron_idx.min() < 0 or neuron_idx.max() >= self.neuron_cnt:
                raise ValueError(
                    f"neuron_indices out of range [0, {self.neuron_cnt - 1}]"
                )

        if trial_indices is None:
            trial_idx = np.arange(self.trial_cnt)
        else:
            trial_idx = np.asarray(trial_indices, dtype=int)
            if trial_idx.ndim != 1:
                raise ValueError("trial_indices must be 1D")
            if trial_idx.min() < 0 or trial_idx.max() >= self.trial_cnt:
                raise ValueError(
                    f"trial_indices out of range [0, {self.trial_cnt - 1}]"
                )

        n_neurons_plot = len(neuron_idx)
        n_trials_plot  = len(trial_idx)

        # Auto-decide whether to overlay per-trial lines. With too many
        # trials they pile up into spaghetti and obscure the mean/band.
        if show_trials is None:
            show_trials = n_trials_plot <= self.MAX_TRIALS_OVERLAY

        # --- 2. Slice the data once ----------------------------------------
        # Fancy indexing: pick a subset of trials AND a subset of neurons.
        # np.ix_ gives the cartesian product so shapes broadcast cleanly.
        # data shape: (n_trials_plot, T_steps, n_neurons_plot)
        data = self.observation[np.ix_(trial_idx,
                                        np.arange(self.timestep_cnt),
                                        neuron_idx)]

        # Trial-mean and trial-std OVER THE SELECTED TRIALS.
        # We don't reuse self._trial_mean here because that's over all trials.
        mean_t = data.mean(axis=0)                          # (T_steps, n_neurons_plot)
        if n_trials_plot >= 2:
            std_t = data.std(axis=0, ddof=1)                # (T_steps, n_neurons_plot)
        else:
            std_t = None                                    # band undefined

        # --- 3. Lay out the grid -------------------------------------------
        fig, axes = self._neuron_grid(n_neurons_plot, ncols=ncols)
        nrows, ncols = axes.shape

        # --- 4. Draw each neuron -------------------------------------------
        t = np.arange(self.timestep_cnt)                    # x-axis
        cmap = plt.get_cmap("tab20")                        # 20 distinct colours, covers N=16

        for panel_i, neuron in enumerate(neuron_idx):
            row, col = divmod(panel_i, ncols)
            ax = axes[row, col]
            color = cmap(neuron % cmap.N)                   # neuron-id → colour, stable across plots

            # 4a. Thin per-trial lines.
            # data[:, :, panel_i] has shape (n_trials_plot, T_steps).
            if show_trials:
                for tr in range(n_trials_plot):
                    ax.plot(t, data[tr, :, panel_i],
                            color=color, alpha=0.3, linewidth=0.8)

            # 4b. Bold trial-mean.
            if show_mean:
                ax.plot(t, mean_t[:, panel_i],
                        color=color, linewidth=2.0, label="mean")

            # 4c. ±1 SD shaded band (only if defined). Use a neutral gray
            # with a visible edge so the band's extent reads clearly
            # against the neuron-coloured mean / per-trial lines.
            if show_band and std_t is not None:
                ax.fill_between(
                    t,
                    mean_t[:, panel_i] - std_t[:, panel_i],
                    mean_t[:, panel_i] + std_t[:, panel_i],
                    facecolor="0.55", alpha=0.30,
                    edgecolor="0.25", linewidth=0.8,
                )

            ax.set_title(f"neuron {neuron}", fontsize=9)
            ax.tick_params(labelsize=8)

        # 4d. Blank out any unused panels (grid not exactly filled).
        self._blank_unused(axes, n_neurons_plot)

        # --- 5. Shared axis labels and title -------------------------------
        # One y-label on the left column, one x-label on the bottom row.
        for row in range(nrows):
            axes[row, 0].set_ylabel("activity", fontsize=9)
        for col in range(ncols):
            # Find the bottom-most visible row in this column.
            for row in range(nrows - 1, -1, -1):
                if axes[row, col].get_visible():
                    axes[row, col].set_xlabel("timestep", fontsize=9)
                    break

        title = (f"Time series  "
                 f"({n_trials_plot} trial{'s' if n_trials_plot != 1 else ''}, "
                 f"{n_neurons_plot} neuron{'s' if n_neurons_plot != 1 else ''})")
        if n_trials_plot == 1 and show_band:
            title += "  [band suppressed: single trial]"
        elif not show_trials and n_trials_plot > self.MAX_TRIALS_OVERLAY:
            title += f"  [per-trial lines hidden: >{self.MAX_TRIALS_OVERLAY} trials]"
        fig.suptitle(title, fontsize=11)

        fig.tight_layout()
        return fig

    def plot_heatmap(self, trial_index: int | None = None, zscore: bool = False):
        """
        Show population activity as a (Neurons x Time) heatmap.

        Parameters
        ----------
        trial_index : int, optional
            Which trial to display. If None (default), the trial-averaged
            signal is shown.
        zscore : bool, default False
            If True, z-score each row independently across time and use a
            diverging colormap centred at 0. Useful when neurons differ
            widely in absolute scale and you want to compare temporal
            dynamics. Rows with zero temporal variance are left at 0
            (no division by zero).

        Returns
        -------
        matplotlib.figure.Figure
        """
        # --- 1. Select the (N, T) matrix to display -----------------------
        if trial_index is None:
            data = self._trial_mean.T                       # (N, T)
            source = "trial mean"
        else:
            if not (0 <= trial_index < self.trial_cnt):
                raise ValueError(
                    f"trial_index out of range [0, {self.trial_cnt - 1}]; "
                    f"got {trial_index}"
                )
            data = self.observation[trial_index].T          # (N, T)
            source = f"trial {trial_index}"

        # --- 2. Optional per-neuron z-scoring across time -----------------
        if zscore:
            row_mean = data.mean(axis=1, keepdims=True)     # (N, 1)
            row_std  = data.std(axis=1, ddof=1, keepdims=True)
            # Guard zero-variance rows: keep them at 0 instead of NaN/inf.
            safe_std = np.where(row_std == 0, 1.0, row_std)
            display = (data - row_mean) / safe_std
            cmap = "RdBu_r"
            vmax = float(np.abs(display).max()) or 1.0      # symmetric, non-zero
            vmin = -vmax
            cbar_label = "z-score"
        else:
            display = data
            cmap = "viridis"
            vmin, vmax = float(display.min()), float(display.max())
            cbar_label = "activity"

        # --- 3. Draw -------------------------------------------------------
        # Height scales with neuron count so tick labels stay readable.
        fig, ax = plt.subplots(figsize=(8, 0.3 * self.neuron_cnt + 1.5))
        im = ax.imshow(
            display, aspect="auto", cmap=cmap,
            vmin=vmin, vmax=vmax,
            origin="lower", interpolation="nearest",
        )

        ax.set_xlabel("timestep")
        ax.set_ylabel("neuron")
        ax.set_yticks(np.arange(self.neuron_cnt))
        title = f"Population activity - {source}"
        if zscore:
            title += " (z-scored per neuron)"
        ax.set_title(title)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)

        fig.tight_layout()
        return fig

    def plot_autocorrelation(
        self,
        max_lag: int | None = None,
        neuron_indices=None,
        trial_indices=None,
        mode: str = "overlay",
    ):
        """
        Plot the temporal autocorrelation function (ACF) per neuron.

        For each neuron `n` the ACF at lag τ is

            ACF_n(τ) = E_{r,t}[ (y^(r)_{n,t} − μ_n)(y^(r)_{n,t+τ} − μ_n) ]
                       / Var(y_n)

        where μ_n and Var(y_n) are pooled across every (trial, timestep)
        pair. The lag-τ expectation is taken over all (r, t) for which
        both endpoints lie inside the trial window. The estimator is
        biased (1/N normalisation, not 1/(N−1)) — this is the standard
        choice for ACFs and guarantees ACF_n(0) = 1.

        Interpretation: exponential decay → real eigenvalue of A,
        oscillation with sign changes → complex-conjugate pair, slow /
        plateaued decay → eigenvalue near 1, near-zero at lag 1 →
        noise-dominated.

        Parameters
        ----------
        max_lag : int, optional
            Largest lag to compute. Defaults to T // 4 (15 for T=60),
            a conventional choice that captures decay without leaning
            on the lag-T-1 tail, where only a handful of samples survive
            and estimates get noisy. Must satisfy 1 <= max_lag < T.
        neuron_indices : array-like of int, optional
            Which neurons to include. Defaults to all neurons.
        trial_indices : array-like of int, optional
            Which trials to pool over for the ACF estimate. Defaults to
            all trials. Useful for split-half stationarity checks or
            comparing ACFs across condition-grouped trial subsets.
        mode : {"overlay", "subplots", "heatmap"}, default "overlay"
            How to display the (max_lag+1, n_neurons) ACF array.

        Returns
        -------
        matplotlib.figure.Figure
        """
        # --- 1. Resolve and validate neuron / trial selection -------------
        if neuron_indices is None:
            neuron_idx = np.arange(self.neuron_cnt)
        else:
            neuron_idx = np.asarray(neuron_indices, dtype=int)
            if neuron_idx.ndim != 1:
                raise ValueError("neuron_indices must be 1D")
            if neuron_idx.min() < 0 or neuron_idx.max() >= self.neuron_cnt:
                raise ValueError(
                    f"neuron_indices out of range [0, {self.neuron_cnt - 1}]"
                )

        if trial_indices is None:
            trial_idx = np.arange(self.trial_cnt)
        else:
            trial_idx = np.asarray(trial_indices, dtype=int)
            if trial_idx.ndim != 1:
                raise ValueError("trial_indices must be 1D")
            if trial_idx.min() < 0 or trial_idx.max() >= self.trial_cnt:
                raise ValueError(
                    f"trial_indices out of range [0, {self.trial_cnt - 1}]"
                )

        # --- 2. Default and validate max_lag ------------------------------
        T = self.timestep_cnt
        if max_lag is None:
            max_lag = max(1, T // 4)
        if max_lag < 1:
            raise ValueError(f"max_lag must be >= 1; got {max_lag}")
        if max_lag >= T:
            raise ValueError(
                f"max_lag={max_lag} is too large for T={T} timesteps; "
                f"need max_lag < T"
            )
        if max_lag > T // 4:
            # Per-trial sample count at lag τ is T-τ, so estimates near
            # max_lag rest on few pairs and get noisy. T//4 is the
            # conventional safe ceiling and the default this method uses.
            warnings.warn(
                f"max_lag={max_lag} exceeds T//4={T // 4}; the longest "
                f"lag uses only {T - max_lag} pairs per trial and the "
                f"tail of the ACF may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        if mode not in ("overlay", "subplots", "heatmap"):
            raise ValueError(
                f"mode must be 'overlay', 'subplots', or 'heatmap'; "
                f"got {mode!r}"
            )

        # --- 3. Compute the autocorrelation -------------------------------
        # Pool across all trials AND timesteps for μ and Var (see
        # `_acf_pooled` for the full rationale). Biased estimator, NaN
        # for constant-variance neurons, ACF_n(0) = 1.
        data = self.observation[np.ix_(                     # (r_sel, T, n_sel)
            trial_idx, np.arange(self.timestep_cnt), neuron_idx
        )]
        acf = self._acf_pooled(data, max_lag)

        # --- 4. Plot ------------------------------------------------------
        lags = np.arange(max_lag + 1)
        n_sel = len(neuron_idx)
        cmap = plt.get_cmap("tab20")

        if mode == "overlay":
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            for i, n in enumerate(neuron_idx):
                ax.plot(lags, acf[:, i],
                        color=cmap(int(n) % cmap.N),
                        linewidth=1.5, label=f"n{n}")
            ax.axhline(0.0, color="k", linewidth=0.6, linestyle="--")
            ax.set_xlabel("lag τ (timesteps)")
            ax.set_ylabel("autocorrelation")
            ax.set_title(
                f"Autocorrelation per neuron  "
                f"(R={len(trial_idx)}, T={T}, lags 0..{max_lag})"
            )
            ax.legend(
                ncol=max(1, n_sel // 8 + 1),
                fontsize=8, loc="best", frameon=False,
            )

        elif mode == "subplots":
            fig, axes = self._neuron_grid(n_sel, sharey=True)
            nrows, ncols = axes.shape
            for i, n in enumerate(neuron_idx):
                row, col = divmod(i, ncols)
                ax = axes[row, col]
                ax.plot(lags, acf[:, i],
                        color=cmap(int(n) % cmap.N), linewidth=1.5)
                ax.axhline(0.0, color="k", linewidth=0.6, linestyle="--")
                ax.set_title(f"neuron {n}", fontsize=9)
                ax.tick_params(labelsize=8)
            self._blank_unused(axes, n_sel)
            for row in range(nrows):
                axes[row, 0].set_ylabel("ACF", fontsize=9)
            for col in range(ncols):
                for row in range(nrows - 1, -1, -1):
                    if axes[row, col].get_visible():
                        axes[row, col].set_xlabel("lag τ", fontsize=9)
                        break
            fig.suptitle(
                f"Autocorrelation per neuron  "
                f"(R={len(trial_idx)}, T={T}, lags 0..{max_lag})",
                fontsize=11,
            )

        else:  # mode == "heatmap"
            fig, ax = plt.subplots(figsize=(8, 0.3 * n_sel + 1.5))
            # Symmetric diverging scale around 0: positive and negative
            # correlations both carry meaning (decay vs. oscillation).
            vabs = float(np.nanmax(np.abs(acf))) if np.any(np.isfinite(acf)) else 1.0
            vabs = vabs or 1.0
            im = ax.imshow(
                acf.T,                                      # (n_sel, max_lag+1)
                aspect="auto", cmap="RdBu_r",
                vmin=-vabs, vmax=vabs,
                origin="lower", interpolation="nearest",
                extent=[-0.5, max_lag + 0.5, -0.5, n_sel - 0.5],
            )
            ax.set_xlabel("lag τ (timesteps)")
            ax.set_ylabel("neuron")
            ax.set_yticks(np.arange(n_sel))
            ax.set_yticklabels([str(n) for n in neuron_idx])
            ax.set_title(
                f"Autocorrelation heatmap  "
                f"(R={len(trial_idx)}, T={T}, lags 0..{max_lag})"
            )
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("autocorrelation")

        fig.tight_layout()
        return fig

    def compute_snr(self, plot: bool = True) -> np.ndarray:
        """
        Per-neuron signal-to-noise ratio, exploiting the trial axis.

        Decomposes each neuron's variability into (a) variance shared
        across trials at matched timesteps — "signal" — and (b)
        trial-to-trial variability around the trial mean — "noise".
        The ratio quantifies how reproducible the neuron is across
        repeated trials.

        Parameters
        ----------
        plot : bool, default True
            Whether to draw the diagnostic plot. The array is returned
            regardless of this flag.

        Returns
        -------
        np.ndarray of shape (N,)
            Bias-corrected SNR per neuron, in original neuron order.
            Clipped to >= 0 (negative estimates are sampling noise).
            Neurons with zero trial-to-trial variability are returned
            as NaN — technically infinite SNR, i.e. maximally reliable.

        Raises
        ------
        ValueError
            If `trial_cnt == 1` — trial-to-trial variability is
            undefined with a single trial.

        Notes
        -----
        The σ²_noise computed here is the total trial-to-trial
        variability, which in the underlying dynamical system is a
        mixture of observation noise v_t and accumulated process noise
        w_t. It is NOT the observation-noise covariance R from the
        system model. It answers "how reproducible is this neuron
        across repeated trials?" — not "what is the sensor noise floor?".

        The decomposition is valid only when all trials share the same
        input sequence u_t. If trials were collected under different
        inputs, σ²_noise will absorb input-driven variability,
        understating noise and overstating signal.
        """
        if self.trial_cnt < 2:
            raise ValueError(
                "SNR requires at least 2 trials. With a single trial, "
                "trial-to-trial variability is undefined. Use "
                "`plot_timeseries` or `plot_autocorrelation` to explore "
                "the data instead."
            )

        R = self.trial_cnt

        # Step 1: trial-averaged time series per neuron — already cached
        # as self._trial_mean, shape (T, N).

        # Step 2: noise variance per neuron. Across-trial variance at
        # each timestep, then mean over time. ddof=1 for sample variance.
        var_across_trials = self.observation.var(axis=0, ddof=1)  # (T, N)
        sigma2_noise = var_across_trials.mean(axis=0)             # (N,)

        # Step 3: raw signal+noise variance — temporal variance of the
        # trial-averaged trace, ddof=1.
        sigma2_raw = self._trial_mean.var(axis=0, ddof=1)         # (N,)

        # Step 4: bias-correct. The trial mean still carries a noise
        # residual of σ²_noise / R, so subtract it. Without this a
        # pure-noise neuron would have an apparent SNR of 1/R.
        sigma2_signal = sigma2_raw - sigma2_noise / R             # (N,)

        # Step 5: SNR. Guard the divide for neurons with zero noise —
        # those are perfectly reproducible (∞ SNR) and become NaN so
        # the caller can spot them. Clip negative SNR (sampling noise
        # around zero) to 0; NaN passes through clip unchanged.
        zero_noise = sigma2_noise == 0
        safe_noise = np.where(zero_noise, 1.0, sigma2_noise)
        snr = sigma2_signal / safe_noise
        snr = np.clip(snr, a_min=0.0, a_max=None)
        snr[zero_noise] = np.nan

        if plot:
            self._plot_snr(snr)

        return snr

    def _plot_snr(self, snr: np.ndarray) -> None:
        """Render the two-panel SNR diagnostic for `compute_snr`."""
        N = self.neuron_cnt
        R = self.trial_cnt
        T = self.timestep_cnt
        cmap = plt.get_cmap("tab20")

        # Bounded reliability fraction for display only. NaN propagates,
        # which we handle explicitly below.
        r2e = snr / (1.0 + snr)                                 # (N,)

        # Sort descending by r²_e. NaN ("∞ SNR") sorts to the top.
        sort_key = np.where(np.isnan(r2e), np.inf, r2e)
        order = np.argsort(-sort_key)
        r2e_sorted = r2e[order]
        is_nan = np.isnan(r2e_sorted)

        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.5, 4.5))

        # --- Left: bar chart of r²_e --------------------------------
        bar_colors = [cmap(int(n) % cmap.N) for n in order]
        # Draw NaN bars at full height with a hatched pattern, so the
        # neuron is visible rather than a blank slot.
        bar_heights = np.where(is_nan, 1.0, r2e_sorted)
        bars = ax_l.bar(
            np.arange(N), bar_heights,
            color=bar_colors, edgecolor="0.25", linewidth=0.6,
        )
        for i, nan in enumerate(is_nan):
            if nan:
                bars[i].set_hatch("///")
                ax_l.text(
                    i, 1.02, "∞ SNR",
                    ha="center", va="bottom", fontsize=7, rotation=90,
                )
        ax_l.axhline(
            0.5, color="k", linewidth=0.8, linestyle="--",
            label="signal = noise",
        )
        ax_l.set_xticks(np.arange(N))
        ax_l.set_xticklabels([str(n) for n in order], fontsize=8)
        ax_l.set_xlabel("neuron index (sorted)")
        ax_l.set_ylabel(r"explainable variance $r^2_e$")
        ax_l.set_ylim(0, 1.12)
        ax_l.set_title(f"Signal reliability per neuron (R={R} trials)")
        ax_l.legend(loc="upper right", fontsize=8, frameon=False)

        # --- Right: trial mean ± 1 SD for top / bottom neurons ------
        # Only finite-r²_e neurons rank meaningfully; NaN neurons are
        # "off the scale" reliable and would dominate the comparison.
        finite_order = order[~is_nan]
        if len(finite_order) < 6:
            chosen = list(finite_order)
        else:
            chosen = list(finite_order[:3]) + list(finite_order[-3:])

        t = np.arange(T)
        for n in chosen:
            color = cmap(int(n) % cmap.N)
            m = self._trial_mean[:, n]
            s = self._trial_std[:, n]
            ax_r.plot(
                t, m, color=color, linewidth=2.0,
                label=f"n{n}: r²_e={r2e[n]:.2f}",
            )
            ax_r.fill_between(
                t, m - s, m + s,
                color=color, alpha=0.20, linewidth=0,
            )
        ax_r.set_xlabel("timestep")
        ax_r.set_ylabel("activity")
        ax_r.set_title("Top vs bottom reliability — trial mean ± 1 SD")
        ax_r.legend(fontsize=8, loc="best", frameon=False)

        fig.tight_layout()

    def plot_correlation_matrix(
        self,
        trial_index: int | None = None,
        zscore: bool = False,
        cluster: bool = True,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Pairwise Pearson-correlation heatmap of the neural population.

        For each pair of neurons (i, j) the entry is the Pearson
        correlation between their time series across the T timesteps.
        By default the time series is the trial average; pass
        `trial_index` to use a single trial instead. Block structure
        in the displayed matrix is direct evidence of shared
        low-dimensional latent drive.

        Parameters
        ----------
        trial_index : int, optional
            Single trial to use. Defaults to the trial-averaged signal.
        zscore : bool, default False
            Z-score each neuron's time series before computing. This is
            mathematically a no-op — `np.corrcoef` already standardises
            internally — and exists only for API symmetry with
            `plot_heatmap` and to make the standardisation explicit at
            the call site. The returned correlation values are
            identical with or without this flag.
        cluster : bool, default True
            If True, reorder rows/columns using average-linkage
            hierarchical clustering on (1 - C) so co-correlated neurons
            sit next to each other in the display. The returned matrix
            is always in *original* neuron order; only the display
            changes.

        Returns
        -------
        (fig, C) : (matplotlib.figure.Figure, np.ndarray of shape (N, N))
            `C[i, j]` is the Pearson correlation between neuron i and
            neuron j in original neuron order. Rows/columns for
            constant-in-time neurons are NaN.

        Raises
        ------
        ValueError
            If `neuron_cnt < 2`, or if `trial_index` is out of range.

        Notes
        -----
        Correlations are computed on the trial-averaged signal by
        default. The result therefore reflects shared *deterministic*
        structure — what the neurons do together reliably across
        trials. Trial-to-trial noise, which `compute_snr` measures, is
        averaged out before this method sees the data.

        Correlation measures linear co-movement. Two neurons driven by
        the same latent dimension of x_t will appear strongly
        correlated. A neuron uncorrelated with everything is either
        noise-dominated (consistent with a low SNR from `compute_snr`)
        or driven by a latent dimension that no other recorded neuron
        observes.

        Correlation does not imply a direct connection between neurons.
        It reflects shared input from the hidden state x_t through the
        observation matrix C, not a direct neuron-to-neuron coupling.
        """
        # --- 1. Validate ---------------------------------------------------
        N = self.neuron_cnt
        if N < 2:
            raise ValueError(
                "plot_correlation_matrix requires at least 2 neurons; "
                "a 1x1 correlation matrix is always [[1]] and carries "
                "no information."
            )

        if trial_index is None:
            data = self._trial_mean                  # (T, N)
            source = "trial mean"
        else:
            if not (0 <= trial_index < self.trial_cnt):
                raise ValueError(
                    f"trial_index out of range [0, {self.trial_cnt - 1}]; "
                    f"got {trial_index}"
                )
            data = self.observation[trial_index]     # (T, N)
            source = f"trial {trial_index}"

        # --- 2. Optional z-scoring (no-op for the returned C; corrcoef
        #         standardises internally). Kept for API symmetry with
        #         plot_heatmap, and explicit at the call site.
        if zscore:
            temporal_std = data.std(axis=0, ddof=1)
            safe_std = np.where(temporal_std == 0, 1.0, temporal_std)
            data = (data - data.mean(axis=0)) / safe_std

        # --- 3. Correlation matrix + constant-neuron mask -----------------
        C, constant = self._corrcoef_with_nan(data)

        # --- 4. Hierarchical clustering for display order -----------------
        do_cluster = cluster and (~constant).sum() >= 2
        display_order = (
            self._cluster_order(C) if do_cluster else np.arange(N)
        )

        # --- 5. Plot ------------------------------------------------------
        display = C[np.ix_(display_order, display_order)]
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("lightgray")                    # NaN cells distinct
        masked = np.ma.masked_invalid(display)

        side = 0.35 * N + 3.0
        fig, ax = plt.subplots(figsize=(side, side))
        im = ax.imshow(
            masked, cmap=cmap, vmin=-1.0, vmax=1.0,
            origin="upper", interpolation="nearest",
        )

        tick_labels = [
            f"{n}*" if constant[n] else str(n) for n in display_order
        ]
        ax.set_xticks(np.arange(N))
        ax.set_yticks(np.arange(N))
        ax.set_xticklabels(tick_labels, fontsize=8, rotation=90)
        ax.set_yticklabels(tick_labels, fontsize=8)
        ax.set_xlabel("neuron")
        ax.set_ylabel("neuron")

        title = f"Correlation matrix - {source}"
        if do_cluster:
            title += " (clustered)"
        ax.set_title(title)

        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        cbar.set_label("Pearson correlation")

        if constant.any():
            constants_str = ", ".join(
                str(int(n)) for n in np.where(constant)[0]
            )
            fig.text(
                0.5, 0.01,
                f"* constant in time, correlation undefined "
                f"(neurons: {constants_str})",
                ha="center", fontsize=8, color="0.3",
            )
            fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
        else:
            fig.tight_layout()

        return fig, C

    # ------------------------------------------------------------------
    # Shared static helpers
    # ------------------------------------------------------------------
    # Pure utilities used by more than one method. Kept at the bottom
    # of the class so the public-method narrative reads top-to-bottom.

    @staticmethod
    def _neuron_grid(n_panels: int, sharey: bool = False,
                     ncols: int | None = None):
        """
        Build a per-neuron subplot grid sized to taste.

        Defaults to ~square, capped at 4 columns wide. Used by every
        method that draws one subplot per neuron.
        """
        if ncols is None:
            ncols = min(4, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(3.2 * ncols, 2.2 * nrows),
            sharex=True, sharey=sharey, squeeze=False,
        )
        return fig, axes

    @staticmethod
    def _blank_unused(axes, n_panels: int) -> None:
        """Hide the leftover panels when the grid is not exactly filled."""
        for ax in axes.flat[n_panels:]:
            ax.set_visible(False)

    @staticmethod
    def _acf_pooled(obs: np.ndarray, max_lag: int) -> np.ndarray:
        """
        Per-neuron ACF for (R, T, N) data, pooled across all trials.

        Pooled μ and Var per neuron, lag-τ expectation over every valid
        (r, t) pair, biased 1/N normalisation. Constant-in-time neurons
        get NaN. Used by `plot_autocorrelation`.
        """
        mu = obs.mean(axis=(0, 1))
        c = obs - mu
        var = (c * c).mean(axis=(0, 1))
        safe_var = np.where(var == 0, 1.0, var)
        acf = np.empty((max_lag + 1, obs.shape[2]))
        for tau in range(max_lag + 1):
            prod = c * c if tau == 0 else c[:, :-tau, :] * c[:, tau:, :]
            acf[tau] = prod.mean(axis=(0, 1))
        acf /= safe_var
        acf[:, var == 0] = np.nan
        return acf

    @staticmethod
    def _corrcoef_with_nan(
        data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Pearson correlation on a (T, N) time series with the same
        constant-in-time handling as `plot_correlation_matrix`. Returns
        (C, constant_mask). Used by `plot_correlation_matrix`.
        """
        temporal_std = data.std(axis=0)
        scale = np.maximum(np.abs(data).max(axis=0), 1.0)
        constant = temporal_std <= 1e-12 * scale
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.corrcoef(data.T)
        if constant.any():
            C[constant, :] = np.nan
            C[:, constant] = np.nan
        return C, constant

    @staticmethod
    def _cluster_order(C: np.ndarray) -> np.ndarray:
        """
        Average-linkage display order from a Pearson matrix.

        Constants (NaN diagonal) are appended after the clustered group
        so they remain visible. Caller is responsible for checking that
        at least two non-constant neurons exist.
        """
        constant = np.isnan(np.diag(C))
        valid = np.where(~constant)[0]
        sub = C[np.ix_(valid, valid)]
        dist = (1.0 - sub + (1.0 - sub).T) / 2.0
        np.fill_diagonal(dist, 0.0)
        dist = np.clip(dist, 0.0, 2.0)
        Z = linkage(squareform(dist, checks=False), method="average")
        return np.concatenate(
            [valid[leaves_list(Z)], np.where(constant)[0]]
        )