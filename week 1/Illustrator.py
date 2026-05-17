"""
Illustrator: exploration and visualisation of neural activity data
with shape (Trials, Timepoints, Neurons).

Works on both real experimental data and Simulator output.
"""

import numpy as np
import matplotlib.pyplot as plt


class Illustrator:
    """
    Inspect, summarise, and visualise a neural-activity dataset.

    The dataset is a 3D numpy array of shape
    (trial_cnt, timestep_cnt, neuron_cnt). Every method operates on
    this convention; axis 0 is always trials, axis 1 is always time,
    axis 2 is always neurons.
    """

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
        # With trial_cnt == 1, _trial_std is identically zero — methods
        # that depend on across-trial variability must guard against this.
        self._trial_mean = self.observation.mean(axis=0)              # (T_steps, N)
        self._trial_std  = self.observation.std(axis=0, ddof=1)       # (T_steps, N)
    
    # Beyond this many trials, per-trial lines turn into spaghetti and
    # we default to mean + band only. Caller can override via show_trials.
    MAX_TRIALS_OVERLAY = 10

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
        if ncols is None:
            # Aim for ~square, cap at 4 columns wide.
            ncols = min(4, n_neurons_plot)
        nrows = int(np.ceil(n_neurons_plot / ncols))

        # Scale figure size with grid, but keep individual panels readable.
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(3.2 * ncols, 2.2 * nrows),
            sharex=True,
            squeeze=False,                                  # always 2D, simplifies indexing
        )

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
        for panel_i in range(n_neurons_plot, nrows * ncols):
            row, col = divmod(panel_i, ncols)
            axes[row, col].set_visible(False)

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

        if mode not in ("overlay", "subplots", "heatmap"):
            raise ValueError(
                f"mode must be 'overlay', 'subplots', or 'heatmap'; "
                f"got {mode!r}"
            )

        # --- 3. Compute the autocorrelation -------------------------------
        # Pool across all trials AND timesteps for μ and Var: the ACF is a
        # within-trial estimator that assumes stationarity, so a single
        # global mean per neuron is the right reference. Each lag-τ
        # estimate averages over R*(T-τ) pairs, so the tail necessarily
        # uses fewer samples than the head — biased estimator is standard.
        data = self.observation[np.ix_(                     # (r_sel, T, n_sel)
            trial_idx, np.arange(self.timestep_cnt), neuron_idx
        )]
        mu  = data.mean(axis=(0, 1))                        # (n_sel,)
        c   = data - mu                                     # (R, T, n_sel)
        var = (c * c).mean(axis=(0, 1))                     # (n_sel,)

        # Constant / silent neurons: avoid 0/0, mark their ACF NaN so the
        # caller sees they were skipped rather than a fake flat curve.
        safe_var = np.where(var == 0, 1.0, var)

        acf = np.empty((max_lag + 1, len(neuron_idx)))
        for tau in range(max_lag + 1):
            if tau == 0:
                prod = c * c
            else:
                prod = c[:, :-tau, :] * c[:, tau:, :]       # (R, T-τ, n_sel)
            acf[tau] = prod.mean(axis=(0, 1))
        acf /= safe_var
        acf[:, var == 0] = np.nan

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
            ncols = min(4, n_sel)
            nrows = int(np.ceil(n_sel / ncols))
            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(3.2 * ncols, 2.2 * nrows),
                sharex=True, sharey=True, squeeze=False,
            )
            for i, n in enumerate(neuron_idx):
                row, col = divmod(i, ncols)
                ax = axes[row, col]
                ax.plot(lags, acf[:, i],
                        color=cmap(int(n) % cmap.N), linewidth=1.5)
                ax.axhline(0.0, color="k", linewidth=0.6, linestyle="--")
                ax.set_title(f"neuron {n}", fontsize=9)
                ax.tick_params(labelsize=8)
            for i in range(n_sel, nrows * ncols):
                row, col = divmod(i, ncols)
                axes[row, col].set_visible(False)
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