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
from numpy.typing import ArrayLike

# a neuron is treated as constant (zero variance) when its std is below
# this fraction of its peak absolute value. Mirrors the guard used in
# _corrcoef_with_nan and applied consistently across SNR and ACF helpers.
_CONSTANT_TOLERANCE = 1e-12


class Illustrator:
    """
    Inspect, summarise, and visualise a neural-activity dataset.

    The dataset is a 3D numpy array of shape
    (trial_cnt, timestep_cnt, neuron_cnt). Every method operates on
    this convention; axis 0 is always trials, axis 1 is always time,
    axis 2 is always neurons.
    """

    # beyond this many trials, per-trial lines are omitted
    # caller can override via show_trials.
    MAX_TRIALS_OVERLAY = 10

    # beyond this many neurons, the SNR bar chart auto-selects
    # the top and bottom MAX_SNR_BAR_EACH neurons to stay readable.
    MAX_SNR_BAR_EACH = 5
    MAX_SNR_BAR_NEURONS = 20

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

        # store as float64 for numerical stability in later reductions
        self.observation = np.asarray(observation, dtype=np.float64)
        self.trial_cnt, self.timestep_cnt, self.neuron_cnt = self.observation.shape

        # cache once: trial-averaged signal and trial-to-trial std.
        self._trial_mean = self.observation.mean(axis=0)              # (T_steps, N)
        self._trial_std  = self.observation.std(
            axis=0, ddof=min(1, self.trial_cnt - 1),
        )                                                              # (T_steps, N)

    def plot_timeseries(
        self,
        neuron_indices: ArrayLike | None = None,
        trial_indices: ArrayLike | None = None,
        show_trials: bool | None = None,
        show_mean: bool = True,
        show_band: bool = True,
        n_cols: int | None = None,
        title: str | None = None,
    ) -> plt.Figure:
        """
        Plot neural activity over time, one subplot per neuron.

        Each panel shows the time course of one neuron. Individual trials
        are drawn as thin semi-transparent lines; the trial-averaged mean
        is a bold line; a shaded band spans +/-1 standard deviation across
        trials. Panels are arranged in a grid (at most 4 columns wide).

        Parameters
        ----------
        neuron_indices : array-like of int, optional
            Indices of the neurons to plot. Negative indices are supported.
            Defaults to all neurons. Duplicates are silently dropped.
        trial_indices : array-like of int, optional
            Indices of the trials to include. The per-trial lines, the
            mean, and the +/-1 SD band are all computed from this subset
            only. Defaults to all trials.
        show_trials : bool, optional
            Whether to draw the thin per-trial lines. If None (default),
            lines are shown when the number of selected trials is at most
            ``MAX_TRIALS_OVERLAY`` (currently 10) and hidden otherwise,
            to avoid an unreadable spaghetti plot.
        show_mean : bool, default True
            Whether to overlay the trial-averaged mean as a bold line.
        show_band : bool, default True
            Whether to shade the +/-1 SD region around the mean. The band
            is automatically suppressed when fewer than 2 trials are
            selected, because a standard deviation requires at least 2
            observations.
        n_cols : int, optional
            Number of columns in the subplot grid. Defaults to an
            approximately-square layout, capped at 4 columns.
        title : str, optional
            Override or suppress the automatic figure title. Pass a
            string to use a custom title, or ``""`` to remove the title
            entirely. The default (``None``) uses an auto-generated title.

        Returns
        -------
        fig : plt.Figure
            The figure containing all neuron subplots. Pass to
            ``fig.savefig(path)`` to save, or ``plt.show()`` to display.

        Raises
        ------
        ValueError
            If ``neuron_indices`` or ``trial_indices`` is empty, not 1-D,
            or contains indices outside ``[-n, n-1]`` for the relevant
            axis size ``n``.

        Notes
        -----
        The +/-1 SD band captures trial-to-trial variability at each
        timestep, not measurement noise. A wide band means the neuron's
        response changes substantially from trial to trial. For a
        quantitative reliability score that separates signal variance from
        noise variance, see ``compute_snr``.

        The band uses sample standard deviation (ddof=1).

        Examples
        --------
        >>> ill = Illustrator(data)          # data shape: (trials, timesteps, neurons)
        >>> fig = ill.plot_timeseries()      # all neurons, all trials
        >>> fig = ill.plot_timeseries(       # two neurons, first 5 trials only
        ...     neuron_indices=[0, 2],
        ...     trial_indices=range(5),
        ... )
        >>> fig.savefig("timeseries.png", dpi=150)
        """
        # resolve and validate neuron/trial selections
        neuron_idx = self._resolve_indices(neuron_indices, self.neuron_cnt, "neuron_indices")
        trial_idx  = self._resolve_indices(trial_indices,  self.trial_cnt,  "trial_indices")

        n_neurons_plot = len(neuron_idx)
        n_trials_plot  = len(trial_idx)

        # decide whether to show per-trial lines
        if show_trials is None:
            show_trials = n_trials_plot <= self.MAX_TRIALS_OVERLAY

        # slice the data once
        data = self.observation[np.ix_(trial_idx,
                                        np.arange(self.timestep_cnt),
                                        neuron_idx)]       # (n_trials_plot, T_steps, n_neurons_plot)

        # mean and std over selected trials
        mean_t = data.mean(axis=0)                          # (T_steps, n_neurons_plot)
        if n_trials_plot >= 2:
            std_t = data.std(axis=0, ddof=1)                # (T_steps, n_neurons_plot)
        else:
            std_t = None                                    # band suppressed

        # lay out the grid 
        fig, axes = self._neuron_grid(n_neurons_plot, ncols=n_cols)
        _, ncols = axes.shape

        t = np.arange(self.timestep_cnt)                    # x-axis, timestep count

        for panel_i, neuron in enumerate(neuron_idx):
            row, col = divmod(panel_i, ncols)               # gives (0,0), (0,1), ...
            ax = axes[row, col]
            color = self._neuron_color(neuron)

            # thin per-trial lines
            if show_trials:
                for tr in range(n_trials_plot):
                    ax.plot(t, data[tr, :, panel_i],
                            color=color, alpha=0.3, linewidth=0.8)

            # bold trial-mean
            if show_mean:
                ax.plot(t, mean_t[:, panel_i],
                        color=color, linewidth=2.0, label="mean")

            # shaded std band
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

        # blank out any unused panels
        self._blank_unused(axes, n_neurons_plot)

        # --- 5. Shared axis labels and title -------------------------------
        # One y-label on the left column, one x-label on the bottom row.
        self._label_left_column(axes, "activity")
        self._label_bottom_row(axes, "timestep")


        ##review##
        _auto_title = (f"Time series  "
                       f"({n_trials_plot} trial{'s' if n_trials_plot != 1 else ''}, "
                       f"{n_neurons_plot} neuron{'s' if n_neurons_plot != 1 else ''})")
        if n_trials_plot == 1 and not show_band:
            _auto_title += "  [band suppressed: single trial]"
        elif not show_trials and n_trials_plot > self.MAX_TRIALS_OVERLAY:
            _auto_title += f"  [per-trial lines hidden: >{self.MAX_TRIALS_OVERLAY} trials]"
        _show_title = _auto_title if title is None else title
        if _show_title:
            fig.suptitle(_show_title, fontsize=11)

        if _show_title:
            fig.tight_layout(rect=[0, 0, 1, 0.96])
        else:
            fig.tight_layout()
        return fig

    def plot_heatmap(
        self,
        trial_index: int | None = None,
        neuron_indices: ArrayLike | None = None,
        cosine_sort: bool = False,
        zscore: bool = False,
        title: str | None = None,
    ) -> plt.Figure:
        """
        Show population activity as a (Neurons x Time) heatmap.

        Each row is one neuron; each column is one timestep; colour
        encodes activity. By default the trial-averaged signal is
        displayed. Pass ``trial_index`` to inspect a single trial instead.

        Parameters
        ----------
        trial_index : int, optional
            Index of the trial to display. Negative indices are supported.
            If None (default), the trial-averaged signal is shown.
        neuron_indices : array-like of int, optional
            Indices of the neurons to include. Defaults to all neurons.
            Pass the output of ``np.where(snr > threshold)[0]`` from
            ``compute_snr`` to restrict to reliable neurons before
            inspecting the population structure.
        cosine_sort : bool, default False
            If True, reorder rows so that neurons with similar temporal
            patterns are grouped together. The neuron whose time course
            is most similar on average to all others is used as the
            reference, and remaining neurons are sorted by cosine
            similarity to it (descending).
        zscore : bool, default False
            If True, z-score each row independently across time and use a
            diverging colormap centred at 0. Useful when neurons differ
            widely in absolute scale and you want to compare temporal
            dynamics rather than raw activity levels.
        title : str, optional
            Override or suppress the automatic title. Pass a string to
            use a custom title, or ``""`` to remove it. Defaults to an
            auto-generated title.

        Returns
        -------
        fig : plt.Figure
            The figure containing the heatmap. Pass to
            ``fig.savefig(path)`` to save, or ``plt.show()`` to display.

        Raises
        ------
        ValueError
            If ``neuron_indices`` is empty, not 1-D, or contains indices
            outside ``[-n, n-1]`` for the neuron axis size ``n``.
            If ``trial_index`` is outside ``[-R, R-1]`` for the trial
            count ``R``.

        Examples
        --------
        >>> ill = Illustrator(data)          # data shape: (trials, timesteps, neurons)
        >>> fig = ill.plot_heatmap()         # trial-averaged, all neurons
        >>> fig = ill.plot_heatmap(trial_index=0, zscore=True)
        >>> fig = ill.plot_heatmap(neuron_indices=np.where(snr > 0.5)[0],
        ...                        cosine_sort=True)
        """
        # --- 1. Select the (N, T) matrix to display -----------------------
        neuron_idx = self._resolve_indices(neuron_indices, self.neuron_cnt, "neuron_indices")
        data, source = self._select_display_data(trial_index)
        data = data[:, neuron_idx].T  # (N_sel, T)

        # optional cosine similarity sort
        if cosine_sort:
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            normalised_data = data / np.where(norms == 0, 1.0, norms)
            # find the index of the neuron most similar on average to the others
            similarities = normalised_data @ normalised_data.T  # (N_sel, N_sel)
            avg_sim = similarities.mean(axis=1)
            most_similar_index = np.argmax(avg_sim)

            # sort by cosine similarity to the most similar neuron
            sort_order = np.argsort(similarities[most_similar_index])[::-1]
            data = data[sort_order]
            neuron_idx = neuron_idx[sort_order]
          

        # --- 2. Optional per-neuron z-scoring across time -----------------
        if zscore:
            row_mean = data.mean(axis=1, keepdims=True)     # (N_sel, 1)
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
            if vmin == vmax:
                vmin, vmax = vmin - 1.0, vmax + 1.0
            cbar_label = "activity"

        # --- 3. Draw -------------------------------------------------------
        n_sel = len(neuron_idx)
        fig, ax = plt.subplots(figsize=(8, 0.3 * n_sel + 1.5))
        im = ax.imshow(
            display, aspect="auto", cmap=cmap,
            vmin=vmin, vmax=vmax,
            origin="lower", interpolation="nearest",
        )

        ax.set_xlabel("timestep")
        ax.set_ylabel("neuron")
        ax.set_yticks(np.arange(n_sel))
        ax.set_yticklabels([str(n) for n in neuron_idx])
        _auto_title = f"Population activity - {source}"
        if zscore:
            _auto_title += " (z-scored per neuron)"
        _show_title = _auto_title if title is None else title
        if _show_title:
            fig.suptitle(_show_title, fontsize=11)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)

        if _show_title:
            fig.tight_layout(rect=[0, 0, 1, 0.95])
        else:
            fig.tight_layout()
        return fig

    def plot_autocorrelation(
        self,
        max_lag: int | None = None,
        neuron_indices: ArrayLike | None = None,
        trial_indices: ArrayLike | None = None,
        mode: str = "overlay",
        title: str | None = None,
    ) -> plt.Figure:
        """
        Plot the temporal autocorrelation function (ACF) per neuron.

        For each neuron the ACF measures how similar its activity is to
        itself at different time offsets (lags). At lag 0 the value is
        always 1 by definition. The ACF is estimated by pooling all selected 
        trials and timesteps, so including more trials gives a more reliable 
        curve. Neurons with no trial-to-trial variation are shown as NaN.

        Parameters
        ----------
        max_lag : int, optional
            Largest time offset (in timesteps) to compute. Defaults to
            ``T // 4``, a conventional choice that captures the main
            decay while avoiding the long tail where very few sample
            pairs survive and estimates become unreliable. Must satisfy
            ``1 <= max_lag < T``. Values above ``T // 4`` are accepted
            but trigger a ``UserWarning``.
        neuron_indices : array-like of int, optional
            Indices of the neurons to include. Negative indices are
            supported. Defaults to all neurons. Duplicates are silently
            dropped.
        trial_indices : array-like of int, optional
            Indices of the trials to pool over when estimating the ACF.
            Defaults to all trials. Including more trials reduces
            estimation noise.
        mode : {"overlay", "subplots", "heatmap"}, default "overlay"
            How to display the results.

            - ``"overlay"``: all neurons on a single axes, each drawn
              in a distinct colour with a legend. Best for comparing a
              small number of neurons directly.
            - ``"subplots"``: one panel per neuron arranged in a grid,
              matching the layout of ``plot_timeseries``. Best when
              individual detail matters more than cross-neuron
              comparison.
            - ``"heatmap"``: a colour image with lag on the x-axis and
              neuron on the y-axis, using a diverging colourmap centred
              at zero. Best for a population-level overview.
        title : str, optional
            Override or suppress the automatic title. Pass a string to
            use a custom title, or ``""`` to remove it. Defaults to an
            auto-generated title.

        Returns
        -------
        fig : plt.Figure
            The figure containing the ACF plot. Pass to
            ``fig.savefig(path)`` to save, or ``plt.show()`` to display.

        Raises
        ------
        ValueError
            If ``neuron_indices`` or ``trial_indices`` is empty, not
            1-D, or contains indices outside ``[-n, n-1]`` for the
            relevant axis size ``n``.
            If ``max_lag`` is outside ``[1, T-1]``.
            If ``mode`` is not one of ``"overlay"``, ``"subplots"``, or
            ``"heatmap"``.

        Examples
        --------
        >>> ill = Illustrator(data)          # data shape: (trials, timesteps, neurons)
        >>> fig = ill.plot_autocorrelation()                     # all neurons, overlay
        >>> fig = ill.plot_autocorrelation(mode="subplots",      # per-neuron panels
        ...                                neuron_indices=[0, 2])
        >>> fig = ill.plot_autocorrelation(mode="heatmap",       # population overview
        ...                                max_lag=20)
        >>> fig.savefig("acf.png", dpi=150)
        """
        # --- 1. Resolve and validate neuron / trial selection -------------
        neuron_idx = self._resolve_indices(neuron_indices, self.neuron_cnt, "neuron_indices")
        trial_idx  = self._resolve_indices(trial_indices,  self.trial_cnt,  "trial_indices")

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

        if mode == "overlay":
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            for i, n in enumerate(neuron_idx):
                ax.plot(lags, acf[:, i],
                        color=self._neuron_color(n),
                        linewidth=1.5, label=f"n{n}")
            ax.axhline(0.0, color="k", linewidth=0.6, linestyle="--")
            ax.set_xlabel("lag τ (timesteps)")
            ax.set_ylabel("autocorrelation")
            _auto_title = (f"Autocorrelation per neuron  "
                           f"(R={len(trial_idx)}, T={T}, lags 0..{max_lag})")
            _show_title = _auto_title if title is None else title
            if _show_title:
                fig.suptitle(_show_title, fontsize=11)
            ax.legend(
                ncol=max(1, n_sel // 8 + 1),
                fontsize=8, loc="best", frameon=False,
            )

        elif mode == "subplots":
            fig, axes = self._neuron_grid(n_sel, sharey=True)
            _, ncols = axes.shape
            for i, n in enumerate(neuron_idx):
                row, col = divmod(i, ncols)
                ax = axes[row, col]
                ax.plot(lags, acf[:, i],
                        color=self._neuron_color(n), linewidth=1.5)
                ax.axhline(0.0, color="k", linewidth=0.6, linestyle="--")
                ax.set_title(f"neuron {n}", fontsize=9)
                ax.tick_params(labelsize=8)
            self._blank_unused(axes, n_sel)
            self._label_left_column(axes, "ACF")
            self._label_bottom_row(axes, "lag τ")
            _auto_title = (f"Autocorrelation per neuron  "
                           f"(R={len(trial_idx)}, T={T}, lags 0..{max_lag})")
            _show_title = _auto_title if title is None else title
            if _show_title:
                fig.suptitle(_show_title, fontsize=11)

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
            _auto_title = (f"Autocorrelation heatmap  "
                           f"(R={len(trial_idx)}, T={T}, lags 0..{max_lag})")
            _show_title = _auto_title if title is None else title
            if _show_title:
                fig.suptitle(_show_title, fontsize=11)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("autocorrelation")

        if _show_title:
            fig.tight_layout(rect=[0, 0, 1, 0.95])
        else:
            fig.tight_layout()
        return fig

    def compute_snr(self, plot: bool = False, neuron_indices: ArrayLike | None = None) -> tuple[np.ndarray, plt.Figure | None]:
        """
        Compute a per-neuron signal-to-noise ratio (SNR) reliability score.

        The estimator is bias-corrected so that a pure-noise neuron scores
        around 0 rather than a spurious positive value. Optionally
        returns a two-panel diagnostic figure: a bar chart of
        reliability per neuron alongside the trial-averaged time
        courses of the most and least reliable neurons.

        Parameters
        ----------
        plot : bool, default False
            If True, also build and return a diagnostic figure (see
            Returns). If False, only the SNR array is computed.
        neuron_indices : array-like of int, optional
            Restrict the diagnostic bar chart to these neurons.
            Negative indices are supported. Has no effect on the
            returned SNR array, which always covers every neuron, and
            is ignored when ``plot=False``. If left as None and the
            dataset contains more than 20 neurons, the bar chart
            auto-selects the top 5 and bottom 5 neurons by reliability
            to stay readable.

        Returns
        -------
        snr : np.ndarray
            A 1D array of shape (N,) with one SNR value per neuron, in
            the original neuron order. Values are clipped at 0 from
            below — a true zero-signal neuron may otherwise yield a
            small negative estimate due to finite sampling. Neurons
            with no trial-to-trial variability at all are returned as
            ``np.nan``: their response is perfectly reproducible and
            the SNR is effectively infinite.
        fig : plt.Figure or None
            The diagnostic figure if ``plot=True``, otherwise None.
            Pass to ``fig.savefig(path)`` to save, or ``plt.show()`` to
            display. In the bar chart, neurons with ``np.nan`` SNR are
            drawn at height 1 with a hatched pattern.

        Raises
        ------
        ValueError
            If the dataset contains fewer than 2 trials, since
            trial-to-trial noise cannot be estimated from a single
            trial.
            If ``neuron_indices`` is empty, not 1-D, or contains
            indices outside ``[-N, N-1]`` for the neuron axis size
            ``N``.

        Examples
        --------
        >>> ill = Illustrator(data)          # data shape: (trials, timesteps, neurons)
        >>> snr, _ = ill.compute_snr()       # SNR per neuron, no figure
        >>> snr, fig = ill.compute_snr(plot=True)
        >>> fig.savefig("snr.png", dpi=150)
        >>> reliable = np.where(snr > 0.5)[0]                 # filter to reliable neurons
        >>> fig = ill.plot_heatmap(neuron_indices=reliable)   # follow-up inspection
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
        scale = np.maximum(np.abs(self.observation).max(axis=(0, 1)), 1.0)
        zero_noise = np.sqrt(sigma2_noise) <= _CONSTANT_TOLERANCE * scale
        safe_noise = np.where(zero_noise, 1.0, sigma2_noise)
        snr = sigma2_signal / safe_noise
        snr = np.clip(snr, a_min=0.0, a_max=None)
        snr[zero_noise] = np.nan

        if plot:
            bar_idx = (
                self._resolve_indices(neuron_indices, self.neuron_cnt, "neuron_indices")
                if neuron_indices is not None else None
            )
            fig = self._plot_snr(snr, bar_indices=bar_idx)
        else:
            fig = None
        return snr, fig

    def _plot_snr(self, snr: np.ndarray, bar_indices: np.ndarray | None = None) -> plt.Figure:
        """Render the two-panel SNR diagnostic for `compute_snr`."""
        N = self.neuron_cnt
        R = self.trial_cnt
        T = self.timestep_cnt

        # Bounded reliability fraction for display only. NaN propagates,
        # which we handle explicitly below.
        r2e = snr / (1.0 + snr)                                 # (N,)

        # Sort descending by r²_e. NaN ("∞ SNR") sorts to the top.
        sort_key = np.where(np.isnan(r2e), np.inf, r2e)
        order = np.argsort(-sort_key)

        # Determine which neurons appear in the bar chart.
        # Priority: explicit selection → auto top/bottom → all.
        if bar_indices is not None:
            bar_set = set(bar_indices.tolist())
            show_order = np.array([n for n in order if n in bar_set])
            bar_title_suffix = ""
        elif N > self.MAX_SNR_BAR_NEURONS:
            k = self.MAX_SNR_BAR_EACH
            show_order = np.concatenate([order[:k], order[-k:]])
            bar_title_suffix = f"  (top {k} / bottom {k} shown)"
        else:
            show_order = order
            bar_title_suffix = ""

        r2e_show  = r2e[show_order]
        is_nan_show = np.isnan(r2e_show)
        n_bars = len(show_order)

        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.5, 4.5))

        # --- Left: bar chart of r²_e --------------------------------
        bar_colors = [self._neuron_color(n) for n in show_order]
        bar_heights = np.where(is_nan_show, 1.0, r2e_show)
        bars = ax_l.bar(
            np.arange(n_bars), bar_heights,
            color=bar_colors, edgecolor="0.25", linewidth=0.6,
        )
        for i, nan in enumerate(is_nan_show):
            if nan:
                bars[i].set_hatch("///")
                ax_l.text(
                    i, 1.02, "∞ SNR",
                    ha="center", va="bottom", fontsize=7, rotation=90,
                )
        # Visual separator between the top and bottom groups when auto-selected.
        if bar_title_suffix:
            k = self.MAX_SNR_BAR_EACH
            ax_l.axvline(k - 0.5, color="0.5", linewidth=0.8, linestyle=":")
        ax_l.axhline(
            0.5, color="k", linewidth=0.8, linestyle="--",
            label="signal = noise",
        )
        ax_l.set_xticks(np.arange(n_bars))
        ax_l.set_xticklabels([str(n) for n in show_order], fontsize=8)
        ax_l.set_xlabel("neuron index (sorted by reliability)")
        ax_l.set_ylabel(r"explainable variance $r^2_e$")
        ax_l.set_ylim(0, 1.12)
        ax_l.set_title(f"Signal reliability per neuron (R={R} trials){bar_title_suffix}")
        ax_l.legend(loc="upper right", fontsize=8, frameon=False)

        # --- Right: trial mean ± 1 SD for top / bottom neurons ------
        # Only finite-r²_e neurons rank meaningfully; NaN neurons are
        # "off the scale" reliable and would dominate the comparison.
        finite_order = order[~np.isnan(r2e[order])]
        if len(finite_order) < 4:
            chosen = list(finite_order)
        else:
            chosen = list(finite_order[:2]) + list(finite_order[-2:])

        t = np.arange(T)
        for n in chosen:
            color = self._neuron_color(n)
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
        if chosen:
            ax_r.legend(fontsize=8, loc="best", frameon=False)

        fig.tight_layout()
        return fig

    def plot_correlation_matrix(
        self,
        trial_index: int | None = None,
        neuron_indices: ArrayLike | None = None,
        cluster: bool = True,
        title: str | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Show pairwise Pearson correlations as a (Neurons x Neurons) heatmap.

        Each cell (i, j) is the correlation between two neurons' time
        courses across timesteps. By default the trial-averaged signal
        is used; pass ``trial_index`` to use a single trial instead.
        Block structure along the diagonal flags groups of neurons that
        co-vary over time.

        Parameters
        ----------
        trial_index : int, optional
            Index of the trial to use. Negative indices are supported.
            If None (default), the trial-averaged signal is used.
        neuron_indices : array-like of int, optional
            Indices of the neurons to include. Defaults to all neurons.
            Pass the output of ``np.where(snr > threshold)[0]`` from
            ``compute_snr`` to restrict to reliable neurons before
            inspecting the correlation structure.
        cluster : bool, default True
            If True, reorder rows/columns by average-linkage
            hierarchical clustering so co-varying neurons sit next to
            each other. Only the display order changes; the returned
            matrix is always in the original neuron order.
        title : str, optional
            Override or suppress the automatic title. Pass a string to
            use a custom title, or ``""`` to remove it. Defaults to an
            auto-generated title.

        Returns
        -------
        C : np.ndarray
            A (N_sel, N_sel) matrix of Pearson correlations between the
            selected neurons, in the order given by ``neuron_indices``.
            Rows and columns for neurons whose activity is constant in
            time are NaN (correlation undefined).
        fig : plt.Figure
            The figure containing the correlation heatmap. Pass to
            ``fig.savefig(path)`` to save, or ``plt.show()`` to display.
            Constant-in-time neurons appear as gray rows/columns and
            are listed in a caption below the figure.

        Raises
        ------
        ValueError
            If fewer than 2 neurons are selected, since a 1x1
            correlation matrix carries no information.
            If ``neuron_indices`` is empty, not 1-D, or contains
            indices outside ``[-N, N-1]`` for the neuron axis size
            ``N``.
            If ``trial_index`` is outside ``[-R, R-1]`` for the trial
            count ``R``.

        Examples
        --------
        >>> ill = Illustrator(data)          # data shape: (trials, timesteps, neurons)
        >>> C, fig = ill.plot_correlation_matrix()                   # all neurons, clustered
        >>> C, fig = ill.plot_correlation_matrix(trial_index=0,      # single-trial, no clustering
        ...                                      cluster=False)
        >>> reliable = np.where(snr > 0.5)[0]
        >>> C, fig = ill.plot_correlation_matrix(neuron_indices=reliable)
        """
        neuron_idx = self._resolve_indices(
            neuron_indices, self.neuron_cnt, "neuron_indices"
        )
        # --- 1. Validate ---------------------------------------------------
        N = len(neuron_idx)
        if N < 2:
            raise ValueError(
                "plot_correlation_matrix requires at least 2 neurons; "
                "a 1x1 correlation matrix is always [[1]] and carries "
                "no information."
            )

        data, source = self._select_display_data(trial_index)
        data = data[:, neuron_idx]   # (T, N_sel)

        # --- 2. Correlation matrix + constant-neuron mask -----------------
        C, constant = self._corrcoef_with_nan(data)

        # --- 3. Hierarchical clustering for display order -----------------
        do_cluster = cluster and (~constant).sum() >= 2
        display_order = (
            self._cluster_order(C) if do_cluster else np.arange(N)
        )

        # --- 4. Plot ------------------------------------------------------
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

        _auto_title = f"Correlation matrix - {source}"
        if do_cluster:
            _auto_title += " (clustered)"
        _show_title = _auto_title if title is None else title
        if _show_title:
            fig.suptitle(_show_title, fontsize=11)

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
            top = 0.95 if _show_title else 1.0
            fig.tight_layout(rect=(0.0, 0.04, 1.0, top))
        else:
            if _show_title:
                fig.tight_layout(rect=[0, 0, 1, 0.95])
            else:
                fig.tight_layout()

        return C, fig

    # pure shared static helpers kept separate for clarity
    
    @staticmethod
    def _resolve_indices(indices, axis_size, name):
        if indices is None:
            return np.arange(axis_size)
        arr = np.asarray(indices, dtype=int)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1D")
        if arr.size == 0:
            raise ValueError(f"{name} must not be empty")
        if arr.min() < -axis_size or arr.max() >= axis_size:
            raise ValueError(
                f"{name} out of range [{-axis_size}, {axis_size - 1}]"
            )
        arr = np.where(arr < 0, arr + axis_size, arr)
        _, keep = np.unique(arr, return_index=True) # preserve order, drop duplicates
        return arr[np.sort(keep)] # unique values in original order

    def _select_display_data(self, trial_index):
        """
        Return ((T, N) array, source label) for either the trial mean
        or a single trial. Validates `trial_index` if provided.
        """
        if trial_index is None:
            return self._trial_mean, "trial mean"
        if not (-self.trial_cnt <= trial_index < self.trial_cnt):
            raise ValueError(
                f"trial_index out of range [{-self.trial_cnt}, "
                f"{self.trial_cnt - 1}]; got {trial_index}"
            )
        if trial_index < 0:
            trial_index += self.trial_cnt
        return self.observation[trial_index], f"trial {trial_index}"

    @staticmethod
    def _label_left_column(axes, label):
        for row in range(axes.shape[0]):
            axes[row, 0].set_ylabel(label, fontsize=9)

    @staticmethod
    def _label_bottom_row(axes, label):
        nrows, ncols = axes.shape
        for col in range(ncols):
            for row in range(nrows - 1, -1, -1):
                if axes[row, col].get_visible():
                    axes[row, col].set_xlabel(label, fontsize=9)
                    break

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

    _PALETTE = plt.get_cmap("tab20")

    @classmethod
    def _neuron_color(cls, n):
        """Return a stable tab20 colour for neuron index `n`."""
        return cls._PALETTE(int(n) % cls._PALETTE.N)

    @staticmethod
    def _acf_pooled(obs: np.ndarray, max_lag: int) -> np.ndarray:
        """
        Per-neuron ACF for (R, T, N) data, pooled across all trials.

        Pooled μ and Var per neuron, lag-τ expectation over every valid
        (r, t) pair, biased 1/N normalisation. Constant-in-time neurons
        get NaN. Used by `plot_autocorrelation`.
        """
        R, T = obs.shape[:2]
        mu = obs.mean(axis=(0, 1))
        c = obs - mu
        var = (c * c).mean(axis=(0, 1))
        scale = np.maximum(np.abs(obs).max(axis=(0, 1)), 1.0)
        constant = np.sqrt(var) <= _CONSTANT_TOLERANCE * scale
        safe_var = np.where(constant, 1.0, var)
        acf = np.empty((max_lag + 1, obs.shape[2]))
        for tau in range(max_lag + 1):
            prod = c * c if tau == 0 else c[:, :-tau, :] * c[:, tau:, :]
            acf[tau] = prod.sum(axis=(0, 1)) / (R * T)
        acf /= safe_var
        acf[:, constant] = np.nan
        return acf

    @staticmethod
    def _corrcoef_with_nan(
        data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Pearson correlation on a (T, N) time series. Neurons whose
        temporal std is effectively zero (≤ 1e-12 × peak absolute
        value) are flagged as constant; their rows and columns of C
        are masked with NaN. Returns (C, constant_mask).
        Used by `plot_correlation_matrix`.
        """
        temporal_std = data.std(axis=0)
        scale = np.maximum(np.abs(data).max(axis=0), 1.0)
        constant = temporal_std <= _CONSTANT_TOLERANCE * scale
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