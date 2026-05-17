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


    def summary(self) -> dict:
        """
        Print and return basic statistics of the dataset.

        Returns
        -------
        dict with keys:
            'shape'            : tuple (trials, timesteps, neurons)
            'global_mean'      : float, mean over all entries
            'global_std'       : float, std over all entries
            'global_min'       : float
            'global_max'       : float
            'per_neuron_mean'  : np.ndarray, shape (neurons,)
            'per_neuron_std'   : np.ndarray, shape (neurons,)
            'per_neuron_min'   : np.ndarray, shape (neurons,)
            'per_neuron_max'   : np.ndarray, shape (neurons,)

        Notes
        -----
        Per-neuron stats pool over both trials and time. The
        decomposition into trial-to-trial vs within-trial variability
        belongs to `compute_snr`.
        """
        obs = self.observation                          # (T_trials, T_steps, N)

        # Reduce over trials AND time → one value per neuron.
        per_neuron_mean = obs.mean(axis=(0, 1))         # (N,)
        per_neuron_std  = obs.std(axis=(0, 1), ddof = 1)          # (N,)
        per_neuron_min  = obs.min(axis=(0, 1))          # (N,)
        per_neuron_max  = obs.max(axis=(0, 1))          # (N,)

        result = {
            "shape": (self.trial_cnt, self.timestep_cnt, self.neuron_cnt),
            "global_mean": float(obs.mean()),
            "global_std":  float(obs.std(ddof=1)),
            "global_min":  float(obs.min()),
            "global_max":  float(obs.max()),
            "per_neuron_mean": per_neuron_mean,
            "per_neuron_std":  per_neuron_std,
            "per_neuron_min":  per_neuron_min,
            "per_neuron_max":  per_neuron_max,
        }

        # Compact human-readable print. Full per-neuron arrays are in
        # the returned dict for programmatic use.
        print(f"Dataset: {self.trial_cnt} trials x "
              f"{self.timestep_cnt} timesteps x {self.neuron_cnt} neurons")
        print(f"  global    mean={result['global_mean']:+.4f}  "
              f"std={result['global_std']:.4f}  "
              f"min={result['global_min']:+.4f}  "
              f"max={result['global_max']:+.4f}")
        print(f"  per-neuron range across {self.neuron_cnt} neurons:")
        print(f"    mean in [{per_neuron_mean.min():+.4f}, "
              f"{per_neuron_mean.max():+.4f}]")
        print(f"    std  in [{per_neuron_std.min():.4f}, "
              f"{per_neuron_std.max():.4f}]")

        return result
    
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