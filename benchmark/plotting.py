"""Plotting module."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from config import get_config


_CFG = get_config()


def _set_nice_log_xticks(ax):
    """Set the x-tickts locator to be adapted to a log scale."""
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0,), numticks=12))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))


def plot_stats_evol(
    results: Iterable[Iterable[Tuple[str, str, str, np.ndarray]]],
    *,
    axis=None,
    xlogscale=False,
    ylogscale=False,
    nice_xlogscale=True,
    ytwinlogscale=False,
    main_axis=None,
    secondary_axis=None,
    main_axis_ref_line=None,
    secondary_axis_ref_line=None,
    sharex=False,
    sharey=False,
    uncertainty=True,
    figsize=(4.0, 4.0),
    title="",
    xlabel="",
    lw=_CFG.LW,
    alpha=_CFG.ALPHA,
    fontsize=_CFG.FONTSIZE,
    ncols_legend=1,
    legend_y_pad=1.3,
    figures_dir=_CFG.FIGURES_DIR,
    filename="figure",
    filename_suffix="",
    verbose=False,
    ignore_nan=True,
    start=0,
    ylim=None,
    ylim_secondary=None,
) -> str | Tuple[plt.Figure, plt.Axes]:
    """Plot the evolution of statistics, with an optional secondary y-axis.

    Parameters
    ----------
    results
        Iterable over statistics; each stat is itself an iterable of trials,
        where each trial is a tuple (name, color, linestyle, values).
    axis
        Optional axis to plot onto.
    xlogscale, ylogscale, ytwinlogscale
        Whether to apply log scaling on axes.
    nice_xlogscale
        Option to enable 'nice' x-axis log-scale.
    main_axis
        Names to be plotted on the main (left) y-axis.
    secondary_axis
        Names to be plotted on the secondary (right) y-axis.
    main_axis_ref_line, secondary_axis_ref_line
        Optional horizontal reference lines for primary/secondary axes.
    uncertainty
        If True, display the standard deviation in transparence.
    sharex, sharey
        Option to share the main axis scale.
    figsize
        Matplotlib figure size.
    title
        Title name.
    xlabel
        X-axis label.
    lw, alpha, fontsize
        Line width, alpha, and font size for plots.
    ncols_legend
        Number of columns in the legend.
    legend_y_pad
        Vertical padding for the legend.
    figures_dir, filename, filename_suffix
        Base directory, filename and optional suffix for the saved PDF.
    verbose
        If True, print info messages.
    ignore_nan
        If True, use nan-safe stats and warn once if NaNs are present.
    start
        Starting index for plotting (to skip initial points if desired).
    ylim
        (ymin, ymax) for the main (left) y-axis. If `ylogscale = True`,
        both bounds must be > 0.
    ylim_secondary
        (ymin, ymax) for the secondary (right) y-axis (if present).
        If `ytwinlogscale = True`, both bounds must be > 0.

    Returns
    -------
    str | (plt.Figure, plt.Axes)
        The filepath of the saved figure if a new figure was created,
        otherwise the (figure, axis) tuple when plotting onto an existing axis.
    """
    # Set up
    main_names = set(main_axis or [])
    sec_names = set(secondary_axis or [])

    if axis is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True, sharex=sharex, sharey=sharey)
        embedded_plot = False
    else:
        ax = axis
        fig = axis.figure
        embedded_plot = True

    ax_twin = ax.twinx() if sec_names else None
    warned_nan = False

    # Plot curves
    for result in zip(*results):
        name, color, linestyle, tt, test_value = zip(*result)

        tt = np.asarray(tt[0]).ravel()
        vals = np.asarray(test_value)
        if vals.ndim == 1:
            vals = vals[None, :]

        if ignore_nan and np.isnan(vals).any():
            if verbose and not warned_nan:
                print("[WARN] NaNs detected in values; using nanmean/nanstd.")
                warned_nan = True
            mean_stats = np.nanmean(vals, axis=0)
            if uncertainty:
                std_stats = np.nanstd(vals, axis=0)
        else:
            mean_stats = np.mean(vals, axis=0)
            if uncertainty:
                std_stats = np.std(vals, axis=0)

        if name[0] in main_names:
            ax_use = ax
        elif ax_twin is not None and name[0] in sec_names:
            ax_use = ax_twin
        else:
            if verbose:
                print(f"[INFO] '{name[0]}' not displayed (not in main/secondary_axis).")
            ax_use = None

        if ax_use is not None:
            ax_use.plot(
                tt[start:],
                mean_stats[start:],
                lw=lw,
                color=color[0],
                linestyle=linestyle[0],
                label=name[0],
                alpha=alpha,
            )
            if uncertainty:
                ax_use.fill_between(
                    tt[start:],
                    mean_stats[start:] - std_stats[start:],
                    mean_stats[start:] + std_stats[start:],
                    color=color[0],
                    alpha=alpha / 3,
                )

    # Reference lines
    if main_axis_ref_line is not None:
        ax.axhline(main_axis_ref_line, lw=lw, linestyle="dashed", color="black", alpha=alpha / 2)

    if secondary_axis_ref_line is not None and ax_twin is not None:
        ax_twin.axhline(
            secondary_axis_ref_line,
            lw=lw,
            linestyle="dashed",
            color="black",
            alpha=alpha / 2,
        )

    # Scaling
    if xlogscale:
        ax.set_xscale("log", base=10)
        if nice_xlogscale:
            _set_nice_log_xticks(ax)

    if ylogscale:
        ax.set_yscale("log", base=10)

    if ytwinlogscale and ax_twin is not None:
        ax_twin.set_yscale("log", base=10)

    # Apply y-limits (after plotting & scaling)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ylim_secondary is not None and ax_twin is not None:
        ax_twin.set_ylim(*ylim_secondary)

    # Ticks and labels
    tick_fs = int(0.9 * fontsize)
    ax.tick_params(axis="x", which="both", labelsize=tick_fs)
    ax.tick_params(axis="y", which="both", labelsize=tick_fs)
    if ax_twin is not None:
        ax_twin.tick_params(axis="y", which="both", labelsize=tick_fs)

    if title:
        ax.set_title(title, fontsize=fontsize)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if ax_twin is not None:
        h2, l2 = ax_twin.get_legend_handles_labels()
        handles += h2
        labels += l2

    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, lab in zip(handles, labels):
        if lab != "_nolegend_" and lab not in seen:
            uniq_handles.append(h)
            uniq_labels.append(lab)
            seen.add(lab)

    if uniq_handles:
        legend = fig.legend(
            uniq_handles,
            uniq_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y_pad),
            ncol=ncols_legend,
            frameon=False,
            fontsize=tick_fs,
        )
        legend.set_in_layout(True)
        for legline in legend.get_lines():
            legline.set_linewidth(2 * lw)

    # Saving
    if not embedded_plot:
        figures_dir = Path(figures_dir)
        suffix = f"_{filename_suffix}" if filename_suffix else ""
        out_path = figures_dir / f"{filename}{suffix}.pdf"

        if verbose:
            print(f"[INFO] Saving figure to '{str(out_path)}'")
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")

        return str(out_path)

    if verbose:
        print("[INFO] Added plots to provided axis.")
    return ax.figure, ax


plot_stats_evolution = plot_stats_evol
