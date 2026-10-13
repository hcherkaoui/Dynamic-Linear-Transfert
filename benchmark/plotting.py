"""Plotting module."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple
import warnings
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
if __package__:
    from .experiments import get_config
else:
    from experiments import get_config

PRETTY_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 16,
    "axes.labelsize": 13,
    "axes.titlesize": 17,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 13,
    "figure.titlesize": 17,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "lines.linewidth": 2.2,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "figure.dpi": 160,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "savefig.transparent": False,
}
plt.rcParams.update(PRETTY_RCPARAMS)

_CFG = get_config()


def _set_nice_ticks(ax, *, xscale=None, yscale=None, nbins=6, y_decimals=3):
    """Apply readable tick locators/formatters for linear or log axes.
    Uses short decimals and clean $10^k$ labels, avoiding verbose scientific notation.
    """
    xscale = ax.get_xscale() if xscale is None else xscale
    yscale = ax.get_yscale() if yscale is None else yscale

    def _fmt_plain(v, _):
        if not np.isfinite(v):
            return ""
        if abs(v) < 1e-12:
            return "0"
        return f"{v:.{y_decimals}f}".rstrip("0").rstrip(".")

    def _fmt_pow10(v, _):
        if not np.isfinite(v) or v <= 0.0:
            return ""
        e = int(np.round(np.log10(v)))
        p = 10.0 ** e
        if abs(v - p) <= 1e-10 * max(1.0, abs(v)):
            return "1" if e == 0 else rf"$10^{{{e}}}$"
        return f"{v:.{y_decimals}f}".rstrip("0").rstrip(".")

    def _set_linear_axis(axis):
        axis.set_major_locator(ticker.MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]))
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_major_formatter(ticker.FuncFormatter(_fmt_plain))

    def _set_log_axis(axis, lim0, lim1):
        axis.set_minor_locator(ticker.NullLocator())
        if lim0 > 0.0 and lim1 > 0.0 and (lim1 / lim0) < 10.0:
            loc = ticker.MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10])
            ticks = loc.tick_values(lim0, lim1)
            ticks = ticks[(ticks >= min(lim0, lim1)) & (ticks <= max(lim0, lim1)) & (ticks > 0.0)]
            if ticks.size >= 2:
                axis.set_major_locator(ticker.FixedLocator(ticks))
            else:
                axis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0,), numticks=12))
            axis.set_major_formatter(ticker.FuncFormatter(_fmt_plain))
        else:
            axis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0,), numticks=12))
            axis.set_major_formatter(ticker.FuncFormatter(_fmt_pow10))

    if xscale == "log":
        x0, x1 = ax.get_xlim()
        _set_log_axis(ax.xaxis, x0, x1)
    else:
        _set_linear_axis(ax.xaxis)

    if yscale == "log":
        y0, y1 = ax.get_ylim()
        _set_log_axis(ax.yaxis, y0, y1)
    else:
        _set_linear_axis(ax.yaxis)

    if ax.get_xticks().size < 2:
        x0, x1 = ax.get_xlim()
        ax.xaxis.set_major_locator(ticker.FixedLocator([x0, x1]))

    if ax.get_yticks().size < 2:
        y0, y1 = ax.get_ylim()
        ax.yaxis.set_major_locator(ticker.FixedLocator([y0, y1]))


def plot_stats_evol(
    results: Iterable[Iterable[Tuple[str, str, str, np.ndarray]]],
    *,
    axis=None,
    xlogscale=False,
    ylogscale=False,
    nice_ticks=True,
    ytwinlogscale=False,
    main_axis=None,
    secondary_axis=None,
    main_axis_ref_line=None,
    secondary_axis_ref_line=None,
    sharex=False,
    sharey=False,
    uncertainty=True,
    uncertainty_style="band",
    figsize=(4.0, 4.0),
    title="",
    xlabel="",
    ylabel="",
    lw=_CFG.LW,
    alpha=_CFG.ALPHA,
    fontsize=_CFG.FONTSIZE,
    markersize=_CFG.MARKERSIZE,
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
    ylabel
        Y-axis label.
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

    if ax_twin is not None:
        ax_twin.spines["right"].set_visible(True)
        ax_twin.spines["left"].set_visible(False)
        ax_twin.yaxis.set_ticks_position("right")
        ax_twin.yaxis.set_label_position("right")
        ax_twin.tick_params(axis="y", which="both", right=True, labelright=True)
        ax_twin.patch.set_visible(False)

    # Plot curves
    for result in zip(*results):
        if isinstance(result, dict):
            result_list = [result]
        else:
            result_list = list(result)

        meta = result_list[0]
        name = meta.get("name", "Unknown")
        color = meta.get("color", "gray")
        linestyle = meta.get("style", "-")
        markerstyle = meta.get("marker", "")
        tt = np.asarray(meta.get("n_samples", [])).ravel()

        metric_vals = [r.get("metric", []) for r in result_list]
        vals = np.asarray(metric_vals)
        if vals.ndim == 1:
            vals = vals[None, :]

        if ignore_nan and np.isnan(vals).any():
            if verbose and not warned_nan:
                print("[WARN] NaNs detected in values; using nanmean/nanstd.")
                warned_nan = True
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_stats = np.nanmean(vals, axis=0)
                if uncertainty:
                    std_stats = np.nanstd(vals, axis=0)
        else:
            mean_stats = np.mean(vals, axis=0)
            if uncertainty:
                std_stats = np.std(vals, axis=0)

        if name in main_names:
            ax_use = ax
        elif ax_twin is not None and name in sec_names:
            ax_use = ax_twin
        else:
            if verbose:
                print(f"[INFO] '{name}' not displayed (not in main/secondary_axis).")
            ax_use = None

        if ax_use is not None:
            ax_use.plot(
                tt[start:],
                mean_stats[start:],
                lw=lw,
                color=color,
                linestyle=linestyle,
                marker=markerstyle if markerstyle else "",
                markersize=markersize,
                label=name,
                alpha=alpha,
            )
            if uncertainty:
                if uncertainty_style == "errorbar":
                    ax_use.errorbar(
                        tt[start:],
                        mean_stats[start:],
                        yerr=std_stats[start:],
                        fmt="-",
                        capsize=3,
                        color=color,
                        alpha=alpha,
                    )
                else:
                    ax_use.fill_between(
                        tt[start:],
                        mean_stats[start:] - std_stats[start:],
                        mean_stats[start:] + std_stats[start:],
                        color=color,
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

    if ylogscale:
        ax.set_yscale("log", base=10)

    if ytwinlogscale and ax_twin is not None:
        ax_twin.set_yscale("log", base=10)

    # Apply y-limits (after plotting & scaling)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ylim_secondary is not None and ax_twin is not None:
        ax_twin.set_ylim(*ylim_secondary)

    if nice_ticks:
        _set_nice_ticks(ax, xscale=ax.get_xscale(), yscale=ax.get_yscale())
        if ax_twin is not None:
            _set_nice_ticks(ax_twin, xscale=ax.get_xscale(), yscale=ax_twin.get_yscale())

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
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)

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
        figures_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{filename_suffix}" if filename_suffix else ""
        out_path = figures_dir / f"{filename}{suffix}.pdf"

        if verbose:
            print(f"[INFO] Saving figure to '{str(out_path)}'")
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

        return str(out_path)

    if verbose:
        print("[INFO] Added plots to provided axis.")
    return ax.figure, ax


plot_stats_evolution = plot_stats_evol
