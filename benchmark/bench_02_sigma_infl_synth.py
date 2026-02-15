"""Collaboration criteria illustration."""
from __future__ import annotations
import argparse
import time
from joblib import Parallel, delayed
from dyntr.utils import format_duration, set_random_state
from experiments import run_one_sigma_infl, get_config
from plotting import plot_stats_evol


_CFG = get_config()
N_JOBS = _CFG.N_JOBS
N_RUNS = _CFG.N_RUNS
FONTSIZE = _CFG.FONTSIZE
LW = _CFG.LW
ALPHA = _CFG.ALPHA
MARKERSIZE = _CFG.MARKERSIZE
LCB_ALPHA = _CFG.MODEL_ALPHA
FIGURES_DIR = _CFG.FIGURES_DIR
SEED = _CFG.SEED
LBDA_DEFAULT = _CFG.LBDA_DEFAULT
D = _CFG.D
L_SIGMA_S = _CFG.L_SIGMA_S
JOBLIB_VERBOSE = _CFG.JOBLIB_VERBOSE

t0 = time.time()


###############################################################################
# Settings
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=float, default=0.1, help="L2-norm approximation error.")
    args = parser.parse_args()

    set_random_state(seed=SEED)

###############################################################################
# Main
    print("[INFO] Running optimal sample sharing experiment...")
    kwargs = dict(d=D, eps=args.eps, l_sigma_s=L_SIGMA_S, lbda_s=LBDA_DEFAULT, alpha=LCB_ALPHA)
    delayed_runs = [delayed(run_one_sigma_infl)(**kwargs) for _ in range(N_RUNS)]

    results = Parallel(verbose=JOBLIB_VERBOSE, n_jobs=N_JOBS)(delayed_runs)

###############################################################################
# Plotting
    main_axis = [r"$\Delta^{\star}$"]
    secondary_axis = [r"$\hat{n}$"]
    uncertainty = True
    title = r"$\varepsilon = $" + f"{args.eps:.1f}" + r"$ ~, ~~ \sigma_T = $" + f"{1.0:.1f}"
    plot_stats_evol(
        results=results,
        xlogscale=True,
        ylogscale=False,
        nice_ticks=True,
        main_axis=main_axis,
        secondary_axis=secondary_axis,
        main_axis_ref_line=0.0,
        figsize=(3.5, 3.6),
        uncertainty=uncertainty,
        title=title,
        xlabel=r"$\sigma_S$",
        lw=LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        markersize=MARKERSIZE,
        ncols_legend=2,
        legend_y_pad=1.25,
        figures_dir=FIGURES_DIR,
        filename="02_sigma_influence",
        filename_suffix=f"_eps_{args.eps:.1f}",
        verbose=True,
    )

###############################################################################
# Timing
    print(f"[INFO] Experiment duration: {format_duration(time.time() - t0)}.")
