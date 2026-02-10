"""Collaboration criteria illustration."""
from __future__ import annotations

import argparse
import time

import numpy as np
from joblib import Parallel, delayed

from dyntr.datasets import fetch_synthetic_datasets
from dyntr.utils import format_duration, print_dataset_info, set_random_state

from config import get_config
from experiments import run_one_performance_comparison
from plotting import plot_stats_evolution


_CFG = get_config()
N_JOBS = _CFG.N_JOBS
N_RUNS = _CFG.N_RUNS
FONTSIZE = _CFG.FONTSIZE
LW = _CFG.LW
ALPHA = _CFG.ALPHA
FIGURES_DIR = _CFG.FIGURES_DIR
SEED = _CFG.SEED
LBDA_DEFAULT = _CFG.LBDA_DEFAULT
D = _CFG.D
SIGMA = _CFG.SIGMA

t0 = time.time()


###############################################################################
# Settings
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=float, required=True, help="L2-norm approximation error.")
    parser.add_argument("-n_t", type=int, default=500, help="Number of samples to keep from dataset target.")
    parser.add_argument("-n_s", type=int, default=1000, help="Number of samples to keep from dataset source.")
    parser.add_argument("--alpha", type=float, required=True, help="LCB parameter.")
    parser.add_argument("--joblib-verbose", type=int, default=0, help="Verbosity for joblib (0 = silent).")
    args = parser.parse_args()

    set_random_state(seed=SEED)

###############################################################################
# Data
    sigma_t = SIGMA
    sigma_s = SIGMA
    data = fetch_synthetic_datasets(
        n_t_train=args.n_t,
        n_s_train=args.n_s,
        d=D,
        eps=args.eps,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
    )

    X_t, y_t = data["target_train"]
    X_t_val, y_t_val = data["target_val"]
    X_t_test, y_t_test = data["target_test"]
    l_X_s, l_y_s = data["source_train"]

    X_s = np.vstack(l_X_s)
    y_s = np.concatenate(l_y_s)

    theta_t = data["meta"]["theta_t"]
    theta_s = data["meta"]["theta_s"]

    print("[INFO] Running optimal sample sharing experiment...")
    print_dataset_info(X_t, y_t, X_s, y_s, X_t_val, X_t_test, theta_t, sigma_t, sigma_s)

###############################################################################
# Main
    delayed_runs = []
    for r in range(N_RUNS):
        kwargs = dict(
            n_samples_target=args.n_t,
            theta_t=theta_t,
            theta_s=theta_s,
            X_t_train=X_t,
            y_t_train=y_t,
            X_t_val=X_t_val,
            y_t_val=y_t_val,
            X_t_test=X_t_test,
            y_t_test=y_t_test,
            l_X_s=l_X_s,
            l_y_s=l_y_s,
            sigma_t=sigma_t,
            sigma_s=sigma_s,
            lbda_s=LBDA_DEFAULT,
            alpha=args.alpha,
            seed=r,
            synthetic_data=True,
        )
        delayed_runs.append(delayed(run_one_performance_comparison)(**kwargs))

    results = Parallel(verbose=args.joblib_verbose, n_jobs=N_JOBS)(delayed_runs)
    results = [res["error:"] for res in results]

###############################################################################
# Plotting
    main_axis = [
        "Target-only ridge",
        "Mixed-task ridge",
        "Proposed (greedy)",
        "Proposed (uniform)",
        "Chen et al, 2014",
        "Obst et al, 2021",
        "Hierarchical",
    ]
    secondary_axis = None
    uncertainty = False
    plot_stats_evolution(
        results=results,
        xlogscale=True,
        ylogscale=True,
        main_axis=main_axis,
        secondary_axis=secondary_axis,
        figsize=(4.0, 4.5),
        uncertainty=uncertainty,
        title=r"$\varepsilon  =  $" + f"{args.eps:.2f}",
        xlabel=r"$n_T$",
        lw=LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        ncols_legend=2,
        legend_y_pad=1.25,
        figures_dir=FIGURES_DIR,
        filename="4_performance_comparison",
        filename_suffix=f"__eps_{args.eps:.2f}",
        verbose=True,
    )

###############################################################################
# Timing
    print(f"[INFO] Experiment duration: {format_duration(time.time() - t0)}.")
