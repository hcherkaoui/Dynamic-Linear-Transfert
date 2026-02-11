"""Collaboration criteria illustration."""
from __future__ import annotations
import argparse
import time
import numpy as np
from joblib import Parallel, delayed
from dyntr.datasets import fetch_real_datasets
from dyntr.utils import compute_exp_vars, format_duration, print_dataset_info, set_random_state
from config import get_config
from experiments import run_one_regr_perf_cmp
from plotting import plot_stats_evol


_CFG = get_config()
N_JOBS = _CFG.N_JOBS
N_RUNS = _CFG.N_RUNS
FONTSIZE = _CFG.FONTSIZE
LW = _CFG.LW
ALPHA = _CFG.ALPHA
FIGURES_DIR = _CFG.FIGURES_DIR
SEED = _CFG.SEED
LBDA_DEFAULT = _CFG.LBDA_DEFAULT


t0 = time.time()


###############################################################################
# Settings
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name.")
    parser.add_argument("-n_t", type=int, default=500, help="Number of samples to keep from dataset target.")
    parser.add_argument("-n_s", type=int, default=1000, help="Number of samples to keep from dataset source.")
    parser.add_argument("--alpha", type=float, required=True, help="LCB parameter.")
    parser.add_argument("--joblib-verbose", type=int, default=0, help="Verbosity for joblib (0 = silent).")
    args = parser.parse_args()

    set_random_state(seed=SEED)

###############################################################################
# Data
    data = fetch_real_datasets(
        dataset_name=args.dataset_name,
        n_t_train=args.n_t,
        n_t_val=max(1, args.n_t // 5),
        n_t_test=max(1, args.n_t // 5),
        n_s_train=args.n_s,
        n_s_val=max(1, args.n_s // 5),
        n_s_test=max(1, args.n_s // 5),
        sigma_t=1.0,
        sigma_s=0.0,
        permutation=True,
    )

    X_t, y_t = data["target_train"]
    X_t_val, y_t_val = data["target_val"]
    X_t_test, y_t_test = data["target_test"]
    l_X_s, l_y_s = data["source_train"]

    X_s = np.vstack(l_X_s)
    y_s = np.concatenate(l_y_s)

    exp_vars = compute_exp_vars(
        X_t=X_t,
        X_s=X_s,
        y_t=y_t,
        y_s=y_s,
        X_t_val=X_t_val,
        y_t_val=y_t_val,
        lbda_s=LBDA_DEFAULT,
    )
    theta_t = exp_vars["theta_t"]
    theta_s = exp_vars["theta_s"]
    sigma_t = exp_vars["sigma_t"]
    sigma_s = exp_vars["sigma_s"]

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
            synthetic_data=False,
        )
        delayed_runs.append(delayed(run_one_regr_perf_cmp)(**kwargs))

    results = Parallel(verbose=args.joblib_verbose, n_jobs=N_JOBS)(delayed_runs)

###############################################################################
# Plotting
    main_axis = [
        "Target-only ridge",
        "Mixed-task ridge",
        "Hierarchical regression",
        "Chen et al, 2014",
        "Obst et al, 2021",
        "Proposed (greedy)",
    ]
    secondary_axis = None
    uncertainty = False
    plot_stats_evol(
        results=[res["error:"] for res in results],
        xlogscale=True,
        ylogscale=False,
        main_axis=main_axis,
        secondary_axis=secondary_axis,
        figsize=(3.75, 4.5),
        uncertainty=uncertainty,
        title=args.dataset_name,
        xlabel=r"$n_T$",
        lw=LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        ncols_legend=2,
        legend_y_pad=1.25,
        figures_dir=FIGURES_DIR,
        filename="5_performance_error_comparison",
        filename_suffix=f"__dataset_{args.dataset_name}",
        verbose=True,
    )
    plot_stats_evol(
        results=[res["mse:"] for res in results],
        xlogscale=True,
        ylogscale=True,
        main_axis=main_axis,
        secondary_axis=secondary_axis,
        figsize=(3.75, 4.5),
        uncertainty=uncertainty,
        title=args.dataset_name,
        xlabel=r"$n_T$",
        lw=LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        ncols_legend=2,
        legend_y_pad=1.35,
        figures_dir=FIGURES_DIR,
        filename="5_performance_mse_comparison",
        filename_suffix=f"__dataset_{args.dataset_name}",
        verbose=True,
    )

###############################################################################
# Timing
    print(f"[INFO] Experiment duration: {format_duration(time.time() - t0)}.")
