"""Collaboration criteria illustration."""
from __future__ import annotations
import argparse
import time
import numpy as np
from joblib import Parallel, delayed
from dyntr.datasets import real_regr_data
from dyntr.utils import compute_exp_vars, format_duration, set_random_state
from experiments import run_one_regr_perf_cmp, get_config
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
N_T = _CFG.N_T
N_S = _CFG.N_S
JOBLIB_VERBOSE = _CFG.JOBLIB_VERBOSE
CHUNK_SIZE = _CFG.CHUNK_SIZE
STANDARDIZE_REAL = _CFG.STANDARDIZE_REAL


t0 = time.time()


###############################################################################
# Settings
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name.")
    args = parser.parse_args()

    set_random_state(seed=SEED)

###############################################################################
# Data
    data = real_regr_data(
        dataset_name=args.dataset_name,
        n_t_train=N_T,
        n_t_val=max(1, N_T // 20),
        n_t_test=max(1, N_T // 20),
        n_s_train=N_S,
        n_s_val=max(1, N_S // 20),
        n_s_test=max(1, N_S // 20),
        sigma_t=1.0,
        sigma_s=0.0,
        permutation=True,
        standardize=STANDARDIZE_REAL,
    )

    X_t_train, y_t_train = data["target_train"]
    X_t_val, y_t_val = data["target_val"]
    X_t_test, y_t_test = data["target_test"]
    l_X_s, l_y_s = data["source_train"]

    X_s = np.vstack(l_X_s)
    y_s = np.concatenate(l_y_s)

    exp_vars = compute_exp_vars(
        X_t=X_t_train,
        X_s=X_s,
        y_t=y_t_train,
        y_s=y_s,
        X_t_val=X_t_val,
        y_t_val=y_t_val,
        lbda_s=LBDA_DEFAULT,
    )
    theta_t = exp_vars["theta_t"]
    theta_s = exp_vars["theta_s"]
    sigma_t = exp_vars["sigma_t"]
    sigma_s = exp_vars["sigma_s"]

    n_t = data["meta"]["n_t_train"]

    print("[INFO] Running optimal sample sharing experiment...")

###############################################################################
# Main
    delayed_runs = []
    for r in range(N_RUNS):
        kwargs = dict(
            n_samples_target=n_t,
            theta_t=theta_t,
            theta_s=theta_s,
            X_t_train=X_t_train,
            y_t_train=y_t_train,
            X_t_val=X_t_val,
            y_t_val=y_t_val,
            X_t_test=X_t_test,
            y_t_test=y_t_test,
            l_X_s=l_X_s,
            l_y_s=l_y_s,
            sigma_t=sigma_t,
            sigma_s=sigma_s,
            lbda_s=LBDA_DEFAULT,
            alpha=LCB_ALPHA,
            chunk_size=CHUNK_SIZE,
            seed=r,
            synthetic_data=False,
        )
        delayed_runs.append(delayed(run_one_regr_perf_cmp)(**kwargs))

    results = Parallel(verbose=JOBLIB_VERBOSE, n_jobs=N_JOBS)(delayed_runs)

###############################################################################
# Plotting
    main_axis = [
        "Target-only",
        "Mixed",
        "Hierarchical",
        "Zhang et al, 2025",
        "Chen et al, 2014",
        "Obst et al, 2021",
        "Proposed",
    ]
    if args.dataset_name == "email": main_axis.remove("Obst et al, 2021")  # XXX # noqa
    secondary_axis = None
    uncertainty = False
    plot_stats_evol(
        results=[res["error"] for res in results],
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
        markersize=MARKERSIZE,
        ncols_legend=2,
        legend_y_pad=1.25,
        figures_dir=FIGURES_DIR,
        filename="05_performance_error_comparison",
        filename_suffix=f"_dataset_{args.dataset_name}",
        verbose=True,
    )

    plot_stats_evol(
        results=[res["mse"] for res in results],
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
        markersize=MARKERSIZE,
        ncols_legend=2,
        legend_y_pad=1.25,
        figures_dir=FIGURES_DIR,
        filename="05_performance_mse_comparison",
        filename_suffix=f"_dataset_{args.dataset_name}",
        verbose=True,
    )

###############################################################################
# Timing
    print(f"[INFO] Experiment duration: {format_duration(time.time() - t0)}.")
