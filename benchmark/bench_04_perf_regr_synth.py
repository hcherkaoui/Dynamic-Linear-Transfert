"""Collaboration criteria illustration."""
from __future__ import annotations
import argparse
import time
import numpy as np
from joblib import Parallel, delayed
from dyntr.datasets import synth_regr_data
from dyntr.utils import format_duration, set_random_state
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
D = _CFG.D
SIGMA = _CFG.SIGMA
N_T = _CFG.N_T
N_S = _CFG.N_TRAIN_SAMPLES
JOBLIB_VERBOSE = _CFG.JOBLIB_VERBOSE
CHUNK_SIZE = _CFG.CHUNK_SIZE

t0 = time.time()


###############################################################################
# Settings
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=float, required=True, help="L2-norm approximation error.")
    args = parser.parse_args()

    set_random_state(seed=SEED)

###############################################################################
# Data
    data = synth_regr_data(
        n_t_train=N_T,
        n_t_val=max(1, N_T // 50),
        n_t_test=max(1, N_T // 50),
        n_s_train=N_S,
        n_s_val=max(1, N_S // 50),
        n_s_test=max(1, N_S // 50),
        d=D,
        eps=args.eps,
        sigma_t=SIGMA,
        sigma_s=SIGMA,
    )

    X_t, y_t = data["target_train"]
    X_t_val, y_t_val = data["target_val"]
    X_t_test, y_t_test = data["target_test"]
    l_X_s, l_y_s = data["source_train"]

    X_s = np.vstack(l_X_s)
    y_s = np.concatenate(l_y_s)

    theta_t = data["meta"]["theta_t"]
    theta_s = data["meta"]["theta_s"]

###############################################################################
# Main
    delayed_runs = []
    for r in range(N_RUNS):
        kwargs = dict(
            n_samples_target=N_T,
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
            sigma_t=SIGMA,
            sigma_s=SIGMA,
            lbda_s=LBDA_DEFAULT,
            alpha=LCB_ALPHA,
            chunk_size=CHUNK_SIZE,
            seed=r,
            synthetic_data=True,
        )
        delayed_runs.append(delayed(run_one_regr_perf_cmp)(**kwargs))

    results = Parallel(verbose=JOBLIB_VERBOSE, n_jobs=N_JOBS)(delayed_runs)

###############################################################################
# Plotting
    main_axis = [
        "Target-only",
        "Mixed",
        "Hierarchical",
        "Chen et al, 2014",
        "Obst et al, 2021",
        "Zhang et al, 2025",
        "Proposed",
        "Proposed (oracle)",
    ]
    secondary_axis = None
    uncertainty = False
    uncertainty_style = "band"
    plot_stats_evol(
        results=[res["error"] for res in results],
        xlogscale=True,
        ylogscale=True,
        main_axis=main_axis,
        secondary_axis=secondary_axis,
        figsize=(3.75, 4.5),
        uncertainty=uncertainty,
        uncertainty_style=uncertainty_style,
        title=r"$\varepsilon  =  $" + f"{args.eps:.1f}" + r"$\,\cdot\,\|\theta_T^{\star}\|_2$",
        xlabel=r"$n_T$",
        lw=LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        markersize=MARKERSIZE,
        ncols_legend=2,
        legend_y_pad=1.35,
        figures_dir=FIGURES_DIR,
        filename="04_performance_error_comparison",
        filename_suffix=f"_eps_{args.eps:.1f}",
        verbose=True,
    )
    plot_stats_evol(
        results=[res["mse"] for res in results],
        xlogscale=True,
        ylogscale=True,
        main_axis=main_axis,
        secondary_axis=secondary_axis,
        figsize=(3.75, 4.5),
        uncertainty=uncertainty,
        uncertainty_style=uncertainty_style,
        title=r"$\varepsilon  =  $" + f"{args.eps:.1f}" + r"$\,\cdot\,\|\theta_T^{\star}\|_2$",
        xlabel=r"$n_T$",
        lw=LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        markersize=MARKERSIZE,
        ncols_legend=2,
        legend_y_pad=1.35,
        figures_dir=FIGURES_DIR,
        filename="04_performance_mse_comparison",
        filename_suffix=f"_eps_{args.eps:.1f}",
        verbose=True,
    )

###############################################################################
# Timing
    print(f"[INFO] Experiment duration: {format_duration(time.time() - t0)}.")
