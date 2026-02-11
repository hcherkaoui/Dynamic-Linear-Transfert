"""Collaboration criteria illustration."""
from __future__ import annotations
import argparse
import time
import numpy as np
from joblib import Parallel, delayed
from dyntr.datasets import fetch_synthetic_datasets
from dyntr.utils import format_duration, print_dataset_info, set_random_state
from config import get_config
from experiments import run_one_lbda_infl
from plotting import plot_stats_evol


_CFG = get_config()
N_JOBS = _CFG.N_JOBS
N_RUNS = _CFG.N_RUNS
FONTSIZE = _CFG.FONTSIZE
LW = _CFG.LW
ALPHA = _CFG.ALPHA
FIGURES_DIR = _CFG.FIGURES_DIR
SEED = _CFG.SEED
D = _CFG.D
SIGMA = _CFG.SIGMA
EPS = _CFG.EPS

t0 = time.time()


###############################################################################
# Settings
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=1.0, help="LCB parameter.")
    parser.add_argument("-n_t", type=int, default=500, help="Number of samples to keep from dataset target.")
    parser.add_argument("-n_s", type=int, default=1000, help="Number of samples to keep from dataset source.")
    parser.add_argument("--joblib-verbose", type=int, default=0, help="Verbosity for joblib (0 = silent).")
    args = parser.parse_args()

###############################################################################
# Data
    set_random_state(seed=SEED)
    sigma_t = SIGMA
    sigma_s = 0.5 * SIGMA
    eps = EPS[0] if isinstance(EPS, list) else EPS
    data = fetch_synthetic_datasets(
        n_t_train=args.n_t,
        n_s_train=args.n_s,
        d=D,
        eps=eps,
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
            alpha=args.alpha,
            seed=r,
            synthetic_data=True,
        )
        delayed_runs.append(delayed(run_one_lbda_infl)(**kwargs))

    results = Parallel(verbose=args.joblib_verbose, n_jobs=N_JOBS)(delayed_runs)

###############################################################################
# Plotting
    main_axis = [r"$\Delta^{\mathrm{emp}}$"]
    secondary_axis = [r"$n^\star$"]
    uncertainty = True
    title = ""
    plot_stats_evol(
        results=results,
        xlogscale=True,
        ylogscale=False,
        nice_xlogscale=False,
        main_axis=main_axis,
        secondary_axis=secondary_axis,
        main_axis_ref_line=0.0,
        figsize=(3.5, 3.2),
        uncertainty=uncertainty,
        title=title,
        xlabel=r"$\lambda_S$",
        lw=LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        ncols_legend=1,
        legend_y_pad=1.25,
        figures_dir=FIGURES_DIR,
        filename="6_lbda_influence",
        filename_suffix="",
        verbose=True,
    )

###############################################################################
# Timing
    print(f"[INFO] Experiment duration: {format_duration(time.time() - t0)}.")
