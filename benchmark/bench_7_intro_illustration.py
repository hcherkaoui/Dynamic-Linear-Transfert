"""Collaboration criteria illustration."""

# Authors: Hamza Cherkaoui

from __future__ import annotations
import time
import argparse
from joblib import Parallel, delayed
from dyntr.utils import format_duration, set_random_state, print_dataset_info
from dyntr.datasets import fetch_synthetic_datasets
from plotting import plot_stats_evolution
from experiments import run_one_performance_comparison
from config import get_config


_CFG = get_config()
N_JOBS = _CFG.N_JOBS
N_RUNS = _CFG.N_RUNS
ALPHA = _CFG.ALPHA
FIGURES_DIR = _CFG.FIGURES_DIR
SEED = _CFG.SEED
LBDA_DEFAULT = _CFG.LBDA_DEFAULT
D = _CFG.D
SIGMA = _CFG.SIGMA

t0 = time.time()

FONTSIZE_LOCAL = 19     # text will be cropped anyway
FIGSIZE = (3.75, 3.5)
N_T_MIN = 10            # to center the curves crossroad
LW_LOCAL = 6

###############################################################################
# Settings
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=float, required=True,
                        help="L2-norm approximation error.")
    parser.add_argument("-n_t", type=int, default=500,
                        help="Number of samples to keep from dataset target.")
    parser.add_argument("-n_s", type=int, default=1000,
                        help="Number of samples to keep from dataset source.")
    parser.add_argument("--alpha", type=float, required=True,
                        help="LCB parameter.")
    parser.add_argument("--joblib-verbose", type=int, default=0,
                        help="Verbosity for joblib (0 = silent).")
    args = parser.parse_args()

    set_random_state(seed=SEED)

###############################################################################
# Data
    sigma_t = SIGMA
    sigma_s = SIGMA
    data, meta = fetch_synthetic_datasets(n_train_samples=args.n_s,
                                          d=D,
                                          eps=args.eps,
                                          sigma_t=sigma_t,
                                          sigma_s=sigma_s,
                                          only_train=False,
                                          )

    (X_t, X_s, y_t, y_s), (X_t_val, _, y_t_val, _), (X_t_test, _, y_t_test, _) = data

    theta_t = meta['theta_t']
    theta_s = meta['theta_s']

    print("[INFO] Running optimal sample sharing experiment...")
    print_dataset_info(X_t, y_t, X_s, y_s, X_t_val, X_t_test, theta_t, sigma_t, sigma_s)

###############################################################################
# Main
    delayed_runs = []
    for r in range(N_RUNS):
        kwargs = dict(n_samples_target=args.n_t, theta_t=theta_t, theta_s=theta_s,
                      X_t_train=X_t, y_t_train=y_t, X_t_val=X_t_val, y_t_val=y_t_val,
                      X_t_test=X_t_test, y_t_test=y_t_test, X_s=X_s, y_s=y_s,
                      sigma_t=sigma_t, sigma_s=sigma_s, lbda_s=LBDA_DEFAULT, alpha=args.alpha,
                      min_n_samples_target=N_T_MIN, seed=r, synthetic_data=True)
        delayed_runs.append(delayed(run_one_performance_comparison)(**kwargs))

    results = Parallel(verbose=args.joblib_verbose, n_jobs=N_JOBS)(delayed_runs)
    results = [res["error:"] for res in results]

###############################################################################
# Plotting
    main_axis = ["Target-only ridge", "Mixed-task ridge", "Proposed"]
    secondary_axis = None
    uncertainty = False
    plot_stats_evolution(results=results, xlogscale=True, ylogscale=False,
                         main_axis=main_axis, secondary_axis=secondary_axis, figsize=FIGSIZE,
                         uncertainty=uncertainty, title="", xlabel=r"$n_T$", lw=LW_LOCAL, alpha=ALPHA,
                         fontsize=FONTSIZE_LOCAL, ncols_legend=2, legend_y_pad=1.25, figures_dir=FIGURES_DIR,
                         filename="7_intro_illustration", filename_suffix=f"__eps_{args.eps:.2f}",
                         pdf_cropping=True, verbose=True,
                         )

###############################################################################
# Timing
    print(f"[INFO] Experiment duration: {format_duration(time.time() - t0)}.")
