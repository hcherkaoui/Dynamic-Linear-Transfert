"""Collaboration criteria illustration."""

# Authors: Hamza Cherkaoui

from __future__ import annotations
import time
import argparse
from joblib import Parallel, delayed
from dyntr.utils import format_duration, set_random_state, print_dataset_info
from dyntr.datasets import fetch_synthetic_datasets
from dyntr.plotting import plot_stats_evolution
from benchmark.experiments import run_one_delta_illustration
from dyntr.constants import (N_JOBS, N_RUNS, FONTSIZE, LW, ALPHA, FIGURES_DIR,
                             SEED, LBDA_DEFAULT, D, SIGMA)


t0 = time.time()


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
    parser.add_argument("--joblib-verbose", type=int, default=0,
                        help="Verbosity for joblib (0 = silent).")
    args = parser.parse_args()

    set_random_state(seed=SEED)

###############################################################################
# Data
    sigma_t = SIGMA
    sigma_s = 0.5 * SIGMA
    data, meta = fetch_synthetic_datasets(n_train_samples=args.n_s,
                                          d=D,
                                          eps=args.eps,
                                          sigma_t=sigma_t,
                                          sigma_s=sigma_s,
                                          only_train=False,
                                          )

    (X_t, X_s, y_t, y_s), (X_t_val, _, _, _), (X_t_test, _, _, _) = data

    theta_t = meta['theta_t']
    theta_s = meta['theta_s']

    X_t = X_t[: args.n_t]
    y_t = y_t[: args.n_t]

    print("[INFO] Running optimal sample sharing experiment...")
    print_dataset_info(X_t, y_t, X_s, y_s, X_t_val, X_t_test, theta_t, sigma_t, sigma_s)

###############################################################################
# Main
    delayed_runs = []
    for r in range(N_RUNS):
        kwargs = dict(theta_t=theta_t, theta_s=theta_s, X_t=X_t, X_s=X_s, y_t=y_t, y_s=y_s,
                      X_t_val=X_t_val, X_t_test=X_t_test, sigma_t=sigma_t, sigma_s=sigma_s,
                      lbda_t=LBDA_DEFAULT, lbda_s=LBDA_DEFAULT, seed=r, synthetic_data=True)
        delayed_runs.append(delayed(run_one_delta_illustration)(**kwargs))

    results = Parallel(verbose=args.joblib_verbose, n_jobs=N_JOBS)(delayed_runs)

###############################################################################
# Plotting
    plot_stats_evolution(results=results, xlogscale=False, ylogscale=False,
                         main_axis=[r"$\hat{\Delta}$", r"$\Delta^{\mathrm{emp}}$"],
                         main_axis_ref_line=0.0, figsize=(3.5, 3.2), uncertainty=True,
                         title=r"$\varepsilon = $" + f"{args.eps:.2f}", xlabel=r"$n$",
                         lw=LW, alpha=ALPHA, fontsize=FONTSIZE, ncols_legend=2, legend_y_pad=1.15,
                         figures_dir=FIGURES_DIR, filename="1_delta_illustration",
                         filename_suffix=f"__eps_{args.eps:.2f}", pdf_cropping=True, verbose=True,
                         )

###############################################################################
# Timing
    print(f"[INFO] Experiment duration: {format_duration(time.time() - t0)}.")
