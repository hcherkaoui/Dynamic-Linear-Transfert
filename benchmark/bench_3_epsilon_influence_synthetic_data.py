"""Collaboration criteria illustration."""

# Authors: Hamza Cherkaoui

from __future__ import annotations
import time
import argparse
from joblib import Parallel, delayed
from dyntr.utils import format_duration, set_random_state
from dyntr.plotting import plot_stats_evolution
from benchmark.experiments import run_one_epsilon_influence
from dyntr.constants import (N_JOBS, N_RUNS, FONTSIZE, LW, ALPHA, FIGURES_DIR,
                             SEED, LBDA_DEFAULT, D, SIGMA, L_EPS)


t0 = time.time()


###############################################################################
# Settings
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="LCB parameter.")
    parser.add_argument("--joblib-verbose", type=int, default=0,
                        help="Verbosity for joblib (0 = silent).")
    args = parser.parse_args()

    set_random_state(seed=SEED)

###############################################################################
# Main
    print("[INFO] Running optimal sample sharing experiment...")
    kwargs = dict(d=D, l_eps=L_EPS, lbda_s=LBDA_DEFAULT, alpha=args.alpha)
    delayed_runs = [delayed(run_one_epsilon_influence)(**kwargs)
                    for _ in range(N_RUNS)]

    results = Parallel(verbose=args.joblib_verbose, n_jobs=N_JOBS)(delayed_runs)

###############################################################################
# Plotting
    main_axis = [r"$\Delta^{\mathrm{emp}}$",]
    secondary_axis = [r"$n^\star$"]
    uncertainty = True
    title = r"$\sigma_T = \sigma_S = $" + f"{SIGMA:.2f}"
    plot_stats_evolution(results=results, xlogscale=True, ylogscale=False, nice_xlogscale=False,
                         main_axis=main_axis, secondary_axis=secondary_axis, main_axis_ref_line=0.0,
                         figsize=(3.5, 3.2), uncertainty=uncertainty, title=title, xlabel=r'$\varepsilon$',
                         lw=LW, alpha=ALPHA, fontsize=FONTSIZE, ncols_legend=1, legend_y_pad=1.25,
                         figures_dir=FIGURES_DIR, filename="3_epsilon_influence",
                         filename_suffix=f"__sigma_{SIGMA:.2f}", pdf_cropping=True, verbose=True,
                         )

###############################################################################
# Timing
    print(f"[INFO] Experiment duration: {format_duration(time.time() - t0)}.")
