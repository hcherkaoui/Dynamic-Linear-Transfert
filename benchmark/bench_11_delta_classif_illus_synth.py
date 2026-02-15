"""Collaboration criteria illustration."""
from __future__ import annotations
import argparse
import time
from joblib import Parallel, delayed
from dyntr.datasets import synth_classif_data
from dyntr.utils import format_duration, set_random_state
from experiments import run_one_delta_illus, get_config
from plotting import plot_stats_evol


_CFG = get_config()
N_JOBS = _CFG.N_JOBS
N_RUNS = _CFG.N_RUNS
FONTSIZE = _CFG.FONTSIZE
LW = _CFG.LW
ALPHA = _CFG.ALPHA
MARKERSIZE = _CFG.MARKERSIZE
FIGURES_DIR = _CFG.FIGURES_DIR
SEED = _CFG.SEED
LBDA_DEFAULT = _CFG.LBDA_DEFAULT
D = _CFG.D
SIGMA = _CFG.SIGMA
N_T = _CFG.N_T
N_S = _CFG.N_S
JOBLIB_VERBOSE = _CFG.JOBLIB_VERBOSE


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
    sigma_t = SIGMA
    sigma_s = 0.5 * SIGMA
    data = synth_classif_data(
        n_t_train=N_T,
        n_t_val=max(1, N_T // 50),
        n_t_test=max(1, N_T // 50),
        n_s_train=N_S,
        n_s_val=max(1, N_S // 50),
        n_s_test=max(1, N_S // 50),
        d=D,
        eps=args.eps,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
    )

    X_t, y_t = data["target_train"]
    X_t_val, y_t_val = data["target_val"]
    X_t_test, y_t_test = data["target_test"]
    l_X_s, l_y_s = data["source_train"]

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
            sigma_t=sigma_t,
            sigma_s=sigma_s,
            lbda_t=LBDA_DEFAULT,
            lbda_s=LBDA_DEFAULT,
            min_n_samples_target=10,
            seed=r,
            synthetic_data=True,
        )
        delayed_runs.append(delayed(run_one_delta_illus)(**kwargs))

    results = Parallel(verbose=JOBLIB_VERBOSE, n_jobs=N_JOBS)(delayed_runs)

###############################################################################
# Plotting
    plot_stats_evol(
        results=results,
        xlogscale=False,
        ylogscale=False,
        main_axis=[r"$\hat{\Delta}$", r"$\Delta^{\star}$"],
        main_axis_ref_line=0.0,
        figsize=(3.5, 3.2),
        uncertainty=True,
        title=r"$\varepsilon  =  $" + f"{args.eps:.1f}",
        xlabel=r"$n$",
        lw=LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        markersize=MARKERSIZE,
        ncols_legend=2,
        legend_y_pad=1.15,
        figures_dir=FIGURES_DIR,
        filename="11_delta_classif_illus",
        filename_suffix=f"_eps_{args.eps:.1f}",
        verbose=True,
    )

###############################################################################
# Timing
    print(f"[INFO] Experiment duration: {format_duration(time.time() - t0)}.")
