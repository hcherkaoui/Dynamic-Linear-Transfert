"""Synthetic performance comparison (classification, multi-source)."""
from __future__ import annotations
import argparse
import time
from joblib import Parallel, delayed
from dyntr.datasets import fetch_synthetic_classification_datasets
from dyntr.utils import format_duration, set_random_state
from config import get_config
from experiments import run_one_classif_perf_cmp
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
D = _CFG.D
SIGMA = _CFG.SIGMA

CHUNK_SIZE = 10
START_PLOT = 2


t0 = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=float, required=True, help="Transfer gap epsilon.")
    parser.add_argument("-n_t", type=int, default=500, help="Number of samples to keep for target.")
    parser.add_argument("-n_s", type=int, default=1000, help="Number of samples per source.")
    parser.add_argument("--alpha", type=float, required=True, help="LCB parameter.")
    parser.add_argument("--joblib-verbose", type=int, default=0, help="Verbosity for joblib (0 = silent).")
    args = parser.parse_args()

    set_random_state(seed=SEED)

    data = fetch_synthetic_classification_datasets(
        n_t_train=args.n_t,
        n_t_val=max(1, args.n_t // 5),
        n_t_test=max(1, args.n_t // 5),
        n_s_train=args.n_s,
        n_s_val=max(1, args.n_s // 5),
        n_s_test=max(1, args.n_s // 5),
        d=D,
        eps=args.eps,
        sigma_t=SIGMA,
        sigma_s=SIGMA,
        n_sources=CHUNK_SIZE,
        random_state=SEED,
    )

    X_t, y_t = data["target_train"]
    X_t_val, y_t_val = data["target_val"]
    X_t_test, y_t_test = data["target_test"]
    l_X_s, l_y_s = data["source_train"]
    theta_t = data["meta"]["theta_t"]
    theta_s = data["meta"]["theta_s"]

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
            sigma_t=0.0,
            sigma_s=0.0,
            lbda_s=LBDA_DEFAULT,
            alpha=args.alpha,
            seed=r,
            synthetic_data=True,
        )
        delayed_runs.append(delayed(run_one_classif_perf_cmp)(**kwargs))

    results = Parallel(verbose=args.joblib_verbose, n_jobs=N_JOBS)(delayed_runs)

    main_axis = [
        "Target-only ridge",
        "Source-only ridge",
        "Mixed-task ridge",
        "Hierarchical classification",
    ]
    secondary_axis = None
    uncertainty = False
    plot_stats_evol(
        results=[res["acc:"] for res in results],
        xlogscale=True,
        ylogscale=False,
        start=START_PLOT,
        main_axis=main_axis,
        secondary_axis=secondary_axis,
        figsize=(3.75, 4.5),
        uncertainty=uncertainty,
        title=r"$\varepsilon  =  $" + f"{args.eps:.2f}",
        xlabel=r"$n_T$",
        lw=LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        ncols_legend=2,
        legend_y_pad=1.25,
        figures_dir=FIGURES_DIR,
        filename="8_performance_classif_acc_multi_source",
        filename_suffix=f"__eps_{args.eps:.2f}",
        verbose=True,
    )

    print(f"[INFO] Experiment duration: {format_duration(time.time() - t0)}.")
