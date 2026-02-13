"""Synthetic performance comparison (classification, multi-source)."""
from __future__ import annotations
import time
import argparse
from joblib import Parallel, delayed
from dyntr.datasets import synth_classif_data
from dyntr.utils import format_duration, set_random_state
from experiments import run_one_classif_perf_cmp, get_config
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
START_PLOT = _CFG.START_PLOT

t0 = time.time()


###############################################################################
# Settings
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=float, required=True, help="L2-norm approximation error.")
    args = parser.parse_args()

    set_random_state(seed=SEED)

    l_eps = [args.eps, args.eps / 2.0, args.eps / 3.0]
    per_line = 2

    str_norm = r"\,\cdot\,\|\theta_T^{\star}\|_2"
    parts = [(rf"$\varepsilon_{{{i}}} = 0$" if abs(eps) < 1e-12 else rf"$\varepsilon_{{{i}}} = {eps:.1f}\,{str_norm}$")
             for i, eps in enumerate(l_eps, start=1)
             ]
    lines = []
    for j in range(0, len(parts), per_line):
        chunk = parts[j:j + per_line]
        lines.append(" , ".join(chunk) + (" , " if j + per_line < len(parts) else ""))
    title = "\n".join(lines) + " ."

###############################################################################
# Data
    kwargs = dict(n_t_train=N_T,
                  n_t_val=max(1, N_T // 50),
                  n_t_test=max(1, N_T // 50),
                  n_s_train=max(1, N_S // len(l_eps)),
                  n_s_val=max(1, N_S // (50)),
                  n_s_test=max(1, N_S // (50 * len(l_eps))),
                  d=D,
                  sigma_t=SIGMA,
                  sigma_s=SIGMA,
                  )

    l_X_s, l_y_s = [], []
    for eps in l_eps:
        kwargs['eps'] = eps
        dataset = synth_classif_data(**kwargs)
        l_X_s.append(dataset["source_train"][0][0])
        l_y_s.append(dataset["source_train"][1][0])

    X_t_train, y_t_train = dataset["target_train"]
    X_t_val, y_t_val = dataset["target_val"]
    X_t_test, y_t_test = dataset["target_test"]
    theta_t = dataset["meta"]["theta_t"]
    theta_s = dataset["meta"]["theta_s"]

    print("[INFO] Running optimal sample sharing experiment...")

###############################################################################
# Main
    delayed_runs = []
    for r in range(N_RUNS):
        kwargs = dict(
            n_samples_target=N_T,
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
            sigma_t=2 * SIGMA,
            sigma_s=SIGMA,
            lbda_s=LBDA_DEFAULT,
            alpha=LCB_ALPHA,
            chunk_size=CHUNK_SIZE,
            seed=r,
            synthetic_data=True,
        )
        delayed_runs.append(delayed(run_one_classif_perf_cmp)(**kwargs))

    results = Parallel(verbose=JOBLIB_VERBOSE, n_jobs=N_JOBS)(delayed_runs)

###############################################################################
# Plotting
    main_axis = [
        "Target-only",
        "Mixed",
        "Hierarchical",
        "Proposed",
        "Proposed (oracle)",
    ]
    secondary_axis = None
    uncertainty = False
    plot_stats_evol(
        results=[res["acc"] for res in results],
        xlogscale=True,
        ylogscale=False,
        start=START_PLOT,
        main_axis=main_axis,
        secondary_axis=secondary_axis,
        figsize=(3.75, 4.5),
        uncertainty=uncertainty,
        title=title,
        xlabel=r"$n_T$",
        lw=LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        markersize=MARKERSIZE,
        ncols_legend=2,
        legend_y_pad=1.35,
        figures_dir=FIGURES_DIR,
        filename="10_performance_classif_acc_multi_source",
        filename_suffix=f"_eps_{args.eps:.1f}",
        verbose=True,
    )

    print(f"[INFO] Experiment duration: {format_duration(time.time() - t0)}.")
