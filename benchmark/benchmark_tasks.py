"""Benchmark task registry used by compute and plot entry points."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Callable
from joblib import Parallel, delayed
import numpy as np
from dyntr.datasets import real_regr_data, synth_classif_data, synth_regr_data
from dyntr.utils import compute_exp_vars, set_random_state
from .experiments import (
    get_config,
    run_one_classif_perf_cmp,
    run_one_delta_illus,
    run_one_eps_infl,
    run_one_lbda_infl,
    run_one_regr_perf_cmp,
    run_one_sigma_infl,
)
from .pipeline import add_mode_subparsers, figures_dir, run_task_cli, ylimit_for
from .plotting import plot_stats_evol


_CFG = get_config()
N_JOBS = _CFG.N_JOBS
N_RUNS = _CFG.N_RUNS
REAL_N_RUNS = _CFG.REAL_N_RUNS
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
EPS = _CFG.EPS
N_T = _CFG.N_T
N_S = _CFG.N_S
L_EPS = _CFG.L_EPS
L_SIGMA_S = _CFG.L_SIGMA_S
JOBLIB_VERBOSE = _CFG.JOBLIB_VERBOSE
CHUNK_SIZE = _CFG.CHUNK_SIZE
START_PLOT = _CFG.START_PLOT
STANDARDIZE_REAL = _CFG.STANDARDIZE_REAL

REAL_YLIM_QUANTILE_DEFAULT = 0.70
LOGNORMAL_NOISE = "lognormal"
LOGNORMAL_SHAPE = 1.0
INTRO_FIGSIZE = (4.75, 3.75)
INTRO_N_T_MIN = 10
INTRO_LW = 6
INTRO_EPS = 0.5


@dataclass(frozen=True)
class ExperimentTask:
    """Executable benchmark experiment specification."""

    key: str
    description: str
    add_args: Callable
    result_name: Callable
    params: Callable
    compute: Callable
    plot: Callable


BenchmarkTask = ExperimentTask


def _add_eps_arg(parser) -> None:
    parser.add_argument("--eps", type=float, required=True, help="L2-norm approximation error.")


def _add_classif_eps_arg(parser) -> None:
    parser.add_argument("--eps", type=float, required=True, help="Transfer gap epsilon.")


def _add_dataset_arg(parser) -> None:
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name.")


def _add_no_args(_parser) -> None:
    return None


def _eps_result(prefix: str, args) -> str:
    return f"{prefix}_eps_{args.eps:.1f}"


def _dataset_result(prefix: str, args) -> str:
    return f"{prefix}_dataset_{args.dataset_name}"


def _no_arg_result(prefix: str) -> str:
    return prefix


def _eps_params(args) -> dict:
    return {"eps": float(args.eps)}


def _dataset_params(args) -> dict:
    return {"dataset_name": args.dataset_name}


def _no_params(_args) -> dict:
    return {}


def _parallel(delayed_runs):
    return Parallel(verbose=JOBLIB_VERBOSE, n_jobs=N_JOBS)(delayed_runs)


def _regression_main_axis(*, include_oracle: bool = True) -> list[str]:
    axis = [
        "Target-only",
        "Mixed",
        "Hierarchical",
        "Chen et al, 2014",
        "Obst et al, 2021",
        "Zhang et al, 2025",
        "Proposed",
    ]
    if include_oracle:
        axis.append("Proposed (oracle)")
    return axis


def _multi_source_title(l_eps) -> str:
    per_line = 2
    str_norm = r"\,\cdot\,\|\theta_T^{\star}\|_2"
    parts = [
        (rf"$\varepsilon_{{{i}}} = 0$" if abs(eps) < 1e-12 else rf"$\varepsilon_{{{i}}} = {eps:.1f}\,{str_norm}$")
        for i, eps in enumerate(l_eps, start=1)
    ]
    lines = []
    for j in range(0, len(parts), per_line):
        chunk = parts[j:j + per_line]
        lines.append(" , ".join(chunk) + (" , " if j + per_line < len(parts) else ""))
    return "\n".join(lines) + " ."


def _skewness(x):
    x = np.asarray(x, dtype=float)
    std = float(np.std(x))
    if std == 0.0:
        return 0.0
    centered = x - float(np.mean(x))
    return float(np.mean(centered ** 3) / std ** 3)


def _print_noise_check(X_t, y_t, X_s, y_s, theta_t, theta_s):
    residual_t = y_t - X_t @ theta_t
    residual_s = y_s - X_s @ theta_s
    for name, residual in (("target", residual_t), ("source", residual_s)):
        print(
            "[INFO] "
            f"{name} {LOGNORMAL_NOISE} residuals: "
            f"mean={np.mean(residual):.4g}, "
            f"std={np.std(residual):.4g}, "
            f"skew={_skewness(residual):.4g}",
            flush=True,
        )


def _compute_delta_regr(args):
    set_random_state(seed=SEED)
    sigma_t = SIGMA
    sigma_s = 0.5 * SIGMA
    data = synth_regr_data(
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

    delayed_runs = []
    for r in range(N_RUNS):
        delayed_runs.append(delayed(run_one_delta_illus)(
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
        ))
    return _parallel(delayed_runs)


def _plot_delta_regr(args, results):
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
        figures_dir=figures_dir(args),
        filename="01_delta_regr_illus",
        filename_suffix=f"_eps_{args.eps:.1f}",
        verbose=True,
    )


def _compute_sigma_infl(args):
    set_random_state(seed=SEED)
    print("[INFO] Running optimal sample sharing experiment...", flush=True)
    kwargs = dict(d=D, eps=args.eps, l_sigma_s=L_SIGMA_S, lbda_s=LBDA_DEFAULT, alpha=LCB_ALPHA)
    return _parallel([delayed(run_one_sigma_infl)(**kwargs) for _ in range(N_RUNS)])


def _plot_sigma_infl(args, results):
    main_axis = [r"$\Delta^{\star}$"]
    secondary_axis = [r"$\hat{n}$"]
    title = r"$\varepsilon = $" + f"{args.eps:.1f}" + r"$ ~, ~~ \sigma_T = $" + f"{1.0:.1f}"
    plot_stats_evol(
        results=results,
        xlogscale=True,
        ylogscale=False,
        nice_ticks=True,
        main_axis=main_axis,
        secondary_axis=secondary_axis,
        main_axis_ref_line=0.0,
        figsize=(3.5, 3.6),
        uncertainty=True,
        title=title,
        xlabel=r"$\sigma_S$",
        lw=LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        markersize=MARKERSIZE,
        ncols_legend=2,
        legend_y_pad=1.25,
        figures_dir=figures_dir(args),
        filename="02_sigma_influence",
        filename_suffix=f"_eps_{args.eps:.1f}",
        verbose=True,
    )


def _compute_eps_infl(args):
    set_random_state(seed=SEED)
    print("[INFO] Running optimal sample sharing experiment...", flush=True)
    kwargs = dict(d=D, l_eps=L_EPS, lbda_s=LBDA_DEFAULT, alpha=LCB_ALPHA)
    return _parallel([delayed(run_one_eps_infl)(**kwargs) for _ in range(N_RUNS)])


def _plot_eps_infl(args, results):
    main_axis = [r"$\Delta^{\star}$"]
    secondary_axis = [r"$\hat{n}$"]
    title = r"$\sigma_T = \sigma_S = $" + f"{SIGMA:.2f}"
    plot_stats_evol(
        results=results,
        xlogscale=True,
        ylogscale=False,
        nice_ticks=True,
        main_axis=main_axis,
        secondary_axis=secondary_axis,
        main_axis_ref_line=0.0,
        figsize=(3.5, 3.6),
        uncertainty=True,
        title=title,
        xlabel=r"$\varepsilon$",
        lw=LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        markersize=MARKERSIZE,
        ncols_legend=2,
        legend_y_pad=1.25,
        figures_dir=figures_dir(args),
        filename="03_epsilon_influence",
        filename_suffix=f"_sigma_{SIGMA:.2f}",
        verbose=True,
    )


def _single_regr_payload(args, *, noise="gaussian", lognormal_shape=1.0):
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
        noise=noise,
        lognormal_shape=lognormal_shape,
    )
    X_t, y_t = data["target_train"]
    X_t_val, y_t_val = data["target_val"]
    X_t_test, y_t_test = data["target_test"]
    l_X_s, l_y_s = data["source_train"]
    theta_t = data["meta"]["theta_t"]
    theta_s = data["meta"]["theta_s"]
    return data, X_t, y_t, X_t_val, y_t_val, X_t_test, y_t_test, l_X_s, l_y_s, theta_t, theta_s


def _compute_regr_perf(args, *, noise="gaussian", lognormal_shape=1.0, start_message=True):
    set_random_state(seed=SEED)
    data, X_t, y_t, X_t_val, y_t_val, X_t_test, y_t_test, l_X_s, l_y_s, theta_t, theta_s = _single_regr_payload(
        args,
        noise=noise,
        lognormal_shape=lognormal_shape,
    )
    if noise == LOGNORMAL_NOISE:
        X_s = np.vstack(l_X_s)
        y_s = np.concatenate(l_y_s)
        _print_noise_check(X_t, y_t, X_s, y_s, theta_t, theta_s)
    if start_message:
        print("[INFO] Running optimal sample sharing experiment...", flush=True)

    delayed_runs = []
    for r in range(N_RUNS):
        delayed_runs.append(delayed(run_one_regr_perf_cmp)(
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
            noise=noise,
            lognormal_shape=lognormal_shape,
        ))
    return _parallel(delayed_runs)


def _plot_regr_perf(args, results, *, number: str, filename_error: str, filename_mse: str, title: str, start: int = 0):
    main_axis = _regression_main_axis(include_oracle=True)
    uncertainty = False
    uncertainty_style = "band"
    plot_stats_evol(
        results=[res["error"] for res in results],
        xlogscale=True,
        ylogscale=True,
        start=start,
        main_axis=main_axis,
        secondary_axis=None,
        figsize=(3.75, 4.5),
        uncertainty=uncertainty,
        uncertainty_style=uncertainty_style,
        title=title,
        xlabel=r"$n_T$",
        lw=LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        markersize=MARKERSIZE,
        ncols_legend=2,
        legend_y_pad=1.35,
        figures_dir=figures_dir(args),
        filename=filename_error,
        filename_suffix=f"_eps_{args.eps:.1f}",
        ylim=ylimit_for(args, "error", [res["error"] for res in results], main_axis, logscale=True),
        verbose=True,
    )
    plot_stats_evol(
        results=[res["mse"] for res in results],
        xlogscale=True,
        ylogscale=True,
        start=start,
        main_axis=main_axis,
        secondary_axis=None,
        figsize=(3.75, 4.5),
        uncertainty=uncertainty,
        uncertainty_style=uncertainty_style,
        title=title,
        xlabel=r"$n_T$",
        lw=LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        markersize=MARKERSIZE,
        ncols_legend=2,
        legend_y_pad=1.35,
        figures_dir=figures_dir(args),
        filename=filename_mse,
        filename_suffix=f"_eps_{args.eps:.1f}",
        ylim=ylimit_for(args, "mse", [res["mse"] for res in results], main_axis, logscale=True),
        verbose=True,
    )


def _compute_regr_perf_gaussian(args):
    return _compute_regr_perf(args)


def _plot_regr_perf_gaussian(args, results):
    title = r"$\varepsilon  =  $" + f"{args.eps:.1f}" + r"$\,\cdot\,\|\theta_T^{\star}\|_2$"
    _plot_regr_perf(
        args,
        results,
        number="04",
        filename_error="04_performance_error_comparison",
        filename_mse="04_performance_mse_comparison",
        title=title,
    )


def _compute_real_regr(args):
    set_random_state(seed=SEED)
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

    print(
        "[INFO] Real benchmark settings: "
        f"dataset={args.dataset_name}, "
        f"n_runs={REAL_N_RUNS}, "
        f"n_jobs={N_JOBS}, "
        f"n_t={X_t_train.shape[0]}, "
        f"n_s={sum(X.shape[0] for X in l_X_s)}, "
        f"d={X_t_train.shape[1]}, "
        f"chunk_size={CHUNK_SIZE}",
        flush=True,
    )
    print("[INFO] Running optimal sample sharing experiment...", flush=True)

    main_axis = [
        "Target-only",
        "Mixed",
        "Hierarchical",
        "Zhang et al, 2025",
        "Chen et al, 2014",
        "Obst et al, 2021",
        "Proposed",
    ]
    if args.dataset_name == "email" or args.dataset_name.endswith("_chemberta"):
        main_axis.remove("Obst et al, 2021")
    min_n_samples_target = 20 if args.dataset_name.endswith("_chemberta") else 100

    alpha = 0.0 if args.dataset_name.endswith("_chemberta") else LCB_ALPHA

    delayed_runs = []
    for r in range(REAL_N_RUNS):
        delayed_runs.append(delayed(run_one_regr_perf_cmp)(
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
            alpha=alpha,
            chunk_size=CHUNK_SIZE,
            seed=r,
            synthetic_data=False,
            min_n_samples_target=min_n_samples_target,
            method_names=main_axis,
        ))
    return _parallel(delayed_runs)


def _real_main_axis(dataset_name: str) -> list[str]:
    main_axis = [
        "Target-only",
        "Mixed",
        "Hierarchical",
        "Zhang et al, 2025",
        "Chen et al, 2014",
        "Obst et al, 2021",
        "Proposed",
    ]
    if dataset_name == "email" or dataset_name.endswith("_chemberta"):
        main_axis.remove("Obst et al, 2021")
    return main_axis


def _plot_real_regr(args, results):
    main_axis = _real_main_axis(args.dataset_name)
    q = None
    if args.dataset_name.endswith("_chemberta"):
        q = args.real_ylim_quantile if args.real_ylim_quantile is not None else REAL_YLIM_QUANTILE_DEFAULT

    error_results = [res["error"] for res in results]
    mse_results = [res["mse"] for res in results]
    error_ylim = (
        tuple(float(x) for x in args.real_error_ylim)
        if args.real_error_ylim is not None
        else ylimit_for(args, "error", error_results, main_axis, default_quantile=q)
    )
    mse_ylim = (
        tuple(float(x) for x in args.real_mse_ylim)
        if args.real_mse_ylim is not None
        else ylimit_for(args, "mse", mse_results, main_axis, default_quantile=q)
    )
    plot_stats_evol(
        results=error_results,
        xlogscale=True,
        ylogscale=False,
        main_axis=main_axis,
        secondary_axis=None,
        figsize=(3.75, 4.5),
        uncertainty=False,
        title=args.dataset_name,
        xlabel=r"$n_T$",
        lw=LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        markersize=MARKERSIZE,
        ncols_legend=2,
        legend_y_pad=1.25,
        figures_dir=figures_dir(args),
        filename="05_performance_error_comparison",
        filename_suffix=f"_dataset_{args.dataset_name}",
        ylim=error_ylim,
        verbose=True,
    )
    plot_stats_evol(
        results=mse_results,
        xlogscale=True,
        ylogscale=False,
        main_axis=main_axis,
        secondary_axis=None,
        figsize=(3.75, 4.5),
        uncertainty=False,
        title=args.dataset_name,
        xlabel=r"$n_T$",
        lw=LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        markersize=MARKERSIZE,
        ncols_legend=2,
        legend_y_pad=1.25,
        figures_dir=figures_dir(args),
        filename="05_performance_mse_comparison",
        filename_suffix=f"_dataset_{args.dataset_name}",
        ylim=mse_ylim,
        verbose=True,
    )


def _compute_lbda_infl(args):
    set_random_state(seed=SEED)
    sigma_t = SIGMA
    sigma_s = 0.5 * SIGMA
    eps = EPS[0] if isinstance(EPS, list) else EPS
    data = synth_regr_data(
        n_t_train=N_T,
        n_t_val=max(1, N_T // 50),
        n_t_test=max(1, N_T // 50),
        n_s_train=N_S,
        n_s_val=max(1, N_S // 50),
        n_s_test=max(1, N_S // 50),
        d=D,
        eps=eps,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
    )

    X_t, y_t = data["target_train"]
    X_t_val, y_t_val = data["target_val"]
    X_t_test, y_t_test = data["target_test"]
    l_X_s, l_y_s = data["source_train"]
    theta_t = data["meta"]["theta_t"]
    theta_s = data["meta"]["theta_s"]

    print("[INFO] Running optimal sample sharing experiment...", flush=True)
    delayed_runs = []
    for r in range(N_RUNS):
        delayed_runs.append(delayed(run_one_lbda_infl)(
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
            alpha=LCB_ALPHA,
            seed=r,
            synthetic_data=True,
        ))
    return _parallel(delayed_runs)


def _plot_lbda_infl(args, results):
    main_axis = [r"$\Delta^{\star}$"]
    secondary_axis = [r"$\hat{n}$"]
    plot_stats_evol(
        results=results,
        xlogscale=True,
        ylogscale=False,
        nice_ticks=True,
        main_axis=main_axis,
        secondary_axis=secondary_axis,
        main_axis_ref_line=0.0,
        figsize=(3.5, 3.6),
        uncertainty=True,
        title="",
        xlabel=r"$\lambda_S$",
        lw=LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        markersize=MARKERSIZE,
        ncols_legend=2,
        legend_y_pad=1.25,
        figures_dir=figures_dir(args),
        filename="06_lbda_influence",
        filename_suffix="",
        verbose=True,
    )


def _compute_intro_regr(args):
    set_random_state(seed=SEED)
    sigma_t = SIGMA
    sigma_s = SIGMA
    data = synth_regr_data(
        n_t_train=N_T,
        n_t_val=max(1, N_T // 50),
        n_t_test=max(1, N_T // 50),
        n_s_train=N_S,
        n_s_val=max(1, N_S // 50),
        n_s_test=max(1, N_S // 50),
        d=D,
        eps=INTRO_EPS,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
    )

    X_t, y_t = data["target_train"]
    X_t_val, y_t_val = data["target_val"]
    X_t_test, y_t_test = data["target_test"]
    l_X_s, l_y_s = data["source_train"]
    theta_t = data["meta"]["theta_t"]
    theta_s = data["meta"]["theta_s"]

    print("[INFO] Running optimal sample sharing experiment...", flush=True)
    delayed_runs = []
    for r in range(N_RUNS):
        delayed_runs.append(delayed(run_one_regr_perf_cmp)(
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
            lbda_s=LBDA_DEFAULT,
            alpha=LCB_ALPHA,
            chunk_size=CHUNK_SIZE,
            min_n_samples_target=INTRO_N_T_MIN,
            seed=r,
            synthetic_data=True,
        ))
    return _parallel(delayed_runs)


def _plot_intro_regr(args, results):
    main_axis = ["Target-only", "Mixed", "Proposed", "Proposed (oracle)"]
    plot_stats_evol(
        results=[res["error"] for res in results],
        xlogscale=True,
        ylogscale=True,
        main_axis=main_axis,
        secondary_axis=None,
        figsize=INTRO_FIGSIZE,
        uncertainty=False,
        title="",
        xlabel=r"$n_T$",
        lw=INTRO_LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        markersize=MARKERSIZE,
        ncols_legend=2,
        legend_y_pad=1.25,
        figures_dir=figures_dir(args),
        filename="07_intro_illus",
        filename_suffix=f"_eps_{INTRO_EPS:.2f}",
        ylim=ylimit_for(args, "error", [res["error"] for res in results], main_axis, logscale=True),
        verbose=True,
    )


def _compute_regr_multi(args):
    set_random_state(seed=SEED)
    l_eps = [args.eps, args.eps / 2.0, args.eps / 3.0]
    kwargs = dict(
        n_t_train=N_T,
        n_t_val=max(1, N_T // 50),
        n_t_test=max(1, N_T // 50),
        n_s_train=max(1, N_S // len(l_eps)),
        n_s_val=max(1, N_S // (50 * len(l_eps))),
        n_s_test=max(1, N_S // (50 * len(l_eps))),
        d=D,
        sigma_t=SIGMA,
        sigma_s=SIGMA,
    )
    l_X_s, l_y_s = [], []
    dataset = None
    for eps in l_eps:
        kwargs["eps"] = eps
        dataset = synth_regr_data(**kwargs)
        l_X_s.append(dataset["source_train"][0][0])
        l_y_s.append(dataset["source_train"][1][0])

    X_t_train, y_t_train = dataset["target_train"]
    X_t_val, y_t_val = dataset["target_val"]
    X_t_test, y_t_test = dataset["target_test"]
    theta_t = dataset["meta"]["theta_t"]
    theta_s = dataset["meta"]["theta_s"]

    print("[INFO] Running optimal sample sharing experiment...", flush=True)
    delayed_runs = []
    for r in range(N_RUNS):
        delayed_runs.append(delayed(run_one_regr_perf_cmp)(
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
            sigma_t=SIGMA,
            sigma_s=SIGMA,
            lbda_s=LBDA_DEFAULT,
            alpha=LCB_ALPHA,
            chunk_size=CHUNK_SIZE,
            seed=r,
            synthetic_data=True,
        ))
    return _parallel(delayed_runs)


def _plot_regr_multi(args, results):
    title = _multi_source_title([args.eps, args.eps / 2.0, args.eps / 3.0])
    _plot_regr_perf(
        args,
        results,
        number="08",
        filename_error="08_performance_error_comparison_multi_source",
        filename_mse="08_performance_mse_comparison_multi_source",
        title=title,
        start=START_PLOT,
    )


def _compute_classif_perf(args):
    set_random_state(seed=SEED)
    data = synth_classif_data(
        n_t_train=N_T,
        n_t_val=max(1, N_T // 50),
        n_t_test=max(1, N_T // 50),
        n_s_train=N_S,
        n_s_val=max(1, N_S // 50),
        n_s_test=max(1, N_S // 50),
        d=D,
        eps=args.eps,
        sigma_t=2 * SIGMA,
        sigma_s=SIGMA,
    )

    X_t, y_t = data["target_train"]
    X_t_val, y_t_val = data["target_val"]
    X_t_test, y_t_test = data["target_test"]
    l_X_s, l_y_s = data["source_train"]
    theta_t = data["meta"]["theta_t"]
    theta_s = data["meta"]["theta_s"]

    delayed_runs = []
    for r in range(N_RUNS):
        delayed_runs.append(delayed(run_one_classif_perf_cmp)(
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
        ))
    return _parallel(delayed_runs)


def _plot_classif_perf(args, results):
    main_axis = ["Target-only", "Mixed", "Hierarchical", "Proposed", "Proposed (oracle)"]
    title = r"$\varepsilon  =  $" + f"{args.eps:.1f}" + r"$\,\cdot\,\|\theta_T^{\star}\|_2$"
    probit_results = [res["probit"] for res in results]
    acc_results = [res["acc"] for res in results]
    plot_stats_evol(
        results=probit_results,
        xlogscale=True,
        ylogscale=False,
        main_axis=main_axis,
        secondary_axis=None,
        figsize=(3.75, 4.5),
        uncertainty=False,
        title=title,
        xlabel=r"$n_T$",
        lw=LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        markersize=MARKERSIZE,
        ncols_legend=2,
        legend_y_pad=1.25,
        figures_dir=figures_dir(args),
        filename="09_performance_classif_probit",
        filename_suffix=f"_eps_{args.eps:.1f}",
        ylim=ylimit_for(args, "probit", probit_results, main_axis),
        verbose=True,
    )
    plot_stats_evol(
        results=acc_results,
        xlogscale=True,
        ylogscale=False,
        main_axis=main_axis,
        secondary_axis=None,
        figsize=(3.75, 4.5),
        uncertainty=False,
        title=title,
        xlabel=r"$n_T$",
        lw=LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        markersize=MARKERSIZE,
        ncols_legend=2,
        legend_y_pad=1.25,
        figures_dir=figures_dir(args),
        filename="09_performance_classif_acc",
        filename_suffix=f"_eps_{args.eps:.1f}",
        ylim=ylimit_for(args, "acc", acc_results, main_axis),
        verbose=True,
    )


def _compute_classif_multi(args):
    set_random_state(seed=SEED)
    l_eps = [args.eps, args.eps / 2.0, args.eps / 3.0]
    kwargs = dict(
        n_t_train=N_T,
        n_t_val=max(1, N_T // 50),
        n_t_test=max(1, N_T // 50),
        n_s_train=max(1, N_S // len(l_eps)),
        n_s_val=max(1, N_S // 50),
        n_s_test=max(1, N_S // (50 * len(l_eps))),
        d=D,
        sigma_t=SIGMA,
        sigma_s=SIGMA,
    )

    l_X_s, l_y_s = [], []
    dataset = None
    for eps in l_eps:
        kwargs["eps"] = eps
        dataset = synth_classif_data(**kwargs)
        l_X_s.append(dataset["source_train"][0][0])
        l_y_s.append(dataset["source_train"][1][0])

    X_t_train, y_t_train = dataset["target_train"]
    X_t_val, y_t_val = dataset["target_val"]
    X_t_test, y_t_test = dataset["target_test"]
    theta_t = dataset["meta"]["theta_t"]
    theta_s = dataset["meta"]["theta_s"]

    print("[INFO] Running optimal sample sharing experiment...", flush=True)
    delayed_runs = []
    for r in range(N_RUNS):
        delayed_runs.append(delayed(run_one_classif_perf_cmp)(
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
        ))
    return _parallel(delayed_runs)


def _plot_classif_multi(args, results):
    main_axis = ["Target-only", "Mixed", "Hierarchical", "Proposed", "Proposed (oracle)"]
    title = _multi_source_title([args.eps, args.eps / 2.0, args.eps / 3.0])
    probit_results = [res["probit"] for res in results]
    acc_results = [res["acc"] for res in results]
    plot_stats_evol(
        results=probit_results,
        xlogscale=True,
        ylogscale=False,
        start=START_PLOT,
        main_axis=main_axis,
        secondary_axis=None,
        figsize=(3.75, 4.5),
        uncertainty=False,
        title=title,
        xlabel=r"$n_T$",
        lw=LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        markersize=MARKERSIZE,
        ncols_legend=2,
        legend_y_pad=1.35,
        figures_dir=figures_dir(args),
        filename="10_performance_classif_probit_multi_source",
        filename_suffix=f"_eps_{args.eps:.1f}",
        ylim=ylimit_for(args, "probit", probit_results, main_axis),
        verbose=True,
    )
    plot_stats_evol(
        results=acc_results,
        xlogscale=True,
        ylogscale=False,
        start=START_PLOT,
        main_axis=main_axis,
        secondary_axis=None,
        figsize=(3.75, 4.5),
        uncertainty=False,
        title=title,
        xlabel=r"$n_T$",
        lw=LW,
        alpha=ALPHA,
        fontsize=FONTSIZE,
        markersize=MARKERSIZE,
        ncols_legend=2,
        legend_y_pad=1.35,
        figures_dir=figures_dir(args),
        filename="10_performance_classif_acc_multi_source",
        filename_suffix=f"_eps_{args.eps:.1f}",
        ylim=ylimit_for(args, "acc", acc_results, main_axis),
        verbose=True,
    )


def _compute_delta_classif(args):
    set_random_state(seed=SEED)
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

    delayed_runs = []
    for r in range(N_RUNS):
        delayed_runs.append(delayed(run_one_delta_illus)(
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
        ))
    return _parallel(delayed_runs)


def _plot_delta_classif(args, results):
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
        figures_dir=figures_dir(args),
        filename="11_delta_classif_illus",
        filename_suffix=f"_eps_{args.eps:.1f}",
        verbose=True,
    )


def _compute_regr_lognormal(args):
    return _compute_regr_perf(args, noise=LOGNORMAL_NOISE, lognormal_shape=LOGNORMAL_SHAPE)


def _plot_regr_lognormal(args, results):
    title = (r"$\varepsilon  =  $" + f"{args.eps:.1f}" + r"$\,\cdot\,\|\theta_T^{\star}\|_2$" + "\ncentered log-normal residuals")
    _plot_regr_perf(
        args,
        results,
        number="12",
        filename_error="12_performance_error_comparison_lognormal_noise",
        filename_mse="12_performance_mse_comparison_lognormal_noise",
        title=title,
        start=START_PLOT,
    )


_TASKS = {
    "bench_01_delta_regr_illus_synth": BenchmarkTask(
        key="bench_01_delta_regr_illus_synth",
        description="Regression collaboration-criteria illustration.",
        add_args=_add_eps_arg,
        result_name=lambda args: _eps_result("01_delta_regr_illus_synth", args),
        params=_eps_params,
        compute=_compute_delta_regr,
        plot=_plot_delta_regr,
    ),
    "bench_02_sigma_infl_synth": BenchmarkTask(
        key="bench_02_sigma_infl_synth",
        description="Regression sigma influence benchmark.",
        add_args=_add_eps_arg,
        result_name=lambda args: _eps_result("02_sigma_infl_synth", args),
        params=_eps_params,
        compute=_compute_sigma_infl,
        plot=_plot_sigma_infl,
    ),
    "bench_03_eps_infl_synth": BenchmarkTask(
        key="bench_03_eps_infl_synth",
        description="Regression epsilon influence benchmark.",
        add_args=_add_no_args,
        result_name=lambda args: _no_arg_result(f"03_eps_infl_synth_sigma_{SIGMA:.2f}"),
        params=_no_params,
        compute=_compute_eps_infl,
        plot=_plot_eps_infl,
    ),
    "bench_04_perf_regr_synth": BenchmarkTask(
        key="bench_04_perf_regr_synth",
        description="Synthetic regression performance comparison.",
        add_args=_add_eps_arg,
        result_name=lambda args: _eps_result("04_perf_regr_synth", args),
        params=_eps_params,
        compute=_compute_regr_perf_gaussian,
        plot=_plot_regr_perf_gaussian,
    ),
    "bench_05_perf_regr_real": BenchmarkTask(
        key="bench_05_perf_regr_real",
        description="Real-data regression performance comparison.",
        add_args=_add_dataset_arg,
        result_name=lambda args: _dataset_result("05_perf_regr_real", args),
        params=_dataset_params,
        compute=_compute_real_regr,
        plot=_plot_real_regr,
    ),
    "bench_06_lbda_infl_synth": BenchmarkTask(
        key="bench_06_lbda_infl_synth",
        description="Regression lambda influence benchmark.",
        add_args=_add_no_args,
        result_name=lambda args: _no_arg_result("06_lbda_infl_synth"),
        params=_no_params,
        compute=_compute_lbda_infl,
        plot=_plot_lbda_infl,
    ),
    "bench_07_intro_regr_illus": BenchmarkTask(
        key="bench_07_intro_regr_illus",
        description="Introductory regression illustration.",
        add_args=_add_no_args,
        result_name=lambda args: _no_arg_result(f"07_intro_regr_illus_eps_{INTRO_EPS:.2f}"),
        params=_no_params,
        compute=_compute_intro_regr,
        plot=_plot_intro_regr,
    ),
    "bench_08_perf_regr_multi_synth": BenchmarkTask(
        key="bench_08_perf_regr_multi_synth",
        description="Synthetic multi-source regression performance comparison.",
        add_args=_add_eps_arg,
        result_name=lambda args: _eps_result("08_perf_regr_multi_synth", args),
        params=_eps_params,
        compute=_compute_regr_multi,
        plot=_plot_regr_multi,
    ),
    "bench_09_perf_classif_synth": BenchmarkTask(
        key="bench_09_perf_classif_synth",
        description="Synthetic classification performance comparison.",
        add_args=_add_classif_eps_arg,
        result_name=lambda args: _eps_result("09_perf_classif_synth", args),
        params=_eps_params,
        compute=_compute_classif_perf,
        plot=_plot_classif_perf,
    ),
    "bench_10_perf_classif_multi_synth": BenchmarkTask(
        key="bench_10_perf_classif_multi_synth",
        description="Synthetic multi-source classification performance comparison.",
        add_args=_add_eps_arg,
        result_name=lambda args: _eps_result("10_perf_classif_multi_synth", args),
        params=_eps_params,
        compute=_compute_classif_multi,
        plot=_plot_classif_multi,
    ),
    "bench_11_delta_classif_illus_synth": BenchmarkTask(
        key="bench_11_delta_classif_illus_synth",
        description="Classification collaboration-criteria illustration.",
        add_args=_add_eps_arg,
        result_name=lambda args: _eps_result("11_delta_classif_illus_synth", args),
        params=_eps_params,
        compute=_compute_delta_classif,
        plot=_plot_delta_classif,
    ),
    "bench_12_perf_regr_lognormal_synth": BenchmarkTask(
        key="bench_12_perf_regr_lognormal_synth",
        description="Synthetic regression comparison with centered log-normal residuals.",
        add_args=_add_eps_arg,
        result_name=lambda args: _eps_result("12_perf_regr_lognormal_synth", args),
        params=_eps_params,
        compute=_compute_regr_lognormal,
        plot=_plot_regr_lognormal,
    ),
}


TASKS = _TASKS


def _unknown_experiment_error(task_key: str) -> ValueError:
    """Build a clear unknown-experiment error."""
    available = "\n".join(f"- {key}" for key in sorted(TASKS))
    return ValueError(
        f"Unknown experiment: {task_key}\n"
        "Available experiments:\n"
        f"{available}\n"
        "Use `python -m benchmark_tasks list`."
    )


def get_task(task_key: str) -> ExperimentTask:
    """Return an executable experiment task."""
    try:
        return TASKS[task_key]
    except KeyError as exc:
        raise _unknown_experiment_error(task_key) from exc


def main(task_key: str, argv: list[str] | None = None, seed: int | None = None) -> None:
    """Run one benchmark task."""
    global SEED
    task = get_task(task_key)
    if seed is not None:
        SEED = int(seed)
    parser = argparse.ArgumentParser(description=task.description)
    add_mode_subparsers(parser, task.add_args)
    args = parser.parse_args(argv)
    run_task_cli(
        args,
        result_name=task.result_name(args),
        params=task.params(args),
        compute_fn=task.compute,
        plot_fn=task.plot,
    )
