"""Shared helpers for compute/plot benchmark entry points."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Callable
from joblib import dump, load
import numpy as np
from dyntr.utils import format_duration
if __package__:
    from .experiments import get_config
else:
    from experiments import get_config


SCHEMA_VERSION = 1
DEFAULT_RESULTS_DIR = "_results"


def add_common_args(parser, *, include_plot_args: bool = False) -> None:
    """Add result/plot arguments shared by benchmark scripts."""
    parser.add_argument(
        "--results-dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing serialized benchmark curves.",
    )
    if include_plot_args:
        parser.add_argument(
            "--figures-dir",
            type=str,
            default=None,
            help="Directory where figures are written. Defaults to plotting.figures_dir.",
        )
        parser.add_argument(
            "--ylim-quantile",
            type=float,
            default=None,
            help="Override quantile-based y-limits for plots that support them.",
        )
        parser.add_argument(
            "--real-ylim-quantile",
            type=float,
            default=None,
            help="Override quantile-based y-limits for real ChemBERTa plots.",
        )
        parser.add_argument(
            "--real-error-ylim",
            nargs=2,
            type=float,
            metavar=("LOW", "HIGH"),
            default=None,
            help="Manual y-limits for real-data error plots.",
        )
        parser.add_argument(
            "--real-mse-ylim",
            nargs=2,
            type=float,
            metavar=("LOW", "HIGH"),
            default=None,
            help="Manual y-limits for real-data MSE plots.",
        )
        parser.add_argument(
            "--error-ylim",
            nargs=2,
            type=float,
            metavar=("LOW", "HIGH"),
            default=None,
            help="Manual y-limits for error plots.",
        )
        parser.add_argument(
            "--mse-ylim",
            nargs=2,
            type=float,
            metavar=("LOW", "HIGH"),
            default=None,
            help="Manual y-limits for MSE plots.",
        )
        parser.add_argument(
            "--probit-ylim",
            nargs=2,
            type=float,
            metavar=("LOW", "HIGH"),
            default=None,
            help="Manual y-limits for probit/classification-loss plots.",
        )
        parser.add_argument(
            "--acc-ylim",
            nargs=2,
            type=float,
            metavar=("LOW", "HIGH"),
            default=None,
            help="Manual y-limits for accuracy plots.",
        )


def add_mode_subparsers(parser, add_task_args: Callable) -> None:
    """Create explicit compute and plot subcommands for one benchmark script."""
    subparsers = parser.add_subparsers(dest="mode", required=True)

    compute_parser = subparsers.add_parser("compute", help="Run computation and store curves.")
    add_task_args(compute_parser)
    add_common_args(compute_parser, include_plot_args=False)

    plot_parser = subparsers.add_parser("plot", help="Load stored curves and write figures.")
    add_task_args(plot_parser)
    add_common_args(plot_parser, include_plot_args=True)


def result_path(results_dir: str | Path, result_name: str) -> Path:
    return Path(results_dir) / f"{result_name}.joblib"


def save_result(results_dir: str | Path, result_name: str, *, params: dict, payload) -> Path:
    """Serialize computed curves and small metadata."""
    path = result_path(results_dir, result_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    cfg = get_config()
    config_payload = asdict(cfg) if is_dataclass(cfg) else {}
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "result_name": result_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "config": config_payload,
        "payload": payload,
    }
    dump(bundle, path, compress=3)
    return path


def load_result(results_dir: str | Path, result_name: str):
    path = result_path(results_dir, result_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark result: {path}")
    bundle = load(path)
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported result schema in {path}")
    return bundle


def finite_metric_values(results, main_axis) -> np.ndarray | None:
    names = set(main_axis or [])
    values = []
    for run in results:
        for series in run:
            if names and series.get("name") not in names:
                continue
            metric = np.asarray(series.get("metric", []), dtype=float).ravel()
            metric = metric[np.isfinite(metric)]
            if metric.size:
                values.append(metric)
    if not values:
        return None
    return np.concatenate(values)


def quantile_ylim(results, main_axis, quantile: float, *, logscale: bool = False):
    """Return y-limits from displayed finite values."""
    values = finite_metric_values(results, main_axis)
    if values is None:
        return None

    upper = float(np.quantile(values, quantile))
    if logscale:
        positive = values[values > 0.0]
        if not positive.size:
            return None
        lower_q = max(0.0, 1.0 - quantile)
        lower = float(np.quantile(positive, lower_q))
    else:
        lower = 0.0 if float(np.min(values)) >= 0.0 else float(np.quantile(values, 1.0 - quantile))

    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return None
    return lower, upper


def ylimit_for(args, key: str, results, main_axis, *, default_quantile=None, logscale=False):
    """Resolve manual or quantile y-limits for a plot metric."""
    explicit = getattr(args, f"{key}_ylim", None)
    if explicit is not None:
        return tuple(float(x) for x in explicit)

    quantile = args.ylim_quantile if getattr(args, "ylim_quantile", None) is not None else default_quantile
    if quantile is None:
        return None
    return quantile_ylim(results, main_axis, quantile=float(quantile), logscale=logscale)


def figures_dir(args) -> str:
    return args.figures_dir or get_config().FIGURES_DIR


def run_task_cli(args, *, result_name: str, params: dict, compute_fn: Callable, plot_fn: Callable) -> None:
    t0 = time.time()
    if args.mode == "compute":
        payload = compute_fn(args)
        path = save_result(args.results_dir, result_name, params=params, payload=payload)
        print(f"[INFO] Saved benchmark curves to '{path}'.", flush=True)
    elif args.mode == "plot":
        bundle = load_result(args.results_dir, result_name)
        plot_fn(args, bundle["payload"])
    else:
        raise ValueError(f"Unknown benchmark mode: {args.mode}")

    print(f"[INFO] Command duration: {format_duration(time.time() - t0)}.", flush=True)
