"""Experiment utilities helpers module."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os
import yaml
from typing import List, Tuple, Any, Mapping
import numpy as np
from scipy import special
from dyntr.augmenter import SampleAugmenter
from dyntr.datasets import sample_synthetic_noise, synth_regr_data
from dyntr.datasets._datasets import _generate_thetas
from dyntr.linear import compute_delta_classif, compute_delta_regr, opt_lambda
from dyntr.utils import check_random_state
from dyntr.linear import (
    target_regr,
    full_regr,
    hier_regr,
    dyn_greedy_regr,
    oracle_greedy_regr,
    chen_regr,
    obst_regr,
    zhang_regr,
    target_classif,
    full_classif,
    hier_classif,
    dyn_greedy_classif,
    oracle_greedy_classif,
)

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _ROOT / "benchmark" / "configs" / "config.yml"


@dataclass(frozen=True)
class Config:
    FIGURES_DIR: str
    FONTSIZE: int
    LW: float
    ALPHA: float
    MARKERSIZE: float
    MODEL_ALPHA: float
    N_JOBS: int
    N_RUNS: int
    REAL_N_RUNS: int
    JOBLIB_VERBOSE: int
    SEED: int
    L_EPS: np.ndarray
    L_SIGMA_S: np.ndarray
    N_S: int
    N_T: int
    N_POINTS: int
    SIGMA: float
    D: int
    EPS: list[float]
    STANDARDIZE_REAL: bool
    LBDA_DEFAULT: float
    CHUNK_SIZE: int
    START_PLOT: int


def _logspace(spec: Any) -> np.ndarray:
    if isinstance(spec, Mapping):
        start = float(spec["start"])
        stop = float(spec["stop"])
        num = int(spec["num"])
        return np.logspace(start=start, stop=stop, num=num)
    return np.asarray(spec, dtype=float)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=None)
def get_config(path: str | None = None) -> Config:
    config_path = Path(path or os.environ.get("DYNTR_CONFIG", _DEFAULT_CONFIG)).resolve()
    data = _load_yaml(config_path)

    plotting = data.get("plotting", {})
    launcher = data.get("launcher", {})
    real_cfg = data.get("real", {})
    data_cfg = data.get("data", {})
    bench_cfg = data.get("bench", {})

    figures_dir = str(plotting.get("figures_dir", "_figures"))
    os.makedirs(figures_dir, exist_ok=True)

    return Config(
        FIGURES_DIR=figures_dir,
        FONTSIZE=int(plotting.get("fontsize", 18)),
        LW=float(plotting.get("lw", 5)),
        ALPHA=float(plotting.get("alpha", 0.4)),
        MARKERSIZE=float(plotting.get("markersize", 6)),
        MODEL_ALPHA=float(data.get("model", {}).get("alpha", 0.01)),
        N_JOBS=int(launcher.get("n_jobs", 1)),
        N_RUNS=int(launcher.get("n_runs", 1)),
        REAL_N_RUNS=int(real_cfg.get("n_runs", launcher.get("n_runs", 1))),
        JOBLIB_VERBOSE=int(launcher.get("joblib_verbose", 0)),
        SEED=int(launcher.get("seed", 0)),
        L_EPS=_logspace(data_cfg.get("l_eps", {"start": -3, "stop": 3, "num": 50})),
        L_SIGMA_S=_logspace(data_cfg.get("l_sigma_s", {"start": -3, "stop": 3, "num": 50})),
        N_S=int(data_cfg.get("n_s", 1000)),
        N_T=int(data_cfg.get("n_t", 1000)),
        N_POINTS=int(data_cfg.get("n_points", 20)),
        SIGMA=float(data_cfg.get("sigma", 1.0)),
        D=int(data_cfg.get("d", 50)),
        EPS=list(data_cfg.get("eps", [0.5])),
        STANDARDIZE_REAL=bool(data_cfg.get("standardize_real", True)),
        LBDA_DEFAULT=float(data_cfg.get("lbda_default", 1.0)),
        CHUNK_SIZE=int(bench_cfg.get("chunk_size", 10)),
        START_PLOT=int(bench_cfg.get("start_plot", 2)),
    )


_CFG = get_config()
SIGMA = _CFG.SIGMA
N_S = _CFG.N_S
N_T = _CFG.N_T
N_POINTS = _CFG.N_POINTS

_COLORS = {
    "target": "#4E79A7",
    "source": "#59A14F",
    "mixed": "#B07AA1",
    "proposed": "#F28E2B",
    "oracle": "#E15759",
    "zhang": "#76B7B2",
    "chen": "#9C755F",
    "obst": "#EDC948",
    "hier": "#BAB0AC",
}

_MARKERS = {
    "target": "o",
    "source": "o",
    "mixed": "o",
    "proposed": "X",
    "oracle": "X",
    "zhang": "s",
    "chen": "s",
    "obst": "s",
    "hier": "^",
}


def _ridge(X, y, lbda):
    return np.linalg.solve(X.T @ X + lbda * np.eye(X.shape[1]), X.T @ y)


def _concat_sources(l_X_s, l_y_s):
    X_s = np.vstack(l_X_s)
    y_s = np.concatenate(l_y_s)
    return X_s, y_s


def _regen_thetas(theta_t, theta_s, rng):
    eps = float(np.linalg.norm(theta_s - theta_t))
    d = int(theta_t.shape[0])
    theta_s_new, theta_t_new = _generate_thetas(d, eps, random_state=rng)
    return theta_t_new, theta_s_new


def _sample_response(X, theta, rng, sigma, noise, lognormal_shape):
    residuals = sample_synthetic_noise(
        rng,
        size=X.shape[0],
        sigma=sigma,
        noise=noise,
        lognormal_shape=lognormal_shape,
    )
    return X @ theta + residuals


def run_one_delta_illus(
    n_samples_target: int,
    theta_t: np.ndarray,
    theta_s: np.ndarray,
    X_t_train: np.ndarray,
    y_t_train: np.ndarray,
    X_t_val: np.ndarray,
    y_t_val: np.ndarray,
    X_t_test: np.ndarray,
    y_t_test: np.ndarray,
    l_X_s: list[np.ndarray],
    l_y_s: list[np.ndarray],
    sigma_t: float,
    sigma_s: float,
    lbda_t: float,
    lbda_s: float,
    min_n_samples_target: int = 100,
    seed: int | None = None,
    synthetic_data: bool = True,
    noise: str = "gaussian",
    lognormal_shape: float = 1.0,
) -> List[Tuple[str, str, str, np.ndarray, np.ndarray]]:
    """
    Run the experiment for a vector of source sample sizes and return series for plotting.

    Parameters
    ----------
    theta_t, theta_s : ndarray of shape (d,)
        Ground-truth parameters for datasets target and source.
    X_t_train, y_t_train : ndarray
        Training features matrices and targets.
    X_t_val, y_t_val : ndarray
        Validation features matrices and targets.
    X_t_test, y_t_test : ndarray
        Test features matrices and targets.
    l_X_s, l_y_s: list of ndarray
        List of source training features matrices and targets.
    sigma_t, sigma_s : float
        Noise stds.
    lbda_t, lbda_s : float
        Ridge penalties.
    seed : int
        Random seed.
    synthetic_data : bool
        Flag for synthetic data.
    noise : {"gaussian", "lognormal"}
        Synthetic residual distribution used when ``synthetic_data`` is True.
    lognormal_shape : float
        Shape parameter for log-normal residuals.

    Returns
    -------
    list of tuples
        [(name, color, linestyle, x_values, y_values), ...]
    """
    # assert proper dimenssions
    for X_s in l_X_s:
        assert X_s.shape[1] == l_X_s[0].shape[1]
    assert X_t_train.shape[1] == l_X_s[0].shape[1]
    assert X_t_val.shape[1] == l_X_s[0].shape[1]
    assert X_t_test.shape[1] == l_X_s[0].shape[1]

    rng = check_random_state(seed)
    _, d = l_X_s[0].shape
    min_n_samples_target = max(2 * d, min_n_samples_target)
    nn_samples_target = np.logspace(
        np.log10(min_n_samples_target),
        np.log10(n_samples_target + 1),
        num=N_POINTS,
        dtype=int,
    )
    nn_samples_target = np.unique(nn_samples_target)

    # copy to avoid modifying the original data in-place (e.g., shuffling)
    X_t_train = X_t_train.copy()
    y_t_train = y_t_train.copy()
    y_t_val = y_t_val.copy()
    X_t_test = X_t_test.copy()
    y_t_test = y_t_test.copy()
    l_X_s = [X_s.copy() for X_s in l_X_s]
    l_y_s = [y_s.copy() for y_s in l_y_s]

    # sanityze feature matrixs
    X_t_train = np.array(X_t_train, copy=True, order="C")
    X_t_val = np.array(X_t_val, copy=True, order="C")
    X_t_test = np.array(X_t_test, copy=True, order="C")
    l_X_s = [np.array(X_s, copy=True, order="C") for X_s in l_X_s]

    if synthetic_data:
        theta_t, theta_s = _regen_thetas(theta_t, theta_s, rng)
        y_t_train = _sample_response(X_t_train, theta_t, rng, sigma_t, noise, lognormal_shape)
        y_t_val = _sample_response(X_t_val, theta_t, rng, sigma_t, noise, lognormal_shape)
        y_t_test = _sample_response(X_t_test, theta_t, rng, sigma_t, noise, lognormal_shape)
        l_y_s = [
            _sample_response(X_s, theta_s, rng, sigma_s, noise, lognormal_shape)
            for X_s in l_X_s
        ]
    else:
        # For real data, we just shuffle the samples to have variability across runs.
        idx_t = rng.permutation(X_t_train.shape[0])
        X_t_train = X_t_train[idx_t]
        y_t_train = y_t_train[idx_t]
        for i, (X_s, y_s) in enumerate(zip(l_X_s, l_y_s)):
            idx_s = rng.permutation(X_s.shape[0])
            l_X_s[i], l_y_s[i] = X_s[idx_s], y_s[idx_s]

    # sanityze targets
    y_t_train = np.array(y_t_train, copy=True, order="C")
    y_t_val = np.array(y_t_val, copy=True, order="C")
    y_t_test = np.array(y_t_test, copy=True, order="C")
    l_y_s = [np.array(y_s, copy=True, order="C") for y_s in l_y_s]

    # for now, concatenate all source datasets
    X_s, y_s = _concat_sources(l_X_s, l_y_s)

    lbda_t = opt_lambda(X_t_train, y_t_train, X_t_val, y_t_val)
    eye = np.eye(d)

    G_t = X_t_train.T @ X_t_train
    b_t = X_t_train.T @ y_t_train
    A_t_inv = np.linalg.inv(G_t + lbda_t * eye)
    theta_t_hat = A_t_inv @ b_t

    l_Delta_hat, l_Delta_true = [], []
    y_t_unique = np.unique(y_t_train)

    is_binary = np.all(np.isin(y_t_unique, [-1.0, 1.0])) or np.all(np.isin(y_t_unique, [0.0, 1.0]))
    y_t_val_classif = y_t_val
    if is_binary and np.all(np.isin(y_t_unique, [0.0, 1.0])):
        y_t_val_classif = np.where(y_t_val > 0.0, 1.0, -1.0)

    for i in nn_samples_target:
        Xi_s = X_s[:i]
        yi_s = y_s[:i]

        G_s = Xi_s.T @ Xi_s
        b_s = Xi_s.T @ yi_s

        A_s_inv = np.linalg.inv(G_s + lbda_s * eye)
        theta_s_hat = A_s_inv @ b_s

        A_c_inv = np.linalg.inv(G_s + G_t + (lbda_s + lbda_t) * eye)
        theta_c_hat = A_c_inv @ (b_s + b_t)

        if is_binary:
            Delta_hat = compute_delta_classif(
                theta_t=theta_t_hat,
                theta_c=theta_c_hat,
                X_val=X_t_val,
                y_val=y_t_val_classif,
            )
        else:
            Delta_hat = compute_delta_regr(
                theta_s=theta_s_hat,
                theta_t=theta_t_hat,
                G_s=G_s,
                G_t=G_t,
                A_t_inv=A_t_inv,
                A_s_inv=A_s_inv,
                A_c_inv=A_c_inv,
                X_val=X_t_val,
                sigma_s=sigma_s,
                sigma_t=sigma_t,
                lbda_t=lbda_t,
                lbda_s=lbda_s,
            )

        Delta_true = float(np.linalg.norm(X_t_val @ (theta_t_hat - theta_t)) ** 2 - np.linalg.norm(X_t_val @ (theta_c_hat - theta_t)) ** 2)
        l_Delta_hat.append(Delta_hat)
        l_Delta_true.append(Delta_true)

    return [
        {
            "name": r"$\hat{\Delta}$",
            "color": _COLORS["proposed"],
            "style": "solid",
            "marker": "o",
            "n_samples": nn_samples_target,
            "metric": np.asarray(l_Delta_hat),
        },
        {
            "name": r"$\Delta^{\star}$",
            "color": _COLORS["target"],
            "style": "solid",
            "marker": "o",
            "n_samples": nn_samples_target,
            "metric": np.asarray(l_Delta_true),
        },
    ]


def run_one_regr_perf_cmp(
    n_samples_target: int,
    theta_t: np.ndarray,
    theta_s: np.ndarray,
    X_t_train: np.ndarray,
    y_t_train: np.ndarray,
    X_t_val: np.ndarray,
    y_t_val: np.ndarray,
    X_t_test: np.ndarray,
    y_t_test: np.ndarray,
    l_X_s: list[np.ndarray],
    l_y_s: list[np.ndarray],
    sigma_t: float,
    sigma_s: float,
    lbda_s: float,
    alpha: float = 0.1,
    min_n_samples_target: int = 100,
    chunk_size: int = 5,
    seed: int | None = None,
    synthetic_data: bool = True,
    noise: str = "gaussian",
    lognormal_shape: float = 1.0,
    method_names: list[str] | None = None,
) -> List[Tuple[str, str, str, np.ndarray]]:
    """Run the experiment for a vector of sample sizes and return series for plotting.

    Parameters
    ----------
    n_samples_target : int
        Max number of samples for dataset target.
    theta_t, theta_s : ndarray of shape (d,)
        Ground-truth parameters for datasets target and source.
    X_t_train, y_t_train : ndarray
        Training features matrices and targets.
    X_t_val, y_t_val : ndarray
        Validation features matrices and targets.
    X_t_test, y_t_test : ndarray
        Test features matrices and targets.
    l_X_s, l_y_s: list of ndarray
        List of source training features matrices and targets.
    sigma_t, sigma_s : float
        Noise stds.
    lbda_s : float
        Ridge penalties.
    alpha : float
        LCB parameter
    min_n_samples_target : int
        Minimum number of target samples.
    chunk_size : int
        Chunk size for dynamic collaboration methods.
    seed : int
        Random seed.
    synthetic_data : bool
        Flag for synthetic data.
    noise : {"gaussian", "lognormal"}
        Synthetic residual distribution used when ``synthetic_data`` is True.
    lognormal_shape : float
        Shape parameter for log-normal residuals.
    method_names : list of str, optional
        Method names to evaluate. If omitted, all methods are run.

    Returns
    -------
    list of tuples
        ``[(name, color, linestyle, values), ...]``
    """
    # assert proper dimenssions
    for X_s in l_X_s:
        assert X_s.shape[1] == l_X_s[0].shape[1]
    assert X_t_train.shape[1] == l_X_s[0].shape[1]
    assert X_t_val.shape[1] == l_X_s[0].shape[1]
    assert X_t_test.shape[1] == l_X_s[0].shape[1]

    rng = check_random_state(seed)
    _, d = l_X_s[0].shape
    if synthetic_data:
        min_n_samples_target = max(2 * d, min_n_samples_target)
    else:
        min_n_samples_target = max(1, min_n_samples_target)
    nn_samples_target = np.logspace(
        np.log10(min_n_samples_target),
        np.log10(n_samples_target),
        num=N_POINTS,
        dtype=int,
    )
    nn_samples_target = np.unique(nn_samples_target)
    nn_samples_target[0] = int(min_n_samples_target)
    nn_samples_target[-1] = int(n_samples_target)
    nn_samples_target = np.unique(nn_samples_target)

    # copy to avoid modifying the original data in-place (e.g., shuffling)
    X_t_train = X_t_train.copy()
    y_t_train = y_t_train.copy()
    y_t_val = y_t_val.copy()
    X_t_test = X_t_test.copy()
    y_t_test = y_t_test.copy()
    l_X_s = [X_s.copy() for X_s in l_X_s]
    l_y_s = [y_s.copy() for y_s in l_y_s]

    # sanityze feature matrixs
    X_t_train = np.array(X_t_train, copy=True, order="C")
    X_t_val = np.array(X_t_val, copy=True, order="C")
    X_t_test = np.array(X_t_test, copy=True, order="C")
    l_X_s = [np.array(X_s, copy=True, order="C") for X_s in l_X_s]

    if synthetic_data:
        theta_t, theta_s = _regen_thetas(theta_t, theta_s, rng)
        y_t_train = _sample_response(X_t_train, theta_t, rng, sigma_t, noise, lognormal_shape)
        y_t_val = _sample_response(X_t_val, theta_t, rng, sigma_t, noise, lognormal_shape)
        y_t_test = _sample_response(X_t_test, theta_t, rng, sigma_t, noise, lognormal_shape)
        l_y_s = [
            _sample_response(X_s, theta_s, rng, sigma_s, noise, lognormal_shape)
            for X_s in l_X_s
        ]
    else:
        # For real data, we just shuffle the samples to have variability across runs.
        idx_t = rng.permutation(X_t_train.shape[0])
        X_t_train = X_t_train[idx_t]
        y_t_train = y_t_train[idx_t]
        for i, (X_s, y_s) in enumerate(zip(l_X_s, l_y_s)):
            idx_s = rng.permutation(X_s.shape[0])
            l_X_s[i], l_y_s[i] = X_s[idx_s], y_s[idx_s]

    # sanityze targets
    y_t_train = np.array(y_t_train, copy=True, order="C")
    y_t_val = np.array(y_t_val, copy=True, order="C")
    y_t_test = np.array(y_t_test, copy=True, order="C")
    l_y_s = [np.array(y_s, copy=True, order="C") for y_s in l_y_s]

    lbda_t = opt_lambda(X_t_train, y_t_train, X_t_val, y_t_val)

    def _mse(theta):
        return np.mean((X_t_test @ theta - y_t_test) ** 2)

    def _err(theta):
        return np.linalg.norm(X_t_test @ (theta - theta_t)) ** 2 / len(X_t_test)

    methods = [
        ("Target-only", _COLORS["target"], "solid", _MARKERS["target"], target_regr),
        ("Mixed", _COLORS["mixed"], "solid", _MARKERS["mixed"], full_regr),
        ("Hierarchical", _COLORS["hier"], "solid", _MARKERS["hier"], hier_regr),
        ("Proposed", _COLORS["proposed"], "solid", _MARKERS["proposed"], dyn_greedy_regr),
        ("Proposed (oracle)", _COLORS["oracle"], "solid", _MARKERS["oracle"], oracle_greedy_regr),
        ("Zhang et al, 2025", _COLORS["zhang"], "dotted", _MARKERS["zhang"], zhang_regr),
        ("Chen et al, 2014", _COLORS["chen"], "dotted", _MARKERS["chen"], chen_regr),
        ("Obst et al, 2021", _COLORS["obst"], "dotted", _MARKERS["obst"], obst_regr),
    ]
    if method_names is not None:
        method_names = set(method_names)
        methods = [method for method in methods if method[0] in method_names]
        if not methods:
            raise ValueError("method_names did not match any regression methods.")

    series = {name: {"err": [], "mse": [], "n_s": []} for name, *_ in methods}

    kwargs = dict(
        theta_t=theta_t,
        X_t_val=X_t_val,
        l_X_s=l_X_s,
        l_y_s=l_y_s,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        alpha=alpha,
        chunk_size=chunk_size
    )

    for n_t in nn_samples_target:
        kwargs["X_t_train"] = X_t_train[:n_t]
        kwargs["y_t_train"] = y_t_train[:n_t]

        for name, _, _, _, fn in methods:
            theta, _, n_s = fn(**kwargs)
            series[name]["err"].append(_err(theta))
            series[name]["mse"].append(_mse(theta))
            series[name]["n_s"].append(n_s)

    def _pack(metric_key):
        return [
            {
                "name": name,
                "color": color,
                "style": style,
                "marker": marker,
                "n_samples": np.array(nn_samples_target),
                "metric": np.array(series[name][metric_key]),
            }
            for name, color, style, marker, _ in methods
        ]

    return {
        "error": _pack("err"),
        "mse": _pack("mse"),
        "n_s": _pack("n_s"),
    }


def run_one_classif_perf_cmp(
    n_samples_target: int,
    theta_t: np.ndarray,
    theta_s: np.ndarray,
    X_t_train: np.ndarray,
    y_t_train: np.ndarray,
    X_t_val: np.ndarray,
    y_t_val: np.ndarray,
    X_t_test: np.ndarray,
    y_t_test: np.ndarray,
    l_X_s: list[np.ndarray],
    l_y_s: list[np.ndarray],
    sigma_t: float,
    sigma_s: float,
    lbda_s: float,
    alpha: float = 0.1,
    min_n_samples_target: int = 100,
    chunk_size: int = 5,
    seed: int | None = None,
    synthetic_data: bool = True,
    noise: str = "gaussian",
    lognormal_shape: float = 1.0,
) -> List[Tuple[str, str, str, np.ndarray]]:
    """Run the classification experiment for a vector of sample sizes."""
    for X_s in l_X_s:
        assert X_s.shape[1] == l_X_s[0].shape[1]
    assert X_t_train.shape[1] == l_X_s[0].shape[1]
    assert X_t_val.shape[1] == l_X_s[0].shape[1]
    assert X_t_test.shape[1] == l_X_s[0].shape[1]

    rng = check_random_state(seed)
    _, d = l_X_s[0].shape
    min_n_samples_target = max(2 * d, min_n_samples_target)
    nn_samples_target = np.logspace(
        np.log10(min_n_samples_target),
        np.log10(n_samples_target + 1),
        num=N_POINTS,
        dtype=int,
    )
    nn_samples_target = np.unique(nn_samples_target)

    X_t_train = X_t_train.copy()
    y_t_train = y_t_train.copy()
    y_t_val = y_t_val.copy()
    X_t_test = X_t_test.copy()
    y_t_test = y_t_test.copy()
    l_X_s = [X_s.copy() for X_s in l_X_s]
    l_y_s = [y_s.copy() for y_s in l_y_s]

    X_t_train = np.array(X_t_train, copy=True, order="C")
    X_t_val = np.array(X_t_val, copy=True, order="C")
    X_t_test = np.array(X_t_test, copy=True, order="C")
    l_X_s = [np.array(X_s, copy=True, order="C") for X_s in l_X_s]

    if synthetic_data:
        theta_t, theta_s = _regen_thetas(theta_t, theta_s, rng)

        def _binarize(z):
            return np.where(z >= 0.0, 1.0, -1.0)

        y_t_train = _binarize(_sample_response(X_t_train, theta_t, rng, sigma_t, noise, lognormal_shape))
        y_t_val = _binarize(_sample_response(X_t_val, theta_t, rng, sigma_t, noise, lognormal_shape))
        y_t_test = _binarize(_sample_response(X_t_test, theta_t, rng, sigma_t, noise, lognormal_shape))
        l_y_s = [
            _binarize(_sample_response(X_s, theta_s, rng, sigma_s, noise, lognormal_shape))
            for X_s in l_X_s
        ]
    else:
        idx_t = rng.permutation(X_t_train.shape[0])
        X_t_train = X_t_train[idx_t]
        y_t_train = y_t_train[idx_t]
        for i, (X_s, y_s) in enumerate(zip(l_X_s, l_y_s)):
            idx_s = rng.permutation(X_s.shape[0])
            l_X_s[i], l_y_s[i] = X_s[idx_s], y_s[idx_s]

    y_t_train = np.array(y_t_train, copy=True, order="C")
    y_t_val = np.array(y_t_val, copy=True, order="C")
    y_t_test = np.array(y_t_test, copy=True, order="C")
    l_y_s = [np.array(y_s, copy=True, order="C") for y_s in l_y_s]

    lbda_t = opt_lambda(X_t_train, y_t_train, X_t_val, y_t_val)

    def _probit(theta):
        scores = X_t_test @ theta
        z = special.erf(-y_t_test * scores / np.sqrt(2.0))
        return float(np.mean(0.5 * (1.0 + z)))

    def _acc(theta):
        y = np.where((X_t_test @ theta) >= 0.0, 1.0, -1.0)
        return float(np.mean(y == y_t_test))

    methods = [
        ("Target-only", _COLORS["target"], "solid", _MARKERS["target"], target_classif),
        ("Mixed", _COLORS["mixed"], "solid", _MARKERS["mixed"], full_classif),
        ("Hierarchical", _COLORS["hier"], "solid", _MARKERS["hier"], hier_classif),
        ("Proposed", _COLORS["proposed"], "solid", _MARKERS["proposed"], dyn_greedy_classif),
        ("Proposed (oracle)", _COLORS["oracle"], "solid", _MARKERS["oracle"], oracle_greedy_classif),
    ]
    series = {name: {"probit": [], "acc": [], "n_s": []} for name, *_ in methods}

    kwargs = dict(
        theta_t=theta_t,
        X_t_val=X_t_val,
        l_X_s=l_X_s,
        l_y_s=l_y_s,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        alpha=alpha,
        chunk_size=chunk_size,
    )

    for n_t in nn_samples_target:
        kwargs["X_t_train"] = X_t_train[:n_t]
        kwargs["y_t_train"] = y_t_train[:n_t]

        for name, _, _, _, fn in methods:
            theta, _, n_s = fn(**kwargs)
            series[name]["probit"].append(_probit(theta))
            series[name]["acc"].append(_acc(theta))
            series[name]["n_s"].append(n_s)

    def _pack(metric_key):
        return [
            {
                "name": name,
                "color": color,
                "style": style,
                "marker": marker,
                "n_samples": np.array(nn_samples_target + 1),
                "metric": np.array(series[name][metric_key]),
            }
            for name, color, style, marker, _ in methods
        ]

    return {
        "probit": _pack("probit"),
        "acc": _pack("acc"),
        "n_s": _pack("n_s"),
    }


def run_one_eps_infl(
    d: int,
    l_eps: float,
    lbda_s: float,
    alpha: float = 0.01,
    noise: str = "gaussian",
    lognormal_shape: float = 1.0,
) -> List[Tuple[str, str, str, np.ndarray]]:
    """Run the experiment for a vector of sample sizes and return series for plotting.

    Parameters
    ----------
    d : int,
        Dimension of the problem.
    l_eps :  float,
        L2-norm difference between linear parameter of datasets target and source.
    lbda_s : float
        Ridge penalties.
    alpha : float
        LCB parameter

    Returns
    -------
    list of tuples
        ``[(name, color, linestyle, values), ...]``
    """
    l_n, error_diff = [], []

    for eps in l_eps:
        # generate new samples in this setting
        theta_s, theta_t = _generate_thetas(d, eps, random_state=None)
        theta_t_and_theta_s = (theta_t, theta_s)

        data = synth_regr_data(
            n_t_train=N_T,
            n_s_train=N_S,
            d=d,
            eps=eps,
            sigma_t=SIGMA,
            sigma_s=SIGMA,
            theta_t_and_theta_s=theta_t_and_theta_s,
            noise=noise,
            lognormal_shape=lognormal_shape,
        )
        X_t, y_t = data["target_train"]
        X_t_val, y_t_val = data["target_val"]
        X_t_test, _ = data["target_test"]
        l_X_s, l_y_s = data["source_train"]
        X_t, y_t = X_t[:N_T], y_t[:N_T]

        lbda_t = opt_lambda(X_t, y_t, X_t_val, y_t_val)
        lbda_c = lbda_s + lbda_t

        augmenter = SampleAugmenter(alpha=alpha, strategy="greedy").fit(
            X_t, y_t, X_t_val, lbda=lbda_t, sigma=1.0
        )
        X_s, y_s = _concat_sources(l_X_s, l_y_s)
        X_c, y_c = augmenter.transform([X_s], [y_s], lbda=lbda_s, sigma=SIGMA)

        l_n.append(augmenter.get_n_selected())

        err_t = np.linalg.norm(X_t_val @ (_ridge(X_t, y_t, lbda_t) - theta_t)) ** 2 / len(X_t_val)
        err_c = np.linalg.norm(X_t_val @ (_ridge(X_c, y_c, lbda_c) - theta_t)) ** 2 / len(X_t_val)
        error_diff.append(err_t - err_c)

    return [
        {
            "name": r"$\Delta^{\star}$",
            "color": _COLORS["proposed"],
            "style": "solid",
            "marker": "o",
            "n_samples": np.array(l_eps),
            "metric": np.array(error_diff),
        },
        {
            "name": r"$\hat{n}$",
            "color": _COLORS["hier"],
            "style": "dashed",
            "marker": "o",
            "n_samples": np.array(l_eps),
            "metric": np.array(l_n),
        },
    ]


def run_one_sigma_infl(
    d: int,
    eps: float,
    l_sigma_s: np.ndarray,
    lbda_s: float,
    alpha: float = 0.01,
    noise: str = "gaussian",
    lognormal_shape: float = 1.0,
) -> List[Tuple[str, str, str, np.ndarray]]:
    """Run the experiment for a vector of sample sizes and return series for plotting.

    Parameters
    ----------
    d : int,
        Dimension of the problem.
    eps :  float,
        L2-norm difference between linear parameter of datasets target and source.
    l_sigma_s : np.ndarray
        List of noise standard deviation of dataset source.
    lbda_s : float
        Ridge penalties.
    alpha : float
        LCB parameter

    Returns
    -------
    list of tuples
        ``[(name, color, linestyle, values), ...]``
    """
    theta_s, theta_t = _generate_thetas(d, eps, random_state=None)
    theta_t_and_theta_s = (theta_t, theta_s)

    l_n, error_diff = [], []

    for sigma_s in l_sigma_s:
        data = synth_regr_data(
            n_t_train=N_T,
            n_s_train=N_S,
            d=d,
            eps=eps,
            sigma_t=SIGMA,
            sigma_s=sigma_s,
            theta_t_and_theta_s=theta_t_and_theta_s,
            noise=noise,
            lognormal_shape=lognormal_shape,
        )
        X_t, y_t = data["target_train"]
        X_t_val, y_t_val = data["target_val"]
        X_t_test, _ = data["target_test"]
        l_X_s, l_y_s = data["source_train"]
        X_t, y_t = X_t[:N_T], y_t[:N_T]

        lbda_t = opt_lambda(X_t, y_t, X_t_val, y_t_val)
        lbda_c = lbda_s + lbda_t

        augmenter = SampleAugmenter(alpha=alpha, strategy="greedy").fit(
            X_t, y_t, X_t_val, lbda=lbda_t, sigma=SIGMA
        )
        X_s, y_s = _concat_sources(l_X_s, l_y_s)
        X_c, y_c = augmenter.transform([X_s], [y_s], lbda=lbda_s, sigma=sigma_s)

        l_n.append(augmenter.get_n_selected())

        err_t = np.linalg.norm(X_t_val @ (_ridge(X_t, y_t, lbda_t) - theta_t)) ** 2 / len(X_t_val)
        err_c = np.linalg.norm(X_t_val @ (_ridge(X_c, y_c, lbda_c) - theta_t)) ** 2 / len(X_t_val)
        error_diff.append(err_t - err_c)

    return [
        {
            "name": r"$\Delta^{\star}$",
            "color": _COLORS["proposed"],
            "style": "solid",
            "marker": "o",
            "n_samples": np.array(l_sigma_s),
            "metric": np.array(error_diff),
        },
        {
            "name": r"$\hat{n}$",
            "color": _COLORS["hier"],
            "style": "dashed",
            "marker": "o",
            "n_samples": np.array(l_sigma_s),
            "metric": np.array(l_n),
        },
    ]


def run_one_lbda_infl(
    theta_t: np.ndarray,
    theta_s: np.ndarray,
    X_t_train: np.ndarray,
    y_t_train: np.ndarray,
    X_t_val: np.ndarray,
    y_t_val: np.ndarray,
    X_t_test: np.ndarray,
    y_t_test: np.ndarray,
    l_X_s: list[np.ndarray],
    l_y_s: list[np.ndarray],
    sigma_t: float,
    sigma_s: float,
    alpha: float,
    seed: int | None = None,
    synthetic_data: bool = True,
    noise: str = "gaussian",
    lognormal_shape: float = 1.0,
) -> List[Tuple[str, str, str, np.ndarray]]:
    """Run the experiment for a vector of sample sizes and return series for plotting."""
    # assert proper dimenssions
    for X_s in l_X_s:
        assert X_s.shape[1] == l_X_s[0].shape[1]
    assert X_t_train.shape[1] == l_X_s[0].shape[1]
    assert X_t_val.shape[1] == l_X_s[0].shape[1]
    assert X_t_test.shape[1] == l_X_s[0].shape[1]

    rng = check_random_state(seed)
    _, d = l_X_s[0].shape

    # copy to avoid modifying the original data in-place (e.g., shuffling)
    X_t_train = X_t_train.copy()
    y_t_train = y_t_train.copy()
    y_t_val = y_t_val.copy()
    X_t_test = X_t_test.copy()
    y_t_test = y_t_test.copy()
    l_X_s = [X_s.copy() for X_s in l_X_s]
    l_y_s = [y_s.copy() for y_s in l_y_s]

    # sanityze feature matrixs
    X_t_train = np.array(X_t_train, copy=True, order="C")
    X_t_val = np.array(X_t_val, copy=True, order="C")
    X_t_test = np.array(X_t_test, copy=True, order="C")
    l_X_s = [np.array(X_s, copy=True, order="C") for X_s in l_X_s]

    if synthetic_data:
        theta_t, theta_s = _regen_thetas(theta_t, theta_s, rng)
        y_t_train = _sample_response(X_t_train, theta_t, rng, sigma_t, noise, lognormal_shape)
        y_t_val = _sample_response(X_t_val, theta_t, rng, sigma_t, noise, lognormal_shape)
        y_t_test = _sample_response(X_t_test, theta_t, rng, sigma_t, noise, lognormal_shape)
        l_y_s = [
            _sample_response(X_s, theta_s, rng, sigma_s, noise, lognormal_shape)
            for X_s in l_X_s
        ]
    else:
        # For real data, we just shuffle the samples to have variability across runs.
        idx_t = rng.permutation(X_t_train.shape[0])
        X_t_train = X_t_train[idx_t]
        y_t_train = y_t_train[idx_t]
        for i, (X_s, y_s) in enumerate(zip(l_X_s, l_y_s)):
            idx_s = rng.permutation(X_s.shape[0])
            l_X_s[i], l_y_s[i] = X_s[idx_s], y_s[idx_s]

    # sanityze targets
    y_t_train = np.array(y_t_train, copy=True, order="C")
    y_t_val = np.array(y_t_val, copy=True, order="C")
    y_t_test = np.array(y_t_test, copy=True, order="C")
    l_y_s = [np.array(y_s, copy=True, order="C") for y_s in l_y_s]

    # for now, concatenate all source datasets
    X_s, y_s = _concat_sources(l_X_s, l_y_s)

    lbda_t = opt_lambda(X_t_train, y_t_train, X_t_val, y_t_val)
    l_lbda_c = np.logspace(np.log(lbda_t / 20.0), np.log(lbda_t * 20.0), N_POINTS)

    l_n, error_diff = [], []

    for lbda_c in l_lbda_c:
        augmenter = SampleAugmenter(alpha=alpha, strategy="greedy").fit(
            X_t_train, y_t_train, X_t_val, lbda=lbda_t, sigma=sigma_t
        )
        X_c, y_c = augmenter.transform([X_s], [y_s], lbda=lbda_c, sigma=sigma_s)

        l_n.append(augmenter.get_n_selected())

        err_t = np.linalg.norm(X_t_val @ (_ridge(X_t_train, y_t_train, lbda_t) - theta_t)) ** 2 / len(X_t_val)
        err_c = np.linalg.norm(X_t_val @ (_ridge(X_c, y_c, lbda_c) - theta_t)) ** 2 / len(X_t_val)
        error_diff.append(float(err_t - err_c))

    return [
        {
            "name": r"$\Delta^{\star}$",
            "color": _COLORS["proposed"],
            "style": "solid",
            "marker": "o",
            "n_samples": np.array(l_lbda_c),
            "metric": np.array(error_diff),
        },
        {
            "name": r"$\hat{n}$",
            "color": _COLORS["hier"],
            "style": "dashed",
            "marker": "o",
            "n_samples": np.array(l_lbda_c),
            "metric": np.array(l_n),
        },
    ]
