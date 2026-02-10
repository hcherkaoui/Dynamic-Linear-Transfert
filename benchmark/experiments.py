"""Experiment utilities helpers module."""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from dyntr.augmenter import SampleAugmenter
from dyntr.datasets import fetch_synthetic_datasets
from dyntr.statistics import compute_delta, opt_lambda
from dyntr.utils import check_random_state
from dyntr.regression import (
    chen_et_al_regression,
    dynamic_greedy_collaboration_regression,
    dynamic_uniform_collaboration_regression,
    full_collaboration_regression,
    hierical_regression,
    obst_et_al_regression,
    oracle_dynamic_collaboration_regression,
    source_single_regression,
    target_single_regression,
)
from config import get_config


_CFG = get_config()
SIGMA = _CFG.SIGMA
N_TRAIN_SAMPLES = _CFG.N_TRAIN_SAMPLES
N_T = _CFG.N_T
N_POINTS = _CFG.N_POINTS


def _ridge(X, y, lbda):
    return np.linalg.solve(X.T @ X + lbda * np.eye(X.shape[1]), X.T @ y)


def _concat_sources(l_X_s, l_y_s):
    X_s = np.vstack(l_X_s)
    y_s = np.concatenate(l_y_s)
    return X_s, y_s


def run_one_delta_illustration(
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
        # For synthetic data, we can regenerate the targets with noise to have more variability across runs.
        y_t_train = X_t_train @ theta_t + rng.normal(0.0, sigma_t, size=X_t_train.shape[0])
        l_y_s = [X_s @ theta_s + rng.normal(0.0, sigma_s, size=X_s.shape[0]) for X_s in l_X_s]
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

    l_Delta_hat, l_Delta_emp = [], []

    for i in nn_samples_target:
        Xi_s = X_s[:i]
        yi_s = y_s[:i]

        G_s = Xi_s.T @ Xi_s
        b_s = Xi_s.T @ yi_s

        A_s_inv = np.linalg.inv(G_s + lbda_s * eye)
        theta_s_hat = A_s_inv @ b_s

        A_c_inv = np.linalg.inv(G_s + G_t + (lbda_s + lbda_t) * eye)
        theta_c_hat = A_c_inv @ (b_s + b_t)

        Delta_hat = compute_delta(
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

        # Empirical: test-feature risk vs theta (not using y_test noise)
        Delta_emp = float(np.linalg.norm(X_t_test @ (theta_t_hat - theta_t)) ** 2 - np.linalg.norm(X_t_test @ (theta_c_hat - theta_t)) ** 2)

        l_Delta_hat.append(Delta_hat)
        l_Delta_emp.append(Delta_emp)

    return [
        (r"$\hat{\Delta}$", "tab:orange", "solid", nn_samples_target, np.asarray(l_Delta_hat)),
        (
            r"$\Delta^{\mathrm{emp}}$",
            "tab:blue",
            "solid",
            nn_samples_target,
            np.asarray(l_Delta_emp),
        ),
    ]


def run_one_performance_comparison(
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
        # For synthetic data, we can regenerate the targets with noise to have more variability across runs.
        y_t_train = X_t_train @ theta_t + rng.normal(0.0, sigma_t, size=X_t_train.shape[0])
        l_y_s = [X_s @ theta_s + rng.normal(0.0, sigma_s, size=X_s.shape[0]) for X_s in l_X_s]
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
        ("Target-only ridge", "tab:blue", "dotted", target_single_regression),
        ("Source-only ridge", "tab:blue", "dashed", source_single_regression),
        ("Mixed-task ridge", "tab:blue", "solid", full_collaboration_regression),
        ("Proposed (greedy)", "tab:orange", "solid", dynamic_greedy_collaboration_regression),
        # ("Proposed (uniform)", "tab:olive", "dotted", dynamic_uniform_collaboration_regression),
        ("Proposed (oracle)", "tab:red", "solid", oracle_dynamic_collaboration_regression),
        ("Chen et al, 2014", "tab:gray", "solid", chen_et_al_regression),
        ("Obst et al, 2021", "tab:green", "solid", obst_et_al_regression),
        ("Hierarchical", "tab:brown", "solid", hierical_regression),
    ]
    series = {name: {"err": [], "mse": [], "n_s": []} for name, *_ in methods}

    kwargs = dict(
        theta_t=theta_t,
        X_t_val=X_t_val,
        l_X_s=l_X_s,
        l_y_s=l_y_s,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        lbda_c=lbda_s + lbda_t,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        alpha=alpha,
        chunk_size=chunk_size
    )

    for n_t in nn_samples_target:
        kwargs["X_t_train"] = X_t_train[:n_t]
        kwargs["y_t_train"] = y_t_train[:n_t]

        for name, _color, _style, fn in methods:
            theta, _, n_s = fn(**kwargs)
            series[name]["err"].append(_err(theta))
            series[name]["mse"].append(_mse(theta))
            series[name]["n_s"].append(n_s)

    def _pack(metric_key):
        return [
            (
                name,
                color,
                style,
                np.array(nn_samples_target + 1),
                np.array(series[name][metric_key]),
            )
            for name, color, style, _ in methods
        ]

    return {
        "error:": _pack("err"),
        "mse:": _pack("mse"),
        "n_s:": _pack("n_s"),
    }


def run_one_epsilon_influence(
    d: int,
    l_eps: float,
    lbda_s: float,
    alpha: float = 0.01,
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
        theta_t = np.array([1.0] + [0.0] * (d - 1))
        theta_s = np.array([1.0] + [eps] + [0.0] * (d - 2))
        theta_t_and_theta_s = (theta_t, theta_s)

        data = fetch_synthetic_datasets(
            n_t_train=N_TRAIN_SAMPLES,
            n_s_train=N_TRAIN_SAMPLES,
            d=d,
            eps=eps,
            sigma_t=SIGMA,
            sigma_s=SIGMA,
            theta_t_and_theta_s=theta_t_and_theta_s,
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

        error_diff.append(np.linalg.norm(X_t_test @ (_ridge(X_t, y_t, lbda_t) - theta_t)) ** 2 / len(X_t_test) - np.linalg.norm(X_t_test @ (_ridge(X_c, y_c, lbda_c) - theta_t)) ** 2 / len(X_t_test))

    return [
        (r"$\Delta^{\mathrm{emp}}$", "tab:orange", "solid", np.array(l_eps), np.array(error_diff)),
        (r"$n^\star$", "tab:gray", "dashed", np.array(l_eps), np.array(l_n)),
    ]


def run_one_sigma_influence(
    d: int,
    eps: float,
    l_sigma_s: np.ndarray,
    lbda_s: float,
    alpha: float = 0.01,
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
    theta_t = np.array([1.0] + [0.0] * (d - 1))
    theta_s = np.array([1.0] + [eps] + [0.0] * (d - 2))
    theta_t_and_theta_s = (theta_t, theta_s)

    l_n, error_diff = [], []

    for sigma_s in l_sigma_s:
        data = fetch_synthetic_datasets(
            n_t_train=N_TRAIN_SAMPLES,
            n_s_train=N_TRAIN_SAMPLES,
            d=d,
            eps=eps,
            sigma_t=SIGMA,
            sigma_s=sigma_s,
            theta_t_and_theta_s=theta_t_and_theta_s,
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

        error_diff.append(np.linalg.norm(X_t_test @ (_ridge(X_t, y_t, lbda_t) - theta_t)) ** 2 / len(X_t_test) - np.linalg.norm(X_t_test @ (_ridge(X_c, y_c, lbda_c) - theta_t)) ** 2 / len(X_t_test))

    return [
        (r"$\Delta^{\mathrm{emp}}$", "tab:orange", "solid", np.array(l_sigma_s), np.array(error_diff)),
        (r"$n^\star$", "tab:gray", "dashed", np.array(l_sigma_s), np.array(l_n)),
    ]


def run_one_lbda_influence(
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
        # For synthetic data, we can regenerate the targets with noise to have more variability across runs.
        y_t_train = X_t_train @ theta_t + rng.normal(0.0, sigma_t, size=X_t_train.shape[0])
        l_y_s = [
            X_s @ theta_s + rng.normal(0.0, sigma_s, size=X_s.shape[0])
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
    l_lbda_c = np.logspace(np.log(lbda_t / 20.0), np.log(lbda_t * 20.0), 50)

    l_n, error_diff = [], []

    for lbda_c in l_lbda_c:
        augmenter = SampleAugmenter(alpha=alpha, strategy="greedy").fit(
            X_t_train, y_t_train, X_t_val, lbda=lbda_t, sigma=sigma_t
        )
        X_c, y_c = augmenter.transform([X_s], [y_s], lbda=lbda_c, sigma=sigma_s)

        l_n.append(augmenter.get_n_selected())

        error_diff.append(np.linalg.norm(X_t_test @ (_ridge(X_t_train, y_t_train, lbda_t) - theta_t)) ** 2 / len(X_t_test) - np.linalg.norm(X_t_test @ (_ridge(X_c, y_c, lbda_c) - theta_t)) ** 2 / len(X_t_test))

    return [
        (r"$\Delta^{\mathrm{emp}}$", "tab:orange", "solid", np.array(l_lbda_c), np.array(error_diff)),
        (r"$n^\star$", "tab:gray", "dashed", np.array(l_lbda_c), np.array(l_n)),
    ]
