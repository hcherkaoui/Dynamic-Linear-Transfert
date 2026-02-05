"""Experiment utilities helpers module."""

# Authors: Hamza Cherkaoui

from __future__ import annotations
from typing import Tuple, List
import numpy as np
from dyntr.utils import check_random_state
from dyntr.datasets import fetch_synthetic_datasets
from dyntr.statistics import compute_delta, opt_lambda
from dyntr.preprocessing import GreedySampleAugmenter
from dyntr.constants import SIGMA, N_TRAIN_SAMPLES, N_T, STEP
from dyntr.regression import (target_single_regression, source_single_regression,
                              full_collaboration_regression, dynamic_collaboration_regression,
                              oracle_dynamic_collaboration_regression, chen_et_al_regression,
                              obst_et_al_regression, hierical_regression)


def _ridge(X, y, lbda):
    return np.linalg.solve(X.T @ X + lbda * np.eye(X.shape[1]), X.T @ y)


def run_one_delta_illustration(
    theta_t: np.ndarray,
    theta_s: np.ndarray,
    X_t: np.ndarray,
    X_s: np.ndarray,
    y_t: np.ndarray,
    y_s: np.ndarray,
    X_t_val: np.ndarray,
    X_t_test: np.ndarray,
    sigma_t: float,
    sigma_s: float,
    lbda_t: float,
    lbda_s: float,
    seed: int | None = None,
    synthetic_data: bool = True,
) -> List[Tuple[str, str, str, np.ndarray, np.ndarray]]:
    """
    Run the experiment for a vector of source sample sizes and return series for plotting.

    Parameters
    ----------
    theta_t, theta_s : ndarray of shape (d,)
        Ground-truth parameters for datasets target and source.
    X_t, X_s : ndarray
        Training features matrices.
    y_t, y_s : ndarray of shape (n_s,), (n_s,)
        Targets.
    X_t_val, X_t_test :  np.ndarray,
        Validation and test features matrices.
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
    rng = check_random_state(seed)

    n, d = X_s.shape
    n_min = max(2 * d, 10)
    nn = np.arange(n_min, n + 1, step=STEP)

    X_t = np.array(X_t, copy=True, order="C")
    X_s = np.array(X_s, copy=True, order="C")

    if synthetic_data:
        y_t = X_t @ theta_t + rng.normal(0.0, sigma_t, size=X_t.shape[0])
        y_s = X_s @ theta_s + rng.normal(0.0, sigma_s, size=X_s.shape[0])
    else:
        y_t = np.array(y_t, copy=True, order="C")
        y_s = np.array(y_s, copy=True, order="C")
        idx_t = rng.permutation(X_t.shape[0])
        X_t, y_t = X_t[idx_t], y_t[idx_t]
        idx_s = rng.permutation(X_s.shape[0])
        X_s, y_s = X_s[idx_s], y_s[idx_s]

    eye = np.eye(d)

    lbda_c = lbda_s + lbda_t

    G_t = X_t.T @ X_t
    b_t = X_t.T @ y_t
    A_t_inv = np.linalg.inv(G_t + lbda_t * eye)
    theta_t_hat = A_t_inv @ b_t

    l_Delta_hat, l_Delta_emp = [], []

    for i in nn:
        Xi_s = X_s[:i]
        yi_s = y_s[:i]

        G_s = Xi_s.T @ Xi_s
        b_s = Xi_s.T @ yi_s

        A_s_inv = np.linalg.inv(G_s + lbda_s * eye)
        theta_s_hat = A_s_inv @ b_s

        A_c_inv = np.linalg.inv(G_s + G_t + lbda_c * eye)
        theta_c_hat = A_c_inv @ (b_s + b_t)

        Delta_hat = compute_delta(
            theta_s=theta_s_hat, theta_t=theta_t_hat, G_s=G_s, G_t=G_t,
            A_t_inv=A_t_inv, A_s_inv=A_s_inv, A_c_inv=A_c_inv, X_val=X_t_val,
            sigma_s=sigma_s, sigma_t=sigma_t, lbda_t=lbda_t, lbda_s=lbda_s,
        )

        # Empirical: test-feature risk vs theta (not using y_test noise)
        Delta_emp = float(np.linalg.norm(X_t_test @ (theta_t_hat - theta_t))**2
                          - np.linalg.norm(X_t_test @ (theta_c_hat - theta_t))**2)  # noqa

        l_Delta_hat.append(Delta_hat)
        l_Delta_emp.append(Delta_emp)

    return [
        (r"$\hat{\Delta}$", "tab:orange", "solid", nn, np.asarray(l_Delta_hat)),
        (r"$\Delta^{\mathrm{emp}}$", "tab:blue", "solid", nn, np.asarray(l_Delta_emp)),
    ]


def run_one_performance_comparison(
    n: int,
    theta_t: np.ndarray,
    theta_s: np.ndarray,
    X_t: np.ndarray,
    X_s: np.ndarray,
    y_t: np.ndarray,
    y_s: np.ndarray,
    X_t_val: np.ndarray,
    y_t_val: np.ndarray,
    X_t_test: np.ndarray,
    sigma_t: float,
    sigma_s: float,
    lbda_s: float,
    alpha: float = 0.1,
    n_min: int = 100,
    seed: int | None = None,
    synthetic_data: bool = True,
) -> List[Tuple[str, str, str, np.ndarray]]:
    """Run the experiment for a vector of sample sizes and return series for plotting.

    Parameters
    ----------
    n : int
        Max number of samples for dataset target.
    theta_t, theta_s : ndarray of shape (d,)
        Ground-truth parameters for datasets target and source.
    X_t, X_s : ndarray
        Training features matrices.
    y_t, y_s : ndarray of shape (n_s,), (n_s,)
        Targets.
    X_t_val, y_t_val, X_t_test :  np.ndarray,
        Validation and test features matrices.
    sigma_t, sigma_s : float
        Noise stds.
    lbda_s : float
        Ridge penalties.
    alpha : float
        LCB parameter
    n_min : int
        Minimum number of target samples.
    seed : int
        Random seed.
    synthetic_data : bool
        Flag for synthetic data.

    Returns
    -------
    list of tuples
        ``[(name, color, linestyle, values), ...]``
    """
    rng = check_random_state(seed)

    _, d = X_s.shape
    n_min = max(2 * d, n_min)
    nn = np.arange(n_min, n + 1, step=STEP)

    X_t = np.array(X_t, copy=True, order="C")
    X_s = np.array(X_s, copy=True, order="C")

    if synthetic_data:
        y_t = X_t @ theta_t + rng.normal(0.0, sigma_t, size=X_t.shape[0])
        y_s = X_s @ theta_s + rng.normal(0.0, sigma_s, size=X_s.shape[0])
    else:
        y_t = np.array(y_t, copy=True, order="C")
        y_s = np.array(y_s, copy=True, order="C")
        idx_t = rng.permutation(X_t.shape[0])
        X_t, y_t = X_t[idx_t], y_t[idx_t]
        idx_s = rng.permutation(X_s.shape[0])
        X_s, y_s = X_s[idx_s], y_s[idx_s]

    X_t = X_t.copy()
    y_t = y_t.copy()

    lbda_t = opt_lambda(X_t, y_t, X_t_val, y_t_val)
    lbda_c = lbda_s + lbda_t

    error_target, error_source, error_union = [], [], []
    error_c, error_c_oracle = [], []
    error_chen_et_al, error_obst_et_al = [], []
    error_hierical = []

    for i in nn:

        X_t_, y_t_ = X_t[:i], y_t[:i]
        X_s_, y_s_ = X_s[:i], y_s[:i]

        kwargs = dict(theta_t=theta_t, X_t=X_t_, X_s=X_s_, y_t=y_t_, y_s=y_s_, X_t_val=X_t_val,
                      lbda_t=lbda_t, lbda_s=lbda_s, lbda_c=lbda_c, sigma_t=sigma_t, sigma_s=sigma_s,
                      alpha=alpha)

        # target
        theta, _, _ = target_single_regression(**kwargs)
        error_target.append(np.linalg.norm(X_t_test @ (theta - theta_t))**2 / len(X_t_test))

        # source
        theta, _, _ = source_single_regression(**kwargs)
        error_source.append(np.linalg.norm(X_t_test @ (theta - theta_t))**2 / len(X_t_test))

        # union
        theta, _, _ = full_collaboration_regression(**kwargs)
        error_union.append(np.linalg.norm(X_t_test @ (theta - theta_t))**2 / len(X_t_test))

        # collab
        theta, _, _ = dynamic_collaboration_regression(**kwargs)
        error_c.append(np.linalg.norm(X_t_test @ (theta - theta_t))**2 / len(X_t_test))

        # oracle
        theta, _, _ = oracle_dynamic_collaboration_regression(**kwargs)
        error_c_oracle.append(np.linalg.norm(X_t_test @ (theta - theta_t))**2 / len(X_t_test))

        # Chen et al
        theta, _, _ = chen_et_al_regression(**kwargs)
        error_chen_et_al.append(np.linalg.norm(X_t_test @ (theta - theta_t))**2 / len(X_t_test))

        # Obst et al
        theta, _, _ = obst_et_al_regression(**kwargs)
        error_obst_et_al.append(np.linalg.norm(X_t_test @ (theta - theta_t))**2 / len(X_t_test))

        # Hierarichal
        theta, _, _ = hierical_regression(**kwargs)
        error_hierical.append(np.linalg.norm(X_t_test @ (theta - theta_t))**2 / len(X_t_test))

    return [
        ("Target-only ridge", "tab:blue", "dotted", np.array(nn + 1), np.array(error_target)),
        ("Source-only ridge", "yellow", "dashed", np.array(nn + 1), np.array(error_source)),
        ("Mixed-task ridge", "tab:purple", "dashed", np.array(nn + 1), np.array(error_union)),
        ("Proposed", "tab:orange", "solid", np.array(nn + 1), np.array(error_c)),
        ("Proposed (oracle)", "tab:red", "solid", np.array(nn + 1), np.array(error_c_oracle)),
        ("Chen et al, 2014", "tab:gray", "solid", np.array(nn + 1), np.array(error_chen_et_al)),
        ("Obst et al, 2021", "tab:green", "solid", np.array(nn + 1), np.array(error_obst_et_al)),
        ("Hierarchical", "tab:olive", "solid", np.array(nn + 1), np.array(error_hierical)),
    ]


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

        data, _ = fetch_synthetic_datasets(
            n_train_samples=N_TRAIN_SAMPLES,
            d=d,
            eps=eps,
            sigma_t=SIGMA,
            sigma_s=SIGMA,
            theta_t_and_theta_s=theta_t_and_theta_s,
            only_train=False,
        )
        (X_t, X_s, y_t, y_s), (X_t_val, _, y_t_val, _), (X_t_test, _, _, _) = data
        X_t, y_t = X_t[:N_T], y_t[:N_T]

        lbda_t = opt_lambda(X_t, y_t, X_t_val, y_t_val)
        lbda_c = lbda_s + lbda_t

        augmenter = GreedySampleAugmenter(alpha=alpha).fit(X_t, y_t, X_t_val, lbda=lbda_t, sigma=1.0)
        X_c, y_c = augmenter.transform(X_s, y_s, lbda=lbda_s, sigma=SIGMA)

        l_n.append(augmenter.get_n_selected())

        error_diff.append(
            np.linalg.norm(X_t_test @ (_ridge(X_t, y_t, lbda_t) - theta_t)) ** 2 / len(X_t_test)  # noqa
            - np.linalg.norm(X_t_test @ (_ridge(X_c, y_c, lbda_c) - theta_t)) ** 2 / len(X_t_test)  # noqa
        )

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
        data, _ = fetch_synthetic_datasets(
            n_train_samples=N_TRAIN_SAMPLES,
            d=d,
            eps=eps,
            sigma_t=SIGMA,
            sigma_s=sigma_s,
            theta_t_and_theta_s=theta_t_and_theta_s,
            only_train=False,
        )
        (X_t, X_s, y_t, y_s), (X_t_val, _, y_t_val, _), (X_t_test, _, _, _) = data
        X_t, y_t = X_t[:N_T], y_t[:N_T]

        lbda_t = opt_lambda(X_t, y_t, X_t_val, y_t_val)
        lbda_c = lbda_s + lbda_t

        augmenter = GreedySampleAugmenter(alpha=alpha).fit(X_t, y_t, X_t_val, lbda=lbda_t, sigma=SIGMA)
        X_c, y_c = augmenter.transform(X_s, y_s, lbda=lbda_s, sigma=sigma_s)

        l_n.append(augmenter.get_n_selected())

        error_diff.append(
            np.linalg.norm(X_t_test @ (_ridge(X_t, y_t, lbda_t) - theta_t)) ** 2 / len(X_t_test)  # noqa
            - np.linalg.norm(X_t_test @ (_ridge(X_c, y_c, lbda_c) - theta_t)) ** 2 / len(X_t_test)  # noqa
        )

    return [
        (r"$\Delta^{\mathrm{emp}}$", "tab:orange", "solid", np.array(l_sigma_s), np.array(error_diff)),
        (r"$n^\star$", "tab:gray", "dashed", np.array(l_sigma_s), np.array(l_n)),
    ]


def run_one_lbda_influence(
    theta_t: np.ndarray,
    theta_s: np.ndarray,
    X_t: np.ndarray,
    X_s: np.ndarray,
    y_t: np.ndarray,
    y_s: np.ndarray,
    X_t_val: np.ndarray,
    y_t_val: np.ndarray,
    X_t_test: np.ndarray,
    sigma_t: float,
    sigma_s: float,
    alpha: float,
    seed: int | None = None,
    synthetic_data: bool = True,
) -> List[Tuple[str, str, str, np.ndarray]]:
    """Run the experiment for a vector of sample sizes and return series for plotting.

    Parameters
    ----------
    theta_t, theta_s : ndarray of shape (d,)
        Ground-truth parameters for datasets target and source.
    X_t, X_s : ndarray
        Training features matrices.
    y_t, y_s : ndarray of shape (n_s,), (n_s,)
        Targets.
    X_t_val, y_t_val, X_t_test :  np.ndarray,
        Validation and test features matrices.
    sigma_t, sigma_s : float
        Noise stds.
    alpha : float
        LCB parameter
    seed : int
        Random seed.
    synthetic_data : bool
        Flag for synthetic data.

    Returns
    -------
    list of tuples
        ``[(name, color, linestyle, values), ...]``
    """
    rng = check_random_state(seed)

    X_t = np.array(X_t, copy=True, order="C")
    X_s = np.array(X_s, copy=True, order="C")

    if synthetic_data:
        y_t = X_t @ theta_t + rng.normal(0.0, sigma_t, size=X_t.shape[0])
        y_s = X_s @ theta_s + rng.normal(0.0, sigma_s, size=X_s.shape[0])
    else:
        y_t = np.array(y_t, copy=True, order="C")
        y_s = np.array(y_s, copy=True, order="C")
        idx_t = rng.permutation(X_t.shape[0])
        X_t, y_t = X_t[idx_t], y_t[idx_t]
        idx_s = rng.permutation(X_s.shape[0])
        X_s, y_s = X_s[idx_s], y_s[idx_s]

    lbda_t = opt_lambda(X_t, y_t, X_t_val, y_t_val)
    l_lbda_c = np.logspace(np.log(lbda_t / 20.0), np.log(lbda_t * 20.0), 50)

    l_n, error_diff = [], []

    for lbda_c in l_lbda_c:
        augmenter = GreedySampleAugmenter(alpha=alpha).fit(X_t, y_t, X_t_val, lbda=lbda_t, sigma=sigma_t)
        X_c, y_c = augmenter.transform(X_s, y_s, lbda=lbda_c, sigma=sigma_s)

        l_n.append(augmenter.get_n_selected())

        error_diff.append(
            np.linalg.norm(X_t_test @ (_ridge(X_t, y_t, lbda_t) - theta_t)) ** 2 / len(X_t_test)  # noqa
            - np.linalg.norm(X_t_test @ (_ridge(X_c, y_c, lbda_c) - theta_t)) ** 2 / len(X_t_test)  # noqa
        )

    return [
        (r"$\Delta^{\mathrm{emp}}$", "tab:orange", "solid", np.array(l_lbda_c), np.array(error_diff)),
        (r"$n^\star$", "tab:gray", "dashed", np.array(l_lbda_c), np.array(l_n)),
    ]
