"""Linear models module (regression + classification + statistics)."""
from __future__ import annotations
from functools import partial
from textwrap import dedent
from typing import Tuple
import math
import numba
import numpy as np


_INV_SQRT2 = 0.70710678118654752440  # 1/sqrt(2)


@numba.jit(nopython=True, cache=True, fastmath=True)
def _phi_vec(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = 0.5 * math.erfc(-x[i] * _INV_SQRT2)
    return out


@numba.jit(nopython=True, cache=True, fastmath=True)
def compute_delta_regr(
    theta_t: np.ndarray,
    theta_s: np.ndarray,
    G_s: np.ndarray,
    G_t: np.ndarray,
    A_t_inv: np.ndarray,
    A_s_inv: np.ndarray,
    A_c_inv: np.ndarray,
    X_val: np.ndarray,
    sigma_s: float,
    sigma_t: float,
    lbda_t: float,
    lbda_s: float,
) -> float:
    sigma_t_sq = sigma_t * sigma_t
    sigma_s_sq = sigma_s * sigma_s
    eye = np.eye(G_s.shape[0])
    A_s = G_s + lbda_s * eye
    theta_diff = theta_s - theta_t
    U = X_val @ A_t_inv
    V = X_val @ A_c_inv
    UTU = U.T @ U
    VTV = V.T @ V
    v1 = lbda_t * U @ theta_t
    v2 = V @ (G_s @ theta_diff - lbda_t * theta_t)
    L1 = sigma_t_sq * G_t
    L2 = -lbda_t * lbda_t * sigma_t_sq * A_t_inv @ G_t @ A_t_inv
    K1 = -sigma_s_sq * G_s @ A_s_inv @ G_s @ A_s_inv
    K2 = -sigma_t_sq * A_s @ A_t_inv @ G_t @ A_t_inv @ A_s
    K3 = sigma_s_sq * G_s + sigma_t_sq * G_t
    N = K1 + K2 + K3
    M = L1 + L2
    return float(v1 @ v1) - float(v2 @ v2) + np.trace(M @ UTU) - np.trace(N @ VTV)


@numba.jit(nopython=True, cache=True, fastmath=True)
def compute_delta_regr_variance(
    theta_s: np.ndarray,
    theta_t: np.ndarray,
    G_s: np.ndarray,
    G_t: np.ndarray,
    A_t_inv: np.ndarray,
    A_s_inv: np.ndarray,
    A_c_inv: np.ndarray,
    X_val: np.ndarray,
    sigma_s: float,
    sigma_t: float,
    lbda_t: float,
    lbda_s: float,
) -> float:
    sigma_t_sq = sigma_t * sigma_t
    sigma_s_sq = sigma_s * sigma_s
    d = G_s.shape[0]
    eye = np.eye(d)
    A_s = G_s + lbda_s * eye
    U = X_val @ A_t_inv
    V = X_val @ A_c_inv
    UTU = U.T @ U
    VTV = V.T @ V
    D = np.zeros((2 * d, 2 * d))
    D[0:d, 0:d] = -G_s @ (VTV @ G_s)
    D[0:d, d:2 * d] = G_s @ (VTV @ A_s)
    D[d:2 * d, 0:d] = A_s @ (VTV @ G_s)
    D[d:2 * d, d:2 * d] = lbda_t * lbda_t * UTU - A_s @ VTV @ A_s
    mu = np.zeros(2 * d)
    mu[0:d] = A_s_inv @ G_s @ theta_s
    mu[d:2 * d] = A_t_inv @ G_t @ theta_t
    Sigma = np.zeros((2 * d, 2 * d))
    Sigma[0:d, 0:d] = sigma_s_sq * A_s_inv @ G_s @ A_s_inv
    Sigma[d:2 * d, d:2 * d] = sigma_t_sq * A_t_inv @ G_t @ A_t_inv
    DS = D @ Sigma
    return float(2.0 * np.trace(DS @ DS) + 4.0 * mu @ DS @ D @ mu)


@numba.jit(nopython=True, cache=True, fastmath=True)
def compute_delta_regr_lower_bound(
    theta_t: np.ndarray,
    theta_s: np.ndarray,
    G_t: np.ndarray,
    G_s: np.ndarray,
    A_t_inv: np.ndarray,
    A_s_inv: np.ndarray,
    A_c_inv: np.ndarray,
    X_val: np.ndarray,
    sigma_t: float,
    sigma_s: float,
    lbda_t: float = 1.0,
    lbda_s: float = 1.0,
    alpha: float = 0.01,
) -> float:
    delta_hat = compute_delta_regr(
        theta_t,
        theta_s,
        G_s,
        G_t,
        A_t_inv,
        A_s_inv,
        A_c_inv,
        X_val,
        sigma_s,
        sigma_t,
        lbda_t,
        lbda_s,
    )
    var_delta_hat = compute_delta_regr_variance(
        theta_s,
        theta_t,
        G_s,
        G_t,
        A_t_inv,
        A_s_inv,
        A_c_inv,
        X_val,
        sigma_s,
        sigma_t,
        lbda_t,
        lbda_s,
    )
    if var_delta_hat < 0.0:
        var_delta_hat = 0.0
    return delta_hat - alpha * np.sqrt(var_delta_hat)


@numba.jit(nopython=True, cache=True, fastmath=True)
def compute_delta_classif(
    theta_t: np.ndarray,
    theta_c: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    if X_val.shape[0] == 0:
        return 0.0
    s_t = X_val @ theta_t
    s_c = X_val @ theta_c
    z_t = -y_val * s_t
    z_c = -y_val * s_c
    ell_t = _phi_vec(z_t)
    ell_c = _phi_vec(z_c)
    return np.mean(ell_t) - np.mean(ell_c)


@numba.jit(nopython=True, cache=True, fastmath=True)
def compute_delta_classif_variance(
    theta_t: np.ndarray,
    theta_c: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    n = X_val.shape[0]
    if n <= 1:
        return 0.0
    s_t = X_val @ theta_t
    s_c = X_val @ theta_c
    z_t = -y_val * s_t
    z_c = -y_val * s_c
    ell_t = _phi_vec(z_t)
    ell_c = _phi_vec(z_c)
    d = ell_t - ell_c
    mean_d = np.mean(d)
    var_pop = np.mean(d * d) - mean_d * mean_d
    if var_pop < 0.0:
        var_pop = 0.0
    return var_pop / (n - 1.0)


@numba.jit(nopython=True, cache=True, fastmath=True)
def compute_delta_classif_lower_bound(
    theta_t: np.ndarray,
    theta_c: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alpha: float = 0.01,
) -> float:
    delta_hat = compute_delta_classif(
        theta_t,
        theta_c,
        X_val,
        y_val,
    )
    var_delta_hat = compute_delta_classif_variance(
        theta_t,
        theta_c,
        X_val,
        y_val,
    )
    if var_delta_hat < 0.0:
        var_delta_hat = 0.0
    return delta_hat - alpha * np.sqrt(var_delta_hat)


@numba.jit(nopython=True, cache=True, fastmath=True)
def opt_lambda(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lbda_grid=np.logspace(-3, 3, 50),
) -> float:
    _, d = X.shape
    eye = np.eye(d)
    G = X.T @ X
    b = X.T @ y
    m = lbda_grid.shape[0]
    errors = np.empty(m)
    for i in range(m):
        r = X_val @ np.linalg.solve(G + lbda_grid[i] * eye, b) - y_val
        errors[i] = r @ r
    return float(lbda_grid[np.argmin(errors)])


@numba.jit(nopython=True, cache=True, fastmath=True)
def _ridge(
    X: np.ndarray,
    y: np.ndarray,
    lbda: float,
) -> np.ndarray:
    return np.linalg.solve(X.T @ X + lbda * np.eye(X.shape[1]), X.T @ y)


def _check_n_max(l_X_s: list[np.ndarray], n_max: int | None) -> int:
    if n_max is None:
        n_max = max(len(X_s) for X_s in l_X_s) if l_X_s else 0
    if len(l_X_s) > 1 and n_max > 0:
        n_max = max(1, int(n_max // len(l_X_s)))
    return int(n_max)


def _check_strategy(mode: str) -> str:
    strategy = "Unknown"
    strategy = "uniform" if "uniform" in mode else strategy
    strategy = "greedy" if "greedy" in mode else strategy
    return strategy


@numba.jit(nopython=True, cache=True, fastmath=True)
def _fit_stats(
    X: np.ndarray,
    y: np.ndarray,
    lbda: float,
):
    n, d = X.shape
    G = X.T @ X + lbda * np.eye(d)
    G_inv = np.linalg.inv(G)
    theta = G_inv @ (X.T @ y)
    r = y - X @ theta
    s2 = float(r.T @ r) / max(n - d, 1)
    return theta, G, G_inv, s2, n, d


@numba.jit(nopython=True, cache=True, fastmath=True)
def _default_alpha(
    G: np.ndarray,
    factor: float = 10.0,
) -> float:
    w = np.linalg.eigvalsh(G)
    lam_min = max(float(w[0]), 1e-12)
    lam_max = float(w[-1])
    return 2.0 / (lam_max + lam_min) / factor


@numba.jit(nopython=True, cache=True, fastmath=True)
def _make_cache(
    G_t: np.ndarray,
    alpha: float,
    k: int,
):
    w, P = np.linalg.eigh(G_t)
    one = 1.0 - alpha * w
    ak = np.power(one, k)
    A_k = (P * ak) @ P.T
    eye = np.eye(G_t.shape[0])
    I_m_Ak = eye - A_k
    w_inv = np.where(w > 0, 1.0 / w, 0.0)
    G_t_inv = (P * w_inv) @ P.T
    Omega_k = (1.0 / alpha) * G_t_inv @ I_m_Ak
    return A_k, Omega_k, G_t_inv


@numba.jit(nopython=True, cache=True, fastmath=True)
def _theta_k(
    theta_s: np.ndarray,
    theta_t: np.ndarray,
    A_k: np.ndarray,
) -> np.ndarray:
    d = A_k.shape[0]
    return A_k @ theta_s + (np.eye(d) - A_k) @ theta_t


@numba.jit(nopython=True, cache=True, fastmath=True)
def _V_k(
    s2_s: float,
    s2_t: float,
    A_k: np.ndarray,
    Omega_k: np.ndarray,
    G_t: np.ndarray,
    G_s_inv: np.ndarray,
    alpha: float,
) -> np.ndarray:
    t1 = s2_s * (A_k @ G_s_inv @ A_k)
    t2 = s2_t * (alpha ** 2) * (Omega_k @ G_t @ Omega_k)
    return t1 + t2


@numba.jit(nopython=True, cache=True, fastmath=True)
def _U_k(
    X: np.ndarray,
    G_t_inv: np.ndarray,
    V: np.ndarray,
    s2_t: float,
) -> float:
    n = X.shape[0]
    acc = 0.0
    for i in range(n):
        x = X[i]
        gx = G_t_inv @ x
        vx = V @ x
        a = float(np.dot(x, gx))
        b = float(np.dot(x, vx))
        denom = s2_t * a
        if denom < 1e-30:
            denom = 1e-30
        r = b / denom
        if r < 1e-30:
            r = 1e-30
        acc += -s2_t * a * np.log(r)
    return acc / n


def _select_k(
    Xk: np.ndarray,
    fit_s,
    fit_t,
    alpha: float,
    k_grid: np.ndarray,
    k_min: int = 5,
) -> int:
    _, _, G_s_inv, s2_s, _, _ = fit_s
    _, G_t, _, s2_t, _, _ = fit_t

    scores = np.empty(len(k_grid), dtype=np.float64)
    for i, k in enumerate(k_grid):
        A_k, Omega_k, G_t_inv = _make_cache(G_t, alpha, int(k))
        V = _V_k(s2_s, s2_t, A_k, Omega_k, G_t, G_s_inv, alpha)
        scores[i] = _U_k(Xk, G_t_inv, V, s2_t)

    cand = []
    for i in range(1, len(k_grid) - 1):
        if k_grid[i] >= k_min and scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1]:
            cand.append(i)
    if cand:
        j = cand[int(np.argmax(scores[cand]))]
        return int(k_grid[j])

    mask = k_grid >= k_min
    if np.any(mask):
        j = int(np.argmax(scores[mask]))
        return int(k_grid[mask][j])
    return int(k_grid[int(np.argmax(scores))])


@numba.jit(nopython=True, cache=True, fastmath=True)
def _hierarchical_theta(
    X_t_train: np.ndarray,
    y_t_train: np.ndarray,
    X_s: np.ndarray,
    y_s: np.ndarray,
    sigma_t: float,
    sigma_s: float,
    lbda_t: float,
    lbda_s: float,
):
    _, d = X_t_train.shape
    assert X_s.shape[1] == d, "X_t_train and X_s must have the same dimension."

    sig2_t = sigma_t ** 2
    sig2_s = sigma_s ** 2

    lbda_c = 0.5 * (lbda_t + lbda_s)
    w = 0.5
    lambda_b = (w / (1.0 - w)) * lbda_c
    lambda_0 = 0.1 * lbda_c

    G_t = (X_t_train.T @ X_t_train) / sig2_t
    G_s = (X_s.T @ X_s) / sig2_s
    g_t = (X_t_train.T @ y_t_train) / sig2_t
    g_s = (X_s.T @ y_s) / sig2_s

    B_t = G_t + (lambda_b + lbda_t) * np.eye(d)
    B_s = G_s + (lambda_b + lbda_s) * np.eye(d)

    B_t_inv = np.linalg.inv(B_t)
    B_s_inv = np.linalg.inv(B_s)

    A_mu = (lambda_0 + 2.0 * lambda_b) * np.eye(d) - (lambda_b * lambda_b) * (B_t_inv + B_s_inv)

    mu = np.linalg.solve(A_mu, lambda_b * (B_t_inv @ g_t + B_s_inv @ g_s))
    theta_t = np.linalg.solve(B_t, g_t + lambda_b * mu)

    return theta_t


_DOC = dedent(
    """
    Estimate the linear parameter from a combination of target and source.

    Parameters
    ----------
    theta_t : ndarray of shape (d,)
        Ground-truth parameters for the target dataset (oracle mode only).
    X_t_train, y_t_train : ndarray
        Training feature matrices and targets for the target dataset.
    X_t_val : ndarray
        Validation features for target.
    l_X_s, l_y_s : list of ndarray of shape (n_t,), (n_s,)
        Training feature matrices and targets for the source datasets.
    sigma_t, sigma_s : float
        Noise standard deviations.
    lbda_t, lbda_s, lbda_c : float
        Ridge penalties.
    alpha : float
        LCB parameter

    Returns
    -------
    theta : ndarray
        Estimated linear parameter.
    n_t : int
        Number of target samples used.
    n_s : int
        Number of source samples used.
    """
)


def regr(
    theta_t: np.ndarray,
    X_t_train: np.ndarray,
    y_t_train: np.ndarray,
    X_t_val: np.ndarray,
    l_X_s: list[np.ndarray],
    l_y_s: list[np.ndarray],
    sigma_t: float,
    sigma_s: float,
    lbda_t: float,
    lbda_s: float,
    alpha: float = 0.05,
    mode: str = "target",
    chunk_size: int = 5,
    n_max: int | None = None,
    criterion: str = "regr",
    y_t_val: np.ndarray | None = None,
) -> Tuple[np.ndarray, int, int]:
    """Dispatch function modes and return (theta, n_t, n_s)."""

    if mode == "target":  # simple baseline
        return _ridge(X_t_train, y_t_train, lbda_t), len(X_t_train), 0

    if mode == "source":  # simple baseline: use all sources samples (only)

        # Designed for a mono-source setting
        X_s = np.vstack(l_X_s)
        y_s = np.concatenate(l_y_s)

        return _ridge(X_s, y_s, lbda_s), 0, len(X_s)

    if mode == "full":  # simple baseline: use all samples

        # Designed for a mono-source setting
        X_s = np.vstack(l_X_s)
        y_s = np.concatenate(l_y_s)
        X_c = np.r_[X_t_train, X_s]
        y_c = np.r_[y_t_train, y_s]

        return _ridge(X_c, y_c, lbda_t), len(X_t_train), len(X_s)

    n_max = _check_n_max(l_X_s, n_max)

    if "dynamic" in mode:
        from .augmenter import SampleAugmenter

        strategy = _check_strategy(mode)
        assert strategy in ["uniform", "greedy"], f"Unknown strategy type, got {strategy}"

        # our proposed approach
        aug = SampleAugmenter(
            alpha=alpha,
            strategy=strategy,
            chunk_size=chunk_size,
            n_max=n_max,
            criterion=criterion,
        )
        aug = aug.fit(X_t_train, y_t_train, X_t_val, lbda=lbda_t, sigma=sigma_t, y_t_val=y_t_val)
        X_c, y_c = aug.transform(l_X_s, l_y_s, sigma=sigma_s, lbda=lbda_s)

        return _ridge(X_c, y_c, lbda_t + lbda_s), len(X_t_train), int(aug.get_n_selected())

    if "oracle" in mode:
        from .augmenter import OracleSampleAugmenter

        strategy = _check_strategy(mode)
        assert strategy in ["uniform", "greedy"], f"Unknown strategy type, got {strategy}"

        # our proposed oracle approach
        ora = OracleSampleAugmenter(
            strategy=strategy,
            chunk_size=chunk_size,
            n_max=n_max,
            criterion=criterion,
        )
        ora = ora.fit(X_t_train, y_t_train, X_t_val, lbda=lbda_t, theta_t_true=theta_t)
        X_c, y_c = ora.transform(l_X_s, l_y_s, lbda=lbda_t + lbda_s)

        return _ridge(X_c, y_c, lbda_t + lbda_s), len(X_t_train), int(ora.get_n_selected())

    if mode == "zhang_et_al":  # Zhang et al, 2025: https://arxiv.org/pdf/2601.10779
        from .augmenter import UOWQSampleAugmenter

        aug = UOWQSampleAugmenter(n_max=n_max)
        aug = aug.fit(X_t_train, y_t_train, X_t_val, lbda=lbda_t, sigma=sigma_t)
        X_c, y_c = aug.transform(l_X_s, l_y_s, lbda=lbda_s, sigma=sigma_s)

        return _ridge(X_c, y_c, lbda_t), len(X_t_train), int(aug.get_n_selected())

    if mode == "chen_et_al":  # Chen et al, 2014: https://arxiv.org/pdf/1304.1837

        # the original paper consider only a mono-source setting
        X_s = np.vstack(l_X_s)
        y_s = np.concatenate(l_y_s)

        # for consistency we consider our ridge case (not the OLS as the original papier)
        theta_t_ = _ridge(X=X_t_train, y=y_t_train, lbda=lbda_t)
        theta_s_ = _ridge(X=X_s, y=y_s, lbda=lbda_s)

        eye = np.eye(X_t_train.shape[1])
        G_t, G_s, G_t_val = X_t_train.T @ X_t_train, X_s.T @ X_s, X_t_val.T @ X_t_val
        G_s_inv_G_t = np.linalg.solve(G_s, G_t)

        # tranfer matrix
        lbda = 0.1  # as with our approach: not hyper-opt the amout of tranfer (ie alpha)
        W = np.linalg.solve(
            G_t + lbda * (G_t_val @ G_s_inv_G_t) + lbda * G_t_val,
            G_t + lbda * (G_t_val @ G_s_inv_G_t),
        )
        return W @ theta_t_ + (eye - W) @ theta_s_, len(X_t_train), len(X_s)

    if mode == "obst_et_al":  # Obst et al, 2021: https://arxiv.org/pdf/2102.09504

        # Designed for a mono-source setting
        X_s = np.vstack(l_X_s)
        y_s = np.concatenate(l_y_s)

        fit_s = _fit_stats(X_s, y_s, lbda_s)
        fit_t = _fit_stats(X_t_train, y_t_train, lbda_t)

        alpha_ft = _default_alpha(fit_t[1], factor=10.0)
        k_grid = np.arange(0, 1001, 50, dtype=np.int64)  # trying to give a fair grid

        # find the best number of gradient step
        k_star = _select_k(X_t_val, fit_s, fit_t, alpha_ft, k_grid, k_min=1)
        A_k, _, _ = _make_cache(fit_t[1], alpha_ft, k_star)

        # re-fit final theta_t
        theta = _theta_k(fit_s[0], fit_t[0], A_k)

        return theta, len(X_t_train), len(X_s)

    if mode == "hierarchical":

        # Designed for a mono-source setting
        X_s = np.vstack(l_X_s)
        y_s = np.concatenate(l_y_s)

        # fit a mixture of Gaussian then return the target corresponding theta
        theta = _hierarchical_theta(
            X_t_train=X_t_train,
            y_t_train=y_t_train,
            X_s=X_s,
            y_s=y_s,
            sigma_t=sigma_t,
            sigma_s=sigma_s,
            lbda_t=lbda_t,
            lbda_s=lbda_s,
        )

        return theta, len(X_t_train), len(X_s)

    raise ValueError(f"Unknown mode: {mode}")


def classif(
    theta_t: np.ndarray,
    X_t_train: np.ndarray,
    y_t_train: np.ndarray,
    X_t_val: np.ndarray,
    l_X_s: list[np.ndarray],
    l_y_s: list[np.ndarray],
    sigma_t: float,
    sigma_s: float,
    lbda_t: float,
    lbda_s: float,
    alpha: float = 0.05,
    mode: str = "target",
    chunk_size: int = 5,
    n_max: int | None = None,
    y_t_val: np.ndarray | None = None,
) -> Tuple[np.ndarray, int, int]:
    """Dispatch function modes and return (theta, n_t, n_s)."""

    if y_t_val is None:
        y_t_val = np.where(X_t_val @ theta_t >= 0.0, 1.0, -1.0)

    kwargs = dict(
        theta_t=theta_t, X_t_train=X_t_train, y_t_train=y_t_train,
        X_t_val=X_t_val, l_X_s=l_X_s, l_y_s=l_y_s, sigma_t=sigma_t,
        sigma_s=sigma_s, lbda_t=lbda_t, lbda_s=lbda_s, alpha=alpha,
        mode=mode, chunk_size=chunk_size, n_max=n_max, criterion="classif",
        y_t_val=y_t_val,
    )

    if mode in ['chen_et_al', 'obst_et_al', 'zhang_et_al']:
        raise ValueError(f"{mode} not designed for classification task.")

    else:
        return regr(**kwargs)


target_regr = partial(regr, mode="target")
source_regr = partial(regr, mode="source")
full_regr = partial(regr, mode="full")
hier_regr = partial(regr, mode="hierarchical")

dyn_greedy_regr = partial(regr, mode="dynamic-greedy")
dyn_uniform_regr = partial(regr, mode="dynamic-uniform")
oracle_greedy_regr = partial(regr, mode="oracle-greedy")
oracle_uniform_regr = partial(regr, mode="oracle-uniform")

zhang_regr = partial(regr, mode="zhang_et_al")
chen_regr = partial(regr, mode="chen_et_al")
obst_regr = partial(regr, mode="obst_et_al")

target_classif = partial(classif, mode="target")
source_classif = partial(classif, mode="source")
full_classif = partial(classif, mode="full")
hier_classif = partial(classif, mode="hierarchical")

dyn_greedy_classif = partial(classif, mode="dynamic-greedy")
dyn_uniform_classif = partial(classif, mode="dynamic-uniform")
oracle_greedy_classif = partial(classif, mode="oracle-greedy")
oracle_uniform_classif = partial(classif, mode="oracle-uniform")

target_regr.__doc__ = _DOC
source_regr.__doc__ = _DOC
full_regr.__doc__ = _DOC
hier_regr.__doc__ = _DOC

dyn_greedy_regr.__doc__ = _DOC
dyn_uniform_regr.__doc__ = _DOC
oracle_greedy_regr.__doc__ = _DOC
oracle_uniform_regr.__doc__ = _DOC

zhang_regr.__doc__ = _DOC
chen_regr.__doc__ = _DOC
obst_regr.__doc__ = _DOC
hier_regr.__doc__ = _DOC

target_classif.__doc__ = _DOC
source_classif.__doc__ = _DOC
full_classif.__doc__ = _DOC
hier_classif.__doc__ = _DOC

dyn_greedy_classif.__doc__ = _DOC
dyn_uniform_classif.__doc__ = _DOC
oracle_greedy_classif.__doc__ = _DOC
oracle_uniform_classif.__doc__ = _DOC

__all__ = [

    # Delta related computation functions
    "compute_delta_regr",
    "compute_delta_regr_variance",
    "compute_delta_regr_lower_bound",
    "compute_delta_classif",
    "compute_delta_classif_variance",
    "compute_delta_classif_lower_bound",

    # usefull helper
    "opt_lambda",

    # regression approaches
    "target_regr",
    "source_regr",
    "full_regr",
    "dyn_greedy_regr",
    "dyn_uniform_regr",
    "oracle_greedy_regr",
    "oracle_uniform_regr",
    "zhang_regr",
    "chen_regr",
    "obst_regr",
    "hier_regr",

    # classification approaches
    "target_classif",
    "source_classif",
    "full_classif",
    "hier_classif",
    "dyn_greedy_classif",
    "dyn_uniform_classif",
    "oracle_greedy_classif",
    "oracle_uniform_classif",
]
