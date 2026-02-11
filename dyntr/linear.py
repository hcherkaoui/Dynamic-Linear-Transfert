"""Linear models module (regression + classification + statistics)."""
from __future__ import annotations
from functools import partial
from textwrap import dedent
from typing import Tuple
import numba
import numpy as np


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
    lbda_t=1.0,
    lbda_s=1.0,
    alpha=0.01,
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
def _opt_lambda_corrected_aic(
    theta_t: np.ndarray,
    theta_s: np.ndarray,
    X_t_train: np.ndarray,
    y_t_train: np.ndarray,
    G_t: np.ndarray,
    G_s_inv_G_t: np.ndarray,
    G_t_val: np.ndarray,
    eye: np.ndarray,
    lbda_grid: np.ndarray,
) -> float:
    n, _ = X_t_train.shape
    best_aicc, best_lbda = np.inf, lbda_grid[0]
    for lbda in lbda_grid:
        W = np.linalg.solve(
            G_t + lbda * (G_t_val @ G_s_inv_G_t) + lbda * G_t_val,
            G_t + lbda * (G_t_val @ G_s_inv_G_t),
        )
        theta_c = W @ theta_t + (eye - W) @ theta_s
        df = np.trace(W)
        r = y_t_train - X_t_train @ theta_c
        sigma_s = (r @ r) / max(1.0, n - df)
        den = 1.0 - df + 2.0 / n
        if den <= 0 or sigma_s <= 0:
            aicc = np.inf
        else:
            aicc = n * np.log(sigma_s) + n * ((1.0 + df / n) / den)
        if aicc < best_aicc:
            best_aicc = aicc
            best_lbda = lbda
    return best_lbda


def _hierarchical_theta(
    X_t_train: np.ndarray,
    y_t_train: np.ndarray,
    X_s: np.ndarray,
    y_s: np.ndarray,
    sigma_t: float,
    sigma_s: float,
    lbda_t: float,
    lbda_s: float,
    lbda_c: float,
) -> np.ndarray:
    n_t, d = X_t_train.shape
    n_s, d_s = X_s.shape
    if d_s != d:
        raise ValueError("X_t_train and X_s must have the same number of columns.")

    sig2_t = max(float(sigma_t) * float(sigma_t), 1e-12)
    sig2_s = max(float(sigma_s) * float(sigma_s), 1e-12)

    lambda_0 = max(float(lbda_c), 1e-8)
    lambda_b = max(0.5 * (float(lbda_t) + float(lbda_s)), 1e-8)

    G_t = (X_t_train.T @ X_t_train) / sig2_t
    G_s = (X_s.T @ X_s) / sig2_s
    g_t = (X_t_train.T @ y_t_train) / sig2_t
    g_s = (X_s.T @ y_s) / sig2_s

    eye = np.eye(d)
    A = (lambda_0 + 2.0 * lambda_b) * eye + G_t + G_s
    b = g_t + g_s
    mu = np.linalg.solve(A, b)

    B = lambda_b * eye + G_t
    theta_t = np.linalg.solve(B, g_t + lambda_b * mu)
    return theta_t


_REGR_DOC = dedent(
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
    lbda_c: float,
    alpha: float = 0.05,
    mode: str = "dynamic-greedy",
    chunk_size: int = 5,
    n_max: int | None = None,
) -> Tuple[np.ndarray, int, int]:
    """Dispatch regression modes and return (theta, n_t, n_s)."""

    if mode == "target":
        return _ridge(X_t_train, y_t_train, lbda_t), len(X_t_train), 0

    if mode == "source":
        X_s = np.vstack(l_X_s)
        y_s = np.concatenate(l_y_s)
        return _ridge(X_s, y_s, lbda_s), 0, len(X_s)

    if mode == "full":
        X_s = np.vstack(l_X_s)
        y_s = np.concatenate(l_y_s)
        X_c = np.r_[X_t_train, X_s]
        y_c = np.r_[y_t_train, y_s]
        return _ridge(X_c, y_c, lbda_c), len(X_t_train), len(X_s)

    n_sources = len(l_X_s)
    n_max_val = n_max
    if n_max_val is None:
        n_max_val = max(len(X_s) for X_s in l_X_s) if l_X_s else 0
    if n_sources > 1 and n_max_val > 0:
        n_max_val = max(1, int(n_max_val // n_sources))

    if mode in {"dynamic", "dynamic-greedy", "dynamic-uniform"}:
        from .augmenter import SampleAugmenter

        strategy = "greedy" if mode in {"dynamic", "dynamic-greedy"} else "uniform"
        if n_max_val is None or n_max_val <= 0:
            aug = SampleAugmenter(alpha=alpha, strategy=strategy, chunk_size=chunk_size)
        else:
            aug = SampleAugmenter(
                alpha=alpha,
                strategy=strategy,
                chunk_size=chunk_size,
                n_max=int(n_max_val),
            )
        aug = aug.fit(
            X_t_train,
            y_t_train,
            X_t_val,
            lbda=lbda_t,
            sigma=sigma_t,
        )
        X_c, y_c = aug.transform(
            l_X_s,
            l_y_s,
            sigma=sigma_s,
            lbda=lbda_s,
        )
        return _ridge(X_c, y_c, lbda_c), len(X_t_train), int(aug.get_n_selected())

    if mode in {"oracle_dynamic", "oracle-dynamic", "oracle-dynamic-greedy", "oracle-dynamic-uniform"}:
        from .augmenter import OracleSampleAugmenter

        strategy = "uniform" if mode == "oracle-dynamic-uniform" else "greedy"
        if n_max_val is None or n_max_val <= 0:
            ora = OracleSampleAugmenter(strategy=strategy, chunk_size=chunk_size)
        else:
            ora = OracleSampleAugmenter(
                n_max=int(n_max_val),
                strategy=strategy,
                chunk_size=chunk_size,
            )
        ora = ora.fit(
            X_t_train,
            y_t_train,
            X_t_val,
            lbda=lbda_t,
            theta_t_true=theta_t,
        )
        X_c, y_c = ora.transform(
            l_X_s,
            l_y_s,
            lbda=lbda_c,
        )
        return _ridge(X_c, y_c, lbda_c), len(X_t_train), int(ora.get_n_selected())

    if mode == "zhang_et_al":
        from .augmenter import UOWQSampleAugmenter

        if n_max_val is None or n_max_val <= 0:
            aug = UOWQSampleAugmenter()
        else:
            aug = UOWQSampleAugmenter(n_max=int(n_max_val))
        aug = aug.fit(
            X_t_train,
            y_t_train,
            X_t_val,
            lbda=lbda_t,
            sigma=sigma_t,
        )
        X_c, y_c = aug.transform(
            l_X_s,
            l_y_s,
            lbda=lbda_s,
            sigma=sigma_s,
        )
        return _ridge(X_c, y_c, lbda_c), len(X_t_train), int(aug.get_n_selected())

    if mode == "chen_et_al":
        X_s = np.vstack(l_X_s)
        y_s = np.concatenate(l_y_s)
        theta_t_ = _ridge(X=X_t_train, y=y_t_train, lbda=0.0)
        theta_s_ = _ridge(X=X_s, y=y_s, lbda=0.0)
        eye = np.eye(X_t_train.shape[1])
        G_t, G_s, G_t_val = X_t_train.T @ X_t_train, X_s.T @ X_s, X_t_val.T @ X_t_val
        G_s_inv_G_t = np.linalg.solve(G_s, G_t)
        lbda_grid = np.logspace(-3, 3, 100)
        lbda = _opt_lambda_corrected_aic(
            theta_t_,
            theta_s_,
            X_t_train,
            y_t_train,
            G_t,
            G_s_inv_G_t,
            G_t_val,
            eye,
            lbda_grid,
        )
        W = np.linalg.solve(
            G_t + lbda * (G_t_val @ G_s_inv_G_t) + lbda * G_t_val,
            G_t + lbda * (G_t_val @ G_s_inv_G_t),
        )
        return W @ theta_t_ + (eye - W) @ theta_s_, len(X_t_train), len(X_s)

    if mode == "obst_et_al":
        X_s = np.vstack(l_X_s)
        y_s = np.concatenate(l_y_s)
        fit_s = _fit_stats(X_s, y_s, lbda_s)
        fit_t = _fit_stats(X_t_train, y_t_train, lbda_t)
        alpha_ft = _default_alpha(fit_t[1], factor=10.0)
        k_grid = np.arange(0, 2001, 10, dtype=np.int64)
        Xk = X_t_val if (X_t_val is not None and X_t_val.size) else np.vstack([X_s, X_t_train])
        k_star = _select_k(Xk, fit_s, fit_t, alpha_ft, k_grid, k_min=10)
        A_k, _, _ = _make_cache(fit_t[1], alpha_ft, k_star)
        theta = _theta_k(fit_s[0], fit_t[0], A_k)
        return theta, len(X_t_train), len(X_s)

    if mode == "hierarchical":
        X_s = np.vstack(l_X_s)
        y_s = np.concatenate(l_y_s)
        theta = _hierarchical_theta(
            X_t_train=X_t_train,
            y_t_train=y_t_train,
            X_s=X_s,
            y_s=y_s,
            sigma_t=sigma_t,
            sigma_s=sigma_s,
            lbda_t=lbda_t,
            lbda_s=lbda_s,
            lbda_c=lbda_c,
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
    lbda_c: float,
    alpha: float = 0.05,
    mode: str = "target",
) -> Tuple[np.ndarray, int, int]:
    """Dispatch classification modes and return (w, n_t, n_s)."""
    _ = (theta_t, X_t_val, sigma_t, sigma_s, alpha)

    if mode == "target":
        return _ridge(X_t_train, y_t_train, lbda_t), len(X_t_train), 0

    if mode == "source":
        X_s = np.vstack(l_X_s)
        y_s = np.concatenate(l_y_s)
        return _ridge(X_s, y_s, lbda_s), 0, len(X_s)

    if mode == "full":
        X_s = np.vstack(l_X_s)
        y_s = np.concatenate(l_y_s)
        X_c = np.r_[X_t_train, X_s]
        y_c = np.r_[y_t_train, y_s]
        return _ridge(X_c, y_c, lbda_c), len(X_t_train), len(X_s)

    if mode == "hierarchical":
        X_s = np.vstack(l_X_s)
        y_s = np.concatenate(l_y_s)
        w = _hierarchical_theta(
            X_t_train=X_t_train,
            y_t_train=y_t_train,
            X_s=X_s,
            y_s=y_s,
            sigma_t=1.0,
            sigma_s=1.0,
            lbda_t=lbda_t,
            lbda_s=lbda_s,
            lbda_c=lbda_c,
        )
        return w, len(X_t_train), len(X_s)

    raise ValueError(f"Unknown mode: {mode}")


target_regr = partial(regr, mode="target")
source_regr = partial(regr, mode="source")
full_regr = partial(regr, mode="full")
dyn_greedy_regr = partial(regr, mode="dynamic-greedy")
dyn_uniform_regr = partial(regr, mode="dynamic-uniform")
oracle_dyn_regr = partial(regr, mode="oracle_dynamic")
zhang_regr = partial(regr, mode="zhang_et_al")
chen_regr = partial(regr, mode="chen_et_al")
obst_regr = partial(regr, mode="obst_et_al")
hier_regr = partial(regr, mode="hierarchical")

target_classif = partial(classif, mode="target")
source_classif = partial(classif, mode="source")
full_classif = partial(classif, mode="full")
hier_classif = partial(classif, mode="hierarchical")

target_regr.__doc__ = _REGR_DOC
source_regr.__doc__ = _REGR_DOC
full_regr.__doc__ = _REGR_DOC
dyn_greedy_regr.__doc__ = _REGR_DOC
dyn_uniform_regr.__doc__ = _REGR_DOC
oracle_dyn_regr.__doc__ = _REGR_DOC
zhang_regr.__doc__ = _REGR_DOC
chen_regr.__doc__ = _REGR_DOC
obst_regr.__doc__ = _REGR_DOC
hier_regr.__doc__ = _REGR_DOC

_CLASS_DOC = dedent(
    """
    Estimate the linear classifier from a combination of target and source.

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
    w : ndarray
        Estimated linear classifier weights.
    n_t : int
        Number of target samples used.
    n_s : int
        Number of source samples used.
    """
)

target_classif.__doc__ = _CLASS_DOC
source_classif.__doc__ = _CLASS_DOC
full_classif.__doc__ = _CLASS_DOC
hier_classif.__doc__ = _CLASS_DOC

__all__ = [
    "compute_delta_regr",
    "compute_delta_regr_variance",
    "compute_delta_regr_lower_bound",
    "opt_lambda",
    "regr",
    "classif",
    "target_regr",
    "source_regr",
    "full_regr",
    "dyn_greedy_regr",
    "dyn_uniform_regr",
    "oracle_dyn_regr",
    "zhang_regr",
    "chen_regr",
    "obst_regr",
    "hier_regr",
    "target_classif",
    "source_classif",
    "full_classif",
    "hier_classif",
]
