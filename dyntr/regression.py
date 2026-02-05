"""Regression module."""

# Authors: Hamza Cherkaoui

from __future__ import annotations
from functools import partial
from textwrap import dedent
from typing import Tuple
import numpy as np
import numba
from .preprocessing import GreedySampleAugmenter, OracleGreedySampleAugmenter


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
    lbda: float
):
    n, d = X.shape
    G = X.T @ X + lbda * np.eye(d)
    G_inv = np.linalg.inv(G)
    theta = G_inv @ (X.T @ y)
    r = y - X @ theta
    s2 = float(r.T @ r) / max(n - d, 1)
    return theta, G, G_inv, s2, n, d


def _default_alpha(
    G: np.ndarray,
    factor: float = 10.0,
) -> float:
    w = np.linalg.eigvalsh(G)
    lam_min = max(float(w[0]), 1e-12)
    lam_max = float(w[-1])
    return 2.0 / (lam_max + lam_min) / factor


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
    k_grid,
) -> int:
    _, _, G_s_inv, s2_s, _, _ = fit_s
    _, G_t, _, s2_t, _, _ = fit_t
    best_k, best = None, -np.inf

    for k in k_grid:
        A_k, Omega_k, G_t_inv = _make_cache(G_t, alpha, k)
        V = _V_k(s2_s, s2_t, A_k, Omega_k, G_t, G_s_inv, alpha)
        score = _U_k(Xk, G_t_inv, V, s2_t)
        if score > best:
            best, best_k = score, k

    return int(best_k)


@numba.jit(nopython=True, cache=True, fastmath=True)
def _opt_lambda_corrected_aic(
    theta_t: np.ndarray,
    theta_s: np.ndarray,
    X_t: np.ndarray,
    y_t: np.ndarray,
    G_t: np.ndarray,
    G_s_inv_G_t: np.ndarray,
    G_t_val: np.ndarray,
    eye: np.ndarray,
    lbda_grid: np.ndarray,
) -> float:
    n, _ = X_t.shape
    best_aicc, best_lbda = np.inf, lbda_grid[0]
    for lbda in lbda_grid:
        W = np.linalg.solve(
            G_t + lbda * (G_t_val @ G_s_inv_G_t) + lbda * G_t_val,
            G_t + lbda * (G_t_val @ G_s_inv_G_t),
        )
        theta_c = W @ theta_t + (eye - W) @ theta_s
        df = np.trace(W)
        r = y_t - X_t @ theta_c
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


def _hierical_theta(
    X_t: np.ndarray,
    y_t: np.ndarray,
    X_s: np.ndarray,
    y_s: np.ndarray,
    sigma_t: float,
    sigma_s: float,
    lbda_t: float,
    lbda_s: float,
    lbda_c: float,
) -> np.ndarray:
    """
    Two-task hierarchical Bayesian linear regression (Gaussian–Gaussian model).

    Prior:
        mu ~ N(0, (lambda_0 I)^{-1})
        beta_t | mu ~ N(mu, (lambda_b I)^{-1})
        beta_s | mu ~ N(mu, (lambda_b I)^{-1})

    Likelihood:
        y_t ~ N(X_t beta_t, sigma_t^2 I)
        y_s ~ N(X_s beta_s, sigma_s^2 I)

    Returns posterior mean (MAP) for beta_t.
    """
    n_t, d = X_t.shape
    n_s, d_s = X_s.shape
    if d_s != d:
        raise ValueError("X_t and X_s must have the same number of columns.")

    # Stabilize variances
    sig2_t = max(float(sigma_t) * float(sigma_t), 1e-12)
    sig2_s = max(float(sigma_s) * float(sigma_s), 1e-12)

    # Use ridge parameters as hyperparameters for the hierarchy
    lambda_0 = max(float(lbda_c), 1e-8)
    lambda_b = max(0.5 * (float(lbda_t) + float(lbda_s)), 1e-8)

    G_t = (X_t.T @ X_t) / sig2_t
    G_s = (X_s.T @ X_s) / sig2_s
    g_t = (X_t.T @ y_t) / sig2_t
    g_s = (X_s.T @ y_s) / sig2_s

    eye = np.eye(d)
    gamma = (lambda_b ** 2) / (lambda_0 + 2.0 * lambda_b)
    c = lambda_b * (lambda_0 + lambda_b) / (lambda_0 + 2.0 * lambda_b)

    H_tt = G_t + c * eye
    H_ss = G_s + c * eye

    # Block elimination: solve for beta_t only
    H_ss_inv = np.linalg.inv(H_ss) if n_s > 0 else (1.0 / c) * eye
    B = H_tt - (gamma ** 2) * H_ss_inv
    rhs = g_t + gamma * (H_ss_inv @ g_s)

    beta_t = np.linalg.solve(B, rhs)
    return beta_t


_REGR_DOC = dedent(
    """
    Estimate the linear parameter from a combination of target and source.

    Parameters
    ----------
    theta_t : ndarray of shape (d,)
        Ground-truth parameters for the target dataset (oracle mode only).
    X_t, X_s : ndarray
        Training feature matrices.
    y_t, y_s : ndarray of shape (n_t,), (n_s,)
        Targets.
    X_t_val : ndarray
        Validation features for target.
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


def regression(
    theta_t: np.ndarray,
    X_t: np.ndarray,
    X_s: np.ndarray,
    y_t: np.ndarray,
    y_s: np.ndarray,
    X_t_val: np.ndarray,
    lbda_t: float,
    lbda_s: float,
    lbda_c: float,
    sigma_t: float,
    sigma_s: float,
    alpha: float,
    mode: str,
) -> Tuple[np.ndarray, int, int]:
    if mode == "target":
        return _ridge(X_t, y_t, lbda_t), len(X_t), 0

    if mode == "source":
        return _ridge(X_s, y_s, lbda_s), 0, len(X_s)

    if mode == "full":
        return _ridge(np.r_[X_t, X_s], np.r_[y_t, y_s], lbda_c), len(X_t), len(X_s)

    if mode == "dynamic":
        aug = GreedySampleAugmenter(alpha=alpha).fit(X_t, y_t, X_t_val, lbda=lbda_t, sigma=sigma_t)
        X_c, y_c = aug.transform(X_s, y_s, sigma=sigma_s, lbda=lbda_s)
        return _ridge(X_c, y_c, lbda_c), len(X_t), int(aug.get_n_selected())

    if mode == "oracle_dynamic":
        ora = OracleGreedySampleAugmenter().fit(
            X_t, y_t, X_t_val, lbda=lbda_t, theta_t_true=theta_t
        )
        X_c, y_c = ora.transform(X_s, y_s, lbda=lbda_c)
        return _ridge(X_c, y_c, lbda_c), len(X_t), int(ora.get_n_selected())

    if mode == "chen_et_al":
        theta_t_ = _ridge(X=X_t, y=y_t, lbda=0.0)
        theta_s_ = _ridge(X=X_s, y=y_s, lbda=0.0)
        eye = np.eye(X_t.shape[1])
        G_t, G_s, G_t_val = X_t.T @ X_t, X_s.T @ X_s, X_t_val.T @ X_t_val
        G_s_inv_G_t = np.linalg.solve(G_s, G_t)
        lbda_grid = np.logspace(-3, 3, 100)
        lbda = _opt_lambda_corrected_aic(
            theta_t_, theta_s_, X_t, y_t, G_t, G_s_inv_G_t, G_t_val, eye, lbda_grid
        )
        W = np.linalg.solve(
            G_t + lbda * (G_t_val @ G_s_inv_G_t) + lbda * G_t_val,
            G_t + lbda * (G_t_val @ G_s_inv_G_t),
        )
        return W @ theta_t_ + (eye - W) @ theta_s_, len(X_t), len(X_s)

    if mode == "obst_et_al":
        fit_s = _fit_stats(X_s, y_s, lbda_s)
        fit_t = _fit_stats(X_t, y_t, lbda_t)
        alpha = _default_alpha(fit_t[1], factor=10.0)
        k_grid = range(0, 50, 100)
        Xk = X_t_val if X_t_val is not None and X_t_val.size else np.vstack([X_s, X_t])
        k_star = _select_k(Xk, fit_s, fit_t, alpha, k_grid)
        A_k, _, _ = _make_cache(fit_t[1], alpha, k_star)
        theta = _theta_k(fit_s[0], fit_t[0], A_k)
        return theta, len(X_t), k_star

    if mode == "hierical":
        theta = _hierical_theta(
            X_t=X_t,
            y_t=y_t,
            X_s=X_s,
            y_s=y_s,
            sigma_t=sigma_t,
            sigma_s=sigma_s,
            lbda_t=lbda_t,
            lbda_s=lbda_s,
            lbda_c=lbda_c,
        )
        return theta, len(X_t), len(X_s)

    raise ValueError(f"Unknown mode: {mode}")


target_single_regression = partial(regression, mode="target")
source_single_regression = partial(regression, mode="source")
full_collaboration_regression = partial(regression, mode="full")
dynamic_collaboration_regression = partial(regression, mode="dynamic")
oracle_dynamic_collaboration_regression = partial(regression, mode="oracle_dynamic")
chen_et_al_regression = partial(regression, mode="chen_et_al")
obst_et_al_regression = partial(regression, mode="obst_et_al")
hierical_regression = partial(regression, mode="hierical")

target_single_regression.__doc__ = _REGR_DOC
source_single_regression.__doc__ = _REGR_DOC
full_collaboration_regression.__doc__ = _REGR_DOC
dynamic_collaboration_regression.__doc__ = _REGR_DOC
oracle_dynamic_collaboration_regression.__doc__ = _REGR_DOC
chen_et_al_regression.__doc__ = _REGR_DOC
obst_et_al_regression.__doc__ = _REGR_DOC
hierical_regression.__doc__ = _REGR_DOC

__all__ = [
    "target_single_regression",
    "source_single_regression",
    "full_collaboration_regression",
    "dynamic_collaboration_regression",
    "oracle_dynamic_collaboration_regression",
    "chen_et_al_regression",
    "obst_et_al_regression",
    "hierical_regression",
]
