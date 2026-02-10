"""Stats helpers module."""
from __future__ import annotations

import numba
import numpy as np


@numba.jit(nopython=True, cache=True, fastmath=True)
def compute_delta(
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
    """
    Compute
        Δ_hat(n)  =  || λ_t U_T hat{θ}_T ||^2
                - || V ( G(n)(hat{θ}_S - hat{θ}_T) - λ_t hat{θ}_T ) ||^2
                + tr( U_T M_T U_T^T ) - tr( V N(n) V^T )

    Parameters
    ----------
    theta_t, theta_s : np.ndarray,
        Estimate parameter for task target and source.
    G_s, G_t : np.ndarray,
        Gram matrices for task target and source.
    A_t_inv, A_s_inv, A_c_inv : np.ndarray,
        Inverse regularized Gram matrices for task target and source.
    X_val : np.ndarray
        Test target dataset.
    sigma_s, sigma_t : float,
        Noise standard-deviation for task target and source.
    lbda_t, lbda_s : float,
        Ridge regularization parameter.

    Returns
    -------
    Δ_hat(n) : float
        Estimated transfer gain
    """
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
def compute_delta_variance(
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
    """
    Compute Var(Δ_hat(n)) with
      Var(Δ_hat(n))  =  2 Tr( (D Σ)^2 ) + 4 μ^T D Σ D μ

    Parameters
    ----------
    theta_s, theta_t : np.ndarray,
        Estimate parameter for task target and source.
    G_s, G_t : np.ndarray,
        Gram matrices for task target and source.
    A_t_inv, A_s_inv, A_c_inv : np.ndarray,
        Inverse regularized Gram matrices for task target and source.
    X_val : np.ndarray
        Test target dataset.
    sigma_s, sigma_t : float,
        Noise standard-deviation for task target and source.
    lbda_t, lbda_s : float,
        Ridge regularization parameter.

    Returns
    -------
    Var(Δ_hat(n)) : float
        Estimated transfer gain
    """
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
def compute_delta_lower_bound(
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
    """
    Compute a lower bound of Δ_hat(n).

    Inspired from the one-sided Chebyshev test: Δ_hat(n) - sqrt(Var(Δ_hat(n)) * (1-δ)/δ)

    Parameters
    ----------
    theta_t, theta_s : np.ndarray,
        Estimate parameter for task target and source.
    G_s, G_t : np.ndarray,
        Gram matrices for task target and source.
    A_t_inv, A_s_inv : np.ndarray,
        Inverse regularized Gram matrices for task target and source.
    X_val : np.ndarray
        Test target dataset.
    sigma_s, sigma_t : float,
        Noise standard-deviation for task target and source.
    lbda_t, lbda_s : float,
        Ridge regularization parameter.

    Returns
    -------
    upper-bound : float
        Delta upper-bound.
    """
    delta_hat = compute_delta(
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
    var_delta_hat = compute_delta_variance(
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
    """
    Grid-search λ to minimize: J(λ)  =  || X_val (θ̂_t - θ_t) ||^2

    Parameters
    ----------
    X: ndarray
        Training feature matrix
    y: ndarray
        Training feature matrix
    theta: ndarray
        True linear parameter
    X_val: ndarray
        Validation feature matrix
    lbda_grid: ndarray
        Lambda search grid

    Returns
    -------
    best_lbda : float
    """
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
