"""Utilities helpers module."""
from __future__ import annotations
import os
import random
from typing import Optional, Tuple
import numba
import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_array
from .linear import opt_lambda


def set_random_state(seed: int) -> None:
    """Fix randomness for the current experiment."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def check_random_state(seed):
    """Turn seed into a np.random.Generator instance."""
    if seed is None or seed is np.random:
        return np.random.default_rng(seed=None)
    if isinstance(seed, (int, np.integer)):
        return np.random.default_rng(seed=seed)
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError(
        f"{seed} cannot be used to seed a numpy.random.Generator instance"
    )


def format_duration(seconds: float) -> str:
    """Convert a duration in seconds to HH:MM:SS format."""
    s, ns = divmod(seconds, 1)
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s {int(1000.0 * ns):03d}ms"


def print_dataset_info(
    X_t,
    y_t,
    X_s,
    y_s,
    X_t_val,
    X_t_test,
    theta_t,
    sigma_t,
    sigma_s,
) -> None:
    """Helper to print dataset statistics."""
    print(
        "[INFO] Dataset statistics:\n"
        f"       d = {X_t.shape[1]}\n"
        f"       n_t = {len(X_t)}\n"
        f"       n_s = {len(X_s)}\n"
        f"       sigma_t = {sigma_t:.2e}\n"
        f"       sigma_s = {sigma_s:.2e}\n"
        f"       X_t shape = {X_t.shape}, y_t shape = {y_t.shape}\n"
        f"       X_s shape = {X_s.shape}, y_s shape = {y_s.shape}\n"
        f"       X_t_val shape = {X_t_val.shape}\n"
        f"       X_t_test shape = {X_t_test.shape}\n"
        f"       cond(X_t^T X_t)  =  {np.linalg.cond(X_t.T @ X_t):.2e}\n"
        f"       cond(X_s^T X_s)  =  {np.linalg.cond(X_s.T @ X_s):.2e}\n"
        f"       ||theta_t||^2    =  {np.linalg.norm(theta_t):.2e}\n"
    )


@numba.jit(nopython=True, cache=True, fastmath=True)
def std_estimation(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    mode=0,
    lbda=0.0,
    G=None,
    return_variance=False,
) -> float:
    """Estimate residual noise (std or variance) for a linear model.

    Parameters
    ----------
    theta : ndarray of shape (d,)
        Parameter vector.
    X : ndarray of shape (n, d)
        Design matrix.
    y : ndarray of shape (n,)
        Targets.
    mode = 0
        Degrees-of-freedom mode. ``0``: OLS ``(n - d)``;
        ``1``: ridge effective DoF ``tr(A (A + lbda I)^{-1})``.
    lbda = 0.0
        Ridge regularization parameter (used when ``mode = 1``).
    G : ndarray of shape (d, d), optional
        Gram matrix ``X^T X``. Must be provided when ``mode = 1``.
    return_variance = False
        If ``True``, return variance instead of standard deviation.

    Returns
    -------
    float
        Estimated noise standard deviation or variance.
        Returns ``nan`` if effective DoF ``<= 0``.
    """
    n, d = X.shape
    r = y - X.dot(theta)

    rss = 0.0
    for i in range(r.shape[0]):
        rss += r[i] * r[i]

    if mode == 0:
        dof = n - d
    else:
        M = np.linalg.solve(G + lbda * np.eye(d), G)
        df_lambda = 0.0
        for i in range(d):
            df_lambda += M[i, i]
        dof = n - df_lambda

    if dof <= 0.0:
        return np.nan

    var = rss / dof
    if return_variance:
        return var
    return np.sqrt(var)


def compute_exp_vars(
    X_t: np.ndarray,
    X_s: np.ndarray,
    y_t: np.ndarray,
    y_s: np.ndarray,
    X_t_val: np.ndarray,
    y_t_val: np.ndarray,
    lbda_s: float,
) -> dict:
    """Compute ridge estimators and noise levels for two datasets.

    Parameters
    ----------
    X_t, X_s : ndarray
        Training design matrices
    y_t, y_s : ndarray
        Targets, shapes
    X_t_val : ndarray
        Validation design matrix
    lbda_t, lbda_s, lbda_c : float
        Ridge penalty.

    Returns
    -------
    dict
        Dictionary containing
    theta_t, theta_s, theta_c, G_t, G_s, b_t, b_s, A_t_inv, A_s_inv, A_c_inv, sigma_t, sigma_s.
    """
    _, d = X_t.shape

    eye = np.eye(d)
    G_t = X_t.T.dot(X_t)
    G_s = X_s.T.dot(X_s)

    lbda_t = opt_lambda(X_t, y_t, X_t_val, y_t_val)
    lbda_c = lbda_t + lbda_s

    A_t_inv = np.linalg.inv(G_t + lbda_t * eye)
    A_s_inv = np.linalg.inv(G_s + lbda_s * eye)
    A_c_inv = np.linalg.inv(G_t + G_s + lbda_c * eye)

    b_t = X_t.T.dot(y_t)
    b_s = X_s.T.dot(y_s)

    theta_t = A_t_inv.dot(b_t)
    theta_s = A_s_inv.dot(b_s)
    theta_c = A_c_inv.dot(b_t + b_s)

    noise_std_t = std_estimation(theta_t, X_t, y_t, mode=1, lbda=lbda_t, G=G_t)
    noise_std_s = std_estimation(theta_s, X_s, y_s, mode=1, lbda=lbda_s, G=G_s)

    return {
        "theta_t": theta_t,
        "theta_s": theta_s,
        "theta_c": theta_c,
        "G_t": G_t,
        "G_s": G_s,
        "b_t": b_t,
        "b_s": b_s,
        "A_t_inv": A_t_inv,
        "A_s_inv": A_s_inv,
        "A_c_inv": A_c_inv,
        "sigma_t": noise_std_t,
        "sigma_s": noise_std_s,
        "lbda_t": lbda_t,
        "lbda_s": lbda_s,
        "lbda_c": lbda_c,
    }


def _as_1d_float(y: ArrayLike) -> np.ndarray:
    return np.asarray(y, dtype=float).reshape(-1)


def _check_X(X: ArrayLike, *, n_features: Optional[int] = None) -> np.ndarray:
    X = check_array(X, ensure_2d=True, dtype=float, accept_sparse=False)
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(
            f"Incompatible feature dimension: got {X.shape[1]}, expected {n_features}."
        )
    return X


def _check_Xy(X: ArrayLike, y: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    X = _check_X(X)
    y = _as_1d_float(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have inconsistent shapes: {X.shape[0]} != {y.shape[0]}"
        )
    return X, y
