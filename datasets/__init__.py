"""Datasets module."""

# Authors: Hamza Cherkaoui

from __future__ import annotations
from typing import Tuple, Union, Optional, Dict, Any
import numpy as np
from numpy.typing import ArrayLike
from ._datasets import _condition_blocks, _assert_is_regression
from ._datasets import (_load_boston_dataset, _load_hillstrom_dataset,
                        _load_open_ecommerce_dataset, _load_wine_quality_dataset)


Array4 = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def generate_thetas(
    d: int,
    epsilon: float,
    random_state=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return theta_s, theta_t such that ||theta_s - theta_t||_2 = epsilon."""
    if d <= 0:
        raise ValueError("d must be positive.")
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")

    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, (int, np.integer)):
        rng = np.random.default_rng(random_state)

    theta_s = rng.normal(size=d)
    v = rng.normal(size=d)

    return theta_s, theta_s + (epsilon / np.linalg.norm(v)) * v


def fetch_synthetic_datasets(
    n_train_samples: int = 10000,
    n_val_samples: int = 1000,
    n_test_samples: int = 1000,
    d: int = 30,
    eps: float = 0.5,
    sigma_t: float = 1.0,
    sigma_s: float = 1.0,
    theta_t_and_theta_s: Optional[Tuple[ArrayLike, ArrayLike]] = None,
    only_train: bool = False,
    random_state: Optional[int] = None,
) -> Union[Tuple[Array4, Dict[str, Any]], Tuple[Array4, Array4, Array4, Dict[str, Any]]]:
    """
    Generate paired synthetic linear datasets (target, source).

    The features of each dataset are i.i.d. standard normal. Targets follow
    linear models with additive Gaussian noise:
        y_t = X_t θ_t + sigma_t ε_t,   y_s = X_s θ_s + sigma_s ε_s.

    Parameters
    ----------
    n_train_samples : int, default=10000
        Number of training samples per dataset.
    n_val_samples : int, default=10000
        Number of validation samples per dataset.
    n_test_samples : int, default=1000
        Number of test samples per dataset.
    d : int, default=30
        Number of features.
    eps : float, default=0.5
        Second coordinate for θ_s when `theta_t_and_theta_s` is None; θ_s := (1, eps, 0, ...).
    sigma_t : float, default=1.0
        Noise standard deviation for dataset target.
    sigma_s : float, default=1.0
        Noise standard deviation for dataset source.
    theta_t_and_theta_s : tuple of array-like, optional
        Custom parameters (θ_t, θ_s). If provided, both are broadcast to shape (d,)
        and used as-is.
    only_train : bool, default=False
        If True, return only the training split and metadata.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    train_out : tuple
        (X_t_train, X_s_train, y_t_train, y_s_train).
    val_out : tuple
        (X_t_val, X_s_val, y_t_val, y_s_val). Returned only if `only_train=False`.
    test_out : tuple
        (X_t_test, X_s_test, y_t_test, y_s_test). Returned only if `only_train=False`.
    meta : dict
        Metadata with keys:
            - 'theta_t' : ndarray of shape (d,)
            - 'theta_s' : ndarray of shape (d,)
            - 'sigma_t', 'sigma_s' : float
            - 'n_train_samples', 'n_val_samples', 'n_test_samples', 'd', 'eps'
            - 'random_state' : int or None
    """
    if d <= 0:
        raise ValueError("d must be positive.")
    if n_train_samples <= 0 or n_test_samples < 0 or n_val_samples < 0:
        raise ValueError("n_train_samples must be > 0 and n_val_samples >= 0 and n_test_samples >= 0.")
    if sigma_t < 0 or sigma_s < 0:
        raise ValueError("sigma_t and sigma_s must be non-negative.")

    rng = np.random.default_rng(random_state)

    X_t_train = rng.normal(size=(n_train_samples, d)).astype(float, copy=False)
    X_s_train = rng.normal(size=(n_train_samples, d)).astype(float, copy=False)
    X_t_val = rng.normal(size=(n_val_samples, d)).astype(float, copy=False) if n_val_samples > 0 else np.empty((0, d))
    X_s_val = rng.normal(size=(n_val_samples, d)).astype(float, copy=False) if n_val_samples > 0 else np.empty((0, d))
    X_t_test = rng.normal(size=(n_test_samples, d)).astype(float, copy=False) if n_test_samples > 0 else np.empty((0, d))
    X_s_test = rng.normal(size=(n_test_samples, d)).astype(float, copy=False) if n_test_samples > 0 else np.empty((0, d))

    if theta_t_and_theta_s is None:
        if d < 2 and eps != 0.0:
            raise ValueError("d must be at least 2 when using default θ_s with nonzero eps.")
        theta_s, theta_t = generate_thetas(d, eps, random_state=random_state)
    else:
        t_t, t_s = (np.asarray(theta_t_and_theta_s[0], dtype=float).reshape(-1),
                    np.asarray(theta_t_and_theta_s[1], dtype=float).reshape(-1))
        if t_t.size != d or t_s.size != d:
            raise ValueError(f"theta_t_and_theta_s must have length d={d}.")
        theta_t, theta_s = t_t, t_s

    y_t_train = (X_t_train @ theta_t) + sigma_t * rng.normal(size=n_train_samples)
    y_s_train = (X_s_train @ theta_s) + sigma_s * rng.normal(size=n_train_samples)
    y_t_val = (X_t_val @ theta_t) + sigma_t * rng.normal(size=n_val_samples) if n_val_samples > 0 else np.empty((0,))
    y_s_val = (X_s_val @ theta_s) + sigma_s * rng.normal(size=n_val_samples) if n_val_samples > 0 else np.empty((0,))
    y_t_test = (X_t_test @ theta_t) + sigma_t * rng.normal(size=n_test_samples) if n_test_samples > 0 else np.empty((0,))
    y_s_test = (X_s_test @ theta_s) + sigma_s * rng.normal(size=n_test_samples) if n_test_samples > 0 else np.empty((0,))

    train_out = (X_t_train.copy(), X_s_train.copy(), y_t_train.copy(), y_s_train.copy())
    val_out = (X_t_val.copy(), X_s_val.copy(), y_t_val.copy(), y_s_val.copy())
    test_out = (X_t_test.copy(), X_s_test.copy(), y_t_test.copy(), y_s_test.copy())

    meta: Dict[str, Any] = {
        "theta_t": theta_t.copy(),
        "theta_s": theta_s.copy(),
        "sigma_t": float(sigma_t),
        "sigma_s": float(sigma_s),
        "n_train_samples": int(n_train_samples),
        "n_test_samples": int(n_test_samples),
        "d": int(d),
        "eps": float(eps),
        "random_state": random_state if isinstance(random_state, int) else None,
    }

    if only_train:
        return train_out, meta

    return (train_out, val_out, test_out), meta


def fetch_real_datasets(
    dataset_name: str,
    n_train_samples: int = 500,
    n_test_samples: int = 100,
    sigma_t: float | None = 1.0,
    permutation: bool = True,
    only_train: bool = True,
    *,
    standardize: bool = True,
    var_threshold: float = 1e-8,
    val_fraction: float = 0.3,
) -> Array4 | Tuple[Array4, Array4, Array4]:
    """
    Load a pair of related datasets (target and source),
    create an optional validation split, and condition the designs to avoid
    ill-conditioning.

    Returns
    -------
    - If only_train is True:
        train_out
    - Else:
        (train_out, val_out, test_out)

    where each *_out is a 4-tuple: (X_t, X_s, y_t, y_s).
    """
    valid_names = {"boston", "email", "ecommerce", "wine"}
    if dataset_name not in valid_names:
        raise ValueError(f"Dataset '{dataset_name}' is not supported. "
                         f"Valid values: {sorted(valid_names)}")

    if dataset_name == "boston":
        train_data, test_data = _load_boston_dataset()
        add_intercept = True

    elif dataset_name == "email":
        train_data, test_data = _load_hillstrom_dataset()
        add_intercept = True

    elif dataset_name == "ecommerce":
        train_data, test_data = _load_open_ecommerce_dataset()
        add_intercept = False

    elif dataset_name == "wine":
        train_data, test_data = _load_wine_quality_dataset()
        add_intercept = True

    (X_t_train, y_t_train), (X_s_train, y_s_train) = train_data
    (X_t_test, y_t_test), (X_s_test, y_s_test) = test_data

    _assert_is_regression(np.r_[y_t_train, y_t_test], "y_t")
    _assert_is_regression(np.r_[y_s_train, y_s_test], "y_s")

    if permutation:
        for X, y in ((X_t_train, y_t_train), (X_s_train, y_s_train),
                     (X_t_test, y_t_test), (X_s_test, y_s_test)):
            idx = np.random.permutation(len(X))
            X[:] = X[idx]
            y[:] = y[idx]

    n_tr_t = min(n_train_samples, X_t_train.shape[0])
    n_tr_s = min(n_train_samples, X_s_train.shape[0])
    n_te_t = min(n_test_samples, X_t_test.shape[0])
    n_te_s = min(n_test_samples, X_s_test.shape[0])

    X_t_train, y_t_train = X_t_train[:n_tr_t], y_t_train[:n_tr_t]
    X_s_train, y_s_train = X_s_train[:n_tr_s], y_s_train[:n_tr_s]
    X_t_test, y_t_test = X_t_test[:n_te_t], y_t_test[:n_te_t]
    X_s_test, y_s_test = X_s_test[:n_te_s], y_s_test[:n_te_s]

    v1 = int(np.floor(val_fraction * len(X_t_train))) if not only_train else 0
    v2 = int(np.floor(val_fraction * len(X_s_train))) if not only_train else 0

    X_t_val, y_t_val = X_t_train[:v1], y_t_train[:v1]
    X_s_val, y_s_val = X_s_train[:v2], y_s_train[:v2]
    X_t_train, y_t_train = X_t_train[v1:], y_t_train[v1:]
    X_s_train, y_s_train = X_s_train[v2:], y_s_train[v2:]

    if not only_train:
        X_t_valtest = np.concatenate([X_t_val, X_t_test], axis=0) if len(X_t_val) else X_t_test
        X_s_valtest = np.concatenate([X_s_val, X_s_test], axis=0) if len(X_s_val) else X_s_test

        X_t_train, X_s_train, X_t_valtest, X_s_valtest = _condition_blocks(
            X_t_train, X_s_train, X_t_valtest, X_s_valtest,
            standardize=standardize,
            pca_var_threshold=0.75,
            var_threshold=var_threshold,
            add_intercept=add_intercept,
        )

        if len(X_t_val):
            X_t_val_c, X_t_test_c = X_t_valtest[:len(X_t_val)], X_t_valtest[len(X_t_val):]
            X_s_val_c, X_s_test_c = X_s_valtest[:len(X_s_val)], X_s_valtest[len(X_s_val):]
        else:
            X_t_val_c, X_s_val_c = X_t_val, X_s_val  # empty
            X_t_test_c, X_s_test_c = X_t_valtest, X_s_valtest
    else:
        X_t_train, X_s_train, _, _ = _condition_blocks(
            X_t_train, X_s_train, X_t_test, X_s_test,
            standardize=standardize,
            pca_var_threshold=0.75,
            var_threshold=var_threshold,
            add_intercept=add_intercept,
        )

    def _as64(*arrs):
        return [np.ascontiguousarray(a, dtype=np.float64) for a in arrs]

    if sigma_t is not None and sigma_t > 0.0:
        y_t_train += sigma_t * np.random.randn(*y_t_train.shape)
        y_t_val += sigma_t * np.random.randn(*y_t_val.shape)
        y_t_test += sigma_t * np.random.randn(*y_t_test.shape)

    X_t_train, X_s_train = _as64(X_t_train, X_s_train)
    y_t_train, y_s_train = _as64(y_t_train, y_s_train)

    train_out: Array4 = (X_t_train, X_s_train, y_t_train, y_s_train)

    if only_train:
        return train_out

    X_t_val_c, X_s_val_c = _as64(X_t_val_c, X_s_val_c)
    y_t_val, y_s_val = _as64(y_t_val, y_s_val)
    val_out: Array4 = (X_t_val_c, X_s_val_c, y_t_val, y_s_val)

    X_t_test_c, X_s_test_c = _as64(X_t_test_c, X_s_test_c)
    y_t_test, y_s_test = _as64(y_t_test, y_s_test)
    test_out: Array4 = (X_t_test_c, X_s_test_c, y_t_test, y_s_test)

    return train_out, val_out, test_out
