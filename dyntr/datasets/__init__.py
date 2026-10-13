"""Datasets module."""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
from ._datasets import (
    _assert_is_regression,
    _bootstrap_sources,
    _condition_blocks,
    _generate_thetas,
)
from ._datasets import CHEMBERTA_EMBEDDING_DATASETS, EMBEDDING_REGRESSION_DATASETS  # noqa: F401
from ._datasets import EMBEDDING_REGRESSION_METADATA_KEYS  # noqa: F401
from ._datasets import _load_boston_dataset, _load_hillstrom_dataset  # noqa: F401
from ._datasets import _load_chemberta_embedding_dataset, _load_embedding_regression_dataset  # noqa: F401


_SYNTHETIC_NOISE_ALIASES = {
    "gaussian": "gaussian",
    "normal": "gaussian",
    "lognormal": "lognormal",
    "log-normal": "lognormal",
    "centered_lognormal": "lognormal",
    "centered-lognormal": "lognormal",
}


def _normalize_synthetic_noise(noise: str) -> str:
    """Normalize synthetic noise aliases."""
    try:
        normalized = _SYNTHETIC_NOISE_ALIASES[str(noise).lower()]
    except KeyError as exc:
        valid = sorted(_SYNTHETIC_NOISE_ALIASES)
        raise ValueError(f"noise must be one of {valid}; got {noise!r}.") from exc
    return normalized


def sample_synthetic_noise(
    rng: np.random.Generator,
    size,
    sigma: float,
    noise: str = "gaussian",
    lognormal_shape: float = 1.0,
) -> np.ndarray:
    """Sample additive synthetic residuals with standard deviation ``sigma``."""
    sigma = float(sigma)
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    noise = _normalize_synthetic_noise(noise)
    if noise == "gaussian":
        return sigma * rng.normal(size=size)

    lognormal_shape = float(lognormal_shape)
    if lognormal_shape <= 0:
        raise ValueError("lognormal_shape must be positive.")

    raw = rng.lognormal(
        mean=-0.5 * lognormal_shape ** 2,
        sigma=lognormal_shape,
        size=size,
    )
    return sigma * (raw - 1.0) / np.sqrt(np.expm1(lognormal_shape ** 2))


def synth_regr_data(
    n_t_train=10000,
    n_t_val=1000,
    n_t_test=1000,
    n_s_train=10000,
    n_s_val=1000,
    n_s_test=1000,
    d=30,
    eps=0.5,
    sigma_t=1.0,
    sigma_s=1.0,
    theta_t_and_theta_s=None,
    n_sources=1,
    noise="gaussian",
    lognormal_shape=1.0,
    random_state=None,
) -> Dict[str, Any]:
    """
    Generate paired synthetic linear datasets (target, sources).

    The features of each dataset are i.i.d. standard normal. Targets follow
    linear models with additive noise:
        y_t = X_t θ_t + sigma_t ε_t, y_s = X_s θ_s + sigma_s ε_s.

    Parameters
    ----------
    n_t_train, n_t_val, n_t_test, n_s_train, n_s_val, n_s_val
        Number of samples per dataset.
    d = 30
        Number of features.
    eps = 0.5
        Second coordinate for θ_s when `theta_t_and_theta_s` is None; θ_s := (1, eps, 0, ...).
    sigma_t = 1.0
        Noise standard deviation for dataset target.
    sigma_s = 1.0
        Noise standard deviation for dataset source.
    theta_t_and_theta_s : tuple of array-like, optional
        Custom parameters (θ_t, θ_s). If provided, both are broadcast to shape (d,)
        and used as-is.
    n_sources : int,
        number of source (with the SAME θ_s)
    noise : {"gaussian", "lognormal"}
        Additive residual distribution. The log-normal option is centered and
        variance-normalized so sigma_t and sigma_s keep their std meaning.
    lognormal_shape : float
        Shape parameter for log-normal residuals.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    out : dict
        Data and metadata with keys:
            - 'target_train' : tuple
            - 'target_val' : tuple
            - 'target_test' : tuple
            - 'source_train' : tuple
            - 'source_val' : tuple
            - 'source_test' : tuple
            - 'meta' : dict
                Metadata with keys:
                    - 'theta_t' : ndarray
                    - 'theta_s' : ndarray
                    - 'sigma_t', 'sigma_s' : float
                    - 'n_train_samples', 'n_val_samples', 'n_test_samples', 'd', 'eps' : int
                    - 'random_state' : int or None
    """
    if d <= 0:
        raise ValueError("d must be positive.")
    if n_t_train <= 0 or n_t_val <= 0 or n_t_test <= 0:
        raise ValueError("n_t_train must be > 0 and n_t_val > 0 and n_t_test > 0.")
    if n_s_train <= 0 or n_s_val <= 0 or n_s_test <= 0:
        raise ValueError("n_s_train must be > 0 and n_s_val > 0 and n_s_test > 0.")
    if sigma_t < 0 or sigma_s < 0:
        raise ValueError("sigma_t and sigma_s must be non-negative.")
    noise = _normalize_synthetic_noise(noise)
    lognormal_shape = float(lognormal_shape)
    if noise == "lognormal" and lognormal_shape <= 0:
        raise ValueError("lognormal_shape must be positive.")

    rng = np.random.default_rng(random_state)

    X_t_train = rng.normal(size=(n_t_train, d)).astype(float, copy=False)
    X_t_val = rng.normal(size=(n_t_val, d)).astype(float, copy=False)
    X_t_test = rng.normal(size=(n_t_test, d)).astype(float, copy=False)

    X_s_train = rng.normal(size=(n_s_train, d)).astype(float, copy=False)
    X_s_val = rng.normal(size=(n_s_val, d)).astype(float, copy=False)
    X_s_test = rng.normal(size=(n_s_test, d)).astype(float, copy=False)

    if theta_t_and_theta_s is None:
        if d < 2 and eps != 0.0:
            raise ValueError("d must be at least 2 when using default θ_s with nonzero eps.")
        theta_s, theta_t = _generate_thetas(d, eps, random_state=random_state)
    else:
        theta_t = np.asarray(theta_t_and_theta_s[0], dtype=float).reshape(-1)
        theta_s = np.asarray(theta_t_and_theta_s[1], dtype=float).reshape(-1)
        if theta_t.size != d or theta_s.size != d:
            raise ValueError(f"theta_t_and_theta_s must have length d = {d}.")

    y_t_train = (X_t_train @ theta_t) + sample_synthetic_noise(
        rng, size=n_t_train, sigma=sigma_t, noise=noise, lognormal_shape=lognormal_shape
    )
    y_t_val = (X_t_val @ theta_t) + sample_synthetic_noise(
        rng, size=n_t_val, sigma=sigma_t, noise=noise, lognormal_shape=lognormal_shape
    )
    y_t_test = (X_t_test @ theta_t) + sample_synthetic_noise(
        rng, size=n_t_test, sigma=sigma_t, noise=noise, lognormal_shape=lognormal_shape
    )

    y_s_train = (X_s_train @ theta_s) + sample_synthetic_noise(
        rng, size=n_s_train, sigma=sigma_s, noise=noise, lognormal_shape=lognormal_shape
    )
    y_s_val = (X_s_val @ theta_s) + sample_synthetic_noise(
        rng, size=n_s_val, sigma=sigma_s, noise=noise, lognormal_shape=lognormal_shape
    )
    y_s_test = (X_s_test @ theta_s) + sample_synthetic_noise(
        rng, size=n_s_test, sigma=sigma_s, noise=noise, lognormal_shape=lognormal_shape
    )

    l_X_s_train, l_y_s_train = _bootstrap_sources(
        X_s_train,
        y_s_train,
        n_sources=n_sources,
        rng=rng,
    )
    l_X_s_val, l_y_s_val = _bootstrap_sources(
        X_s_val,
        y_s_val,
        n_sources=n_sources,
        rng=rng,
    )
    l_X_s_test, l_y_s_test = _bootstrap_sources(
        X_s_test,
        y_s_test,
        n_sources=n_sources,
        rng=rng,
    )

    out = {
        "target_train": (X_t_train, y_t_train),
        "target_val": (X_t_val, y_t_val),
        "target_test": (X_t_test, y_t_test),
        "source_train": (l_X_s_train, l_y_s_train),
        "source_val": (l_X_s_val, l_y_s_val),
        "source_test": (l_X_s_test, l_y_s_test),
        "meta": {
            "theta_t": theta_t.copy(),
            "theta_s": theta_s.copy(),
            "sigma_t": float(sigma_t),
            "sigma_s": float(sigma_s),
            "n_t_train": int(n_t_train),
            "n_t_val": int(n_t_val),
            "n_t_test": int(n_t_test),
            "n_s_train": int(n_s_train),
            "n_s_val": int(n_s_val),
            "n_s_test": int(n_s_test),
            "d": int(d),
            "eps": float(eps),
            "noise": noise,
            "lognormal_shape": float(lognormal_shape),
            "random_state": random_state if isinstance(random_state, int) else None,
        },
    }
    return out


def synth_classif_data(
    n_t_train=10000,
    n_t_val=1000,
    n_t_test=1000,
    n_s_train=10000,
    n_s_val=1000,
    n_s_test=1000,
    d=30,
    eps=0.5,
    sigma_t=1.0,
    sigma_s=1.0,
    theta_t_and_theta_s=None,
    n_sources=1,
    noise="gaussian",
    lognormal_shape=1.0,
    random_state=None,
) -> Dict[str, Any]:
    """
    Generate paired synthetic linear classification datasets (target, sources).

    Labels are in {-1, +1} based on the sign of X @ theta plus synthetic noise.
    """
    data = synth_regr_data(
        n_t_train=n_t_train,
        n_t_val=n_t_val,
        n_t_test=n_t_test,
        n_s_train=n_s_train,
        n_s_val=n_s_val,
        n_s_test=n_s_test,
        d=d,
        eps=eps,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        theta_t_and_theta_s=theta_t_and_theta_s,
        n_sources=n_sources,
        noise=noise,
        lognormal_shape=lognormal_shape,
        random_state=random_state,
    )

    def _binarize(y):
        """Binarize regression targets by sign."""
        y = np.asarray(y)
        return np.where(y >= 0.0, 1.0, -1.0)

    X_t, y_t = data["target_train"]
    X_t_val, y_t_val = data["target_val"]
    X_t_test, y_t_test = data["target_test"]
    l_X_s_train, l_y_s_train = data["source_train"]
    l_X_s_val, l_y_s_val = data["source_val"]
    l_X_s_test, l_y_s_test = data["source_test"]

    data["target_train"] = (X_t, _binarize(y_t))
    data["target_val"] = (X_t_val, _binarize(y_t_val))
    data["target_test"] = (X_t_test, _binarize(y_t_test))
    data["source_train"] = (l_X_s_train, [_binarize(y) for y in l_y_s_train])
    data["source_val"] = (l_X_s_val, [_binarize(y) for y in l_y_s_val])
    data["source_test"] = (l_X_s_test, [_binarize(y) for y in l_y_s_test])
    data["meta"]["task"] = "classification"
    return data


def list_real_regression_datasets() -> Dict[str, Dict[str, Any]]:
    """Return metadata for datasets accepted by ``real_regr_data``."""
    out = {
        "boston": {
            "filename": None,
            "n_samples": 20640,
            "d": 8,
            "target": "median house value",
            "feature_type": "tabular",
            "source": "California Housing (historical boston alias)",
            "default_split": "median income split into target/source pools",
            "description": "California Housing regression exposed as the legacy boston dataset.",
            "cache_required": False,
        },
        "email": {
            "filename": "hillstrom.csv",
            "n_samples": None,
            "d": None,
            "target": "spend",
            "feature_type": "tabular",
            "source": "Hillstrom email marketing dataset",
            "default_split": "high-value versus low-value customer split",
            "description": "Email marketing spend regression from the Hillstrom dataset.",
            "cache_required": False,
        },
    }
    for name, spec in EMBEDDING_REGRESSION_DATASETS.items():
        row = {key: spec[key] for key in EMBEDDING_REGRESSION_METADATA_KEYS}
        row["cache_required"] = True
        out[name] = row
    return {name: values.copy() for name, values in out.items()}


def real_regr_data(
    dataset_name: str,
    n_t_train=500,
    n_t_val=200,
    n_t_test=200,
    n_s_train=500,
    n_s_val=200,
    n_s_test=200,
    sigma_t=1.0,
    sigma_s=0.0,
    permutation=True,
    *,
    standardize=True,
    var_threshold=1e-8,
    n_sources=1,
    random_state=None,
) -> Dict[str, Any]:
    """
    Load a pair of related datasets (target and sources) and return train/val/test splits.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier. Supported values are "boston", "email", or any
        name registered in EMBEDDING_REGRESSION_DATASETS, including
        "lipophilicity_chemberta", "aqsoldb_chemberta",
        "moleculenet_lipo_ecfp", "moleculenet_esol_ecfp", and
        "moleculenet_freesolv_ecfp", "tape_fluorescence_esm2", and
        "tape_stability_esm2", "amazon_reviews_sentence", and
        "yelp_reviews_sentence", "e2006_tfidf_svd512",
        "gdsc1_expression_fingerprint", and "gdsc2_expression_fingerprint".
        Embedding datasets are loaded from local .npz caches.
    n_t_train, n_t_val, n_t_test
        Number of target samples for train/val/test.
    n_s_train, n_s_val, n_s_test
        Number of source samples for train/val/test.
    sigma_t, sigma_s : float
        Optional noise standard deviation added to target/source labels.
    permutation : bool
        Whether to shuffle samples before slicing.
    standardize : bool
        Whether to standardize features (shared transform for target/source).
    var_threshold : float
        Threshold for low-variance feature removal.
    n_sources : int
        Number of bootstrapped source datasets to return.
    random_state : int | None
        Seed for reproducibility.

    Returns
    -------
    out : dict
        Data and metadata with keys:
            - 'target_train' : tuple
            - 'target_val' : tuple
            - 'target_test' : tuple
            - 'source_train' : tuple
            - 'source_val' : tuple
            - 'source_test' : tuple
            - 'meta' : dict
    """
    valid_names = {"boston", "email"} | set(EMBEDDING_REGRESSION_DATASETS)
    if dataset_name not in valid_names:
        raise ValueError(
            f"Dataset '{dataset_name}' is not supported. "
            f"Valid values: {sorted(valid_names)}"
        )

    if dataset_name == "boston":
        train_data, test_data = _load_boston_dataset()
        add_intercept = True
        pca_var_threshold = 0.75
    elif dataset_name == "email":
        train_data, test_data = _load_hillstrom_dataset()
        add_intercept = True
        pca_var_threshold = 0.75
    elif dataset_name in EMBEDDING_REGRESSION_DATASETS:
        train_data, test_data = _load_embedding_regression_dataset(dataset_name)
        add_intercept = False
        pca_var_threshold = None
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    (X_t_pool, y_t_pool), (X_s_pool, y_s_pool) = train_data
    (X_t_test_pool, y_t_test_pool), (X_s_test_pool, y_s_test_pool) = test_data

    _assert_is_regression(np.r_[y_t_pool, y_t_test_pool], "y_t")
    _assert_is_regression(np.r_[y_s_pool, y_s_test_pool], "y_s")

    rng = np.random.default_rng(random_state)
    if permutation:
        for X, y in (
            (X_t_pool, y_t_pool),
            (X_s_pool, y_s_pool),
            (X_t_test_pool, y_t_test_pool),
            (X_s_test_pool, y_s_test_pool),
        ):
            idx = rng.permutation(len(X))
            X[:] = X[idx]
            y[:] = y[idx]

    n_tr_t = min(n_t_train + n_t_val, X_t_pool.shape[0])
    n_tr_s = min(n_s_train + n_s_val, X_s_pool.shape[0])
    n_te_t = min(n_t_test, X_t_test_pool.shape[0])
    n_te_s = min(n_s_test, X_s_test_pool.shape[0])

    X_t_pool, y_t_pool = X_t_pool[:n_tr_t], y_t_pool[:n_tr_t]
    X_s_pool, y_s_pool = X_s_pool[:n_tr_s], y_s_pool[:n_tr_s]
    X_t_test, y_t_test = X_t_test_pool[:n_te_t], y_t_test_pool[:n_te_t]
    X_s_test, y_s_test = X_s_test_pool[:n_te_s], y_s_test_pool[:n_te_s]

    n_t_val = min(n_t_val, X_t_pool.shape[0])
    n_s_val = min(n_s_val, X_s_pool.shape[0])

    X_t_val, y_t_val = X_t_pool[:n_t_val], y_t_pool[:n_t_val]
    X_s_val, y_s_val = X_s_pool[:n_s_val], y_s_pool[:n_s_val]
    X_t_train = X_t_pool[n_t_val: n_t_val + n_t_train]
    y_t_train = y_t_pool[n_t_val: n_t_val + n_t_train]
    X_s_train = X_s_pool[n_s_val: n_s_val + n_s_train]
    y_s_train = y_s_pool[n_s_val: n_s_val + n_s_train]

    X_t_valtest = np.concatenate([X_t_val, X_t_test], axis=0) if len(X_t_val) else X_t_test
    X_s_valtest = np.concatenate([X_s_val, X_s_test], axis=0) if len(X_s_val) else X_s_test

    X_t_train, X_s_train, X_t_valtest, X_s_valtest = _condition_blocks(
        X_t_train,
        X_s_train,
        X_t_valtest,
        X_s_valtest,
        standardize=standardize,
        pca_var_threshold=pca_var_threshold,
        var_threshold=var_threshold,
        add_intercept=add_intercept,
    )

    if len(X_t_val):
        X_t_val_c, X_t_test_c = X_t_valtest[:len(X_t_val)], X_t_valtest[len(X_t_val):]
        X_s_val_c, X_s_test_c = X_s_valtest[:len(X_s_val)], X_s_valtest[len(X_s_val):]
    else:
        X_t_val_c, X_s_val_c = X_t_val, X_s_val
        X_t_test_c, X_s_test_c = X_t_valtest, X_s_valtest

    def _as64(*arrs):
        """Convert arrays to contiguous float64 arrays."""
        return [np.ascontiguousarray(a, dtype=np.float64) for a in arrs]

    if sigma_t is not None and sigma_t > 0.0:
        y_t_train += sigma_t * rng.normal(size=y_t_train.shape)
        y_t_val += sigma_t * rng.normal(size=y_t_val.shape)
        y_t_test += sigma_t * rng.normal(size=y_t_test.shape)

    if sigma_s is not None and sigma_s > 0.0:
        y_s_train += sigma_s * rng.normal(size=y_s_train.shape)
        y_s_val += sigma_s * rng.normal(size=y_s_val.shape)
        y_s_test += sigma_s * rng.normal(size=y_s_test.shape)

    X_t_train, X_s_train = _as64(X_t_train, X_s_train)
    y_t_train, y_s_train = _as64(y_t_train, y_s_train)

    n_sources = int(n_sources)
    if n_sources <= 0:
        raise ValueError("n_sources must be positive.")

    l_X_s_train, l_y_s_train = _bootstrap_sources(
        X_s_train,
        y_s_train,
        n_sources=n_sources,
        rng=rng,
    )
    l_X_s_val, l_y_s_val = _bootstrap_sources(X_s_val_c, y_s_val, n_sources=n_sources, rng=rng)
    l_X_s_test, l_y_s_test = _bootstrap_sources(X_s_test_c, y_s_test, n_sources=n_sources, rng=rng)

    out = {
        "target_train": (X_t_train, y_t_train),
        "target_val": (X_t_val_c, y_t_val),
        "target_test": (X_t_test_c, y_t_test),
        "source_train": (l_X_s_train, l_y_s_train),
        "source_val": (l_X_s_val, l_y_s_val),
        "source_test": (l_X_s_test, l_y_s_test),
        "meta": {
            "dataset": dataset_name,
            "sigma_t": float(sigma_t) if sigma_t is not None else None,
            "sigma_s": float(sigma_s) if sigma_s is not None else None,
            "n_t_train": int(X_t_train.shape[0]),
            "n_t_val": int(X_t_val_c.shape[0]),
            "n_t_test": int(X_t_test_c.shape[0]),
            "n_s_train": int(X_s_train.shape[0]),
            "n_s_val": int(X_s_val_c.shape[0]),
            "n_s_test": int(X_s_test_c.shape[0]),
            "d": int(X_t_train.shape[1]),
            "random_state": random_state if isinstance(random_state, int) else None,
        },
    }
    return out
