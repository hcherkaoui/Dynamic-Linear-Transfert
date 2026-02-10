"""Tests for dyntr.augmenter."""
from __future__ import annotations

import numpy as np
import pytest

from dyntr.augmenter import OracleSampleAugmenter, SampleAugmenter


def _make_data(rng, n_t=20, n_s=30, d=5):
    X_t = rng.normal(size=(n_t, d))
    X_s = rng.normal(size=(n_s, d))
    theta_t = rng.normal(size=d)
    theta_s = rng.normal(size=d)
    y_t = X_t @ theta_t + rng.normal(scale=0.1, size=n_t)
    y_s = X_s @ theta_s + rng.normal(scale=0.1, size=n_s)
    X_t_val = rng.normal(size=(10, d))
    return X_t, y_t, X_s, y_s, X_t_val, theta_t


def test_greedy_sample_augmenter_fit_transform():
    rng = np.random.default_rng(0)
    X_t, y_t, X_s, y_s, X_t_val, _ = _make_data(rng)
    aug = SampleAugmenter(n_max=10, alpha=0.1, random_state=0)
    aug.fit(X_t, y_t, X_t_val, lbda=1.0, sigma=0.1)
    Xc, yc = aug.transform([X_s], [y_s], lbda=0.5, sigma=0.2)
    assert Xc.shape[0] == yc.shape[0]
    assert Xc.shape[1] == X_t.shape[1]
    assert aug.get_n_selected() >= 0


def test_greedy_sample_augmenter_bad_shapes():
    rng = np.random.default_rng(0)
    X_t, y_t, X_s, y_s, X_t_val, _ = _make_data(rng)
    aug = SampleAugmenter(n_max=10)
    with pytest.raises(ValueError):
        aug.fit(X_t, y_t[:-1], X_t_val, lbda=1.0, sigma=0.1)
    aug.fit(X_t, y_t, X_t_val, lbda=1.0, sigma=0.1)
    with pytest.raises(ValueError):
        aug.transform([X_s[:, :-1]], [y_s], lbda=0.5, sigma=0.2)


def test_greedy_sample_augmenter_bad_nmax():
    rng = np.random.default_rng(0)
    X_t, y_t, _, _, X_t_val, _ = _make_data(rng)
    aug = SampleAugmenter(n_max=0)
    with pytest.raises(ValueError):
        aug.fit(X_t, y_t, X_t_val, lbda=1.0, sigma=0.1)


def test_oracle_greedy_sample_augmenter_fit_transform():
    rng = np.random.default_rng(1)
    X_t, y_t, X_s, y_s, X_t_val, theta_t = _make_data(rng)
    aug = OracleSampleAugmenter(n_max=8, random_state=0)
    aug.fit(X_t, y_t, X_t_val, lbda=1.0, theta_t_true=theta_t)
    Xc, yc = aug.transform([X_s], [y_s], lbda=1.5)
    assert Xc.shape[0] == yc.shape[0]
    assert Xc.shape[1] == X_t.shape[1]
    assert aug.get_n_selected() >= 0


def test_oracle_greedy_sample_augmenter_bad_shapes():
    rng = np.random.default_rng(2)
    X_t, y_t, X_s, y_s, X_t_val, theta_t = _make_data(rng)
    aug = OracleSampleAugmenter(n_max=5)
    with pytest.raises(ValueError):
        aug.fit(X_t, y_t, X_t_val, lbda=1.0, theta_t_true=np.zeros(X_t.shape[1] + 1))
    aug.fit(X_t, y_t, X_t_val, lbda=1.0, theta_t_true=theta_t)
    with pytest.raises(ValueError):
        aug.transform([X_s[:, :-1]], [y_s], lbda=1.0)
