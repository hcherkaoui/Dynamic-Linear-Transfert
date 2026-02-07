"""Tests for dyntr.samplers."""

from __future__ import annotations

import numpy as np

from dyntr.samplers import (
    sample_uniform,
    sample_ridge_leverage,
    sample_maxvol_greedy,
    sample_weighted_sparsifier,
)


def _make_X():
    rng = np.random.default_rng(0)
    return rng.normal(size=(30, 5))


def test_sample_uniform_shapes():
    X = _make_X()
    idx, w = sample_uniform(X, m=10, replace=True, rng=0)
    assert idx.shape == (10,)
    assert w.shape == (10,)


def test_sample_ridge_leverage_shapes():
    X = _make_X()
    idx, w = sample_ridge_leverage(X, m=8, lam=0.1, rng=0)
    assert idx.shape == (8,)
    assert w.shape == (8,)
    assert np.all(w > 0)


def test_sample_ridge_leverage_zero_scores_fallback():
    X = np.zeros((10, 3))
    idx, w = sample_ridge_leverage(X, m=5, lam=1.0, rng=0)
    assert idx.shape == (5,)
    assert np.allclose(w, 1.0)


def test_sample_maxvol_greedy():
    X = _make_X()
    idx, w = sample_maxvol_greedy(X, m=6)
    assert idx.shape[0] <= 6
    assert w.shape == (idx.shape[0],)


def test_sample_weighted_sparsifier():
    X = _make_X()
    idx, w = sample_weighted_sparsifier(X, m=6, tau=1e-6)
    assert idx.shape[0] <= 6
    assert w.shape == (idx.shape[0],)
    assert np.all(w >= 0)
