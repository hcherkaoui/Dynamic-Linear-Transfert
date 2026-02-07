"""Tests for dyntr.utils."""

from __future__ import annotations

import numpy as np
import pytest

from dyntr import utils


def test_check_random_state_variants():
    rng = utils.check_random_state(123)
    assert isinstance(rng, np.random.Generator)

    rng2 = np.random.default_rng(0)
    assert utils.check_random_state(rng2) is rng2

    with pytest.raises(ValueError):
        utils.check_random_state("bad")


def test_format_duration():
    assert utils.format_duration(0.0) == "00h 00m 00s 000ms"
    assert utils.format_duration(1.234) == "00h 00m 01s 234ms"


def test_std_estimation_basic():
    X = np.eye(4)
    theta = np.zeros(4)
    y = np.zeros(4)
    std = utils.std_estimation(theta, X, y, mode=0, return_variance=False, G=X.T @ X)
    var = utils.std_estimation(theta, X, y, mode=0, return_variance=True, G=X.T @ X)
    assert np.isnan(std)
    assert np.isnan(var)


def test_std_estimation_mode1_and_dof_nan():
    X = np.eye(2)
    theta = np.zeros(2)
    y = np.zeros(2)
    G = X.T @ X
    std = utils.std_estimation(theta, X, y, mode=1, lbda=1.0, G=G)
    assert std >= 0.0

    X_small = np.eye(3)
    y_small = np.zeros(3)
    std_nan = utils.std_estimation(theta=np.zeros(3), X=X_small, y=y_small, mode=0, G=X_small.T @ X_small)
    assert np.isnan(std_nan)


def test_std_estimation_mode1_requires_G():
    X = np.eye(2)
    theta = np.zeros(2)
    y = np.zeros(2)
    with pytest.raises(Exception):
        utils.std_estimation(theta, X, y, mode=1, lbda=1.0, G=None)


def test_compute_exp_vars_shapes():
    rng = np.random.default_rng(0)
    X_t = rng.normal(size=(10, 4))
    X_s = rng.normal(size=(12, 4))
    theta = rng.normal(size=4)
    y_t = X_t @ theta + rng.normal(size=10)
    y_s = X_s @ theta + rng.normal(size=12)
    X_t_val = rng.normal(size=(5, 4))
    y_t_val = X_t_val @ theta + rng.normal(size=5)

    out = utils.compute_exp_vars(
        X_t=X_t,
        X_s=X_s,
        y_t=y_t,
        y_s=y_s,
        X_t_val=X_t_val,
        y_t_val=y_t_val,
        lbda_s=0.1,
    )

    assert out["theta_t"].shape == (4,)
    assert out["theta_s"].shape == (4,)
    assert out["G_t"].shape == (4, 4)
    assert out["G_s"].shape == (4, 4)
