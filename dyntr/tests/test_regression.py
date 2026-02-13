"""Regression test coverage for dyntr.linear."""
import numpy as np
import pytest
from dyntr.linear import (
    chen_regr,
    dyn_greedy_regr,
    full_regr,
    hier_regr,
    obst_regr,
    oracle_greedy_regr,
    regr,
    source_regr,
    target_regr,
)


def _make_pair_data(rng, n_t=12, n_s=14, d=3, sigma_t=0.1, sigma_s=0.2):
    theta_t = rng.normal(size=d)
    theta_s = rng.normal(size=d)
    X_t = rng.normal(size=(n_t, d))
    X_s = rng.normal(size=(n_s, d))
    y_t = X_t @ theta_t + sigma_t * rng.normal(size=n_t)
    y_s = X_s @ theta_s + sigma_s * rng.normal(size=n_s)
    X_t_val = rng.normal(size=(6, d))
    return theta_t, theta_s, X_t, X_s, y_t, y_s, X_t_val


def _assert_theta(theta, d):
    assert isinstance(theta, np.ndarray)
    assert theta.shape == (d,)


def test_regression_modes_run():
    """Smoke-test all regression modes."""
    rng = np.random.default_rng(0)
    theta_t, _, X_t, X_s, y_t, y_s, X_t_val = _make_pair_data(rng)
    d = X_t.shape[1]
    lbda_t = 1.0
    lbda_s = 0.5
    sigma_t = 0.1
    sigma_s = 0.2
    alpha = 0.1

    theta, n_t, n_s = target_regr(
        theta_t=theta_t,
        X_t_train=X_t,
        l_X_s=[X_s],
        y_t_train=y_t,
        l_y_s=[y_s],
        X_t_val=X_t_val,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        alpha=alpha,
    )
    _assert_theta(theta, d)
    assert n_t == len(X_t)
    assert n_s == 0

    theta, n_t, n_s = source_regr(
        theta_t=theta_t,
        X_t_train=X_t,
        l_X_s=[X_s],
        y_t_train=y_t,
        l_y_s=[y_s],
        X_t_val=X_t_val,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        alpha=alpha,
    )
    _assert_theta(theta, d)
    assert n_t == 0
    assert n_s == len(X_s)

    theta, n_t, n_s = full_regr(
        theta_t=theta_t,
        X_t_train=X_t,
        l_X_s=[X_s],
        y_t_train=y_t,
        l_y_s=[y_s],
        X_t_val=X_t_val,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        alpha=alpha,
    )
    _assert_theta(theta, d)
    assert n_t == len(X_t)
    assert n_s == len(X_s)

    theta, n_t, n_s = dyn_greedy_regr(
        theta_t=theta_t,
        X_t_train=X_t,
        l_X_s=[X_s],
        y_t_train=y_t,
        l_y_s=[y_s],
        X_t_val=X_t_val,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        alpha=alpha,
    )
    _assert_theta(theta, d)
    assert n_t == len(X_t)
    assert 0 <= n_s <= len(X_s)

    theta, n_t, n_s = oracle_greedy_regr(
        theta_t=theta_t,
        X_t_train=X_t,
        l_X_s=[X_s],
        y_t_train=y_t,
        l_y_s=[y_s],
        X_t_val=X_t_val,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        alpha=alpha,
    )
    _assert_theta(theta, d)
    assert n_t == len(X_t)
    assert 0 <= n_s <= len(X_s)

    theta, n_t, n_s = regr(
        theta_t=theta_t,
        X_t_train=X_t,
        l_X_s=[X_s],
        y_t_train=y_t,
        l_y_s=[y_s],
        X_t_val=X_t_val,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        alpha=alpha,
        mode="oracle-dynamic-uniform",
    )
    _assert_theta(theta, d)
    assert n_t == len(X_t)
    assert 0 <= n_s <= len(X_s)

    theta, n_t, n_s = chen_regr(
        theta_t=theta_t,
        X_t_train=X_t,
        l_X_s=[X_s],
        y_t_train=y_t,
        l_y_s=[y_s],
        X_t_val=X_t_val,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        alpha=alpha,
    )
    _assert_theta(theta, d)
    assert n_t == len(X_t)
    assert n_s == len(X_s)

    theta, n_t, n_s = obst_regr(
        theta_t=theta_t,
        X_t_train=X_t,
        l_X_s=[X_s],
        y_t_train=y_t,
        l_y_s=[y_s],
        X_t_val=X_t_val,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        alpha=alpha,
    )
    _assert_theta(theta, d)
    assert n_t == len(X_t)
    assert isinstance(n_s, int)

    theta, n_t, n_s = hier_regr(
        theta_t=theta_t,
        X_t_train=X_t,
        l_X_s=[X_s],
        y_t_train=y_t,
        l_y_s=[y_s],
        X_t_val=X_t_val,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        alpha=alpha,
    )
    _assert_theta(theta, d)
    assert n_t == len(X_t)
    assert n_s == len(X_s)


def test_regression_unknown_mode_raises():
    """Unknown regression mode should raise."""
    rng = np.random.default_rng(0)
    theta_t, _, X_t, X_s, y_t, y_s, X_t_val = _make_pair_data(rng)
    with pytest.raises(ValueError):
        regr(
            theta_t=theta_t,
            X_t_train=X_t,
            y_t_train=y_t,
            X_t_val=X_t_val,
            l_X_s=[X_s],
            l_y_s=[y_s],
            lbda_t=1.0,
            lbda_s=1.0,
            sigma_t=0.1,
            sigma_s=0.1,
            alpha=0.1,
            mode="unknown",
        )
