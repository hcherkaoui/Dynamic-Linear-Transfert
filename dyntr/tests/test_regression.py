"""Regression test coverage for dyntr.regression."""

import numpy as np
import pytest

from dyntr.regression import (
    target_single_regression,
    source_single_regression,
    full_collaboration_regression,
    dynamic_collaboration_regression,
    oracle_dynamic_collaboration_regression,
    chen_et_al_regression,
    obst_et_al_regression,
    hierical_regression,
    regression,
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
    theta_t, theta_s, X_t, X_s, y_t, y_s, X_t_val = _make_pair_data(rng)
    d = X_t.shape[1]

    lbda_t = 1.0
    lbda_s = 0.5
    lbda_c = lbda_t + lbda_s
    sigma_t = 0.1
    sigma_s = 0.2
    alpha = 0.1

    theta, n_t, n_s = target_single_regression(
        theta_t=theta_t,
        X_t=X_t,
        X_s=X_s,
        y_t=y_t,
        y_s=y_s,
        X_t_val=X_t_val,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        lbda_c=lbda_c,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        alpha=alpha,
    )
    _assert_theta(theta, d)
    assert n_t == len(X_t)
    assert n_s == 0

    theta, n_t, n_s = source_single_regression(
        theta_t=theta_t,
        X_t=X_t,
        X_s=X_s,
        y_t=y_t,
        y_s=y_s,
        X_t_val=X_t_val,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        lbda_c=lbda_c,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        alpha=alpha,
    )
    _assert_theta(theta, d)
    assert n_t == 0
    assert n_s == len(X_s)

    theta, n_t, n_s = full_collaboration_regression(
        theta_t=theta_t,
        X_t=X_t,
        X_s=X_s,
        y_t=y_t,
        y_s=y_s,
        X_t_val=X_t_val,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        lbda_c=lbda_c,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        alpha=alpha,
    )
    _assert_theta(theta, d)
    assert n_t == len(X_t)
    assert n_s == len(X_s)

    theta, n_t, n_s = dynamic_collaboration_regression(
        theta_t=theta_t,
        X_t=X_t,
        X_s=X_s,
        y_t=y_t,
        y_s=y_s,
        X_t_val=X_t_val,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        lbda_c=lbda_c,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        alpha=alpha,
    )
    _assert_theta(theta, d)
    assert n_t == len(X_t)
    assert 0 <= n_s <= len(X_s)

    theta, n_t, n_s = oracle_dynamic_collaboration_regression(
        theta_t=theta_t,
        X_t=X_t,
        X_s=X_s,
        y_t=y_t,
        y_s=y_s,
        X_t_val=X_t_val,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        lbda_c=lbda_c,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        alpha=alpha,
    )
    _assert_theta(theta, d)
    assert n_t == len(X_t)
    assert 0 <= n_s <= len(X_s)

    theta, n_t, n_s = chen_et_al_regression(
        theta_t=theta_t,
        X_t=X_t,
        X_s=X_s,
        y_t=y_t,
        y_s=y_s,
        X_t_val=X_t_val,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        lbda_c=lbda_c,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        alpha=alpha,
    )
    _assert_theta(theta, d)
    assert n_t == len(X_t)
    assert n_s == len(X_s)

    theta, n_t, n_s = obst_et_al_regression(
        theta_t=theta_t,
        X_t=X_t,
        X_s=X_s,
        y_t=y_t,
        y_s=y_s,
        X_t_val=X_t_val,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        lbda_c=lbda_c,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        alpha=alpha,
    )
    _assert_theta(theta, d)
    assert n_t == len(X_t)
    assert isinstance(n_s, int)

    theta, n_t, n_s = hierical_regression(
        theta_t=theta_t,
        X_t=X_t,
        X_s=X_s,
        y_t=y_t,
        y_s=y_s,
        X_t_val=X_t_val,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        lbda_c=lbda_c,
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
        regression(
            theta_t=theta_t,
            X_t=X_t,
            X_s=X_s,
            y_t=y_t,
            y_s=y_s,
            X_t_val=X_t_val,
            lbda_t=1.0,
            lbda_s=1.0,
            lbda_c=2.0,
            sigma_t=0.1,
            sigma_s=0.1,
            alpha=0.1,
            mode="unknown",
        )
