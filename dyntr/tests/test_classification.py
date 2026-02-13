"""Classification tests."""
from __future__ import annotations
import numpy as np
from dyntr.linear import (
    dyn_greedy_classif,
    full_classif,
    hier_classif,
    oracle_greedy_classif,
    source_classif,
    target_classif,
)
from dyntr.datasets import synth_classif_data


def _make_data(rng, n_t=20, n_s=30, d=4):
    theta_t = rng.normal(size=d)
    theta_s = rng.normal(size=d)
    X_t = rng.normal(size=(n_t, d))
    X_s = rng.normal(size=(n_s, d))
    y_t = np.where(X_t @ theta_t >= 0.0, 1.0, -1.0)
    y_s = np.where(X_s @ theta_s >= 0.0, 1.0, -1.0)
    X_t_val = rng.normal(size=(8, d))
    return theta_t, theta_s, X_t, X_s, y_t, y_s, X_t_val


def test_classification_modes_run():
    rng = np.random.default_rng(0)
    theta_t, theta_s, X_t, X_s, y_t, y_s, X_t_val = _make_data(rng)
    lbda_t = 1.0
    lbda_s = 0.5

    w, n_t, n_s = target_classif(
        theta_t=theta_t,
        X_t_train=X_t,
        y_t_train=y_t,
        X_t_val=X_t_val,
        l_X_s=[X_s],
        l_y_s=[y_s],
        sigma_t=0.1,
        sigma_s=0.1,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        alpha=0.1,
    )
    assert w.shape == (X_t.shape[1],)
    assert n_t == len(X_t)
    assert n_s == 0

    w, n_t, n_s = source_classif(
        theta_t=theta_t,
        X_t_train=X_t,
        y_t_train=y_t,
        X_t_val=X_t_val,
        l_X_s=[X_s],
        l_y_s=[y_s],
        sigma_t=0.1,
        sigma_s=0.1,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        alpha=0.1,
    )
    assert w.shape == (X_t.shape[1],)
    assert n_t == 0
    assert n_s == len(X_s)

    w, n_t, n_s = full_classif(
        theta_t=theta_t,
        X_t_train=X_t,
        y_t_train=y_t,
        X_t_val=X_t_val,
        l_X_s=[X_s],
        l_y_s=[y_s],
        sigma_t=0.1,
        sigma_s=0.1,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        alpha=0.1,
    )
    assert w.shape == (X_t.shape[1],)
    assert n_t == len(X_t)
    assert n_s == len(X_s)

    w, n_t, n_s = dyn_greedy_classif(
        theta_t=theta_t,
        X_t_train=X_t,
        y_t_train=y_t,
        X_t_val=X_t_val,
        l_X_s=[X_s],
        l_y_s=[y_s],
        sigma_t=0.1,
        sigma_s=0.1,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        alpha=0.1,
    )
    assert w.shape == (X_t.shape[1],)
    assert n_t == len(X_t)
    assert 0 <= n_s <= len(X_s)

    w, n_t, n_s = oracle_greedy_classif(
        theta_t=theta_t,
        X_t_train=X_t,
        y_t_train=y_t,
        X_t_val=X_t_val,
        l_X_s=[X_s],
        l_y_s=[y_s],
        sigma_t=0.1,
        sigma_s=0.1,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        alpha=0.1,
    )
    assert w.shape == (X_t.shape[1],)
    assert n_t == len(X_t)
    assert 0 <= n_s <= len(X_s)

    w, n_t, n_s = hier_classif(
        theta_t=theta_t,
        X_t_train=X_t,
        y_t_train=y_t,
        X_t_val=X_t_val,
        l_X_s=[X_s],
        l_y_s=[y_s],
        sigma_t=0.1,
        sigma_s=0.1,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        alpha=0.1,
    )
    assert w.shape == (X_t.shape[1],)
    assert n_t == len(X_t)
    assert n_s == len(X_s)


def test_fetch_synthetic_classification_datasets():
    data = synth_classif_data(
        n_t_train=10,
        n_t_val=5,
        n_t_test=5,
        n_s_train=10,
        n_s_val=5,
        n_s_test=5,
        d=3,
        n_sources=2,
        random_state=0,
    )
    _, y_t = data["target_train"]
    _, y_t_val = data["target_val"]
    _, y_t_test = data["target_test"]
    _, l_y_s = data["source_train"]

    assert set(np.unique(y_t)).issubset({-1.0, 1.0})
    assert set(np.unique(y_t_val)).issubset({-1.0, 1.0})
    assert set(np.unique(y_t_test)).issubset({-1.0, 1.0})
    assert len(l_y_s) == 2
    assert set(np.unique(l_y_s[0])).issubset({-1.0, 1.0})
