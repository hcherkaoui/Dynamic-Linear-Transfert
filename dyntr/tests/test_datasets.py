"""Tests for dyntr.datasets with mocked CSV inputs."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import dyntr.datasets as datasets_mod
from dyntr.datasets import synth_regr_data
from dyntr.datasets._datasets import (
    _assert_is_regression,
    _autodetect_user_id,
    _condition_blocks,
    _generate_thetas,
    _guess_column,
    _load_boston_dataset,
    _load_hillstrom_dataset,
)


def test_fetch_synthetic_datasets_basic():
    out = synth_regr_data(
        n_t_train=20,
        n_t_val=5,
        n_t_test=5,
        n_s_train=20,
        n_s_val=5,
        n_s_test=5,
        d=4,
        n_sources=3,
    )
    X_t, y_t = out["target_train"]
    X_t_val, y_t_val = out["target_val"]
    X_t_test, y_t_test = out["target_test"]
    l_X_s, l_y_s = out["source_train"]
    l_X_s_val, l_y_s_val = out["source_val"]
    l_X_s_test, l_y_s_test = out["source_test"]
    meta = out["meta"]

    assert X_t.shape == (20, 4)
    assert y_t.shape == (20,)
    assert X_t_val.shape == (5, 4)
    assert y_t_val.shape == (5,)
    assert X_t_test.shape == (5, 4)
    assert y_t_test.shape == (5,)

    assert isinstance(l_X_s, list)
    assert len(l_X_s) == 3
    assert l_X_s[0].shape == (20, 4)

    assert isinstance(l_y_s, list)
    assert len(l_y_s) == 3
    assert l_y_s[0].shape == (20,)

    assert len(l_X_s_val) == 3
    assert l_X_s_val[0].shape == (5, 4)

    assert len(l_y_s_val) == 3
    assert l_y_s_val[0].shape == (5,)

    assert len(l_X_s_test) == 3
    assert l_X_s_test[0].shape == (5, 4)

    assert len(l_y_s_test) == 3
    assert l_y_s_test[0].shape == (5,)

    assert meta["theta_t"].shape == (4,)
    assert meta["theta_s"].shape == (4,)


def test_fetch_synthetic_datasets_invalid():
    with pytest.raises(ValueError):
        synth_regr_data(d=0)
    with pytest.raises(ValueError):
        synth_regr_data(n_t_train=0)
    with pytest.raises(ValueError):
        synth_regr_data(d=1, eps=0.5)
    with pytest.raises(ValueError):
        synth_regr_data(n_s_train=0)
    with pytest.raises(ValueError):
        synth_regr_data(sigma_t=-0.1)
    with pytest.raises(ValueError):
        synth_regr_data(theta_t_and_theta_s=(np.zeros(3), np.zeros(2)), d=3)


def test_generate_thetas_validation():
    with pytest.raises(AssertionError):
        _generate_thetas(0, 0.5)
    with pytest.raises(AssertionError):
        _generate_thetas(2, -0.1)

    theta_s, theta_t = _generate_thetas(3, 0.0, random_state=0)
    assert theta_s.shape == (3,)
    assert theta_t.shape == (3,)
    assert np.isclose(np.linalg.norm(theta_s - theta_t), 0.0)

    rng = np.random.default_rng(123)
    theta_s, theta_t = _generate_thetas(2, 0.1, random_state=rng)
    assert theta_s.shape == (2,)
    assert theta_t.shape == (2,)


def test_condition_blocks_intercept_and_variance():
    X1_tr = np.array([[1.0, 0.0], [1.0, 0.0]])
    X2_tr = np.array([[2.0, 0.0], [2.0, 0.0]])
    X1_te = np.array([[1.5, 0.0]])
    X2_te = np.array([[2.5, 0.0]])

    X1_tr_c, X2_tr_c, X1_te_c, X2_te_c = _condition_blocks(
        X1_tr,
        X2_tr,
        X1_te,
        X2_te,
        pca_var_threshold=None,
        standardize=False,
        var_threshold=1e-6,
        add_intercept=True,
    )
    assert X1_tr_c.shape[1] == X2_tr_c.shape[1]
    assert X1_te_c.shape[1] == X2_te_c.shape[1]
    assert X1_tr_c.shape[1] >= 1


def test_assert_is_regression_rejects_binary():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    with pytest.raises(AssertionError):
        _assert_is_regression(y, "y")


def test_bootstrap_sources_behavior(monkeypatch):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 2))
    y = rng.normal(size=5)
    l_X, l_y = datasets_mod._bootstrap_sources(X, y, n_sources=1, rng=rng)
    assert len(l_X) == 1
    assert len(l_y) == 1
    assert l_X[0].shape == X.shape
    assert l_y[0].shape == y.shape

    l_X2, l_y2 = datasets_mod._bootstrap_sources(X, y, n_sources=2, rng=rng)
    assert len(l_X2) == 2
    assert len(l_y2) == 2


def test_fetch_real_datasets_mocked(monkeypatch):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    y = rng.normal(size=10)
    train_data = (X[:5], y[:5]), (X[5:], y[5:])
    test_data = (X[:3], y[:3]), (X[3:6], y[3:6])
    monkeypatch.setattr(datasets_mod, "_load_boston_dataset", lambda: (train_data, test_data))

    out = datasets_mod.real_regr_data(
        "boston",
        n_t_train=3,
        n_t_val=1,
        n_t_test=2,
        n_s_train=3,
        n_s_val=1,
        n_s_test=2,
        n_sources=2,
        permutation=True,
        sigma_t=0.0,
        sigma_s=0.0,
        random_state=0,
    )
    X_t, l_X_s, y_t, l_y_s = (
        out["target_train"][0],
        out["source_train"][0],
        out["target_train"][1],
        out["source_train"][1],
    )

    assert X_t.shape[0] == 3
    assert isinstance(l_X_s, list)
    assert len(l_X_s) == 2
    assert l_X_s[0].shape[0] == 3
    assert y_t.shape[0] == 3
    assert isinstance(l_y_s, list)
    assert len(l_y_s) == 2
    assert l_y_s[0].shape[0] == 3

    with pytest.raises(ValueError):
        datasets_mod.real_regr_data("unknown")

    with pytest.raises(ValueError):
        datasets_mod.real_regr_data(
            "boston",
            n_t_train=1,
            n_t_val=0,
            n_t_test=1,
            n_s_train=1,
            n_s_val=0,
            n_s_test=1,
            n_sources=0,
            permutation=False,
        )

    out = datasets_mod.real_regr_data(
        "boston",
        n_t_train=2,
        n_t_val=0,
        n_t_test=1,
        n_s_train=2,
        n_s_val=0,
        n_s_test=1,
        n_sources=1,
        permutation=True,
        sigma_t=0.1,
        sigma_s=0.1,
        random_state=0,
    )
    assert out["target_val"][0].shape[0] == 0
    assert out["source_val"][0][0].shape[0] == 0


def test_load_hillstrom_dataset_mocked(monkeypatch):
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "history": [0.0, 10.0, 5.0, 2.0, 8.0, 9.0, 1.0, 3.0],
            "spend": [0.0, 2.0, 1.0, 0.5, 1.5, 2.5, 0.2, 0.7],
            "segment": [
                "Womens E-Mail",
                "Mens E-Mail",
                "Control",
                "Control",
                "Womens E-Mail",
                "Mens E-Mail",
                "Control",
                "Control",
            ],
        }
    )

    def _fake_read_csv(_path):
        return df.copy()

    monkeypatch.setattr(pd, "read_csv", _fake_read_csv)
    monkeypatch.setattr("os.path.exists", lambda _p: True)

    train_data, test_data = _load_hillstrom_dataset(split="high_value_vs_low_value")
    (X1_tr, y1_tr), (X2_tr, y2_tr) = train_data
    (X1_te, y1_te), (X2_te, y2_te) = test_data
    assert X1_tr.shape[1] == X2_tr.shape[1]
    assert X1_te.shape[1] == X2_te.shape[1]
    assert y1_tr.ndim == 1 and y2_tr.ndim == 1

    with pytest.raises(ValueError):
        _load_hillstrom_dataset(split="unknown_split")


def test_load_hillstrom_dataset_missing_columns(monkeypatch):
    df = pd.DataFrame({"history": [1.0, 2.0], "segment": ["A", "B"]})

    def _fake_read_csv(_path):
        return df.copy()

    monkeypatch.setattr(pd, "read_csv", _fake_read_csv)
    monkeypatch.setattr("os.path.exists", lambda _p: True)

    with pytest.raises(ValueError):
        _load_hillstrom_dataset(target="spend")

    df2 = pd.DataFrame({"spend": [0.1, 0.2], "segment": ["Womens", "Control"]})

    def _fake_read_csv2(_path):
        return df2.copy()

    monkeypatch.setattr(pd, "read_csv", _fake_read_csv2)
    with pytest.raises(ValueError):
        _load_hillstrom_dataset(split="existing_vs_new")

    df3 = pd.DataFrame({"spend": [0.1, 0.2], "history": [0.0, 1.0]})

    def _fake_read_csv3(_path):
        return df3.copy()

    monkeypatch.setattr(pd, "read_csv", _fake_read_csv3)
    with pytest.raises(ValueError):
        _load_hillstrom_dataset(split="womens_vs_control")


def test_load_boston_dataset_mocked(monkeypatch):
    class _Bunch:
        def __init__(self, data, target):
            self.data = data
            self.target = target

    rng = np.random.default_rng(1)
    data = rng.normal(size=(8, 4))
    target = rng.normal(size=8)
    monkeypatch.setattr("dyntr.datasets._datasets.fetch_california_housing", lambda: _Bunch(data, target))

    train_data, test_data = _load_boston_dataset()
    (X1_tr, y1_tr), (X2_tr, y2_tr) = train_data
    (X1_te, y1_te), (X2_te, y2_te) = test_data
    assert X1_tr.shape[1] == X2_tr.shape[1]
    assert y1_tr.ndim == 1 and y2_tr.ndim == 1


def test_guess_column_and_autodetect_user_id():
    df = pd.DataFrame({"User_ID": [1, 2], "value": [3, 4]})
    col = _guess_column(df, ["user_id", "id"], required=True, case_insensitive=True)
    assert col == "User_ID"

    purchases = pd.DataFrame({"user_id": [1, 2], "amount": [10, 20]})
    survey = pd.DataFrame({"user_id": [1, 2], "age": [30, 40]})
    detected = _autodetect_user_id(purchases, survey)
    assert detected.lower() == "user_id"

    df = pd.DataFrame({"x": [1, 2]})
    col = _guess_column(df, ["missing"], required=False, case_insensitive=True)
    assert col is None

    purchases = pd.DataFrame({"alpha": [1, 2], "beta": [3, 4]})
    survey = pd.DataFrame({"beta": [1, 2], "gamma": [3, 4]})
    detected = _autodetect_user_id(purchases, survey)
    assert detected.lower() == "beta"

    purchases = pd.DataFrame({"a": [1, 2]})
    survey = pd.DataFrame({"b": [3, 4]})
    with pytest.raises(ValueError):
        _autodetect_user_id(purchases, survey)
