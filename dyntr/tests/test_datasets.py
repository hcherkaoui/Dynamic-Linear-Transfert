"""Tests for dyntr.datasets with mocked CSV inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import dyntr.datasets as datasets_mod
from dyntr.datasets import fetch_synthetic_datasets
from dyntr.datasets._datasets import _autodetect_user_id, _guess_column, _load_boston_dataset, _load_hillstrom_dataset


def test_fetch_synthetic_datasets_basic():
    out = fetch_synthetic_datasets(
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
        fetch_synthetic_datasets(d=0)
    with pytest.raises(ValueError):
        fetch_synthetic_datasets(n_t_train=0)
    with pytest.raises(ValueError):
        fetch_synthetic_datasets(d=1, eps=0.5)


def test_fetch_real_datasets_mocked(monkeypatch):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    y = rng.normal(size=10)
    train_data = (X[:5], y[:5]), (X[5:], y[5:])
    test_data = (X[:3], y[:3]), (X[3:6], y[3:6])
    monkeypatch.setattr(datasets_mod, "_load_boston_dataset", lambda: (train_data, test_data))

    out = datasets_mod.fetch_real_datasets(
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
        datasets_mod.fetch_real_datasets("unknown")


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
