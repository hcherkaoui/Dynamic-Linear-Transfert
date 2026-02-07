"""Tests for dyntr.datasets with mocked CSV inputs."""

from __future__ import annotations
import pandas as pd
import pytest
from dyntr.datasets import fetch_synthetic_datasets
from dyntr.datasets._datasets import _load_hillstrom_dataset, _load_open_ecommerce_dataset


def test_fetch_synthetic_datasets_basic():
    data, meta = fetch_synthetic_datasets(n_train_samples=20, n_val_samples=5, n_test_samples=5, d=4)
    (X_t, X_s, y_t, y_s), (X_t_val, _, y_t_val, _), (X_t_test, _, y_t_test, _) = data
    assert X_t.shape == (20, 4)
    assert X_s.shape == (20, 4)
    assert y_t.shape == (20,)
    assert y_s.shape == (20,)
    assert X_t_val.shape == (5, 4)
    assert y_t_val.shape == (5,)
    assert X_t_test.shape == (5, 4)
    assert y_t_test.shape == (5,)
    assert meta["theta_t"].shape == (4,)
    assert meta["theta_s"].shape == (4,)


def test_fetch_synthetic_datasets_invalid():
    with pytest.raises(ValueError):
        fetch_synthetic_datasets(d=0)
    with pytest.raises(ValueError):
        fetch_synthetic_datasets(n_train_samples=0)
    with pytest.raises(ValueError):
        fetch_synthetic_datasets(d=1, eps=0.5)


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


def test_load_open_ecommerce_dataset_mocked(monkeypatch):
    purchases = pd.DataFrame(
        {
            "Survey ResponseID": [1, 1, 2, 2, 3, 3, 4, 5],
            "Shipping Address State": ["CA", "CA", "NY", "NY", "NY", "CA", "TX", "TX"],
            "Purchase Price Per Unit": [10.0, 5.0, 7.5, 8.0, 12.0, 20.0, 3.0, 4.0],
            "Quantity": [1, 2, 1, 1, 1, 1, 2, 1],
        }
    )
    survey = pd.DataFrame(
        {
            "Survey ResponseID": [1, 2, 3, 4, 5],
            "age": [30, 40, 50, 35, 45],
            "state": ["CA", "NY", "NY", "TX", "TX"],
        }
    )

    def _fake_read_csv(path, *args, **kwargs):
        if "amazon-purchases" in path:
            return purchases.copy()
        return survey.copy()

    monkeypatch.setattr(pd, "read_csv", _fake_read_csv)
    monkeypatch.setattr("os.path.exists", lambda _p: True)

    train_data, test_data = _load_open_ecommerce_dataset()
    (X1_tr, y1_tr), (X2_tr, y2_tr) = train_data
    (X1_te, y1_te), (X2_te, y2_te) = test_data
    assert X1_tr.shape[1] == X2_tr.shape[1]
    assert y1_tr.ndim == 1 and y2_tr.ndim == 1
