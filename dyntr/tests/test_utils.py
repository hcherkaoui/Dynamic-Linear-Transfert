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


def test_set_random_state_reproducible():
    utils.set_random_state(123)
    a = np.random.rand(3)
    utils.set_random_state(123)
    b = np.random.rand(3)
    assert np.allclose(a, b)


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


def test_center_and_as_generator_helpers():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Xc, mu = utils._center(X, center=True)
    assert Xc.shape == X.shape
    assert mu.shape == (2,)
    Xn, mu2 = utils._center(X, center=False)
    assert np.allclose(Xn, X)
    assert mu2.shape == (2,)

    rng = utils._as_generator(0)
    assert isinstance(rng, np.random.Generator)
    with pytest.raises(ValueError):
        utils._as_generator("bad")


def test_safe_cholesky_and_svec():
    A = np.array([[2.0, 0.5], [0.5, 1.0]])
    L = utils._safe_cholesky(A)
    assert L.shape == (2, 2)

    v = utils._svec_psd(A)
    assert v.shape == (3,)


def test_crop_pdf_missing_and_no_pdfcrop(monkeypatch, tmp_path):
    missing = tmp_path / "missing.pdf"
    ok = utils.crop_pdf(str(missing), verbose=False)
    assert ok is False

    def _raise_file_not_found(*_args, **_kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(utils.subprocess, "run", _raise_file_not_found)
    dummy = tmp_path / "dummy.pdf"
    dummy.write_bytes(b"%PDF-1.4")
    ok2 = utils.crop_pdf(str(dummy), verbose=False)
    assert ok2 is False


def test_compute_exp_vars_shapes(monkeypatch):
    rng = np.random.default_rng(0)
    X_t = rng.normal(size=(10, 4))
    X_s = rng.normal(size=(12, 4))
    theta = rng.normal(size=4)
    y_t = X_t @ theta + rng.normal(size=10)
    y_s = X_s @ theta + rng.normal(size=12)
    X_t_val = rng.normal(size=(5, 4))
    y_t_val = X_t_val @ theta + rng.normal(size=5)

    monkeypatch.setattr(utils, "opt_lambda", lambda *_args, **_kwargs: 0.1)
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
