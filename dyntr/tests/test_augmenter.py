"""Tests for dyntr.augmenter."""
from __future__ import annotations
import numpy as np
import pytest
from dyntr.augmenter import (
    OracleSampleAugmenter,
    SampleAugmenter,
    UOWQSampleAugmenter,
    _BaseSampleAugmenter,
    _estimate_sigma,
    _oracle_score_update_batch,
    _permute,
    _prepare_sources,
    _woodbury_update,
    prepare_sources,
    project_simplex,
    select_multi_source,
    solve_simplex_qp,
)


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


def test_greedy_sample_augmenter_bad_params():
    rng = np.random.default_rng(0)
    X_t, y_t, _, _, X_t_val, _ = _make_data(rng)
    with pytest.raises(ValueError):
        SampleAugmenter(n_max=10, chunk_size=0).fit(X_t, y_t, X_t_val, lbda=1.0, sigma=0.1)
    with pytest.raises(ValueError):
        SampleAugmenter(n_max=10, strategy="bad").fit(X_t, y_t, X_t_val, lbda=1.0, sigma=0.1)


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


def test_uowq_sample_augmenter_fit_transform():
    rng = np.random.default_rng(3)
    X_t, y_t, X_s, y_s, X_t_val, _ = _make_data(rng)
    aug = UOWQSampleAugmenter(n_max=8, n_iter_qp=50, random_state=0)
    aug.fit(X_t, y_t, X_t_val, lbda=1.0, sigma=0.1)
    Xc, yc = aug.transform([X_s], [y_s], lbda=0.5, sigma=0.2)
    assert Xc.shape[0] == yc.shape[0]
    assert Xc.shape[1] == X_t.shape[1]
    assert aug.get_n_selected() >= 0


def test_prepare_sources_validation():
    rng = np.random.RandomState(0)
    X = np.ones((3, 2))
    y = np.ones(3)

    with pytest.raises(ValueError):
        _prepare_sources(rng, [X], [y, y], n_features=2)
    with pytest.raises(ValueError):
        prepare_sources([X], [y, y], n_features=2, rng=rng)

    with pytest.raises(ValueError):
        _prepare_sources(rng, [X], [y[:-1]], n_features=2)
    with pytest.raises(ValueError):
        prepare_sources([X], [y[:-1]], n_features=2, rng=rng)


def test_project_simplex_and_qp():
    v = np.array([-1.0, 2.0, 0.5])
    w = project_simplex(v)
    assert np.all(w >= 0.0)
    assert np.isclose(w.sum(), 1.0)

    M = np.eye(3)
    a = solve_simplex_qp(M, n_iter=10)
    assert np.all(a >= 0.0)
    assert np.isclose(a.sum(), 1.0)

    single = solve_simplex_qp(np.array([[2.0]]))
    assert np.allclose(single, np.array([1.0]))


def test_estimate_sigma_edge_case():
    X = np.ones((1, 2))
    y = np.array([1.0])
    assert _estimate_sigma(X, y, lbda=1.0) == 0.0


def test_woodbury_and_oracle_batch():
    A_inv = np.eye(2)
    empty = _woodbury_update(A_inv, np.empty((0, 2)))
    assert np.allclose(empty, A_inv)

    X_b = np.array([[1.0, 0.0]])
    y_b = np.array([1.0])
    X_t_val = np.array([[1.0, 0.0]])
    theta_t_true = np.array([0.5, 0.0])
    base_err = 1.0
    delta, Kc_new, b_new = _oracle_score_update_batch(
        A_inv,
        np.zeros(2),
        X_b,
        y_b,
        X_t_val,
        theta_t_true,
        base_err,
    )
    assert isinstance(delta, float)
    assert Kc_new.shape == (2, 2)
    assert b_new.shape == (2,)


def test_select_multi_source_edges():
    rng = np.random.RandomState(0)
    Xs = [np.zeros((2, 2)), np.ones((2, 2))]
    ys = [np.zeros(2), np.ones(2)]

    out = select_multi_source(rng, "greedy", Xs, ys, 0, lambda s, X, y: (0.0, s), None, chunk_size=1)
    deltas, sel_src, X_seq, y_seq, chunk_sizes = out
    assert deltas.size == 0
    assert sel_src.size == 0
    assert X_seq.size == 0
    assert y_seq.size == 0
    assert chunk_sizes.size == 0

    with pytest.raises(ValueError):
        select_multi_source(rng, "bad", Xs, ys, 1, lambda s, X, y: (0.0, s), None, chunk_size=1)

    out = select_multi_source(rng, "uniform", Xs, ys, 1, lambda s, X, y: (0.1, s), None, chunk_size=1)
    deltas, sel_src, X_seq, y_seq, chunk_sizes = out
    assert deltas.size == 1
    assert sel_src.size == 1
    assert X_seq.shape[0] == 1
    assert y_seq.shape[0] == 1
    assert chunk_sizes.size == 1

    out = select_multi_source(
        rng,
        "greedy",
        [np.empty((0, 2))],
        [np.empty((0,))],
        1,
        lambda s, X, y: (0.0, s),
        None,
        chunk_size=1,
    )
    assert out[0].size == 0


def test_permutes_deterministically():
    rng = np.random.RandomState(0)
    X = np.arange(6.0).reshape(3, 2)
    y = np.array([1.0, 2.0, 3.0])
    Xp, yp = _permute(rng, X, y)
    assert Xp.shape == X.shape
    assert yp.shape == y.shape


def test_finalize_selection_empty_and_negative():
    class _Dummy(_BaseSampleAugmenter):
        def fit(self):
            return self

    rng = np.random.RandomState(0)
    dummy = _Dummy()
    dummy._rng_ = rng
    dummy.X_t_ = np.zeros((2, 2))
    dummy.y_t_ = np.zeros(2)
    X_out, y_out = dummy._finalize_selection(
        np.array([], dtype=float),
        np.array([], dtype=int),
        np.empty((0, 2)),
        np.empty((0,)),
        np.array([], dtype=int),
    )
    assert X_out.shape == dummy.X_t_.shape
    assert y_out.shape == dummy.y_t_.shape

    X_out, y_out = dummy._finalize_selection(
        np.array([-1.0], dtype=float),
        np.array([0], dtype=int),
        np.zeros((1, 2)),
        np.zeros(1),
        np.array([1], dtype=int),
    )
    assert X_out.shape == dummy.X_t_.shape
    assert y_out.shape == dummy.y_t_.shape
