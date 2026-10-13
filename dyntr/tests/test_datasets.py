"""Tests for dyntr.datasets with mocked CSV inputs."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import dyntr.datasets as datasets_mod
from dyntr.datasets import (
    list_real_regression_datasets,
    sample_synthetic_noise,
    synth_regr_data,
)
from dyntr.datasets._datasets import (
    EMBEDDING_REGRESSION_DATASETS,
    EMBEDDING_REGRESSION_METADATA_KEYS,
    _assert_is_regression,
    _autodetect_user_id,
    _condition_blocks,
    _default_data_dir,
    _generate_thetas,
    _guess_column,
    _load_boston_dataset,
    _load_chemberta_embedding_dataset,
    _load_embedding_regression_dataset,
    _load_hillstrom_dataset,
)

_FAKE_EMBEDDING_DATASET = "fake_embedding_regression"
_FAKE_EMBEDDING_SPEC = {
    "filename": "fake_embedding_regression.npz",
    "n_samples": 40,
    "d": 8,
    "target": "y",
    "feature_type": "fake embedding",
    "source": "unit test",
    "default_split": "unit test",
    "description": "Synthetic cache for tests.",
}


def _fake_embedding_arrays(d=8):
    """Provide fake test data or behavior."""
    def _X(n, offset):
        """X."""
        values = np.arange(n * d, dtype=np.float32).reshape(n, d)
        return values / 10.0 + offset

    return {
        "X_target_train": _X(12, 0.0),
        "y_target_train": np.linspace(0.1, 1.2, 12, dtype=np.float32).reshape(-1, 1),
        "X_source_train": _X(13, 2.0),
        "y_source_train": np.linspace(1.1, 2.3, 13, dtype=np.float32),
        "X_target_test": _X(7, 4.0),
        "y_target_test": np.linspace(2.1, 2.7, 7, dtype=np.float32),
        "X_source_test": _X(8, 6.0),
        "y_source_test": np.linspace(3.1, 3.8, 8, dtype=np.float32),
    }


def _register_fake_embedding_dataset(monkeypatch, tmp_path):
    """Register fake embedding dataset."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("DYNTR_DATA_DIR", str(data_dir))
    monkeypatch.setitem(
        EMBEDDING_REGRESSION_DATASETS,
        _FAKE_EMBEDDING_DATASET,
        _FAKE_EMBEDDING_SPEC.copy(),
    )
    return data_dir


def _write_fake_embedding_cache(data_dir, arrays=None):
    """Write fake embedding cache."""
    arrays = _fake_embedding_arrays() if arrays is None else arrays
    cache_dir = data_dir / _FAKE_EMBEDDING_DATASET
    cache_dir.mkdir(parents=True)
    path = cache_dir / _FAKE_EMBEDDING_SPEC["filename"]
    np.savez_compressed(path, **arrays, feature_type=np.array("fake embedding"))
    return path


def _registered_embedding_arrays(d, seed):
    """Registered embedding arrays."""
    rng = np.random.default_rng(seed)

    def _X(n, offset):
        """X."""
        return rng.normal(size=(n, d)).astype(np.float32) + offset

    return {
        "X_target_train": _X(6, 0.0),
        "y_target_train": np.linspace(0.1, 0.6, 6, dtype=np.float32),
        "X_source_train": _X(7, 2.0),
        "y_source_train": np.linspace(1.1, 1.7, 7, dtype=np.float32),
        "X_target_test": _X(4, 4.0),
        "y_target_test": np.linspace(2.1, 2.4, 4, dtype=np.float32),
        "X_source_test": _X(4, 6.0),
        "y_source_test": np.linspace(3.1, 3.4, 4, dtype=np.float32),
    }


def _write_registered_embedding_cache(data_dir, dataset_name, spec, seed):
    """Write registered embedding cache."""
    d = spec["d"] if spec["d"] is not None else 9
    cache_dir = data_dir / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / spec["filename"]
    np.savez_compressed(path, **_registered_embedding_arrays(d, seed))
    return path, d


def test_embedding_regression_registry_metadata_is_complete():
    """Test embedding regression registry metadata is complete."""
    for name, spec in EMBEDDING_REGRESSION_DATASETS.items():
        missing = [
            key
            for key in EMBEDDING_REGRESSION_METADATA_KEYS
            if key not in spec
        ]
        assert missing == [], name


def test_list_real_regression_datasets_returns_metadata_copy():
    """Test list real regression datasets returns metadata copy."""
    info = list_real_regression_datasets()

    assert "boston" in info
    assert "email" in info
    assert set(EMBEDDING_REGRESSION_DATASETS) <= set(info)
    for name, spec in EMBEDDING_REGRESSION_DATASETS.items():
        for key in EMBEDDING_REGRESSION_METADATA_KEYS:
            assert info[name][key] == spec[key]
        assert info[name]["cache_required"] is True

    info["moleculenet_lipo_ecfp"]["d"] = -1
    assert EMBEDDING_REGRESSION_DATASETS["moleculenet_lipo_ecfp"]["d"] == 2048


def test_fetch_synthetic_datasets_basic():
    """Test fetch synthetic datasets basic."""
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
    """Test fetch synthetic datasets invalid."""
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
    with pytest.raises(ValueError):
        synth_regr_data(noise="poisson")
    with pytest.raises(ValueError):
        synth_regr_data(noise="lognormal", lognormal_shape=0.0)


def test_synthetic_lognormal_noise_is_centered_scaled_and_skewed():
    """Test synthetic lognormal noise is centered scaled and skewed."""
    rng = np.random.default_rng(0)
    residuals = sample_synthetic_noise(rng, size=50_000, sigma=2.0, noise="log-normal")
    centered = residuals - np.mean(residuals)
    skewness = np.mean(centered ** 3) / np.std(residuals) ** 3

    assert abs(np.mean(residuals)) < 0.05
    assert np.isclose(np.std(residuals), 2.0, rtol=0.05)
    assert skewness > 3.0

    out = synth_regr_data(
        n_t_train=30,
        n_t_val=5,
        n_t_test=5,
        n_s_train=30,
        n_s_val=5,
        n_s_test=5,
        d=3,
        noise="lognormal",
        random_state=0,
    )
    assert out["meta"]["noise"] == "lognormal"


def test_generate_thetas_validation():
    """Test generate thetas validation."""
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
    """Test condition blocks intercept and variance."""
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
    """Test assert is regression rejects binary."""
    y = np.array([0.0, 1.0, 0.0, 1.0])
    with pytest.raises(AssertionError):
        _assert_is_regression(y, "y")


def test_bootstrap_sources_behavior(monkeypatch):
    """Test bootstrap sources behavior."""
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
    """Test fetch real datasets mocked."""
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


def test_fetch_real_chemberta_embedding_dataset_mocked(monkeypatch):
    """Test fetch real chemberta embedding dataset mocked."""
    rng = np.random.default_rng(0)
    X_t = rng.normal(size=(6, 768))
    y_t = np.linspace(0.1, 0.6, 6)
    X_s = rng.normal(size=(7, 768))
    y_s = np.linspace(1.0, 1.6, 7)
    train_data = (X_t[:5], y_t[:5]), (X_s[:6], y_s[:6])
    test_data = (X_t[5:], y_t[5:]), (X_s[6:], y_s[6:])
    monkeypatch.setattr(
        datasets_mod,
        "_load_embedding_regression_dataset",
        lambda _dataset_name: (train_data, test_data),
    )

    out = datasets_mod.real_regr_data(
        "lipophilicity_chemberta",
        n_t_train=3,
        n_t_val=1,
        n_t_test=1,
        n_s_train=3,
        n_s_val=1,
        n_s_test=1,
        n_sources=1,
        permutation=False,
        standardize=True,
        sigma_t=0.0,
        sigma_s=0.0,
    )

    assert out["meta"]["dataset"] == "lipophilicity_chemberta"
    assert out["target_train"][0].shape == (3, 768)
    assert out["target_val"][0].shape == (1, 768)
    assert out["target_test"][0].shape == (1, 768)
    assert out["source_train"][0][0].shape == (3, 768)


def test_load_embedding_regression_dataset_fake_cache(monkeypatch, tmp_path):
    """Test load embedding regression dataset fake cache."""
    data_dir = _register_fake_embedding_dataset(monkeypatch, tmp_path)
    _write_fake_embedding_cache(data_dir)

    train_data, test_data = _load_embedding_regression_dataset(_FAKE_EMBEDDING_DATASET)
    (X_t_train, y_t_train), (X_s_train, y_s_train) = train_data
    (X_t_test, y_t_test), (X_s_test, y_s_test) = test_data

    assert X_t_train.shape == (12, 8)
    assert y_t_train.shape == (12,)
    assert X_s_train.shape == (13, 8)
    assert y_s_train.shape == (13,)
    assert X_t_test.shape == (7, 8)
    assert y_t_test.shape == (7,)
    assert X_s_test.shape == (8, 8)
    assert y_s_test.shape == (8,)
    assert X_t_train.dtype == np.float64
    assert y_t_train.dtype == np.float64
    assert X_t_train.flags.c_contiguous
    assert y_t_train.flags.c_contiguous


def test_load_embedding_regression_dataset_missing_file(monkeypatch, tmp_path):
    """Test load embedding regression dataset missing file."""
    _register_fake_embedding_dataset(monkeypatch, tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match=f"{_FAKE_EMBEDDING_DATASET}.*Expected cache file",
    ):
        _load_embedding_regression_dataset(_FAKE_EMBEDDING_DATASET)


def test_load_embedding_regression_dataset_missing_required_key(monkeypatch, tmp_path):
    """Test load embedding regression dataset missing required key."""
    data_dir = _register_fake_embedding_dataset(monkeypatch, tmp_path)
    arrays = _fake_embedding_arrays()
    arrays.pop("y_source_test")
    _write_fake_embedding_cache(data_dir, arrays)

    with pytest.raises(ValueError, match="missing keys.*y_source_test"):
        _load_embedding_regression_dataset(_FAKE_EMBEDDING_DATASET)


def test_load_embedding_regression_dataset_wrong_feature_dimension(monkeypatch, tmp_path):
    """Test load embedding regression dataset wrong feature dimension."""
    data_dir = _register_fake_embedding_dataset(monkeypatch, tmp_path)
    arrays = _fake_embedding_arrays()
    arrays["X_target_train"] = arrays["X_target_train"][:, :-1]
    _write_fake_embedding_cache(data_dir, arrays)

    with pytest.raises(
        ValueError,
        match="wrong embedding dimension.*expected d=8.*got 7",
    ):
        _load_embedding_regression_dataset(_FAKE_EMBEDDING_DATASET)


def test_load_embedding_regression_dataset_non_finite(monkeypatch, tmp_path):
    """Test load embedding regression dataset non finite."""
    data_dir = _register_fake_embedding_dataset(monkeypatch, tmp_path)
    arrays = _fake_embedding_arrays()
    arrays["X_source_test"][0, 0] = np.nan
    _write_fake_embedding_cache(data_dir, arrays)

    with pytest.raises(ValueError, match="finite values"):
        _load_embedding_regression_dataset(_FAKE_EMBEDDING_DATASET)


def test_load_embedding_regression_dataset_incompatible_lengths(monkeypatch, tmp_path):
    """Test load embedding regression dataset incompatible lengths."""
    data_dir = _register_fake_embedding_dataset(monkeypatch, tmp_path)
    arrays = _fake_embedding_arrays()
    arrays["y_source_train"] = arrays["y_source_train"][:-1]
    _write_fake_embedding_cache(data_dir, arrays)

    with pytest.raises(ValueError, match="incompatible lengths"):
        _load_embedding_regression_dataset(_FAKE_EMBEDDING_DATASET)


def test_real_regr_data_embedding_regression_integration(monkeypatch, tmp_path):
    """Test real regr data embedding regression integration."""
    data_dir = _register_fake_embedding_dataset(monkeypatch, tmp_path)
    _write_fake_embedding_cache(data_dir)

    out = datasets_mod.real_regr_data(
        _FAKE_EMBEDDING_DATASET,
        n_t_train=5,
        n_t_val=3,
        n_t_test=4,
        n_s_train=6,
        n_s_val=2,
        n_s_test=3,
        n_sources=1,
        permutation=False,
        standardize=False,
        sigma_t=0.0,
        sigma_s=0.0,
    )

    assert out["meta"]["dataset"] == _FAKE_EMBEDDING_DATASET
    assert out["target_train"][0].shape == (5, 8)
    assert out["target_val"][0].shape == (3, 8)
    assert out["target_test"][0].shape == (4, 8)
    assert out["source_train"][0][0].shape == (6, 8)
    assert out["meta"]["d"] == 8


def test_real_regr_data_embedding_regression_n_sources_and_standardize(monkeypatch, tmp_path):
    """Test real regr data embedding regression n sources and standardize."""
    data_dir = _register_fake_embedding_dataset(monkeypatch, tmp_path)
    _write_fake_embedding_cache(data_dir)

    for standardize in (True, False):
        out = datasets_mod.real_regr_data(
            _FAKE_EMBEDDING_DATASET,
            n_t_train=4,
            n_t_val=2,
            n_t_test=3,
            n_s_train=5,
            n_s_val=2,
            n_s_test=3,
            n_sources=3,
            permutation=False,
            standardize=standardize,
            sigma_t=0.0,
            sigma_s=0.0,
            random_state=0,
        )

        l_X_s_train, l_y_s_train = out["source_train"]
        l_X_s_val, l_y_s_val = out["source_val"]
        l_X_s_test, l_y_s_test = out["source_test"]

        assert isinstance(l_X_s_train, list)
        assert isinstance(l_y_s_train, list)
        assert len(l_X_s_train) == 3
        assert len(l_y_s_train) == 3
        assert l_X_s_train[0].shape == (5, 8)
        assert l_y_s_train[0].shape == (5,)
        assert len(l_X_s_val) == 3
        assert l_X_s_val[0].shape == (2, 8)
        assert len(l_y_s_val) == 3
        assert len(l_X_s_test) == 3
        assert l_X_s_test[0].shape == (3, 8)
        assert len(l_y_s_test) == 3


def test_real_regr_data_accepts_all_registered_embedding_caches(monkeypatch, tmp_path):
    """Test real regr data accepts all registered embedding caches."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("DYNTR_DATA_DIR", str(data_dir))
    dimensions = {}
    for i, (dataset_name, spec) in enumerate(EMBEDDING_REGRESSION_DATASETS.items()):
        _path, dimensions[dataset_name] = _write_registered_embedding_cache(
            data_dir,
            dataset_name,
            spec,
            seed=i,
        )

    for dataset_name, d in dimensions.items():
        out = datasets_mod.real_regr_data(
            dataset_name,
            n_t_train=2,
            n_t_val=1,
            n_t_test=2,
            n_s_train=2,
            n_s_val=1,
            n_s_test=2,
            n_sources=1,
            permutation=False,
            standardize=False,
            sigma_t=0.0,
            sigma_s=0.0,
            random_state=0,
        )

        assert out["meta"]["dataset"] == dataset_name
        assert out["meta"]["d"] == d
        assert out["target_train"][0].shape == (2, d)
        assert out["target_val"][0].shape == (1, d)
        assert out["target_test"][0].shape == (2, d)
        assert out["source_train"][0][0].shape == (2, d)


def test_load_chemberta_embedding_dataset_from_cache(tmp_path):
    """Test load chemberta embedding dataset from cache."""
    data_dir = tmp_path / "data"
    cache_dir = data_dir / "aqsoldb_chemberta"
    cache_dir.mkdir(parents=True)
    rng = np.random.default_rng(1)
    np.savez_compressed(
        cache_dir / "aqsoldb_chemberta.npz",
        X_target_train=rng.normal(size=(4, 768)).astype(np.float32),
        y_target_train=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        X_source_train=rng.normal(size=(5, 768)).astype(np.float32),
        y_source_train=np.array([1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float32),
        X_target_test=rng.normal(size=(2, 768)).astype(np.float32),
        y_target_test=np.array([0.5, 0.6], dtype=np.float32),
        X_source_test=rng.normal(size=(3, 768)).astype(np.float32),
        y_source_test=np.array([1.6, 1.7, 1.8], dtype=np.float32),
    )

    train_data, test_data = _load_chemberta_embedding_dataset(
        "aqsoldb_chemberta",
        data_dir=str(data_dir),
    )
    (X_t_train, y_t_train), (X_s_train, y_s_train) = train_data
    (X_t_test, y_t_test), (X_s_test, y_s_test) = test_data

    assert X_t_train.shape == (4, 768)
    assert y_t_train.shape == (4,)
    assert X_s_train.shape == (5, 768)
    assert y_s_train.shape == (5,)
    assert X_t_test.shape == (2, 768)
    assert y_t_test.shape == (2,)
    assert X_s_test.shape == (3, 768)
    assert y_s_test.shape == (3,)

    train_data, test_data = _load_embedding_regression_dataset(
        "aqsoldb_chemberta",
        data_dir=str(data_dir),
    )
    assert train_data[0][0].dtype == np.float64
    assert train_data[0][0].flags.c_contiguous


def test_load_chemberta_embedding_dataset_missing_cache(tmp_path):
    """Test load chemberta embedding dataset missing cache."""
    with pytest.raises(FileNotFoundError):
        _load_chemberta_embedding_dataset(
            "lipophilicity_chemberta",
            data_dir=str(tmp_path),
        )


def test_embedding_data_dir_resolution(monkeypatch):
    """Test embedding data dir resolution."""
    monkeypatch.setenv("DYNTR_DATA_DIR", "/tmp/dyntr-data")
    assert _default_data_dir() == "/tmp/dyntr-data"

    monkeypatch.delenv("DYNTR_DATA_DIR")
    assert _default_data_dir().endswith("dyntr/datasets/data")


def test_load_hillstrom_dataset_mocked(monkeypatch):
    """Test load hillstrom dataset mocked."""
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
        """Provide fake test data or behavior."""
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
    """Test load hillstrom dataset missing columns."""
    df = pd.DataFrame({"history": [1.0, 2.0], "segment": ["A", "B"]})

    def _fake_read_csv(_path):
        """Provide fake test data or behavior."""
        return df.copy()

    monkeypatch.setattr(pd, "read_csv", _fake_read_csv)
    monkeypatch.setattr("os.path.exists", lambda _p: True)

    with pytest.raises(ValueError):
        _load_hillstrom_dataset(target="spend")

    df2 = pd.DataFrame({"spend": [0.1, 0.2], "segment": ["Womens", "Control"]})

    def _fake_read_csv2(_path):
        """Provide fake test data or behavior."""
        return df2.copy()

    monkeypatch.setattr(pd, "read_csv", _fake_read_csv2)
    with pytest.raises(ValueError):
        _load_hillstrom_dataset(split="existing_vs_new")

    df3 = pd.DataFrame({"spend": [0.1, 0.2], "history": [0.0, 1.0]})

    def _fake_read_csv3(_path):
        """Provide fake test data or behavior."""
        return df3.copy()

    monkeypatch.setattr(pd, "read_csv", _fake_read_csv3)
    with pytest.raises(ValueError):
        _load_hillstrom_dataset(split="womens_vs_control")


def test_load_boston_dataset_mocked(monkeypatch):
    """Test load boston dataset mocked."""
    class _Bunch:
        """_Bunch helper class."""
        def __init__(self, data, target):
            """Initialize the instance."""
            self.data = data
            self.target = target

    rng = np.random.default_rng(1)
    data = rng.normal(size=(8, 4))
    target = rng.normal(size=8)
    monkeypatch.setattr(
        "dyntr.datasets._datasets.fetch_california_housing",
        lambda: _Bunch(data, target),
    )

    train_data, test_data = _load_boston_dataset()
    (X1_tr, y1_tr), (X2_tr, y2_tr) = train_data
    (X1_te, y1_te), (X2_te, y2_te) = test_data
    assert X1_tr.shape[1] == X2_tr.shape[1]
    assert y1_tr.ndim == 1 and y2_tr.ndim == 1


def test_guess_column_and_autodetect_user_id():
    """Test guess column and autodetect user id."""
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
