"""Tests for the embedding-regression cache preparation skeleton."""
from __future__ import annotations
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from dyntr.datasets._datasets import EMBEDDING_REGRESSION_DATASETS
from scripts import prepare_embedding_regression_dataset as prep


def _prepared_arrays():
    """Prepared arrays."""
    return {
        "X_target_train": np.ones((3, 2), dtype=np.float32),
        "y_target_train": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "X_source_train": np.ones((4, 2), dtype=np.float32),
        "y_source_train": np.array([1.1, 1.2, 1.3, 1.4], dtype=np.float32),
        "X_target_test": np.ones((2, 2), dtype=np.float32),
        "y_target_test": np.array([0.4, 0.5], dtype=np.float32),
        "X_source_test": np.ones((2, 2), dtype=np.float32),
        "y_source_test": np.array([1.5, 1.6], dtype=np.float32),
    }


def test_prepare_embedding_regression_cli_help_runs():
    """Test prepare embedding regression cli help runs."""
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "prepare_embedding_regression_dataset.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--dataset" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--batch-size" in result.stdout
    assert "--device" in result.stdout
    assert "--input-path" in result.stdout
    assert "--hf-config" in result.stdout
    assert "--text-column" in result.stdout
    assert "--train-path" in result.stdout
    assert "--test-path" in result.stdout
    assert "--fingerprint-dim" in result.stdout
    assert "Optional and dataset-dependent" in result.stdout


def test_prepare_embedding_regression_unknown_dataset_raises(tmp_path):
    """Test prepare embedding regression unknown dataset raises."""
    with pytest.raises(ValueError, match="not implemented.*Available preparers"):
        prep.main(
            [
                "--dataset",
                "unknown_dataset",
                "--output-dir",
                str(tmp_path),
                "--max-samples",
                "10",
                "--test-size",
                "0.2",
                "--random-state",
                "0",
            ]
        )


def test_save_embedding_cache_writes_expected_keys(tmp_path):
    """Test save embedding cache writes expected keys."""
    path = tmp_path / "fake_embedding_regression.npz"
    prep._save_embedding_cache(
        path,
        _prepared_arrays(),
        metadata={"dataset": "fake_embedding_regression"},
    )

    with np.load(path, allow_pickle=False) as cache:
        for key in prep.REQUIRED_KEYS:
            assert key in cache
        assert cache["X_target_train"].dtype == np.float64
        assert cache["y_target_train"].shape == (3,)
        assert cache["dataset"].item() == "fake_embedding_regression"


def test_optional_import_error_mentions_dataset_requirements(monkeypatch):
    """Test optional import error mentions dataset requirements."""
    def _fake_import_module(package_name):
        """Provide fake test data or behavior."""
        raise ImportError(f"No module named {package_name}")

    monkeypatch.setattr(prep.importlib, "import_module", _fake_import_module)

    with pytest.raises(
        ImportError,
        match=(
            "Optional dependency 'missing_pkg'.*dataset preparer.*"
            "requirements-datasets.txt"
        ),
    ):
        prep._check_optional_import("missing_pkg", install_name="missing-install")


def test_moleculenet_ecfp_preparers_are_registered():
    """Test moleculenet ecfp preparers are registered."""
    expected = {
        "moleculenet_lipo_ecfp",
        "moleculenet_esol_ecfp",
        "moleculenet_freesolv_ecfp",
    }

    assert expected <= set(prep.PREPARERS)
    assert expected <= set(EMBEDDING_REGRESSION_DATASETS)
    for dataset_name in expected:
        assert EMBEDDING_REGRESSION_DATASETS[dataset_name]["d"] == 2048


def test_protein_esm2_preparers_are_registered():
    """Test protein esm2 preparers are registered."""
    expected = {
        "tape_fluorescence_esm2",
        "tape_stability_esm2",
    }

    assert expected <= set(prep.PREPARERS)
    assert expected <= set(EMBEDDING_REGRESSION_DATASETS)
    for dataset_name in expected:
        assert EMBEDDING_REGRESSION_DATASETS[dataset_name]["d"] == 1280


def test_review_sentence_preparers_are_registered():
    """Test review sentence preparers are registered."""
    expected = {
        "amazon_reviews_sentence",
        "yelp_reviews_sentence",
    }

    assert expected <= set(prep.PREPARERS)
    assert expected <= set(EMBEDDING_REGRESSION_DATASETS)
    for dataset_name in expected:
        assert EMBEDDING_REGRESSION_DATASETS[dataset_name]["d"] == 384


def test_e2006_tfidf_svd_preparer_is_registered():
    """Test e2006 tfidf svd preparer is registered."""
    assert "e2006_tfidf_svd512" in prep.PREPARERS
    assert EMBEDDING_REGRESSION_DATASETS["e2006_tfidf_svd512"]["d"] == 512
    assert EMBEDDING_REGRESSION_DATASETS["e2006_tfidf_svd512"]["feature_type"] == "TF-IDF + TruncatedSVD"


def test_tdc_drug_response_preparers_are_registered():
    """Test tdc drug response preparers are registered."""
    expected = {
        "gdsc1_expression_fingerprint",
        "gdsc2_expression_fingerprint",
    }

    assert expected <= set(prep.PREPARERS)
    assert expected <= set(EMBEDDING_REGRESSION_DATASETS)
    for dataset_name in expected:
        assert EMBEDDING_REGRESSION_DATASETS[dataset_name]["d"] is None


def test_moleculenet_ecfp_preparer_uses_deepchem_loader(monkeypatch, tmp_path):
    """Test moleculenet ecfp preparer uses deepchem loader."""
    calls = {}

    class _FakeDataset:
        """_FakeDataset helper class."""
        def __init__(self, n_samples, offset):
            """Initialize the instance."""
            self.X = np.full((n_samples, 2048), offset, dtype=np.float32)
            self.y = np.arange(n_samples, dtype=np.float32).reshape(-1, 1) + offset

    class _FakeCircularFingerprint:
        """_FakeCircularFingerprint helper class."""
        def __init__(self, size):
            """Initialize the instance."""
            calls["fingerprint_size"] = size

    def _load_lipo(featurizer=None, splitter=None):
        """Load lipo."""
        calls["loader"] = "load_lipo"
        calls["splitter"] = splitter
        calls["featurizer"] = featurizer
        datasets = (
            _FakeDataset(8, 1.0),
            _FakeDataset(3, 2.0),
            _FakeDataset(3, 3.0),
        )
        return ["logD"], datasets, []

    fake_dc = SimpleNamespace(
        feat=SimpleNamespace(CircularFingerprint=_FakeCircularFingerprint),
        molnet=SimpleNamespace(load_lipo=_load_lipo),
    )
    imported = []

    def _fake_optional_import(package_name, install_name=None):
        """Provide fake test data or behavior."""
        imported.append(package_name)
        if package_name == "deepchem":
            return fake_dc
        if package_name == "numpy":
            return np
        raise AssertionError(package_name)

    monkeypatch.setattr(prep, "_check_optional_import", _fake_optional_import)

    out_path = prep.main(
        [
            "--dataset",
            "moleculenet_lipo_ecfp",
            "--output-dir",
            str(tmp_path),
            "--max-samples",
            "0",
            "--test-size",
            "0.25",
            "--random-state",
            "0",
        ]
    )

    assert imported == ["deepchem", "numpy"]
    assert calls["loader"] == "load_lipo"
    assert calls["splitter"] == "scaffold"
    assert calls["fingerprint_size"] == 2048
    assert out_path == tmp_path / "moleculenet_lipo_ecfp" / "moleculenet_lipo_ecfp.npz"

    with np.load(out_path, allow_pickle=False) as cache:
        for key in prep.REQUIRED_KEYS:
            assert key in cache
        assert cache["X_target_train"].shape[1] == 2048
        assert cache["X_source_train"].shape[1] == 2048
        assert cache["X_target_train"].dtype == np.float64
        assert cache["dataset"].item() == "moleculenet_lipo_ecfp"
        assert cache["embedding_model"].item() == "ecfp"
        assert cache["target_domain"].item() == "target"
        assert cache["source_domains"].item() == "source"


def test_protein_esm2_preparer_uses_hf_dataset_and_model_defaults(monkeypatch, tmp_path):
    """Test protein esm2 preparer uses hf dataset and model defaults."""
    calls = {"embeddings": []}

    class _FakeSplit:
        """_FakeSplit helper class."""
        def __init__(self, sequences, targets):
            """Initialize the instance."""
            self.column_names = ["primary", "fluorescence"]
            self._columns = {
                "primary": sequences,
                "fluorescence": targets,
            }

        def __getitem__(self, key):
            """Getitem."""
            return self._columns[key]

    class _FakeDatasets:
        """_FakeDatasets helper class."""
        @staticmethod
        def load_dataset(name):
            """Load dataset."""
            calls["dataset_source"] = name
            return {
                "train": _FakeSplit(["AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF"], range(6)),
                "validation": _FakeSplit(["GGGG", "HHHH", "IIII"], [6, 7, 8]),
                "test": _FakeSplit(["JJJJ", "KKKK", "LLLL"], [9, 10, 11]),
            }

    class _FakeCuda:
        """_FakeCuda helper class."""
        @staticmethod
        def is_available():
            """Is available."""
            return False

    class _FakeTorch:
        """_FakeTorch helper class."""
        cuda = _FakeCuda()

    class _FakeTokenizer:
        """_FakeTokenizer helper class."""
        pass

    class _FakeModel:
        """_FakeModel helper class."""
        def to(self, device):
            """To."""
            calls["device"] = device
            return self

        def eval(self):
            """Eval."""
            calls["eval"] = True
            return self

    class _FakeAutoTokenizer:
        """_FakeAutoTokenizer helper class."""
        @staticmethod
        def from_pretrained(name):
            """From pretrained."""
            calls["tokenizer_model"] = name
            return _FakeTokenizer()

    class _FakeAutoModel:
        """_FakeAutoModel helper class."""
        @staticmethod
        def from_pretrained(name):
            """From pretrained."""
            calls["model_name"] = name
            return _FakeModel()

    fake_transformers = SimpleNamespace(
        AutoTokenizer=_FakeAutoTokenizer,
        AutoModel=_FakeAutoModel,
    )

    def _fake_optional_import(package_name, install_name=None):
        """Provide fake test data or behavior."""
        if package_name == "datasets":
            return _FakeDatasets
        if package_name == "torch":
            return _FakeTorch
        if package_name == "transformers":
            return fake_transformers
        raise AssertionError(package_name)

    def _fake_embed(sequences, tokenizer, model, torch_module, device, batch_size):
        """Provide fake test data or behavior."""
        calls["embeddings"].append((tuple(sequences), device, batch_size))
        return np.ones((len(sequences), 1280), dtype=np.float64)

    monkeypatch.setattr(prep, "_check_optional_import", _fake_optional_import)
    monkeypatch.setattr(prep, "_embed_protein_sequences", _fake_embed)

    out_path = prep.main(
        [
            "--dataset",
            "tape_fluorescence_esm2",
            "--output-dir",
            str(tmp_path),
            "--max-samples",
            "0",
            "--test-size",
            "0.25",
            "--random-state",
            "0",
        ]
    )

    assert calls["dataset_source"] == "SaProtHub/Dataset-Fluorescence-TAPE"
    assert calls["tokenizer_model"] == prep.DEFAULT_ESM2_MODEL
    assert calls["model_name"] == prep.DEFAULT_ESM2_MODEL
    assert calls["device"] == "cpu"
    assert calls["eval"] is True
    assert calls["embeddings"][0][1:] == ("cpu", 8)
    assert calls["embeddings"][1][1:] == ("cpu", 8)
    assert out_path == tmp_path / "tape_fluorescence_esm2" / "tape_fluorescence_esm2.npz"

    with np.load(out_path, allow_pickle=False) as cache:
        for key in prep.REQUIRED_KEYS:
            assert key in cache
        assert cache["X_target_train"].shape[1] == 1280
        assert cache["X_source_train"].shape[1] == 1280
        assert cache["embedding_model"].item() == prep.DEFAULT_ESM2_MODEL
        assert cache["dataset_source"].item() == "SaProtHub/Dataset-Fluorescence-TAPE"
        assert cache["target_name"].item() == "fluorescence"
        assert cache["split_name"].item() == "official:train->source,validation+test->target"


def test_amazon_reviews_sentence_requires_hf_config(tmp_path):
    """Test amazon reviews sentence requires hf config."""
    with pytest.raises(ValueError, match="--hf-config is required"):
        prep.main(
            [
                "--dataset",
                "amazon_reviews_sentence",
                "--output-dir",
                str(tmp_path),
                "--max-samples",
                "8",
                "--test-size",
                "0.25",
                "--random-state",
                "0",
            ]
        )


def test_amazon_reviews_sentence_preparer_uses_config_and_sentence_model(monkeypatch, tmp_path):
    """Test amazon reviews sentence preparer uses config and sentence model."""
    import pandas as pd

    calls = {}

    class _FakeDatasets:
        """_FakeDatasets helper class."""
        @staticmethod
        def load_dataset(name, config):
            """Load dataset."""
            calls["dataset_source"] = name
            calls["hf_config"] = config
            return {
                "train": {
                    "text": [f"review {i}" for i in range(10)],
                    "rating": [float((i % 5) + 1) for i in range(10)],
                    "main_category": ["books"] * 5 + ["music"] * 5,
                }
            }

    class _FakeSentenceTransformer:
        """_FakeSentenceTransformer helper class."""
        def __init__(self, model_name, device=None):
            """Initialize the instance."""
            calls["model_name"] = model_name
            calls["device"] = device

        def encode(self, texts, batch_size=None, convert_to_numpy=True, show_progress_bar=True):
            """Encode."""
            calls.setdefault("encoded", []).append((tuple(texts), batch_size))
            return np.ones((len(texts), 384), dtype=np.float32)

    fake_sentence_transformers = SimpleNamespace(
        SentenceTransformer=_FakeSentenceTransformer
    )

    def _fake_optional_import(package_name, install_name=None):
        """Provide fake test data or behavior."""
        if package_name == "datasets":
            return _FakeDatasets
        if package_name == "pandas":
            return pd
        if package_name == "sentence_transformers":
            return fake_sentence_transformers
        raise AssertionError(package_name)

    monkeypatch.setattr(prep, "_check_optional_import", _fake_optional_import)
    monkeypatch.setattr(prep, "_sentence_device", lambda _device: "cpu")

    out_path = prep.main(
        [
            "--dataset",
            "amazon_reviews_sentence",
            "--output-dir",
            str(tmp_path),
            "--hf-config",
            "raw_review_Books",
            "--max-samples",
            "8",
            "--test-size",
            "0.25",
            "--random-state",
            "0",
        ]
    )

    assert calls["dataset_source"] == "McAuley-Lab/Amazon-Reviews-2023"
    assert calls["hf_config"] == "raw_review_Books"
    assert calls["model_name"] == prep.DEFAULT_SENTENCE_MODEL
    assert calls["device"] == "cpu"
    assert out_path == tmp_path / "amazon_reviews_sentence" / "amazon_reviews_sentence.npz"

    with np.load(out_path, allow_pickle=False) as cache:
        for key in prep.REQUIRED_KEYS:
            assert key in cache
        assert cache["X_target_train"].shape[1] == 384
        assert cache["X_source_train"].shape[1] == 384
        assert cache["embedding_model"].item() == prep.DEFAULT_SENTENCE_MODEL
        assert cache["dataset_source"].item() == "McAuley-Lab/Amazon-Reviews-2023"
        assert cache["data_source"].item() == "raw_review_Books"
        assert cache["target_column"].item() == "rating"
        assert cache["domain_column"].item() == "main_category"


def test_yelp_reviews_sentence_missing_input_path_raises(tmp_path):
    """Test yelp reviews sentence missing input path raises."""
    with pytest.raises(FileNotFoundError, match="Yelp input path not found"):
        prep.main(
            [
                "--dataset",
                "yelp_reviews_sentence",
                "--output-dir",
                str(tmp_path),
                "--input-path",
                str(tmp_path / "missing"),
                "--max-samples",
                "8",
                "--test-size",
                "0.25",
                "--random-state",
                "0",
            ]
        )


def test_e2006_tfidf_svd_requires_train_and_test_paths(tmp_path):
    """Test e2006 tfidf svd requires train and test paths."""
    with pytest.raises(ValueError, match="--train-path is required"):
        prep.main(
            [
                "--dataset",
                "e2006_tfidf_svd512",
                "--output-dir",
                str(tmp_path),
                "--max-samples",
                "0",
                "--test-size",
                "0.25",
                "--random-state",
                "0",
            ]
        )


def test_e2006_tfidf_svd_preparer_uses_sparse_svd(monkeypatch, tmp_path):
    """Test e2006 tfidf svd preparer uses sparse svd."""
    calls = {}
    train_path = tmp_path / "E2006.train.bz2"
    test_path = tmp_path / "E2006.test.bz2"
    train_path.write_text("unused")
    test_path.write_text("unused")

    class _FakeSparse:
        """_FakeSparse helper class."""
        def __init__(self, n_rows, label):
            """Initialize the instance."""
            self.shape = (n_rows, 150360)
            self.label = label

        def __getitem__(self, idx):
            """Getitem."""
            if isinstance(idx, tuple):
                idx = idx[0]
            if isinstance(idx, slice):
                n_rows = len(range(*idx.indices(self.shape[0])))
            else:
                n_rows = len(np.atleast_1d(idx))
            return _FakeSparse(n_rows, self.label)

    def _fake_load_svmlight_file(path, n_features=None):
        """Provide fake test data or behavior."""
        calls.setdefault("loaded", []).append((Path(path).name, n_features))
        if Path(path).name == "E2006.train.bz2":
            return _FakeSparse(8, "source"), np.arange(8, dtype=np.float64)
        return _FakeSparse(6, "target"), np.arange(6, dtype=np.float64) + 10.0

    class _FakeSVD:
        """_FakeSVD helper class."""
        def __init__(self, n_components, random_state=None):
            """Initialize the instance."""
            calls["n_components"] = n_components
            calls["random_state"] = random_state

        def fit(self, X):
            """Fit."""
            calls["fit_shape"] = X.shape
            return self

        def transform(self, X):
            """Transform."""
            calls.setdefault("transformed_shapes", []).append(X.shape)
            return np.ones((X.shape[0], 512), dtype=np.float32)

    def _fake_vstack(blocks, format=None):
        """Provide fake test data or behavior."""
        calls["vstack_format"] = format
        calls["vstack_shapes"] = [block.shape for block in blocks]
        return _FakeSparse(sum(block.shape[0] for block in blocks), "pool")

    def _fake_optional_import(package_name, install_name=None):
        """Provide fake test data or behavior."""
        if package_name == "sklearn.datasets":
            return SimpleNamespace(load_svmlight_file=_fake_load_svmlight_file)
        if package_name == "sklearn.decomposition":
            return SimpleNamespace(TruncatedSVD=_FakeSVD)
        if package_name == "scipy.sparse":
            return SimpleNamespace(vstack=_fake_vstack)
        raise AssertionError(package_name)

    monkeypatch.setattr(prep, "_check_optional_import", _fake_optional_import)

    with pytest.warns(RuntimeWarning, match="150360.*512"):
        out_path = prep.main(
            [
                "--dataset",
                "e2006_tfidf_svd512",
                "--output-dir",
                str(tmp_path),
                "--train-path",
                str(train_path),
                "--test-path",
                str(test_path),
                "--max-samples",
                "0",
                "--test-size",
                "0.25",
                "--random-state",
                "7",
            ]
        )

    assert calls["loaded"] == [
        ("E2006.train.bz2", 150360),
        ("E2006.test.bz2", 150360),
    ]
    assert calls["n_components"] == 512
    assert calls["random_state"] == 7
    assert calls["vstack_format"] == "csr"
    assert calls["fit_shape"][1] == 150360
    assert out_path == tmp_path / "e2006_tfidf_svd512" / "e2006_tfidf_svd512.npz"

    with np.load(out_path, allow_pickle=False) as cache:
        for key in prep.REQUIRED_KEYS:
            assert key in cache
        assert cache["X_target_train"].shape[1] == 512
        assert cache["X_source_train"].shape[1] == 512
        assert cache["X_target_train"].dtype == np.float64
        assert cache["raw_tfidf_dim"].item() == 150360
        assert cache["svd_n_components"].item() == 512


def test_tdc_drug_response_preparer_uses_fingerprint_and_fallback(monkeypatch, tmp_path):
    """Test tdc drug response preparer uses fingerprint and fallback."""
    import pandas as pd

    calls = {}

    class _FakeDrugRes:
        """_FakeDrugRes helper class."""
        def __init__(self, name):
            """Initialize the instance."""
            calls["tdc_name"] = name

        def get_data(self):
            """Get data."""
            return pd.DataFrame(
                {
                    "Drug": ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CCI", "CCF", "COC"],
                    "Drug_ID": [f"drug_{i}" for i in range(8)],
                    "Cell Line": ["A", "B", "C", "D", "A", "B", "C", "D"],
                    "cancer_lineage": [
                        "lung",
                        "lung",
                        "breast",
                        "breast",
                        "lung",
                        "breast",
                        "lung",
                        "breast",
                    ],
                    "Y": np.arange(8, dtype=np.float64),
                }
            )

    class _FakeChem:
        """_FakeChem helper class."""
        @staticmethod
        def MolFromSmiles(smiles):
            """Molfromsmiles."""
            return smiles if smiles else None

    class _FakeFingerprint:
        """_FakeFingerprint helper class."""
        def __init__(self, n_bits):
            """Initialize the instance."""
            self.n_bits = n_bits

        def ToBitString(self):
            """Tobitstring."""
            return ("10" * ((self.n_bits + 1) // 2))[: self.n_bits]

    class _FakeGenerator:
        """_FakeGenerator helper class."""
        def __init__(self, n_bits):
            """Initialize the instance."""
            self.n_bits = n_bits

        def GetFingerprint(self, mol):
            """Getfingerprint."""
            calls.setdefault("fingerprinted", []).append(mol)
            return _FakeFingerprint(self.n_bits)

    class _FakeRdFingerprintGenerator:
        """_FakeRdFingerprintGenerator helper class."""
        @staticmethod
        def GetMorganGenerator(radius, fpSize):
            """Getmorgangenerator."""
            calls["fingerprint_radius"] = radius
            calls["fingerprint_dim"] = fpSize
            return _FakeGenerator(fpSize)

    fake_tdc = SimpleNamespace(
        multi_pred=SimpleNamespace(DrugRes=_FakeDrugRes),
    )

    def _fake_optional_import(package_name, install_name=None):
        """Provide fake test data or behavior."""
        if package_name == "tdc":
            return fake_tdc
        if package_name == "rdkit":
            return SimpleNamespace()
        if package_name == "rdkit.Chem":
            return _FakeChem
        if package_name == "rdkit.Chem.rdFingerprintGenerator":
            return _FakeRdFingerprintGenerator
        if package_name == "pandas":
            return pd
        if package_name == "numpy":
            return np
        raise AssertionError(package_name)

    monkeypatch.setattr(prep, "_check_optional_import", _fake_optional_import)

    with pytest.warns(RuntimeWarning, match="falling back.*one-hot"):
        out_path = prep.main(
            [
                "--dataset",
                "gdsc1_expression_fingerprint",
                "--output-dir",
                str(tmp_path),
                "--max-samples",
                "0",
                "--test-size",
                "0.25",
                "--random-state",
                "0",
                "--fingerprint-dim",
                "8",
            ]
        )

    assert calls["tdc_name"] == "GDSC1"
    assert calls["fingerprint_radius"] == 2
    assert calls["fingerprint_dim"] == 8
    assert len(calls["fingerprinted"]) == 8
    assert out_path == tmp_path / "gdsc1_expression_fingerprint" / "gdsc1_expression_fingerprint.npz"

    with np.load(out_path, allow_pickle=False) as cache:
        for key in prep.REQUIRED_KEYS:
            assert key in cache
        assert cache["X_target_train"].shape[1] == 12
        assert cache["X_source_train"].shape[1] == 12
        assert cache["X_target_train"].dtype == np.float64
        assert cache["fingerprint_dim"].item() == 8
        assert cache["actual_d"].item() == 12
        assert cache["cell_feature_type"].item() == "cell_line_one_hot"
        assert cache["expression_features_available"].item() is False
        assert cache["domain_column"].item() == "cancer_lineage"


def test_tdc_drug_response_missing_required_column_raises(monkeypatch, tmp_path):
    """Test tdc drug response missing required column raises."""
    import pandas as pd

    class _FakeDrugRes:
        """_FakeDrugRes helper class."""
        def __init__(self, name):
            """Initialize the instance."""
            self.name = name

        def get_data(self):
            """Get data."""
            return pd.DataFrame(
                {
                    "Drug": ["CCO", "CCN", "CCC", "CCCl"],
                    "Y": [0.0, 1.0, 2.0, 3.0],
                }
            )

    fake_tdc = SimpleNamespace(
        multi_pred=SimpleNamespace(DrugRes=_FakeDrugRes),
    )

    def _fake_optional_import(package_name, install_name=None):
        """Provide fake test data or behavior."""
        if package_name == "tdc":
            return fake_tdc
        if package_name == "rdkit":
            return SimpleNamespace()
        if package_name == "pandas":
            return pd
        if package_name == "numpy":
            return np
        raise AssertionError(package_name)

    monkeypatch.setattr(prep, "_check_optional_import", _fake_optional_import)

    with pytest.raises(ValueError, match="Could not detect cell-line column"):
        prep.main(
            [
                "--dataset",
                "gdsc2_expression_fingerprint",
                "--output-dir",
                str(tmp_path),
                "--max-samples",
                "0",
                "--test-size",
                "0.25",
                "--random-state",
                "0",
            ]
        )
