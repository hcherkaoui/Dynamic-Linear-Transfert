"""Prepare local npz caches for embedding-regression datasets."""
from __future__ import annotations
import argparse
import importlib
from pathlib import Path
import tempfile
import urllib.parse
import urllib.request
import warnings
import numpy as np


REQUIRED_KEYS = (
    "X_target_train",
    "y_target_train",
    "X_source_train",
    "y_source_train",
    "X_target_test",
    "y_target_test",
    "X_source_test",
    "y_source_test",
)

MOLECULENET_ECFP_DATASETS = {
    "moleculenet_lipo_ecfp": {
        "loader": "load_lipo",
        "target": "logD",
        "source": "MoleculeNet Lipophilicity",
    },
    "moleculenet_esol_ecfp": {
        "loader": "load_delaney",
        "target": "solubility",
        "source": "MoleculeNet ESOL",
    },
    "moleculenet_freesolv_ecfp": {
        "loader": "load_sampl",
        "target": "hydration free energy",
        "source": "MoleculeNet FreeSolv",
    },
}

DEFAULT_ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"
DEFAULT_SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROTEIN_ESM2_DATASETS = {
    "tape_fluorescence_esm2": {
        "dataset_source": "SaProtHub/Dataset-Fluorescence-TAPE",
        "target_name": "fluorescence",
    },
    "tape_stability_esm2": {
        "dataset_source": "SaProtHub/Dataset-Stability-TAPE",
        "target_name": "stability",
    },
}

REVIEW_SENTENCE_DATASETS = {
    "amazon_reviews_sentence": {
        "dataset_source": "McAuley-Lab/Amazon-Reviews-2023",
        "target_name": "star rating",
    },
    "yelp_reviews_sentence": {
        "dataset_source": "local Yelp JSON files",
        "target_name": "star rating",
    },
}

E2006_TFIDF_SVD_DATASETS = {
    "e2006_tfidf_svd512": {
        "target": "E2006 regression target",
        "feature_type": "TF-IDF + TruncatedSVD",
        "raw_dim": 150360,
        "d": 512,
    },
}

TDC_DRUG_RESPONSE_DATASETS = {
    "gdsc1_expression_fingerprint": {
        "tdc_name": "GDSC1",
        "target": "IC50 or drug response",
        "source": "TDC DrugRes GDSC1",
    },
    "gdsc2_expression_fingerprint": {
        "tdc_name": "GDSC2",
        "target": "IC50 or drug response",
        "source": "TDC DrugRes GDSC2",
    },
}


def _check_optional_import(package_name, install_name=None):
    """Import an optional dependency with a clear error."""
    try:
        return importlib.import_module(package_name)
    except ImportError as exc:
        install_name = install_name or package_name
        raise ImportError(
            f"Optional dependency '{package_name}' is required for this "
            "dataset preparer. "
            f"Install it with `pip install {install_name}` or "
            "`pip install -r requirements-datasets.txt`. The base dyntr "
            "install intentionally does not include dataset-preparation "
            "dependencies."
        ) from exc


def _split_indices(n_samples, test_size, rng):
    """Split indices into train and test subsets."""
    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size as a float must be in (0, 1).")
        n_test = int(np.ceil(n_samples * test_size))
    else:
        n_test = int(test_size)
        if not 0 < n_test < n_samples:
            raise ValueError("test_size as an int must be in [1, n_samples - 1].")

    n_test = min(max(n_test, 1), n_samples - 1)
    idx = rng.permutation(n_samples)
    return idx[n_test:], idx[:n_test]


def _split_arrays(X, y, test_size, rng):
    """Split arrays into train and test subsets."""
    train_idx, test_idx = _split_indices(X.shape[0], test_size, rng)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def _train_test_split_domain(
    X,
    y,
    domain,
    target_domain,
    source_domains,
    test_size,
    random_state,
):
    """Split arrays by source and target domains."""
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    domain = np.asarray(domain)
    source_domains = list(source_domains)

    if X.ndim != 2:
        raise ValueError(f"X must be a 2D matrix, got shape {X.shape}.")
    if X.shape[0] != y.shape[0] or X.shape[0] != domain.shape[0]:
        raise ValueError("X, y, and domain must have the same number of rows.")
    if domain.ndim != 1:
        raise ValueError("domain must be a 1D array.")
    if not source_domains:
        raise ValueError("source_domains must contain at least one domain.")

    target_mask = domain == target_domain
    source_mask = np.isin(domain, source_domains)
    if target_mask.sum() < 2:
        raise ValueError("Target domain must contain at least two samples.")
    if source_mask.sum() < 2:
        raise ValueError("Source domains must contain at least two samples.")

    rng = np.random.default_rng(random_state)
    X_target_train, y_target_train, X_target_test, y_target_test = _split_arrays(
        X[target_mask],
        y[target_mask],
        test_size,
        rng,
    )
    X_source_train, y_source_train, X_source_test, y_source_test = _split_arrays(
        X[source_mask],
        y[source_mask],
        test_size,
        rng,
    )

    return {
        "X_target_train": X_target_train,
        "y_target_train": y_target_train,
        "X_source_train": X_source_train,
        "y_source_train": y_source_train,
        "X_target_test": X_target_test,
        "y_target_test": y_target_test,
        "X_source_test": X_source_test,
        "y_source_test": y_source_test,
    }


def _validate_prepared_arrays(arrays, expected_dim=None):
    """Validate prepared cache arrays."""
    missing = [key for key in REQUIRED_KEYS if key not in arrays]
    if missing:
        raise ValueError(f"Prepared arrays are missing keys: {missing}")

    out = {}
    for prefix in ("target_train", "source_train", "target_test", "source_test"):
        X_key = f"X_{prefix}"
        y_key = f"y_{prefix}"
        X = np.asarray(arrays[X_key], dtype=np.float64)
        y = np.asarray(arrays[y_key], dtype=np.float64).reshape(-1)
        if X.ndim != 2:
            raise ValueError(f"{X_key} must be a 2D matrix, got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"{X_key} and {y_key} have incompatible lengths: "
                f"{X.shape[0]} and {y.shape[0]}."
            )
        if expected_dim is not None and X.shape[1] != expected_dim:
            raise ValueError(
                f"{X_key} must have embedding dimension {expected_dim}, "
                f"got {X.shape[1]}."
            )
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError(f"{X_key} and {y_key} must contain finite values.")
        out[X_key] = np.ascontiguousarray(X, dtype=np.float64)
        out[y_key] = np.ascontiguousarray(y, dtype=np.float64)

    return out


def _save_embedding_cache(path, arrays, metadata=None, overwrite=False):
    """Save a validated embedding cache."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing cache: {path}")

    payload = _validate_prepared_arrays(arrays)
    if metadata:
        for key, value in metadata.items():
            if key in payload:
                raise ValueError(f"Metadata key conflicts with array key: {key}")
            payload[key] = np.asarray(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def _dense_float64(X, np_module=np):
    """Convert an array-like object to dense float64."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np_module.ascontiguousarray(np_module.asarray(X, dtype=np_module.float64))


def _dataset_xy(dataset, np_module=np):
    """Extract dense features and targets from a dataset."""
    X = _dense_float64(dataset.X, np_module=np_module)
    y = np_module.ascontiguousarray(
        np_module.asarray(dataset.y, dtype=np_module.float64).reshape(-1)
    )
    return X, y


def _limit_samples(X, y, max_samples, rng):
    """Limit arrays to a maximum number of samples."""
    if max_samples is None or max_samples <= 0 or X.shape[0] <= max_samples:
        return X, y
    idx = rng.permutation(X.shape[0])[:max_samples]
    return X[idx], y[idx]


def _limit_records(sequences, y, max_samples, rng):
    """Limit sequence records to a maximum sample count."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if max_samples is None or max_samples <= 0 or len(sequences) <= max_samples:
        return list(sequences), y
    idx = rng.permutation(len(sequences))[:max_samples]
    return [sequences[i] for i in idx], y[idx]


def _limit_frame(df, max_samples, rng):
    """Limit a dataframe to a maximum sample count."""
    if max_samples is None or max_samples <= 0 or len(df) <= max_samples:
        return df
    idx = rng.permutation(len(df))[:max_samples]
    return df.iloc[idx].reset_index(drop=True)


def _make_ecfp_featurizer(dc):
    """Create a DeepChem ECFP featurizer."""
    try:
        return dc.feat.CircularFingerprint(size=2048)
    except AttributeError:
        return "ECFP"


def _load_moleculenet_splits(dc, loader_name, featurizer):
    """Load DeepChem MoleculeNet splits."""
    loader = getattr(dc.molnet, loader_name)
    try:
        return loader(featurizer=featurizer, splitter="scaffold")
    except TypeError:
        return loader(featurizer="ECFP", splitter="scaffold")


def _prepare_moleculenet_ecfp(args, spec):
    """Prepare a MoleculeNet ECFP cache."""
    dc = _check_optional_import("deepchem")
    np_module = _check_optional_import("numpy")
    embedding_model = args.embedding_model or "ecfp"
    target_domain = args.target_domain or "target"
    source_domains = args.source_domains or ["source"]
    featurizer = _make_ecfp_featurizer(dc)
    tasks, datasets, _transformers = _load_moleculenet_splits(
        dc,
        spec["loader"],
        featurizer,
    )
    train_dataset, valid_dataset, test_dataset = datasets

    rng = np_module.random.default_rng(args.random_state)
    X_source, y_source = _dataset_xy(train_dataset, np_module=np_module)
    X_valid, y_valid = _dataset_xy(valid_dataset, np_module=np_module)
    X_test, y_test = _dataset_xy(test_dataset, np_module=np_module)
    X_target = np_module.vstack([X_valid, X_test])
    y_target = np_module.concatenate([y_valid, y_test])

    X_source, y_source = _limit_samples(X_source, y_source, args.max_samples, rng)
    X_target, y_target = _limit_samples(X_target, y_target, args.max_samples, rng)

    X_target_train, y_target_train, X_target_test, y_target_test = _split_arrays(
        X_target,
        y_target,
        args.test_size,
        rng,
    )
    X_source_train, y_source_train, X_source_test, y_source_test = _split_arrays(
        X_source,
        y_source,
        args.test_size,
        rng,
    )
    arrays = {
        "X_target_train": X_target_train,
        "y_target_train": y_target_train,
        "X_source_train": X_source_train,
        "y_source_train": y_source_train,
        "X_target_test": X_target_test,
        "y_target_test": y_target_test,
        "X_source_test": X_source_test,
        "y_source_test": y_source_test,
    }
    arrays = _validate_prepared_arrays(arrays, expected_dim=2048)
    metadata = {
        "dataset": args.dataset,
        "target": spec["target"],
        "feature_type": "ECFP fingerprint",
        "source": spec["source"],
        "embedding_model": embedding_model,
        "target_domain": target_domain,
        "source_domains": ",".join(source_domains),
        "tasks": ",".join(str(task) for task in tasks),
    }
    return arrays, metadata


def _make_moleculenet_ecfp_preparer(spec):
    """Build a MoleculeNet ECFP preparer."""
    def _preparer(args):
        """Run the configured preparer."""
        return _prepare_moleculenet_ecfp(args, spec)

    return _preparer


def _detect_column(columns, candidates, kind):
    """Detect a column from candidate names."""
    lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        column = lower.get(candidate.lower())
        if column is not None:
            return column
    raise ValueError(
        f"Could not detect {kind} column. Tried {candidates}. "
        f"Available columns: {list(columns)}"
    )


def _split_to_sequences_and_targets(dataset):
    """Extract protein sequences and targets."""
    columns = list(dataset.column_names)
    sequence_col = _detect_column(
        columns,
        ("sequence", "protein_sequence", "primary"),
        "sequence",
    )
    target_col = _detect_column(
        columns,
        ("target", "label", "fluorescence", "stability"),
        "target",
    )
    sequences = [str(sequence) for sequence in dataset[sequence_col]]
    y = np.asarray(dataset[target_col], dtype=np.float64).reshape(-1)
    if len(sequences) != y.shape[0]:
        raise ValueError("Sequence and target columns must have the same length.")
    if not sequences:
        raise ValueError("Protein dataset split is empty.")
    return sequences, y, target_col


def _find_split_name(split_names, candidates):
    """Find a dataset split by candidate names."""
    lower = {name.lower(): name for name in split_names}
    for candidate in candidates:
        name = lower.get(candidate.lower())
        if name is not None:
            return name
    return None


def _concat_records(records):
    """Concatenate sequence-target records."""
    sequences = []
    targets = []
    target_col = None
    for seqs, y, col in records:
        sequences.extend(seqs)
        targets.append(y)
        target_col = target_col or col
    if not targets:
        raise ValueError("No protein records found.")
    return sequences, np.concatenate(targets), target_col


def _all_protein_records(dataset):
    """Collect all protein records from a dataset."""
    if hasattr(dataset, "keys"):
        return _concat_records(
            [_split_to_sequences_and_targets(dataset[name]) for name in dataset.keys()]
        )
    return _split_to_sequences_and_targets(dataset)


def _protein_source_target_records(dataset, test_size, random_state, max_samples):
    """Build source and target protein records."""
    rng = np.random.default_rng(random_state)
    if hasattr(dataset, "keys"):
        split_names = list(dataset.keys())
        train_name = _find_split_name(split_names, ("train", "training"))
        target_names = [
            name
            for name in (
                _find_split_name(split_names, ("validation", "valid", "dev")),
                _find_split_name(split_names, ("test", "testing")),
            )
            if name is not None
        ]
        if train_name is not None and target_names:
            source_sequences, source_y, target_col = _split_to_sequences_and_targets(
                dataset[train_name]
            )
            target_sequences, target_y, target_col = _concat_records(
                [_split_to_sequences_and_targets(dataset[name]) for name in target_names]
            )
            source_sequences, source_y = _limit_records(
                source_sequences,
                source_y,
                max_samples,
                rng,
            )
            target_sequences, target_y = _limit_records(
                target_sequences,
                target_y,
                max_samples,
                rng,
            )
            split_name = f"official:{train_name}->source,{'+'.join(target_names)}->target"
            return source_sequences, source_y, target_sequences, target_y, target_col, split_name

    sequences, y, target_col = _all_protein_records(dataset)
    sequences, y = _limit_records(sequences, y, max_samples, rng)
    if len(sequences) < 4:
        raise ValueError("Protein dataset needs at least four samples for random splitting.")
    n_target = int(np.ceil(len(sequences) * test_size))
    n_target = min(max(n_target, 2), len(sequences) - 2)
    idx = rng.permutation(len(sequences))
    target_idx = idx[:n_target]
    source_idx = idx[n_target:]
    source_sequences = [sequences[i] for i in source_idx]
    target_sequences = [sequences[i] for i in target_idx]
    return (
        source_sequences,
        y[source_idx],
        target_sequences,
        y[target_idx],
        target_col,
        "deterministic_random",
    )


def _resolve_torch_device(torch_module, device):
    """Resolve the torch device to use."""
    if device is None:
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    return device


def _embed_protein_sequences(
    sequences,
    tokenizer,
    model,
    torch_module,
    device,
    batch_size,
):
    """Embed protein sequences with a transformer model."""
    embeddings = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_special_tokens_mask=True,
        )
        special_tokens_mask = encoded.pop("special_tokens_mask", None)
        model_inputs = {key: value.to(device) for key, value in encoded.items()}
        attention_mask = model_inputs["attention_mask"]
        if special_tokens_mask is None:
            special_tokens_mask = torch_module.zeros_like(attention_mask)
        else:
            special_tokens_mask = special_tokens_mask.to(device)

        with torch_module.no_grad():
            output = model(**model_inputs)
            hidden = (
                output.last_hidden_state
                if hasattr(output, "last_hidden_state")
                else output[0]
            )
            mask = attention_mask.bool() & ~special_tokens_mask.bool()
            counts = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / counts
        embeddings.append(pooled.detach().cpu().numpy())

    return np.ascontiguousarray(np.vstack(embeddings), dtype=np.float64)


def _prepare_protein_esm2(args, spec):
    """Prepare an ESM-2 protein cache."""
    datasets_module = _check_optional_import("datasets")
    torch_module = _check_optional_import("torch")
    transformers = _check_optional_import("transformers")
    embedding_model = args.embedding_model or DEFAULT_ESM2_MODEL
    device = _resolve_torch_device(torch_module, args.device)
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    dataset = datasets_module.load_dataset(spec["dataset_source"])
    (
        source_sequences,
        source_y,
        target_sequences,
        target_y,
        target_name,
        split_name,
    ) = _protein_source_target_records(
        dataset,
        args.test_size,
        args.random_state,
        args.max_samples,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(embedding_model)
    model = transformers.AutoModel.from_pretrained(embedding_model)
    model.to(device)
    model.eval()

    X_source = _embed_protein_sequences(
        source_sequences,
        tokenizer,
        model,
        torch_module,
        device,
        batch_size,
    )
    X_target = _embed_protein_sequences(
        target_sequences,
        tokenizer,
        model,
        torch_module,
        device,
        batch_size,
    )

    rng = np.random.default_rng(args.random_state)
    X_target_train, y_target_train, X_target_test, y_target_test = _split_arrays(
        X_target,
        target_y,
        args.test_size,
        rng,
    )
    X_source_train, y_source_train, X_source_test, y_source_test = _split_arrays(
        X_source,
        source_y,
        args.test_size,
        rng,
    )
    arrays = _validate_prepared_arrays(
        {
            "X_target_train": X_target_train,
            "y_target_train": y_target_train,
            "X_source_train": X_source_train,
            "y_source_train": y_source_train,
            "X_target_test": X_target_test,
            "y_target_test": y_target_test,
            "X_source_test": X_source_test,
            "y_source_test": y_source_test,
        },
        expected_dim=1280,
    )
    metadata = {
        "dataset": args.dataset,
        "embedding_model": embedding_model,
        "dataset_source": spec["dataset_source"],
        "target_name": target_name or spec["target_name"],
        "split_name": split_name,
    }
    return arrays, metadata


def _make_protein_esm2_preparer(spec):
    """Build an ESM-2 protein preparer."""
    def _preparer(args):
        """Run the configured preparer."""
        return _prepare_protein_esm2(args, spec)

    return _preparer


def _require_arg(value, flag, dataset_name):
    """Require a dataset-specific CLI argument."""
    if value is None:
        raise ValueError(f"{flag} is required for dataset '{dataset_name}'.")
    return value


def _dataset_to_frame(pd, dataset):
    """Convert a dataset object to a dataframe."""
    if hasattr(dataset, "to_pandas"):
        return dataset.to_pandas()
    if hasattr(dataset, "column_names"):
        return pd.DataFrame({column: dataset[column] for column in dataset.column_names})
    return pd.DataFrame(dataset)


def _hf_dataset_to_frame(pd, dataset):
    """Convert a Hugging Face dataset to a dataframe."""
    if hasattr(dataset, "keys"):
        frames = []
        for split_name in dataset.keys():
            frame = _dataset_to_frame(pd, dataset[split_name])
            frame["_split"] = split_name
            frames.append(frame)
        if not frames:
            raise ValueError("Hugging Face dataset has no splits.")
        return pd.concat(frames, ignore_index=True)
    frame = _dataset_to_frame(pd, dataset)
    frame["_split"] = "all"
    return frame


def _select_column(df, override, candidates, kind):
    """Select a column using an optional override."""
    if override is not None:
        if override not in df.columns:
            raise ValueError(
                f"{kind} column '{override}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        return override
    return _detect_column(df.columns, candidates, kind)


def _combined_domain(df, domain_column, default_value):
    """Build domain labels from one or more columns."""
    if domain_column is None:
        return np.asarray([default_value] * len(df), dtype=object), str(default_value)

    columns = [
        column.strip()
        for column in domain_column.replace("+", ",").split(",")
        if column.strip()
    ]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Domain column(s) {missing} not found. Available columns: {list(df.columns)}"
        )
    if len(columns) == 1:
        return df[columns[0]].astype(str).to_numpy(), columns[0]

    series = df[columns[0]].astype(str)
    for column in columns[1:]:
        series = series + "::" + df[column].astype(str)
    return series.to_numpy(), "+".join(columns)


def _auto_review_domain(df, dataset_name, fallback):
    """Infer review domain labels."""
    if dataset_name == "amazon_reviews_sentence":
        for column in ("category", "main_category", "product_category"):
            if column in df.columns:
                return df[column].astype(str).to_numpy(), column
        return np.asarray([fallback] * len(df), dtype=object), "hf_config"

    if "city" in df.columns and "categories" in df.columns:
        series = df["city"].astype(str) + "::" + df["categories"].astype(str)
        return series.to_numpy(), "city+categories"
    for column in ("city", "categories", "business_category"):
        if column in df.columns:
            return df[column].astype(str).to_numpy(), column
    return np.asarray([fallback] * len(df), dtype=object), "constant"


def _review_source_target_indices(domain, args):
    """Choose review source and target indices."""
    rng = np.random.default_rng(args.random_state)
    n = len(domain)
    if args.target_domain is not None or args.source_domains is not None:
        if args.target_domain is None or args.source_domains is None:
            raise ValueError(
                "--target-domain and --source-domains must be provided together "
                "for domain-based review splits."
            )
        target_mask = domain == args.target_domain
        source_mask = np.isin(domain, args.source_domains)
        if target_mask.sum() < 2:
            raise ValueError("Target domain must contain at least two reviews.")
        if source_mask.sum() < 2:
            raise ValueError("Source domains must contain at least two reviews.")
        return np.flatnonzero(source_mask), np.flatnonzero(target_mask), "domain"

    if n < 4:
        raise ValueError("Review dataset needs at least four rows for random splitting.")
    n_target = int(np.ceil(n * args.test_size))
    n_target = min(max(n_target, 2), n - 2)
    idx = rng.permutation(n)
    return idx[n_target:], idx[:n_target], "deterministic_random"


def _sentence_device(device):
    """Resolve the sentence embedding device."""
    if device is not None:
        return device
    try:
        torch_module = importlib.import_module("torch")
    except ImportError:
        return "cpu"
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def _encode_sentence_texts(texts, model, batch_size):
    """Encode review texts with a sentence model."""
    try:
        X = model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
    except TypeError:
        X = model.encode(list(texts), batch_size=batch_size)
    return np.ascontiguousarray(np.asarray(X, dtype=np.float64))


def _review_arrays_from_frame(df, args, dataset_name, default_domain):
    """Extract review texts, targets, and domains."""
    rng = np.random.default_rng(args.random_state)
    df = _limit_frame(df, args.max_samples, rng).reset_index(drop=True)
    text_col = _select_column(
        df,
        args.text_column,
        ("text", "review_text", "reviewText", "body", "content", "review"),
        "text",
    )
    target_col = _select_column(
        df,
        args.target_column,
        ("stars", "rating", "overall", "star_rating", "score"),
        "target",
    )
    if args.domain_column is not None:
        domain, domain_name = _combined_domain(df, args.domain_column, default_domain)
    else:
        domain, domain_name = _auto_review_domain(df, dataset_name, default_domain)

    texts = df[text_col].astype(str).fillna("").tolist()
    y = df[target_col].astype(float).to_numpy(dtype=np.float64)
    source_idx, target_idx, split_name = _review_source_target_indices(domain, args)
    return (
        [texts[i] for i in source_idx],
        y[source_idx],
        [texts[i] for i in target_idx],
        y[target_idx],
        text_col,
        target_col,
        domain_name,
        split_name,
    )


def _load_amazon_reviews_frame(args):
    """Load Amazon reviews into a dataframe."""
    datasets_module = _check_optional_import("datasets")
    pd = _check_optional_import("pandas")
    hf_config = _require_arg(args.hf_config, "--hf-config", args.dataset)
    dataset = datasets_module.load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        hf_config,
    )
    return _hf_dataset_to_frame(pd, dataset), hf_config


def _find_first_existing(root, candidates):
    """Find the first existing candidate path."""
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    return None


def _load_yelp_reviews_frame(args):
    """Load Yelp reviews into a dataframe."""
    pd = _check_optional_import("pandas")
    input_path = Path(_require_arg(args.input_path, "--input-path", args.dataset))
    if not input_path.exists():
        raise FileNotFoundError(f"Yelp input path not found: {input_path}")

    business_path = None
    if input_path.is_dir():
        review_path = _find_first_existing(
            input_path,
            (
                "yelp_academic_dataset_review.json",
                "review.json",
                "reviews.json",
            ),
        )
        business_path = _find_first_existing(
            input_path,
            (
                "yelp_academic_dataset_business.json",
                "business.json",
                "businesses.json",
            ),
        )
        if review_path is None:
            raise FileNotFoundError(
                f"No Yelp review JSON file found under {input_path}."
            )
    else:
        review_path = input_path

    reviews = pd.read_json(review_path, lines=True)
    if business_path is not None and "business_id" in reviews.columns:
        business = pd.read_json(business_path, lines=True)
        if "business_id" in business.columns:
            reviews = reviews.merge(
                business,
                on="business_id",
                how="left",
                suffixes=("", "_business"),
            )
    return reviews, str(review_path)


def _prepare_review_sentence(args, spec):
    """Prepare a review sentence-embedding cache."""
    embedding_model = args.embedding_model or DEFAULT_SENTENCE_MODEL
    device = _sentence_device(args.device)
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if args.dataset == "amazon_reviews_sentence":
        df, data_source = _load_amazon_reviews_frame(args)
    elif args.dataset == "yelp_reviews_sentence":
        df, data_source = _load_yelp_reviews_frame(args)
    else:
        raise ValueError(f"Unknown review sentence dataset: {args.dataset}")

    (
        source_texts,
        source_y,
        target_texts,
        target_y,
        text_col,
        target_col,
        domain_name,
        split_name,
    ) = _review_arrays_from_frame(df, args, args.dataset, data_source)

    sentence_transformers = _check_optional_import("sentence_transformers")
    try:
        model = sentence_transformers.SentenceTransformer(embedding_model, device=device)
    except TypeError:
        model = sentence_transformers.SentenceTransformer(embedding_model)

    X_source = _encode_sentence_texts(source_texts, model, batch_size)
    X_target = _encode_sentence_texts(target_texts, model, batch_size)
    rng = np.random.default_rng(args.random_state)
    X_target_train, y_target_train, X_target_test, y_target_test = _split_arrays(
        X_target,
        target_y,
        args.test_size,
        rng,
    )
    X_source_train, y_source_train, X_source_test, y_source_test = _split_arrays(
        X_source,
        source_y,
        args.test_size,
        rng,
    )
    arrays = _validate_prepared_arrays(
        {
            "X_target_train": X_target_train,
            "y_target_train": y_target_train,
            "X_source_train": X_source_train,
            "y_source_train": y_source_train,
            "X_target_test": X_target_test,
            "y_target_test": y_target_test,
            "X_source_test": X_source_test,
            "y_source_test": y_source_test,
        },
        expected_dim=384,
    )
    metadata = {
        "dataset": args.dataset,
        "embedding_model": embedding_model,
        "dataset_source": spec["dataset_source"],
        "data_source": data_source,
        "target_name": spec["target_name"],
        "text_column": text_col,
        "target_column": target_col,
        "domain_column": domain_name,
        "split_name": split_name,
    }
    return arrays, metadata


def _make_review_sentence_preparer(spec):
    """Build a review sentence preparer."""
    def _preparer(args):
        """Run the configured preparer."""
        return _prepare_review_sentence(args, spec)

    return _preparer


def _is_url(value):
    """Return whether a value is a URL."""
    scheme = urllib.parse.urlparse(str(value)).scheme
    return scheme in {"http", "https", "ftp", "file"}


def _resolve_path_or_url(value, tmpdir, label):
    """Resolve a local path or URL."""
    value = _require_arg(value, f"--{label.replace('_', '-')}", "e2006_tfidf_svd512")
    if _is_url(value):
        parsed = urllib.parse.urlparse(value)
        filename = Path(parsed.path).name or f"{label}.bz2"
        path = Path(tmpdir) / filename
        urllib.request.urlretrieve(value, str(path))
        return path

    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"E2006 {label} path not found: {path}")
    return path


def _load_e2006_file(load_svmlight_file, path, n_features):
    """Load an E2006 svmlight file."""
    X, y = load_svmlight_file(str(path), n_features=n_features)
    return X, np.asarray(y, dtype=np.float64).reshape(-1)


def _prepare_e2006_tfidf_svd(args, spec):
    """Prepare the E2006 SVD cache."""
    train_value = _require_arg(args.train_path, "--train-path", args.dataset)
    test_value = _require_arg(args.test_path, "--test-path", args.dataset)
    sklearn_datasets = _check_optional_import("sklearn.datasets", "scikit-learn")
    sklearn_decomposition = _check_optional_import("sklearn.decomposition", "scikit-learn")
    scipy_sparse = _check_optional_import("scipy.sparse", "scipy")
    raw_dim = int(spec["raw_dim"])
    n_components = int(spec["d"])
    warnings.warn(
        f"E2006 raw TF-IDF dimension is {raw_dim}; this cache stores "
        f"{n_components}-dimensional dense TruncatedSVD embeddings.",
        RuntimeWarning,
        stacklevel=2,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = _resolve_path_or_url(train_value, tmpdir, "train_path")
        test_path = _resolve_path_or_url(test_value, tmpdir, "test_path")
        X_source, y_source = _load_e2006_file(
            sklearn_datasets.load_svmlight_file,
            train_path,
            raw_dim,
        )
        X_target, y_target = _load_e2006_file(
            sklearn_datasets.load_svmlight_file,
            test_path,
            raw_dim,
        )

    rng = np.random.default_rng(args.random_state)
    X_source, y_source = _limit_samples(X_source, y_source, args.max_samples, rng)
    X_target, y_target = _limit_samples(X_target, y_target, args.max_samples, rng)
    X_target_train_raw, y_target_train, X_target_test_raw, y_target_test = _split_arrays(
        X_target,
        y_target,
        args.test_size,
        rng,
    )
    X_source_train_raw, y_source_train, X_source_test_raw, y_source_test = _split_arrays(
        X_source,
        y_source,
        args.test_size,
        rng,
    )

    train_pool = scipy_sparse.vstack(
        [X_source_train_raw, X_target_train_raw],
        format="csr",
    )
    svd = sklearn_decomposition.TruncatedSVD(
        n_components=n_components,
        random_state=args.random_state,
    )
    svd.fit(train_pool)

    arrays = _validate_prepared_arrays(
        {
            "X_target_train": _dense_float64(svd.transform(X_target_train_raw)),
            "y_target_train": y_target_train,
            "X_source_train": _dense_float64(svd.transform(X_source_train_raw)),
            "y_source_train": y_source_train,
            "X_target_test": _dense_float64(svd.transform(X_target_test_raw)),
            "y_target_test": y_target_test,
            "X_source_test": _dense_float64(svd.transform(X_source_test_raw)),
            "y_source_test": y_source_test,
        },
        expected_dim=n_components,
    )
    metadata = {
        "dataset": args.dataset,
        "target": spec["target"],
        "feature_type": spec["feature_type"],
        "raw_tfidf_dim": raw_dim,
        "svd_n_components": n_components,
        "train_path": args.train_path,
        "test_path": args.test_path,
        "split_name": "E2006.train->source,E2006.test->target",
    }
    return arrays, metadata


def _make_e2006_tfidf_svd_preparer(spec):
    """Build an E2006 SVD preparer."""
    def _preparer(args):
        """Run the configured preparer."""
        return _prepare_e2006_tfidf_svd(args, spec)

    return _preparer


def _column_key(value):
    """Normalize a column name for matching."""
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _detect_column_loose(columns, candidates, kind, required=True):
    """Detect a column with loose name matching."""
    exact = {str(column).lower(): column for column in columns}
    normalized = {}
    for column in columns:
        normalized.setdefault(_column_key(column), column)

    for candidate in candidates:
        column = exact.get(str(candidate).lower())
        if column is not None:
            return column
        column = normalized.get(_column_key(candidate))
        if column is not None:
            return column

    if not required:
        return None
    raise ValueError(
        f"Could not detect {kind} column. Tried {candidates}. "
        f"Available columns: {list(columns)}"
    )


def _select_column_loose(df, override, candidates, kind):
    """Select a loosely matched column."""
    if override is not None:
        column = _detect_column_loose(df.columns, (override,), kind, required=False)
        if column is None:
            raise ValueError(
                f"{kind} column '{override}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        return column
    return _detect_column_loose(df.columns, candidates, kind)


def _load_tdc_drug_response_frame(tdc_multi_pred, pd, spec):
    """Load a TDC DrugRes dataframe."""
    dataset = tdc_multi_pred.DrugRes(name=spec["tdc_name"])
    if not hasattr(dataset, "get_data"):
        raise ValueError("TDC DrugRes object does not expose get_data().")

    try:
        frame = dataset.get_data()
    except TypeError:
        frame = dataset.get_data(format="df")
    if hasattr(frame, "columns"):
        return frame.copy()
    return pd.DataFrame(frame)


def _tdc_morgan_fingerprint_function(fingerprint_dim):
    """Build the TDC Morgan fingerprint function."""
    chem = _check_optional_import("rdkit.Chem", "rdkit")
    try:
        rd_fingerprint_generator = _check_optional_import(
            "rdkit.Chem.rdFingerprintGenerator",
            "rdkit",
        )
        generator = rd_fingerprint_generator.GetMorganGenerator(
            radius=2,
            fpSize=fingerprint_dim,
        )
        return chem, generator.GetFingerprint
    except (ImportError, AttributeError):
        all_chem = _check_optional_import("rdkit.Chem.AllChem", "rdkit")
        return chem, lambda mol: all_chem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=fingerprint_dim,
        )


def _fingerprint_to_array(fingerprint, fingerprint_dim, np_module):
    """Convert a fingerprint to a dense array."""
    if hasattr(fingerprint, "ToBitString"):
        bit_string = fingerprint.ToBitString()
        if len(bit_string) != fingerprint_dim:
            raise ValueError(
                f"Fingerprint has length {len(bit_string)}, expected {fingerprint_dim}."
            )
        return np_module.fromiter(
            (1.0 if bit == "1" else 0.0 for bit in bit_string),
            dtype=np_module.float64,
            count=fingerprint_dim,
        )

    arr = np_module.asarray(fingerprint, dtype=np_module.float64).reshape(-1)
    if arr.size != fingerprint_dim:
        raise ValueError(
            f"Fingerprint has length {arr.size}, expected {fingerprint_dim}."
        )
    return arr


def _tdc_fingerprint_matrix(smiles, chem, fingerprint_fn, fingerprint_dim, np_module):
    """Build a TDC drug fingerprint matrix."""
    rows = []
    invalid_rows = []
    for i, value in enumerate(smiles):
        mol = chem.MolFromSmiles(str(value))
        if mol is None:
            invalid_rows.append(i)
            continue
        rows.append(_fingerprint_to_array(fingerprint_fn(mol), fingerprint_dim, np_module))

    if invalid_rows:
        raise ValueError(
            "Invalid SMILES values at prepared rows: "
            f"{invalid_rows[:10]}{'...' if len(invalid_rows) > 10 else ''}"
        )
    if not rows:
        raise ValueError("No valid SMILES values found.")
    return np_module.ascontiguousarray(np_module.vstack(rows), dtype=np_module.float64)


def _looks_like_expression_column(column):
    """Return whether a column looks like expression data."""
    key = _column_key(column)
    return any(
        token in key
        for token in ("expression", "expr", "rna", "transcriptome", "transcript")
    )


def _stack_vector_column(series, np_module):
    """Stack vector-valued dataframe cells."""
    vectors = []
    for value in series:
        vector = np_module.asarray(value, dtype=np_module.float64).reshape(-1)
        vectors.append(vector)
    widths = {vector.size for vector in vectors}
    if len(widths) != 1:
        raise ValueError("Expression feature vectors must have a consistent length.")
    return np_module.ascontiguousarray(np_module.vstack(vectors), dtype=np_module.float64)


def _tdc_expression_features(df, pd, np_module, excluded_columns):
    """Extract expression features from a TDC table."""
    candidate_columns = [
        column
        for column in df.columns
        if column not in excluded_columns and _looks_like_expression_column(column)
    ]
    numeric_columns = []
    for column in candidate_columns:
        series = df[column]
        first_valid = series.dropna().iloc[0] if series.notna().any() else None
        if isinstance(first_valid, (list, tuple, np_module.ndarray)):
            return (
                _stack_vector_column(series, np_module),
                "cell_line_expression_vector",
                True,
                str(column),
            )

        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().all():
            numeric_columns.append(column)

    if numeric_columns:
        return (
            np_module.ascontiguousarray(
                df[numeric_columns].astype(float).to_numpy(dtype=np_module.float64),
                dtype=np_module.float64,
            ),
            "cell_line_expression_columns",
            True,
            ",".join(str(column) for column in numeric_columns),
        )

    return None, "cell_line_one_hot", False, ""


def _one_hot(values, np_module):
    """One-hot encode categorical values."""
    labels = [str(value) for value in values]
    levels = sorted(set(labels))
    if not levels:
        raise ValueError("Cannot one-hot encode an empty cell-line column.")
    mapping = {value: i for i, value in enumerate(levels)}
    X = np_module.zeros((len(labels), len(levels)), dtype=np_module.float64)
    for row, value in enumerate(labels):
        X[row, mapping[value]] = 1.0
    return np_module.ascontiguousarray(X, dtype=np_module.float64), levels


def _tdc_cell_features(df, pd, np_module, cell_line_col, excluded_columns):
    """Build TDC cell-line feature arrays."""
    X_expr, feature_type, expression_available, feature_names = _tdc_expression_features(
        df,
        pd,
        np_module,
        excluded_columns,
    )
    if X_expr is not None:
        return X_expr, feature_type, expression_available, feature_names

    warnings.warn(
        "TDC DrugRes data did not include directly usable expression features; "
        "falling back to drug fingerprint plus one-hot cell-line encoding.",
        RuntimeWarning,
        stacklevel=2,
    )
    X_cell, levels = _one_hot(df[cell_line_col].astype(str), np_module)
    return X_cell, feature_type, expression_available, ",".join(levels)


def _tdc_domain_values(df, args, drug_col, drug_id_col, cell_line_col, lineage_col):
    """Choose TDC domain labels."""
    if args.domain_column is not None:
        requested = str(args.domain_column).strip()
        if "," in requested or "+" in requested:
            return _combined_domain(df, requested, "tdc")

        alias = _column_key(requested)
        aliases = {
            "drug": drug_id_col or drug_col,
            "drugid": drug_id_col or drug_col,
            "cell": cell_line_col,
            "cellline": cell_line_col,
            "cancerlineage": lineage_col,
            "lineage": lineage_col,
        }
        if alias in aliases:
            column = aliases[alias]
            if column is None:
                raise ValueError(
                    f"Domain alias '{requested}' is not available in this TDC table."
                )
            return df[column].astype(str).to_numpy(), str(column)
        return _combined_domain(df, requested, "tdc")

    if lineage_col is not None:
        return df[lineage_col].astype(str).to_numpy(), str(lineage_col)
    return df[cell_line_col].astype(str).to_numpy(), str(cell_line_col)


def _tdc_source_target_indices(domain, args):
    """Choose TDC source and target indices."""
    rng = np.random.default_rng(args.random_state)
    n = len(domain)
    if args.target_domain is not None or args.source_domains is not None:
        if args.target_domain is None or args.source_domains is None:
            raise ValueError(
                "--target-domain and --source-domains must be provided together "
                "for domain-based TDC DrugRes splits."
            )
        target_mask = domain == args.target_domain
        source_mask = np.isin(domain, args.source_domains)
        if target_mask.sum() < 2:
            raise ValueError(
                "Target domain must contain at least two drug-response samples."
            )
        if source_mask.sum() < 2:
            raise ValueError(
                "Source domains must contain at least two drug-response samples."
            )
        return np.flatnonzero(source_mask), np.flatnonzero(target_mask), "domain"

    if n < 4:
        raise ValueError(
            "TDC DrugRes dataset needs at least four rows for random splitting."
        )
    n_target = int(np.ceil(n * args.test_size))
    n_target = min(max(n_target, 2), n - 2)
    idx = rng.permutation(n)
    return idx[n_target:], idx[:n_target], "deterministic_random"


def _prepare_tdc_drug_response(args, spec):
    """Prepare a TDC drug-response cache."""
    tdc = _check_optional_import("tdc", "PyTDC")
    tdc_multi_pred = getattr(tdc, "multi_pred", None)
    if tdc_multi_pred is None:
        tdc_multi_pred = _check_optional_import("tdc.multi_pred", "PyTDC")
    _check_optional_import("rdkit", "rdkit")
    pd = _check_optional_import("pandas")
    np_module = _check_optional_import("numpy")

    fingerprint_dim = int(args.fingerprint_dim)
    if fingerprint_dim <= 0:
        raise ValueError("fingerprint_dim must be positive.")

    frame = _load_tdc_drug_response_frame(tdc_multi_pred, pd, spec)
    if frame.empty:
        raise ValueError(f"TDC DrugRes dataset '{spec['tdc_name']}' is empty.")

    smiles_col = _select_column_loose(
        frame,
        args.text_column,
        (
            "smiles",
            "canonical_smiles",
            "drug_smiles",
            "Drug_SMILES",
            "Drug SMILES",
            "Drug",
        ),
        "SMILES",
    )
    target_col = _select_column_loose(
        frame,
        args.target_column,
        (
            "Y",
            "label",
            "target",
            "IC50",
            "LN_IC50",
            "response",
            "drug_response",
            "drug response",
            "AUC",
        ),
        "target",
    )
    drug_id_col = _detect_column_loose(
        frame.columns,
        (
            "Drug_ID",
            "drug_id",
            "Drug ID",
            "drug_name",
            "Drug_Name",
            "compound_id",
            "compound_name",
        ),
        "drug id",
        required=False,
    )
    cell_line_col = _detect_column_loose(
        frame.columns,
        (
            "Cell Line",
            "Cell_Line",
            "CellLine",
            "cell_line",
            "Cell_Line_ID",
            "Cell Line_ID",
            "cell_line_id",
            "cell line id",
            "COSMIC_ID",
            "sample_id",
            "sample",
        ),
        "cell-line",
    )
    lineage_col = _detect_column_loose(
        frame.columns,
        (
            "cancer_lineage",
            "cancer lineage",
            "lineage",
            "cancer_type",
            "cancer type",
            "tissue",
            "tissue_type",
            "primary_disease",
        ),
        "cancer lineage",
        required=False,
    )

    y = pd.to_numeric(frame[target_col], errors="coerce")
    required = frame[[smiles_col, cell_line_col]].notna().all(axis=1) & y.notna()
    if not required.all():
        dropped = int((~required).sum())
        warnings.warn(
            f"Dropping {dropped} rows with missing SMILES, cell-line, or target values.",
            RuntimeWarning,
            stacklevel=2,
        )
        frame = frame.loc[required].reset_index(drop=True)
        y = y.loc[required].reset_index(drop=True)
    if len(frame) < 4:
        raise ValueError("TDC DrugRes dataset needs at least four valid rows.")

    rng = np_module.random.default_rng(args.random_state)
    if args.max_samples is not None and args.max_samples > 0 and len(frame) > args.max_samples:
        idx = rng.permutation(len(frame))[:args.max_samples]
        frame = frame.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)
        y = y.reset_index(drop=True)
    if len(frame) < 4:
        raise ValueError("TDC DrugRes dataset needs at least four rows after sampling.")
    domain, domain_name = _tdc_domain_values(
        frame,
        args,
        smiles_col,
        drug_id_col,
        cell_line_col,
        lineage_col,
    )

    chem, fingerprint_fn = _tdc_morgan_fingerprint_function(fingerprint_dim)
    X_drug = _tdc_fingerprint_matrix(
        frame[smiles_col].astype(str),
        chem,
        fingerprint_fn,
        fingerprint_dim,
        np_module,
    )
    excluded_columns = {
        smiles_col,
        target_col,
        drug_id_col,
        cell_line_col,
        lineage_col,
        domain_name,
    }
    excluded_columns.discard(None)
    X_cell, cell_feature_type, expression_available, cell_feature_names = _tdc_cell_features(
        frame,
        pd,
        np_module,
        cell_line_col,
        excluded_columns,
    )
    X = np_module.ascontiguousarray(
        np_module.hstack([X_drug, X_cell]),
        dtype=np_module.float64,
    )
    y_array = np_module.ascontiguousarray(
        np_module.asarray(y, dtype=np_module.float64).reshape(-1),
        dtype=np_module.float64,
    )
    source_idx, target_idx, split_name = _tdc_source_target_indices(domain, args)
    X_target_train, y_target_train, X_target_test, y_target_test = _split_arrays(
        X[target_idx],
        y_array[target_idx],
        args.test_size,
        rng,
    )
    X_source_train, y_source_train, X_source_test, y_source_test = _split_arrays(
        X[source_idx],
        y_array[source_idx],
        args.test_size,
        rng,
    )
    actual_d = int(X.shape[1])
    arrays = _validate_prepared_arrays(
        {
            "X_target_train": X_target_train,
            "y_target_train": y_target_train,
            "X_source_train": X_source_train,
            "y_source_train": y_source_train,
            "X_target_test": X_target_test,
            "y_target_test": y_target_test,
            "X_source_test": X_source_test,
            "y_source_test": y_source_test,
        },
        expected_dim=actual_d,
    )
    metadata = {
        "dataset": args.dataset,
        "dataset_source": spec["source"],
        "tdc_name": spec["tdc_name"],
        "target": spec["target"],
        "target_column": target_col,
        "smiles_column": smiles_col,
        "drug_id_column": drug_id_col or "",
        "cell_line_column": cell_line_col,
        "domain_column": domain_name,
        "split_name": split_name,
        "fingerprint_type": "Morgan/ECFP",
        "fingerprint_dim": fingerprint_dim,
        "cell_feature_type": cell_feature_type,
        "cell_feature_names": cell_feature_names,
        "expression_features_available": bool(expression_available),
        "actual_d": actual_d,
    }
    return arrays, metadata


def _make_tdc_drug_response_preparer(spec):
    """Build a TDC drug-response preparer."""
    def _preparer(args):
        """Run the configured preparer."""
        return _prepare_tdc_drug_response(args, spec)

    return _preparer


PREPARERS = {
    name: _make_moleculenet_ecfp_preparer(spec)
    for name, spec in MOLECULENET_ECFP_DATASETS.items()
}
PREPARERS.update(
    {
        name: _make_protein_esm2_preparer(spec)
        for name, spec in PROTEIN_ESM2_DATASETS.items()
    }
)
PREPARERS.update(
    {
        name: _make_review_sentence_preparer(spec)
        for name, spec in REVIEW_SENTENCE_DATASETS.items()
    }
)
PREPARERS.update(
    {
        name: _make_e2006_tfidf_svd_preparer(spec)
        for name, spec in E2006_TFIDF_SVD_DATASETS.items()
    }
)
PREPARERS.update(
    {
        name: _make_tdc_drug_response_preparer(spec)
        for name, spec in TDC_DRUG_RESPONSE_DATASETS.items()
    }
)


def _build_parser():
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Prepare local npz caches for embedding-regression datasets."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--target-domain",
        default=None,
        help="Optional and dataset-dependent; required by explicit-domain preparers.",
    )
    parser.add_argument(
        "--source-domains",
        nargs="+",
        default=None,
        help="Optional and dataset-dependent; required by explicit-domain preparers.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Optional and dataset-dependent; defaults may be provided by preparers.",
    )
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--test-size", type=float, required=True)
    parser.add_argument("--random-state", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device",
        default=None,
        help=(
            'Device for neural embedding preparers; defaults to "cuda" when '
            'available, else "cpu".'
        ),
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Local input path for file-based preparers such as Yelp.",
    )
    parser.add_argument(
        "--hf-config",
        default=None,
        help="Hugging Face dataset config/category; required by Amazon reviews.",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="Optional text column override.",
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="Optional regression target column override.",
    )
    parser.add_argument(
        "--domain-column",
        default=None,
        help="Optional domain column override; combine columns with comma or plus.",
    )
    parser.add_argument(
        "--fingerprint-dim",
        type=int,
        default=2048,
        help="Morgan/ECFP fingerprint dimension for fingerprint-based preparers.",
    )
    parser.add_argument(
        "--train-path",
        default=None,
        help="Local path or URL for E2006.train.bz2.",
    )
    parser.add_argument(
        "--test-path",
        default=None,
        help="Local path or URL for E2006.test.bz2.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _run(args):
    """Run the selected dataset preparer."""
    if args.dataset not in PREPARERS:
        available = sorted(PREPARERS)
        raise ValueError(
            f"Embedding-regression dataset '{args.dataset}' is not implemented. "
            f"Available preparers: {available}"
        )

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    result = PREPARERS[args.dataset](args)
    if isinstance(result, tuple) and len(result) == 2:
        arrays, metadata = result
    else:
        arrays, metadata = result, None

    cache_path = output_dir / f"{args.dataset}.npz"
    return _save_embedding_cache(
        cache_path,
        arrays,
        metadata=metadata,
        overwrite=args.overwrite,
    )


def main(argv=None):
    """Run the preparation CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    main()
