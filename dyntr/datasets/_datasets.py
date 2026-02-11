"""Datasets module."""
from __future__ import annotations
import os
from typing import Iterable, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def _generate_thetas(
    d: int,
    epsilon: float,
    random_state=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return theta_s, theta_t such that ||theta_s - theta_t||_2  =  epsilon."""
    if d <= 0:
        raise ValueError("d must be positive.")
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")

    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, (int, np.integer)):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    theta_s = rng.normal(size=d)
    v = rng.normal(size=d)
    return theta_s, theta_s + (epsilon / np.linalg.norm(v)) * v


def _assert_is_regression(y: np.ndarray, name: str, min_unique: int = 3) -> None:
    """Raise if y looks like classification (e.g., binary)."""
    y = np.asarray(y)
    uq = np.unique(y[~np.isnan(y)])
    assert uq.size >= min_unique, (
        f"[{name}] target looks like classification "
        f"(only {uq.size} unique values: {uq[:10]}). "
        f"Use a classification loader or change the dataset."
    )


def _bootstrap_sources(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_sources: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if n_sources == 1:
        return [X.copy()], [y.copy()]
    l_X = []
    l_y = []
    n = X.shape[0]
    for _ in range(n_sources):
        idx = rng.integers(0, n, size=n)
        l_X.append(X[idx].copy())
        l_y.append(y[idx].copy())
    return l_X, l_y


def _condition_blocks(
    X1_tr: np.ndarray,
    X2_tr: np.ndarray,
    X1_te: np.ndarray,
    X2_te: np.ndarray,
    *,
    pca_var_threshold=0.99,
    standardize=True,
    var_threshold=1e-12,
    add_intercept=False,
):
    """
    Condition the two design matrices jointly (fit transforms on TRAIN only).
    Returns transformed (X1_tr, X2_tr, X1_te, X2_te).
    """
    Xtr_all = np.vstack([X1_tr, X2_tr]).astype(np.float64, copy=False)

    if pca_var_threshold:
        pca = PCA(n_components=pca_var_threshold, whiten=True)
        X1_tr = pca.fit_transform(X1_tr)
        X2_tr = pca.transform(X2_tr)
        X1_te = pca.transform(X1_te)
        X2_te = pca.transform(X2_te)
        Xtr_all = pca.transform(Xtr_all)

    mu = Xtr_all.mean(axis=0)
    std = Xtr_all.std(axis=0, ddof=0)
    keep_var = std > var_threshold
    if not np.all(keep_var):
        X1_tr = X1_tr[:, keep_var]
        X2_tr = X2_tr[:, keep_var]
        X1_te = X1_te[:, keep_var]
        X2_te = X2_te[:, keep_var]
        mu = mu[keep_var]
        std = std[keep_var]

    if standardize:
        std_safe = np.where(std > 0.0, std, 1.0)
        X1_tr = (X1_tr - mu) / std_safe
        X2_tr = (X2_tr - mu) / std_safe
        X1_te = (X1_te - mu) / std_safe
        X2_te = (X2_te - mu) / std_safe

    if add_intercept:
        ones1_tr = np.ones((X1_tr.shape[0], 1), dtype=np.float64)
        ones2_tr = np.ones((X2_tr.shape[0], 1), dtype=np.float64)
        ones1_te = np.ones((X1_te.shape[0], 1), dtype=np.float64)
        ones2_te = np.ones((X2_te.shape[0], 1), dtype=np.float64)
        X1_tr = np.hstack([X1_tr, ones1_tr])
        X2_tr = np.hstack([X2_tr, ones2_tr])
        X1_te = np.hstack([X1_te, ones1_te])
        X2_te = np.hstack([X2_te, ones2_te])

    X1_tr = np.ascontiguousarray(X1_tr, dtype=np.float64)
    X2_tr = np.ascontiguousarray(X2_tr, dtype=np.float64)
    X1_te = np.ascontiguousarray(X1_te, dtype=np.float64)
    X2_te = np.ascontiguousarray(X2_te, dtype=np.float64)
    return X1_tr, X2_tr, X1_te, X2_te


def _load_boston_dataset():
    """
    Load California Housing dataset (Boston deprecated).

    Returns
    -------
    train_data : tuple
        ((X1_train, y1_train), (X2_train, y2_train))
    test_data : tuple
        ((X1_test, y1_test), (X2_test, y2_test))
    """
    data = fetch_california_housing()
    X_full = data.data.astype(np.float64)
    y_full = data.target.astype(np.float64)
    n = len(y_full)
    split = n // 2

    X1_train, y1_train = X_full[: split // 2], y_full[: split // 2]
    X2_train, y2_train = X_full[split // 2:split], y_full[split // 2:split]
    X1_test, y1_test = X_full[split: split + (n - split) // 2], y_full[
        split: split + (n - split) // 2
    ]
    X2_test, y2_test = X_full[split + (n - split) // 2:], y_full[split + (n - split) // 2:]

    scaler = StandardScaler()
    X_all = np.vstack([X1_train, X2_train, X1_test, X2_test])
    scaler.fit(X_all)
    X1_train = scaler.transform(X1_train)
    X2_train = scaler.transform(X2_train)
    X1_test = scaler.transform(X1_test)
    X2_test = scaler.transform(X2_test)

    train_data = (X1_train, y1_train), (X2_train, y2_train)
    test_data = (X1_test, y1_test), (X2_test, y2_test)
    return train_data, test_data


def _load_hillstrom_dataset(
    data_dir=os.path.join(DATA_DIR, "data"),
    filename="Hillstrom.csv",
    target="spend",
    split="high_value_vs_low_value",
    pct_high_value=0.3,
) -> Tuple[
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
]:
    """
    Load Hillstrom Email dataset and split tasks by a business cluster.

    The two tasks (D1, D2) are defined by the cluster, then each task is split
    50/50 into train/test. Returns the same API as _load_boston_dataset.

    Returns
    -------
    train_data : tuple
        ((X1_train, y1_train), (X2_train, y2_train))
    test_data : tuple
        ((X1_test, y1_test), (X2_test, y2_test))
    """
    path = os.path.join(data_dir, "email", filename)
    if not os.path.exists(path):
        path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataset.")

    y_full = df[target].to_numpy(dtype=np.float64)
    drop_cols = [target] + (["customer_id"] if "customer_id" in df.columns else [])
    X_raw = df.drop(columns=drop_cols)
    X = pd.get_dummies(X_raw, drop_first=True).to_numpy(dtype=np.float64)

    if split == "existing_vs_new":
        if "history" not in df.columns:
            raise ValueError("Column 'history' is required for split = 'existing_vs_new'.")
        mask = df["history"].to_numpy(dtype=float) > 0.0
    elif split == "womens_vs_control":
        if "segment" not in df.columns:
            raise ValueError("Column 'segment' is required for split = 'womens_vs_control'.")
        mask = df["segment"].astype(str).str.contains("Wom", case=False, regex=False).to_numpy()
        ctrl = df["segment"].astype(str).str.contains("Control", case=False, regex=False).to_numpy()
        if ctrl.sum() == 0:
            ctrl = ~mask
        mask = mask
        comp_mask = ctrl
    elif split == "high_value_vs_low_value":
        if "history" not in df.columns:
            raise ValueError("Column 'history' is required for split = 'high_value_vs_low_value'.")
        q = np.quantile(df["history"].to_numpy(dtype=float), 1.0 - float(pct_high_value))
        mask = df["history"].to_numpy(dtype=float) >= q
    else:
        raise ValueError("Unknown split option.")

    if split != "womens_vs_control":
        comp_mask = ~mask

    X1, y1 = X[mask], y_full[mask]
    X2, y2 = X[comp_mask], y_full[comp_mask]

    n1, n2 = len(y1), len(y2)
    if n1 == 0 or n2 == 0:
        raise ValueError("One cluster is empty after the split.")

    n1_train, n2_train = n1 // 2, n2 // 2
    X1_train, y1_train = X1[:n1_train], y1[:n1_train]
    X1_test, y1_test = X1[n1_train:], y1[n1_train:]
    X2_train, y2_train = X2[:n2_train], y2[:n2_train]
    X2_test, y2_test = X2[n2_train:], y2[n2_train:]

    scaler = StandardScaler()
    X_all = np.vstack([X1_train, X2_train, X1_test, X2_test])
    scaler.fit(X_all)
    X1_train = scaler.transform(X1_train)
    X2_train = scaler.transform(X2_train)
    X1_test = scaler.transform(X1_test)
    X2_test = scaler.transform(X2_test)

    train_data = (X1_train, y1_train), (X2_train, y2_train)
    test_data = (X1_test, y1_test), (X2_test, y2_test)
    return train_data, test_data


def _guess_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    required=True,
    case_insensitive=True,
) -> Optional[str]:
    cols = list(df.columns)
    if case_insensitive:
        lower = {c.lower(): c for c in cols}
        for cand in candidates:
            c = lower.get(cand.lower())
            if c is not None:
                return c
    else:
        for cand in candidates:
            if cand in df.columns:
                return cand

    if required:
        raise ValueError(
            "None of the candidate columns found. "
            f"Tried {list(candidates)}. Available: {cols}"
        )
    return None


def _autodetect_user_id(purchases: pd.DataFrame, survey: pd.DataFrame) -> str:
    id_candidates = [
        "user_id",
        "userid",
        "user",
        "id",
        "uid",
        "profile_id",
        "respondent_id",
        "participant_id",
        "customer_id",
        "account_id",
        "member_id",
    ]
    purch_id = None
    surv_id = None

    try:
        purch_id = _guess_column(purchases, id_candidates, required=False)
    except ValueError:
        purch_id = None

    try:
        surv_id = _guess_column(survey, id_candidates, required=False)
    except ValueError:
        surv_id = None

    if purch_id and surv_id and purch_id.lower() == surv_id.lower():
        return surv_id

    inter = set(c.lower() for c in purchases.columns) & set(c.lower() for c in survey.columns)
    inter_idlike = [c for c in inter if any(tok in c for tok in ["id", "user", "uid", "account", "profile"])]
    inter_scan = inter_idlike if inter_idlike else list(inter)
    if inter_scan:
        surv_lower = {c.lower(): c for c in survey.columns}
        for c in inter_scan:
            return surv_lower[c]

    surv_uniques = sorted(
        ((c, survey[c].nunique(dropna=True)) for c in survey.columns),
        key=lambda t: -t[1],
    )
    purch_cols_lower = {c.lower() for c in purchases.columns}
    for cname, _ in surv_uniques:
        if cname.lower() in purch_cols_lower:
            return cname

    raise ValueError(
        "Could not auto-detect a common user ID column present in both files.\n"
        f"Purchases columns: {list(purchases.columns)}\n"
        f"Survey columns:    {list(survey.columns)}\n"
        "Pass user_id_col = '...' explicitly."
    )
