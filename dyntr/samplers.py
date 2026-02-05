"""Samplers helpers module."""

# Authors: Hamza Cherkaoui

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import numba
from .utils import _as_generator, _center, _safe_cholesky, _row_norms_sq_numba, _svec_psd, _argmax_ignore_selected


def sample_uniform(
    X: np.ndarray,
    m: int,
    *,
    replace: bool = True,
    center: bool = True,
    mean: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform row sampling.

    Parameters
    ----------
    X : ndarray of shape (n, d)
    m : int
        Number of rows to sample.
    replace : bool, default=True
        Sample with replacement.
    center, mean, rng :
        Present for API consistency; not used.

    Returns
    -------
    indices : ndarray of shape (m,)
    weights : ndarray of shape (m,)
        All ones (use with the "importance" estimator).
    """
    n = int(np.asarray(X).shape[0])
    g = _as_generator(rng)

    idx = g.choice(n, size=int(m), replace=replace)
    w = np.ones((len(idx),), dtype=np.asarray(X).dtype)

    return idx, w


def sample_ridge_leverage(
    X: np.ndarray,
    m: int,
    *,
    lam: Optional[float] = None,
    lam_alpha: float = 1e-2,
    center: bool = True,
    mean: Optional[np.ndarray] = None,
    clip_p_max: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ridge leverage-score importance sampling.

    Parameters
    ----------
    X : ndarray of shape (n, d)
    m : int
        Number of draws (with replacement).
    lam : float, optional
        Ridge parameter for scores. If ``None``, uses
        ``lam_alpha * trace(Xc^T Xc) / n``.
    lam_alpha : float, default=1e-2
        Scale for the default ``lam``.
    center : bool, default=True
        If ``True``, center rows before scoring.
    mean : ndarray, optional
        Pre-computed mean for centering.
    clip_p_max : float, optional
        Cap per-row probability and renormalize.
    rng : Generator, optional

    Returns
    -------
    indices : ndarray of shape (m,)
    weights : ndarray of shape (m,)
        Importance weights ``1/sqrt(m p_i)``.
    """
    X = np.asarray(X)
    Xc, _ = _center(X, center=center, mean=mean)
    n, d = Xc.shape
    XtX = Xc.T @ Xc
    if lam is None:
        lam = float(lam_alpha * np.trace(XtX) / max(n, 1))

    L = _safe_cholesky(XtX + lam * np.eye(d, dtype=Xc.dtype))
    Z = np.linalg.solve(L, Xc.T)
    Z = np.linalg.solve(L.T, Z)
    lev = np.einsum("ij,ij->i", Xc, Z.T, optimize=True)

    lev = np.clip(lev, a_min=0.0, a_max=None)
    total = float(lev.sum())
    if total <= 0.0:
        return sample_uniform(X, m, center=center, mean=mean, rng=rng)

    p = lev / total
    if clip_p_max is not None:
        p = np.minimum(p, float(clip_p_max))
        p /= p.sum()

    g = _as_generator(rng)
    idx = g.choice(n, size=int(m), replace=True, p=p)
    w = 1.0 / np.sqrt(np.maximum(m * p[idx], 1e-32))

    return idx, w


@numba.njit(cache=True, fastmath=True)
def _sample_maxvol_greedy_core(Xc: np.ndarray, m: int) -> np.ndarray:
    n, d = Xc.shape
    m = min(m, n)

    base = _row_norms_sq_numba(Xc)
    selected_mask = np.zeros(n, dtype=np.uint8)
    selected_idx = np.full(m, -1, dtype=np.int64)

    Vs = np.zeros((m, d), dtype=Xc.dtype)
    dots = np.zeros((n, m), dtype=Xc.dtype)

    t = 0
    while t < m:

        res = np.empty(n, dtype=Xc.dtype)
        if t == 0:
            for i in range(n):
                res[i] = base[i]
        else:
            for i in range(n):
                s = 0.0
                for k in range(t):
                    v = dots[i, k]
                    s += v * v
                res[i] = base[i] - s

        i = _argmax_ignore_selected(res, selected_mask)
        if i == -1 or res[i] <= 0.0:
            break

        vi = np.empty(d, dtype=Xc.dtype)
        for j in range(d):
            vi[j] = Xc[i, j]
        for k in range(t):
            coeff = dots[i, k]
            for j in range(d):
                vi[j] -= coeff * Vs[k, j]

        norm2 = 0.0
        for j in range(d):
            norm2 += vi[j] * vi[j]
        if norm2 <= 0.0:
            selected_mask[i] = 1
            selected_idx[t] = i
            t += 1
            continue

        inv_norm = 1.0 / np.sqrt(norm2)
        for j in range(d):
            vi[j] *= inv_norm

        for j in range(d):
            Vs[t, j] = vi[j]

        for r in range(n):
            s = 0.0
            for j in range(d):
                s += Xc[r, j] * vi[j]
            dots[r, t] = s

        selected_mask[i] = 1
        selected_idx[t] = i
        t += 1

    if t < m:
        out = np.empty(t, dtype=np.int64)
        for k in range(t):
            out[k] = selected_idx[k]
        return out
    return selected_idx


def sample_maxvol_greedy(
    X: np.ndarray,
    m: int,
    *,
    center: bool = True,
    mean: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,  # kept for API symmetry; not used
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy max-volume subset (no replacement, unit weights).

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Design matrix (rows = samples).
    m : int
        Number of rows to select (``m <= n``).
    center : bool, default=True
        If True, subtract row-wise mean (or provided mean) before selection.
    mean : ndarray of shape (d,), optional
        Precomputed mean for centering. Used if ``center=True``.
    rng : Generator, optional
        Ignored. Selection is deterministic.

    Returns
    -------
    indices : ndarray of shape (m,)
    weights : ndarray of shape (m,)
        All ones (use with the "importance" estimator).
    """
    X = np.asarray(X)
    if center:
        mu = X.mean(axis=0) if mean is None else np.asarray(mean)
        Xc = X - mu
    else:
        Xc = X

    idx = _sample_maxvol_greedy_core(Xc.astype(np.float64, copy=False), int(m))
    w = np.ones((idx.shape[0],), dtype=X.dtype)
    return idx, w


def sample_weighted_sparsifier(
    X: np.ndarray,
    m: int,
    *,
    center: bool = True,
    mean: Optional[np.ndarray] = None,
    tau: float = 1e-8,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic weighted sparsifier (BSS-style proxy).

    Parameters
    ----------
    X : ndarray of shape (n, d)
    m : int
        Subset size.
    center : bool, default=True
    mean : ndarray, optional
    tau : float, default=1e-8
        Ridge penalty in the weight-fitting least-squares.
    rng : Generator, optional

    Returns
    -------
    indices : ndarray of shape (m,)
    weights : ndarray of shape (m,)
        Square-roots of nonnegative weights ``alpha`` fitted to match ``X^T X``
        in Frobenius sense.
    """
    X = np.asarray(X)
    Xc, _ = _center(X, center=center, mean=mean)
    _, d = Xc.shape

    idx, _ = sample_maxvol_greedy(X, m, center=center, mean=mean, rng=rng)
    XS = Xc[idx]

    b = _svec_psd(Xc.T @ Xc)
    p = b.shape[0]
    A = np.empty((p, len(idx)), dtype=Xc.dtype)
    for j in range(len(idx)):
        A[:, j] = _svec_psd(np.outer(XS[j], XS[j]))

    ATA = A.T @ A
    ATb = A.T @ b
    ATA.flat[:: len(idx) + 1] += float(tau)
    try:
        alpha = np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(ATA, ATb, rcond=None)[0]

    alpha = np.maximum(alpha, 0.0)
    Sigma_tilde = sum((alpha[j] * np.outer(XS[j], XS[j]) for j in range(len(idx))),
                      start=np.zeros((d, d), dtype=Xc.dtype))
    tr_target = float(np.trace(Xc.T @ Xc))
    tr_tilde = float(np.trace(Sigma_tilde))
    scale = 1.0 if tr_tilde <= 0 else tr_target / tr_tilde
    alpha *= scale

    w = np.sqrt(alpha, dtype=X.dtype)

    return idx, w
