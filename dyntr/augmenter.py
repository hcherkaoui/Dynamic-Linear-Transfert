from __future__ import annotations
from typing import Callable, Sequence, Tuple
import numba
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from .statistics import compute_delta_lower_bound
from .utils import _check_X, _check_Xy, std_estimation


N_MAX = 10000


def _permute(
    rng: np.random.RandomState,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    p = rng.permutation(X.shape[0])
    return X[p], y[p]


@numba.njit(cache=True, fastmath=True)
def _ridge_stats(
    X: np.ndarray,
    y: np.ndarray,
    lbda: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = X.shape[1]
    G = X.T @ X
    b = X.T @ y
    A_inv = np.linalg.inv(G + float(lbda) * np.eye(d))
    theta = A_inv @ b
    return G, b, A_inv, theta


def _prepare_sources(
    rng: np.random.RandomState,
    l_X: Sequence[ArrayLike],
    l_y: Sequence[ArrayLike],
    n_features: int,
) -> Tuple[list[np.ndarray], list[np.ndarray]]:
    if len(l_X) != len(l_y):
        raise ValueError("l_X and l_y must have the same length.")
    Xs = []
    ys = []
    for X, y in zip(l_X, l_y):
        X = _check_X(np.asarray(X), n_features=n_features)
        y = np.asarray(y).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("A source dataset has X and y with mismatched lengths.")
        X, y = _permute(rng, X, y)
        Xs.append(X)
        ys.append(y)
    return Xs, ys


def _estimate_sigma(
    X: np.ndarray,
    y: np.ndarray,
    lbda: float,
) -> float:
    if X.shape[0] <= 1:
        return 0.0
    G, _, _, theta = _ridge_stats(X, y, lbda)
    return float(std_estimation(theta, X, y, mode=1, lbda=float(lbda), G=G, return_variance=False))


@numba.njit(cache=True, fastmath=True)
def _woodbury_update(A_inv: np.ndarray, X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return A_inv
    mid = np.eye(X.shape[0]) + X @ A_inv @ X.T
    K = np.linalg.inv(mid)
    return A_inv - A_inv @ X.T @ K @ X @ A_inv


@numba.njit(cache=True, fastmath=True)
def _oracle_score_update_batch(
    Kc: np.ndarray,
    b: np.ndarray,
    X_b: np.ndarray,
    y_b: np.ndarray,
    X_t_val: np.ndarray,
    theta_t_true: np.ndarray,
    base_err: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    Kc = _woodbury_update(Kc, X_b)
    b = b + X_b.T @ y_b
    theta = Kc @ b
    v = X_t_val @ (theta - theta_t_true)
    delta = base_err - float(v @ v)
    return delta, Kc, b


@numba.njit(cache=True, fastmath=True)
def _sample_score_update_batch(
    G_s: np.ndarray,
    b_s: np.ndarray,
    A_s_inv: np.ndarray,
    A_c_inv: np.ndarray,
    X_b: np.ndarray,
    y_b: np.ndarray,
    theta_t: np.ndarray,
    G_t: np.ndarray,
    A_t_inv: np.ndarray,
    X_t_val: np.ndarray,
    sigma_t: float,
    sigma_s: float,
    lbda_t: float,
    lbda_s: float,
    alpha: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    G_s = G_s + X_b.T @ X_b
    b_s = b_s + X_b.T @ y_b
    A_s_inv = _woodbury_update(A_s_inv, X_b)
    A_c_inv = _woodbury_update(A_c_inv, X_b)
    theta_s = A_s_inv @ b_s
    delta = compute_delta_lower_bound(
        theta_t,
        theta_s,
        G_t,
        G_s,
        A_t_inv,
        A_s_inv,
        A_c_inv,
        X_t_val,
        sigma_t,
        sigma_s,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        alpha=alpha,
    )
    return float(delta), G_s, b_s, A_s_inv, A_c_inv


def select_multi_source(
    rng: np.random.RandomState,
    strategy: str,
    Xs: list[np.ndarray],
    ys: list[np.ndarray],
    n_rounds: int,
    score_and_update_batch: Callable[[object, np.ndarray, np.ndarray], Tuple[float, object]],
    init_state: object,
    chunk_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_rounds <= 0:
        return (
            np.empty((0,), float),
            np.empty((0,), int),
            np.empty((0,)),
            np.empty((0,)),
            np.empty((0,), int),
        )
    if strategy not in {"greedy", "uniform"}:
        raise ValueError("strategy must be in {'greedy','uniform'}.")

    n_src = len(Xs)
    rem = [np.arange(X.shape[0]) for X in Xs]
    deltas = []
    sel_src = []
    X_sel = []
    y_sel = []
    chunk_sizes = []
    state = init_state
    n_selected = 0

    def active_sources() -> list[int]:
        return [s for s in range(n_src) if rem[s].size > 0]

    while n_selected < n_rounds:
        act = active_sources()
        if not act:
            break

        if strategy == "uniform":
            best_s = int(rng.choice(act))
            best_state = None
            best_delta = None
            best_chunk_idx = None
        else:
            best_delta = -np.inf
            best_state = None
            best_s = -1
            best_chunk_idx = None
            for s in act:
                r = rem[s]
                if r.size <= 0:
                    continue
                size = min(int(chunk_size), int(r.size), int(n_rounds - n_selected))
                idx = r[:size]
                X_b = Xs[s][idx]
                y_b = ys[s][idx]
                d_new, st_new = score_and_update_batch(state, X_b, y_b)
                if d_new > best_delta:
                    best_delta = float(d_new)
                    best_state = st_new
                    best_s = int(s)
                    best_chunk_idx = np.array(idx, copy=True)

            if best_state is None:
                break

        r = rem[best_s]
        size = min(int(chunk_size), int(r.size), int(n_rounds - n_selected))
        if size <= 0:
            break

        if strategy == "uniform":
            best_chunk_idx = r[:size]
            X_b = Xs[best_s][best_chunk_idx]
            y_b = ys[best_s][best_chunk_idx]
            best_delta, best_state = score_and_update_batch(state, X_b, y_b)
            best_delta = float(best_delta)
        else:
            if best_chunk_idx is None:
                break
            if best_chunk_idx.size != size:
                remaining = r[~np.isin(r, best_chunk_idx)]
                extra = remaining[: size - best_chunk_idx.size]
                best_chunk_idx = np.concatenate([best_chunk_idx, extra])
                X_b = Xs[best_s][best_chunk_idx]
                y_b = ys[best_s][best_chunk_idx]
                best_delta, best_state = score_and_update_batch(state, X_b, y_b)
                best_delta = float(best_delta)

        deltas.append(float(best_delta))
        sel_src.append(int(best_s))
        X_sel.append(Xs[best_s][best_chunk_idx])
        y_sel.append(ys[best_s][best_chunk_idx])
        chunk_sizes.append(int(best_chunk_idx.size))
        state = best_state
        rem[best_s] = r[~np.isin(r, best_chunk_idx)]
        n_selected += int(best_chunk_idx.size)

    X_seq = np.vstack(X_sel) if X_sel else np.empty((0, Xs[0].shape[1] if Xs else 0))
    y_seq = np.concatenate(y_sel) if y_sel else np.empty((0,))
    return (
        np.asarray(deltas, dtype=float),
        np.asarray(sel_src, dtype=int),
        X_seq,
        y_seq,
        np.asarray(chunk_sizes, dtype=int),
    )


class _AugmenterBase(BaseEstimator):
    def __init__(self, n_max=N_MAX, strategy="greedy", chunk_size=1, random_state=None):
        self.n_max = n_max
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.random_state = random_state

    def _validate(self) -> None:
        if int(self.n_max) <= 0:
            raise ValueError("n_max must be positive.")
        if int(self.chunk_size) <= 0:
            raise ValueError("chunk_size must be positive.")
        if str(self.strategy) not in {"greedy", "uniform"}:
            raise ValueError("strategy must be in {'greedy','uniform'}.")

    def get_n_selected(self) -> int:
        check_is_fitted(self, "n_selected_")
        return int(self.n_selected_)


class _BaseSampleAugmenter(_AugmenterBase):
    def _run_selection(
        self,
        Xs: list[np.ndarray],
        ys: list[np.ndarray],
        n_rounds: int,
        score_and_update_batch: Callable[[object, np.ndarray, np.ndarray], Tuple[float, object]],
        init_state: object,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return select_multi_source(
            rng=self._rng_,
            strategy=str(self.strategy),
            Xs=Xs,
            ys=ys,
            n_rounds=n_rounds,
            score_and_update_batch=score_and_update_batch,
            init_state=init_state,
            chunk_size=int(self.chunk_size),
        )

    def _finalize_selection(
        self,
        deltas: np.ndarray,
        sel_src: np.ndarray,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        chunk_sizes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.deltas_ = deltas
        self.selected_sources_ = sel_src

        if deltas.size:
            k_chunks = int(np.argmax(deltas) + 1)
            cum = np.cumsum(chunk_sizes)
            n_selected = int(cum[k_chunks - 1]) if k_chunks > 0 else 0
        else:
            k_chunks = 0
            n_selected = 0

        self.n_selected_ = n_selected
        self.max_delta_ = float(deltas[k_chunks - 1]) if k_chunks > 0 else -np.inf

        if self.max_delta_ < 0.0:
            return self.X_t_.copy(), self.y_t_.copy()

        Xc = np.concatenate([self.X_t_, X_seq[:n_selected]], axis=0)
        yc = np.concatenate([self.y_t_, y_seq[:n_selected]], axis=0)
        return _permute(self._rng_, Xc, yc)


class OracleSampleAugmenter(_BaseSampleAugmenter):
    """Multi-source augmentation using oracle Δ_emp (requires θ_t^*)."""

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_t_val: ArrayLike,
        lbda: float,
        theta_t_true: ArrayLike,
    ) -> "OracleSampleAugmenter":
        X_t, y_t = _check_Xy(X, y)
        d = X_t.shape[1]
        X_t_val = _check_X(X_t_val, n_features=d)

        self._validate()

        theta_t_true = np.asarray(theta_t_true, float).reshape(-1)
        if theta_t_true.shape[0] != d:
            raise ValueError("theta_t_true has incompatible dimension with X.")

        G_t, b_t, _, _ = _ridge_stats(X_t, y_t, float(lbda))
        theta_t_hat = np.linalg.inv(G_t + float(lbda) * np.eye(d)) @ b_t

        self._rng_ = check_random_state(self.random_state)
        self.n_features_ = d
        self.lbda_t_ = float(lbda)
        self.X_t_ = X_t
        self.y_t_ = y_t
        self.X_t_val_ = X_t_val
        self.G_t_ = G_t
        self.b_t_ = b_t
        self.theta_t_hat_ = theta_t_hat
        self.theta_t_true_ = theta_t_true

        return self

    def transform(
        self,
        l_X: Sequence[ArrayLike],
        l_y: Sequence[ArrayLike],
        lbda: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ("X_t_", "G_t_", "b_t_", "theta_t_true_", "_rng_"))
        Xs, ys = _prepare_sources(self._rng_, l_X, l_y, self.n_features_)
        total = sum(X.shape[0] for X in Xs)
        n_rounds = min(int(self.n_max), int(total))
        if n_rounds <= 0:
            raise ValueError("No source samples available.")

        eye = np.eye(self.n_features_)
        Kc0 = np.linalg.inv(self.G_t_ + float(lbda) * eye)
        b0 = self.b_t_.copy()
        v0 = self.X_t_val_ @ (self.theta_t_hat_ - self.theta_t_true_)
        base_err = float(v0 @ v0)

        def score_and_update_batch(state, X_b: np.ndarray, y_b: np.ndarray):
            Kc, b = state
            delta, Kc, b = _oracle_score_update_batch(
                Kc,
                b,
                X_b,
                y_b,
                self.X_t_val_,
                self.theta_t_true_,
                base_err,
            )
            return float(delta), (Kc, b)

        deltas, sel_src, X_seq, y_seq, chunk_sizes = self._run_selection(
            Xs=Xs,
            ys=ys,
            n_rounds=n_rounds,
            score_and_update_batch=score_and_update_batch,
            init_state=(Kc0, b0),
        )
        return self._finalize_selection(deltas, sel_src, X_seq, y_seq, chunk_sizes)


class SampleAugmenter(_BaseSampleAugmenter):
    """Multi-source augmentation using Δ lower bound maximization."""

    def __init__(self, n_max=N_MAX, alpha=0.05, strategy="greedy", chunk_size=5, random_state=None):
        super().__init__(
            n_max=n_max,
            strategy=strategy,
            chunk_size=chunk_size,
            random_state=random_state,
        )
        self.alpha = alpha

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_t_val: ArrayLike,
        lbda: float,
        sigma: float | None = None,
    ) -> "SampleAugmenter":
        X_t, y_t = _check_Xy(X, y)
        d = X_t.shape[1]
        X_t_val = _check_X(X_t_val, n_features=d)

        self._validate()

        G_t, b_t, A_t_inv, theta_t = _ridge_stats(X_t, y_t, float(lbda))
        sigma_t = float(sigma) if sigma is not None else _estimate_sigma(X_t, y_t, float(lbda))

        self._rng_ = check_random_state(self.random_state)
        self.n_features_ = d
        self.lbda_t_ = float(lbda)
        self.alpha_ = float(self.alpha)
        self.sigma_t_ = float(sigma_t)
        self.X_t_ = X_t
        self.y_t_ = y_t
        self.X_t_val_ = X_t_val
        self.G_t_ = G_t
        self.b_t_ = b_t
        self.A_t_inv_ = A_t_inv
        self.theta_t_ = theta_t

        return self

    def transform(
        self,
        l_X: Sequence[ArrayLike],
        l_y: Sequence[ArrayLike],
        lbda: float,
        sigma: float | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ("X_t_", "G_t_", "b_t_", "A_t_inv_", "theta_t_", "_rng_"))
        Xs, ys = _prepare_sources(self._rng_, l_X, l_y, self.n_features_)
        total = sum(X.shape[0] for X in Xs)
        n_rounds = min(int(self.n_max), int(total))
        if n_rounds <= 0:
            raise ValueError("No source samples available.")

        sigma_s = float(sigma) if sigma is not None else 0.0

        eye = np.eye(self.n_features_)
        G_s0 = np.zeros((self.n_features_, self.n_features_), dtype=float)
        b_s0 = np.zeros((self.n_features_,), dtype=float)
        A_s_inv0 = np.linalg.inv(G_s0 + float(lbda) * eye)
        A_c_inv0 = np.linalg.inv(self.G_t_ + float(lbda + self.lbda_t_) * eye)

        def score_and_update_batch(state, X_b: np.ndarray, y_b: np.ndarray):
            G_s, b_s, A_s_inv, A_c_inv = state
            delta, G_s, b_s, A_s_inv, A_c_inv = _sample_score_update_batch(
                G_s,
                b_s,
                A_s_inv,
                A_c_inv,
                X_b,
                y_b,
                self.theta_t_,
                self.G_t_,
                self.A_t_inv_,
                self.X_t_val_,
                self.sigma_t_,
                sigma_s,
                self.lbda_t_,
                float(lbda),
                self.alpha_,
            )
            return float(delta), (G_s, b_s, A_s_inv, A_c_inv)

        deltas, sel_src, X_seq, y_seq, chunk_sizes = self._run_selection(
            Xs=Xs,
            ys=ys,
            n_rounds=n_rounds,
            score_and_update_batch=score_and_update_batch,
            init_state=(G_s0, b_s0, A_s_inv0, A_c_inv0),
        )
        return self._finalize_selection(deltas, sel_src, X_seq, y_seq, chunk_sizes)
