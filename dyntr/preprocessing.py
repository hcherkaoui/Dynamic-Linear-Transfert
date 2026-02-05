"""Preprocessing module."""

# Authors: Hamza Cherkaoui

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_array as sk_check_array
from .statistics import compute_delta_lower_bound
from .utils import std_estimation


N_MAX = 1000


def _to_1d_float(y: ArrayLike) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return y.ravel()


def _check_xy(X: ArrayLike, y: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    X = sk_check_array(X, ensure_2d=True, dtype=float, accept_sparse=False)
    y = _to_1d_float(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y have inconsistent shapes: {X.shape[0]} != {y.shape[0]}")
    return X, y


@dataclass
class OracleGreedySampleAugmenter(BaseEstimator, TransformerMixin):
    """
    Oracle greedy sample augmenter (Δ_emp argmin with oracle θ₁*).

    This estimator receives θ₁* at fit time and scans dataset source at transform
    time to minimize the empirical delta
        Δ_emp(n) = ||θ̂_c(n) - θ₁*||² - ||θ̂₁ - θ₁*||²
    over prefix sizes n = 1..n_max. It updates the collaborative ridge solution
    via Sherman-Morrison. If min Δ_emp > 0, it returns (X₁, y₁); otherwise it
    concatenates the first n_selected rows of (X₂, y₂) to (X₁, y₁).

    Parameters
    ----------
    n_max : int
        Maximum number of candidate rows from X₂ to consider (prefix length).
    random_state : int or numpy.random.RandomState, optional
        Controls the permutation of X₂ before the greedy pass.

    Attributes
    ----------
    lbda_t, lbda_c : float
        Ridge regularization strength used for both dataset target and the
        collaborative model.
    n_features_ : int
        Number of features.
    n_samples_ : int
        Number of samples in X_t.
    X_t_ : ndarray of shape (n_t, d)
        Stored X₁.
    y_t_ : ndarray of shape (n_t,)
        Stored y₁.
    X_t_val_ : array-like of shape (n_t, d)
        Validation features matrix for dataset target.
    G_t_ : ndarray of shape (d, d)
        X₁ᵀX₁.
    b_t_ : ndarray of shape (d,)
        X₁ᵀy₁.
    A_t_inv_ : ndarray of shape (d, d)
        (X_tᵀX_t + λI)⁻¹.
    theta_t_hat_ : ndarray of shape (d,)
        Ridge estimator on dataset target, A_t_inv_ @ b_t_.
    theta_t_true_ : ndarray of shape (d,)
        Oracle parameter θ₁*.
    n_selected_ : int
        Number of rows selected from the permuted X₂.
    min_delta_emp_ : float
        Minimum empirical delta achieved.
    deltas_emp_ : ndarray of shape (n_max,)
        Trajectory of Δ_emp across prefix sizes.
    permutation_ : ndarray of shape (n2,)
        Permutation applied to X₂, y₂ in the last transform.
    indices_selected_ : ndarray of shape (n_selected_,)
        Indices [0..n_selected_-1] within the permuted X₂.
    """

    n_max: int = N_MAX
    random_state: Optional[int] = None

    # Fitted attrs init (set in fit or tranform)
    lbda_t: float = None
    lbda_c: float = None
    n_features_: int = 0
    n_samples_: int = 0
    X_t_: Optional[np.ndarray] = None
    y_t_: Optional[np.ndarray] = None
    X_t_val_: Optional[np.ndarray] = None
    G_t_: Optional[np.ndarray] = None
    b_t_: Optional[np.ndarray] = None
    A_t_inv_: Optional[np.ndarray] = None
    theta_t_hat_: Optional[np.ndarray] = None
    theta_t_true_: Optional[np.ndarray] = None
    n_selected_: int = 0
    min_delta_emp_: float = 0.0
    deltas_emp_: Optional[np.ndarray] = None
    permutation_: Optional[np.ndarray] = None
    indices_selected_: Optional[np.ndarray] = None

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_t_val: ArrayLike,
        lbda: float,
        theta_t_true: ArrayLike,
    ) -> "OracleGreedySampleAugmenter":
        """
        Fit on (X_t, y_t) with oracle θ_t*.

        Parameters
        ----------
        X : array-like of shape (n_t, d)
            Design matrix of dataset target.
        y : array-like of shape (n_t,)
            Targets of dataset target.
        X_t_val : array-like of shape (n_t, d)
            Validation features matrix for dataset target.
        lbda : float
            Ridge regularization strength used for the target model.
        theta_t_true : array-like of shape (d,)
            Oracle parameter θ_t*.

        Returns
        -------
        self : OracleGreedySampleAugmenter
            Fitted estimator.
        """
        X_t, y_t = _check_xy(X, y)
        theta_t_true = np.asarray(theta_t_true, dtype=float).ravel()
        self.lbda_t = lbda

        if self.n_max <= 0:
            raise ValueError("n_max must be positive.")
        if theta_t_true.shape[0] != X_t.shape[1]:
            raise ValueError("theta_t_true has incompatible dimension with X_t.")

        n_t, d = X_t.shape
        eye = np.eye(d)

        G_t = X_t.T @ X_t
        b_t = X_t.T @ y_t
        A_t_inv = np.linalg.inv(G_t + self.lbda_t * eye)
        theta_t_hat = A_t_inv @ b_t

        self.n_features_ = d
        self.n_samples_ = n_t
        self.X_t_ = X_t
        self.y_t_ = y_t
        self.X_t_val_ = X_t_val
        self.G_t_ = G_t
        self.b_t_ = b_t
        self.A_t_inv_ = A_t_inv
        self.theta_t_hat_ = theta_t_hat
        self.theta_t_true_ = theta_t_true
        return self

    def transform(
        self,
        X: ArrayLike,
        y: ArrayLike,
        lbda: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run greedy Δ_emp minimization on (X₂, y₂) and return the augmented dataset.

        Parameters
        ----------
        X : array-like of shape (n2, d)
            Candidate design matrix (dataset source).
        y : array-like of shape (n2,)
            Candidate targets.
        lbda : float
            Ridge regularization strength used for the collaborative model.

        Returns
        -------
        Xc : ndarray of shape (n1 + n_selected_, d)
            Augmented design matrix (or X₁ if no augmentation improves Δ_emp).
        yc : ndarray of shape (n1 + n_selected_,)
            Augmented targets (or y₁ if no augmentation improves Δ_emp).
        """
        check_is_fitted(self, ("X_t_", "A_t_inv_", "theta_t_true_", "theta_t_hat_"))

        X_s, y_s = _check_xy(X, y)
        if X_s.shape[1] != self.n_features_:
            raise ValueError(f"Incompatible feature dimension: got {X_s.shape[1]}, expected {self.n_features_}.")

        self.lbda_c = lbda

        rng = check_random_state(self.random_state)
        perm = rng.permutation(X_s.shape[0])
        X_s_p = X_s[perm]
        y_s_p = y_s[perm]
        self.permutation_ = perm

        n2 = X_s_p.shape[0]
        n_max = min(int(self.n_max), n2)
        if n_max <= 0:
            self.n_selected_ = 0
            self.min_delta_emp_ = 0.0
            self.deltas_emp_ = np.empty(0, dtype=float)
            self.indices_selected_ = np.empty(0, dtype=int)
            return self.X_t_.copy(), self.y_t_.copy()

        G_t = self.G_t_
        b_t = self.b_t_
        theta_t_star = self.theta_t_true_
        theta_t_hat = self.theta_t_hat_
        d = self.n_features_
        eye = np.eye(d)

        Kc = np.linalg.inv(G_t + float(self.lbda_c) * eye)
        theta_c = Kc @ b_t

        deltas = np.empty(n_max, dtype=float)

        for i in range(n_max):
            x = X_s_p[i]
            y_i = y_s_p[i]

            Kcx = Kc @ x
            kappac = 1.0 + float(x @ Kcx)
            residual = float(y_i - x @ theta_c)
            theta_c = theta_c + (residual / kappac) * Kcx
            Kc = Kc - np.outer(Kcx, Kcx) / kappac

            deltas[i] = float(
                np.dot(self.X_t_val_ @ (theta_t_hat - theta_t_star), self.X_t_val_ @ (theta_t_hat - theta_t_star))  # noqa
                - np.dot(self.X_t_val_ @ (theta_c - theta_t_star), self.X_t_val_ @ (theta_c - theta_t_star))  # noqa
            )

        idx_max = int(np.argmax(deltas))
        n_selected = idx_max + 1
        max_delta = float(deltas[idx_max])

        self.n_selected_ = n_selected if max_delta > 0.0 else 0
        self.max_delta_emp_ = max_delta
        self.deltas_emp_ = deltas
        self.indices_selected_ = np.arange(self.n_selected_, dtype=int)

        if self.n_selected_ == 0:
            return self.X_t_.copy(), self.y_t_.copy()

        Xc = np.concatenate([self.X_t_, X_s_p[: self.n_selected_]], axis=0)
        yc = np.concatenate([self.y_t_, y_s_p[: self.n_selected_]], axis=0)

        p = rng.permutation(Xc.shape[0])
        return Xc[p], yc[p]

    def get_delta(
        self,
    ) -> int:
        """
        Return the estimated maximum transfer gain.

        Returns
        -------
        float
            Estimated maximum transfer gain.
        """
        check_is_fitted(self, "max_delta_emp_")
        return float(self.max_delta_emp_)

    def get_deltas(
        self,
    ) -> int:
        """
        Return the estimated transfer gains.

        Returns
        -------
        float
            Estimated transfer gains.
        """
        check_is_fitted(self, "deltas_emp_")
        return np.array(self.deltas_emp_)

    def get_n_selected(
        self,
    ) -> int:
        """
        Return the number of rows from X₂ selected in the last transform.

        Returns
        -------
        int
            Number of selected rows (0 if none).
        """
        check_is_fitted(self, "n_selected_")
        return int(self.n_selected_)


@dataclass
class GreedySampleAugmenter(BaseEstimator, TransformerMixin):
    """
    Greedy sample augmentation via Δ upper bound minimization.

    This transformer selects a prefix of rows from a candidate dataset (X_s, y_s)
    to augment a base dataset (X_t, y_t). It iteratively updates the collaborative
    ridge inverse via Sherman-Morrison and evaluates a user-provided upper bound
    Δ̂_U(n) = delta_upper_bound_estimation(...) on the collaborative risk delta.
    It returns (Xc, yc) equal to either (X_t, y_t) if the minimum upper bound is
    positive, or concatenation of (X_t, y_t) with the first n2_best rows of
    (X_s, y_s) that minimize the bound.

    Parameters
    ----------
    n_max : int
        Maximum number of candidate rows from X_s to consider.
    alpha : float, default=0.01
        Extra tuning parameter relayed to `delta_upper_bound_estimation`:
        LCB parameter
    random_state : int or RandomState, optional
        Controls shuffling before processing. Use to ensure reproducibility.

    Attributes
    ----------
    lbda_t, lbda_c : float
        Ridge regularization strength used for both dataset target and the
        collaborative model.
    n_features_ : int
        Number of features (d).
    n_samples_ : int
        Number of rows in X_t used during `fit`.
    I_ : ndarray of shape (d, d)
        Identity matrix.
    X_t_ : ndarray of shape (n1, d)
        Design matrix for dataset target.
    y_t_ : ndarray of shape (n1,)
        Target vector for dataset target.
    X_t_val_ : array-like of shape (n1, d)
        Validation features matrix for dataset target.
    G_t_ : ndarray of shape (d, d)
        Gram matrix X_t_.T @ X_t_.
    b_t_ : ndarray of shape (d,)
        Vector X_t_.T @ y_t_.
    A_t_inv_ : ndarray of shape (d, d)
        Inverse of the ridge-regularized Gram for dataset target.
    theta_t_ : ndarray of shape (d,)
        Ridge solution on dataset target.
    sigma_t_ : float
        Noise standard deviation for dataset target (estimated if not provided in `fit`).
    """

    n_max: int = N_MAX
    alpha: float = 0.1
    random_state: Optional[int] = None

    # Fitted attrs init (set in fit or tranform)
    lbda_t: float = None
    lbda_s: float = None
    lbda_c: float = None
    n_features_: int = 0
    n_samples_: int = 0
    I_: Optional[np.ndarray] = None
    X_t_: Optional[np.ndarray] = None
    y_t_: Optional[np.ndarray] = None
    X_t_val_: Optional[np.ndarray] = None
    G_t_: Optional[np.ndarray] = None
    b_t_: Optional[np.ndarray] = None
    A_t_inv_: Optional[np.ndarray] = None
    theta_t_: Optional[np.ndarray] = None
    sigma_t_: Optional[float] = None

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_t_val: ArrayLike,
        lbda: float,
        sigma: Optional[float] = None,
    ) -> "GreedySampleAugmenter":
        """
        Fit on the base dataset (X_t, y_t).

        Parameters
        ----------
        X : array-like of shape (n_t, d)
            Training features matrix for dataset target.
        y : array-like of shape (n_t,)
            Targets for dataset target.
        X_t_val : array-like of shape (n_t, d)
            Validation features matrix for dataset target.
        lbda : float
            Ridge regularization strength used for the target model.
        sigma : float, optional
            Noise standard deviation for dataset target. If None, estimated via `std_estimation`.

        Returns
        -------
        self : DataAugmentation
            Fitted estimator.
        """
        X, y = _check_xy(X, y)

        self.lbda_t = lbda

        if self.n_max <= 0:
            raise ValueError("n_max must be positive.")

        n_t, d = X.shape
        eye = np.eye(d)

        G_t = X.T @ X
        b_t = X.T @ y
        A_t_inv = np.linalg.inv(G_t + self.lbda_t * eye)
        theta_t = A_t_inv @ b_t

        if sigma is None:  # should be avoid if possible
            sigma_t = float(
                std_estimation(theta_t, X, y, mode=1, lbda=self.lbda_t, G=G_t, return_variance=False)  # type: ignore[name-defined]
            )
        else:
            sigma_t = float(sigma)

        self.n_features_ = d
        self.n_samples_ = n_t
        self.I_ = eye
        self.X_t_ = X
        self.y_t_ = y
        self.X_t_val_ = X_t_val
        self.G_t_ = G_t
        self.b_t_ = b_t
        self.A_t_inv_ = A_t_inv
        self.theta_t_ = theta_t
        self.sigma_t_ = sigma_t
        return self

    def transform(
        self,
        X: ArrayLike,
        y: ArrayLike,
        lbda: float,
        sigma: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select a prefix of (X_s, y_s) to augment (X_t, y_t) by maximizing the Δ upper bound.

        Parameters
        ----------
        X : array-like of shape (n_s, d)
            Candidate design matrix to augment dataset target.
        y : array-like of shape (n_s,)
            Candidate targets.
        lbda : float
            Ridge regularization strength used for the collaborative model.
        sigma : float, optional
            Noise standard deviation for dataset source. If None, estimated on full (X, y).

        Returns
        -------
        Xc : ndarray of shape (n_t + k, d)
            Augmented design matrix. If the minimum upper bound is positive, k=0.
        yc : ndarray of shape (n_t + k,)
            Augmented target vector.
        """
        check_is_fitted(self, ("X_t_", "A_t_inv_", "sigma_t_"))
        X_s, y_s = _check_xy(X, y)

        lbda_s = lbda
        lbda_c = self.lbda_t + lbda_s

        d_expected = self.n_features_
        if X_s.shape[1] != d_expected:
            raise ValueError(f"Incompatible feature dimension: got {X_s.shape[1]}, expected {d_expected}.")

        rng = check_random_state(self.random_state)
        p_s = rng.permutation(X_s.shape[0])
        X_s_p = X_s[p_s]
        y_s_p = y_s[p_s]

        n_s = X_s_p.shape[0]
        n_max = int(min(self.n_max, n_s))
        if n_max <= 0:
            return self.X_t_.copy(), self.y_t_.copy()  # type: ignore[return-value]

        if sigma is None:  # should be avoid if possible
            G_s_full = X_s_p.T @ X_s_p
            b_s_full = X_s_p.T @ y_s_p
            A_s_inv_full = np.linalg.inv(G_s_full + lbda_s * self.I_)  # type: ignore[arg-type]
            theta_s_full = A_s_inv_full @ b_s_full
            sigma_s = float(
                std_estimation(theta_s_full, X_s_p, y_s_p, mode=1, lbda=lbda_s, G=G_s_full, return_variance=False)  # type: ignore[name-defined]
            )
        else:
            sigma_s = float(sigma)

        A_t_inv = self.A_t_inv_  # type: ignore[assignment]
        G_t = self.G_t_          # type: ignore[assignment]
        eye = self.I_            # type: ignore[assignment]
        theta_t = self.theta_t_  # type: ignore[assignment]
        sigma_t = self.sigma_t_  # type: ignore[assignment]

        A_c_inv = np.linalg.inv(G_t + lbda_c * eye)
        G_s = np.zeros((d_expected, d_expected))
        A_s_inv = np.linalg.inv(G_s + lbda_s * eye)
        b_s = np.zeros((d_expected,))

        deltas = np.empty(n_max, dtype=float)

        for i in range(n_max):
            x = X_s_p[i]
            y_i = y_s_p[i]

            G_s += np.outer(x, x)
            b_s += x * y_i

            # sherman morrison on A2_inv
            K_s_x = A_s_inv @ x
            kappas = 1.0 + float(x @ K_s_x)
            A_s_inv -= np.outer(K_s_x, K_s_x) / kappas

            # sherman morrison on Ac_inv
            Kcx = A_c_inv @ x
            kappac = 1.0 + float(x @ Kcx)
            A_c_inv -= np.outer(Kcx, Kcx) / kappac

            theta_s = A_s_inv @ b_s

            delta_lb = float(
                compute_delta_lower_bound(  # type: ignore[name-defined]
                    theta_t,
                    theta_s,
                    G_t,
                    G_s,
                    A_t_inv,
                    A_s_inv,
                    A_c_inv,
                    self.X_t_val_,
                    sigma_t,
                    sigma_s,
                    lbda_t=self.lbda_t,
                    lbda_s=lbda_s,
                    alpha=self.alpha,
                )
            )
            deltas[i] = delta_lb

        n_selected_best = int(np.argmax(deltas) + 1)
        max_delta = float(deltas[n_selected_best - 1])

        self.n_selected_ = n_selected_best
        self.max_delta_ = max_delta
        self.deltas_ = deltas.copy()

        if max_delta < 0.0:
            Xc = self.X_t_.copy()
            yc = self.y_t_.copy()
            return Xc, yc

        Xc = np.concatenate([self.X_t_, X_s_p[:n_selected_best]], axis=0)
        yc = np.concatenate([self.y_t_, y_s_p[:n_selected_best]], axis=0)

        p = rng.permutation(Xc.shape[0])
        return Xc[p], yc[p]

    def get_n_selected(
        self,
    ) -> int:
        """
        Return the number of rows from X2 selected during the last transform.

        Returns
        -------
        int
            Number of candidate rows used (0 if none).
        """
        check_is_fitted(self, "n_selected_")
        return self.n_selected_

    def get_delta(
        self,
    ) -> int:
        """
        Return the estimated maximum transfer gain.

        Returns
        -------
        float
            Estimated maximum transfer gain.
        """
        check_is_fitted(self, "max_delta_")
        return float(self.max_delta_)

    def get_deltas(
        self,
    ) -> int:
        """
        Return the estimated transfer gains.

        Returns
        -------
        float
            Estimated transfer gains.
        """
        check_is_fitted(self, "deltas_")
        return np.array(self.deltas_)
