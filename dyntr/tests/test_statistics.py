import numpy as np

from dyntr.statistics import compute_delta, compute_delta_variance, compute_delta_lower_bound, opt_lambda


def _make_linear_data(rng, n, d, sigma):
    theta = rng.normal(size=d)
    X = rng.normal(size=(n, d))
    y = X @ theta + sigma * rng.normal(size=n)
    return X, y, theta


def test_compute_delta_and_variance_are_finite():
    """Compute-delta helpers should return finite values."""
    rng = np.random.default_rng(0)
    d = 3
    X_t, y_t, theta_t = _make_linear_data(rng, n=12, d=d, sigma=0.1)
    X_s, y_s, theta_s = _make_linear_data(rng, n=15, d=d, sigma=0.2)
    X_val = rng.normal(size=(8, d))

    lbda_t = 1.0
    lbda_s = 0.5
    lbda_c = lbda_t + lbda_s
    eye = np.eye(d)
    G_t = X_t.T @ X_t
    G_s = X_s.T @ X_s
    A_t_inv = np.linalg.inv(G_t + lbda_t * eye)
    A_s_inv = np.linalg.inv(G_s + lbda_s * eye)
    A_c_inv = np.linalg.inv(G_t + G_s + lbda_c * eye)

    delta = compute_delta(
        theta_t=theta_t,
        theta_s=theta_s,
        G_s=G_s,
        G_t=G_t,
        A_t_inv=A_t_inv,
        A_s_inv=A_s_inv,
        A_c_inv=A_c_inv,
        X_val=X_val,
        sigma_s=0.2,
        sigma_t=0.1,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
    )
    var = compute_delta_variance(
        theta_s=theta_s,
        theta_t=theta_t,
        G_s=G_s,
        G_t=G_t,
        A_t_inv=A_t_inv,
        A_s_inv=A_s_inv,
        A_c_inv=A_c_inv,
        X_val=X_val,
        sigma_s=0.2,
        sigma_t=0.1,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
    )
    delta_lb = compute_delta_lower_bound(
        theta_t=theta_t,
        theta_s=theta_s,
        G_t=G_t,
        G_s=G_s,
        A_t_inv=A_t_inv,
        A_s_inv=A_s_inv,
        A_c_inv=A_c_inv,
        X_val=X_val,
        sigma_t=0.1,
        sigma_s=0.2,
        lbda_t=lbda_t,
        lbda_s=lbda_s,
        alpha=0.1,
    )

    assert np.isfinite(delta)
    assert np.isfinite(var)
    assert np.isfinite(delta_lb)


def test_opt_lambda_returns_value_in_grid():
    """opt_lambda should return a value from the provided grid."""
    rng = np.random.default_rng(1)
    d = 4
    X, y, _ = _make_linear_data(rng, n=20, d=d, sigma=0.1)
    X_val = rng.normal(size=(10, d))
    y_val = X_val @ rng.normal(size=d) + 0.1 * rng.normal(size=10)

    grid = np.array([1e-3, 1e-2, 1e-1, 1.0])
    lbda = opt_lambda(X=X, y=y, X_val=X_val, y_val=y_val, lbda_grid=grid)

    assert lbda in grid
