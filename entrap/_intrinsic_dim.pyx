# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
from sklearn.linear_model import LinearRegression

cnp.import_array()

cdef extern from "twonn.h":
    void entrap_compute_twonn_mu(
        const double *dist,
        size_t        N,
        double       *mu_out
    )


def estimate_intrinsic_dimension_twenn(
    cnp.ndarray X      not None,
    bint        X_is_dist = False
) -> float:
    cdef cnp.npy_intp N = X.shape[0]

    if N < 3:
        return 1.0

    if X_is_dist:
        dist = np.ascontiguousarray(X, dtype=np.float64)
    else:
        from scipy.spatial.distance import squareform, pdist
        dist = np.ascontiguousarray(
            squareform(pdist(X, metric='euclidean')),
            dtype=np.float64
        )

    cdef cnp.ndarray[cnp.float64_t, ndim=1] mu = np.empty(N, dtype=np.float64)

    entrap_compute_twonn_mu(
        <const double *>cnp.PyArray_DATA(dist),
        <size_t>N,
        <double *>cnp.PyArray_DATA(mu)
    )

    sort_idx          = np.argsort(mu)
    Femp              = np.arange(N, dtype=np.float64) / N
    log_mu            = np.log(mu[sort_idx] + 1e-12).reshape(-1, 1)
    neg_log_1_minus_F = -np.log(1.0 - Femp + 1e-12).reshape(-1, 1)

    lr = LinearRegression(fit_intercept=False)
    lr.fit(log_mu, neg_log_1_minus_F)

    d_hat   = float(lr.coef_[0][0])
    max_dim = float(X.shape[1] if not X_is_dist else N)
    return float(np.clip(d_hat, 1.0, max_dim))
