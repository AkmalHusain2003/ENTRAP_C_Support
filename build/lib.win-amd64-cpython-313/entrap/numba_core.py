try:
    from entrap._numba_core import (
        compute_cov_from_rows,
        compute_cluster_mean,
        compute_cluster_covariance,
        compute_mahalanobis_sq,
        logistic_mapping,
        logistic_mapping_batch,
    )

except ImportError:
    import numpy as np

    def compute_cov_from_rows(neighbor_distances: np.ndarray) -> np.ndarray:
        nd = np.asarray(neighbor_distances, dtype=np.float64)
        if nd.shape[1] == 0:
            return np.full(nd.shape[0], np.inf, dtype=np.float64)

        means = nd.mean(axis=1)
        stds  = nd.std(axis=1)

        mask       = means > 1e-12
        safe_means = np.where(mask, means, 1.0)

        return np.where(mask, stds / safe_means, np.inf).astype(np.float64)

    def compute_cluster_mean(cluster_points: np.ndarray) -> np.ndarray:
        cp = np.asarray(cluster_points, dtype=np.float64)
        return cp.mean(axis=0)

    def compute_cluster_covariance(
        cluster_points: np.ndarray,
        mean:           np.ndarray,
        ridge_epsilon:  float
    ) -> np.ndarray:
        cp   = np.asarray(cluster_points, dtype=np.float64)
        mu   = np.asarray(mean,           dtype=np.float64)
        n, d = cp.shape

        X_c = cp - mu
        cov = np.dot(X_c.T, X_c) / n

        diag_idx = np.arange(d)
        cov[diag_idx, diag_idx] += ridge_epsilon

        return cov

    def compute_mahalanobis_sq(
        diff:      np.ndarray,
        Sigma_inv: np.ndarray
    ) -> float:
        d  = np.asarray(diff,      dtype=np.float64)
        Si = np.asarray(Sigma_inv, dtype=np.float64)
        return float(d @ Si @ d)

    def logistic_mapping(
        cov_value: float,
        cov_10:    float,
        cov_50:    float,
        cov_90:    float,
        q_min:     float,
        q_max:     float,
        alpha:     float = 10.0
    ) -> float:
        delta     = 1e-12
        a         = alpha / (cov_90 - cov_10 + delta)
        exponent  = -a * (cov_value - cov_50)
        sigmoid   = 1.0 / (1.0 + np.exp(exponent))
        q         = q_min + (q_max - q_min) * sigmoid
        if q < q_min:
            return q_min
        if q > q_max:
            return q_max
        return float(q)

    def logistic_mapping_batch(
        cov_values: np.ndarray,
        cov_10:     float,
        cov_50:     float,
        cov_90:     float,
        q_min:      float,
        q_max:      float,
        alpha:      float = 10.0
    ) -> np.ndarray:
        cv    = np.asarray(cov_values, dtype=np.float64)
        delta = 1e-12
        a     = alpha / (cov_90 - cov_10 + delta)

        exponent = -a * (cv - cov_50)
        sigmoid  = 1.0 / (1.0 + np.exp(exponent))

        q = q_min + (q_max - q_min) * sigmoid
        return np.clip(q, q_min, q_max).astype(np.float64)
