# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef extern from "cov_entropy.h":
    void entrap_compute_cov_from_rows(
        const double *neighbor_distances,
        size_t        n_points,
        size_t        m,
        double       *covs_out
    )
    void entrap_compute_cluster_mean(
        const double *cluster_points,
        size_t        n,
        size_t        d,
        double       *mean_out
    )
    void entrap_compute_cluster_covariance(
        const double *cluster_points,
        size_t        n,
        size_t        d,
        const double *mean,
        double        ridge_epsilon,
        double       *cov_out
    )
    double entrap_compute_mahalanobis_sq(
        const double *diff,
        const double *sigma_inv,
        size_t        d
    )
    double entrap_logistic_mapping(
        double cov_value,
        double cov_10,
        double cov_50,
        double cov_90,
        double q_min,
        double q_max,
        double alpha
    )
    void entrap_logistic_mapping_batch(
        const double *cov_values,
        size_t        n,
        double        cov_10,
        double        cov_50,
        double        cov_90,
        double        q_min,
        double        q_max,
        double        alpha,
        double       *q_out
    )


def compute_cov_from_rows(
    cnp.ndarray neighbor_distances not None
) -> cnp.ndarray:
    neighbor_distances = np.ascontiguousarray(neighbor_distances, dtype=np.float64)

    cdef:
        cnp.npy_intp n_points = neighbor_distances.shape[0]
        cnp.npy_intp m        = neighbor_distances.shape[1]
        cnp.ndarray[cnp.float64_t, ndim=1] covs = np.empty(n_points, dtype=np.float64)

    entrap_compute_cov_from_rows(
        <const double *>cnp.PyArray_DATA(neighbor_distances),
        <size_t>n_points,
        <size_t>m,
        <double *>cnp.PyArray_DATA(covs)
    )
    return covs


def compute_cluster_mean(
    cnp.ndarray cluster_points not None
) -> cnp.ndarray:
    cluster_points = np.ascontiguousarray(cluster_points, dtype=np.float64)

    cdef:
        cnp.npy_intp n = cluster_points.shape[0]
        cnp.npy_intp d = cluster_points.shape[1]
        cnp.ndarray[cnp.float64_t, ndim=1] mean = np.empty(d, dtype=np.float64)

    entrap_compute_cluster_mean(
        <const double *>cnp.PyArray_DATA(cluster_points),
        <size_t>n,
        <size_t>d,
        <double *>cnp.PyArray_DATA(mean)
    )
    return mean


def compute_cluster_covariance(
    cnp.ndarray cluster_points not None,
    cnp.ndarray mean           not None,
    double      ridge_epsilon
) -> cnp.ndarray:
    cluster_points = np.ascontiguousarray(cluster_points, dtype=np.float64)
    mean           = np.ascontiguousarray(mean,           dtype=np.float64)

    cdef:
        cnp.npy_intp n = cluster_points.shape[0]
        cnp.npy_intp d = cluster_points.shape[1]
        cnp.ndarray[cnp.float64_t, ndim=2] cov = np.empty((d, d), dtype=np.float64)

    entrap_compute_cluster_covariance(
        <const double *>cnp.PyArray_DATA(cluster_points),
        <size_t>n,
        <size_t>d,
        <const double *>cnp.PyArray_DATA(mean),
        ridge_epsilon,
        <double *>cnp.PyArray_DATA(cov)
    )
    return cov


def compute_mahalanobis_sq(
    cnp.ndarray diff      not None,
    cnp.ndarray Sigma_inv not None
) -> float:
    diff      = np.ascontiguousarray(diff,      dtype=np.float64)
    Sigma_inv = np.ascontiguousarray(Sigma_inv, dtype=np.float64)

    cdef cnp.npy_intp d = diff.shape[0]

    return entrap_compute_mahalanobis_sq(
        <const double *>cnp.PyArray_DATA(diff),
        <const double *>cnp.PyArray_DATA(Sigma_inv),
        <size_t>d
    )


def logistic_mapping(
    double cov_value,
    double cov_10,
    double cov_50,
    double cov_90,
    double q_min,
    double q_max,
    double alpha = 10.0
) -> float:
    return entrap_logistic_mapping(
        cov_value, cov_10, cov_50, cov_90, q_min, q_max, alpha
    )


def logistic_mapping_batch(
    cnp.ndarray cov_values not None,
    double cov_10,
    double cov_50,
    double cov_90,
    double q_min,
    double q_max,
    double alpha = 10.0
) -> cnp.ndarray:
    cov_values = np.ascontiguousarray(cov_values, dtype=np.float64)
    cdef cnp.npy_intp n = cov_values.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] q_out = np.empty(n, dtype=np.float64)

    entrap_logistic_mapping_batch(
        <const double *>cnp.PyArray_DATA(cov_values),
        <size_t>n,
        cov_10, cov_50, cov_90,
        q_min, q_max, alpha,
        <double *>cnp.PyArray_DATA(q_out)
    )
    return q_out
