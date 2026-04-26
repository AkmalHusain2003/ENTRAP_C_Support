#ifndef ENTRAP_COV_ENTROPY_H
#define ENTRAP_COV_ENTROPY_H

#include <stddef.h>

void entrap_compute_cov_from_rows(
    const double *neighbor_distances,
    size_t        n_points,
    size_t        m,
    double       *covs_out
);

void entrap_compute_cluster_mean(
    const double *cluster_points,
    size_t        n,
    size_t        d,
    double       *mean_out
);

void entrap_compute_cluster_covariance(
    const double *cluster_points,
    size_t        n,
    size_t        d,
    const double *mean,
    double        ridge_epsilon,
    double       *cov_out
);

double entrap_compute_mahalanobis_sq(
    const double *diff,
    const double *sigma_inv,
    size_t        d
);

double entrap_logistic_mapping(
    double cov_value,
    double cov_10,
    double cov_50,
    double cov_90,
    double q_min,
    double q_max,
    double alpha
);

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
);

#endif /* ENTRAP_COV_ENTROPY_H */
