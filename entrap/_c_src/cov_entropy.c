#include "cov_entropy.h"

#include <math.h>
#include <stddef.h>
#include <string.h>


void entrap_compute_cov_from_rows(
    const double *neighbor_distances,
    size_t        n_points,
    size_t        m,
    double       *covs_out
) {
    size_t i, j;
    double row_sum, mean, var_sum, diff, std_val;

    for (i = 0; i < n_points; i++) {
        row_sum = 0.0;
        for (j = 0; j < m; j++) {
            row_sum += neighbor_distances[i * m + j];
        }
        mean = row_sum / (double)m;

        if (mean > 1e-12) {
            var_sum = 0.0;
            for (j = 0; j < m; j++) {
                diff     = neighbor_distances[i * m + j] - mean;
                var_sum += diff * diff;
            }
            std_val    = sqrt(var_sum / (double)m);
            covs_out[i] = std_val / mean;
        } else {
            covs_out[i] = HUGE_VAL;
        }
    }
}


void entrap_compute_cluster_mean(
    const double *cluster_points,
    size_t        n,
    size_t        d,
    double       *mean_out
) {
    size_t i, j;

    for (j = 0; j < d; j++) {
        mean_out[j] = 0.0;
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < d; j++) {
            mean_out[j] += cluster_points[i * d + j];
        }
    }

    for (j = 0; j < d; j++) {
        mean_out[j] /= (double)n;
    }
}


void entrap_compute_cluster_covariance(
    const double *cluster_points,
    size_t        n,
    size_t        d,
    const double *mean,
    double        ridge_epsilon,
    double       *cov_out
) {
    size_t i, j, k;
    double diff_j;

    for (j = 0; j < d * d; j++) {
        cov_out[j] = 0.0;
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < d; j++) {
            diff_j = cluster_points[i * d + j] - mean[j];
            for (k = 0; k < d; k++) {
                cov_out[j * d + k] +=
                    diff_j * (cluster_points[i * d + k] - mean[k]);
            }
        }
    }

    for (j = 0; j < d; j++) {
        for (k = 0; k < d; k++) {
            cov_out[j * d + k] /= (double)n;
            if (j == k) {
                cov_out[j * d + k] += ridge_epsilon;
            }
        }
    }
}


double entrap_compute_mahalanobis_sq(
    const double *diff,
    const double *sigma_inv,
    size_t        d
) {
    size_t  i, j;
    double  result = 0.0;
    double  temp;

    for (i = 0; i < d; i++) {
        temp = 0.0;
        for (j = 0; j < d; j++) {
            temp += diff[j] * sigma_inv[j * d + i];
        }
        result += diff[i] * temp;
    }

    return result;
}


double entrap_logistic_mapping(
    double cov_value,
    double cov_10,
    double cov_50,
    double cov_90,
    double q_min,
    double q_max,
    double alpha
) {
    const double delta     = 1e-12;
    double a               = alpha / (cov_90 - cov_10 + delta);
    double exponent        = -a * (cov_value - cov_50);
    double sigmoid         = 1.0 / (1.0 + exp(exponent));
    double q_adaptive      = q_min + (q_max - q_min) * sigmoid;

    if (q_adaptive < q_min) return q_min;
    if (q_adaptive > q_max) return q_max;
    return q_adaptive;
}


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
) {
    size_t i;
    for (i = 0; i < n; i++) {
        q_out[i] = entrap_logistic_mapping(
            cov_values[i], cov_10, cov_50, cov_90, q_min, q_max, alpha
        );
    }
}
