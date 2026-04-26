#include "twonn.h"

#include <float.h>
#include <stddef.h>


void entrap_compute_twonn_mu(
    const double *dist,
    size_t        N,
    double       *mu_out
) {
    size_t i, j;
    double d_ij;
    double min1, min2;

    for (i = 0; i < N; i++) {
        min1 = DBL_MAX;
        min2 = DBL_MAX;

        for (j = 0; j < N; j++) {
            if (j == i) {
                continue;
            }
            d_ij = dist[i * N + j];

            if (d_ij < min1) {
                min2 = min1;
                min1 = d_ij;
            } else if (d_ij < min2) {
                min2 = d_ij;
            }
        }

        if (min1 > 1e-12) {
            mu_out[i] = min2 / min1;
        } else {
            mu_out[i] = 1.0;
        }
    }
}
