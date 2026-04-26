#ifndef ENTRAP_TWONN_H
#define ENTRAP_TWONN_H

#include <stddef.h>

void entrap_compute_twonn_mu(
    const double *dist,
    size_t        N,
    double       *mu_out
);

#endif /* ENTRAP_TWONN_H */
