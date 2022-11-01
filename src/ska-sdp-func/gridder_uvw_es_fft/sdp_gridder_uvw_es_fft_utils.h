/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef NIFTY_UTILS_H_
#define NIFTY_UTILS_H_

#include <cmath>
#include <stdarg.h>
#include <cstdio>

#include "ska-sdp-func/utility/sdp_mem.h"


#define MAX_NEWTON_RAPHSON_ITERATIONS 100
#define QUADRATURE_SUPPORT_BOUND 32


template<typename T>
void get_w_range(
        const int num_rows,
        const T* uvw,
        const int num_chan,
        const T* freq_hz,
        double& min_abs_w,
        double& max_abs_w)
{
    for (int i = 0; i < num_rows; ++i)
    {
        const double abs_w = std::fabs(uvw[3 * i + 2]);
        if (abs_w < min_abs_w) min_abs_w = abs_w;
        if (abs_w > max_abs_w) max_abs_w = abs_w;
    }

    double fscaleMin = freq_hz[0] / 299792458.0;
    double fscaleMax = freq_hz[num_chan - 1] / 299792458.0;

    min_abs_w *= fscaleMin;
    max_abs_w *= fscaleMax;
}

void sdp_generate_gauss_legendre_conv_kernel(
        int image_size,
        int grid_size,
        int support,
        double beta,
        double* quadrature_kernel,
        double* quadrature_nodes,
        double* quadrature_weights,
        double* conv_corr_kernel);

void sdp_calculate_params_from_epsilon(
        double epsilon,
        int image_size,
        int vis_precision,
        int &grid_size,
        int &support,
        double &beta,
        sdp_Error* status);

void sdp_calculate_support_and_beta(
        double upsampling,
        double epsilon,
        int &support,
        double &beta,
        int &status);

#endif /* include guard */
