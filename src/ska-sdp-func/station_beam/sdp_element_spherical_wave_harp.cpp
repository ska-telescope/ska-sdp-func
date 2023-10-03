/* See the LICENSE file at the top-level directory of this distribution. */

/* Spherical wave evaluation method based on Matlab code by
 * Christophe Craeye, Quentin Gueuning and Eloy de Lera Acedo.
 * See "Spherical near-field antenna measurements", J. E. Hansen, 1988 */

#include <cmath>
#include <complex>
#include <cstdlib>

#include "ska-sdp-func/math/sdp_legendre_polynomial.h"
#include "ska-sdp-func/station_beam/sdp_element_spherical_wave_harp.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"

using std::complex;

#define SPH_WAVE_HARP(M, A_TE, A_TM, C_THETA, C_PHI) \
    { \
        const complex<FP> qq(-cos_p * dpms, -sin_p * dpms); \
        const complex<FP> dd(-sin_p * pds * (M), cos_p * pds * (M)); \
        C_PHI += qq * A_TM - dd * A_TE; \
        C_THETA += dd * A_TM + qq * A_TE; \
    } \



template<typename FP>
void sdp_spherical_wave_pattern_harp (
        const int num_points,
        const FP* theta,
        const FP* phi_x,
        const FP* phi_y,
        const int l_max,
        const complex<FP>* alpha,
        const int offset,
        complex<FP>* pattern
)
{
    #pragma omp parallel for
    for (int i = 0; i < num_points; ++i)
    {
        FP theta_ = theta[i];
        complex<FP> x_phi(0, 0), x_theta(0, 0), y_phi(0, 0), y_theta(0, 0);

        // Avoid divide-by-zero (also in Matlab code!).
        if (theta_ < (FP)1e-5) theta_ = (FP)1e-5;
        const FP phi_x_ = phi_x[i];
        const FP phi_y_ = phi_y[i];
        if (phi_x_ != phi_x_)
        {
            // Propagate NAN.
            x_theta = x_phi = complex<FP>(phi_x_, phi_x_);
            y_theta = y_phi = complex<FP>(phi_x_, phi_x_);
        }
        else
        {
            const FP sin_t = sin(theta_), cos_t = cos(theta_);
            for (int l = 1; l <= l_max; ++l)
            {
                const int ind0 = l * l - 1 + l;
                const FP f_ = (2 * l + 1) / (4 * ((FP)M_PI) * l * (l + 1));
                for (int abs_m = l; abs_m >= 0; --abs_m)
                {
                    FP p, pds, dpms, sin_p, cos_p;
                    sdp_legendre2(l, abs_m, cos_t, sin_t, p, pds, dpms);
                    if (abs_m == 0)
                    {
                        sin_p = (FP)0;
                        cos_p = sqrt(f_);
                        const complex<FP> te_x = alpha[4 * ind0 + 0];
                        const complex<FP> tm_x = alpha[4 * ind0 + 1];
                        const complex<FP> te_y = alpha[4 * ind0 + 2];
                        const complex<FP> tm_y = alpha[4 * ind0 + 3];
                        SPH_WAVE_HARP(0, te_x, tm_x, x_theta, x_phi)
                        SPH_WAVE_HARP(0, te_y, tm_y, y_theta, y_phi)
                    }
                    else
                    {
                        FP d_fact = (FP)1, s_fact = (FP)1;
                        const int d_ = l - abs_m, s_ = l + abs_m;
                        for (int i_ = 2; i_ <= d_; ++i_)
                            d_fact *= i_;
                        for (int i_ = 2; i_ <= s_; ++i_)
                            s_fact *= i_;
                        const FP ff = f_ * d_fact / s_fact;
                        const FP nf = sqrt(ff);
                        const int ind_m = 4 * (ind0 - abs_m);
                        const int ind_p = 4 * (ind0 + abs_m);
                        const complex<FP> te_x_m = alpha[ind_m + 0];
                        const complex<FP> tm_x_m = alpha[ind_m + 1];
                        const complex<FP> te_y_m = alpha[ind_m + 2];
                        const complex<FP> tm_y_m = alpha[ind_m + 3];
                        const complex<FP> te_x_p = alpha[ind_p + 0];
                        const complex<FP> tm_x_p = alpha[ind_p + 1];
                        const complex<FP> te_y_p = alpha[ind_p + 2];
                        const complex<FP> tm_y_p = alpha[ind_p + 3];
                        p = -abs_m * phi_x_;
                        sin_p = sin(p) * nf;
                        cos_p = cos(p) * nf;
                        SPH_WAVE_HARP(-abs_m, te_x_m, tm_x_m, x_theta, x_phi)
                        sin_p = -sin_p;
                        SPH_WAVE_HARP( abs_m, te_x_p, tm_x_p, x_theta, x_phi)
                        p = -abs_m * phi_y_;
                        sin_p = sin(p) * nf;
                        cos_p = cos(p) * nf;
                        SPH_WAVE_HARP(-abs_m, te_y_m, tm_y_m, y_theta, y_phi)
                        sin_p = -sin_p;
                        SPH_WAVE_HARP( abs_m, te_y_p, tm_y_p, y_theta, y_phi)
                    }
                }
            }
        }
        // For some reason the theta/phi components must be reversed.
        const int i_out = 4 * (i + offset);
        pattern[i_out + 0] = x_phi;
        pattern[i_out + 1] = x_theta;
        pattern[i_out + 2] = y_phi;
        pattern[i_out + 3] = y_theta;
    }
}


void sdp_element_beam_spherical_wave_harp(
        int num_points,
        const sdp_Mem* theta_rad,
        const sdp_Mem* phi_x_rad,
        const sdp_Mem* phi_y_rad,
        int l_max,
        const sdp_Mem* coeffs,
        int index_offset_element_beam,
        sdp_Mem* element_beam,
        sdp_Error* status
)
{
    if (*status) return;
    const int location = sdp_mem_location(element_beam);
    const int element_beam_is_complex4 = sdp_mem_is_complex4(element_beam);
    const sdp_MemType type = sdp_mem_type(element_beam);
    if (sdp_mem_location(theta_rad) != location ||
            sdp_mem_location(phi_x_rad) != location ||
            sdp_mem_location(phi_y_rad) != location ||
            sdp_mem_location(coeffs) != location)
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Memory location mismatch");
        return;
    }
    if (!sdp_mem_is_complex(element_beam))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Element beam array must be of complex type");
        return;
    }
    if (location == SDP_MEM_CPU)
    {
        if (element_beam_is_complex4)
        {
            if (type == SDP_MEM_COMPLEX_FLOAT)
            {
                sdp_spherical_wave_pattern_harp<float>(num_points,
                        (const float*)sdp_mem_data_const(theta_rad),
                        (const float*)sdp_mem_data_const(phi_x_rad),
                        (const float*)sdp_mem_data_const(phi_y_rad),
                        l_max,
                        (const complex<float>*)sdp_mem_data_const(coeffs),
                        index_offset_element_beam,
                        (complex<float>*)sdp_mem_data(element_beam)
                );
            }
            else if (type == SDP_MEM_COMPLEX_DOUBLE)
            {
                sdp_spherical_wave_pattern_harp<double>(num_points,
                        (const double*)sdp_mem_data_const(theta_rad),
                        (const double*)sdp_mem_data_const(phi_x_rad),
                        (const double*)sdp_mem_data_const(phi_y_rad),
                        l_max,
                        (const complex<double>*)sdp_mem_data_const(coeffs),
                        index_offset_element_beam,
                        (complex<double>*)sdp_mem_data(element_beam)
                );
            }
            else
            {
                *status = SDP_ERR_DATA_TYPE;
                SDP_LOG_ERROR("Unsupported data type");
            }
        }
        else // complex scalar output
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type");
        }
    }
    else if (location == SDP_MEM_GPU)
    {
        const uint64_t num_threads[] = {256, 1, 1};
        const uint64_t num_blocks[] = {
            (num_points + num_threads[0] - 1) / num_threads[0], 1, 1
        };
        const char* kernel_name = 0;
        if (element_beam_is_complex4)
        {
            if (type == SDP_MEM_COMPLEX_FLOAT)
            {
                kernel_name = "sdp_spherical_wave_pattern_harp<float>";
            }
            else if (type == SDP_MEM_COMPLEX_DOUBLE)
            {
                kernel_name = "sdp_spherical_wave_pattern_harp<double>";
            }
            else
            {
                *status = SDP_ERR_DATA_TYPE;
                SDP_LOG_ERROR("Unsupported data type");
            }
            const void* args[] = {
                &num_points,
                sdp_mem_gpu_buffer_const(theta_rad, status),
                sdp_mem_gpu_buffer_const(phi_x_rad, status),
                sdp_mem_gpu_buffer_const(phi_y_rad, status),
                &l_max,
                sdp_mem_gpu_buffer_const(coeffs, status),
                &index_offset_element_beam,
                sdp_mem_gpu_buffer(element_beam, status)
            };
            sdp_launch_cuda_kernel(
                    kernel_name, num_blocks, num_threads, 0, 0, args, status
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type");
        }
    }
}
