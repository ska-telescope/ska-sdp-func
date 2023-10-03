/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ska-sdp-func/math/sdp_legendre_polynomial.h"
#include "ska-sdp-func/station_beam/sdp_element_spherical_wave_harp.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"

#define C_0 299792458.0

using std::complex;
typedef complex<double> complexd;
typedef complex<float> complexf;


#define SPH_WAVE_HARP(M, A_TE, A_TM, C_THETA, C_PHI) \
    { \
        const complex<FP> qq(-cos_p * dpms, -sin_p * dpms); \
        const complex<FP> dd(-sin_p * pds * (M), cos_p * pds * (M)); \
        C_PHI += qq * A_TM - dd * A_TE; \
        C_THETA += dd * A_TM + qq * A_TE; \
    } \



template<typename FP>
static void check_results(
        const char* test_name,
        const int num_points,
        const FP* theta,
        const FP* phi_x,
        const FP* phi_y,
        const int l_max,
        const complex<FP>* alpha,
        const int offset,
        complex<FP>* pattern,
        const sdp_Error* status
)
{
    if (*status)
    {
        SDP_LOG_ERROR("%s: Test failed (error signalled)", test_name);
        return;
    }
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
        assert(fabs(pattern[i_out + 0].real() - x_phi.real()) < 1e-5);
        assert(fabs(pattern[i_out + 0].imag() - x_phi.imag()) < 1e-5);
        assert(fabs(pattern[i_out + 1].real() - x_theta.real()) < 1e-5);
        assert(fabs(pattern[i_out + 1].imag() - x_theta.imag()) < 1e-5);
        assert(fabs(pattern[i_out + 2].real() - y_phi.real()) < 1e-5);
        assert(fabs(pattern[i_out + 2].imag() - y_phi.imag()) < 1e-5);
        assert(fabs(pattern[i_out + 3].real() - y_theta.real()) < 1e-5);
        assert(fabs(pattern[i_out + 3].imag() - y_theta.imag()) < 1e-5);
    }
    SDP_LOG_INFO("%s: Test passed", test_name);
}


static void run_and_check(
        const char* test_name,
        bool expect_pass,
        sdp_MemType data_type,
        sdp_MemLocation in_location,
        sdp_MemLocation out_location,
        sdp_Error* status
)
{
    // Generate test data and copy to specified location.
    const sdp_MemType cplx = (sdp_MemType)(data_type | SDP_MEM_COMPLEX);

    // Coordinates.
    const int64_t num_points = 100;
    sdp_Mem* theta = sdp_mem_create(
            data_type, SDP_MEM_CPU, 1, &num_points, status
    );
    sdp_Mem* phi_x = sdp_mem_create(
            data_type, SDP_MEM_CPU, 1, &num_points, status
    );
    sdp_Mem* phi_y = sdp_mem_create(
            data_type, SDP_MEM_CPU, 1, &num_points, status
    );
    sdp_mem_random_fill(theta, status);
    sdp_mem_random_fill(phi_x, status);
    sdp_mem_random_fill(phi_y, status);
    sdp_Mem* in_theta = sdp_mem_create_copy(theta, in_location, status);
    sdp_Mem* in_phi_x = sdp_mem_create_copy(phi_x, in_location, status);
    sdp_Mem* in_phi_y = sdp_mem_create_copy(phi_y, in_location, status);

    // Coefficients.
    const int l_max = 3;
    const int num_coeffs = (l_max + 1) * (l_max + 1) - 1;
    const int64_t coeff_shape[] = {num_coeffs, 4};
    sdp_Mem* coeffs = sdp_mem_create(cplx, SDP_MEM_CPU, 2, coeff_shape, status);
    if (data_type == SDP_MEM_DOUBLE)
    {
        complexd* coe = (complexd*)sdp_mem_data(coeffs);
        for (int degree = 1, i_coeff = 0; degree <= l_max; ++degree)
        {
            for (int order = -degree; order <= degree; ++order, i_coeff += 4)
            {
                coe[i_coeff + 0] = complexd(1.23 * degree, -0.12 * order);
                coe[i_coeff + 1] = complexd(1.45 * degree, 0.24 * order);
                coe[i_coeff + 2] = complexd(-1.67 * degree, -0.36 * order);
                coe[i_coeff + 3] = complexd(1.89 * degree, 0.48 * order);
            }
        }
    }
    else
    {
        complexf* coe = (complexf*)sdp_mem_data(coeffs);
        for (int degree = 1, i_coeff = 0; degree <= l_max; ++degree)
        {
            for (int order = -degree; order <= degree; ++order, i_coeff += 4)
            {
                coe[i_coeff + 0] = complexf(1.23 * degree, -0.12 * order);
                coe[i_coeff + 1] = complexf(1.45 * degree, 0.24 * order);
                coe[i_coeff + 2] = complexf(-1.67 * degree, -0.36 * order);
                coe[i_coeff + 3] = complexf(1.89 * degree, 0.48 * order);
            }
        }
    }
    sdp_Mem* in_coeffs = sdp_mem_create_copy(coeffs, in_location, status);

    // Output beam array.
    const int64_t data_shape[] = {num_points, 4};
    sdp_Mem* beam = sdp_mem_create(cplx, out_location, 2, data_shape, status);
    sdp_mem_clear_contents(beam, status);

    // Call the function to test.
    SDP_LOG_INFO("Running test: %s", test_name);
    sdp_element_beam_spherical_wave_harp(
            (int) num_points, in_theta, in_phi_x, in_phi_y,
            l_max, coeffs, 0, beam, status
    );

    // Copy the output for checking.
    sdp_Mem* out_beam = sdp_mem_create_copy(beam, SDP_MEM_CPU, status);

    // Check output only if test is expected to pass.
    if (expect_pass)
    {
        if (data_type == SDP_MEM_DOUBLE)
        {
            check_results<double>(
                    test_name,
                    num_points,
                    (double*)sdp_mem_data(theta),
                    (double*)sdp_mem_data(phi_x),
                    (double*)sdp_mem_data(phi_y),
                    l_max,
                    (complex<double>*)sdp_mem_data(coeffs),
                    0,
                    (complex<double>*)sdp_mem_data(out_beam),
                    status
            );
        }
        else
        {
            check_results<float>(
                    test_name,
                    num_points,
                    (float*)sdp_mem_data(theta),
                    (float*)sdp_mem_data(phi_x),
                    (float*)sdp_mem_data(phi_y),
                    l_max,
                    (complex<float>*)sdp_mem_data(coeffs),
                    0,
                    (complex<float>*)sdp_mem_data(out_beam),
                    status
            );
        }
    }
    sdp_mem_ref_dec(theta);
    sdp_mem_ref_dec(in_theta);
    sdp_mem_ref_dec(phi_x);
    sdp_mem_ref_dec(in_phi_x);
    sdp_mem_ref_dec(phi_y);
    sdp_mem_ref_dec(in_phi_y);
    sdp_mem_ref_dec(coeffs);
    sdp_mem_ref_dec(in_coeffs);
    sdp_mem_ref_dec(beam);
    sdp_mem_ref_dec(out_beam);
}


int main()
{
    // Happy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, double precision", true,
                SDP_MEM_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, single precision", true,
                SDP_MEM_FLOAT,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
#if 0
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU, double precision", true,
                SDP_MEM_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU, single precision", true,
                SDP_MEM_FLOAT,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
#endif

    return 0;
}
