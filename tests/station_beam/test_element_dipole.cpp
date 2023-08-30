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

#include "ska-sdp-func/station_beam/sdp_element_dipole.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"

#define C_0 299792458.0

using std::complex;


template<typename FP>
static void sdp_dipole(
        FP kL,
        FP cos_kL,
        FP phi_rad,
        FP sin_theta,
        FP cos_theta,
        complex<FP>& e_theta,
        complex<FP>& e_phi
)
{
    const FP cos_phi = cos(phi_rad);
    const FP denominator =
            (FP)1 + cos_phi * cos_phi * (cos_theta * cos_theta - (FP)1);
    if (denominator == (FP)0)
    {
        e_theta = e_phi = complex<FP>(0, 0);
    }
    else
    {
        const FP temp = (cos(kL * cos_phi * sin_theta) - cos_kL) / denominator;
        e_theta = complex<FP>(-cos_phi * cos_theta * temp, 0);
        e_phi = complex<FP>(sin(phi_rad) * temp, 0);
    }
}


template<typename FP>
static void check_results_pol(
        const char* test_name,
        const int num_points,
        const FP* theta_rad,
        const FP* phi_rad,
        const FP kL,
        const FP cos_kL,
        const int stride,
        const int e_theta_offset,
        const int e_phi_offset,
        const complex<FP>* e_theta,
        const complex<FP>* e_phi,
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
        complex<FP> e_theta_tmp(0, 0), e_phi_tmp(0, 0);
        sdp_dipole(kL, cos_kL,
                phi_rad[i], (FP)sin(theta_rad[i]), (FP)cos(theta_rad[i]),
                e_theta_tmp,
                e_phi_tmp
        );
        assert(fabs(e_theta[i * stride + e_theta_offset] - e_theta_tmp) < 1e-5);
        assert(fabs(e_phi[i * stride + e_phi_offset] - e_phi_tmp) < 1e-5);
    }
    SDP_LOG_INFO("%s: Test passed", test_name);
}


template<typename FP>
static void check_results_scalar(
        const char* test_name,
        const int num_points,
        const FP* theta_rad,
        const FP* phi_rad,
        const FP kL,
        const FP cos_kL,
        const int stride,
        const int offset,
        const complex<FP>* pattern,
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
        FP amp, phi_;
        const FP sin_theta = sin(theta_rad[i]);
        const FP cos_theta = cos(theta_rad[i]);
        phi_ = phi_rad[i];
        complex<FP> x_theta, x_phi, y_theta, y_phi;
        sdp_dipole(kL, cos_kL, phi_, sin_theta, cos_theta, x_theta, x_phi);
        phi_ += ((FP) M_PI / (FP)2);
        sdp_dipole(kL, cos_kL, phi_, sin_theta, cos_theta, y_theta, y_phi);
        amp = x_theta.real() * x_theta.real() + x_phi.real() * x_phi.real() +
                y_theta.real() * y_theta.real() + y_phi.real() * y_phi.real();
        amp /= (FP)2;
        amp = sqrt(amp);
        assert(fabs(pattern[i * stride + offset] - complex<FP>(amp, 0)) < 1e-5);
    }
    SDP_LOG_INFO("%s: Test passed", test_name);
}


static void run_and_check(
        const char* test_name,
        bool expect_pass,
        int num_pol,
        sdp_MemType data_type,
        sdp_MemLocation in_location,
        sdp_MemLocation out_location,
        sdp_Error* status
)
{
    // Generate test data and copy to specified location.
    const double dipole_length_m = 1.5;
    const double freq_hz = 100e6;
    const double kL = dipole_length_m * (M_PI * freq_hz / C_0);
    const double cos_kL = cos(kL);
    const int stride = num_pol == 1 ? 1 : 4;

    // Coordinates.
    const int64_t num_points = 100;
    sdp_Mem* theta = sdp_mem_create(
            data_type, SDP_MEM_CPU, 1, &num_points, status
    );
    sdp_Mem* phi = sdp_mem_create(
            data_type, SDP_MEM_CPU, 1, &num_points, status
    );
    sdp_mem_random_fill(theta, status);
    sdp_mem_random_fill(phi, status);
    sdp_Mem* in_theta = sdp_mem_create_copy(theta, in_location, status);
    sdp_Mem* in_phi = sdp_mem_create_copy(phi, in_location, status);

    // Output beam array.
    int64_t data_shape[] = {num_points, 1, 1};
    if (num_pol != 1) data_shape[1] = data_shape[2] = 2;
    const sdp_MemType cplx = (sdp_MemType)(data_type | SDP_MEM_COMPLEX);
    sdp_Mem* beam = sdp_mem_create(cplx, out_location, 3, data_shape, status);
    sdp_mem_clear_contents(beam, status);

    // Call the function to test.
    SDP_LOG_INFO("Running test: %s", test_name);
    sdp_element_beam_dipole(
            (int) num_points, in_theta, in_phi,
            freq_hz, dipole_length_m, stride, 0, beam, status
    );

    // Copy the output for checking.
    sdp_Mem* out_beam = sdp_mem_create_copy(beam, SDP_MEM_CPU, status);

    // Check output only if test is expected to pass.
    if (expect_pass)
    {
        if (num_pol == 1)
        {
            if (data_type == SDP_MEM_DOUBLE)
            {
                check_results_scalar<double>(
                        test_name,
                        num_points,
                        (double*)sdp_mem_data(theta),
                        (double*)sdp_mem_data(phi),
                        kL,
                        cos_kL,
                        stride,
                        0,
                        (complex<double>*)sdp_mem_data(out_beam),
                        status
                );
            }
            else
            {
                check_results_scalar<float>(
                        test_name,
                        num_points,
                        (float*)sdp_mem_data(theta),
                        (float*)sdp_mem_data(phi),
                        kL,
                        cos_kL,
                        stride,
                        0,
                        (complex<float>*)sdp_mem_data(out_beam),
                        status
                );
            }
        }
        else if (num_pol == 4)
        {
            if (data_type == SDP_MEM_DOUBLE)
            {
                check_results_pol<double>(
                        test_name,
                        num_points,
                        (double*)sdp_mem_data(theta),
                        (double*)sdp_mem_data(phi),
                        kL,
                        cos_kL,
                        stride,
                        0,
                        1,
                        (complex<double>*)sdp_mem_data(out_beam),
                        (complex<double>*)sdp_mem_data(out_beam),
                        status
                );
            }
            else
            {
                check_results_pol<float>(
                        test_name,
                        num_points,
                        (float*)sdp_mem_data(theta),
                        (float*)sdp_mem_data(phi),
                        kL,
                        cos_kL,
                        stride,
                        0,
                        1,
                        (complex<float>*)sdp_mem_data(out_beam),
                        (complex<float>*)sdp_mem_data(out_beam),
                        status
                );
            }
        }
    }
    sdp_mem_ref_dec(theta);
    sdp_mem_ref_dec(in_theta);
    sdp_mem_ref_dec(phi);
    sdp_mem_ref_dec(in_phi);
    sdp_mem_ref_dec(beam);
    sdp_mem_ref_dec(out_beam);
}


int main()
{
    // Happy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, double precision, 1 pol", true, 1,
                SDP_MEM_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, single precision, 1 pol", true, 1,
                SDP_MEM_FLOAT,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, double precision, 4 pols", true, 4,
                SDP_MEM_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, single precision, 4 pols", true, 4,
                SDP_MEM_FLOAT,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
#ifdef SDP_HAVE_CUDA
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU, double precision, 1 pol", true, 1,
                SDP_MEM_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU, single precision, 1 pol", true, 1,
                SDP_MEM_FLOAT,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU, double precision, 4 pols", true, 4,
                SDP_MEM_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU, single precision, 4 pols", true, 4,
                SDP_MEM_FLOAT,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
#endif

    return 0;
}
