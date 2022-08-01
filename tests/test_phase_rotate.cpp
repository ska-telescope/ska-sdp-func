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

#include "ska-sdp-func/phase_rotate/sdp_phase_rotate.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"

#define C_0 299792458.0
#define INDEX_3D(N3, N2, N1, I3, I2, I1)         (N1 * (N2 * I3 + I2) + I1)
#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)

using std::complex;

template<typename FP>
static void check_results_rotate_uvw(
        const char* test_name,
        const sdp_SkyCoord* phase_centre_orig,
        const sdp_SkyCoord* phase_centre_new,
        const int64_t num,
        const FP* uvw_in,
        const FP* uvw_out,
        const sdp_Error* status)
{
    if (*status)
    {
        SDP_LOG_ERROR("%s: Test failed (error signalled)", test_name);
        return;
    }

    // Rotate by -delta_ra around v, then delta_dec around u.
    const double orig_ra_rad = sdp_sky_coord_value(phase_centre_orig, 0);
    const double orig_dec_rad = sdp_sky_coord_value(phase_centre_orig, 1);
    const double new_ra_rad = sdp_sky_coord_value(phase_centre_new, 0);
    const double new_dec_rad = sdp_sky_coord_value(phase_centre_new, 1);
    const double d_a = -(new_ra_rad - orig_ra_rad);
    const double d_d = (new_dec_rad - orig_dec_rad);
    const double sin_d_a = sin(d_a);
    const double cos_d_a = cos(d_a);
    const double sin_d_d = sin(d_d);
    const double cos_d_d = cos(d_d);
    const double tol = 1e-5;
    double mat[9];
    mat[0] =  cos_d_a;           mat[1] = 0.0;     mat[2] =  sin_d_a;
    mat[3] =  sin_d_a * sin_d_d; mat[4] = cos_d_d; mat[5] = -cos_d_a * sin_d_d;
    mat[6] = -sin_d_a * cos_d_d; mat[7] = sin_d_d; mat[8] =  cos_d_a * cos_d_d;

    for (int64_t i_uvw = 0; i_uvw < num; ++i_uvw)
    {
        const int64_t i_uvw3 = i_uvw * 3;
        const FP uu = uvw_in[i_uvw3 + 0];
        const FP vv = uvw_in[i_uvw3 + 1];
        const FP ww = uvw_in[i_uvw3 + 2];
        const FP original_length = sqrt(uu * uu + vv * vv + ww * ww);
        const FP new_length = sqrt(
                pow(uvw_out[i_uvw3 + 0], 2.0) +
                pow(uvw_out[i_uvw3 + 1], 2.0) +
                pow(uvw_out[i_uvw3 + 2], 2.0));

        // Check coordinates.
        assert(fabs(uvw_out[i_uvw3 + 0] - (mat[0] * uu + mat[1] * vv + mat[2] * ww)) < tol);
        assert(fabs(uvw_out[i_uvw3 + 1] - (mat[3] * uu + mat[4] * vv + mat[5] * ww)) < tol);
        assert(fabs(uvw_out[i_uvw3 + 2] - (mat[6] * uu + mat[7] * vv + mat[8] * ww)) < tol);
        assert(fabs(new_length - original_length) < tol);
    }
    SDP_LOG_INFO("%s: Test passed", test_name);
}


template<typename COORD_TYPE, typename VIS_TYPE>
static void check_results_rotate_vis(
        const char* test_name,
        const int64_t num_times,
        const int64_t num_baselines,
        const int64_t num_channels,
        const int64_t num_pols,
        const double channel_start_hz,
        const double channel_step_hz,
        const sdp_SkyCoord* phase_centre_orig,
        const sdp_SkyCoord* phase_centre_new,
        const COORD_TYPE* uvw,
        const complex<VIS_TYPE>* vis_in,
        const complex<VIS_TYPE>* vis_out,
        const sdp_Error* status)
{
    if (*status)
    {
        SDP_LOG_ERROR("%s: Test failed (error signalled)", test_name);
        return;
    }

    // Convert from spherical to tangent-plane to get delta (l, m, n).
    const double orig_ra_rad = sdp_sky_coord_value(phase_centre_orig, 0);
    const double orig_dec_rad = sdp_sky_coord_value(phase_centre_orig, 1);
    const double new_ra_rad = sdp_sky_coord_value(phase_centre_new, 0);
    const double new_dec_rad = sdp_sky_coord_value(phase_centre_new, 1);
    const double d_a = -(new_ra_rad - orig_ra_rad);
    const double sin_d_a = sin(d_a);
    const double cos_d_a = cos(d_a);
    const double sin_dec0 = sin(orig_dec_rad);
    const double cos_dec0 = cos(orig_dec_rad);
    const double sin_dec  = sin(new_dec_rad);
    const double cos_dec  = cos(new_dec_rad);
    const double l1 = cos_dec  * -sin_d_a;
    const double m1 = cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_d_a;
    const double n1 = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_d_a;
    const double delta_l = 0.0 - l1;
    const double delta_m = 0.0 - m1;
    const double delta_n = 1.0 - n1;
    const double tol = 1e-5;

    for (int64_t i_time = 0; i_time < num_times; ++i_time)
    {
        for (int64_t i_baseline = 0; i_baseline < num_baselines; ++i_baseline)
        {
            const int64_t i_uvw = INDEX_3D(
                    num_times, num_baselines, 3,
                    i_time, i_baseline, 0);
            const COORD_TYPE uu = uvw[i_uvw + 0];
            const COORD_TYPE vv = uvw[i_uvw + 1];
            const COORD_TYPE ww = uvw[i_uvw + 2];
            for (int64_t i_channel = 0; i_channel < num_channels; ++i_channel)
            {
                const double inv_wavelength = (
                        channel_start_hz + i_channel * channel_step_hz) / C_0;
                const double phase = 2.0 * M_PI * inv_wavelength * (
                        uu * delta_l + vv * delta_m + ww * delta_n);
                const double cos_phase = cos(phase), sin_phase = sin(phase);
                const std::complex<VIS_TYPE> phasor(cos_phase, sin_phase);
                for (int64_t i_pol = 0; i_pol < num_pols; ++i_pol)
                {
                    const int64_t i_vis = INDEX_4D(
                            num_times, num_baselines, num_channels, num_pols,
                            i_time, i_baseline, i_channel, i_pol);
                    complex<VIS_TYPE> diff = vis_out[i_vis] - vis_in[i_vis] * phasor;
                    assert(fabs(real(diff)) < tol);
                    assert(fabs(imag(diff)) < tol);
                }
            }
        }
    }
    SDP_LOG_INFO("%s: Test passed", test_name);
}


static void run_and_check_rotate_uvw(
        const char* test_name,
        bool expect_pass,
        bool read_only_output,
        sdp_MemType coord_type,
        sdp_MemLocation input_location,
        sdp_MemLocation output_location,
        sdp_Error* status
)
{
    // Generate some test data.
    sdp_SkyCoord* original_phase_centre = sdp_sky_coord_create(
            "icrs", 123.5 * M_PI / 180, 17.8 * M_PI / 180, 0.0);
    sdp_SkyCoord* new_phase_centre = sdp_sky_coord_create(
            "icrs", 148.3 * M_PI / 180, 38.9 * M_PI / 180, 0.0);
    const int num_baselines = 351;
    const int num_times = 10;
    int64_t uvw_shape[] = {num_times, num_baselines, 3};
    sdp_Mem* uvw_in_cpu = sdp_mem_create(
            coord_type, SDP_MEM_CPU, 3, uvw_shape, status);
    sdp_Mem* uvw_out = sdp_mem_create(
            coord_type, output_location, 3, uvw_shape, status);
    sdp_mem_random_fill(uvw_in_cpu, status);
    sdp_mem_clear_contents(uvw_out, status);
    sdp_mem_set_read_only(uvw_out, read_only_output);

    // Copy inputs to specified location.
    sdp_Mem* uvw_in = sdp_mem_create_copy(uvw_in_cpu, input_location, status);

    // Call the function to test.
    SDP_LOG_INFO("Running test: %s", test_name);
    sdp_phase_rotate_uvw(original_phase_centre, new_phase_centre,
            uvw_in, uvw_out, status);
    sdp_mem_ref_dec(uvw_in);

    // Copy the output for checking.
    sdp_Mem* uvw_out_cpu = sdp_mem_create_copy(uvw_out, SDP_MEM_CPU, status);
    sdp_mem_ref_dec(uvw_out);

    // Check output only if test is expected to pass.
    if (expect_pass)
    {
        if (coord_type == SDP_MEM_DOUBLE)
        {
            check_results_rotate_uvw<double>(test_name,
                    original_phase_centre,
                    new_phase_centre,
                    num_times * num_baselines,
                    (const double*)sdp_mem_data_const(uvw_in_cpu),
                    (const double*)sdp_mem_data_const(uvw_out_cpu),
                    status);
        }
        else
        {
            check_results_rotate_uvw<float>(test_name,
                    original_phase_centre,
                    new_phase_centre,
                    num_times * num_baselines,
                    (const float*)sdp_mem_data_const(uvw_in_cpu),
                    (const float*)sdp_mem_data_const(uvw_out_cpu),
                    status);
        }
    }
    sdp_mem_ref_dec(uvw_in_cpu);
    sdp_mem_ref_dec(uvw_out_cpu);
    sdp_sky_coord_free(original_phase_centre);
    sdp_sky_coord_free(new_phase_centre);
}

static void run_and_check_rotate_vis(
        const char* test_name,
        bool expect_pass,
        bool read_only_output,
        sdp_MemType coord_type,
        sdp_MemType vis_type,
        sdp_MemLocation input_location,
        sdp_MemLocation output_location,
        sdp_Error* status
)
{
    // Generate some test data.
    sdp_SkyCoord* original_phase_centre = sdp_sky_coord_create(
            "icrs", 123.5 * M_PI / 180, 17.8 * M_PI / 180, 0.0);
    sdp_SkyCoord* new_phase_centre = sdp_sky_coord_create(
            "icrs", 148.3 * M_PI / 180, 38.9 * M_PI / 180, 0.0);
    const double channel_start_hz = 100e6;
    const double channel_step_hz = 10e6;
    const int num_channels = 3;
    const int num_baselines = 351;
    const int num_times = 10;
    const int num_pols = 4;
    int64_t uvw_shape[] = {num_times, num_baselines, 3};
    int64_t vis_shape[] = {num_times, num_baselines, num_channels, num_pols};
    sdp_Mem* uvw = sdp_mem_create(
            coord_type, SDP_MEM_CPU, 3, uvw_shape, status);
    sdp_Mem* vis_in_cpu = sdp_mem_create(
            vis_type, SDP_MEM_CPU, 4, vis_shape, status);
    sdp_Mem* vis_out = sdp_mem_create(
            vis_type, output_location, 4, vis_shape, status);
    sdp_mem_random_fill(uvw, status);
    sdp_mem_random_fill(vis_in_cpu, status);
    sdp_mem_clear_contents(vis_out, status);
    sdp_mem_set_read_only(vis_out, read_only_output);

    // Copy inputs to specified location.
    sdp_Mem* uvw_in = sdp_mem_create_copy(uvw, input_location, status);
    sdp_Mem* vis_in = sdp_mem_create_copy(vis_in_cpu, input_location, status);

    // Call the function to test.
    SDP_LOG_INFO("Running test: %s", test_name);
    sdp_phase_rotate_vis(original_phase_centre, new_phase_centre,
            channel_start_hz, channel_step_hz, uvw_in, vis_in, vis_out, status);
    sdp_mem_ref_dec(uvw_in);
    sdp_mem_ref_dec(vis_in);

    // Copy the output for checking.
    sdp_Mem* vis_out_cpu = sdp_mem_create_copy(vis_out, SDP_MEM_CPU, status);
    sdp_mem_ref_dec(vis_out);

    // Check output only if test is expected to pass.
    if (expect_pass)
    {
        if (vis_type == SDP_MEM_COMPLEX_DOUBLE)
        {
            check_results_rotate_vis<double, double>(test_name,
                    num_times, num_baselines, num_channels, num_pols,
                    channel_start_hz, channel_step_hz,
                    original_phase_centre, new_phase_centre,
                    (const double*)sdp_mem_data_const(uvw),
                    (const complex<double>*)sdp_mem_data_const(vis_in_cpu),
                    (const complex<double>*)sdp_mem_data_const(vis_out_cpu),
                    status);
        }
        else
        {
            check_results_rotate_vis<float, float>(test_name,
                    num_times, num_baselines, num_channels, num_pols,
                    channel_start_hz, channel_step_hz,
                    original_phase_centre, new_phase_centre,
                    (const float*)sdp_mem_data_const(uvw),
                    (const complex<float>*)sdp_mem_data_const(vis_in_cpu),
                    (const complex<float>*)sdp_mem_data_const(vis_out_cpu),
                    status);
        }
    }
    sdp_mem_ref_dec(uvw);
    sdp_mem_ref_dec(vis_in_cpu);
    sdp_mem_ref_dec(vis_out_cpu);
    sdp_sky_coord_free(original_phase_centre);
    sdp_sky_coord_free(new_phase_centre);
}

int main()
{
    // Happy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_rotate_uvw("CPU, phase_rotate_uvw, double precision",
                true, false, SDP_MEM_DOUBLE, SDP_MEM_CPU, SDP_MEM_CPU, &status);
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_rotate_uvw("CPU, phase_rotate_uvw, single precision",
                true, false, SDP_MEM_FLOAT, SDP_MEM_CPU, SDP_MEM_CPU, &status);
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_rotate_vis("CPU, phase_rotate_vis, double precision",
                true, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status);
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_rotate_vis("CPU, phase_rotate_vis, single precision",
                true, false,
                SDP_MEM_FLOAT, SDP_MEM_COMPLEX_FLOAT,
                SDP_MEM_CPU, SDP_MEM_CPU, &status);
        assert(status == SDP_SUCCESS);
    }
#ifdef SDP_HAVE_CUDA
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_rotate_uvw("GPU, phase_rotate_uvw, double precision",
                true, false, SDP_MEM_DOUBLE, SDP_MEM_GPU, SDP_MEM_GPU, &status);
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_rotate_uvw("GPU, phase_rotate_uvw, single precision",
                true, false, SDP_MEM_FLOAT, SDP_MEM_GPU, SDP_MEM_GPU, &status);
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_rotate_vis("GPU, phase_rotate_vis, double precision",
                true, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status);
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_rotate_vis("GPU, phase_rotate_vis, single precision",
                true, false,
                SDP_MEM_FLOAT, SDP_MEM_COMPLEX_FLOAT,
                SDP_MEM_GPU, SDP_MEM_GPU, &status);
        assert(status == SDP_SUCCESS);
    }
#endif

    // Unhappy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_rotate_uvw("Read-only output (rotate_uvw)",
                false, true, SDP_MEM_DOUBLE, SDP_MEM_CPU, SDP_MEM_CPU, &status);
        assert(status != SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_rotate_vis("Read-only output (rotate_vis)",
                false, true,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status);
        assert(status != SDP_SUCCESS);
    }
#ifdef SDP_HAVE_CUDA
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_rotate_uvw("Location mismatch (rotate_uvw)",
                false, false, SDP_MEM_DOUBLE, SDP_MEM_CPU, SDP_MEM_GPU, &status);
        assert(status == SDP_ERR_MEM_LOCATION);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_rotate_vis("Location mismatch (rotate_vis)",
                false, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_GPU, &status);
        assert(status == SDP_ERR_MEM_LOCATION);
    }
#endif
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_rotate_uvw("Bad data type (rotate_uvw)",
                false, false, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status);
        assert(status != SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_rotate_vis("Bad data type (rotate_vis)",
                false, false,
                SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status);
        assert(status != SDP_SUCCESS);
    }

    return 0;
}
