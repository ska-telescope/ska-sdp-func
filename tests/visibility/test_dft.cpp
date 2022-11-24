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

#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"
#include "ska-sdp-func/visibility/sdp_dft.h"

#define C_0 299792458.0
#define INDEX_3D(N3, N2, N1, I3, I2, I1)         (N1 * (N2 * I3 + I2) + I1)
#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) \
    (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)

using std::complex;


template<
        typename DIR_TYPE,
        typename FLUX_TYPE,
        typename UVW_TYPE,
        typename VIS_TYPE
>
static void check_results_v00(
        const char* test_name,
        int num_components,
        int num_pols,
        int num_channels,
        int num_baselines,
        int num_times,
        const DIR_TYPE* const __restrict__ source_directions,
        const complex<FLUX_TYPE>* const __restrict__ source_fluxes,
        const UVW_TYPE* const __restrict__ uvw_lambda,
        const complex<VIS_TYPE>* const __restrict__ vis,
        const sdp_Error* status
)
{
    if (*status)
    {
        SDP_LOG_ERROR("%s: Test failed (error signalled)", test_name);
        return;
    }
    for (int i_time = 0; i_time < num_times; ++i_time)
    {
        for (int i_channel = 0; i_channel < num_channels; ++i_channel)
        {
            for (int i_baseline = 0; i_baseline < num_baselines; ++i_baseline)
            {
                // Local visibility. (Allow up to 4 polarisations.)
                complex<VIS_TYPE> vis_local[4];
                vis_local[0] = vis_local[1] = vis_local[2] = vis_local[3] = 0;

                // Load uvw-coordinates.
                const unsigned int i_uvw = INDEX_4D(
                        num_times, num_baselines, num_channels, 3,
                        i_time, i_baseline, i_channel, 0
                );
                const UVW_TYPE uu = uvw_lambda[i_uvw];
                const UVW_TYPE vv = uvw_lambda[i_uvw + 1];
                const UVW_TYPE ww = uvw_lambda[i_uvw + 2];

                // Loop over components and calculate phase for each.
                for (int i_component = 0;
                        i_component < num_components; ++i_component)
                {
                    const unsigned int i_dir = 3 * i_component;
                    const DIR_TYPE l = source_directions[i_dir];
                    const DIR_TYPE m = source_directions[i_dir + 1];
                    const DIR_TYPE n = source_directions[i_dir + 2];
                    const double phase = -2.0 * M_PI *
                            (l * uu + m * vv + n * ww);
                    const double cos_phase = cos(phase), sin_phase = sin(phase);
                    const complex<VIS_TYPE> phasor(cos_phase, sin_phase);

                    // Multiply by flux in each polarisation and accumulate.
                    const unsigned int i_pol_start = INDEX_3D(
                            num_components, num_channels, num_pols,
                            i_component, i_channel, 0
                    );
                    for (int i_pol = 0; i_pol < num_pols; ++i_pol)
                    {
                        const complex<FLUX_TYPE> flux =
                                source_fluxes[i_pol_start + i_pol];
                        const complex<VIS_TYPE> flux_cast(
                                real(flux), imag(flux));
                        vis_local[i_pol] += phasor * flux_cast;
                    }
                }

                // Check visibilities.
                for (int i_pol = 0; i_pol < num_pols; ++i_pol)
                {
                    const unsigned int i_out = INDEX_4D(
                            num_times, num_baselines, num_channels, num_pols,
                            i_time, i_baseline, i_channel, i_pol
                    );
                    complex<VIS_TYPE> diff = vis[i_out] - vis_local[i_pol];
                    assert(fabs(real(diff)) < 1e-5);
                    assert(fabs(imag(diff)) < 1e-5);
                }
            }
        }
    }
    SDP_LOG_INFO("%s: Test passed", test_name);
}


static void run_and_check_v00(
        const char* test_name,
        bool expect_pass,
        int num_pols,
        bool read_only_output,
        sdp_MemType coord_type,
        sdp_MemType flux_type,
        sdp_MemType vis_type,
        sdp_MemLocation input_location,
        sdp_MemLocation output_location,
        sdp_Error* status
)
{
    // Generate some test data.
    const int num_components = 20;
    const int num_channels = 3;
    const int num_baselines = 351;
    const int num_times = 10;
    int64_t source_dirs_shape[] = {num_components, 3};
    int64_t source_flux_shape[] = {num_components, num_channels, num_pols};
    int64_t uvw_shape[] = {num_times, num_baselines, num_channels, 3};
    int64_t vis_shape[] = {num_times, num_baselines, num_channels, num_pols};
    sdp_Mem* source_dirs = sdp_mem_create(
            coord_type, SDP_MEM_CPU, 2, source_dirs_shape, status
    );
    sdp_Mem* source_flux = sdp_mem_create(
            flux_type, SDP_MEM_CPU, 3, source_flux_shape, status
    );
    sdp_Mem* uvw = sdp_mem_create(
            coord_type, SDP_MEM_CPU, 4, uvw_shape, status
    );
    sdp_Mem* vis = sdp_mem_create(
            vis_type, output_location, 4, vis_shape, status
    );
    sdp_mem_random_fill(source_dirs, status);
    sdp_mem_random_fill(source_flux, status);
    sdp_mem_random_fill(uvw, status);
    sdp_mem_clear_contents(vis, status);
    sdp_mem_set_read_only(vis, read_only_output);

    // Copy inputs to specified location.
    sdp_Mem* source_dirs_in = sdp_mem_create_copy(
            source_dirs, input_location, status
    );
    sdp_Mem* source_flux_in = sdp_mem_create_copy(
            source_flux, input_location, status
    );
    sdp_Mem* uvw_in = sdp_mem_create_copy(uvw, input_location, status);

    // Call the function to test.
    SDP_LOG_INFO("Running test (v00): %s", test_name);
    sdp_dft_point_v00(source_dirs_in, source_flux_in, uvw_in, vis, status);
    sdp_mem_ref_dec(source_flux_in);
    sdp_mem_ref_dec(source_dirs_in);
    sdp_mem_ref_dec(uvw_in);

    // Copy the output for checking.
    sdp_Mem* vis_out = sdp_mem_create_copy(vis, SDP_MEM_CPU, status);
    sdp_mem_ref_dec(vis);

    // Check output only if test is expected to pass.
    if (expect_pass)
    {
        if (vis_type == SDP_MEM_COMPLEX_DOUBLE)
        {
            check_results_v00(test_name, num_components, num_pols,
                    num_channels, num_baselines, num_times,
                    (const double*)sdp_mem_data_const(source_dirs),
                    (const complex<double>*)sdp_mem_data_const(source_flux),
                    (const double*)sdp_mem_data_const(uvw),
                    (const complex<double>*)sdp_mem_data_const(vis_out),
                    status
            );
        }
        else
        {
            check_results_v00(test_name, num_components, num_pols,
                    num_channels, num_baselines, num_times,
                    (const double*)sdp_mem_data_const(source_dirs),
                    (const complex<double>*)sdp_mem_data_const(source_flux),
                    (const double*)sdp_mem_data_const(uvw),
                    (const complex<float>*)sdp_mem_data_const(vis_out),
                    status
            );
        }
    }
    sdp_mem_ref_dec(source_dirs);
    sdp_mem_ref_dec(source_flux);
    sdp_mem_ref_dec(uvw);
    sdp_mem_ref_dec(vis_out);
}


template<
        typename DIR_TYPE,
        typename FLUX_TYPE,
        typename UVW_TYPE,
        typename VIS_TYPE
>
static void check_results_v01(
        const char* test_name,
        int num_components,
        int num_pols,
        int num_channels,
        int num_baselines,
        int num_times,
        const DIR_TYPE* const __restrict__ source_directions,
        const complex<FLUX_TYPE>* const __restrict__ source_fluxes,
        const UVW_TYPE* const __restrict__ uvw_metres,
        const double channel_start_hz,
        const double channel_step_hz,
        const complex<VIS_TYPE>* const __restrict__ vis,
        const sdp_Error* status
)
{
    if (*status)
    {
        SDP_LOG_ERROR("%s: Test failed (error signalled)", test_name);
        return;
    }
    for (int i_time = 0; i_time < num_times; ++i_time)
    {
        for (int i_baseline = 0; i_baseline < num_baselines; ++i_baseline)
        {
            // Load uvw-coordinates.
            const unsigned int i_uvw = INDEX_3D(
                    num_times, num_baselines, 3,
                    i_time, i_baseline, 0
            );
            const UVW_TYPE uu = uvw_metres[i_uvw];
            const UVW_TYPE vv = uvw_metres[i_uvw + 1];
            const UVW_TYPE ww = uvw_metres[i_uvw + 2];

            for (int i_channel = 0; i_channel < num_channels; ++i_channel)
            {
                // Local visibility. (Allow up to 4 polarisations.)
                complex<VIS_TYPE> vis_local[4];
                vis_local[0] = vis_local[1] = vis_local[2] = vis_local[3] = 0;

                // Get the inverse wavelength.
                const double inv_wavelength =
                        (channel_start_hz + i_channel * channel_step_hz) / C_0;

                // Loop over components and calculate phase for each.
                for (int i_component = 0;
                        i_component < num_components; ++i_component)
                {
                    const unsigned int i_dir = 3 * i_component;
                    const DIR_TYPE l = source_directions[i_dir];
                    const DIR_TYPE m = source_directions[i_dir + 1];
                    const DIR_TYPE n = source_directions[i_dir + 2];
                    const double phase = -2.0 * M_PI * inv_wavelength *
                            (l * uu + m * vv + n * ww);
                    const double cos_phase = cos(phase), sin_phase = sin(phase);
                    const complex<VIS_TYPE> phasor(cos_phase, sin_phase);

                    // Multiply by flux in each polarisation and accumulate.
                    const unsigned int i_pol_start = INDEX_3D(
                            num_components, num_channels, num_pols,
                            i_component, i_channel, 0
                    );
                    for (int i_pol = 0; i_pol < num_pols; ++i_pol)
                    {
                        const complex<FLUX_TYPE> flux =
                                source_fluxes[i_pol_start + i_pol];
                        const complex<VIS_TYPE> flux_cast(
                                real(flux), imag(flux));
                        vis_local[i_pol] += phasor * flux_cast;
                    }
                }

                // Check visibilities.
                for (int i_pol = 0; i_pol < num_pols; ++i_pol)
                {
                    const unsigned int i_out = INDEX_4D(
                            num_times, num_baselines, num_channels, num_pols,
                            i_time, i_baseline, i_channel, i_pol
                    );
                    complex<VIS_TYPE> diff = vis[i_out] - vis_local[i_pol];
                    assert(fabs(real(diff)) < 1e-5);
                    assert(fabs(imag(diff)) < 1e-5);
                }
            }
        }
    }
    SDP_LOG_INFO("%s: Test passed", test_name);
}


static void run_and_check_v01(
        const char* test_name,
        bool expect_pass,
        int num_pols,
        bool read_only_output,
        sdp_MemType coord_type,
        sdp_MemType flux_type,
        sdp_MemType vis_type,
        sdp_MemLocation input_location,
        sdp_MemLocation output_location,
        sdp_Error* status
)
{
    // Generate some test data.
    const int num_components = 20;
    const int num_channels = 3;
    const int num_baselines = 351;
    const int num_times = 10;
    const double channel_start_hz = 100e6;
    const double channel_step_hz = 100e3;
    int64_t source_dirs_shape[] = {num_components, 3};
    int64_t source_flux_shape[] = {num_components, num_channels, num_pols};
    int64_t uvw_shape[] = {num_times, num_baselines, 3};
    int64_t vis_shape[] = {num_times, num_baselines, num_channels, num_pols};
    sdp_Mem* source_dirs = sdp_mem_create(
            coord_type, SDP_MEM_CPU, 2, source_dirs_shape, status
    );
    sdp_Mem* source_flux = sdp_mem_create(
            flux_type, SDP_MEM_CPU, 3, source_flux_shape, status
    );
    sdp_Mem* uvw = sdp_mem_create(
            coord_type, SDP_MEM_CPU, 3, uvw_shape, status
    );
    sdp_Mem* vis = sdp_mem_create(
            vis_type, output_location, 4, vis_shape, status
    );
    sdp_mem_random_fill(source_dirs, status);
    sdp_mem_random_fill(source_flux, status);
    sdp_mem_random_fill(uvw, status);
    sdp_mem_clear_contents(vis, status);
    sdp_mem_set_read_only(vis, read_only_output);

    // Copy inputs to specified location.
    sdp_Mem* source_dirs_in = sdp_mem_create_copy(
            source_dirs, input_location, status
    );
    sdp_Mem* source_flux_in = sdp_mem_create_copy(
            source_flux, input_location, status
    );
    sdp_Mem* uvw_in = sdp_mem_create_copy(uvw, input_location, status);

    // Call the function to test.
    SDP_LOG_INFO("Running test (v01): %s", test_name);
    sdp_dft_point_v01(source_dirs_in, source_flux_in, uvw_in,
            channel_start_hz, channel_step_hz, vis, status
    );
    sdp_mem_ref_dec(source_flux_in);
    sdp_mem_ref_dec(source_dirs_in);
    sdp_mem_ref_dec(uvw_in);

    // Copy the output for checking.
    sdp_Mem* vis_out = sdp_mem_create_copy(vis, SDP_MEM_CPU, status);
    sdp_mem_ref_dec(vis);

    // Check output only if test is expected to pass.
    if (expect_pass)
    {
        if (vis_type == SDP_MEM_COMPLEX_DOUBLE)
        {
            check_results_v01(test_name, num_components, num_pols,
                    num_channels, num_baselines, num_times,
                    (const double*)sdp_mem_data_const(source_dirs),
                    (const complex<double>*)sdp_mem_data_const(source_flux),
                    (const double*)sdp_mem_data_const(uvw),
                    channel_start_hz,
                    channel_step_hz,
                    (const complex<double>*)sdp_mem_data_const(vis_out),
                    status
            );
        }
        else
        {
            check_results_v01(test_name, num_components, num_pols,
                    num_channels, num_baselines, num_times,
                    (const double*)sdp_mem_data_const(source_dirs),
                    (const complex<double>*)sdp_mem_data_const(source_flux),
                    (const double*)sdp_mem_data_const(uvw),
                    channel_start_hz,
                    channel_step_hz,
                    (const complex<float>*)sdp_mem_data_const(vis_out),
                    status
            );
        }
    }
    sdp_mem_ref_dec(source_dirs);
    sdp_mem_ref_dec(source_flux);
    sdp_mem_ref_dec(uvw);
    sdp_mem_ref_dec(vis_out);
}


int main()
{
    // Happy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v00("CPU, double precision, 1 pol", true, 1, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v00("CPU, single precision, 1 pol", true, 1, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_FLOAT,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v00("CPU, double precision, 4 pols", true, 4, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v00("CPU, single precision, 4 pols", true, 4, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_FLOAT,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
#ifdef SDP_HAVE_CUDA
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v00("GPU, double precision, 1 pol", true, 1, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v00("GPU, single precision, 1 pol", true, 1, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_FLOAT,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v00("GPU, double precision, 4 pols", true, 4, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v00("GPU, single precision, 4 pols", true, 4, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_FLOAT,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
#endif

    // Unhappy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v00("Read-only output", false, 4, true,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status != SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v00("Unsupported number of polarisations",
                false, 3, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status != SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v00("Wrong visibility type", false, 4, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_ERR_DATA_TYPE);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v00("Unsupported coordinate types", false, 4, false,
                SDP_MEM_FLOAT, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_ERR_DATA_TYPE);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v00("Unsupported flux type", false, 4, false,
                SDP_MEM_DOUBLE, SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_ERR_DATA_TYPE);
    }
#ifdef SDP_HAVE_CUDA
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v00("Memory location mismatch", false, 4, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_ERR_MEM_LOCATION);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v00("Unsupported coordinate types", false, 4, false,
                SDP_MEM_FLOAT, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_ERR_DATA_TYPE);
    }
#endif

    // Happy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v01("CPU, double precision, 1 pol", true, 1, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v01("CPU, single precision, 1 pol", true, 1, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_FLOAT,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v01("CPU, double precision, 4 pols", true, 4, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v01("CPU, single precision, 4 pols", true, 4, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_FLOAT,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
#ifdef SDP_HAVE_CUDA
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v01("GPU, double precision, 1 pol", true, 1, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v01("GPU, single precision, 1 pol", true, 1, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_FLOAT,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v01("GPU, double precision, 4 pols", true, 4, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v01("GPU, single precision, 4 pols", true, 4, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_FLOAT,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
#endif

    // Unhappy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v01("Read-only output", false, 4, true,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status != SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v01("Unsupported number of polarisations",
                false, 3, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status != SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v01("Wrong visibility type", false, 4, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_ERR_DATA_TYPE);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v01("Unsupported coordinate types", false, 4, false,
                SDP_MEM_FLOAT, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_ERR_DATA_TYPE);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v01("Unsupported flux type", false, 4, false,
                SDP_MEM_DOUBLE, SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_ERR_DATA_TYPE);
    }
#ifdef SDP_HAVE_CUDA
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v01("Memory location mismatch", false, 4, false,
                SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_ERR_MEM_LOCATION);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_v01("Unsupported coordinate types", false, 4, false,
                SDP_MEM_FLOAT, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_ERR_DATA_TYPE);
    }
#endif

    return 0;
}
