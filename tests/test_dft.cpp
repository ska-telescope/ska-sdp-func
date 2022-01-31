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

#include "logging/sdp_logging.h"
#include "mem/sdp_mem.h"
#include "dft/sdp_dft.h"

#define INDEX_3D(N3, N2, N1, I3, I2, I1)         (N1 * (N2 * I3 + I2) + I1)
#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)

template<
        typename FLUX_TYPE,
        typename DIR_TYPE,
        typename UVW_TYPE,
        typename VIS_TYPE
>
void check_results(
        const char* test_name,
        int num_components,
        int num_pols,
        int num_channels,
        int num_baselines,
        int num_times,
        const std::complex<FLUX_TYPE> *const __restrict__ source_fluxes,
        const DIR_TYPE *const __restrict__ source_directions,
        const UVW_TYPE *const __restrict__ uvw_lambda,
        const std::complex<VIS_TYPE> *const __restrict__ vis,
        sdp_Error* status)
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
                std::complex<VIS_TYPE> vis_local[4];
                vis_local[0] = vis_local[1] = vis_local[2] = vis_local[3] = 0;

                // Load uvw-coordinates.
                const unsigned int i_uvw = INDEX_4D(
                        num_times, num_baselines, num_channels, 3,
                        i_time, i_baseline, i_channel, 0);
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
                    const double phase = -2.0 * M_PI * (
                            l * uu + m * vv + n * ww);
                    const double cos_phase = cos(phase), sin_phase = sin(phase);
                    const std::complex<VIS_TYPE> phasor(cos_phase, sin_phase);

                    // Multiply by flux in each polarisation and accumulate.
                    const unsigned int i_pol_start = INDEX_3D(
                            num_components, num_channels, num_pols,
                            i_component, i_channel, 0);
                    for (int i_pol = 0; i_pol < num_pols; ++i_pol)
                    {
                        const std::complex<FLUX_TYPE> flux =
                                source_fluxes[i_pol_start + i_pol];
                        const std::complex<VIS_TYPE> flux_cast(
                                real(flux), imag(flux));
                        vis_local[i_pol] += phasor * flux_cast;
                    }
                }

                // Check visibilities.
                for (int i_pol = 0; i_pol < num_pols; ++i_pol)
                {
                    const unsigned int i_out = INDEX_4D(
                            num_times, num_baselines, num_channels, num_pols,
                            i_time, i_baseline, i_channel, i_pol);
                    std::complex<VIS_TYPE> diff = vis[i_out] - vis_local[i_pol];
                    assert(fabs(real(diff)) < 1e-5);
                    assert(fabs(imag(diff)) < 1e-5);
                }
            }
        }
    }
    SDP_LOG_INFO("%s: Test passed", test_name);
}


int main()
{
    // Generate some test data.
    // Use C++ vectors as an example of externally-managed memory.
    const int num_components = 20;
    const int num_pols = 4;
    const int num_channels = 10;
    const int num_baselines = 351;
    const int num_times = 10;
    std::vector<std::complex<double> > source_fluxes_vec(
        num_components * num_channels * num_pols);
    std::vector<double> source_directions_vec(num_components * 3);
    std::vector<double> uvw_lambda_vec(
        num_times * num_baselines * num_channels * 3);
    std::vector<std::complex<double> > vis_vec(
        num_times * num_baselines * num_channels * num_pols);
    srand(1);
    for (size_t i = 0; i < source_fluxes_vec.size(); ++i)
    {
        source_fluxes_vec[i] = std::complex<double>(
                rand() / (double)RAND_MAX, rand() / (double)RAND_MAX);
    }
    for (size_t i = 0; i < source_directions_vec.size(); ++i)
    {
        source_directions_vec[i] = rand() / (double)RAND_MAX;
    }
    for (size_t i = 0; i < uvw_lambda_vec.size(); ++i)
    {
        uvw_lambda_vec[i] = rand() / (double)RAND_MAX;
    }

    // Wrap pointers to externally-managed memory.
    sdp_Error status = SDP_SUCCESS;
    sdp_Mem* source_fluxes = sdp_mem_create_from_raw(
            source_fluxes_vec.data(), SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU,
            source_fluxes_vec.size(), &status);
    sdp_Mem* source_directions = sdp_mem_create_from_raw(
            source_directions_vec.data(), SDP_MEM_DOUBLE, SDP_MEM_CPU,
            source_directions_vec.size(), &status);
    sdp_Mem* uvw_lambda = sdp_mem_create_from_raw(
            uvw_lambda_vec.data(), SDP_MEM_DOUBLE, SDP_MEM_CPU,
            uvw_lambda_vec.size(), &status);
    sdp_Mem* vis = sdp_mem_create_from_raw(
            vis_vec.data(), SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU,
            vis_vec.size(), &status);

    // Call CPU version of processing function.
    sdp_dft_point_v00(
            num_components, num_pols, num_channels, num_baselines, num_times,
            source_fluxes, source_directions, uvw_lambda, vis,
            &status);

    // Check results.
    check_results("CPU DFT",
            num_components, num_pols, num_channels, num_baselines, num_times,
            (const std::complex<double>*)sdp_mem_data_const(source_fluxes),
            (const double*)sdp_mem_data_const(source_directions),
            (const double*)sdp_mem_data_const(uvw_lambda),
            (const std::complex<double>*)sdp_mem_data_const(vis),
            &status);
    sdp_mem_free(vis);

#ifdef SDP_HAVE_CUDA
    // Copy test data to GPU.
    sdp_Mem* source_fluxes_gpu = sdp_mem_create_copy(
            source_fluxes, SDP_MEM_GPU, &status);
    sdp_Mem* source_directions_gpu = sdp_mem_create_copy(
            source_directions, SDP_MEM_GPU, &status);
    sdp_Mem* uvw_lambda_gpu = sdp_mem_create_copy(
            uvw_lambda, SDP_MEM_GPU, &status);
    sdp_Mem* vis_gpu = sdp_mem_create(
            SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_GPU, vis_vec.size(), &status);
    sdp_mem_clear_contents(vis_gpu, &status);

    // Call GPU version of processing function.
    sdp_dft_point_v00(
            num_components, num_pols, num_channels, num_baselines, num_times,
            source_fluxes_gpu, source_directions_gpu, uvw_lambda_gpu, vis_gpu,
            &status);
    sdp_mem_free(source_fluxes_gpu);
    sdp_mem_free(source_directions_gpu);
    sdp_mem_free(uvw_lambda_gpu);

    // Copy GPU output back to host for checking.
    sdp_Mem* vis2 = sdp_mem_create_copy(vis_gpu, SDP_MEM_CPU, &status);
    sdp_mem_free(vis_gpu);

    // Check results.
    check_results("GPU DFT",
            num_components, num_pols, num_channels, num_baselines, num_times,
            (const std::complex<double>*)sdp_mem_data_const(source_fluxes),
            (const double*)sdp_mem_data_const(source_directions),
            (const double*)sdp_mem_data_const(uvw_lambda),
            (const std::complex<double>*)sdp_mem_data_const(vis2),
            &status);
    sdp_mem_free(vis2);
#endif

    sdp_mem_free(source_fluxes);
    sdp_mem_free(source_directions);
    sdp_mem_free(uvw_lambda);
    return status ? EXIT_FAILURE : EXIT_SUCCESS;
}
