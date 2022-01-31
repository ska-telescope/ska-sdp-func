/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>

#include "dft/sdp_dft.h"
#include "logging/sdp_logging.h"
#include "utility/sdp_device_wrapper.h"

#define INDEX_3D(N3, N2, N1, I3, I2, I1)         (N1 * (N2 * I3 + I2) + I1)
#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)

template<
        typename FLUX_TYPE,
        typename DIR_TYPE,
        typename UVW_TYPE,
        typename VIS_TYPE
>
void dft_point_v00(
        const int num_components,
        const int num_pols,
        const int num_channels,
        const int num_baselines,
        const int num_times,
        const std::complex<FLUX_TYPE> *const __restrict__ source_fluxes,
        const DIR_TYPE *const __restrict__ source_directions,
        const UVW_TYPE *const __restrict__ uvw_lambda,
        std::complex<VIS_TYPE> *__restrict__ vis
)
{
    for (int i_time = 0; i_time < num_times; ++i_time)
    {
        for (int i_channel = 0; i_channel < num_channels; ++i_channel)
        {
            #pragma omp parallel for
            for (int i_baseline = 0; i_baseline < num_baselines; ++i_baseline)
            {
                // Local visibility. Allow up to 4 polarisations.
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

                // Write out local visibility.
                for (int i_pol = 0; i_pol < num_pols; ++i_pol)
                {
                    const unsigned int i_out = INDEX_4D(
                            num_times, num_baselines, num_channels, num_pols,
                            i_time, i_baseline, i_channel, i_pol);
                    vis[i_out] = vis_local[i_pol];
                }
            }
        }
    }
}


void sdp_dft_point_v00(
        int num_components,
        int num_pols,
        int num_channels,
        int num_baselines,
        int num_times,
        const sdp_Mem* source_fluxes,
        const sdp_Mem* source_directions,
        const sdp_Mem* uvw_lambda,
        sdp_Mem* vis,
        sdp_Error* status
)
{
    if (*status) return;
    const sdp_MemLocation location = sdp_mem_location(vis);
    if (sdp_mem_location(source_fluxes) != location ||
            sdp_mem_location(source_directions) != location ||
            sdp_mem_location(uvw_lambda) != location)
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Memory location mismatch");
        return;
    }
    if (num_pols != 4 && num_pols != 1)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Number of polarisations should be 4 or 1");
        return;
    }
    if (location == SDP_MEM_CPU)
    {
        if (sdp_mem_type(source_fluxes) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(source_directions) == SDP_MEM_DOUBLE &&
                sdp_mem_type(uvw_lambda) == SDP_MEM_DOUBLE &&
                sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE)
        {
            dft_point_v00(
                num_components,
                num_pols,
                num_channels,
                num_baselines,
                num_times,
                (const std::complex<double>*)sdp_mem_data_const(source_fluxes),
                (const double*)sdp_mem_data_const(source_directions),
                (const double*)sdp_mem_data_const(uvw_lambda),
                (std::complex<double>*)sdp_mem_data(vis)
            );
        }
        else if (sdp_mem_type(source_fluxes) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(source_directions) == SDP_MEM_DOUBLE &&
                sdp_mem_type(uvw_lambda) == SDP_MEM_DOUBLE &&
                sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT)
        {
            dft_point_v00(
                num_components,
                num_pols,
                num_channels,
                num_baselines,
                num_times,
                (const std::complex<double>*)sdp_mem_data_const(source_fluxes),
                (const double*)sdp_mem_data_const(source_directions),
                (const double*)sdp_mem_data_const(uvw_lambda),
                (std::complex<float>*)sdp_mem_data(vis)
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
        }
    }
    else
    {
        const size_t num_threads[] = {128, 2, 2};
        const size_t num_blocks[] = {
            (num_baselines + num_threads[0] - 1) / num_threads[0],
            (num_channels + num_threads[1] - 1) / num_threads[1],
            (num_times + num_threads[2] - 1) / num_threads[2]
        };
        const char* kernel_name = 0;
        if (sdp_mem_type(source_fluxes) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(source_directions) == SDP_MEM_DOUBLE &&
                sdp_mem_type(uvw_lambda) == SDP_MEM_DOUBLE &&
                sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE)
        {
            kernel_name = "dft_point_v00<double2, double3, double3, double2>";
        }
        else if (sdp_mem_type(source_fluxes) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(source_directions) == SDP_MEM_DOUBLE &&
                sdp_mem_type(uvw_lambda) == SDP_MEM_DOUBLE &&
                sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT)
        {
            kernel_name = "dft_point_v00<double2, double3, double3, float2>";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
        }
        const void* args[] = {
                &num_components,
                &num_pols,
                &num_channels,
                &num_baselines,
                &num_times,
                sdp_mem_gpu_buffer_const(source_fluxes, status),
                sdp_mem_gpu_buffer_const(source_directions, status),
                sdp_mem_gpu_buffer_const(uvw_lambda, status),
                sdp_mem_gpu_buffer(vis, status)
        };
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, 0, 0, args, status);
    }
}
