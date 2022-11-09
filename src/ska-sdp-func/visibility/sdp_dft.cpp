/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>

#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"
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
static void dft_point_v00(
        const int num_components,
        const int num_pols,
        const int num_channels,
        const int num_baselines,
        const int num_times,
        const DIR_TYPE* const __restrict__ source_directions,
        const complex<FLUX_TYPE>* const __restrict__ source_fluxes,
        const UVW_TYPE* const __restrict__ uvw_lambda,
        complex<VIS_TYPE>* __restrict__ vis
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

                // Write out local visibility.
                for (int i_pol = 0; i_pol < num_pols; ++i_pol)
                {
                    const unsigned int i_out = INDEX_4D(
                            num_times, num_baselines, num_channels, num_pols,
                            i_time, i_baseline, i_channel, i_pol
                    );
                    vis[i_out] = vis_local[i_pol];
                }
            }
        }
    }
}


static void check_params_v00(
        const sdp_Mem* source_directions,
        const sdp_Mem* source_fluxes,
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
    if (sdp_mem_is_read_only(vis))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Visibility data must be writable.");
        return;
    }
    if (!sdp_mem_is_c_contiguous(source_fluxes) ||
            !sdp_mem_is_c_contiguous(source_directions) ||
            !sdp_mem_is_c_contiguous(uvw_lambda) ||
            !sdp_mem_is_c_contiguous(vis))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("All arrays must be C contiguous");
        return;
    }
    if (!sdp_mem_is_complex(source_fluxes))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Source flux values must be complex");
        return;
    }
    if (!sdp_mem_is_complex(vis))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Visibility values must be complex");
        return;
    }
    const int64_t num_times      = sdp_mem_shape_dim(vis, 0);
    const int64_t num_baselines  = sdp_mem_shape_dim(vis, 1);
    const int64_t num_channels   = sdp_mem_shape_dim(vis, 2);
    const int64_t num_pols       = sdp_mem_shape_dim(vis, 3);
    const int64_t num_components = sdp_mem_shape_dim(source_directions, 0);
    if (num_pols != 4 && num_pols != 1)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The number of polarisations should be 4 or 1");
        return;
    }
    if (sdp_mem_shape_dim(source_directions, 1) != 3)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The source_directions array must have shape "
                "[num_components, 3] (expected [%d, 3])", num_components
        );
        return;
    }
    if (sdp_mem_shape_dim(source_fluxes, 0) != num_components ||
            sdp_mem_shape_dim(source_fluxes, 1) != num_channels ||
            sdp_mem_shape_dim(source_fluxes, 2) != num_pols)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The source_fluxes array must have shape "
                "[num_components, num_channels, num_pols] "
                "(expected [%d, %d, %d])",
                num_components, num_channels, num_pols
        );
        return;
    }
    if (sdp_mem_shape_dim(uvw_lambda, 0) != num_times ||
            sdp_mem_shape_dim(uvw_lambda, 1) != num_baselines ||
            sdp_mem_shape_dim(uvw_lambda, 2) != num_channels ||
            sdp_mem_shape_dim(uvw_lambda, 3) != 3)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The uvw_lamda array must have shape "
                "[num_times, num_baselines, num_channels, 3] "
                "(expected [%d, %d, %d, 3])",
                num_times, num_baselines, num_channels
        );
        return;
    }
}


void sdp_dft_point_v00(
        const sdp_Mem* source_directions,
        const sdp_Mem* source_fluxes,
        const sdp_Mem* uvw_lambda,
        sdp_Mem* vis,
        sdp_Error* status
)
{
    check_params_v00(source_directions, source_fluxes, uvw_lambda, vis, status);
    if (*status) return;
    const int num_times      = (int)sdp_mem_shape_dim(vis, 0);
    const int num_baselines  = (int)sdp_mem_shape_dim(vis, 1);
    const int num_channels   = (int)sdp_mem_shape_dim(vis, 2);
    const int num_pols       = (int)sdp_mem_shape_dim(vis, 3);
    const int num_components = (int)sdp_mem_shape_dim(source_directions, 0);
    if (sdp_mem_location(vis) == SDP_MEM_CPU)
    {
        if (sdp_mem_type(source_directions) == SDP_MEM_DOUBLE &&
                sdp_mem_type(source_fluxes) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(uvw_lambda) == SDP_MEM_DOUBLE &&
                sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE)
        {
            dft_point_v00(
                    num_components,
                    num_pols,
                    num_channels,
                    num_baselines,
                    num_times,
                    (const double*)sdp_mem_data_const(source_directions),
                    (const complex<double>*)sdp_mem_data_const(source_fluxes),
                    (const double*)sdp_mem_data_const(uvw_lambda),
                    (complex<double>*)sdp_mem_data(vis)
            );
        }
        else if (sdp_mem_type(source_directions) == SDP_MEM_DOUBLE &&
                sdp_mem_type(source_fluxes) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(uvw_lambda) == SDP_MEM_DOUBLE &&
                sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT)
        {
            dft_point_v00(
                    num_components,
                    num_pols,
                    num_channels,
                    num_baselines,
                    num_times,
                    (const double*)sdp_mem_data_const(source_directions),
                    (const complex<double>*)sdp_mem_data_const(source_fluxes),
                    (const double*)sdp_mem_data_const(uvw_lambda),
                    (complex<float>*)sdp_mem_data(vis)
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
        }
    }
    else if (sdp_mem_location(vis) == SDP_MEM_GPU)
    {
        const uint64_t num_threads[] = {128, 2, 2};
        const uint64_t num_blocks[] = {
            (num_baselines + num_threads[0] - 1) / num_threads[0],
            (num_channels + num_threads[1] - 1) / num_threads[1],
            (num_times + num_threads[2] - 1) / num_threads[2]
        };
        const char* kernel_name = 0;
        if (sdp_mem_type(source_directions) == SDP_MEM_DOUBLE &&
                sdp_mem_type(source_fluxes) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(uvw_lambda) == SDP_MEM_DOUBLE &&
                sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE)
        {
            kernel_name = "dft_point_v00<double3, double2, double3, double2>";
        }
        else if (sdp_mem_type(source_directions) == SDP_MEM_DOUBLE &&
                sdp_mem_type(source_fluxes) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(uvw_lambda) == SDP_MEM_DOUBLE &&
                sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT)
        {
            kernel_name = "dft_point_v00<double3, double2, double3, float2>";
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
            sdp_mem_gpu_buffer_const(source_directions, status),
            sdp_mem_gpu_buffer_const(source_fluxes, status),
            sdp_mem_gpu_buffer_const(uvw_lambda, status),
            sdp_mem_gpu_buffer(vis, status)
        };
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, 0, 0, args, status
        );
    }
}


template<
        typename DIR_TYPE,
        typename FLUX_TYPE,
        typename UVW_TYPE,
        typename VIS_TYPE
>
static void dft_point_v01(
        const int num_components,
        const int num_pols,
        const int num_channels,
        const int num_baselines,
        const int num_times,
        const DIR_TYPE* const __restrict__ source_directions,
        const complex<FLUX_TYPE>* const __restrict__ source_fluxes,
        const UVW_TYPE* const __restrict__ uvw_metres,
        const double channel_start_hz,
        const double channel_step_hz,
        complex<VIS_TYPE>* __restrict__ vis
)
{
    for (int i_time = 0; i_time < num_times; ++i_time)
    {
        #pragma omp parallel for
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
                // Local visibility. Allow up to 4 polarisations.
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
                    const complex<VIS_TYPE> phasor(cos(phase), sin(phase));

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

                // Write out local visibility.
                for (int i_pol = 0; i_pol < num_pols; ++i_pol)
                {
                    const unsigned int i_out = INDEX_4D(
                            num_times, num_baselines, num_channels, num_pols,
                            i_time, i_baseline, i_channel, i_pol
                    );
                    vis[i_out] = vis_local[i_pol];
                }
            }
        }
    }
}


static void check_params_v01(
        const sdp_Mem* source_directions,
        const sdp_Mem* source_fluxes,
        const sdp_Mem* uvw,
        sdp_Mem* vis,
        sdp_Error* status
)
{
    if (*status) return;
    const sdp_MemLocation location = sdp_mem_location(vis);
    if (sdp_mem_location(source_fluxes) != location ||
            sdp_mem_location(source_directions) != location ||
            sdp_mem_location(uvw) != location)
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Memory location mismatch");
        return;
    }
    if (sdp_mem_is_read_only(vis))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Visibility data must be writable.");
        return;
    }
    if (!sdp_mem_is_c_contiguous(source_fluxes) ||
            !sdp_mem_is_c_contiguous(source_directions) ||
            !sdp_mem_is_c_contiguous(uvw) ||
            !sdp_mem_is_c_contiguous(vis))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("All arrays must be C contiguous");
        return;
    }
    if (!sdp_mem_is_complex(source_fluxes))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Source flux values must be complex");
        return;
    }
    if (!sdp_mem_is_complex(vis))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Visibility values must be complex");
        return;
    }
    const int64_t num_times      = sdp_mem_shape_dim(vis, 0);
    const int64_t num_baselines  = sdp_mem_shape_dim(vis, 1);
    const int64_t num_channels   = sdp_mem_shape_dim(vis, 2);
    const int64_t num_pols       = sdp_mem_shape_dim(vis, 3);
    const int64_t num_components = sdp_mem_shape_dim(source_directions, 0);
    if (num_pols != 4 && num_pols != 1)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The number of polarisations should be 4 or 1");
        return;
    }
    if (sdp_mem_shape_dim(source_directions, 1) != 3)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The source_directions array must have shape "
                "[num_components, 3] (expected [%d, 3])", num_components
        );
        return;
    }
    if (sdp_mem_shape_dim(source_fluxes, 0) != num_components ||
            sdp_mem_shape_dim(source_fluxes, 1) != num_channels ||
            sdp_mem_shape_dim(source_fluxes, 2) != num_pols)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The source_fluxes array must have shape "
                "[num_components, num_channels, num_pols] "
                "(expected [%d, %d, %d])",
                num_components, num_channels, num_pols
        );
        return;
    }
    if (sdp_mem_shape_dim(uvw, 0) != num_times ||
            sdp_mem_shape_dim(uvw, 1) != num_baselines ||
            sdp_mem_shape_dim(uvw, 2) != 3)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The uvw array must have shape "
                "[num_times, num_baselines, 3] "
                "(expected [%d, %d, 3])",
                num_times, num_baselines
        );
        return;
    }
}


void sdp_dft_point_v01(
        const sdp_Mem* source_directions,
        const sdp_Mem* source_fluxes,
        const sdp_Mem* uvw,
        const double channel_start_hz,
        const double channel_step_hz,
        sdp_Mem* vis,
        sdp_Error* status
)
{
    check_params_v01(source_directions, source_fluxes, uvw, vis, status);
    if (*status) return;
    const int num_times      = (int)sdp_mem_shape_dim(vis, 0);
    const int num_baselines  = (int)sdp_mem_shape_dim(vis, 1);
    const int num_channels   = (int)sdp_mem_shape_dim(vis, 2);
    const int num_pols       = (int)sdp_mem_shape_dim(vis, 3);
    const int num_components = (int)sdp_mem_shape_dim(source_directions, 0);
    if (sdp_mem_location(vis) == SDP_MEM_CPU)
    {
        if (sdp_mem_type(source_directions) == SDP_MEM_DOUBLE &&
                sdp_mem_type(source_fluxes) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(uvw) == SDP_MEM_DOUBLE &&
                sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE)
        {
            dft_point_v01(
                    num_components,
                    num_pols,
                    num_channels,
                    num_baselines,
                    num_times,
                    (const double*)sdp_mem_data_const(source_directions),
                    (const complex<double>*)sdp_mem_data_const(source_fluxes),
                    (const double*)sdp_mem_data_const(uvw),
                    channel_start_hz,
                    channel_step_hz,
                    (complex<double>*)sdp_mem_data(vis)
            );
        }
        else if (sdp_mem_type(source_directions) == SDP_MEM_DOUBLE &&
                sdp_mem_type(source_fluxes) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(uvw) == SDP_MEM_DOUBLE &&
                sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT)
        {
            dft_point_v01(
                    num_components,
                    num_pols,
                    num_channels,
                    num_baselines,
                    num_times,
                    (const double*)sdp_mem_data_const(source_directions),
                    (const complex<double>*)sdp_mem_data_const(source_fluxes),
                    (const double*)sdp_mem_data_const(uvw),
                    channel_start_hz,
                    channel_step_hz,
                    (complex<float>*)sdp_mem_data(vis)
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
        }
    }
    else if (sdp_mem_location(vis) == SDP_MEM_GPU)
    {
        const uint64_t num_threads[] = {128, 2, 2};
        const uint64_t num_blocks[] = {
            (num_baselines + num_threads[0] - 1) / num_threads[0],
            (num_channels + num_threads[1] - 1) / num_threads[1],
            (num_times + num_threads[2] - 1) / num_threads[2]
        };
        const char* kernel_name = 0;
        if (sdp_mem_type(source_directions) == SDP_MEM_DOUBLE &&
                sdp_mem_type(source_fluxes) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(uvw) == SDP_MEM_DOUBLE &&
                sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE)
        {
            kernel_name = "dft_point_v01<double3, double2, double3, double2>";
        }
        else if (sdp_mem_type(source_directions) == SDP_MEM_DOUBLE &&
                sdp_mem_type(source_fluxes) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(uvw) == SDP_MEM_DOUBLE &&
                sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT)
        {
            kernel_name = "dft_point_v01<double3, double2, double3, float2>";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
        }
        const void* args[] = {
            (const void*)&num_components,
            (const void*)&num_pols,
            (const void*)&num_channels,
            (const void*)&num_baselines,
            (const void*)&num_times,
            sdp_mem_gpu_buffer_const(source_directions, status),
            sdp_mem_gpu_buffer_const(source_fluxes, status),
            sdp_mem_gpu_buffer_const(uvw, status),
            (const void*)&channel_start_hz,
            (const void*)&channel_step_hz,
            sdp_mem_gpu_buffer(vis, status)
        };
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, 0, 0, args, status
        );
    }
}
