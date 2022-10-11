/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>

#include "ska-sdp-func/weighting/sdp_weighting.h"
#include "ska-sdp-func/utility/sdp_data_model_checks.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"

#define C_0 299792458.0
#define INDEX_3D(N3, N2, N1, I3, I2, I1)         (N1 * (N2 * I3 + I2) + I1)
#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)

template<typename UVW_TYPE, typename FREQ_TYPE, typename WEIGHT_TYPE>
static void uniform_weights_grid_write(
        const int64_t num_times,
        const int64_t num_baselines,
        const int64_t num_channels,
        const int64_t grid_size,
        const UVW_TYPE* uvw,
        const FREQ_TYPE* freq_hz,
        const double max_abs_uv,
        WEIGHT_TYPE* grid_uv
)
{
    const int64_t half_grid_size = grid_size / 2;
    for (int64_t i_time = 0; i_time < num_times; ++i_time)
    {
        for (int64_t i_baseline = 0; i_baseline < num_baselines; ++i_baseline)
        {
            const int64_t i_uv = INDEX_3D(
                    num_times, num_baselines, 3,
                    i_time, i_baseline, 0);
            for (int64_t i_channel = 0; i_channel < num_channels; ++i_channel)
            {
                const UVW_TYPE inv_wavelength = freq_hz[i_channel] / C_0;
                const UVW_TYPE grid_u = uvw[i_uv + 0] * inv_wavelength;
                const UVW_TYPE grid_v = uvw[i_uv + 1] * inv_wavelength;
                const int64_t idx_u = (int64_t) (grid_u / max_abs_uv * half_grid_size + half_grid_size);
                const int64_t idx_v = (int64_t) (grid_v / max_abs_uv * half_grid_size + half_grid_size);
                if (idx_u >= grid_size || idx_v >= grid_size) continue;
                grid_uv[idx_v * grid_size + idx_u] += 1.0;
            }
        }
    }
}

template<typename UVW_TYPE, typename FREQ_TYPE, typename WEIGHT_TYPE>
static void uniform_weights_grid_read(
        const int64_t num_times,
        const int64_t num_baselines,
        const int64_t num_channels,
        const int64_t num_pols,
        const int64_t grid_size,
        const UVW_TYPE* uvw,
        const FREQ_TYPE* freq_hz,
        const double max_abs_uv,
        const WEIGHT_TYPE* grid_uv,
        WEIGHT_TYPE* weights
)
{
    const int64_t half_grid_size = grid_size / 2;
    for (int64_t i_time = 0; i_time < num_times; ++i_time)
    {
        for (int64_t i_baseline = 0; i_baseline < num_baselines; ++i_baseline)
        {
            const int64_t i_uv = INDEX_3D(
                    num_times, num_baselines, 3,
                    i_time, i_baseline, 0);
            for (int64_t i_channel = 0; i_channel < num_channels; ++i_channel)
            {
                WEIGHT_TYPE weight_val = 1.0;
                const UVW_TYPE inv_wavelength = freq_hz[i_channel] / C_0;
                const UVW_TYPE grid_u = uvw[i_uv + 0] * inv_wavelength;
                const UVW_TYPE grid_v = uvw[i_uv + 1] * inv_wavelength;
                const int64_t idx_u = (int64_t) (grid_u / max_abs_uv * half_grid_size + half_grid_size);
                const int64_t idx_v = (int64_t) (grid_v / max_abs_uv * half_grid_size + half_grid_size);
                if (idx_u < grid_size && idx_v < grid_size)
                {
                    weight_val = 1.0 / grid_uv[idx_v * grid_size + idx_u];
                }

                const int64_t i_pol_start = INDEX_4D(
                        num_times, num_baselines, num_channels, num_pols,
                        i_time, i_baseline, i_channel, 0);
                for (int64_t i_pol = 0; i_pol < num_pols; ++i_pol)
                {
                    weights[i_pol_start + i_pol] = weight_val;
                }
            }
        }
    }
}

void sdp_weighting_uniform(
        const sdp_Mem* uvw,
        const sdp_Mem* freq_hz,
        double max_abs_uv,
        sdp_Mem* grid_uv,
        sdp_Mem* weights,
        sdp_Error* status)
{
    if (*status) return;
    sdp_MemType uvw_type = SDP_MEM_VOID;
    sdp_MemType weights_type = SDP_MEM_VOID;
    sdp_MemLocation uvw_location = SDP_MEM_CPU;
    sdp_MemLocation weights_location = SDP_MEM_CPU;
    int64_t num_times = 0;
    int64_t num_baselines = 0;
    int64_t num_channels = 0;
    int64_t num_pols = 0;
    int64_t grid_size = 0;

    // Check parameters.
    sdp_data_model_check_uvw(uvw, &uvw_type, &uvw_location, 0, 0, status);
    sdp_data_model_check_weights(weights, &weights_type, &weights_location,
            &num_times, &num_baselines, &num_channels, &num_pols, status);
    if (uvw_location != weights_location ||
            sdp_mem_location(freq_hz) != weights_location ||
            sdp_mem_location(grid_uv) != weights_location)
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Memory location mismatch");
        return;
    }
    if (sdp_mem_is_read_only(weights) || sdp_mem_is_read_only(grid_uv))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Weights and grid data must be writable");
        return;
    }
    grid_size = sdp_mem_shape_dim(grid_uv, 0);
    if (sdp_mem_shape_dim(grid_uv, 1) != grid_size)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Grid must be square");
        return;
    }

    // Call the appropriate version of the kernel.
    if (weights_location == SDP_MEM_CPU)
    {
        if (uvw_type == SDP_MEM_DOUBLE &&
                weights_type == SDP_MEM_DOUBLE &&
                sdp_mem_type(freq_hz) == SDP_MEM_DOUBLE &&
                sdp_mem_type(grid_uv) == SDP_MEM_DOUBLE)
        {
            uniform_weights_grid_write(
                num_times,
                num_baselines,
                num_channels,
                grid_size,
                (const double*)sdp_mem_data_const(uvw),
                (const double*)sdp_mem_data_const(freq_hz),
                max_abs_uv,
                (double*)sdp_mem_data(grid_uv)
            );
            uniform_weights_grid_read(
                num_times,
                num_baselines,
                num_channels,
                num_pols,
                grid_size,
                (const double*)sdp_mem_data_const(uvw),
                (const double*)sdp_mem_data_const(freq_hz),
                max_abs_uv,
                (const double*)sdp_mem_data_const(grid_uv),
                (double*)sdp_mem_data(weights)
            );
        }
        else if (uvw_type == SDP_MEM_DOUBLE &&
                weights_type == SDP_MEM_FLOAT &&
                sdp_mem_type(freq_hz) == SDP_MEM_DOUBLE &&
                sdp_mem_type(grid_uv) == SDP_MEM_FLOAT)
        {
            uniform_weights_grid_write(
                num_times,
                num_baselines,
                num_channels,
                grid_size,
                (const double*)sdp_mem_data_const(uvw),
                (const double*)sdp_mem_data_const(freq_hz),
                max_abs_uv,
                (float*)sdp_mem_data(grid_uv)
            );
            uniform_weights_grid_read(
                num_times,
                num_baselines,
                num_channels,
                num_pols,
                grid_size,
                (const double*)sdp_mem_data_const(uvw),
                (const double*)sdp_mem_data_const(freq_hz),
                max_abs_uv,
                (const float*)sdp_mem_data_const(grid_uv),
                (float*)sdp_mem_data(weights)
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
        }
    }
    else if (weights_location == SDP_MEM_GPU)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("A GPU version of the uniform weighting function "
                "is not currently implemented");
    }
}
