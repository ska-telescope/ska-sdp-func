#include <cmath>
#include <complex>
#include <iostream>

#include "ska-sdp-func/utility/sdp_data_model_checks.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/visibility/sdp_weighting.h"

#define C_0 299792458.0

#define INDEX_2D(N2, N1, I2, I1)(N1 * I2 + I1)
#define INDEX_3D(N3, N2, N1, I3, I2, I1)         (N1 * (N2 * I3 + I2) + I1)
#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) \
    (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)


// Generate grid of weights
template<typename UVW_TYPE, typename FREQ_TYPE, typename WEIGHT_TYPE>
static void weights_grid_write(
        const int64_t num_times,
        const int64_t num_baselines,
        const int64_t num_channels,
        const int64_t num_pols,
        const int64_t grid_size,
        const UVW_TYPE* uvw,
        const FREQ_TYPE* freq_hz,
        const double max_abs_uv,
        WEIGHT_TYPE* weights_grid_uv,
        const WEIGHT_TYPE* input_weight
)
{
    const int64_t half_grid_size = grid_size / 2;
    for (int64_t i_time = 0; i_time < num_times; ++i_time)
    {
        for (int64_t i_baseline = 0; i_baseline < num_baselines; ++i_baseline)
        {
            const int64_t i_uv = INDEX_3D(
                    num_times, num_baselines, 3,
                    i_time, i_baseline, 0
            );
            for (int64_t i_channel = 0; i_channel < num_channels; ++i_channel)
            {
                const UVW_TYPE inv_wavelength = freq_hz[i_channel] / C_0;
                const UVW_TYPE grid_u = uvw[i_uv + 0] * inv_wavelength;
                const UVW_TYPE grid_v = uvw[i_uv + 1] * inv_wavelength;
                const int64_t idx_u =
                        (int64_t) (floor(grid_u / max_abs_uv * half_grid_size) +
                        half_grid_size);
                const int64_t idx_v =
                        (int64_t) (floor(grid_v / max_abs_uv * half_grid_size) +
                        half_grid_size);
                if (idx_u >= grid_size || idx_v >= grid_size) continue;
                const int64_t i_pol_s = INDEX_3D(
                        grid_size,
                        grid_size,
                        num_pols,
                        idx_u,
                        idx_v,
                        0
                );
                const int64_t i_pol_start = INDEX_4D(
                        num_times, num_baselines, num_channels, num_pols,
                        i_time, i_baseline, i_channel, 0
                );
                for (int64_t i_pol = 0; i_pol < num_pols; i_pol++)
                {
                    weights_grid_uv[i_pol_s +
                            i_pol] += input_weight[i_pol_start + i_pol];
                }
            }
        }
    }
}


// Calculate the sum of weights and the sum of the gridded weights squared
template<typename UVW_TYPE, typename FREQ_TYPE, typename WEIGHT_TYPE>
static void sum_weights_calc(
        const int64_t num_times,
        const int64_t num_baselines,
        const int64_t num_channels,
        const int64_t num_pols,
        const int64_t grid_size,
        const UVW_TYPE* uvw,
        const FREQ_TYPE* freq_hz,
        const double max_abs_uv,
        const WEIGHT_TYPE* weights_grid_uv,
        double* sumweight,
        double* sumweight2
)
{
    const int64_t half_grid_size = grid_size / 2;
    for (int64_t i_time = 0; i_time < num_times; ++i_time)
    {
        for (int64_t i_baseline = 0; i_baseline < num_baselines; ++i_baseline)
        {
            const int64_t i_uv = INDEX_3D(
                    num_times, num_baselines, 3,
                    i_time, i_baseline, 0
            );
            for (int64_t i_channel = 0; i_channel < num_channels; ++i_channel)
            {
                const UVW_TYPE inv_wavelength = freq_hz[i_channel] / C_0;
                const UVW_TYPE grid_u = uvw[i_uv + 0] * inv_wavelength;
                const UVW_TYPE grid_v = uvw[i_uv + 1] * inv_wavelength;
                const int64_t idx_u =
                        (int64_t) (floor(grid_u / max_abs_uv * half_grid_size) +
                        half_grid_size);
                const int64_t idx_v =
                        (int64_t) (floor(grid_v / max_abs_uv * half_grid_size) +
                        half_grid_size);
                if (idx_u >= grid_size || idx_v >= grid_size) continue;
                const int64_t i_pol_s = INDEX_3D(
                        grid_size,
                        grid_size,
                        num_pols,
                        idx_u,
                        idx_v,
                        0
                );
                for (int64_t i_pol = 0; i_pol < num_pols; ++i_pol)
                {
                    *sumweight += weights_grid_uv[i_pol_s + i_pol];
                    *sumweight2 +=
                            (weights_grid_uv[i_pol_s + i_pol] *
                            weights_grid_uv[i_pol_s + i_pol]);
                }
            }
        }
    }
}


// Calculate the robustness function
static double robustness_calc(
        double sumweight2,
        double sumweight,
        const double robust_param
)
{
    double numerator = pow(5.0 * 1 / (pow(10.0, robust_param)), 2.0);
    double division_param = sumweight2 / sumweight;
    return numerator / division_param;
}


// Read from the grid of weights according to the enum type
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
        const WEIGHT_TYPE* weights_grid_uv,
        WEIGHT_TYPE* output_weights
)
{
    const int64_t half_grid_size = grid_size / 2;
    for (int64_t i_time = 0; i_time < num_times; ++i_time)
    {
        for (int64_t i_baseline = 0; i_baseline < num_baselines; ++i_baseline)
        {
            const int64_t i_uv = INDEX_3D(
                    num_times, num_baselines, 3,
                    i_time, i_baseline, 0
            );
            for (int64_t i_channel = 0; i_channel < num_channels; ++i_channel)
            {
                WEIGHT_TYPE weight_val = 1.0;
                const UVW_TYPE inv_wavelength = freq_hz[i_channel] / C_0;
                const UVW_TYPE grid_u = uvw[i_uv + 0] * inv_wavelength;
                const UVW_TYPE grid_v = uvw[i_uv + 1] * inv_wavelength;
                const int64_t idx_u =
                        (int64_t) (floor(grid_u / max_abs_uv * half_grid_size) +
                        half_grid_size);
                const int64_t idx_v =
                        (int64_t) (floor(grid_v / max_abs_uv * half_grid_size) +
                        half_grid_size);
                if (idx_u < grid_size && idx_v < grid_size)
                {
                    const int64_t i_pol_s = INDEX_3D(
                            grid_size,
                            grid_size,
                            num_pols,
                            idx_u,
                            idx_v,
                            0
                    );
                    const int64_t i_pol_start = INDEX_4D(
                            num_times, num_baselines, num_channels, num_pols,
                            i_time, i_baseline, i_channel, 0
                    );
                    for (int64_t i_pol = 0; i_pol < num_pols; ++i_pol)
                    {
                        weight_val = 1.0 / weights_grid_uv[i_pol_s + i_pol];
                        output_weights[i_pol_start + i_pol] = weight_val;
                    }
                }
            }
        }
    }
}


// Read from the grid of weights according to the enum type
template<typename UVW_TYPE, typename FREQ_TYPE, typename WEIGHT_TYPE>
static void briggs_weights_grid_read(
        const int64_t num_times,
        const int64_t num_baselines,
        const int64_t num_channels,
        const int64_t num_pols,
        const int64_t grid_size,
        const UVW_TYPE* uvw,
        const FREQ_TYPE* freq_hz,
        const double max_abs_uv,
        const WEIGHT_TYPE* weights_grid_uv,
        const WEIGHT_TYPE* input_weights,
        WEIGHT_TYPE* output_weights,
        double robustness
)
{
    const int64_t half_grid_size = grid_size / 2;
    for (int64_t i_time = 0; i_time < num_times; ++i_time)
    {
        for (int64_t i_baseline = 0; i_baseline < num_baselines; ++i_baseline)
        {
            const int64_t i_uv = INDEX_3D(
                    num_times, num_baselines, 3,
                    i_time, i_baseline, 0
            );
            for (int64_t i_channel = 0; i_channel < num_channels; ++i_channel)
            {
                WEIGHT_TYPE weight_val = 1.0;
                const UVW_TYPE inv_wavelength = freq_hz[i_channel] / C_0;
                const UVW_TYPE grid_u = uvw[i_uv + 0] * inv_wavelength;
                const UVW_TYPE grid_v = uvw[i_uv + 1] * inv_wavelength;
                const int64_t idx_u =
                        (int64_t) (floor(grid_u / max_abs_uv * half_grid_size) +
                        half_grid_size);
                const int64_t idx_v =
                        (int64_t) (floor(grid_v / max_abs_uv * half_grid_size) +
                        half_grid_size);
                if (idx_u < grid_size && idx_v < grid_size)
                {
                    const int64_t i_pol_s = INDEX_3D(
                            grid_size,
                            grid_size,
                            num_pols,
                            idx_u,
                            idx_v,
                            0
                    );
                    const int64_t i_pol_start = INDEX_4D(
                            num_times, num_baselines, num_channels, num_pols,
                            i_time, i_baseline, i_channel, 0
                    );
                    for (int64_t i_pol = 0; i_pol < num_pols; ++i_pol)
                    {
                        weight_val = input_weights[i_pol_start + i_pol] /
                                (1 +
                                (robustness *
                                weights_grid_uv[i_pol_s + i_pol]));
                        output_weights[i_pol_start + i_pol] = weight_val;
                    }
                }
            }
        }
    }
}


void sdp_weighting_uniform(
        const sdp_Mem* uvw,
        const sdp_Mem* freq_hz,
        double max_abs_uv,
        sdp_Mem* weights_grid_uv,
        sdp_Mem* input_weight,
        sdp_Mem* output_weight,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_MemType uvw_type = sdp_mem_type(uvw);
    sdp_MemType weights_type = SDP_MEM_VOID;
    sdp_MemLocation weights_location = sdp_mem_location(output_weight);
    int64_t num_times = 0;
    int64_t num_baselines = 0;
    int64_t num_channels = 0;
    int64_t num_pols = 0;

    // Check parameters.
    sdp_data_model_get_weights_metadata(input_weight,
            &weights_type,
            &weights_location,
            &num_times,
            &num_baselines,
            &num_channels,
            &num_pols,
            status
    );

    sdp_data_model_get_weights_metadata(output_weight,
            &weights_type,
            &weights_location,
            &num_times,
            &num_baselines,
            &num_channels,
            &num_pols,
            status
    );

    sdp_data_model_check_uvw(uvw,
            SDP_MEM_VOID,
            weights_location,
            num_times,
            num_baselines,
            status
    );

    sdp_mem_check_location(freq_hz, weights_location, status);
    sdp_mem_check_location(weights_grid_uv, weights_location, status);

    sdp_mem_check_writeable(output_weight, status);
    sdp_mem_check_writeable(weights_grid_uv, status);

    const int32_t num_grid_dims = 3;
    sdp_mem_check_num_dims(weights_grid_uv, num_grid_dims, status);
    int64_t grid_size = sdp_mem_shape_dim(weights_grid_uv, 0);
    // Checking that grid is square
    sdp_mem_check_dim_size(weights_grid_uv, 1, grid_size, status);
    if (*status) return;

    // Call the appropriate version of the kernel.
    if (weights_location == SDP_MEM_CPU)
    {
        if (uvw_type == SDP_MEM_DOUBLE &&
                weights_type == SDP_MEM_DOUBLE &&
                sdp_mem_type(freq_hz) == SDP_MEM_DOUBLE &&
                sdp_mem_type(weights_grid_uv) == SDP_MEM_DOUBLE)
        {
            weights_grid_write(
                    num_times,
                    num_baselines,
                    num_channels,
                    num_pols,
                    grid_size,
                    (const double*)sdp_mem_data_const(uvw),
                    (const double*)sdp_mem_data_const(freq_hz),
                    max_abs_uv,
                    (double*)sdp_mem_data(weights_grid_uv),
                    (const double*)sdp_mem_data_const(input_weight)
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
                    (const double*)sdp_mem_data_const(weights_grid_uv),
                    (double*)sdp_mem_data(output_weight)
            );
        }
        else if (uvw_type == SDP_MEM_DOUBLE &&
                weights_type == SDP_MEM_FLOAT &&
                sdp_mem_type(freq_hz) == SDP_MEM_DOUBLE &&
                sdp_mem_type(weights_grid_uv) == SDP_MEM_FLOAT)
        {
            weights_grid_write(
                    num_times,
                    num_baselines,
                    num_channels,
                    num_pols,
                    grid_size,
                    (const double*)sdp_mem_data_const(uvw),
                    (const double*)sdp_mem_data_const(freq_hz),
                    max_abs_uv,
                    (float*)sdp_mem_data(weights_grid_uv),
                    (const float*)sdp_mem_data_const(input_weight)
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
                    (const float*)sdp_mem_data_const(weights_grid_uv),
                    (float*)sdp_mem_data(output_weight)
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
        const uint64_t num_threads[] = {128, 2, 2};
        const uint64_t num_blocks[] = {
            (num_baselines + num_threads[0] - 1) / num_threads[0],
            (num_channels + num_threads[1] - 1) / num_threads[1],
            (num_times + num_threads[2] - 1) / num_threads[2]
        };
        const char* kernel_name = 0;
        if (uvw_type == SDP_MEM_DOUBLE &&
                weights_type == SDP_MEM_DOUBLE &&
                sdp_mem_type(freq_hz) == SDP_MEM_DOUBLE &&
                sdp_mem_type(weights_grid_uv) == SDP_MEM_DOUBLE)
        {
            kernel_name = "grid_write_gpu<double, double, double>";
        }
        else if (uvw_type == SDP_MEM_DOUBLE &&
                weights_type == SDP_MEM_FLOAT &&
                sdp_mem_type(freq_hz) == SDP_MEM_DOUBLE &&
                sdp_mem_type(weights_grid_uv) == SDP_MEM_FLOAT)
        {
            kernel_name = "grid_write_gpu<double, double, float>";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
        }

        const void* args[]{
            (const void*)&num_times,
            (const void*)&num_baselines,
            (const void*)&num_channels,
            (const void*)&num_pols,
            (const void*)&grid_size,
            sdp_mem_gpu_buffer_const(uvw, status),
            sdp_mem_gpu_buffer_const(freq_hz, status),
            (const void*)&max_abs_uv,
            sdp_mem_gpu_buffer(weights_grid_uv, status),
            sdp_mem_gpu_buffer_const(input_weight, status),
        };

        sdp_launch_cuda_kernel(kernel_name,
                num_blocks,
                num_threads,
                0,
                0,
                args,
                status
        );

        const char* kernel_name2 = 0;
        if (uvw_type == SDP_MEM_DOUBLE &&
                weights_type == SDP_MEM_DOUBLE &&
                sdp_mem_type(freq_hz) == SDP_MEM_DOUBLE &&
                sdp_mem_type(weights_grid_uv) == SDP_MEM_DOUBLE)
        {
            kernel_name2 = "grid_uniform_read_gpu<double, double, double>";
        }
        else if (uvw_type == SDP_MEM_DOUBLE &&
                weights_type == SDP_MEM_FLOAT &&
                sdp_mem_type(freq_hz) == SDP_MEM_DOUBLE &&
                sdp_mem_type(weights_grid_uv) == SDP_MEM_FLOAT)
        {
            kernel_name2 = "grid_uniform_read_gpu<double, double, float>";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
        }

        const void* args2[]{
            (const void*)&num_times,
            (const void*)&num_baselines,
            (const void*)&num_channels,
            (const void*)&num_pols,
            (const void*)&grid_size,
            sdp_mem_gpu_buffer_const(uvw, status),
            sdp_mem_gpu_buffer_const(freq_hz, status),
            (const void*)&max_abs_uv,
            sdp_mem_gpu_buffer_const(weights_grid_uv, status),
            sdp_mem_gpu_buffer_const(input_weight, status),
            sdp_mem_gpu_buffer(output_weight, status)
        };

        sdp_launch_cuda_kernel(kernel_name2,
                num_blocks,
                num_threads,
                0,
                0,
                args2,
                status
        );
    }
}


void sdp_weighting_briggs(
        const sdp_Mem* uvw,
        const sdp_Mem* freq_hz,
        double max_abs_uv,
        const double robust_param,
        sdp_Mem* weights_grid_uv,
        sdp_Mem* input_weight,
        sdp_Mem* output_weight,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_MemType uvw_type = sdp_mem_type(uvw);
    sdp_MemType weights_type = SDP_MEM_VOID;
    sdp_MemLocation weights_location = sdp_mem_location(output_weight);
    double sumweight = 0;
    double sumweight2 = 0;
    int64_t num_times = 0;
    int64_t num_baselines = 0;
    int64_t num_channels = 0;
    int64_t num_pols = 0;

    // Check parameters.
    sdp_data_model_get_weights_metadata(input_weight,
            &weights_type,
            &weights_location,
            &num_times,
            &num_baselines,
            &num_channels,
            &num_pols,
            status
    );

    sdp_data_model_get_weights_metadata(output_weight,
            &weights_type,
            &weights_location,
            &num_times,
            &num_baselines,
            &num_channels,
            &num_pols,
            status
    );

    sdp_data_model_check_uvw(uvw,
            SDP_MEM_VOID,
            weights_location,
            num_times,
            num_baselines,
            status
    );

    sdp_mem_check_location(freq_hz, weights_location, status);
    sdp_mem_check_location(weights_grid_uv, weights_location, status);

    sdp_mem_check_writeable(output_weight, status);
    sdp_mem_check_writeable(weights_grid_uv, status);

    const int32_t num_grid_dims = 3;
    sdp_mem_check_num_dims(weights_grid_uv, num_grid_dims, status);
    int64_t grid_size = sdp_mem_shape_dim(weights_grid_uv, 0);
    // Checking that grid is square
    sdp_mem_check_dim_size(weights_grid_uv, 1, grid_size, status);
    if (*status) return;

    // Call the appropriate version of the kernel.
    if (weights_location == SDP_MEM_CPU)
    {
        if (uvw_type == SDP_MEM_DOUBLE &&
                weights_type == SDP_MEM_DOUBLE &&
                sdp_mem_type(freq_hz) == SDP_MEM_DOUBLE &&
                sdp_mem_type(weights_grid_uv) == SDP_MEM_DOUBLE)
        {
            weights_grid_write(
                    num_times,
                    num_baselines,
                    num_channels,
                    num_pols,
                    grid_size,
                    (const double*)sdp_mem_data_const(uvw),
                    (const double*)sdp_mem_data_const(freq_hz),
                    max_abs_uv,
                    (double*)sdp_mem_data(weights_grid_uv),
                    (const double*)sdp_mem_data_const(input_weight)
            );
            sum_weights_calc(
                    num_times,
                    num_baselines,
                    num_channels,
                    num_pols,
                    grid_size,
                    (const double*)sdp_mem_data_const(uvw),
                    (const double*)sdp_mem_data_const(freq_hz),
                    max_abs_uv,
                    (const double*)sdp_mem_data(weights_grid_uv),
                    &sumweight,
                    &sumweight2
            );

            double robustness = robustness_calc(
                    sumweight2,
                    sumweight,
                    robust_param
            );

            briggs_weights_grid_read(
                    num_times,
                    num_baselines,
                    num_channels,
                    num_pols,
                    grid_size,
                    (const double*)sdp_mem_data_const(uvw),
                    (const double*)sdp_mem_data_const(freq_hz),
                    max_abs_uv,
                    (const double*)sdp_mem_data_const(weights_grid_uv),
                    (const double*)sdp_mem_data_const(input_weight),
                    (double*)sdp_mem_data(output_weight),
                    robustness
            );
        }
        else if (uvw_type == SDP_MEM_DOUBLE &&
                weights_type == SDP_MEM_FLOAT &&
                sdp_mem_type(freq_hz) == SDP_MEM_DOUBLE &&
                sdp_mem_type(weights_grid_uv) == SDP_MEM_FLOAT)
        {
            weights_grid_write(
                    num_times,
                    num_baselines,
                    num_channels,
                    num_pols,
                    grid_size,
                    (const double*)sdp_mem_data_const(uvw),
                    (const double*)sdp_mem_data_const(freq_hz),
                    max_abs_uv,
                    (float*)sdp_mem_data(weights_grid_uv),
                    (const float*)sdp_mem_data_const(input_weight)
            );
            sum_weights_calc(
                    num_times,
                    num_baselines,
                    num_channels,
                    num_pols,
                    grid_size,
                    (const double*)sdp_mem_data_const(uvw),
                    (const double*)sdp_mem_data_const(freq_hz),
                    max_abs_uv,
                    (const float*)sdp_mem_data(weights_grid_uv),
                    &sumweight,
                    &sumweight2
            );
            double robustness = robustness_calc(
                    sumweight2,
                    sumweight,
                    robust_param
            );
            briggs_weights_grid_read(
                    num_times,
                    num_baselines,
                    num_channels,
                    num_pols,
                    grid_size,
                    (const double*)sdp_mem_data_const(uvw),
                    (const double*)sdp_mem_data_const(freq_hz),
                    max_abs_uv,
                    (const float*)sdp_mem_data_const(weights_grid_uv),
                    (const float*)sdp_mem_data_const(input_weight),
                    (float*)sdp_mem_data(output_weight),
                    robustness
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
        const uint64_t num_threads[] = {128, 2, 2};
        const uint64_t num_blocks[] = {
            (num_baselines + num_threads[0] - 1) / num_threads[0],
            (num_channels + num_threads[1] - 1) / num_threads[1],
            (num_times + num_threads[2] - 1) / num_threads[2]
        };
        const char* kernel_name = 0;
        if (uvw_type == SDP_MEM_DOUBLE &&
                weights_type == SDP_MEM_DOUBLE &&
                sdp_mem_type(freq_hz) == SDP_MEM_DOUBLE &&
                sdp_mem_type(weights_grid_uv) == SDP_MEM_DOUBLE)
        {
            kernel_name = "grid_write_gpu<double, double, double>";
        }
        else if (uvw_type == SDP_MEM_DOUBLE &&
                weights_type == SDP_MEM_FLOAT &&
                sdp_mem_type(freq_hz) == SDP_MEM_DOUBLE &&
                sdp_mem_type(weights_grid_uv) == SDP_MEM_FLOAT)
        {
            kernel_name = "grid_write_gpu<double, double, float>";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
        }

        const void* args[]{
            (const void*)&num_times,
            (const void*)&num_baselines,
            (const void*)&num_channels,
            (const void*)&num_pols,
            (const void*)&grid_size,
            sdp_mem_gpu_buffer_const(uvw, status),
            sdp_mem_gpu_buffer_const(freq_hz, status),
            (const void*)&max_abs_uv,
            sdp_mem_gpu_buffer(weights_grid_uv, status),
            sdp_mem_gpu_buffer_const(input_weight, status),
        };

        sdp_launch_cuda_kernel(kernel_name,
                num_blocks,
                num_threads,
                0,
                0,
                args,
                status
        );

        int64_t sumweight_shape[] = {1};
        int64_t sumweight2_shape[] = {1};
        sdp_Mem* sumweight_gpu = sdp_mem_create(SDP_MEM_DOUBLE,
                SDP_MEM_GPU,
                1,
                sumweight_shape,
                status
        );
        sdp_Mem* sumweight2_gpu = sdp_mem_create(SDP_MEM_DOUBLE,
                SDP_MEM_GPU,
                1,
                sumweight2_shape,
                status
        );

        const char* kernel_name1 = 0;
        if (uvw_type == SDP_MEM_DOUBLE &&
                weights_type == SDP_MEM_DOUBLE &&
                sdp_mem_type(freq_hz) == SDP_MEM_DOUBLE &&
                sdp_mem_type(weights_grid_uv) == SDP_MEM_DOUBLE)
        {
            kernel_name1 = "calc_sum_gpu<double, double, double>";
        }
        else if (uvw_type == SDP_MEM_DOUBLE &&
                weights_type == SDP_MEM_FLOAT &&
                sdp_mem_type(freq_hz) == SDP_MEM_DOUBLE &&
                sdp_mem_type(weights_grid_uv) == SDP_MEM_FLOAT)
        {
            kernel_name1 = "calc_sum_gpu<double, double, float>";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
        }

        const void* args1[]{
            (const void*)&num_times,
            (const void*)&num_baselines,
            (const void*)&num_channels,
            (const void*)&num_pols,
            (const void*)&grid_size,
            sdp_mem_gpu_buffer_const(uvw, status),
            sdp_mem_gpu_buffer_const(freq_hz, status),
            (const void*)&max_abs_uv,
            sdp_mem_gpu_buffer_const(weights_grid_uv, status),
            sdp_mem_gpu_buffer_const(input_weight, status),
            sdp_mem_gpu_buffer(sumweight_gpu, status),
            sdp_mem_gpu_buffer(sumweight2_gpu, status)
        };

        sdp_launch_cuda_kernel(kernel_name1,
                num_blocks,
                num_threads,
                0,
                0,
                args1,
                status
        );

        // copy back from gpu sumweight and sumweight2
        sdp_Mem* sumweight2_gpu_cpy = sdp_mem_create_copy(sumweight2_gpu,
                SDP_MEM_CPU,
                status
        );
        sdp_Mem* sumweight_gpu_cpy = sdp_mem_create_copy(sumweight_gpu,
                SDP_MEM_CPU,
                status
        );

        double robustness = robustness_calc(
                *((double*)sdp_mem_data(sumweight2_gpu_cpy)),
                *((double*)sdp_mem_data(sumweight_gpu_cpy)),
                robust_param
        );

        sdp_mem_ref_dec(sumweight2_gpu);
        sdp_mem_ref_dec(sumweight_gpu);
        sdp_mem_ref_dec(sumweight_gpu_cpy);
        sdp_mem_ref_dec(sumweight2_gpu_cpy);

        const char* kernel_name2 = 0;
        if (uvw_type == SDP_MEM_DOUBLE &&
                weights_type == SDP_MEM_DOUBLE &&
                sdp_mem_type(freq_hz) == SDP_MEM_DOUBLE &&
                sdp_mem_type(weights_grid_uv) == SDP_MEM_DOUBLE)
        {
            kernel_name2 = "grid_briggs_read_gpu<double, double, double>";
        }
        else if (uvw_type == SDP_MEM_DOUBLE &&
                weights_type == SDP_MEM_FLOAT &&
                sdp_mem_type(freq_hz) == SDP_MEM_DOUBLE &&
                sdp_mem_type(weights_grid_uv) == SDP_MEM_FLOAT)
        {
            kernel_name2 = "grid_briggs_read_gpu<double, double, float>";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
        }

        const void* args2[]{
            (const void*)&num_times,
            (const void*)&num_baselines,
            (const void*)&num_channels,
            (const void*)&num_pols,
            (const void*)&grid_size,
            sdp_mem_gpu_buffer_const(uvw, status),
            sdp_mem_gpu_buffer_const(freq_hz, status),
            (const void*)&max_abs_uv,
            sdp_mem_gpu_buffer_const(weights_grid_uv, status),
            sdp_mem_gpu_buffer_const(input_weight, status),
            sdp_mem_gpu_buffer(output_weight, status),
            (const void*)&robustness
        };

        sdp_launch_cuda_kernel(kernel_name2,
                num_blocks,
                num_threads,
                0,
                0,
                args2,
                status
        );
    }
}
