#include <cmath>
#include <stdio.h>
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/visibility/sdp_weighting.h"
#define C_0 299792458.0

#define INDEX_2D(N2, N1, I2, I1)(N1 * I2 + I1)
#define INDEX_3D(N3, N2, N1, I3, I2, I1)         (N1 * (N2 * I3 + I2) + I1)
#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) \
    (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)


// The following kernel is not fully optimized to write the weights


// Currently it uses atomic operators for avoiding race conditions


template<typename UVW_TYPE, typename FREQ_TYPE, typename WEIGHT_TYPE>
__global__ void grid_write_gpu(
        const int num_times,
        const int num_baselines,
        const int num_channels,
        const int num_pols,
        const int grid_size,
        const UVW_TYPE* const __restrict__ uvw,
        const FREQ_TYPE* const __restrict__ freq_hz,
        const double max_abs_uv,
        WEIGHT_TYPE* __restrict__ weights_grid_uv,
        const WEIGHT_TYPE* const __restrict__ input_weight
)
{
    const int i_baseline = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_channel = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_time = blockDim.z * blockIdx.z + threadIdx.z;
    const int half_grid_size = grid_size / 2;

    // Bounds Check
    if (i_baseline >= num_baselines ||
            i_channel >= num_channels ||
            i_time >= num_times)
    {
        return;
    }

    const int i_uv = INDEX_3D(
            num_times, num_baselines, 3,
            i_time, i_baseline, 0
    );

    const UVW_TYPE inv_wavelength = freq_hz[i_channel] / C_0;
    const UVW_TYPE grid_u = uvw[i_uv + 0] * inv_wavelength;
    const UVW_TYPE grid_v = uvw[i_uv + 1] * inv_wavelength;

    const int idx_u =
            (int)(floor(grid_u / max_abs_uv * half_grid_size) + half_grid_size);
    const int idx_v =
            (int)(floor(grid_v / max_abs_uv * half_grid_size) + half_grid_size);

    if (idx_u < grid_size && idx_v < grid_size)
    {
        const int i_pol_s = INDEX_3D(grid_size,
                grid_size,
                num_pols,
                idx_u,
                idx_v,
                0
        );
        const int i_pol_start = INDEX_4D(num_times,
                num_baselines,
                num_channels,
                num_pols,
                i_time,
                i_baseline,
                i_channel,
                0
        );

        for (int i_pol = 0; i_pol < num_pols; ++i_pol)
        {
            atomicAdd(&weights_grid_uv[i_pol_s + i_pol],
                    input_weight[i_pol_start + i_pol]
            );
        }
    }
}

SDP_CUDA_KERNEL(grid_write_gpu<double, double, double>)
SDP_CUDA_KERNEL(grid_write_gpu<double, double, float>)


// The following kernel is not fully optimized as it can be better optimized by using reduction to sum the weights


// Currently it uses atomic operators for avoiding race conditions


template<typename UVW_TYPE, typename FREQ_TYPE, typename WEIGHT_TYPE>
__global__ void calc_sum_gpu(
        const int num_times,
        const int num_baselines,
        const int num_channels,
        const int num_pols,
        const int grid_size,
        const UVW_TYPE* const __restrict__ uvw,
        const FREQ_TYPE* const __restrict__ freq_hz,
        const double max_abs_uv,
        const WEIGHT_TYPE* const __restrict__ weights_grid_uv,
        const WEIGHT_TYPE* const __restrict__ input_weight,
        double* sumweight,
        double* sumweight2
)
{
    const int i_baseline = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_channel = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_time = blockDim.z * blockIdx.z + threadIdx.z;
    const int half_grid_size = grid_size / 2;

    // Bounds Check
    if (i_baseline >= num_baselines ||
            i_channel >= num_channels ||
            i_time >= num_times)
    {
        return;
    }

    const int i_uv = INDEX_3D(
            num_times, num_baselines, 3,
            i_time, i_baseline, 0
    );

    const UVW_TYPE inv_wavelength = freq_hz[i_channel] / C_0;
    const UVW_TYPE grid_u = uvw[i_uv + 0] * inv_wavelength;
    const UVW_TYPE grid_v = uvw[i_uv + 1] * inv_wavelength;

    const int idx_u =
            (int)(floor(grid_u / max_abs_uv * half_grid_size) + half_grid_size);
    const int idx_v =
            (int)(floor(grid_v / max_abs_uv * half_grid_size) + half_grid_size);

    if (idx_u < grid_size && idx_v < grid_size)
    {
        const int i_pol_s = INDEX_3D(grid_size,
                grid_size,
                num_pols,
                idx_u,
                idx_v,
                0
        );
        const int i_pol_start = INDEX_4D(num_times,
                num_baselines,
                num_channels,
                num_pols,
                i_time,
                i_baseline,
                i_channel,
                0
        );

        for (int i_pol = 0; i_pol < num_pols; ++i_pol)
        {
            atomicAdd(&(*sumweight), input_weight[i_pol_start + i_pol]);
            atomicAdd(&(*sumweight2),
                    weights_grid_uv[i_pol_s + i_pol] *
                    weights_grid_uv[i_pol_s + i_pol]
            );
        }
    }
}

SDP_CUDA_KERNEL(calc_sum_gpu<double, double, double>)
SDP_CUDA_KERNEL(calc_sum_gpu<double, double, float>)


template<typename UVW_TYPE, typename FREQ_TYPE, typename WEIGHT_TYPE>
__global__ void grid_briggs_read_gpu(
        const int num_times,
        const int num_baselines,
        const int num_channels,
        const int num_pols,
        const int grid_size,
        const UVW_TYPE* const __restrict__ uvw,
        const FREQ_TYPE* const __restrict__ freq_hz,
        const double max_abs_uv,
        const WEIGHT_TYPE* const __restrict__ weights_grid_uv,
        const WEIGHT_TYPE* const __restrict__ input_weights,
        WEIGHT_TYPE* __restrict__ output_weights,
        double robustness
)
{
    const int i_baseline = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_channel = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_time = blockDim.z * blockIdx.z + threadIdx.z;
    const int half_grid_size = grid_size / 2;

    // Bounds Check
    if (i_baseline >= num_baselines ||
            i_channel >= num_channels ||
            i_time >= num_times)
    {
        return;
    }

    const int i_uv = INDEX_3D(
            num_times, num_baselines, 3,
            i_time, i_baseline, 0
    );

    WEIGHT_TYPE weight_val = 1.0;
    const UVW_TYPE inv_wavelength = freq_hz[i_channel] / C_0;
    const UVW_TYPE grid_u = uvw[i_uv + 0] * inv_wavelength;
    const UVW_TYPE grid_v = uvw[i_uv + 1] * inv_wavelength;

    const int idx_u =
            (int)(floor(grid_u / max_abs_uv * half_grid_size) + half_grid_size);
    const int idx_v =
            (int)(floor(grid_v / max_abs_uv * half_grid_size) + half_grid_size);

    if (idx_u < grid_size && idx_v < grid_size)
    {
        const int i_pol_s = INDEX_3D(grid_size,
                grid_size,
                num_pols,
                idx_u,
                idx_v,
                0
        );
        const int i_pol_start = INDEX_4D(num_times,
                num_baselines,
                num_channels,
                num_pols,
                i_time,
                i_baseline,
                i_channel,
                0
        );

        for (int i_pol = 0; i_pol < num_pols; i_pol++)
        {
            weight_val = input_weights[i_pol_start + i_pol] /
                    (1 + (robustness * weights_grid_uv[i_pol_s + i_pol]));
            output_weights[i_pol_start + i_pol] = weight_val;
        }
    }
}

SDP_CUDA_KERNEL(grid_briggs_read_gpu<double, double, double>)
SDP_CUDA_KERNEL(grid_briggs_read_gpu<double, double, float>)


template<typename UVW_TYPE, typename FREQ_TYPE, typename WEIGHT_TYPE>
__global__ void grid_uniform_read_gpu(
        const int num_times,
        const int num_baselines,
        const int num_channels,
        const int num_pols,
        const int grid_size,
        const UVW_TYPE* const __restrict__ uvw,
        const FREQ_TYPE* const __restrict__ freq_hz,
        const double max_abs_uv,
        const WEIGHT_TYPE* const __restrict__ weights_grid_uv,
        const WEIGHT_TYPE* const __restrict__ input_weights,
        WEIGHT_TYPE* __restrict__ output_weights
)
{
    const int i_baseline = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_channel = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_time = blockDim.z * blockIdx.z + threadIdx.z;
    const int half_grid_size = grid_size / 2;

    // Bounds Check
    if (i_baseline >= num_baselines ||
            i_channel >= num_channels ||
            i_time >= num_times)
    {
        return;
    }

    const int i_uv = INDEX_3D(
            num_times, num_baselines, 3,
            i_time, i_baseline, 0
    );

    WEIGHT_TYPE weight_val = 1.0;
    const UVW_TYPE inv_wavelength = freq_hz[i_channel] / C_0;
    const UVW_TYPE grid_u = uvw[i_uv + 0] * inv_wavelength;
    const UVW_TYPE grid_v = uvw[i_uv + 1] * inv_wavelength;

    const int idx_u =
            (int)(floor(grid_u / max_abs_uv * half_grid_size) + half_grid_size);
    const int idx_v =
            (int)(floor(grid_v / max_abs_uv * half_grid_size) + half_grid_size);

    if (idx_u < grid_size && idx_v < grid_size)
    {
        const int i_pol_s = INDEX_3D(grid_size,
                grid_size,
                num_pols,
                idx_u,
                idx_v,
                0
        );
        const int i_pol_start = INDEX_4D(num_times,
                num_baselines,
                num_channels,
                num_pols,
                i_time,
                i_baseline,
                i_channel,
                0
        );

        for (int i_pol = 0; i_pol < num_pols; i_pol++)
        {
            weight_val = 1.0 / weights_grid_uv[i_pol_s + i_pol];
            output_weights[i_pol_start + i_pol] = weight_val;
        }
    }
}

SDP_CUDA_KERNEL(grid_uniform_read_gpu<double, double, double>)
SDP_CUDA_KERNEL(grid_uniform_read_gpu<double, double, float>)
