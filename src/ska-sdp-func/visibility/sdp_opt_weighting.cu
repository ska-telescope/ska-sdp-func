#include <cmath>
#include <cooperative_groups.h>
#include <stdio.h>
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/visibility/sdp_opt_weighting.h"

#define C_0 299792458.0

#define INDEX_5D(N5, N4, N3, N2, N1, I5, I4, I3, I2, I1) \
    (N1 * (N2 * (N3 * (N4 * I5 + I4) + I3) + I2) + I1)

#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) \
    (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)

#define INDEX_3D(N3, N2, N1, I3, I2, I1)         (N1 * (N2 * I3 + I2) + I1)

#define INDEX_2D(N2, N1, I2, I1)                 (N1 * I2 + I1)


template<typename UVW_TYPE, typename WEIGHT_TYPE>
__global__ void sdp_opt_briggs_bucket_gpu(
        const UVW_TYPE* const __restrict__ sorted_uu,
        const UVW_TYPE* const __restrict__ sorted_vv,
        const WEIGHT_TYPE* const __restrict__ sorted_weights,
        const int* const __restrict__ sorted_tile,
        const int* const __restrict__ tile_offsets,
        const int* const __restrict__ num_points_in_tiles,
        const int top_left_u,
        const int top_left_v,
        const int grid_size,
        const int num_tiles,
        const int support,
        const int robust_param,
        const int64_t tile_size_u,
        const int64_t tile_size_v,
        WEIGHT_TYPE* output_weights
)
{
    WEIGHT_TYPE numerator = 0.0;
    WEIGHT_TYPE denominator = 0.0;
    WEIGHT_TYPE robustness = 0.0;
    WEIGHT_TYPE weight_val = 1.0;

    __shared__ WEIGHT_TYPE sw;
    __shared__ WEIGHT_TYPE sw2;
    extern __shared__ WEIGHT_TYPE tile[];
    tile[threadIdx.x] = 0.0;
    sw = 0.0;
    sw2 = 0.0;

    __syncthreads();

    size_t tile_idx = blockIdx.x;
    int grid_centre = grid_size / 2;
    int64_t start_vis = tile_offsets[tile_idx];
    int64_t total_vis = tile_offsets[tile_idx + 1] - tile_offsets[tile_idx];
    const int pu = sorted_tile[start_vis] & 32767;
    const int pv = sorted_tile[start_vis] >> 15;
    const int tile_idx_u = pu * tile_size_u + top_left_u;
    const int tile_idx_v = pv * tile_size_v + top_left_v;

    __syncthreads();

    for (size_t i_vis = threadIdx.x + start_vis;
            i_vis < total_vis;
            i_vis += blockDim.x)
    {
        UVW_TYPE pos_u = sorted_uu[i_vis];
        UVW_TYPE pos_v = sorted_vv[i_vis];
        int64_t grid_u = (int64_t)round(pos_u) + grid_centre;
        int64_t grid_v = (int64_t)round(pos_v) + grid_centre;
        const int64_t tile_grid_u = grid_u - tile_idx_u;
        const int64_t tile_grid_v = grid_v - tile_idx_v;
        if (tile_grid_u >= 0 && tile_grid_u < tile_size_u && tile_grid_v >= 0 &&
                tile_grid_v < tile_size_v)
        {
            const int64_t tile_grid_idx = INDEX_2D(tile_size_u,
                    tile_size_v,
                    tile_grid_u,
                    tile_grid_v
            );
            WEIGHT_TYPE weight = sorted_weights[i_vis];
            atomicAdd(&tile[tile_grid_idx], weight);
        }
    }

    __syncthreads();

    for (size_t i_vis = threadIdx.x + start_vis;
            i_vis < total_vis;
            i_vis += blockDim.x)
    {
        const UVW_TYPE pos_u = sorted_uu[i_vis];
        const UVW_TYPE pos_v = sorted_vv[i_vis];
        const int64_t grid_u = (int64_t)round(pos_u) + grid_centre;
        const int64_t grid_v = (int64_t)round(pos_v) + grid_centre;
        const int64_t tile_grid_u = grid_u - tile_idx_u;
        const int64_t tile_grid_v = grid_v - tile_idx_v;
        if (tile_grid_u >= 0 && tile_grid_u < tile_size_u && tile_grid_v >= 0 &&
                tile_grid_v < tile_size_v)
        {
            const int64_t tile_grid_idx = INDEX_2D(tile_size_u,
                    tile_size_v,
                    tile_grid_u,
                    tile_grid_v
            );
            atomicAdd(&sw, tile[tile_grid_idx]);
            atomicAdd(&sw2, tile[tile_grid_idx] * tile[tile_grid_idx]);
        }

        numerator = pow(5.0 * 1 / (pow(10.0, robust_param)), 2.0);
        denominator = sw2 / sw;
        robustness = numerator / denominator;
    }

    __syncthreads();

    for (size_t i_vis = threadIdx.x + start_vis;
            i_vis < total_vis;
            i_vis += blockDim.x)
    {
        const UVW_TYPE pos_u = sorted_uu[i_vis];
        const UVW_TYPE pos_v = sorted_vv[i_vis];
        const int64_t grid_u = (int64_t)round(pos_u) + grid_centre;
        const int64_t grid_v = (int64_t)round(pos_v) + grid_centre;
        const int64_t tile_grid_v = grid_v - tile_idx_v;
        const int64_t tile_grid_u = grid_u - tile_idx_u;
        if (tile_grid_u >= 0 && tile_grid_u < tile_size_u && tile_grid_v >= 0 &&
                tile_grid_v < tile_size_v)
        {
            const int64_t tile_grid_idx = INDEX_2D(tile_size_u,
                    tile_size_v,
                    tile_grid_u,
                    tile_grid_v
            );
            weight_val = sorted_weights[i_vis] /
                    (1 + (robustness * tile[tile_grid_idx]));
            output_weights[i_vis] = weight_val;
        }
    }
}

SDP_CUDA_KERNEL(sdp_opt_briggs_bucket_gpu<double, double>)


template<typename UVW_TYPE, typename WEIGHT_TYPE, typename FREQ_TYPE>
__global__ void sdp_opt_briggs_index_gpu(
        const UVW_TYPE* const __restrict__ sorted_uu,
        const UVW_TYPE* const __restrict__ sorted_vv,
        const WEIGHT_TYPE* const __restrict__ weights,
        const int* const __restrict__ sorted_index,
        const int* const __restrict__ sorted_tile,
        const int* const __restrict__ tile_offsets,
        const int* const __restrict__ num_points_in_tiles,
        const int top_left_u,
        const int top_left_v,
        const int grid_size,
        const int num_tiles,
        const int support,
        const int robust_param,
        const int num_vis,
        const int64_t num_channels,
        const int64_t tile_size_u,
        const int64_t tile_size_v,
        const double cell_size_rad,
        WEIGHT_TYPE* output_weights
)
{
    WEIGHT_TYPE numerator = 0.0;
    WEIGHT_TYPE denominator = 0.0;
    WEIGHT_TYPE robustness = 0.0;
    WEIGHT_TYPE weight_val = 1.0;
    extern __shared__ WEIGHT_TYPE sw[];
    extern __shared__ WEIGHT_TYPE sw2[];
    extern __shared__ WEIGHT_TYPE tile[];
    tile[threadIdx.x] = 0.0;
    sw[0] = 0.0;
    sw2[0] = 0.0;
    __syncthreads();

    size_t tile_idx = blockIdx.x;
    int grid_centre = grid_size / 2;
    int64_t start_vis = tile_offsets[tile_idx];
    int64_t total_vis = tile_offsets[tile_idx + 1] - tile_offsets[tile_idx];
    const int pu = sorted_tile[start_vis] & 32767;
    const int pv = sorted_tile[start_vis] >> 15;
    const int tile_idx_u = pu * tile_size_u + top_left_u;
    const int tile_idx_v = pv * tile_size_v + top_left_v;

    __syncthreads();

    for (size_t i_thread = threadIdx.x + start_vis;
            i_thread < total_vis;
            i_thread += blockDim.x)
    {
        const double grid_scale = grid_size * cell_size_rad;
        const size_t i_channel = blockDim.y * blockIdx.y + threadIdx.y;
        if (i_channel > num_channels) break;
        int i_vis = sorted_index[i_thread];
        const UVW_TYPE pos_u = sorted_uu[i_thread];
        const UVW_TYPE pos_v = sorted_vv[i_thread];
        int64_t grid_u = (int64_t)round(pos_u) + grid_centre;
        int64_t grid_v = (int64_t)round(pos_v) + grid_centre;
        const int64_t tile_grid_u = grid_u - tile_idx_u;
        const int64_t tile_grid_v = grid_v - tile_idx_v;
        if (tile_grid_u >= 0 && tile_grid_u < tile_size_u && tile_grid_v >= 0 &&
                tile_grid_v < tile_size_v)
        {
            const int64_t tile_grid_idx = INDEX_2D(tile_size_u,
                    tile_size_v,
                    tile_grid_u,
                    tile_grid_v
            );
            WEIGHT_TYPE weight = weights[i_vis];
            atomicAdd(&tile[tile_grid_idx], weight);
        }
    }

    __syncthreads();

    for (size_t i_thread = threadIdx.x + start_vis;
            i_thread < total_vis;
            i_thread += blockDim.x)
    {
        const double grid_scale = grid_size * cell_size_rad;
        const size_t i_channel = blockDim.y * blockIdx.y + threadIdx.y;
        const UVW_TYPE pos_u = sorted_uu[i_thread];
        const UVW_TYPE pos_v = sorted_vv[i_thread];
        const int64_t grid_u = (int64_t)round(pos_u) + grid_centre;
        const int64_t grid_v = (int64_t)round(pos_v) + grid_centre;
        const int64_t tile_grid_u = grid_u - tile_idx_u;
        const int64_t tile_grid_v = grid_v - tile_idx_v;
        if (tile_grid_u >= 0 && tile_grid_u < tile_size_u && tile_grid_v >= 0 &&
                tile_grid_v < tile_size_v)
        {
            const int64_t tile_grid_idx = INDEX_2D(tile_size_u,
                    tile_size_v,
                    tile_grid_u,
                    tile_grid_v
            );
            atomicAdd(&sw[0], tile[tile_grid_idx]);
            atomicAdd(&sw2[0], tile[tile_grid_idx] * tile[tile_grid_idx]);
        }

        numerator = pow(5.0 * 1 / (pow(10.0, robust_param)), 2.0);
        denominator = sw2[0] / sw[0];
        robustness = numerator / denominator;
    }

    __syncthreads();

    for (size_t i_thread = threadIdx.x + start_vis;
            i_thread < total_vis;
            i_thread += blockDim.x)
    {
        const double grid_scale = grid_size * cell_size_rad;
        const size_t i_channel = blockDim.y * blockIdx.y + threadIdx.y;
        int i_vis = sorted_index[i_thread];
        const UVW_TYPE pos_u = sorted_uu[i_thread];
        const UVW_TYPE pos_v = sorted_vv[i_thread];
        const int64_t grid_u = (int64_t)round(pos_u) + grid_centre;
        const int64_t grid_v = (int64_t)round(pos_v) + grid_centre;
        const int64_t tile_grid_v = grid_v - tile_idx_v;
        const int64_t tile_grid_u = grid_u - tile_idx_u;
        if (tile_grid_u >= 0 && tile_grid_u < tile_size_u && tile_grid_v >= 0 &&
                tile_grid_v < tile_size_v)
        {
            const int64_t tile_grid_idx = INDEX_2D(tile_size_u,
                    tile_size_v,
                    tile_grid_u,
                    tile_grid_v
            );
            weight_val = weights[i_vis] /
                    (1 + (robustness * tile[tile_grid_idx]));
            output_weights[i_vis] = weight_val;
        }
    }
}

SDP_CUDA_KERNEL(sdp_opt_briggs_index_gpu<double, double, double>)
