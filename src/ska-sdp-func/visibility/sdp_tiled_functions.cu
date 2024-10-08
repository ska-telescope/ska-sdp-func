#include <cmath>
#include <cooperative_groups.h>
#include <stdio.h>
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/visibility/sdp_tiled_functions.h"

#define C_0 299792458.0

#define INDEX_5D(N5, N4, N3, N2, N1, I5, I4, I3, I2, I1) \
    (N1 * (N2 * (N3 * (N4 * I5 + I4) + I3) + I2) + I1)

#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) \
    (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)

#define INDEX_3D(N3, N2, N1, I3, I2, I1)         (N1 * (N2 * I3 + I2) + I1)

#define INDEX_2D(N2, N1, I2, I1)                 (N1 * I2 + I1)


template<typename T>
__global__ void sdp_prefix_sum_gpu(
        const T num_tiles,
        T* num_points_in_tiles,
        T* tile_offsets
)
{
    extern __shared__ __align__(64) unsigned char my_smem[];
    T* scratch = reinterpret_cast<T*>(my_smem);
    const int num_loops = (num_tiles + blockDim.x) / blockDim.x;
    T running_total = (T)0;
    int idx = threadIdx.x;
    const int t = threadIdx.x + blockDim.x;
    for (int i = 0; i < num_loops; i++)
    {
        T val = (T)0;
        if (idx <= num_tiles && idx > 0)
        {
            val = num_points_in_tiles[idx - 1];
        }
        scratch[threadIdx.x] = (T)0;
        scratch[t] = val;
        for (int j = 1; j < blockDim.x; j <<= 1)
        {
            __syncthreads();
            const T x = scratch[t - j];
            __syncthreads();
            scratch[t] += x;
        }

        __syncthreads();
        if (idx <= num_tiles)
        {
            tile_offsets[idx] = scratch[t] + running_total;
        }
        idx += blockDim.x;
        running_total += scratch[2 * blockDim.x - 1];
    }
}

SDP_CUDA_KERNEL(sdp_prefix_sum_gpu<int>);


template<typename UVW_TYPE, typename FREQ_TYPE>
__global__ void sdp_tile_count_simple_gpu(
        const int64_t support,
        const int num_times,
        const int num_baselines,
        const int num_channels,
        const int grid_size,
        const float inv_tile_size_u,
        const float inv_tile_size_v,
        const UVW_TYPE* uvw,
        const FREQ_TYPE* freqs,
        const int64_t num_tiles_u,
        const int64_t top_left_u,
        const int64_t top_left_v,
        const double cell_size_rad,
        int* num_points_in_tiles,
        int* num_skipped
)
{
    const int grid_centre = grid_size / 2;
    const size_t i_baseline = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t i_channel = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t i_time = blockDim.z * blockIdx.z + threadIdx.z;
    const double grid_scale = grid_size * cell_size_rad;

    if (i_baseline >= num_baselines || i_channel >= num_channels ||
            i_time >= num_times) return;

    const int i_uvw = INDEX_3D(
            num_times, num_baselines, 3,
            i_time, i_baseline, 0
    );

    const UVW_TYPE inv_wavelength = freqs[i_channel] / C_0;
    const UVW_TYPE pos_u = uvw[i_uvw + 0] * inv_wavelength * grid_scale;
    const UVW_TYPE pos_v = uvw[i_uvw + 1] * inv_wavelength * grid_scale;

    const int64_t grid_u = (int64_t)round(pos_u) + grid_centre;
    const int64_t grid_v = (int64_t)round(pos_v) + grid_centre;

    if ((grid_u + support < grid_size) && (grid_u - support >= 0) &&
            (grid_v + support < grid_size) && (grid_v - support) >= 0)
    {
        int tile_u_min, tile_u_max, tile_v_min, tile_v_max;
        const int rel_u = grid_u - top_left_u;
        const int rel_v = grid_v - top_left_v;
        const float u1 = (float)(rel_u - support) * inv_tile_size_u;
        const float u2 = (float)(rel_u + support + 1) * inv_tile_size_u;
        const float v1 = (float)(rel_v - support) * inv_tile_size_v;
        const float v2 = (float)(rel_v + support + 1) * inv_tile_size_v;
        tile_u_min = (int)(floor(u1)); tile_u_max = (int)(ceil(u2));
        tile_v_min = (int)(floor(v1)); tile_v_max = (int)(ceil(v2));

        for (int pv = tile_v_min; pv < tile_v_max; pv++)
        {
            for (int pu = tile_u_min; pu < tile_u_max; pu++)
            {
                atomicAdd(&num_points_in_tiles[pu + pv * num_tiles_u], 1);
            }
        }
    }
    else{atomicAdd(&num_skipped[0], 1);}
}

SDP_CUDA_KERNEL(sdp_tile_count_simple_gpu<double, double>)
SDP_CUDA_KERNEL(sdp_tile_count_simple_gpu<float, float>)


template<typename UVW_TYPE, typename FREQ_TYPE, typename VIS_TYPE,
        typename WEIGHT_TYPE, int NUM_POL>
__global__ void sdp_bucket_sort_simple_gpu(
        const int64_t support,
        const int64_t num_times,
        const int64_t num_baselines,
        const int64_t num_channels,
        const int grid_size,
        const float inv_tile_size_u,
        const float inv_tile_size_v,
        const UVW_TYPE* uvw,
        const FREQ_TYPE* freqs,
        const VIS_TYPE* vis,
        const WEIGHT_TYPE* weight,
        const int64_t num_tiles_u,
        const int64_t top_left_u,
        const int64_t top_left_v,
        int* tile_offsets,
        UVW_TYPE* sorted_uu,
        UVW_TYPE* sorted_vv,
        VIS_TYPE* sorted_vis,
        WEIGHT_TYPE* sorted_weight,
        int* sorted_tile,
        const double cell_size_rad
)
{
    const int grid_centre = grid_size / 2;
    const size_t i_baseline = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t i_channel = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t i_time = blockDim.z * blockIdx.z + threadIdx.z;
    const double grid_scale = grid_size * cell_size_rad;

    if (i_baseline >= num_baselines || i_channel >= num_channels ||
            i_time >= num_times) return;

    const int i_uvw = INDEX_3D(
            num_times, num_baselines, 3,
            i_time, i_baseline, 0
    );

    const UVW_TYPE inv_wavelength = freqs[i_channel] / C_0;
    const UVW_TYPE pos_u = uvw[i_uvw + 0] * inv_wavelength * grid_scale;
    const UVW_TYPE pos_v = uvw[i_uvw + 1] * inv_wavelength * grid_scale;

    const int64_t grid_u = (int64_t)round(pos_u) + grid_centre;
    const int64_t grid_v = (int64_t)round(pos_v) + grid_centre;

    if ((grid_u + support < grid_size) && (grid_u - support >= 0) &&
            (grid_v + support < grid_size) && (grid_v - support) >= 0)
    {
        int tile_u_min, tile_u_max, tile_v_min, tile_v_max;
        const int rel_u = grid_u - top_left_u;
        const int rel_v = grid_v - top_left_v;
        const float u1 = (float)(rel_u - support) * inv_tile_size_u;
        const float u2 = (float)(rel_u + support + 1) * inv_tile_size_u;
        const float v1 = (float)(rel_v - support) * inv_tile_size_v;
        const float v2 = (float)(rel_v + support + 1) * inv_tile_size_v;
        tile_u_min = (int)(floor(u1)); tile_u_max = (int)(ceil(u2));
        tile_v_min = (int)(floor(v1)); tile_v_max = (int)(ceil(v2));

        const int i_vis = INDEX_3D(
                num_times, num_baselines, num_channels,
                i_time, i_baseline, i_channel
        );

        for (int pv = tile_v_min; pv < tile_v_max; pv++)
        {
            for (int pu = tile_u_min; pu < tile_u_max; pu++)
            {
                int off = atomicAdd(&tile_offsets[pu + pv * num_tiles_u], 1);
                sorted_uu[off] = pos_u;
                sorted_vv[off] = pos_v;
                sorted_vis[off] = vis[i_vis];
                sorted_weight[off] = weight[i_vis];
                sorted_tile[off] = pv * 32768 + pu;
            }
        }
    }
}

SDP_CUDA_KERNEL(sdp_bucket_sort_simple_gpu<double, double, double, double, 1>)
SDP_CUDA_KERNEL(sdp_bucket_sort_simple_gpu<float, float, float, float, 1>)


template<typename UVW_TYPE, typename FREQ_TYPE, typename VIS_TYPE,
        typename WEIGHT_TYPE, int NUM_POL>
__global__ void sdp_tiled_indexing_gpu(
        const int64_t support,
        const int64_t num_times,
        const int64_t num_baselines,
        const int64_t num_channels,
        const int grid_size,
        const float inv_tile_size_u,
        const float inv_tile_size_v,
        const UVW_TYPE* uvw,
        const FREQ_TYPE* freqs,
        const int64_t num_tiles_u,
        const int64_t top_left_u,
        const int64_t top_left_v,
        int* tile_offsets,
        UVW_TYPE* sorted_uu,
        UVW_TYPE* sorted_vv,
        int* sorted_vis_index,
        int* sorted_tile,
        const double cell_size_rad
)
{
    const int grid_centre = grid_size / 2;
    const size_t i_baseline = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t i_channel = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t i_time = blockDim.z * blockIdx.z + threadIdx.z;
    const double grid_scale = grid_size * cell_size_rad;

    if (i_baseline >= num_baselines || i_channel >= num_channels ||
            i_time >= num_times) return;

    const int i_uvw = INDEX_3D(
            num_times, num_baselines, 3,
            i_time, i_baseline, 0
    );

    const UVW_TYPE inv_wavelength = freqs[i_channel] / C_0;
    const UVW_TYPE pos_u = uvw[i_uvw + 0] * inv_wavelength * grid_scale;
    const UVW_TYPE pos_v = uvw[i_uvw + 1] * inv_wavelength * grid_scale;
    const int64_t grid_u = (int64_t)round(pos_u) + grid_centre;
    const int64_t grid_v = (int64_t)round(pos_v) + grid_centre;

    if ((grid_u + support < grid_size) && (grid_u - support >= 0) &&
            (grid_v + support < grid_size) && (grid_v - support) >= 0)
    {
        int tile_u_min, tile_u_max, tile_v_min, tile_v_max;
        const int rel_u = grid_u - top_left_u;
        const int rel_v = grid_v - top_left_v;
        const float u1 = (float)(rel_u - support) * inv_tile_size_u;
        const float u2 = (float)(rel_u + support + 1) * inv_tile_size_u;
        const float v1 = (float)(rel_v - support) * inv_tile_size_v;
        const float v2 = (float)(rel_v + support + 1) * inv_tile_size_v;
        tile_u_min = (int)(floor(u1)); tile_u_max = (int)(ceil(u2));
        tile_v_min = (int)(floor(v1)); tile_v_max = (int)(ceil(v2));

        const int i_vis = INDEX_3D(
                num_times, num_baselines, num_channels,
                i_time, i_baseline, i_channel
        );

        for (int pv = tile_v_min; pv < tile_v_max; pv++)
        {
            for (int pu = tile_u_min; pu < tile_u_max; pu++)
            {
                int off = atomicAdd(&tile_offsets[pu + pv * num_tiles_u], 1);
                sorted_uu[off] = pos_u;
                sorted_vv[off] = pos_v;
                sorted_vis_index[off] = i_vis;
                sorted_tile[off] = pv * 32768 + pu;
            }
        }
    }
}

SDP_CUDA_KERNEL(sdp_tiled_indexing_gpu<double, double, double, double, 1>)
SDP_CUDA_KERNEL(sdp_tiled_indexing_gpu<float, float, float, float, 1>)
