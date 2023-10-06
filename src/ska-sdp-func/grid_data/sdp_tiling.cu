#include <cmath>
#include <stdio.h>
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/visibility/sdp_tiling.h"

#define TILE_RANGES(SUPPORT, U_MIN, U_MAX, V_MIN, V_MAX)\
        const int rel_u = grid_u - top_left_u;\
        const int rel_v = grid_v - top_left_v;\
        const float u1 = (float)(rel_u - SUPPORT) * inv_tile_size_u;\
        const float u2 = (float)(rel_u + SUPPORT + 1) * inv_tile_size_u;\
        const float v1 = (float)(rel_v - SUPPORT) * inv_tile_size_v;\
        const float v2 = (float)(rel_v + SUPPORT + 1) * inv_tile_size_v;\
        U_MIN = (int)(floor(u1)); U_MAX = (int)(ceil(u2));\
        V_MIN = (int)(floor(v1)); V_MAX = (int)(ceil(v2));\

#define C_0 299792458.0

#define INDEX_5D(N5, N4, N3, N2, N1, I5, I4, I3, I2, I1) \
    (N1 * (N2 * (N3 * (N4 * I5 + I4) + I3) + I2) + I1)

#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) \
    (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)

#define INDEX_3D(N3, N2, N1, I3, I2, I1)         (N1 * (N2 * I3 + I2) + I1)

#define INDEX_2D(N2, N1, I2, I1)                 (N1 * I2 + I1)


template<typename UVW_TYPE, typename FREQ_TYPE>
__global__ void sdp_tile_count_wproj(
        const int64_t num_w_planes, 
        const int64_t support,  
        const int num_times,
        const int num_baselines,
        const int num_channels,
        const int grid_size,
        const double cell_size,        
        const float interval_tile_size_u,
        const float interval_tile_size_v, 
        const UVW_TYPE* uvw,
        const FREQ_TYPE* freqs,  
        const int64_t num_tiles_u, 
        const int64_t top_left_u, 
        const int64_t top_left_v,
        const float w_scale, 
        int64_t num_points_in_tiles, 
        int64_t num_skipped
        )
{
    const int64_t grid_centre = grid_size/2;
    const int i_baseline = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_channel = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_time = blockDim.z * blockIdx.z + threadIdx.z;

    const int i_uvw = INDEX_3D(
            num_times, num_baselines, 3,
            i_time, i_baseline, 0
    );

    const UVW_TYPE inv_wavelength = freq_hz[i_channel] / C_0;
    const UVW_TYPE pos_u = uvw[i_uvw + 0] * inv_wavelength;
    const UVW_TYPE pos_v = uvw[i_uvw + 1] * inv_wavelength;
    const UVW_TYPE pos_w = uvw[i_uvw + 2] * inv_wavelength * w_scale;

    const int grid_u = (int)(floor(pos_u/ grid_centre) + grid_centre);
    const int grid_v = (int)(floor(pos_v/ grid_centre) + grid_centre);
    const int grid_w = (int) sqrt(fabs(pos_w));

    if(grid_w > num_w_planes) grid_w = num_w_planes - 1;

    const int w_support = support[grid_w];

    if((grid_u + w_support < grid_size) && (grid_u - w_support >= 0) && (grid_v + w_support < grid_size) && (grid_v - w_support) >= 0)
    {
        int tile_u_min, tile_u_max, tile_v_min, tile_v_max;
        TILE_RANGES(w_support, tile_u_min, tile_u_max, tile_v_min, tile_v_max);
        for(int pv = tile_v_min; pv < tile_v_max; pv++)
        {
            for(int pu = tile_u_min; pu < tile_u_max; pu++)
            {
                atomicAdd(&num_points_in_tiles[pu+pv * num_tiles_u], 1);
            }
        }
    }
    else atomicAdd(&num_skipped[0],1);
}


template<typename UVW_TYPE, typename FREQ_TYPE, typename VIS_TYPE, typename WEIGHT_TYPE, int NUM_POL>
__global__ void sdp_bucket_sort_wproj(
        const int64_t num_w_planes, 
        const int64_t support,
        const int num_times,
        const int num_baselines,
        const int num_channels,
        const int grid_size,
        const double cell_size,        
        const float interval_tile_size_u,
        const float interval_tile_size_v, 
        const UVW_TYPE* uvw,
        const FREQ_TYPE* freqs,
        const VIS_TYPE* vis,
        const WEIGHT_TYPE* weight,   
        const int64_t num_tiles_u, 
        const int64_t top_left_u, 
        const int64_t top_left_v,
        const float w_scale,
        int tile_offsets, 
        float sorted_uu, 
        float sorted_vv, 
        int sorted_grid_w,
        VIS_TYPE* sorted_vis, 
        float sorted_weight,
        float sorted_tile
)
{
    const int64_t grid_centre = grid_size/2;
    const int i_baseline = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_channel = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_time = blockDim.z * blockIdx.z + threadIdx.z;

    const int i_uvw = INDEX_3D(
            num_times, num_baselines, 3,
            i_time, i_baseline, 0
    );

    const int i_vis = INDEX_4D(
        num_times, num_baselines, num_channels, NUM_POL, 
        i_time, i_baseline, i_channel, 0 
    )

    const UVW_TYPE inv_wavelength = freq_hz[i_channel] / C_0;
    const UVW_TYPE pos_u = uvw[i_uvw + 0] * inv_wavelength;
    const UVW_TYPE pos_v = uvw[i_uvw + 1] * inv_wavelength;
    const UVW_TYPE pos_w = uvw[i_uvw + 2] * inv_wavelength * w_scale;

    const int grid_u = (int)(floor(pos_u/ grid_centre) + grid_centre);
    const int grid_v = (int)(floor(pos_v/ grid_centre) + grid_centre);
    const int grid_w = (int) sqrt(fabs(pos_w));

    if(grid_w > num_w_planes) grid_w = num_w_planes - 1;

    const int w_support = support[grid_w];

    if((grid_u + w_support < grid_size) && (grid_u - w_support >= 0) && (grid_v + w_support < grid_size) && (grid_v - w_support) >= 0)
    {
        int tile_u_min, tile_u_max, tile_v_min, tile_v_max;
        TILE_RANGES(w_support, tile_u_min, tile_u_max, tile_v_min, tile_v_max);
        for(int pv = tile_v_min; pv < tile_v_max; pv++)
        {
            for(int pu = tile_u_min; pu < tile_u_max; pu++)
            {
                int off = atomicAdd(&tile_offsets[pu+pv], 1);
                sorted_uu[off] = pos_u;
                sorted_vv[off] = pos_v;
                sorted_grid_w[off] = grid_w;
                #pragma unroll
                for(int i =0; i < NUM_POL; i++)
                {
                    sorted_vis[off] = vis[i_vis + i];
                    sorted_weight[off] = weight[i_vis + i];
                }
                sorted_tile[off] = pv * 32768 * pu;
            }
        }

    }

}

template<typename UVW_TYPE, typename FREQ_TYPE>
__global__ void sdp_tile_count_simple(
        const int64_t support,  
        const int num_times,
        const int num_baselines,
        const int num_channels,
        const int grid_size,
        const double cell_size,        
        const float interval_tile_size_u,
        const float interval_tile_size_v, 
        const UVW_TYPE* uvw,
        const FREQ_TYPE* freqs,  
        const int64_t num_tiles_u, 
        const int64_t top_left_u, 
        const int64_t top_left_v,
        const float w_scale, 
        int64_t num_points_in_tiles, 
        int64_t num_skipped
)
{
    const int64_t grid_centre = grid_size/2;
    const int i_baseline = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_channel = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_time = blockDim.z * blockIdx.z + threadIdx.z;

    const int i_uvw = INDEX_3D(
            num_times, num_baselines, 3,
            i_time, i_baseline, 0
    );

    const UVW_TYPE inv_wavelength = freq_hz[i_channel] / C_0;
    const UVW_TYPE pos_u = uvw[i_uvw + 0] * inv_wavelength;
    const UVW_TYPE pos_v = uvw[i_uvw + 1] * inv_wavelength;

    const int grid_u = (int)(floor(pos_u/ grid_centre) + grid_centre);
    const int grid_v = (int)(floor(pos_v/ grid_centre) + grid_centre);
    
    if ((grid_u + support < grid_size) && (grid_u - support >= 0) && (grid_v + support < grid_size) && (grid_v - support) >= 0)
    {
        int tile_u_min, tile_u_max, tile_v_min, tile_v_max;
        TILE_RANGES(support, tile_v_min, tile_u_max, tile_v_min, tile_v_max);
        for(int pv = tile_v_min; pv < tile_v_max; pv++)
        {
            for(int pu = tile_u_min; pu < tile_u_max; pu++)
            {
                atomicAdd(&num_points_in_tiles[pu+pv * num_tiles_u], 1);
            }
        }
    }
    else atomicAdd(&num_skipped[0],1);
}

template<typename UVW_TYPE, typename FREQ_TYPE, typename VIS_TYPE, typename WEIGHT_TYPE, int NUM_POL>
__global__ void sdp_bucket_sort_simple(
        const int64_t support,
        const int num_times,  
        const int num_baselines,
        const int num_channels,
        const int grid_size,
        const double cell_size,        
        const float interval_tile_size_u,
        const float interval_tile_size_v, 
        const UVW_TYPE* uvw,
        const FREQ_TYPE* freqs,
        const VIS_TYPE* vis,
        const WEIGHT_TYPE* weight,   
        const int64_t num_tiles_u, 
        const int64_t top_left_u, 
        const int64_t top_left_v,
        int* tile_offsets, 
        float* sorted_uu, 
        float* sorted_vv, 
        VIS_TYPE* sorted_vis, 
        float* sorted_weight,
        float* sorted_tile
)
{
    const int64_t grid_centre = grid_size/2;
    const int i_baseline = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_channel = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_time = blockDim.z * blockIdx.z + threadIdx.z;

    const int i_uvw = INDEX_3D(
            num_times, num_baselines, 3,
            i_time, i_baseline, 0
    );

    const int i_vis = INDEX_4D(
        num_times, num_baselines, num_channels, NUM_POLS, 
        i_time, i_baseline, i_channel, 0
    )

    const UVW_TYPE inv_wavelength = freq_hz[i_channel] / C_0;
    const UVW_TYPE pos_u = uvw[i_uvw + 0] * inv_wavelength;
    const UVW_TYPE pos_v = uvw[i_uvw + 1] * inv_wavelength;

    const int grid_u = (int)(floor(pos_u/ grid_centre) + grid_centre);
    const int grid_v = (int)(floor(pos_v/ grid_centre) + grid_centre);

    if((grid_u + support < grid_size) && (grid_u - support >= 0) && (grid_v + support < grid_size) && (grid_v - support) >= 0)
    {
        int tile_u_min, tile_u_max, tile_v_min, tile_v_max;
        TILE_RANGES(w_support, tile_u_min, tile_u_max, tile_v_min, tile_v_max);
        for(int pv = tile_v_min; pv < tile_v_max; pv++)
        {
            for(int pu = tile_u_min; pu < tile_u_max; pu++)
            {
                int off = atomicAdd(&tile_offsets[pu+pv], 1);
                sorted_uu[off] = pos_u;
                sorted_vv[off] = pos_v;
                #pragma unroll
                for (int i = 0; i < NUM_POL; i ++)
                {
                    sorted_vis[off] = vis[i_vis + i];
                    sorted_weight[off] = weight[i_vis + i];
                }
                sorted_tile[off] = pv * 32768 * pu;
            }
        }

    }
}
