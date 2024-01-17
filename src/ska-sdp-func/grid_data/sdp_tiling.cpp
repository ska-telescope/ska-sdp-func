#include "ska-sdp-func/grid_data/sdp_tiling.h"
#include "ska-sdp-func/utility/sdp_data_model_checks.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"

#include <cmath>
#include <complex>
#include <vector>

#define TILE_RANGES(SUPPORT, U_MIN, U_MAX, V_MIN, V_MAX) \
    const int rel_u = grid_u - top_left_u; \
    const int rel_v = grid_v - top_left_v; \
    const float u1 = (float)(rel_u - SUPPORT) * inv_tile_size_u; \
    const float u2 = (float)(rel_u + SUPPORT + 1) * inv_tile_size_u; \
    const float v1 = (float)(rel_v - SUPPORT) * inv_tile_size_v; \
    const float v2 = (float)(rel_v + SUPPORT + 1) * inv_tile_size_v; \
    U_MIN = (int)(floor(u1)); U_MAX = (int)(ceil(u2)); \
    V_MIN = (int)(floor(v1)); V_MAX = (int)(ceil(v2)); \

#define C_0 299792458.0

#define INDEX_5D(N5, N4, N3, N2, N1, I5, I4, I3, I2, I1) \
    (N1 * (N2 * (N3 * (N4 * I5 + I4) + I3) + I2) + I1)

#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) \
    (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)

#define INDEX_3D(N3, N2, N1, I3, I2, I1)         (N1 * (N2 * I3 + I2) + I1)

#define INDEX_2D(N2, N1, I2, I1)                 (N1 * I2 + I1)


template<typename UVW_TYPE, typename FREQ_TYPE>
void sdp_tile_count_simple(
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
        const double cell_size_rad,
        int* num_points_in_tiles,
        int* num_skipped
)
{
    const int64_t grid_centre = grid_size / 2;
    const double grid_scale =  grid_size * cell_size_rad;

    for (int i_time = 0; i_time < num_times; i_time++)
    {
        for (int i_baseline = 0; i_baseline < num_baselines; i_baseline++)
        {
            const int i_uv = INDEX_3D(
                    num_times, num_baselines, 3,
                    i_time, i_baseline, 0
            );

            for (int i_channel = 0; i_channel < num_channels; i_channel++)
            {
                const UVW_TYPE inv_wavelength = freqs[i_channel] / C_0;
                const UVW_TYPE pos_u = uvw[i_uv + 0] * inv_wavelength * grid_scale;
                const UVW_TYPE pos_v = uvw[i_uv + 1] * inv_wavelength * grid_scale;

                const int64_t grid_u =
                        (int64_t)(round(pos_u) + grid_centre);
                const int64_t grid_v =
                        (int64_t)(round(pos_v) + grid_centre);

                    if ((grid_u + support < grid_size) && (grid_u - support >= 0) &&
                        (grid_v + support < grid_size) &&
                        (grid_v - support >= 0))
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
                            num_points_in_tiles[pu + pv * num_tiles_u] += 1;
                        }
                    }
                }
                else{num_skipped[0] += 1;}
            }
        }
    }
}

void sdp_prefix_sum_for_bucket_sort(
    const int num_tiles, 
    const int* num_points_in_tiles, 
    int* tile_offsets
)
{
    int sum = 0;
    int i = 0;
    for(i = 0; i < num_tiles; i++)
    {
        int x = num_points_in_tiles[i];
        tile_offsets[i] = sum;
        sum += x;
    }

    tile_offsets[i] = sum;
}


template<typename UVW_TYPE, typename FREQ_TYPE, typename VIS_TYPE,
        typename WEIGHT_TYPE, int NUM_POL>
void sdp_bucket_sort_simple(
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
    const int64_t grid_centre = grid_size / 2;
    const double grid_scale =  grid_size * cell_size_rad;
    for (int i_time = 0; i_time < num_times; i_time++)
    {
        for (int i_baseline = 0; i_baseline < num_baselines; i_baseline++)
        {
            const int i_uv = INDEX_3D(
                    num_times, num_baselines, 3,
                    i_time, i_baseline, 0
            );

            for (int i_channel = 0; i_channel < num_channels; i_channel++)
            {
                const int i_vis = INDEX_4D(
                        num_times, num_baselines, num_channels, NUM_POL,
                        i_time, i_baseline, i_channel, 0
                );

                const UVW_TYPE inv_wavelength = freqs[i_channel] / C_0;
                const UVW_TYPE pos_u = uvw[i_uv + 0] * inv_wavelength * grid_scale;
                const UVW_TYPE pos_v = uvw[i_uv + 1] * inv_wavelength * grid_scale;

                const int grid_u =
                        (int)round(pos_u) + grid_centre;
                const int grid_v =
                        (int)round(pos_v) + grid_centre;

                if ((grid_u + support < grid_size) && (grid_u - support >= 0) &&
                        (grid_v + support < grid_size) &&
                        (grid_v - support) >= 0)
                {
                    int tile_u_min, tile_u_max, tile_v_min, tile_v_max;
                    TILE_RANGES(support,
                            tile_v_min,
                            tile_u_max,
                            tile_v_min,
                            tile_v_max
                    );
                    for (int pv = tile_v_min; pv < tile_v_max; pv++)
                    {
                        for (int pu = tile_u_min; pu < tile_u_max; pu++)
                        {
                            int off = tile_offsets[pu + pv * num_tiles_u];
                            tile_offsets[pu + pv * num_tiles_u] += 1;
                            sorted_uu[off] = pos_u;
                            sorted_vv[off] = pos_v;
                            for (int i = 0; i < NUM_POL; i++)
                            {
                                sorted_vis[off] = vis[i_vis + i];
                                sorted_weight[off] = weight[i_vis + i];
                            }
                            sorted_tile[off] = pv * 32768 + pu;
                        }
                    }
                }
            }
        }
    }
}

// Function to bucket sort the visibilities after the scan operation

void sdp_bucket_simple(
    const int64_t support, 
    const int grid_size, 
    const float inv_tile_size_u,
    const float inv_tile_size_v,
    const int64_t top_left_u,
    const int64_t top_left_v, 
    const int64_t num_tiles_u, 
    const double cell_size_rad, 
    const sdp_Mem* uvw, 
    const sdp_Mem* vis, 
    const sdp_Mem* weights, 
    const sdp_Mem* freqs, 
    sdp_Mem* tile_offsets, 
    sdp_Mem* sorted_uu, 
    sdp_Mem* sorted_vv, 
    sdp_Mem* sorted_vis, 
    sdp_Mem* sorted_weight, 
    sdp_Mem* sorted_tile,
    sdp_Error* status 
)
{
    if (*status) return;
    sdp_MemLocation vis_location = sdp_mem_location(vis);
    sdp_MemType vis_type = sdp_mem_type(vis);
    sdp_MemType uvw_type = sdp_mem_type(uvw);
    sdp_MemType freq_type = sdp_mem_type(freqs);
    sdp_MemType weight_type = sdp_mem_type(weights);
    int64_t num_times = 0;
    int64_t num_baselines = 0;
    int64_t num_channels = 0;
    int64_t num_pols = 0;
    int64_t num_total = 0;

    // Check parameters
    sdp_data_model_get_vis_metadata(
            vis,
            &vis_type,
            &vis_location,
            &num_times,
            &num_baselines,
            &num_channels,
            &num_pols,
            status
    );

    sdp_data_model_check_uvw(
            uvw,
            SDP_MEM_VOID,
            vis_location,
            num_times,
            num_baselines,
            status
    );

    sdp_data_model_check_weights(
            weights,
            SDP_MEM_VOID,
            vis_location,
            num_times,
            num_baselines,
            num_channels,
            num_pols,
            status
    );

    constexpr int NUM_POL = 1;

    if (vis_location == SDP_MEM_CPU)
    {
        if (uvw_type == SDP_MEM_DOUBLE &&
                weight_type == SDP_MEM_DOUBLE &&
                freq_type == SDP_MEM_DOUBLE)
        {
            sdp_bucket_sort_simple<double, double, double, double, NUM_POL>(
                support,
                num_times,
                num_baselines,
                num_channels,
                grid_size,
                inv_tile_size_u,
                inv_tile_size_v,
                (const double*) sdp_mem_data_const(uvw),
                (const double*)sdp_mem_data_const(freqs),
                (const double*)sdp_mem_data_const(vis),
                (const double*)sdp_mem_data_const(weights),
                num_tiles_u,
                top_left_u,
                top_left_v,
                (int*)sdp_mem_data(tile_offsets),
                (double*)sdp_mem_data(sorted_uu),
                (double*)sdp_mem_data(sorted_vv),
                (double*)sdp_mem_data(sorted_vis),
                (double*)sdp_mem_data(sorted_weight),
                (int*)sdp_mem_data(sorted_tile),
                cell_size_rad
            );
        }
    }
    else if (vis_location = SDP_MEM_GPU)
    {
        // Define hyperparameters

        const uint64_t num_threads[] = {128, 2, 2};
        const uint64_t num_blocks[] = {
            (num_baselines + num_threads[0] - 1) / num_threads[0],
            (num_channels + num_threads[1] - 1) / num_threads[1],
            (num_times + num_threads[2] - 1) / num_threads[2]
        };

        // Launch bucket sort kernel

        const char* kernel_name2 = 0;
        if (uvw_type == SDP_MEM_DOUBLE &&
                weight_type == SDP_MEM_DOUBLE &&
                freq_type == SDP_MEM_DOUBLE)
        {
            kernel_name2 =
                    "sdp_bucket_sort_simple_gpu<double, double, double, double, 1>";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
        }

        const void* args2[]{
            (const void*)&support,
            (const void*)&num_times,
            (const void*)&num_baselines,
            (const void*)&num_channels,
            (const void*)&grid_size,
            (const void*)&inv_tile_size_u,
            (const void*)&inv_tile_size_v,
            sdp_mem_gpu_buffer_const(uvw, status),
            sdp_mem_gpu_buffer_const(freqs, status),
            sdp_mem_gpu_buffer_const(vis, status),
            sdp_mem_gpu_buffer_const(weights, status),
            (const void*)&num_tiles_u,
            (const void*)&top_left_u,
            (const void*)&top_left_v,
            sdp_mem_gpu_buffer(tile_offsets, status),
            sdp_mem_gpu_buffer(sorted_uu, status),
            sdp_mem_gpu_buffer(sorted_vv, status),
            sdp_mem_gpu_buffer(sorted_vis, status),
            sdp_mem_gpu_buffer(sorted_weight, status),
            sdp_mem_gpu_buffer(sorted_tile, status), 
            (const void*)&cell_size_rad
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

// Function to calculate the number of visibilites falling into each tile
void sdp_tile_simple(
        const sdp_Mem* uvw,
        const sdp_Mem* freqs,
        const sdp_Mem* vis,
        const sdp_Mem* weights,
        const int grid_size,
        const int64_t support,
        const float inv_tile_size_u,
        const float inv_tile_size_v,
        const int64_t num_tiles_u,
        const int64_t top_left_u,
        const int64_t top_left_v,
        const double cell_size_rad,
        const int num_tiles,
        sdp_Mem* tile_offsets,
        sdp_Mem* num_points_in_tiles,
        sdp_Mem* num_skipped,
        int* num_visibilites,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_MemLocation vis_location = sdp_mem_location(vis);
    sdp_MemType vis_type = sdp_mem_type(vis);
    sdp_MemType uvw_type = sdp_mem_type(uvw);
    sdp_MemType freq_type = sdp_mem_type(freqs);
    sdp_MemType weight_type = sdp_mem_type(weights);
    int64_t num_times = 0;
    int64_t num_baselines = 0;
    int64_t num_channels = 0;
    int64_t num_pols = 0;
    int64_t num_total = 0;

    // Check parameters
    sdp_data_model_get_vis_metadata(
            vis,
            &vis_type,
            &vis_location,
            &num_times,
            &num_baselines,
            &num_channels,
            &num_pols,
            status
    );

    sdp_data_model_check_uvw(
            uvw,
            SDP_MEM_VOID,
            vis_location,
            num_times,
            num_baselines,
            status
    );

    sdp_data_model_check_weights(
            weights,
            SDP_MEM_VOID,
            vis_location,
            num_times,
            num_baselines,
            num_channels,
            num_pols,
            status
    );

    constexpr int NUM_POL = 1;

    if (vis_location == SDP_MEM_CPU)
    {
        if (uvw_type == SDP_MEM_DOUBLE &&
                weight_type == SDP_MEM_DOUBLE &&
                freq_type == SDP_MEM_DOUBLE)
        {
            sdp_tile_count_simple(
                    support,
                    num_times,
                    num_baselines,
                    num_channels,
                    grid_size,
                    inv_tile_size_u,
                    inv_tile_size_v,
                    (const double*) sdp_mem_data_const(uvw),
                    (const double*)sdp_mem_data_const(freqs),
                    num_tiles_u,
                    top_left_u,
                    top_left_v,
                    cell_size_rad,
                    (int*)sdp_mem_data(num_points_in_tiles),
                    (int*)sdp_mem_data(num_skipped)
            );

            sdp_prefix_sum_for_bucket_sort(
                num_tiles, 
                (const int*)sdp_mem_data_const(num_points_in_tiles), 
                (int*)sdp_mem_data(tile_offsets)
            );
            
            *num_visibilites = *((int*)sdp_mem_data(tile_offsets) + num_tiles);

        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
        }
    }
    else if (vis_location = SDP_MEM_GPU)
    {
        // Define hyperparameters

        const uint64_t num_threads[] = {128, 2, 2};
        const uint64_t num_blocks[] = {
            (num_baselines + num_threads[0] - 1) / num_threads[0],
            (num_channels + num_threads[1] - 1) / num_threads[1],
            (num_times + num_threads[2] - 1) / num_threads[2]
        };

        // Launch tile count kernel

        const char* kernel_name = 0;
        if (uvw_type == SDP_MEM_DOUBLE &&
                weight_type == SDP_MEM_DOUBLE &&
                freq_type == SDP_MEM_DOUBLE)
        {
            kernel_name = "sdp_tile_count_simple_gpu<double, double>";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
        }

        const void* args[]{
            (const void*)&support,
            (const void*)&num_times,
            (const void*)&num_baselines,
            (const void*)&num_channels,
            (const void*)&grid_size,
            (const void*)&inv_tile_size_u,
            (const void*)&inv_tile_size_v,
            sdp_mem_gpu_buffer_const(uvw, status),
            sdp_mem_gpu_buffer_const(freqs, status),
            (const void*)&num_tiles_u,
            (const void*)&top_left_u,
            (const void*)&top_left_v,
            (const void*)&cell_size_rad,
            sdp_mem_gpu_buffer(num_points_in_tiles, status),
            sdp_mem_gpu_buffer(num_skipped, status)
        };

        sdp_launch_cuda_kernel(kernel_name,
                num_blocks,
                num_threads,
                0,
                0,
                args,
                status
        );


        //Launch prefix sum / scan kernel
        
        const char* kernel_name_prefix = 0;
        kernel_name_prefix = "sdp_preifx_sum_for_bucket_sort<int>";

        const void* args_prefix[]{
            (const void*)&num_tiles, 
            sdp_mem_gpu_buffer(num_points_in_tiles, status),
            sdp_mem_gpu_buffer(tile_offsets, status)
        };

        sdp_launch_cuda_kernel(kernel_name_prefix, 
            num_blocks, 
            num_threads, 
            0, 
            0, 
            args_prefix, 
            status
        );

        // Copy back tile offsets and get number of visiibilites
        sdp_Mem* tile_offsets_cpy = sdp_mem_create_copy(tile_offsets, SDP_MEM_CPU, status);
        *num_visibilites = *((int*)sdp_mem_data(tile_offsets_cpy) + num_tiles); 
    }
}
