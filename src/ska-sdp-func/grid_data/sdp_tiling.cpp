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
        int* num_points_in_tiles,
        int* num_skipped
)
{
    const int64_t grid_centre = grid_size / 2;

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
                const UVW_TYPE pos_u = uvw[i_uv + 0] * inv_wavelength;
                const UVW_TYPE pos_v = uvw[i_uv + 1] * inv_wavelength;

                const int grid_u =
                        (int)(floor(pos_u / grid_centre) + grid_centre);
                const int grid_v =
                        (int)(floor(pos_v / grid_centre) + grid_centre);

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
                            num_points_in_tiles[pu + pv * num_tiles_u] += 1;
                        }
                    }
                }
                else{num_skipped[0] += 1;}
            }
        }
    }
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
        float* sorted_uu,
        float* sorted_vv,
        VIS_TYPE* sorted_vis,
        WEIGHT_TYPE* sorted_weight,
        float* sorted_tile
)
{
    const int64_t grid_centre = grid_size / 2;

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
                const UVW_TYPE pos_u = uvw[i_uv + 0] * inv_wavelength;
                const UVW_TYPE pos_v = uvw[i_uv + 1] * inv_wavelength;

                const int grid_u =
                        (int)(floor(pos_u / grid_centre) + grid_centre);
                const int grid_v =
                        (int)(floor(pos_v / grid_centre) + grid_centre);

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
                            int off = tile_offsets[pu + pv] += 1;
                            sorted_uu[off] = pos_u;
                            sorted_vv[off] = pos_v;
                            for (int i = 0; i < NUM_POL; i++)
                            {
                                sorted_vis[off] = vis[i_vis + i];
                                sorted_weight[off] = vis[i_vis + i];
                            }
                            sorted_tile[off] = pv * 32768 * pu;
                        }
                    }
                }
            }
        }
    }
}


void sdp_tile_and_bucket_sort_simple(
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
        int* tile_offsets,
        sdp_Mem* sorted_vis,
        float* sorted_uu,
        float* sorted_vv,
        sdp_Mem* sorted_weight,
        sdp_Mem* num_points_in_tiles,
        sdp_Mem* num_skipped,
        float* sorted_tile,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_MemLocation vis_location = sdp_mem_location(sorted_vis);
    sdp_MemType vis_type = sdp_mem_type(vis);
    sdp_MemType uvw_type = sdp_mem_type(uvw);
    sdp_MemType freq_type = sdp_mem_type(freqs);
    sdp_MemType weight_type = sdp_mem_type(weights);
    int64_t num_times = 0;
    int64_t num_baselines = 0;
    int64_t num_channels = 0;
    int64_t num_pols = 0;

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

    sdp_mem_check_writeable(sorted_vis, status);
    sdp_mem_check_writeable(sorted_weight, status);

    constexpr int NUM_POL = 4;

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
                    (int*)sdp_mem_data(num_points_in_tiles),
                    (int*)sdp_mem_data(num_skipped)
            );

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
                    tile_offsets,
                    sorted_uu,
                    sorted_vv,
                    (double*)sdp_mem_data(sorted_vis),
                    (double*)sdp_mem_data(sorted_weight),
                    sorted_tile
            );
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
            kernel_name = "sdp_tile_count_simple_gpu<double,double>";
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
            (const void*) num_tiles_u,
            (const void*) top_left_u,
            (const void*) top_left_v,
            (void*) num_points_in_tiles,
            (void*) num_skipped,
        };

        sdp_launch_cuda_kernel(kernel_name,
                num_blocks,
                num_threads,
                0,
                0,
                args,
                status
        );

        // Launch bucket sort kernel

        const char* kernel_name2 = 0;
        if (uvw_type == SDP_MEM_DOUBLE &&
                weight_type == SDP_MEM_DOUBLE &&
                freq_type == SDP_MEM_DOUBLE)
        {
            kernel_name2 =
                    "sdp_bucket_sort_simple_gpu<double, double, double, double, 4>";
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
            (const void*) num_tiles_u,
            (const void*) top_left_u,
            (const void*) top_left_v,
            (void*) tile_offsets,
            (void*) sorted_uu,
            (void*) sorted_vv,
            sdp_mem_gpu_buffer(sorted_vis, status),
            sdp_mem_gpu_buffer(sorted_weight, status),
            (void*) sorted_tile,
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
