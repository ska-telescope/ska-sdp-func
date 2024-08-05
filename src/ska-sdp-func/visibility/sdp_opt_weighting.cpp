#include <cmath>
#include <complex>
#include <ctime>
#include <iostream>
#include <vector>
#include "ska-sdp-func/utility/sdp_data_model_checks.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/visibility/sdp_opt_weighting.h"


#define C_0 299792458.0

#define INDEX_5D(N5, N4, N3, N2, N1, I5, I4, I3, I2, I1) \
    (N1 * (N2 * (N3 * (N4 * I5 + I4) + I3) + I2) + I1)

#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) \
    (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)

#define INDEX_3D(N3, N2, N1, I3, I2, I1)         (N1 * (N2 * I3 + I2) + I1)

#define INDEX_2D(N2, N1, I2, I1)                 (N1 * I2 + I1)


void sdp_optimized_weighting(
        const sdp_Mem* uvw,
        const sdp_Mem* freqs,
        const sdp_Mem* vis,
        const sdp_Mem* weights,
        const double robust_param,
        const int grid_size,
        const double cell_size_rad,
        const int64_t support,
        int* num_visibilites,
        sdp_Mem* sorted_uu,
        sdp_Mem* sorted_vv,
        sdp_Mem* sorted_weight,
        sdp_Mem* sorted_tile,
        sdp_Mem* sorted_vis,
        sdp_Mem* tile_offsets,
        sdp_Mem* num_points_in_tiles,
        sdp_Mem* output_weights,
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

    // Calculate parameters for tiling
    int64_t grid_centre = grid_size / 2;
    const int64_t tile_size_u = 32;
    const int64_t tile_size_v = 16;
    int64_t ctile_u = grid_centre / tile_size_u;
    int64_t ctile_v = grid_centre / tile_size_v;
    const float inv_tile_size_u = 1.0 / tile_size_u;
    const float inv_tile_size_v = 1.0 / tile_size_v;
    int64_t num_tiles_u = (grid_size + tile_size_u - 1) / tile_size_u;
    int64_t num_tiles_v = (grid_size + tile_size_v - 1) / tile_size_v;
    int64_t num_tiles = num_tiles_u * num_tiles_v;
    int64_t top_left_u = grid_centre - ctile_u * tile_size_u - tile_size_u / 2;
    int64_t top_left_v = grid_centre - ctile_v * tile_size_v - tile_size_v / 2;

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

    if (vis_location == SDP_MEM_CPU)
    {
        if (uvw_type == SDP_MEM_DOUBLE &&
                weight_type == SDP_MEM_DOUBLE &&
                freq_type == SDP_MEM_DOUBLE)
        {
            *status = SDP_ERR_MEM_LOCATION;
            SDP_LOG_ERROR("CPU Briggs Weighting doesn't exist yet!");
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
        }
    }
    else if (vis_location == SDP_MEM_GPU)
    {
        // Define hyperparameters for weighting

        uint64_t n_threads  = tile_size_u * tile_size_v;
        const uint64_t num_threads_briggs[] = {n_threads, 1, 1};
        const uint64_t num_blocks_briggs[] = {(uint64_t)num_tiles - 1, 1, 1};

        // Make sure weights are on the gpu
        sdp_mem_check_location(output_weights, SDP_MEM_GPU, status);

        const char* kernel_name_weights_update = 0;

        if (uvw_type == SDP_MEM_DOUBLE &&
                weight_type == SDP_MEM_DOUBLE && freq_type == SDP_MEM_DOUBLE)
        {
            kernel_name_weights_update =
                    "sdp_opt_briggs_bucket_gpu<double, double>";
        }

        const void* weighting_args[]{
            sdp_mem_gpu_buffer_const(sorted_uu, status),
            sdp_mem_gpu_buffer_const(sorted_vv, status),
            sdp_mem_gpu_buffer_const(sorted_weight, status),
            sdp_mem_gpu_buffer_const(sorted_tile, status),
            sdp_mem_gpu_buffer_const(tile_offsets, status),
            sdp_mem_gpu_buffer_const(num_points_in_tiles, status),
            (const void*)&top_left_u,
            (const void*)&top_left_v,
            (const void*)&grid_size,
            (const void*)&num_tiles,
            (const void*)&support,
            (const void*)&robust_param,
            (const void*)&tile_size_u,
            (const void*)&tile_size_v,
            sdp_mem_gpu_buffer(output_weights, status)
        };

        size_t shared_mem_size = tile_size_u * tile_size_v * sizeof(double);

        sdp_launch_cuda_kernel(
                kernel_name_weights_update,
                num_blocks_briggs,
                num_threads_briggs,
                shared_mem_size,
                0,
                weighting_args,
                status
        );
    }
}


void sdp_optimised_indexed_weighting(
        const sdp_Mem* uvw,
        const sdp_Mem* vis,
        const sdp_Mem* weights,
        const double robust_param,
        const int grid_size,
        const double cell_size_rad,
        const int64_t support,
        const int* num_visibilites,
        sdp_Mem* sorted_tile,
        sdp_Mem* sorted_uu,
        sdp_Mem* sorted_vv,
        sdp_Mem* sorted_vis_index,
        sdp_Mem* tile_offsets,
        sdp_Mem* num_points_in_tiles,
        sdp_Mem* output_weights,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_MemLocation vis_location = sdp_mem_location(vis);
    sdp_MemType vis_type = sdp_mem_type(vis);
    sdp_MemType uvw_type = sdp_mem_type(uvw);
    sdp_MemType weight_type = sdp_mem_type(weights);
    int64_t num_times = 0;
    int64_t num_baselines = 0;
    int64_t num_channels = 0;
    int64_t num_pols = 0;

    // Calculate parameters for tiling
    int64_t grid_centre = grid_size / 2;
    const int64_t tile_size_u = 32;
    const int64_t tile_size_v = 16;
    int64_t ctile_u = grid_centre / tile_size_u;
    int64_t ctile_v = grid_centre / tile_size_v;
    const float inv_tile_size_u = 1.0 / tile_size_u;
    const float inv_tile_size_v = 1.0 / tile_size_v;
    int64_t num_tiles_u = (grid_size + tile_size_u - 1) / tile_size_u;
    int64_t num_tiles_v = (grid_size + tile_size_v - 1) / tile_size_v;
    int64_t num_tiles = num_tiles_u * num_tiles_v;
    int64_t top_left_u = grid_centre - ctile_u * tile_size_u - tile_size_u / 2;
    int64_t top_left_v = grid_centre - ctile_v * tile_size_v - tile_size_v / 2;
    int num_vis = *num_visibilites;

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

    if (vis_location == SDP_MEM_CPU)
    {
        if (uvw_type == SDP_MEM_DOUBLE &&
                weight_type == SDP_MEM_DOUBLE)
        {
            *status = SDP_ERR_MEM_LOCATION;
            SDP_LOG_ERROR("CPU Briggs Weighting doesn't exist yet!");
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
        }
    }
    else if (vis_location == SDP_MEM_GPU)
    {
        // Define hyperparameters for weighting

        uint64_t n_threads  = tile_size_u * tile_size_v;
        const uint64_t num_threads_briggs[] = {n_threads, 1, 1};
        const uint64_t num_blocks_briggs[] = {(uint64_t)num_tiles - 1, 1, 1};

        // Make sure weights are on the gpu
        sdp_mem_check_location(output_weights, SDP_MEM_GPU, status);

        const char* kernel_name_weights_update = 0;

        if (uvw_type == SDP_MEM_DOUBLE &&
                weight_type == SDP_MEM_DOUBLE)
        {
            kernel_name_weights_update =
                    "sdp_opt_briggs_index_gpu<double, double, double>";
        }

        const void* weighting_args[]{
            sdp_mem_gpu_buffer_const(sorted_uu, status),
            sdp_mem_gpu_buffer_const(sorted_vv, status),
            sdp_mem_gpu_buffer_const(weights, status),
            sdp_mem_gpu_buffer_const(sorted_vis_index, status),
            sdp_mem_gpu_buffer_const(sorted_tile, status),
            sdp_mem_gpu_buffer_const(tile_offsets, status),
            sdp_mem_gpu_buffer_const(num_points_in_tiles, status),
            (const void*)&top_left_u,
            (const void*)&top_left_v,
            (const void*)&grid_size,
            (const void*)&num_tiles,
            (const void*)&support,
            (const void*)&robust_param,
            (const void*)&num_vis,
            (const void*)&num_channels,
            (const void*)&tile_size_u,
            (const void*)&tile_size_v,
            (const void*)&cell_size_rad,
            sdp_mem_gpu_buffer(output_weights, status)
        };

        size_t shared_mem_size = (tile_size_u * tile_size_v * sizeof(double)) +
                (sizeof(double) * 2);

        sdp_launch_cuda_kernel(
                kernel_name_weights_update,
                num_blocks_briggs,
                num_threads_briggs,
                shared_mem_size,
                0,
                weighting_args,
                status
        );
    }
}
