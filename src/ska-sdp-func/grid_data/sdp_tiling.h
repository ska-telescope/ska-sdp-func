/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_TILING_H_
#define SKA_SDP_PROC_FUNC_TILING_H_

/**
 * @file sdp_tiling.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup tiling_func
 * @{
 */

/**
 * @brief Calculate the number of visibilities falling into each tile and perform
 * a bucket sort on those visibilties
 *
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @param uvw is 3D and real-valued, with shape:
 *   - [ num_times, num_baselines, 3 ]
 *
 * - @param freqs is 1D and real-valued, with shape:
 *   - [ num_channels ]
 *
 * - @param vis is 4D and complex valued with shape:
 *   - [ time_samples, baselines, channels, polarizations ]
 *
 * - @param weights is 3D and real-valued, with shape:
 *   - [ num_times, num_baselines, num_channels ]
 *
 * - @param tile_offsets is 1D and real-valued, with shape:
 *   - [ num_tiles + 1 ]
 *
 * - @param num_points_in_tiles is 1D and real-valued, with shape:
 *
 *   -[ num_tiles ]
 *
 * - @param num_skipped is 1D and real-valued, with shape:
 *
 *   -[ num_tiles ]
 */

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
);

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
);

/** @} */ /* End group tiling_func. */
#ifdef __cplusplus
}
#endif

#endif /* include guard */
