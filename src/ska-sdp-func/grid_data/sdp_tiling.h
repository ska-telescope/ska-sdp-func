/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_WEIGHTING_H_
#define SKA_SDP_PROC_FUNC_WEIGHTING_H_

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
 *   - [ num_tiles ]
 *
 * - @param sorted_vis is 4D and complex valued, with shape:
 *   - [ time_samples, baselines, channels, polarizations ]
 *
 * - @param sorted_uu is 1D and complex valued, with shape:
 *   - [ num_visibilities ]
 *
 * - @param sorted_vv is 1D and complex valued, with shape:
 *   - [ num_visibilities ]
 *
 * - @param sorted_weights is 3D and real-valued, with shape:
 *   - [ num_times, num_baselines, num_channels ]
 *
 * - @param num_points_in_tiles is 2D and real-valued, with shape:
 *
 *   -[ num_points, num_tiles ]
 *
 * - @param num_skipped is 2D and real-valued, with shape:
 *
 *   -[ num_points, num_tiles ]
 */

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
        sdp_Mem* tile_offsets,
        sdp_Mem* sorted_vis,
        sdp_Mem* sorted_uu,
        sdp_Mem* sorted_vv,
        sdp_Mem* sorted_weight,
        sdp_Mem* num_points_in_tiles,
        sdp_Mem* num_skipped,
        sdp_Mem* sorted_tile,
        sdp_Error* status
);

/** @} */ /* End group tiling_func. */
#ifdef __cplusplus
}
#endif

#endif /* include guard */
