/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_TILED_H_
#define SKA_SDP_PROC_FUNC_TILIED_H_

/**
 * @file sdp_tiled_functions.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup tiled_func
 * @{
 */

/**
 * @brief The functions divide the grid into tiles, then counts the number of visibilities
 * that fall into that particular tile. It then performs a prefix sum on this array
 * with stored number of visibilities in each tile.
 * Tile sizes are hard coded and are expected to be of 32 * 16 size.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p uvw is 3D and real-valued, with shape:
 *   - [ num_times, num_baselines, 3 ]
 *
 * - @p freqs is 1D and real-valued, with shape:
 *   - [ num_channels ]
 *
 * - @p vis is 4D and complex-valued, with shape:
 *   - [ num_times, num_baselines, num_channels, num_pols ]
 *
 * - @p tile_offsets is 1D and real-valued with shape:
 *   - [ num_tiles + 1 ]
 *
 * - @p num_points_in_tiles is 1D and real-valued with shape:
 *   - [ num_tiles ]
 *
 * - @p num_skipped is 1D and real valued with shape:
 *   - [ 1 ]
 *
 * @param uvw Baseline (u,v,w) coordinates, in metres. Dimensions as above.
 * @param freqs Channel frequencies, in Hz. Dimensions as above.
 * @param vis Complex-valued visibilities. Dimensions as above.
 * @param grid_size Size of the grid in one dimension. Assumed to be square.
 * @param tile_size_u Size of the individual tile, in the u-direction.
 * @param tile_size_v Size of the individual tile, in the v-direction.
 * @param cell_size_rad Size of the cell, in radians.
 * @param support Number of cells a visibility contributes to during gridding.
 * @param num_visibilites Number of total visibilities after prefix sum.
 * @param tile_offsets Prefix summed visibilities in each tile. Dimensions as above.
 * @param num_points_in_tiles Number of visibilities in each tile. Dimensions as above.
 * @param num_skipped Number of visibilities skipped during tile count. Dimensions as above.
 * @param status Error status.
 */

void sdp_count_and_prefix_sum(
        const sdp_Mem* uvw,
        const sdp_Mem* freqs,
        const sdp_Mem* vis,
        const int grid_size,
        const int64_t tile_size_u,
        const int64_t tile_size_v,
        const double cell_size_rad,
        const int64_t support,
        int* num_visibilites,
        sdp_Mem* tile_offsets,
        sdp_Mem* num_points_in_tiles,
        sdp_Mem* num_skipped,
        sdp_Error* status
);

/**
 * @brief Performs a bucket sort on the visibilities and associated properties
 * within these tiles according to what tile they fall into.
 * If a visibility falls into more than one tile it is duplicated for the bucket sort.
 * Tile sizes are hard coded and are expected to be of 32 * 16 size.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p uvw is 3D and real-valued, with shape:
 *   - [ num_times, num_baselines, 3 ]
 *
 * - @p freqs is 1D and real-valued, with shape:
 *   - [ num_channels ]
 *
 * - @p vis is 4D and complex-valued, with shape:
 *   - [ num_times, num_baselines, num_channels, num_pols ]
 *
 * - @p weights is 4D and real-valued, with shape:
 *   - [ num_times, num_baselines, num_channels, num_pols ]

 * - @p tile_offsets is 1D and real-valued with shape:
 *   - [ num_tiles + 1 ]
 *
 * - @p num_points_in_tiles is 1D and real-valued with shape:
 *   - [ num_tiles ]
 *
 * - @p sorted_uu is 1D and real valued with shape:
 *   - [ num_visibilities ]
 *
 * - @p sorted_vv is 1D and real valued with shape:
 *   - [ num_visibilties ]
 *
 * - @p sorted_vis is 1D and complex valued with shape:
 *   - [ num_visiblities ]
 *
 * - @p sorted_weight is 1D and real valued with shape:
 *   - [ num_visiblities ]
 *
 * - @p sorted_tile is 1D and real valued with shape:
 *   - [ num_visiblities ]
 *
 * @param uvw Baseline (u,v,w) coordinates, in metres. Dimensions as above.
 * @param freqs Channel frequencies, in Hz. Dimensions as above.
 * @param vis Complex-valued visibilities. Dimensions as above.
 * @param weights Weights for each visibility, as available in the input data. Dimensions as above.
 * @param grid_size Size of the grid in one dimension. Assumed to be square.
 * @param tile_size_u Size of the individual tile, in the u-direction.
 * @param tile_size_v Size of the individual tile, in the v-direction.
 * @param cell_size_rad Size of the cell, in radians.
 * @param support Number of cells a visibility contributes to during gridding.
 * @param sorted_uu Sorted u coordinates after bucket sort. Dimensions as above.
 * @param sorted_vv Sorted v coordinates after bucket sort. Dimensions as above.
 * @param sorted_weight Sorted weights after bucket sort. Dimensions as above.
 * @param sorted_tile Sorted visibilities in tile positions after bucket sort. Dimensions as above.
 * @param sorted_vis Sorted visibilties after bucket sort. Dimensions as above.
 * @param tile_offsets Prefix summed visibilities in each tile. Dimensions as above.
 * @param status Error status.
 */
void sdp_bucket_sort(
        const sdp_Mem* uvw,
        const sdp_Mem* freqs,
        const sdp_Mem* vis,
        const sdp_Mem* weights,
        const int grid_size,
        const int64_t tile_size_u,
        const int64_t tile_size_v,
        const double cell_size_rad,
        const int64_t support,
        sdp_Mem* sorted_uu,
        sdp_Mem* sorted_vv,
        sdp_Mem* sorted_weight,
        sdp_Mem* sorted_tile,
        sdp_Mem* sorted_vis,
        sdp_Mem* tile_offsets,
        sdp_Error* status
);

/**
 * @brief The functions divide the grid into tiles and bucket sort the visibility indices
 * according to what tile they fall into. If a visibility falls into more than one tile it's
 * index is duplicated for the bucket sort. Tile sizes are hard coded and are expected to
 * be of 32 * 16 size.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p uvw is 3D and real-valued, with shape:
 *   - [ num_times, num_baselines, 3 ]
 *
 * - @p freqs is 1D and real-valued, with shape:
 *   - [ num_channels ]
 *
 * - @p sorted_uu is 1D and real valued with shape:
 *   - [ num_visibilities ]
 *
 * - @p sorted_vv is 1D and real valued, with shape:
 *   - [ num_visibilties ]
 *
 * - @p sorted_vis_index is 1D and real valued, with shape:
 *   - [ num_visiblities ]
 *
 * - @p sorted_tile is 1D and real valued, with shape:
 *   - [ num_visiblities ]
 *
 * @param uvw Baseline (u,v,w) coordinates, in metres. Dimensions as above.
 * @param freqs Channel frequencies, in Hz. Dimensions as above.
 * @param grid_size Size of the grid in one dimension. Assumed to be square.
 * @param tile_size_u Size of the individual tile, in the u-direction.
 * @param tile_size_v Size of the individual tile, in the v-direction.
 * @param cell_size_rad Size of the cell, in radians.
 * @param support Number of cells a visibility contributes to during gridding.
 * @param num_channels Number of frequency channels.
 * @param num_baselines Number of baselines.
 * @param num_times Number of time samples.
 * @param num_pol Number of polarizations.
 * @param num_visibilites Number of total visibilities after prefix sum.
 * @param sorted_tile Sorted visibilities in tile positions after bucket sort. Dimensions as above.
 * @param sorted_uu Sorted u coordinates after bucket sort. Dimensions as above.
 * @param sorted_vv Sorted v coordinates after bucket sort. Dimensions as above.
 * @param sorted_vis_index Sorted indices of the visibilities after bucket sort. Dimensions as above.
 * @param tile_offsets Prefix summed visibilities in each tile. Dimensions as above.
 * @param status Error status.
 */
void sdp_tiled_indexing(
        const sdp_Mem* uvw,
        const sdp_Mem* freqs,
        const int grid_size,
        const int64_t tile_size_u,
        const int64_t tile_size_v,
        const double cell_size_rad,
        const int64_t support,
        const int64_t num_channels,
        const int64_t num_baselines,
        const int64_t num_times,
        sdp_Mem* sorted_tile,
        sdp_Mem* sorted_uu,
        sdp_Mem* sorted_vv,
        sdp_Mem* sorted_vis_index,
        sdp_Mem* tile_offsets,
        sdp_Error* status
);

/** @} */ /* End group tiled_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
