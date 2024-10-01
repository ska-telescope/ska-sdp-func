/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_WEIGHTING_OPT_H_
#define SKA_SDP_PROC_FUNC_WEIGHTING_OPT_H_

/**
 * @file sdp_weighting.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup weight_func
 * @{
 */

/**
 * @brief Calculate the number of hits per UV cell and use the inverse of this
 * as the weight, calculate the sum of these weights and the sum of these weights squared
 * and with these sums, adjust the weight according to the robust parameter provided.
 * This is done on each tile, for bucket sorted visibilities and weights.
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
 *
 * - @p sorted_uu is 1D and real valued with shape:
 *   - [ num_visibilities ]
 *
 * - @p sorted_vv is 1D and real valued with shape:
 *   - [ num_visibilties ]
 *
 * - @p sorted_weight is 1D and real valued with shape:
 *   - [ num_visiblities ]
 *
 * - @p sorted_tile is 1D and real valued with shape:
 *   - [ num_visiblities ]
 *
 * - @p tile_offsets is 1D and real-valued with shape:
 *   - [ num_tiles + 1 ]
 *
 * - @p num_points_in_tiles is 1D and real-valued with shape:
 *   - [ num_tiles ]
 *
 * - @p output_weights is 1D and real-valued with shape:
 *   - [ num_visibilities ]
 *
 * @param uvw Baseline (u,v,w) coordinates, in metres. Dimensions as above.
 * @param freqs Channel frequencies, in Hz. Dimensions as above.
 * @param vis Complex-valued visibilities. Dimensions as above.
 * @param weights Weights for each visibility, as available in the input data. Dimensions as above.
 * @param robust_param Input parameter by the user to determine robustness of the weighting. Integer value between -2 and 2.
 * @param output_weights Output of the function including the weights for each visibility. Dimensions as above.
 * @param grid_size Size of the grid in one dimension. Assumed to be square.
 * @param num_visibilites Number of total visibilities after prefix sum.
 * @param sorted_uu Sorted u coordinates after bucket sort. Dimensions as above.
 * @param sorted_vv Sorted v coordinates after bucket sort. Dimensions as above.
 * @param sorted_weight Sorted weights after bucket sort. Dimensions as above.
 * @param sorted_tile Sorted visibilities in tile positions after bucket sort. Dimensions as above.
 * @param tile_offsets Prefix summed visibilities in each tile. Dimensions as above.
 * @param num_points_in_tiles Number of visibilities in each tile. Dimensions as above.
 * @param output_weights Returned weights by the function that are still sorted. Dimensions as above.
 * @param status Error status.
 */

void sdp_optimized_weighting(
        const sdp_Mem* uvw,
        const sdp_Mem* freqs,
        const sdp_Mem* vis,
        const sdp_Mem* weights,
        const double robust_param,
        const int grid_size,
        const int64_t support,
        sdp_Mem* sorted_uu,
        sdp_Mem* sorted_vv,
        sdp_Mem* sorted_weight,
        sdp_Mem* sorted_tile,
        sdp_Mem* tile_offsets,
        sdp_Mem* num_points_in_tiles,
        sdp_Mem* output_weights,
        sdp_Error* status
);

/**
 * @brief Calculate the number of hits per UV cell and use the inverse of this
 * as the weight, calculate the sum of these weights and the sum of these weights squared
 * and with these sums, adjust the weight according to the robust parameter provided.
 * This is done on each tile, for visibilities and weights, through their sorted indices.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p uvw is 3D and real-valued, with shape:
 *   - [ num_times, num_baselines, 3 ]
 *
 * - @p vis is 4D and complex-valued, with shape:
 *   - [ num_times, num_baselines, num_channels, num_pols ]
 *
 * - @p weights is 4D and real-valued, with shape:
 *   - [ num_times, num_baselines, num_channels, num_pols ]
 *
 * - @p sorted_uu is 1D and real valued with shape:
 *   - [ num_visibilities ]
 *
 * - @p sorted_vv is 1D and real valued with shape:
 *   - [ num_visibilties ]
 *
 * - @p sorted_vis_index is 1D and real valued with shape:
 *   - [ num_visiblities ]
 *
 * - @p sorted_tile is 1D and real valued with shape:
 *   - [ num_visiblities ]
 *
 * - @p tile_offsets is 1D and real-valued with shape:
 *   - [ num_tiles + 1 ]
 *
 * - @p num_points_in_tiles is 1D and real-valued with shape:
 *   - [ num_tiles ]
 *
 * - @p output_weights is 1D and real-valued with shape:
 *   - [ num_visibilities ]
 *
 * @param uvw Baseline (u,v,w) coordinates, in metres. Dimensions as above.
 * @param vis Complex-valued visibilities. Dimensions as above.
 * @param weights Weights for each visibility, as available in the input data. Dimensions as above.
 * @param robust_param Input parameter by the user to determine robustness of the weighting. Integer value between -2 and 2.
 * @param grid_size Size of the grid in one dimension. Assumed to be square.
 * @param cell_size_rad Size of the cell, in radians.
 * @param num_visibilites Number of total visibilities after prefix sum.
 * @param sorted_tile Sorted visibilities in tile positions after bucket sort. Dimensions as above.
 * @param sorted_uu Sorted u coordinates after bucket sort. Dimensions as above.
 * @param sorted_vv Sorted v coordinates after bucket sort. Dimensions as above.
 * @param sorted_vis_index Sorted indices of the visibilities after bucket sort. Dimensions as above.
 * @param tile_offsets Prefix summed visibilities in each tile. Dimensions as above.
 * @param num_points_in_tiles Number of visibilities in each tile. Dimensions as above.
 * @param output_weights Returned weights by the function that are still sorted. Dimensions as above.
 * @param status Error status.
 */

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
);

/** @} */ /* End group weight_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
