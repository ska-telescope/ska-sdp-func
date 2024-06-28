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
 * as the weight.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p uvw is 3D and real-valued, with shape:
 *   - [ num_times, num_baselines, 3 ]
 *
 * - @p freq_hz is 1D and real-valued, with shape:
 *   - [ num_channels ]
 *
 * - @p grid_uv is 2D and real-valued (should be zero-initialised), with shape:
 *   - [ num_cells_v, num_cells_u ]
 *
 * - @p weights is 4D and real-valued, with shape:
 *   - [ num_times, num_baselines, num_channels, num_pols ]
 *
 * @param uvw Baseline (u,v,w) coordinates, in metres. Dimensions as above.
 * @param freq_hz Channel frequencies, in Hz. Dimensions as above.
 * @param max_abs_uv Maximum absolute value of UV coordinates in wavelength units.
 * @param weight_grid_uv Output number of hits per grid cell. Dimensions as above.
 * @param input_weights Input weights for the visibilities. Dimensions as above.
 * @param output_weights Output of the function including the weights for each grid cell. Dimensions as above.
 * @param weighting_type Weighting type defined by the user for matching the uv function. Enum Type.
 * @param robust_param Input parameter by the user to determine robustness of the weighting. Integer value between -2 and 2.
 * @param status Error status.
 */


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
);

void sdp_tile_and_prefix_sum(
        const sdp_Mem* uvw,
        const sdp_Mem* freqs,
        const sdp_Mem* vis,
        const int grid_size,
        const double cell_size_rad,
        const int64_t support,
        int* num_visibilites, 
        sdp_Mem* tile_offsets,
        sdp_Mem* num_points_in_tiles,
        sdp_Mem* num_skipped,
        sdp_Error* status
);

void sdp_bucket_sort(
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
    sdp_Error* status
);

void sdp_tiled_indexing(
        const sdp_Mem* uvw,
        const sdp_Mem* freqs,
        const sdp_Mem* vis,
        const sdp_Mem* weights,
        const double robust_param,
        const int grid_size,
        const double cell_size_rad,
        const int64_t support,
        int* num_visibilites,
        sdp_Mem* sorted_tile,
        sdp_Mem* sorted_vis_index,
        sdp_Mem* tile_offsets,
        sdp_Error* status
);

void sdp_optimised_indexed_weighting(
        const sdp_Mem* uvw,
        const sdp_Mem* freqs,
        const sdp_Mem* vis,
        const sdp_Mem* weights,
        const double robust_param,
        const int grid_size,
        const double cell_size_rad,
        const int64_t support,
        int* num_visibilites, 
        sdp_Mem* sorted_tile,
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
