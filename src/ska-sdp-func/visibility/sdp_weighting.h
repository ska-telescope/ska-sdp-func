/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_WEIGHTING_H_
#define SKA_SDP_PROC_FUNC_WEIGHTING_H_

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
 * @param grid_uv Output number of hits per grid cell. Dimensions as above.
 * @param weights Output uniform weights. Dimensions as above.
 * @param status Error status.
 */
void sdp_weighting_uniform(
        const sdp_Mem* uvw,
        const sdp_Mem* freq_hz,
        double max_abs_uv,
        sdp_Mem* grid_uv,
        sdp_Mem* weights,
        sdp_Error* status
);

/** @} */ /* End group weight_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
