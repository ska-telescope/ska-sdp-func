/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_PHASE_ROTATE_H_
#define SKA_SDP_PROC_FUNC_PHASE_ROTATE_H_

/**
 * @file sdp_phase_rotate.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"
#include "ska-sdp-func/utility/sdp_sky_coord.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup pha_rot_func
 * @{
 */

/**
 * @brief Rotate (u,v,w) coordinates to a new phase centre.
 *
 * Parameters @p uvw_in and @p uvw_out are arrays of packed 3D coordinates.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p uvw_in is 3D and real-valued, with shape:
 *   - [ num_times, num_baselines, 3 ]
 *
 * - @p uvw_out is 3D and real-valued, with shape:
 *   - [ num_times, num_baselines, 3 ]
 *
 * @param phase_centre_orig Original phase centre.
 * @param phase_centre_new New phase centre.
 * @param uvw_in Input baseline (u,v,w) coordinates. Dimensions as above.
 * @param uvw_out Output baseline (u,v,w) coordinates. Dimensions as above.
 * @param status Error status.
 */
void sdp_phase_rotate_uvw(
        const sdp_SkyCoord* phase_centre_orig,
        const sdp_SkyCoord* phase_centre_new,
        const sdp_Mem* uvw_in,
        sdp_Mem* uvw_out,
        sdp_Error* status);

/**
 * @brief Rotate visibilities to a new phase centre.
 *
 * Parameter @p uvw is an array of packed 3D coordinates.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p uvw is 3D and real-valued, with shape:
 *   - [ num_times, num_baselines, 3 ]
 *
 * - @p vis_in is 4D and complex-valued, with shape:
 *   - [ num_times, num_baselines, num_channels, num_pols ]
 *
 * - @p vis_out is 4D and complex-valued, with shape:
 *   - [ num_times, num_baselines, num_channels, num_pols ]
 *
 * @param phase_centre_orig Original phase centre.
 * @param phase_centre_new New phase centre.
 * @param channel_start_hz Frequency of first channel, in Hz.
 * @param channel_step_hz Frequency increment between channels, in Hz.
 * @param uvw Original baseline (u,v,w) coordinates, in metres.
 *            Dimensions as above.
 * @param vis_in Input visibility data. Dimensions as above.
 * @param vis_out Output visibility data. Dimensions as above.
 * @param status Error status.
 */
void sdp_phase_rotate_vis(
        const sdp_SkyCoord* phase_centre_orig,
        const sdp_SkyCoord* phase_centre_new,
        const double channel_start_hz,
        const double channel_step_hz,
        const sdp_Mem* uvw,
        const sdp_Mem* vis_in,
        sdp_Mem* vis_out,
        sdp_Error* status);

/** @} */ /* End group pha_rot_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
