/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_GRID_UVW_ES_H_
#define SKA_SDP_GRID_UVW_ES_H_

/**
 * @file sdp_grid_uvw_es.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Grid visibilities onto a (partial) single w-layer.
 *
 * @param uvw Visibility (u,v,w) coordinates in metres, shape [time][baseline][3]
 * @param vis Input visibilities with shape [time][baseline][chan]
 * @param freq_hz Array of channel frequencies, in Hz.
 * @param image_size Required size of final image.
 * @param epsilon Parameter to specify required accuracy of gridding.
 * @param cell_size_rad Cell (pixel) size, in radians.
 * @param w_scale Factor to convert w-coordinates to w-layer index.
 * @param min_plane_w The w-coordinate of the first w-layer.
 * @param sub_grid_start_u Start index of sub-grid in u dimension.
 * @param sub_grid_start_v Start index of sub-grid in v dimension.
 * @param sub_grid_w Index of sub-grid in w-layer stack.
 * @param sub_grid Output sub-grid data with shape [v][u]
 * @param status Error status.
 */
void sdp_grid_uvw_es(
        const sdp_Mem* uvw,
        const sdp_Mem* vis,
        const sdp_Mem* weights,
        const sdp_Mem* freq_hz,
        int image_size,
        double epsilon,
        double cell_size_rad,
        double w_scale,
        double min_plane_w,
        int sub_grid_start_u,
        int sub_grid_start_v,
        int sub_grid_w,
        sdp_Mem* sub_grid,
        sdp_Error* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
