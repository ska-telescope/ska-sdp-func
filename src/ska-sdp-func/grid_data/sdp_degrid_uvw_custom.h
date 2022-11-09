/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_DEGRID_UVW_CUSTOM_H_
#define SKA_SDP_PROC_DEGRID_UVW_CUSTOM_H_

/**
 * @file sdp_degrid_uvw_custom.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Degrid visibilities.
 *
 * Degrids previously gridded visibilities using supplied convolution kernels.
 *
 * @param grid Input grid data with shape [chan][w][v][u][pol]
 * @param uvw Visibility (u,v,w) coordinates with shape [time][baseline][3]
 * @param uv_kernel (u,v)-plane kernel with shape [oversampling][stride]
 * @param w_kernel w-plane kernel with shape [oversampling][stride]
 * @param theta Conversion parameter from (u,v)-coordinates to (x,y)-coordinates x=u*theta
 * @param wstep Conversion parameter from w-coordinates to z-coordinates z=w*wstep
 * @param channel_start_hz Frequency of first channel, in Hz.
 * @param channel_step_hz Frequency increment between channels, in Hz.
 * @param conjugate  Whether to generate conjugated visibilities
 * @param vis Output Visibilities with shape [time][baseline][chan][pol]
 * @param status Error status.
 */
void sdp_degrid_uvw_custom(
        const sdp_Mem* grid,
        const sdp_Mem* uvw,
        const sdp_Mem* uv_kernel,
        const sdp_Mem* w_kernel,
        const double theta,
        const double wstep,
        const double channel_start_hz,
        const double channel_step_hz,
        const int32_t conjugate,
        sdp_Mem* vis,
        sdp_Error* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
