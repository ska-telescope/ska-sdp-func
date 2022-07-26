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
 * @param grid Input grid data with shape [chan][w][v][u][pol]
 * @param uvw u,v,w coordinates of the visibilities with shape [time][baseline][chan][uvw]
 * @param uv_kernel u,v plane kernel
 * @param w_kernel w plane Kernel
 * @param uv_kernel_oversampling U,V plane kernel oversampling
 * @param w_kernel_oversampling W plane kernel oversampling
 * @param theta Conversion parameter from uv coordinates to xy coordinates x=u*theta
 * @param wstep Conversion parameter from w coordinates to z coordinates z=w*wstep 
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
        const int64_t uv_kernel_oversampling,
        const int64_t w_kernel_oversampling,
        const double theta,
        const double wstep,
        const double channel_start_hz,
        const double channel_step_hz,
        const bool conjugate, 
        sdp_Mem* vis,
        sdp_Error* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
