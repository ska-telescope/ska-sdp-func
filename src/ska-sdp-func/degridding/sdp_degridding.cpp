/* See the LICENSE file at the top-level directory of this distribution. */

#include "ska-sdp-func/degridding/sdp_degridding.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"

void sdp_degridding(
        const sdp_Mem* grid,
        const sdp_Mem* vis_coordinates,
        const sdp_Mem* uv_kernel,
        const sdp_Mem* w_kernel,
        const int64_t uv_kernel_oversampling,
        const int64_t w_kernel_oversampling,
        const double theta,
        const double wstep, 
        const bool conjugate, 
        sdp_Mem* vis,
        sdp_Error* status)

{




}