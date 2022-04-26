/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_DEGRIDDING_H_
#define SKA_SDP_PROC_DEGRIDDING_H_

/**
 * @file sdp_degridding.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Brief description of the function.
 *
 * Detailed description of the function, and its inputs and outputs.
 *
 * @param input Description of input array.
 * @param output Description of output array.
 * @param status Error status.
 */
void sdp_degridding(
        const sdp_Mem* input,
        sdp_Mem* output,
        sdp_Error* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */