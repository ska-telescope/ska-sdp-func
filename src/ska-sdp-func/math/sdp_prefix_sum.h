/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SDP_PROC_FUNC_PREFIX_SUM_H_
#define SDP_PROC_FUNC_PREFIX_SUM_H_

/**
 * @file sdp_prefix_sum.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Prefix sum.
 *
 * @details
 * Prefix sum.
 *
 * @param num_elements Number of elements to sum.
 * @param in Input array to sum.
 * @param out Output array containing prefix sum of input.
 * @param status Error status.
 */
void sdp_prefix_sum(
        int num_elements,
        const sdp_Mem* in,
        sdp_Mem* out,
        sdp_Error* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
