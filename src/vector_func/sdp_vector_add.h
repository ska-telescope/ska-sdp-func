/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_VECTOR_ADD_H_
#define SKA_SDP_PROC_FUNC_VECTOR_ADD_H_

/**
 * @file sdp_vector_add.h
 */

#include "mem/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Adds two vectors together, element-wise.
 *
 * @param num_elements Number of elements to add.
 * @param a First input vector.
 * @param b Second input vector.
 * @param out Output vector.
 * @param status Error status.
 */
void sdp_vector_add(
        int num_elements,
        const sdp_Mem* a,
        const sdp_Mem* b,
        sdp_Mem* out,
        sdp_Error* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
