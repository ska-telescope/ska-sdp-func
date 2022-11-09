/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_VECTOR_ADD_H_
#define SKA_SDP_PROC_FUNC_VECTOR_ADD_H_

/**
 * @file sdp_vector_add.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Simple example to add two vectors, element-wise.
 *
 * @param input_a First input vector.
 * @param input_b Second input vector.
 * @param output Output vector.
 * @param status Error status.
 */
void sdp_vector_add(
        const sdp_Mem* input_a,
        const sdp_Mem* input_b,
        sdp_Mem* output,
        sdp_Error* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
