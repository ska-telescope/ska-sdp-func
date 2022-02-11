/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_FUNCTION_A_H_
#define SKA_SDP_PROC_FUNC_FUNCTION_A_H_

/**
 * @file sdp_function_a.h
 */

#include "utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward-declare structure handles for private implementation. */
struct sdp_FunctionA;
typedef struct sdp_FunctionA sdp_FunctionA;

/**
 * @brief Creates processing function A.
 *
 * @param a Value of a.
 * @param b Value of b.
 * @param c Value of c.
 * @param status Error status.
 * @return sdp_FunctionA* Handle to processing function.
 */
sdp_FunctionA* sdp_function_a_create_plan(
        int a,
        int b,
        float c,
        sdp_Error* status
    );

/**
 * @brief Dummy function to demonstrate a function utilising a plan.
 *
 * @param input_a First input vector.
 * @param input_b Second input vector.
 * @param output Output vector.
 * @param status Error status.
 */
void sdp_function_a_exec(
        sdp_FunctionA* plan,
        sdp_Mem* output,
        sdp_Error* status
    );

/**
 * @brief Releases handle to processing function A.
 *
 * @param handle Handle to processing function.
 */
void sdp_function_a_free_plan(sdp_FunctionA* handle);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
