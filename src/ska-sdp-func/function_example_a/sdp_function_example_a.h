/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_FUNCTION_EXAMPLE_A_H_
#define SKA_SDP_PROC_FUNC_FUNCTION_EXAMPLE_A_H_

/**
 * @file sdp_function_example_a.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward-declare structure handles for private implementation. */
struct sdp_FunctionExampleA;
typedef struct sdp_FunctionExampleA sdp_FunctionExampleA;

/**
 * @defgroup FunctionA_func
 * @{
 */

/**
 * @brief Creates processing function A.
 *
 * @param a Value of a.
 * @param b Value of b.
 * @param c Value of c.
 * @param status Error status.
 * @return sdp_FunctionExampleA* Handle to processing function.
 */
sdp_FunctionExampleA* sdp_function_example_a_create_plan(
        int a,
        int b,
        float c,
        sdp_Error* status
    );

/**
 * @brief Demonstrate a function utilising a plan.
 *
 * @param plan Handle to processing function.
 * @param output Output buffer.
 * @param status Error status.
 */
void sdp_function_example_a_exec(
        sdp_FunctionExampleA* plan,
        sdp_Mem* output,
        sdp_Error* status
    );

/**
 * @brief Releases handle to processing function A.
 *
 * @param handle Handle to processing function.
 */
void sdp_function_example_a_free_plan(sdp_FunctionExampleA* handle);

/** @} */ /* End group FunctionA_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
