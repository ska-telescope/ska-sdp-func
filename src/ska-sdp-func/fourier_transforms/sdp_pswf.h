/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_PSWF_H_
#define SKA_SDP_PROC_FUNC_PSWF_H_

/**
 * @file sdp_pswf.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup PSWF_struct
 * @{
 */

/**
 * @struct sdp_PSWF
 *
 * @brief
 * Provides methods to evaluate the prolate spheroidal wave function (PSWF).
 */
struct sdp_PSWF;

/** @} */ /* End group PSWF_struct. */

typedef struct sdp_PSWF sdp_PSWF;

/**
 * @defgroup PSWF_func
 * @{
 */

/**
 * @brief Generate prolate spheroidal wave function (PSWF)
 *
 * Only CPU mode is supported.
 *
 * @param m Mode parameter. 0 is generally best-behaved.
 * @param c Size parameter.
 * @param pswf_out Memory space to fill with function values.
 * @param status Error status.
 */
void sdp_generate_pswf(
        int m,
        double c,
        sdp_Mem* pswf_out,
        sdp_Error* status
);

/**
 * @brief Create plan for PSWF evaluation.
 *
 * @param m Mode parameter. 0 is generally best-behaved.
 * @param c Size parameter.
 * @return Handle to function plan.
 */
sdp_PSWF* sdp_pswf_create(int m, double c);

/**
 * @brief Evaluate PSWF at a single point.
 *
 * @param plan Handle to function plan.
 * @param x Value at which to evaluate function, where |x| < 1.0.
 * @return Value of PSWF at @p x.
 */
double sdp_pswf_evaluate(const sdp_PSWF* plan, double x);

/**
 * @brief Destroy plan for PSWF evaluation.
 *
 * @param plan Handle to function plan.
 */
void sdp_pswf_free(sdp_PSWF* plan);

/** @} */ /* End group PSWF_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
