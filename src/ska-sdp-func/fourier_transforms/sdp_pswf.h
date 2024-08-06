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
 * @defgroup pswf_struct
 * @{
 */

/**
 * @struct sdp_Pswf
 *
 * @brief
 * Provides methods to evaluate the prolate spheroidal wave function (PSWF).
 */
struct sdp_Pswf;

/** @} */ /* End group pswf_struct. */

typedef struct sdp_Pswf sdp_Pswf;

/**
 * @defgroup pswf_func
 * @{
 */

/**
 * @brief Generate prolate spheroidal wave function (PSWF)
 *
 * Only CPU mode is supported.
 *
 * @param m Non-negative mode parameter. 0 is generally best-behaved.
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
 * @param m Non-negative mode parameter. 0 is generally best-behaved.
 * @param c Size parameter.
 * @return Handle to function plan.
 */
sdp_Pswf* sdp_pswf_create(int m, double c);

/**
 * @brief Return internal plan coefficients in the specified memory location.
 *
 * @param plan Handle to function plan.
 * @param location Enumerated memory location of data to return.
 * @param status Error status.
 */
const sdp_Mem* sdp_pswf_coeff(
        sdp_Pswf* plan,
        sdp_MemLocation location,
        sdp_Error* status
);

/**
 * @brief Return internal PSWF values in the specified memory location.
 *
 * @param plan Handle to function plan.
 * @param location Enumerated memory location of data to return.
 * @param status Error status.
 */
const sdp_Mem* sdp_pswf_values(
        sdp_Pswf* plan,
        sdp_MemLocation location,
        sdp_Error* status
);

/**
 * @brief Evaluate PSWF at a single point.
 *
 * @param plan Handle to function plan.
 * @param x Value at which to evaluate function, where |x| < 1.0.
 * @return Value of PSWF at @p x.
 */
double sdp_pswf_evaluate(const sdp_Pswf* plan, double x);

/**
 * @brief Return parameter "c" (size parameter).
 *
 * @param plan Handle to function plan.
 * @return Value of parameter "c" used in plan creation.
 */
double sdp_pswf_par_c(const sdp_Pswf* plan);

/**
 * @brief Return parameter "m" (mode parameter).
 *
 * @param plan Handle to function plan.
 * @return Value of parameter "m" used in plan creation.
 */
double sdp_pswf_par_m(const sdp_Pswf* plan);

/**
 * @brief Destroy plan for PSWF evaluation.
 *
 * @param plan Handle to function plan.
 */
void sdp_pswf_free(sdp_Pswf* plan);

/**
 * @brief Generate prolate spheroidal wave function (PSWF)
 *
 * This function generates PSWF values using the supplied function plan.
 * If the @p out array is not NULL, values will be returned in that;
 * but if not supplied, and if the @p size parameter is greater than 0,
 * then values will instead be cached internally, and available
 * via @fn sdp_pswf_values().
 *
 * @param plan Handle to function plan.
 * @param out Optional memory space to fill with function values.
 * @param size Optional size of internal list of generated values.
 * @param end_correction If true, set the first element to 1e-15.
 * @param status Error status.
 */
void sdp_pswf_generate(
        sdp_Pswf* plan,
        sdp_Mem* out,
        int size,
        int end_correction,
        sdp_Error* status
);

/** @} */ /* End group pswf_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
