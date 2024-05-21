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
 * @brief Generate prolate spheroidal wave function (PSWF) at specified values.
 *
 * Only CPU mode is supported.
 *
 * @param m Mode parameter. 0 is generally best-behaved.
 * @param c Size parameter.
 * @param x Array of values at which to evaluate function, where all |x| < 1.0
 * @param pswf_out Memory space to fill with function values.
 * @param status Error status.
 */
void sdp_generate_pswf_at_x(
        int m,
        double c,
        const sdp_Mem* x,
        sdp_Mem* pswf_out,
        sdp_Error* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
