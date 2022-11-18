/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_PSWF_H_
#define SKA_SDP_PROC_FUNC_PSWF_H_

/**
 * @file sdp_swiftly.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate prolate spheroidal wave function (PSWF)
 *
 * The number of dimensions used for the FFT is specified using the
 * @p num_dims_fft parameter. If this is less than the number of dimensions
 * in the arrays, then the FFT batch size is assumed to be the size of the
 * first (slowest varying) dimension.
 *
 * Only CPU mode is supported.
 *
 * @param m Mode parameter. 0 is generally best-behaved.
 * @param c Size parameter.
 * @param pswf_out Memory space to fill with function values
 * @param status Error status.
 *
 * @return sdp_Fft* Handle to FFT plan.
 */
void sdp_generate_pswf(
        int m,
        double c,
        struct sdp_Mem* pswf_out,
        sdp_Error* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
