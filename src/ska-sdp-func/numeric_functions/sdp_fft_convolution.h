/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_FFT_CONVOLUTION_H_
#define SKA_SDP_PROC_FUNC_FFT_CONVOLUTION_H_

/**
 * @file sdp_fft_convolution.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Perform a convolution on two inputs and return a result the same size as input 1
 *
 * @param in1 Input 1, complex double, 2D with equal dimensions [size1, size1].
 * @param in2 Input 2, complex double, 2D with equal dimensions [size2, size2].
 * @param out Output, complex double, 2D with the same dimentions as input 1 [size1, size1].
 * @param status Error status.
 */

void sdp_fft_convolution(
        const sdp_Mem* in1,
        const sdp_Mem* in2,
        sdp_Mem* out,
        sdp_Error* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */