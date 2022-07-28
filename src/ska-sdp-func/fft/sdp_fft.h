/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_FFT_H_
#define SKA_SDP_PROC_FUNC_FFT_H_

/**
 * @file sdp_fft.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

struct sdp_Fft;
typedef struct sdp_Fft sdp_Fft;

/**
 * @defgroup fft_func
 * @{
 */

/**
 * @brief Creates a plan for FFTs using the supplied input and output buffers.
 *
 * The number of dimensions used for the FFT is specified using the
 * @p num_dims_fft parameter. If this is less than the number of dimensions
 * in the arrays, then the FFT batch size is assumed to be the size of the
 * first (slowest varying) dimension.
 *
 * This wraps cuFFT, so only GPU FFTs are currently supported.
 *
 * @param input Input data.
 * @param output Output data.
 * @param num_dims_fft The number of dimensions for the FFT.
 * @param is_forward Set true if FFT should be "forward", false for "inverse".
 * @param status Error status.
 *
 * @return sdp_Fft* Handle to FFT plan.
 */
sdp_Fft* sdp_fft_create(
        const sdp_Mem* input,
        const sdp_Mem* output,
        int32_t num_dims_fft,
        int32_t is_forward,
        sdp_Error* status);

/**
 * @brief Executes FFT using plan and supplied data.
 *
 * @param fft Handle to FFT plan.
 * @param input Input data.
 * @param output Output data.
 * @param status Error status.
 */
void sdp_fft_exec(sdp_Fft* fft, sdp_Mem* input, sdp_Mem* output,
        sdp_Error* status);

/**
 * @brief Destroys the FFT plan.
 *
 * @param fft Handle to FFT plan.
 */
void sdp_fft_free(sdp_Fft* fft);

/** @} */ /* End group fft_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
