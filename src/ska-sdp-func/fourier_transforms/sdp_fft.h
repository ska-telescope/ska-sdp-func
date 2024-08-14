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

/**
 * @defgroup fft_struct
 * @{
 */

/**
 * @struct sdp_Fft
 *
 * @brief
 * Wrapper for FFT functionality, using either NVIDIA's cuFFT, Intel's MKL,
 * or a stand-alone CPU version as appropriate.
 */
struct sdp_Fft;

/** @} */ /* End group fft_struct. */

/* Typedefs. */
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
 * This wraps cuFFT in addition to CPU FFT code. Advanced data
 * layouts although supported for GPU are not supported for CPU
 * version and thus are discouraged.
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
        sdp_Error* status
);

/**
 * @brief Executes FFT using plan and supplied data.
 *
 * @param fft Handle to FFT plan.
 * @param input Input data.
 * @param output Output data.
 * @param status Error status.
 */
void sdp_fft_exec(
        sdp_Fft* fft,
        sdp_Mem* input,
        sdp_Mem* output,
        sdp_Error* status
);

/**
 * @brief Destroys the FFT plan.
 *
 * @param fft Handle to FFT plan.
 */
void sdp_fft_free(sdp_Fft* fft);

/**
 * @brief Normalises the supplied array by dividing by the number of elements.
 *
 * This is to provide compatibility with numpy's ifft.
 *
 * @param data Array to normalise.
 * @param status Error status.
 */
void sdp_fft_norm(sdp_Mem* data, sdp_Error* status);

/**
 * @brief Provide fftshift() behaviour for complex data.
 *
 * The data are multiplied by a checker-board pattern to achieve the same
 * result as fftshift(), without actually moving memory around.
 * CPU or GPU memory locations are supported.
 *
 * @param data Array to shift. Can be 1D or 2D, but must be of complex type.
 * @param status Error status.
 */
void sdp_fft_phase(sdp_Mem* data, sdp_Error* status);

/** @} */ /* End group fft_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
