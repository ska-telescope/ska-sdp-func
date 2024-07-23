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

struct sdp_Fft_extended;
typedef struct sdp_Fft_extended sdp_Fft_extended;


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

/**
 * @brief Creates a plan for 1D FFTs using the supplied input and output buffers.
 *
 * The number of dimensions used for the FFT should be fixed to 1,
 * @p num_dims_fft parameter is redundant (ToDo - to remove it).
 *
 * This wraps cuFFT in addition to CPU FFT code. Advanced data
 * layouts although supported for GPU are not supported for CPU
 * version and thus are discouraged.
 *
 * @param input Input data (usually cuFFT buffer).
 * @param output Output data (usually cuFFT buffer).
 * @param num_dims_fft The number of dimensions for the FFT (=1)
 * @param is_forward Set true if FFT should be "forward", false for "inverse".
 * @param num_streams The number of CUDA streams.
 * @param batch_size The size of the batch (simultaneously performed 1D FFTs).
 * @param status Error status.
 *
 * @return sdp_Fft* Handle to FFT plan.
 */
sdp_Fft_extended* sdp_fft_extended_create(
        const sdp_Mem* input,
        const sdp_Mem* output,
        int32_t num_dims_fft,
        int32_t is_forward,
        int32_t num_streams,
        int32_t batch_size,
        sdp_Error* status
);

/**
 * @brief Executes 1D->2D FFT using plans, CUDA streams and supplied data.
 *
 * @param fft Handle to FFT plans and CUDA streams.
 * @param input Input data.
 * @param output Output data.
 * @param status Error status.
 */
void sdp_fft_extended_exec(
        sdp_Fft_extended* fft,
        sdp_Mem* input,
        sdp_Mem* output,
        sdp_Error* status
);

/**
 * @brief Destroys the FFT plans and streams.
 *
 * @param fft Handle to FFT plans and streams.
 */void sdp_fft_extended_free(
        sdp_Fft_extended* fft,
        sdp_Error* status
);

/** @} */ /* End group fft_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
