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
 * @enum sdp_FftType
 *
 * @brief Enumerator to specify the type of FFT to perform.
 */
enum sdp_FftType
{
    //! Complex-to-complex type.
    SDP_FFT_C2C = 0
};

/**
 * @brief Creates a plan for FFTs.
 *
 * This wraps cuFFT, so only GPU FFTs are currently supported.
 *
 * @param precision The enumerated precision for the FFT (single or double).
 * @param location The enumerated location for the FFT (CPU or GPU).
 * @param fft_type The enumerated FFT type.
 * @param num_dims The number of dimensions for the FFT.
 * @param dim_size The size of each dimension.
 * @param batch_size The batch size.
 * @param is_forward Set true if FFT should be "forward", false for "inverse".
 * @param status Error status.
 *
 * @return sdp_Fft* Handle to FFT plan.
 */
sdp_Fft* sdp_fft_create(
        sdp_MemType precision,
        sdp_MemLocation location,
        sdp_FftType fft_type,
        int num_dims,
        const int64_t* dim_size,
        int batch_size,
        int is_forward,
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
