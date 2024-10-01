/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_FFT_PADDED_SIZE_H_
#define SKA_SDP_FFT_PADDED_SIZE_H_

/**
 * @file sdp_fft_padded_size.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup fft_padded_func
 * @{
 */

/**
 * @brief Returns the next largest even that is a power of 2, 3, 5, 7 or 11.
 *
 * @param n Minimum input grid size.
 * @param padding_factor Padding factor to multiply input grid size.
 * @return Optimal grid size.
 */
int sdp_fft_padded_size(int n, double padding_factor);

/** @} */ /* End group fft_padded_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
