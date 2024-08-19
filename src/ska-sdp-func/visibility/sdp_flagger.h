/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SDP_FUNC_SDP_FLAGGER_H_
#define SDP_FUNC_SDP_FLAGGER_H_

/**
 * @file sdp_flagger.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @defgroup flag_func
 * @{
 */

/**
 * @brief Basic RFI flagger implementing the FluctuFlagger algorithm
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p vis is 2D and complex-valued, with shape:
 *   - [ num_timesamples, num_channels]
 *
 * - @p thresholds is 1D and real-valued.
 *   - The size of the array is n, where 2^(n-1) = @p max_sequence_length .
 *
 * - @p flags is 2D and integer-valued, with the same shape as @p vis .
 *
 * @param vis complex valued visibilities. Dimensions as above.
 * @param flags output flags. Dimensions as above.
 * @param alpha coefficient for the recursive transit score value
 * @param threshold_magnitudes threshold on the modified z-score
 * for magnitudes
 * @param threshold_variations threshold on the modified z-score
 * for variations in the magnitudes
 * @param threshold_broadband threshold on the modified z-score
 * for median history
 * @param window the number of channels on each side of a flagged
 * visibility to be flagged
 * @param window_median_history the size of the window of time
 * samples for which the median across all channels is maintained
 *
 * @param status Error status.
 */
void sdp_flagger_dynamic_threshold(
        const sdp_Mem* vis,
        sdp_Mem* flags,
        const double alpha,
        const double threshold_magnitudes,
        const double threshold_variations,
        const double threshold_broadband,
        const int sampling_step,
        const int window,
        const int window_median_history,
        sdp_Error* status
);

/** @} */ /* End group flag_func. */


#ifdef __cplusplus
}
#endif

#endif
