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
 * @brief Basic RFI flagger implementing a simplified and experimental 
 * version of FluctuFlagger algorithm with fixed thresholds.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p vis is 4D and complex-valued, with shape:
 *   - [ num_times, num_baselines, num_freqs, num_pols]
 *
 *
 * - @p flags is 4D and integer-valued, with the same shape as @p vis .
 *
 * @param vis Complex valued visibilities. Dimensions as above.
 * @param flags Output flags. Dimensions as above.
 * @param what_quantile_for_vis the quantile for visibility 
 * magnitudes to flag anything above that (between 0 and 1)
 * @param what_quantile_for_changes the quantile for the rate
 * of fluctuations (transit score) to flag anything above that
 * (between 0 and 1)
 * @param sampling_step the intervals at which we take samples
 * (to compute the medians over the samples)
 * @param alpha the coefficient in the recursive equation for
 * transit score (shows how much the rate of changes in the 
 * previous time samples contributes to the transit score.
 * @param window number of channels on each side of a flagged
 * visibility to be flagged.
 * @param status Error status.
 */
void sdp_flagger_fixed_threshold(
        const sdp_Mem* vis,
        sdp_Mem* flags,
        const double what_quantile_for_vis,
        const double what_quantile_for_changes,
        const int sampling_step,
        const double alpha,
        const int window, 
        sdp_Error* status
);

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
