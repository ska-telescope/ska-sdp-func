/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_RFI_FLAGGER_H_
#define SKA_SDP_PROC_FUNC_RFI_FLAGGER_H_

/**
 * @file sdp_rfi_flagger.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup rfi_flag_func
 * @{
 */

/**
 * @brief Basic RFI flagger based on sum-threshold algorithm.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p vis is 4D and complex-valued, with shape:
 *   - [ num_timesamples, num_baselines, num_channels, num_polarisations ]
 *
 * - @p thresholds is 1D and real-valued.
 *   - The size of the array is n, where 2^(n-1) = @p max_sequence_length .
 *
 * - @p flags is 4D and integer-valued, with the same shape as @p vis .
 *
 * - @p max_sequence_length is the maximum length of the sum performed
 *   by the algorithm.
 *
 * @param vis Complex valued visibilities. Dimensions as above.
 * @param thresholds List of thresholds, one for each sequence length.
 * @param flags Output flags. Dimensions as above.
 * @param max_sequence_length Maximum length of the partial sum.
 * @param status Error status.
 */
void sdp_sum_threshold_rfi_flagger(
        const sdp_Mem* vis,
        const sdp_Mem* thresholds,
        sdp_Mem* flags,
        const int64_t max_sequence_length,
        sdp_Error* status);

/** @} */ /* End group rfi_flag_func. */

#ifdef __cplusplus
}
#endif

#endif
