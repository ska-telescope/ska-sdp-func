/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SDP_FUNC_SDP_2SM_RFI_FLAGGER_H
#define SDP_FUNC_SDP_2SM_RFI_FLAGGER_H

/**
* @file sdp_2sm_rfi_flagger.h
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
 * @brief Basic RFI flagger based on two-state machine model.
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
 * @param vis Complex valued visibilities. Dimensions as above.
 * @param thresholds thresholds for first and second order estimate (extrapolation-based) methods .
 * @param flags Output flags. Dimensions as above.
 * @param status Error status.
 */
void sdp_twosm_rfi_flagger(
        const sdp_Mem* vis,
        const sdp_Mem* thresholds,
        sdp_Mem* flags,
        sdp_Error* status);

/** @} */ /* End group rfi_flag_func. */

#ifdef __cplusplus
}
#endif


#endif //SDP_FUNC_SDP_2SM_RFI_FLAGGER_H