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
 * - @p max_sequence_length is the maximum length of the sum performed
 *   by the algorithm.
 *
 * @param vis Complex valued visibilities. Dimensions as above.
 * @param thresholds thresholds for first and second order estimate (extrapolation-based) methods .
 * @param flags Output flags. Dimensions as above.
 * @param status Error status.
 */


#endif //SDP_FUNC_SDP_2SM_RFI_FLAGGER_H