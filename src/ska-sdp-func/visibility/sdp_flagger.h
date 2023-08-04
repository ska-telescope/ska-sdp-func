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
 * @brief Basic RFI flagger based on two-state machine model.
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
 * @param vis Complex valued visibilities. Dimensions as above.
 * @param thresholds thresholds for time and frequency domains.
 * @param flags Output flags. Dimensions as above.
 * @param status Error status.
 */
void sdp_flagger(
        const sdp_Mem* vis,
        const sdp_Mem* parameters,
        sdp_Mem* flags,
        const sdp_Mem* antennas,
        const sdp_Mem* baselines1,
        const sdp_Mem* baselines2,
        sdp_Error* status);

/** @} */ /* End group flag_func. */

#ifdef __cplusplus
}
#endif

#endif
