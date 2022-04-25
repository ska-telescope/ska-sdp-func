
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
 * @brief Basic RFI flagger based on sum-threshold algorithm.
 *
 * Input parameters @p vis and @p flags are 4D arrays.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p vis is 4D and complex-valued, with shape:
 *   - [ num_timesamples, num_baselines, num_channels, num_polarisations ]
 *
 * @param vis Complex valued visibilities.  Dimensions as above.
 * @param thresholds The list of thresholds used for flagging.
 * @param flags The output flags are stored in flags.
 * @param max_sequence_length Size of the window.
 * @param status Error status.
 */
void sdp_rfi_flagger(
        const sdp_Mem* vis,
        const sdp_Mem* thresholds,
        sdp_Mem* flags,
        const int64_t max_sequence_length,
        sdp_Error* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
