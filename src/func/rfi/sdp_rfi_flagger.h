
/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_RFI_FLAGGER_H_
#define SKA_SDP_PROC_FUNC_RFI_FLAGGER_H_

/**
 * @file sdp_rfi_flagger.h
 */

#include "utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Simple example 
 *
 * @param vis Spectrograms from the visibility data.
 * @param sequence The length of continuous channels used for flagging
 * @param thresholds The thresholds used for flagging
 * @param flags	The output flags are stored in flags.
 * @param status Error status.
 */
void sdp_rfi_flagger(
	       	const sdp_Mem* vis,
	       	const sdp_Mem* sequence,
	       	const sdp_Mem* thresholds,
		sdp_Mem*  flags,
        	sdp_Error* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
