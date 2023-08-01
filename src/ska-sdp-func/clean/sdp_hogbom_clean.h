/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_HOGBOM_CLEAN_H_
#define SKA_SDP_PROC_FUNC_HOGBOM_CLEAN_H_

/**
 * @file sdp_hogbom_clean.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Perform the Hogbom CLEAN algorithm on a dirty image
 *
 * @param dirty_img Input dirty image.
 * @param psf Input Point Spread Function.
 * @param cbeam_details Input shape of cbeam [BMAJ, BMINN, THETA]
 * @param loop_gain Gain to be used in the CLEAN loop (typically 0.1)
 * @param threshold Minimum intensity of peak to search for, loop terminates if peak is found under this threshold. 
 * @param cycle_limit Maximum nuber of loops to perform, if the stop threshold is not reached first.
 * @param skymodel Output Skymodel (CLEANed image).
 * @param use_bfloat Use bfloat16 in CLEAN algorithm (only available on GPU)
 * @param status Error status.
 */
void sdp_hogbom_clean(
        const sdp_Mem* dirty_img,
        const sdp_Mem* psf,
        const sdp_Mem* cbeam_details,
        const double loop_gain,
        const double threshold,
        const int cycle_limit,
        sdp_Mem* skymodel,
        const bool use_bfloat,
        sdp_Error* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
