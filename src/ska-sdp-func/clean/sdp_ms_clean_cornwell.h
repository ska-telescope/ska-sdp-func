/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_MS_CLEAN_CORNWELL_H_
#define SKA_SDP_PROC_FUNC_MS_CLEAN_CORNWELL_H_

/**
 * @file sdp_ms_clean_cornwell.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @defgroup CLEAN
 * @{
 */

/**
 * @brief Perform the msCLEAN algorithm on a dirty image.
 *
 * Based on the version decribed by T. J. Cornwell, "Multiscale CLEAN Deconvolution of Radio Synthesis Images,"
 * in IEEE Journal of Selected Topics in Signal Processing, vol. 2, no. 5, pp. 793-801, Oct. 2008,
 * doi: 10.1109/JSTSP.2008.2006388. https://ieeexplore.ieee.org/document/4703304
 *
 * @param dirty_img Input dirty image is 2D and real-valued with square shape: [X SIZE, Y SIZE].
 * @param psf Input Point Spread Function is 2D and real-valued with square shape: [X SIZE * 2, Y SIZE * 2].
 * @param cbeam_details Input shape of CLEAN beam, with the size of the array to be generated [BMAJ, BMINN, THETA, SIZE]
 * @param scale_list List of scales to use, in pixels e.g. [0,4,8,16,32]
 * @param loop_gain Gain to be used in the CLEAN loop (typically 0.1)
 * @param threshold Minimum intensity of peak to search for, loop terminates if peak is found under this threshold. Can be set to a negative number to ensure CLEAN loop is run exactly cycle_limit times.
 * @param cycle_limit Maximum nuber of loops to perform, if the stop threshold is not reached first.
 * @param clean_model Map of CLEAN components, unconvolved pixels.
 * @param residual Residual image, flux remaining after CLEANing.
 * @param skymodel Output Skymodel, CLEAN components convolved with CLEAN beam + residuals.
 * @param status Error status.
 */
void sdp_ms_clean_cornwell(
        const sdp_Mem* dirty_img,
        const sdp_Mem* psf,
        const sdp_Mem* cbeam_details,
        const sdp_Mem* scale_list,
        const double loop_gain,
        const double threshold,
        const int cycle_limit,
        sdp_Mem* clean_model,
        sdp_Mem* residual,
        sdp_Mem* skymodel,
        sdp_Error* status
);

/** @} */ /* End group CLEAN. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
