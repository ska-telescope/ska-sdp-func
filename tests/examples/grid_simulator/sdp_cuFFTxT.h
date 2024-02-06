/*
 * sdp_cuFFTxT.h
 *
 *  Created on: 6 Feb 2024
 *      Author: vlad
 */

#ifndef SDP_CUFFTXT_H_
#define SDP_CUFFTXT_H_

/**
 * @file sdp_function_name.h
 *       (Change this to match the name of the header file)
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Function demonstrator that uses cuFFTxT.
 *
 * The function uses cuFFTxT API to perform inverse FFT of
 * the input 2D complex array with gridded visibilities
 * to construct the dirty image. 
 *
 * @param grid_sim Input grid (2D SDP_MEM_COMPLEX_DOUBLE).
 * @param image_out Output image (2D SDP_MEM_COMPLEX_DOUBLE).
 * @param status Error status.
 */
void sdp_cuFFTxT(
		sdp_Mem* grid_sim,
		sdp_Mem* image_out,
		sdp_Error* status);

#ifdef __cplusplus
}
#endif

#endif /* SDP_CUFFTXT_H_ */
