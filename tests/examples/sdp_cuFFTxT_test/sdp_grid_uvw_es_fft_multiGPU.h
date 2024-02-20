/*
 * sdp_grid_uvw_es_fft_multiGPU.h
 *
 *  Created on: Feb. 20, 2024
 *      Author: vlad
 */

#ifndef SRC_SDP_GRID_UVW_ES_FFT_MULTIGPU_H_
#define SRC_SDP_GRID_UVW_ES_FFT_MULTIGPU_H_

#include "ska-sdp-func/utility/sdp_mem.h"

// #ifdef __cplusplus
// extern "C" {
// #endif


/* Forward-declare structure handles for private implementation. */
struct sdp_GridderUvwEsFft;
typedef struct sdp_GridderUvwEsFft sdp_GridderUvwEsFft;

void sdp_grid_uvw_es_fft_multiGPU(
        sdp_GridderUvwEsFft* plan,
        const sdp_Mem* uvw,
        const sdp_Mem* freq_hz,
        const sdp_Mem* vis,
        const sdp_Mem* weight,
        sdp_Mem* dirty_image,
        sdp_Error* status
);

// #ifdef __cplusplus
// }
// #endif

#endif /* SRC_SDP_GRID_UVW_ES_FFT_MULTIGPU_H_ */
