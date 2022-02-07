/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_DFT_H_
#define SKA_SDP_PROC_FUNC_DFT_H_

/**
 * @file sdp_dft.h
 */

#include "utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup dft_func
 * @{
 */

/**
 * @brief Basic prediction of visibilities from point sources using a DFT.
 *
 * This version of the function is compatible with the memory layout of
 * arrays used by RASCIL.
 *
 * Input parameters @p source_directions and @p uvw_lambda are arrays
 * of packed 3D coordinates.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p source_directions is 2D and real-valued, with shape:
 *   - [ num_components, 3 ]
 *
 * - @p source_fluxes is 3D and complex-valued, with shape:
 *   - [ num_components, num_channels, num_pols ]
 *
 * - @p uvw_lambda is 4D and real-valued, with shape:
 *   - [ num_times, num_baselines, num_channels, 3 ]
 *
 * - @p vis is 4D and complex-valued, with shape:
 *   - [ num_times, num_baselines, num_channels, num_pols ]
 *
 * @param source_directions Source direction cosines. Dimensions as above.
 * @param source_fluxes Complex source fluxes. Dimensions as above.
 * @param uvw_lambda Baseline (u,v,w) coordinates, in wavelengths.
 *                   Dimensions as above.
 * @param vis Output complex visibilities. Dimensions as above.
 * @param status Error status.
 */
void sdp_dft_point_v00(
        const sdp_Mem* source_directions,
        const sdp_Mem* source_fluxes,
        const sdp_Mem* uvw_lambda,
        sdp_Mem* vis,
        sdp_Error* status);

/** @} */ /* End group dft_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
