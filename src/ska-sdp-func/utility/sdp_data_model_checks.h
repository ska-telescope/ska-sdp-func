/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_DATA_MODEL_CHECKS_H_
#define SKA_SDP_PROC_FUNC_DATA_MODEL_CHECKS_H_

/**
 * @file sdp_data_model_checks.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup data_checks
 * @{
 */

/**
 * @brief Check uvw coordinate array matches data model convention.
 *
 * Parameter @p uvw should be an array of packed 3D coordinates.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p uvw is 3D and real-valued, with shape:
 *   - [ num_times, num_baselines, 3 ]
 *
 * @param uvw Baseline (u,v,w) coordinates. Dimensions as above.
 * @param type Enumerated type of data.
 * @param location Enumerated location of data.
 * @param num_times Number of time samples in data.
 * @param num_baselines Number of baselines in data.
 * @param status Error status.
 */
void sdp_data_model_check_uvw(
        const sdp_Mem* uvw,
        sdp_MemType* type,
        sdp_MemLocation* location,
        int64_t* num_times,
        int64_t* num_baselines,
        sdp_Error* status
);

/**
 * @brief Check visibility array matches data model convention.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p vis is 4D and complex-valued, with shape:
 *   - [ num_times, num_baselines, num_channels, num_pols ]
 *
 * @param vis Complex visibility data. Dimensions as above.
 * @param type Enumerated type of data.
 * @param location Enumerated location of data.
 * @param num_times Number of time samples in data.
 * @param num_baselines Number of baselines in data.
 * @param num_channels Number of channels in data.
 * @param num_pols Number of polarisations in data.
 * @param status Error status.
 */
void sdp_data_model_check_vis(
        const sdp_Mem* vis,
        sdp_MemType* type,
        sdp_MemLocation* location,
        int64_t* num_times,
        int64_t* num_baselines,
        int64_t* num_channels,
        int64_t* num_pols,
        sdp_Error* status
);

/**
 * @brief Check weights array matches data model convention.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p weights is 4D and real-valued, with shape:
 *   - [ num_times, num_baselines, num_channels, num_pols ]
 *
 * @param weights Visibility weights. Dimensions as above.
 * @param type Enumerated type of data.
 * @param location Enumerated location of data.
 * @param num_times Number of time samples in data.
 * @param num_baselines Number of baselines in data.
 * @param num_channels Number of channels in data.
 * @param num_pols Number of polarisations in data.
 * @param status Error status.
 */
void sdp_data_model_check_weights(
        const sdp_Mem* weights,
        sdp_MemType* type,
        sdp_MemLocation* location,
        int64_t* num_times,
        int64_t* num_baselines,
        int64_t* num_channels,
        int64_t* num_pols,
        sdp_Error* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
