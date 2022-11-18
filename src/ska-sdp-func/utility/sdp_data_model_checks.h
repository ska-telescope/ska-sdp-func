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
 * @brief Returns the number of timesamples for sdp_Mem containing
 * uvw Baseline (u,v,w) coordinates.
 *
 * Parameter @p uvw coordinate array that matches data model convention.
 *
 * @param uvw uvw Baseline (u,v,w) coordinates.
 */
#define sdp_uvw_timesamples(uvw) sdp_mem_shape_dim(uvw, 0)

/**
 * @brief Returns the number of baselines for sdp_Mem containing
 * uvw Baseline (u,v,w) coordinates.
 *
 * Parameter @p uvw coordinate array that matches data model convention.
 *
 * @param uvw uvw Baseline (u,v,w) coordinates.
 */
#define sdp_uvw_baselines(uvw) sdp_mem_shape_dim(uvw, 1)

/**
 * @brief Returns the number of coordinates for sdp_Mem containing
 * uvw Baseline (u,v,w) coordinates.
 *
 * Parameter @p uvw coordinate array that matches data model convention.
 *
 * @param uvw uvw Baseline (u,v,w) coordinates.
 */
#define sdp_uvw_coord(uvw) sdp_mem_shape_dim(uvw, 2)

/**
 * @brief Checks if uvw coordinate array matches data model convention
 *        and other required parameters.
 *
 * Parameter @p uvw should be an array of packed 3D coordinates.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p uvw is 3D and real-valued, with shape:
 *   - [ num_times, num_baselines, 3 ]
 *
 * Use sdp_data_model_check_uvw(...) macro to automatically fill
 * ``func``, ``file`` and ``line`` by call location.
 *
 * To bypass datatype check pass expected_type = SDP_MEM_VOID
 *
 * @param uvw Baseline (u,v,w) coordinates. Dimensions as above.
 * @param type Enumerated type of data.
 * @param location Enumerated location of data.
 * @param num_times Number of time samples in data.
 * @param num_baselines Number of baselines in data.
 * @param status Error status.
 * @param expr Expression string to report in error message.
 * @param func Function to report in error message.
 * @param file File name to report in error message.
 * @param line Line to report in error message.
 */
void sdp_data_model_check_uvw_at(
        const sdp_Mem* uvw,
        sdp_MemType expected_type,
        sdp_MemLocation expected_location,
        int64_t expected_num_timesamples,
        int64_t expected_num_baselines,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
);

/**
 * @brief Checks if uvw coordinate array matches data model convention
 *        and other required parameters.
 *
 * Parameter @p uvw should be an array of packed 3D coordinates.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p uvw is 3D and real-valued, with shape:
 *   - [ num_times, num_baselines, 3 ]
 *
 * To bypass datatype check pass expected_type = SDP_MEM_VOID
 *
 * @param uvw Baseline (u,v,w) coordinates. Dimensions as above.
 * @param type Enumerated type of data.
 * @param location Enumerated location of data.
 * @param num_times Number of time samples in data.
 * @param num_baselines Number of baselines in data.
 * @param status Error status.
 */
#define sdp_data_model_check_uvw(uvw, \
            expected_type, \
            expected_location, \
            expected_num_timesamples, \
            expected_num_baselines, \
            status)                     \
    sdp_data_model_check_uvw_at(uvw, \
        expected_type, \
        expected_location, \
        expected_num_timesamples, \
        expected_num_baselines, \
        status, \
        #uvw, \
        __func__, \
        __FILE__, \
        __LINE__ \
    )

/**
 * @brief Checks if uvw coordinate array matches data model convention
 * and returns its metadata.
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
void sdp_data_model_get_uvw_metadata(
        const sdp_Mem* uvw,
        sdp_MemType* type,
        sdp_MemLocation* location,
        int64_t* num_times,
        int64_t* num_baselines,
        sdp_Error* status
);

/**
 * @brief Returns the number of timesamples in visibility data array.
 *
 * Parameter @p vis is visibility data array that match data model convention.
 *
 * @param vis visibility data.
 */
#define sdp_vis_timesamples(vis) sdp_mem_shape_dim(vis, 0)

/**
 * @brief Returns the number of baselines in visibility data array.
 *
 * Parameter @p vis is visibility data array that match data model convention.
 *
 * @param vis visibility data.
 */
#define sdp_vis_baselines(vis) sdp_mem_shape_dim(vis, 1)

/**
 * @brief Returns the number of channels in visibility data array.
 *
 * Parameter @p vis is visibility data array that match data model convention.
 *
 * @param vis visibility data.
 */
#define sdp_vis_channels(vis) sdp_mem_shape_dim(vis, 2)

/**
 * @brief Returns the number of polarisations in visibility data array.
 *
 * Parameter @p vis is visibility data array that match data model convention.
 *
 * @param vis visibility data.
 */
#define sdp_vis_pols(vis) sdp_mem_shape_dim(vis,3)

/**
 * @brief Checks if visibility array matches data model convention.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p vis is 4D and complex-valued, with shape:
 *   - [ num_times, num_baselines, num_channels, num_pols ]
 *
 * Use sdp_data_model_check_visibility(...) macro to automatically fill
 * ``func``, ``file`` and ``line`` by call location.
 *
 * To bypass datatype check pass expected_type = SDP_MEM_VOID
 *
 * @param vis Complex visibility data. Dimensions as above.
 * @param expected_type Expected enumerated type of data.
 * @param expected_location Expected enumerated location of data.
 * @param expected_num_timesamples Expected number of time samples in data.
 * @param expected_num_baselines Expected number of baselines in data.
 * @param expected_num_channels Expected number of channels in data.
 * @param expected_num_pols Expected number of polarisations in data.
 * @param status Error status.
 * @param expr Expression string to report in error message
 * @param func Function to report in error message.
 * @param file File name to report in error message.
 * @param line Line to report in error message.
 */
void sdp_data_model_check_visibility_at(
        const sdp_Mem* vis,
        sdp_MemType expected_type,
        sdp_MemLocation expected_location,
        int64_t expected_num_timesamples,
        int64_t expected_num_baselines,
        int64_t expected_num_channels,
        int64_t expected_num_pols,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
);

/**
 * @brief Checks if visibility array matches data model convention.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p vis is 4D and complex-valued, with shape:
 *   - [ num_times, num_baselines, num_channels, num_pols ]
 *
 * To bypass datatype check pass expected_type = SDP_MEM_VOID
 *
 * @param vis Complex visibility data. Dimensions as above.
 * @param expected_type Expected enumerated type of data.
 * @param expected_location Expected enumerated location of data.
 * @param expected_num_timesamples Expected number of time samples in data.
 * @param expected_num_baselines Expected number of baselines in data.
 * @param expected_num_channels Expected number of channels in data.
 * @param expected_num_pols Expected number of polarisations in data.
 * @param status Error status.
 */
 #define sdp_data_model_check_visibility(vis, \
            expected_type, \
            expected_location, \
            expected_num_timesamples, \
            expected_num_baselines, \
            expected_num_channels, \
            expected_num_pols, \
            status)                     \
    sdp_data_model_check_visibility_at(vis, \
        expected_type, \
        expected_location, \
        expected_num_timesamples, \
        expected_num_baselines, \
        expected_num_channels, \
        expected_num_pols, \
        status, \
        #vis, \
        __func__, \
        __FILE__, \
        __LINE__ \
    )

/**
 * @brief Checks if visibility array matches data model convention
 * and return its metadata.
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
void sdp_data_model_get_vis_metadata(
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
 * @brief Returns the number of timesamples in weights data array.
 *
 * Parameter @p weights is weights data array that match data model convention.
 *
 * @param weights weights data.
 */
#define sdp_weights_timesamples(weights) sdp_mem_shape_dim(weights, 0)

/**
 * @brief Returns the number of baselines in weights data array.
 *
 * Parameter @p weights is weights data array that match data model convention.
 *
 * @param weights weights data.
 */
#define sdp_weights_baselines(weights) sdp_mem_shape_dim(weights, 1)

/**
 * @brief Returns the number of channels in weights data array.
 *
 * Parameter @p weights is weights data array that match data model convention.
 *
 * @param weights weights data.
 */
#define sdp_weights_channels(weights) sdp_mem_shape_dim(weights, 2)

/**
 * @brief Returns the number of polarisations in weights data array.
 *
 * Parameter @p weights is weights data array that match data model convention.
 *
 * @param weights weights data.
 */
#define sdp_weights_pols(weights) sdp_mem_shape_dim(weights,3)

/**
 * @brief Check weights array matches data model convention.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p weights is 4D and real-valued, with shape:
 *   - [ num_times, num_baselines, num_channels, num_pols ]
 *
 * Use sdp_data_model_check_weight(...) macro to automatically fill
 * ``func``, ``file`` and ``line`` by call location.
 *
 * To bypass datatype check pass expected_type = SDP_MEM_VOID
 *
 * @param weights Visibility weights. Dimensions as above.
 * @param expected_type Enumerated type of data.
 * @param expected_location Enumerated location of data.
 * @param expected_num_timesamples Number of time samples in data.
 * @param expected_num_baselines Number of baselines in data.
 * @param expected_num_channels Number of channels in data.
 * @param expected_num_pols Number of polarisations in data.
 * @param expr Expression string to report in error message
 * @param func Function to report in error message.
 * @param file File name to report in error message.
 * @param line Line to report in error message.
 * @param status Error status.
 */
void sdp_data_model_check_weights_at(
        const sdp_Mem* weights,
        sdp_MemType expected_type,
        sdp_MemLocation expected_location,
        int64_t expected_num_timesamples,
        int64_t expected_num_baselines,
        int64_t expected_num_channels,
        int64_t expected_num_pols,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
);

/**
 * @brief Check weights array matches data model convention.
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p weights is 4D and real-valued, with shape:
 *   - [ num_times, num_baselines, num_channels, num_pols ]
 *
 * To bypass datatype check pass expected_type = SDP_MEM_VOID
 *
 * @param weights Visibility weights. Dimensions as above.
 * @param expected_type Enumerated type of data.
 * @param expected_location Enumerated location of data.
 * @param expected_num_timesamples Number of time samples in data.
 * @param expected_num_baselines Number of baselines in data.
 * @param expected_num_channels Number of channels in data.
 * @param expected_num_pols Number of polarisations in data.
 * @param expr Expression string to report in error message
 * @param func Function to report in error message.
 * @param file File name to report in error message.
 * @param line Line to report in error message.
 * @param status Error status.
 */
 #define sdp_data_model_check_weights(weights, \
            expected_type, \
            expected_location, \
            expected_num_timesamples, \
            expected_num_baselines, \
            expected_num_channels, \
            expected_num_pols, \
            status)                     \
    sdp_data_model_check_weights_at(weights, \
        expected_type, \
        expected_location, \
        expected_num_timesamples, \
        expected_num_baselines, \
        expected_num_channels, \
        expected_num_pols, \
        status, \
        #weights, \
        __func__, \
        __FILE__, \
        __LINE__ \
    )

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
void sdp_data_model_get_weights_metadata(
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
