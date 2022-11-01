/* See the LICENSE file at the top-level directory of this distribution. */

#include "ska-sdp-func/utility/sdp_data_model_checks.h"
#include "ska-sdp-func/utility/sdp_logging.h"


void sdp_data_model_check_uvw(
        const sdp_Mem* uvw,
        sdp_MemType* type,
        sdp_MemLocation* location,
        int64_t* num_times,
        int64_t* num_baselines,
        sdp_Error* status)
{
    if (*status) return;
    if (sdp_mem_num_dims(uvw) != 3)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The uvw array must be 3D");
        return;
    }
    if (sdp_mem_shape_dim(uvw, 2) != 3)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR(
                "The last dimension of the uvw array must be of length 3");
        return;
    }
    if (sdp_mem_type(uvw) != SDP_MEM_DOUBLE &&
            sdp_mem_type(uvw) != SDP_MEM_FLOAT)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The uvw array must be real-valued");
        return;
    }
    if (type) *type = sdp_mem_type(uvw);
    if (location) *location = sdp_mem_location(uvw);
    if (num_times) *num_times = sdp_mem_shape_dim(uvw, 0);
    if (num_baselines) *num_baselines = sdp_mem_shape_dim(uvw, 1);
}


void sdp_data_model_check_vis(
        const sdp_Mem* vis,
        sdp_MemType* type,
        sdp_MemLocation* location,
        int64_t* num_times,
        int64_t* num_baselines,
        int64_t* num_channels,
        int64_t* num_pols,
        sdp_Error* status)
{
    if (*status) return;
    if (sdp_mem_num_dims(vis) != 4)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The visibility array must be 4D");
        return;
    }
    if (!sdp_mem_is_complex(vis))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The visibility array must be complex");
        return;
    }
    if (type) *type = sdp_mem_type(vis);
    if (location) *location = sdp_mem_location(vis);
    if (num_times) *num_times = sdp_mem_shape_dim(vis, 0);
    if (num_baselines) *num_baselines = sdp_mem_shape_dim(vis, 1);
    if (num_channels) *num_channels = sdp_mem_shape_dim(vis, 2);
    if (num_pols) *num_pols = sdp_mem_shape_dim(vis, 3);
}


void sdp_data_model_check_weights(
        const sdp_Mem* weights,
        sdp_MemType* type,
        sdp_MemLocation* location,
        int64_t* num_times,
        int64_t* num_baselines,
        int64_t* num_channels,
        int64_t* num_pols,
        sdp_Error* status)
{
    if (*status) return;
    if (sdp_mem_num_dims(weights) != 4)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The weights array must be 4D");
        return;
    }
    if (sdp_mem_is_complex(weights))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The weights array cannot be complex");
        return;
    }
    if (type) *type = sdp_mem_type(weights);
    if (location) *location = sdp_mem_location(weights);
    if (num_times) *num_times = sdp_mem_shape_dim(weights, 0);
    if (num_baselines) *num_baselines = sdp_mem_shape_dim(weights, 1);
    if (num_channels) *num_channels = sdp_mem_shape_dim(weights, 2);
    if (num_pols) *num_pols = sdp_mem_shape_dim(weights, 3);
}
