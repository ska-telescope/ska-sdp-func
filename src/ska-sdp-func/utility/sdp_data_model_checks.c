/* See the LICENSE file at the top-level directory of this distribution. */

#include "ska-sdp-func/utility/sdp_data_model_checks.h"
#include "ska-sdp-func/utility/sdp_logging.h"


void sdp_data_model_check_uvw_at(
        const sdp_Mem* uvw,
        sdp_MemType expected_type,
        sdp_MemLocation expected_location,
        int64_t expected_num_times,
        int64_t expected_num_baselines,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
)
{
    if (*status) return;
    if (!sdp_mem_is_floating_point(uvw) || sdp_mem_is_complex(uvw))
    {
        *status = SDP_ERR_DATA_TYPE;
        sdp_log_message(SDP_LOG_LEVEL_ERROR, stderr, func, file, line,
                "%s: The uvw array must be real-valued", func
        );
    }
    const int32_t num_dims = 3;
    const int64_t uvw_shape[] = {expected_num_times, expected_num_baselines, 3};
    sdp_mem_check_shape_at(uvw, num_dims, uvw_shape,
            status, expr, func, file, line
    );
    sdp_mem_check_location_at(uvw, expected_location,
            status, expr, func, file, line
    );
    if (expected_type != SDP_MEM_VOID)
    {
        sdp_mem_check_type_at(uvw, expected_type,
                status, expr, func, file, line
        );
    }
    sdp_mem_check_c_contiguity_at(uvw, status, expr, func, file, line);
}


void sdp_data_model_get_uvw_metadata(
        const sdp_Mem* uvw,
        sdp_MemType* type,
        sdp_MemLocation* location,
        int64_t* num_times,
        int64_t* num_baselines,
        sdp_Error* status
)
{
    if (*status) return;
    if (!sdp_mem_is_floating_point(uvw) || sdp_mem_is_complex(uvw))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("The uvw array must be real-valued");
    }
    const int32_t uvw_num_dims = 3;
    const int32_t uvw_coord_dim = 2;
    const int32_t uvw_coord_size = 3;
    sdp_mem_check_num_dims(uvw, uvw_num_dims, status);
    sdp_mem_check_dim_size(uvw, uvw_coord_dim, uvw_coord_size, status);

    if (type) *type = sdp_mem_type(uvw);
    if (location) *location = sdp_mem_location(uvw);
    if (num_times) *num_times = sdp_uvw_num_times(uvw);
    if (num_baselines) *num_baselines = sdp_uvw_num_baselines(uvw);
}


void sdp_data_model_check_vis_at(
        const sdp_Mem* vis,
        sdp_MemType expected_type,
        sdp_MemLocation expected_location,
        int64_t expected_num_times,
        int64_t expected_num_baselines,
        int64_t expected_num_channels,
        int64_t expected_num_pols,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
)
{
    if (*status) return;
    if (!sdp_mem_is_complex(vis))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("The visibility array must be complex");
        return;
    }
    const int32_t num_dims = 4;
    const int64_t vis_shape[] = {
        expected_num_times,
        expected_num_baselines,
        expected_num_channels,
        expected_num_pols
    };
    sdp_mem_check_shape_at(vis, num_dims, vis_shape,
            status, expr, func, file, line
    );
    sdp_mem_check_location_at(vis, expected_location,
            status, expr, func, file, line
    );
    if (expected_type != SDP_MEM_VOID)
    {
        sdp_mem_check_type_at(vis, expected_type,
                status, expr, func, file, line
        );
    }
    sdp_mem_check_c_contiguity_at(vis, status, expr, func, file, line);
}


void sdp_data_model_get_vis_metadata(
        const sdp_Mem* vis,
        sdp_MemType* type,
        sdp_MemLocation* location,
        int64_t* num_times,
        int64_t* num_baselines,
        int64_t* num_channels,
        int64_t* num_pols,
        sdp_Error* status
)
{
    if (*status) return;
    if (!sdp_mem_is_complex(vis))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("The visibility array must be complex");
        return;
    }
    const int32_t vis_dims = 4;
    sdp_mem_check_num_dims(vis, vis_dims, status);
    if (type) *type = sdp_mem_type(vis);
    if (location) *location = sdp_mem_location(vis);
    if (num_times) *num_times = sdp_vis_num_times(vis);
    if (num_baselines) *num_baselines = sdp_vis_num_baselines(vis);
    if (num_channels) *num_channels = sdp_vis_num_channels(vis);
    if (num_pols)
    {
        *num_pols = sdp_vis_num_pols(vis);

        if (!(*status) // if status is set do nothing
                && *num_pols != 4
                && *num_pols != 1)
        {
            *status = SDP_ERR_INVALID_ARGUMENT;
            SDP_LOG_ERROR("The number of polarisations should be 4 or 1");
            return;
        }
    }
}


void sdp_data_model_check_weights_at(
        const sdp_Mem* weights,
        sdp_MemType expected_type,
        sdp_MemLocation expected_location,
        int64_t expected_num_times,
        int64_t expected_num_baselines,
        int64_t expected_num_channels,
        int64_t expected_num_pols,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
)
{
    if (*status) return;
    if (sdp_mem_is_complex(weights))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("The weights array cannot be complex");
        return;
    }
    const int32_t num_dims = 4;
    const int64_t weights_shape[] = {
        expected_num_times,
        expected_num_baselines,
        expected_num_channels,
        expected_num_pols
    };
    sdp_mem_check_shape_at(weights, num_dims, weights_shape,
            status, expr, func, file, line
    );
    sdp_mem_check_location_at(weights, expected_location,
            status, expr, func, file, line
    );
    if (expected_type != SDP_MEM_VOID)
    {
        sdp_mem_check_type_at(weights, expected_type,
                status, expr, func, file, line
        );
    }
    sdp_mem_check_c_contiguity_at(weights, status, expr, func, file, line);
}


void sdp_data_model_get_weights_metadata(
        const sdp_Mem* weights,
        sdp_MemType* type,
        sdp_MemLocation* location,
        int64_t* num_times,
        int64_t* num_baselines,
        int64_t* num_channels,
        int64_t* num_pols,
        sdp_Error* status
)
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
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("The weights array cannot be complex");
        return;
    }
    if (type) *type = sdp_mem_type(weights);
    if (location) *location = sdp_mem_location(weights);
    if (num_times) *num_times = sdp_weights_num_times(weights);
    if (num_baselines) *num_baselines = sdp_weights_num_baselines(weights);
    if (num_channels) *num_channels = sdp_weights_num_channels(weights);
    if (num_pols) *num_pols = sdp_weights_num_pols(weights);
}
