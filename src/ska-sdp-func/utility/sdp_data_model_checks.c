/* See the LICENSE file at the top-level directory of this distribution. */

#include "ska-sdp-func/utility/sdp_data_model_checks.h"
#include "ska-sdp-func/utility/sdp_logging.h"


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
)
{
    if (*status) return;

    sdp_mem_check_c_contiguity_at(uvw, status, expr, func, file, line);

    const int32_t num_dims = 3;
    int64_t uvw_shape[3] = {
        expected_num_timesamples,
        expected_num_baselines,
        3
    };
    sdp_mem_check_shape_at(uvw,
            num_dims,
            uvw_shape,
            status,
            expr,
            func,
            file,
            line
    );

    if (*status) return;
    if (!sdp_mem_is_floating_point(uvw)
            || sdp_mem_is_complex(uvw))
    {
        *status = SDP_ERR_DATA_TYPE;
        sdp_log_message(
                SDP_LOG_LEVEL_ERROR,
                stderr,
                func,
                file,
                line,
                "%s: The uvw array must be real-valued",
                func
        );
    }

    sdp_mem_check_location_at(uvw,
            expected_location,
            status,
            expr,
            func,
            file,
            line
    );

    if (expected_type != SDP_MEM_VOID)
    {
        sdp_mem_check_type_at(
                uvw, expected_type, status, "uvw", func, file, line
        );
    }
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

    const int32_t uvw_dims = 3;
    sdp_mem_check_num_dims_at(uvw, uvw_dims, status, "uvw", CODE_POS);

    const int32_t uvw_dim = 2;
    const int32_t uvw_coord_size = 3;
    sdp_mem_check_dim_size_at(
            uvw, uvw_dim, uvw_coord_size, status, "uvw", CODE_POS
    );

    if (!(*status) // if status is set do nothing
            && (!sdp_mem_is_floating_point(uvw)
            || sdp_mem_is_complex(uvw)))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("The uvw array must be real-valued");
    }

    if (type) *type = sdp_mem_type(uvw);
    if (location) *location = sdp_mem_location(uvw);
    if (num_times) *num_times = sdp_mem_shape_dim(uvw, 0);
    if (num_baselines) *num_baselines = sdp_mem_shape_dim(uvw, 1);
}


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
)
{
    if (*status) return;

    sdp_mem_check_c_contiguity_at(vis, status, expr, func, file, line);

    const int32_t num_dims = 4;
    int64_t vis_shape[4] = {
        expected_num_timesamples,
        expected_num_baselines,
        expected_num_channels,
        expected_num_pols
    };
    sdp_mem_check_shape_at(vis,
            num_dims,
            vis_shape,
            status,
            expr,
            func,
            file,
            line
    );

    if (!(*status) // if status is set do nothing
            && !sdp_mem_is_complex(vis))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("The visibility array must be complex");
        return;
    }

    sdp_mem_check_location_at(vis,
            expected_location,
            status,
            expr,
            func,
            file,
            line
    );

    if (expected_type != SDP_MEM_VOID)
    {
        sdp_mem_check_type_at(
                vis, expected_type, status, expr, func, file, line
        );
    }
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
    const int32_t vis_dims = 4;
    sdp_mem_check_num_dims_at(vis, vis_dims, status, "vis", CODE_POS);

    if (!sdp_mem_is_complex(vis))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("The visibility array must be complex");
        return;
    }

    if (type) *type = sdp_mem_type(vis);
    if (location) *location = sdp_mem_location(vis);
    if (num_times) *num_times = sdp_mem_shape_dim(vis, 0);
    if (num_baselines) *num_baselines = sdp_mem_shape_dim(vis, 1);
    if (num_channels) *num_channels = sdp_mem_shape_dim(vis, 2);
    if (num_pols)
    {
        *num_pols = sdp_mem_shape_dim(vis, 3);

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


void sdp_data_model_check_weights(
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
