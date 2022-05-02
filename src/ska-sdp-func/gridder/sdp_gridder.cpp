/* See the LICENSE file at the top-level directory of this distribution. */

#include "ska-sdp-func/gridder/sdp_gridder.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"

struct sdp_Gridder
{
    // const sdp_Mem* uvw;
    // const sdp_Mem* freq_hz;  // in Hz
    // const sdp_Mem* vis;
    // const sdp_Mem* weight;
	float pixsize_x_rad; 
	float pixsize_y_rad;
	float epsilon;
    float* workarea;
};

void sdp_gridder_check_inputs(
		const sdp_Mem* uvw,
		const sdp_Mem* freq_hz,  // in Hz
		const sdp_Mem* vis,
		const sdp_Mem* weight,
        sdp_Error* status)
{
    SDP_LOG_DEBUG("Checking inputs...");

	// check location of parameters (CPU or GPU)
    const sdp_MemLocation location = sdp_mem_location(uvw);
	
    if (location != sdp_mem_location(freq_hz) || 
		location != sdp_mem_location(vis) || 
		location != sdp_mem_location(weight))
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Memory location mismatch");
        return;
    }
	
	// check types of parameters (real or complex)
    if (sdp_mem_is_complex(uvw))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("uvw values must be real");
        return;
    }
    if (sdp_mem_is_complex(freq_hz))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Frequency values must be real");
        return;
    }
    if (!sdp_mem_is_complex(vis))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Visibility values must be complex");
        return;
    }
    if (sdp_mem_is_complex(weight))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Weight values must be real");
        return;
    }
	
	// check shapes of parameters
    const int64_t num_vis      = sdp_mem_shape_dim(vis, 0);
    const int64_t num_channels = sdp_mem_shape_dim(vis, 1);
	
	SDP_LOG_DEBUG("vis is %i by %i", num_vis, num_channels);
	SDP_LOG_DEBUG("freq_hz is %i by %i", 
		sdp_mem_shape_dim(freq_hz, 0), 
		sdp_mem_shape_dim(freq_hz, 1));
		
    if (sdp_mem_shape_dim(uvw, 0) != num_vis)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("The number of rows in uvw and vis must match.");
        return;
    }
    if (sdp_mem_shape_dim(uvw, 1) != 3)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("uvw must be N x 3.");
        return;
    }
    if (sdp_mem_shape_dim(freq_hz, 0) != num_channels)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("The number of channels in vis and freq_hz must match.");
        return;
    }

	// check contiguity
    if (!sdp_mem_is_c_contiguous(uvw) ||
        !sdp_mem_is_c_contiguous(freq_hz) ||
        !sdp_mem_is_c_contiguous(vis) ||
        !sdp_mem_is_c_contiguous(weight))
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("All input arrays must be C contiguous");
        return;
    }
}

void sdp_gridder_check_outputs(
		const sdp_Mem* uvw,
		const sdp_Mem* dirty_image,
        sdp_Error* status)
{
    SDP_LOG_DEBUG("Checking outputs...");

    const sdp_MemLocation location = sdp_mem_location(uvw);
	
    if (location != sdp_mem_location(dirty_image))
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Memory location mismatch");
        return;
    }

    if (sdp_mem_is_complex(dirty_image))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Dirty image values must be real");
        return;
    }
    if (sdp_mem_shape_dim(dirty_image, 0) != sdp_mem_shape_dim(dirty_image, 1))
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Dirty image must be square.");
        return;
    }
	
    if (!sdp_mem_is_c_contiguous(dirty_image))
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("All arrays must be C contiguous");
        return;
    }
    if (sdp_mem_is_read_only(dirty_image))
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Visibility data must be writable.");
		return;
    }
}

void sdp_gridder_check_plan(
		sdp_Gridder* plan,
        sdp_Error* status)
{
    if (*status) return;

	if (1)
	{
		SDP_LOG_DEBUG("  plan->pixsize_x_rad is %e", plan->pixsize_x_rad);
		SDP_LOG_DEBUG("  plan->pixsize_y_rad is %e", plan->pixsize_y_rad);
		SDP_LOG_DEBUG("  plan->epsilon is %e",       plan->epsilon);

		SDP_LOG_DEBUG("  plan->workarea is %p",      plan->workarea);
		
		// SDP_LOG_DEBUG("  plan->uvw's     location is %i", sdp_mem_location(plan->uvw));
		// SDP_LOG_DEBUG("  plan->freq_hz's location is %i", sdp_mem_location(plan->freq_hz));
		// SDP_LOG_DEBUG("  plan->vis's     location is %i", sdp_mem_location(plan->vis));
		// SDP_LOG_DEBUG("  plan->weight's  location is %i", sdp_mem_location(plan->weight));		
	}

	if (plan->pixsize_x_rad != plan->pixsize_y_rad)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Only square images supported, so pixsize_x_rad and pixsize_y_rad must be equal.");
        return;
    }
	
	// should check range of epsilon !!
}

sdp_Gridder* sdp_gridder_create_plan(
        const sdp_Mem* uvw,
        const sdp_Mem* freq_hz,  // in Hz
        const sdp_Mem* vis,
        const sdp_Mem* weight,
		const float pixsize_x_rad, 
		const float pixsize_y_rad, 
		const float epsilon,
        sdp_Error* status)
{
    if (*status) return NULL;
	
    sdp_Gridder* plan = (sdp_Gridder*) calloc(1, sizeof(sdp_Gridder));
    // plan->uvw = uvw;
    // plan->freq_hz = freq_hz;
    // plan->vis = vis;
    // plan->weight = weight;
    plan->pixsize_x_rad = pixsize_x_rad;
    plan->pixsize_y_rad = pixsize_y_rad;
    plan->epsilon = epsilon;
	plan->workarea = (float*) calloc(10, sizeof(float)); // TBD!!

	sdp_gridder_check_plan(plan, status);
    if (*status) return NULL;

	sdp_gridder_check_inputs(uvw, freq_hz, vis, weight, status);
    if (*status) return NULL;

    SDP_LOG_INFO("Created sdp_Gridder");
    return plan;
}

void sdp_gridder_exec(
	const sdp_Mem* uvw,
	const sdp_Mem* freq_hz,  // in Hz
	const sdp_Mem* vis,
	const sdp_Mem* weight,
    sdp_Gridder* plan,
    sdp_Mem *dirty_image,
    sdp_Error* status
)
{
    SDP_LOG_DEBUG("Executing sdp_Gridder...");
    if (*status || !plan) return;
	
	sdp_gridder_check_plan(plan, status);
    if (*status) return;

	sdp_gridder_check_inputs(uvw, freq_hz, vis, weight, status);
    if (*status) return;
	
	sdp_gridder_check_outputs(uvw, dirty_image, status);
    if (*status) return;
	
}

void sdp_gridder_free_plan(sdp_Gridder* plan)
{
    if (!plan) return;
    free(plan->workarea);
    free(plan);
    SDP_LOG_INFO("Destroyed sdp_Gridder");
}
