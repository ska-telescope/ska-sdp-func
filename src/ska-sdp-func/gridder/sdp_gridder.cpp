/* See the LICENSE file at the top-level directory of this distribution. */

#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"

#include "ska-sdp-func/fft/sdp_fft.h"

#include "ska-sdp-func/gridder/sdp_gridder.h"
#include "ska-sdp-func/gridder/nifty_utils.h"

#include <complex>

struct sdp_Gridder
{
    // const sdp_Mem* uvw;
    // const sdp_Mem* freq_hz;  // in Hz
    // const sdp_Mem* vis;
    // const sdp_Mem* weight;
	float pixsize_x_rad; 
	float pixsize_y_rad;
	float epsilon;
	bool do_wstacking;
	int num_rows;
	int num_chan;
	double min_plane_w;
	int num_total_w_grids;
	double w_scale; // scaling factor for converting w coord to signed w grid index
    double inv_w_scale;
    float inv_w_scale_f;
    float w_scale_f;
    float min_plane_w_f;
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
		bool do_wstacking,
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
	plan->do_wstacking = do_wstacking;
	plan->workarea = (float*) calloc(10, sizeof(float)); // TBD!!
	plan->num_rows = sdp_mem_shape_dim(vis, 0);
	plan->num_chan = sdp_mem_shape_dim(vis, 1);

	if (plan->do_wstacking)
	{
	}
	else
	{
		plan->min_plane_w = 0.0;
		plan->num_total_w_grids = 1;
		plan->w_scale = 1.0; // scaling factor for converting w coord to signed w grid index
		plan->inv_w_scale = 1.0 / plan->w_scale;
		plan->inv_w_scale_f = (float) plan->inv_w_scale;
		plan->w_scale_f = (float) plan->w_scale;
		plan->min_plane_w_f = (float) plan->min_plane_w;
	}

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

    const int npix_x = (int)sdp_mem_shape_dim(dirty_image, 0);
    const int npix_y = (int)sdp_mem_shape_dim(dirty_image, 1);  // this should be the same as checked by check_params()
	
    int start_row = 0;
    size_t num_threads[] = {1, 1, 1}, num_blocks[] = {1, 1, 1};

    const double upsampling = 2.0;
    const int image_size = npix_x;

    const sdp_MemType vis_type = sdp_mem_type(vis);
    const int vis_precision = (vis_type & SDP_MEM_DOUBLE) ? SDP_MEM_DOUBLE : SDP_MEM_DOUBLE;
	SDP_LOG_DEBUG("vis_type is %#06x", vis_type);

	double beta;
	int support;
	int grid_size;
	
	CalculateParamsFromEpsilon(plan->epsilon, image_size, vis_precision, 
					   grid_size, support, beta, status);	

	//CalculateSupportAndBeta(upsampling, epsilon, support, beta, *status);

	SDP_LOG_DEBUG("support is %i", support);
	SDP_LOG_DEBUG("beta is %e", beta);
    beta *= support; 
	
    //const int max_rows_per_chunk = 2000000 / num_chan;
    const int max_rows_per_chunk = plan->num_rows;
    const int chunk_size = plan->num_rows;
    const int num_w_grids_batched = 1; // fixed, don't change this!!
    const int coord_type = sdp_mem_type(uvw);
    //const int vis_type = mem_type(ms);
    //const int vis_precision = (vis_type & MEM_DOUBLE) ? MEM_DOUBLE : MEM_FLOAT;
    const int dbl_vis = (vis_type & SDP_MEM_DOUBLE);
    const int dbl_coord = (coord_type & SDP_MEM_DOUBLE);
    //const int image_size = npix_x;
   //grid_size = floor(npix_x * upsampling);
    const double uv_scale = grid_size * plan->pixsize_x_rad;
    const double pixel_size = plan->pixsize_x_rad;
    const float beta_f = (float) beta;
    const float uv_scale_f = (float) uv_scale;
    const float pixel_size_f = (float) pixel_size;

	if (1)
	{	
		SDP_LOG_DEBUG("grid_size is %i",  grid_size);
		SDP_LOG_DEBUG("image_size is %i", image_size);
		SDP_LOG_DEBUG("num_total_w_grids is %i", plan->num_total_w_grids);
		SDP_LOG_DEBUG("min_plane_w is %e", plan->min_plane_w);
	}

    // Create the empty grid.
	// THIS SHOULD BE DONE IN THE PLAN!!
    size_t num_w_grid_stack_cells = grid_size * grid_size * num_w_grids_batched;
	int64_t w_grid_stack_shape[] = {grid_size, grid_size, num_w_grids_batched};
    sdp_Mem* d_w_grid_stack = sdp_mem_create(
            vis_type, SDP_MEM_GPU, 3, w_grid_stack_shape, status);
			
   // Create the FFT plan.
    const int64_t fft_dims[] = {grid_size, grid_size};
    sdp_Fft* fft = sdp_fft_create(SDP_MEM_DOUBLE, SDP_MEM_GPU, SDP_FFT_C2C,
            2, fft_dims, num_w_grids_batched, 0, status);

    // Create the FFT plan.
    // const int fft_dims[] = {grid_size, grid_size};
    // FFT* fft = wrapper.fft_create(vis_precision, MEM_GPU, 2, fft_dims, FFT_C2C,
            // num_w_grids_batched, status);

    if (*status) return;

    // Determine how many w grid subset batches to process in total
    const int total_w_grid_batches =
            (plan->num_total_w_grids + num_w_grids_batched - 1) / num_w_grids_batched;
			
    for (int batch = 0; batch < total_w_grid_batches; batch++)
    {
        const int num_w_grids_subset = std::min(
            num_w_grids_batched,
            plan->num_total_w_grids - ((batch * num_w_grids_batched) % plan->num_total_w_grids)
        );
        const int grid_start_w = batch * num_w_grids_batched;
        sdp_mem_clear_contents(d_w_grid_stack, status);
		if (*status) return;

        // Perform gridding on a "chunk" of w grids
        {
            const char* k = 0;
            if (dbl_vis && dbl_coord)
                k = "sdp_cuda_nifty_gridder_gridding_2d<double, double2, double, double2, double3>";
            else if (!dbl_vis && dbl_coord)
                k = "sdp_cuda_nifty_gridder_gridding_2d<float, float2, double, double2, double3>";
            else if (!dbl_vis && !dbl_coord)
                k = "sdp_cuda_nifty_gridder_gridding_2d<float, float2, float, float2, float3>";
            if (k)
            {
                num_threads[0] = 1;
                num_threads[1] = 256;
                num_blocks[0] = (plan->num_chan + num_threads[0] - 1) / num_threads[0];
                num_blocks[1] = (chunk_size + num_threads[1] - 1) / num_threads[1];
                const bool solving = 1;
                const void* args[] = {
                    &chunk_size,
                    &plan->num_chan,
                    sdp_mem_gpu_buffer_const(vis, status),
                    sdp_mem_gpu_buffer_const(weight, status),
                    sdp_mem_gpu_buffer_const(uvw, status),
                    sdp_mem_gpu_buffer_const(freq_hz, status),
                    sdp_mem_gpu_buffer(d_w_grid_stack, status),
                    &grid_size,
                    &grid_start_w,
                    &num_w_grids_subset,
                    &support,
                    dbl_vis ? (const void*)&beta : (const void*)&beta_f,
                    dbl_coord ?
                        (const void*)&uv_scale : (const void*)&uv_scale_f,
                    dbl_coord ?
                        (const void*)&plan->w_scale : (const void*)&plan->w_scale_f,
                    dbl_coord ?
                        (const void*)&plan->min_plane_w : (const void*)&plan->min_plane_w_f,
                    &solving
                };
                sdp_launch_cuda_kernel(k, num_blocks, num_threads, 0, 0, args, status);
            }
        }
		
		SDP_LOG_DEBUG("Finished gridding batch %i of %i batches.",  batch, total_w_grid_batches);
		
		if (1) // write out w-grids
		{
			sdp_Mem* h_w_grid_stack = sdp_mem_create_copy(d_w_grid_stack, SDP_MEM_CPU, status);
			const std::complex<double>* test_grid = (const std::complex<double>*)sdp_mem_data_const(h_w_grid_stack);
			for (size_t i = 1185039 - 5; i <= 1185039 + 5; i++)
			{			
				printf("test_grid[%i] = [%e, %e]\n", i, real(test_grid[i]), imag(test_grid[i]));
			}
			
			int start_w_grid = batch;
			char file_name_buffer[257];
			uint32_t num_w_grid_cells = grid_size * grid_size;
			for(int i = 0; i < num_w_grids_batched; i++)
			{
				// build file name, ie: my/folder/path/w_grid_123.bin
				//snprintf(file_name_buffer, 257, "%s/cw_grid_%d.bin", config->data_output_folder, start_w_grid++);
				snprintf(file_name_buffer, 257, "cw_grid_%d.bin", start_w_grid++);
				printf("Writing image to file: %s ...\n", file_name_buffer);
				FILE *f = fopen(file_name_buffer, "wb");

				// memory offset for "splitting" the binary write process
				uint32_t w_grid_index_offset = i * num_w_grid_cells; 
				fwrite(test_grid + w_grid_index_offset, sizeof(std::complex<double>), num_w_grid_cells, f);
				
				fclose(f);
			}
			
			sdp_mem_free(h_w_grid_stack);
		}
		
		
		if (0)
 		{
			// printf("Trying here...");
			// mem_copy_contents(im, d_w_grid_stack, 0, 0, (npix_x*2 * npix_y*2), status);  // AG for testing (comment out otherwise)
			// printf("success!!\n");
		}

        // Perform 2D FFT on each bound w grid
		sdp_fft_exec(fft, d_w_grid_stack, d_w_grid_stack, status);
        //fft_exec(fft, d_w_grid_stack, d_w_grid_stack, 0, status);
		
		if (1) // write out w-grids
		{
			sdp_Mem* h_w_image_stack = sdp_mem_create_copy(d_w_grid_stack, SDP_MEM_CPU, status);
			const std::complex<double>* test_image = (const std::complex<double>*)sdp_mem_data_const(h_w_image_stack);
			for (size_t i = 1185039 - 5; i <= 1185039 + 5; i++)
			{			
				printf("test_image[%i] = [%e, %e]\n", i, real(test_image[i]), imag(test_image[i]));
			}
			
			int start_w_grid = batch;
			char file_name_buffer[257];
			uint32_t num_w_grid_cells = grid_size * grid_size;
			for(int i = 0; i < num_w_grids_batched; i++)
			{
				// build file name, ie: my/folder/path/w_grid_123.bin
				//snprintf(file_name_buffer, 257, "%s/cw_grid_%d.bin", config->data_output_folder, start_w_grid++);
				snprintf(file_name_buffer, 257, "cw_image_%d.bin", start_w_grid++);
				printf("Writing image to file: %s ...\n", file_name_buffer);
				FILE *f = fopen(file_name_buffer, "wb");

				// memory offset for "splitting" the binary write process
				uint32_t w_grid_index_offset = i * num_w_grid_cells; 
				fwrite(test_image + w_grid_index_offset, sizeof(std::complex<double>), num_w_grid_cells, f);
				
				fclose(f);
			}
			
			sdp_mem_free(h_w_image_stack);
		}
		
		
/*
        // Perform phase shift on a "chunk" of planes and sum into single real plane
        {
            const char* k = dbl_vis ?
                    "apply_w_screen_and_sum<double, double2>" :
                    "apply_w_screen_and_sum<float, float2>";
            num_threads[0] = std::min(32, (npix_x + 1) / 2);
            num_threads[1] = std::min(32, (npix_y + 1) / 2);
            // Allow extra in negative x quadrants, for asymmetric image centre
            num_blocks[0] = (npix_x / 2 + 1 + num_threads[0] - 1) / num_threads[0];
            num_blocks[1] = (npix_y / 2 + 1 + num_threads[1] - 1) / num_threads[1];
			const bool do_FFT_shift = true;
			const bool do_wstacking = false;
            const void* args[] = {
                mem_buffer(d_dirty_image),
                &image_size,
                dbl_vis ?
                    (const void*)&pixel_size : (const void*)&pixel_size_f,
                mem_buffer_const(d_w_grid_stack),
                &grid_size,
                &grid_start_w,
                &num_w_grids_subset,
                dbl_vis ?
                    (const void*)&inv_w_scale : (const void*)&inv_w_scale_f,
                dbl_vis ?
                    (const void*)&min_plane_w : (const void*)&min_plane_w_f,
				&do_FFT_shift,
				&do_wstacking
            };
            launch_kernel(k, num_blocks, num_threads, 0, 0, args, status);
        }
 */
	}	
	
    // Free FFT plan and data.
    sdp_fft_free(fft);	
	
    sdp_mem_free(d_w_grid_stack);

}

void sdp_gridder_free_plan(sdp_Gridder* plan)
{
    if (!plan) return;
    free(plan->workarea);
    free(plan);
    SDP_LOG_INFO("Destroyed sdp_Gridder");
}
