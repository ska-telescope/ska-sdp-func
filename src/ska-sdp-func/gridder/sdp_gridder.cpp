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
	double pixsize_x_rad; 
	double pixsize_y_rad;
	double epsilon;
	bool do_wstacking;
	int num_rows;
	int num_chan;
	int image_size;
	int grid_size;
	int support;
	double beta;
	float  beta_f;

    double pixel_size;
    float  pixel_size_f;

    double uv_scale;
    float  uv_scale_f;

	double min_plane_w;
	double max_plane_w;
	double min_abs_w = 0.0;
	double max_abs_w = 0.0;
	int num_total_w_grids;
	double w_scale; // scaling factor for converting w coord to signed w grid index

    double inv_w_scale;
    double inv_w_range; // final scaling factor for scaling dirty image by w grid accumulation

    float  inv_w_scale_f;
    float  inv_w_range_f;
    float w_scale_f;
    float min_plane_w_f;
    float max_plane_w_f;
	
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


void sdp_gridder_log_plan(
		sdp_Gridder* plan,
        sdp_Error* status)
{
	if (1)
	{
		SDP_LOG_DEBUG("  plan->pixsize_x_rad is %.12e", plan->pixsize_x_rad);
		SDP_LOG_DEBUG("  plan->pixsize_y_rad is %.12e", plan->pixsize_y_rad);
		SDP_LOG_DEBUG("  plan->epsilon is %e",       	plan->epsilon);
		SDP_LOG_DEBUG("  plan->min_abs_w is %e",       	plan->min_abs_w);
		SDP_LOG_DEBUG("  plan->max_abs_w is %e",       	plan->max_abs_w);
		SDP_LOG_DEBUG("  plan->min_plane_w is %e",       	plan->min_plane_w);
		SDP_LOG_DEBUG("  plan->max_plane_w is %e",       	plan->max_plane_w);
		// SDP_LOG_DEBUG("  plan-> is %e",       	plan->);

		SDP_LOG_DEBUG("  plan->workarea is %p",      plan->workarea);
		
		// SDP_LOG_DEBUG("  plan->uvw's     location is %i", sdp_mem_location(plan->uvw));
		// SDP_LOG_DEBUG("  plan->freq_hz's location is %i", sdp_mem_location(plan->freq_hz));
		// SDP_LOG_DEBUG("  plan->vis's     location is %i", sdp_mem_location(plan->vis));
		// SDP_LOG_DEBUG("  plan->weight's  location is %i", sdp_mem_location(plan->weight));		
	}
}

void sdp_gridder_check_plan(
		sdp_Gridder* plan,
        sdp_Error* status)
{
	sdp_gridder_log_plan(plan, status);
    if (*status) return;

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
		const double pixsize_x_rad, 
		const double pixsize_y_rad, 
		const double epsilon,
		const double min_abs_w, 
		const double max_abs_w, 
		bool do_wstacking,
        const sdp_Mem* dirty_image,
        sdp_Error* status)
{
    if (*status) return NULL;
	
    sdp_Gridder* plan = (sdp_Gridder*) calloc(1, sizeof(sdp_Gridder));
	
    plan->pixsize_x_rad = pixsize_x_rad;
    plan->pixsize_y_rad = pixsize_y_rad;
    plan->pixel_size = pixsize_x_rad;  // only square pixels supported
    plan->epsilon = epsilon;
	plan->do_wstacking = do_wstacking;
	plan->workarea = (float*) calloc(10, sizeof(float)); // TBD!!
	plan->num_rows = sdp_mem_shape_dim(vis, 0);
	plan->num_chan = sdp_mem_shape_dim(vis, 1);

	plan->image_size = sdp_mem_shape_dim(dirty_image, 0);

    plan->pixel_size_f = (float) plan->pixel_size;

	int grid_size;
	int support;
	double beta;
	
	const sdp_MemType vis_type = sdp_mem_type(vis);
    const int vis_precision = (vis_type & SDP_MEM_DOUBLE) ? SDP_MEM_DOUBLE : SDP_MEM_FLOAT;
	
	CalculateParamsFromEpsilon(plan->epsilon, plan->image_size, vis_precision, 
					   grid_size, support, beta, status);	
    if (*status) return NULL;

    beta *= support; 

	plan->grid_size = grid_size;
	plan->support = support;
	plan->beta = beta;
	plan->beta_f = (float) beta;

    plan->uv_scale = plan->grid_size * plan->pixel_size;
    plan->uv_scale_f = (float) plan->uv_scale;

	// do_wstacking-dependent parameters
	if (plan->do_wstacking)
	{
		// Determine number of w-planes required.
		const double x0 = -0.5 * plan->image_size * plan->pixel_size;
		const double y0 = -0.5 * plan->image_size * plan->pixel_size;
		double nmin = sqrt(std::max(1.0 - x0 * x0 - y0 * y0, 0.0)) - 1.0;
		if (x0 * x0 + y0 * y0 > 1.0)
			nmin = -sqrt(fabs(1.0 - x0*x0 - y0*y0)) - 1.0;
		double w_scale = 0.25 / fabs(nmin); // scaling factor for converting w coord to signed w grid index
		int num_total_w_grids = (max_abs_w - min_abs_w) / w_scale + 2; //  number of w grids required
		w_scale = 1.0 / ((1.0 + 1e-13) * (max_abs_w - min_abs_w) / (num_total_w_grids - 1));
		const double min_plane_w = min_abs_w - (0.5 * plan->support - 1.0) / w_scale;
		const double max_plane_w = max_abs_w + (0.5 * plan->support - 1.0) / w_scale;
		num_total_w_grids += plan->support - 2;

		plan->min_abs_w = min_abs_w;
		plan->max_abs_w = max_abs_w;
		plan->min_plane_w = min_plane_w;
		plan->max_plane_w = max_plane_w;
		plan->num_total_w_grids = num_total_w_grids;
		plan->w_scale = w_scale; 
		plan->inv_w_range = (plan->max_plane_w - plan->min_plane_w); 
	}
	else
	{
		plan->min_abs_w = 0.0;
		plan->max_abs_w = 0.0;
		plan->min_plane_w = 0.0;
		plan->max_plane_w = 0.0;
		plan->num_total_w_grids = 1;
		plan->inv_w_range = 1.0; 
		plan->w_scale = 1.0; 
	}

	plan->inv_w_scale = 1.0 / plan->w_scale;
	plan->inv_w_scale_f = (float) plan->inv_w_scale;
	plan->w_scale_f =     (float) plan->w_scale;
	plan->min_plane_w_f = (float) plan->min_plane_w;
	plan->max_plane_w_f = (float) plan->max_plane_w;
    plan->inv_w_range_f = (float) plan->inv_w_range;
		
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
	
    //int start_row = 0;
    size_t num_threads[] = {1, 1, 1}, num_blocks[] = {1, 1, 1};

    //const double upsampling = 2.0;

    const sdp_MemType vis_type = sdp_mem_type(vis);
	SDP_LOG_DEBUG("vis_type is %#06x", vis_type);

	
    //const int max_rows_per_chunk = 2000000 / num_chan;
    //const int max_rows_per_chunk = plan->num_rows;
    const int chunk_size = plan->num_rows;
    const int num_w_grids_batched = 1; // fixed, don't change this!!
    const int coord_type = sdp_mem_type(uvw);
    const int dbl_vis = (vis_type & SDP_MEM_DOUBLE);
    const int dbl_coord = (coord_type & SDP_MEM_DOUBLE);
    //const int image_size = npix_x;
   //grid_size = floor(npix_x * upsampling);

	if (1)
	{	
		// SDP_LOG_DEBUG("grid_size is %i",  plan->grid_size);
		// SDP_LOG_DEBUG("image_size is %i", plan->image_size);
		// SDP_LOG_DEBUG("num_total_w_grids is %i", plan->num_total_w_grids);
		// SDP_LOG_DEBUG("min_plane_w is %e", plan->min_plane_w);
		// SDP_LOG_DEBUG("uv_scale is %.12e", uv_scale);
		// SDP_LOG_DEBUG("uv_scale_f is %.12e", uv_scale_f);
		// SDP_LOG_DEBUG("dbl_coord is %i", dbl_coord);		
	}

    // Create the empty grid.
	// THIS SHOULD BE DONE IN THE PLAN!!
	int64_t w_grid_stack_shape[] = {plan->grid_size, plan->grid_size};
    sdp_Mem* d_w_grid_stack = sdp_mem_create(
            vis_type, SDP_MEM_GPU, 2, w_grid_stack_shape, status);
			
    // Create the FFT plan.
    sdp_Fft* fft = sdp_fft_create(d_w_grid_stack, d_w_grid_stack, 2, 0, status);

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
			if (plan->do_wstacking)
			{
				if (dbl_vis && dbl_coord)
					k = "sdp_cuda_nifty_gridder_gridding_3d<double, double2, double, double2, double3>";
				else if (!dbl_vis && dbl_coord)
					k = "sdp_cuda_nifty_gridder_gridding_3d<float, float2, double, double2, double3>";
				else if (!dbl_vis && !dbl_coord)
					k = "sdp_cuda_nifty_gridder_gridding_3d<float, float2, float, float2, float3>";
			}
			else
			{
				if (dbl_vis && dbl_coord)
					k = "sdp_cuda_nifty_gridder_gridding_2d<double, double2, double, double2, double3>";
				else if (!dbl_vis && dbl_coord)
					k = "sdp_cuda_nifty_gridder_gridding_2d<float, float2, double, double2, double3>";
				else if (!dbl_vis && !dbl_coord)
					k = "sdp_cuda_nifty_gridder_gridding_2d<float, float2, float, float2, float3>";
			}
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
                    &plan->grid_size,
                    &grid_start_w,
                    &num_w_grids_subset,
                    &plan->support,
                    dbl_vis ?   (const void*)&plan->beta        : (const void*)&plan->beta_f,
                    dbl_coord ? (const void*)&plan->uv_scale    : (const void*)&plan->uv_scale_f,
                    dbl_coord ? (const void*)&plan->w_scale     : (const void*)&plan->w_scale_f,
                    dbl_coord ? (const void*)&plan->min_plane_w : (const void*)&plan->min_plane_w_f,
                    &solving
                };
                sdp_launch_cuda_kernel(k, num_blocks, num_threads, 0, 0, args, status);
            }
        }
		
		SDP_LOG_DEBUG("Finished gridding batch %i of %i batches.",  batch, total_w_grid_batches);
		
		if (0) // write out w-grids
		{
			sdp_Mem* h_w_grid_stack = sdp_mem_create_copy(d_w_grid_stack, SDP_MEM_CPU, status);
			const std::complex<double>* test_grid = (const std::complex<double>*)sdp_mem_data_const(h_w_grid_stack);
			for (size_t i = 1185039 - 5; i <= 1185039 + 5; i++)
			{			
				//printf("test_grid[%li] = [%e, %e]\n", i, real(test_grid[i]), imag(test_grid[i]));
			}
			
			int start_w_grid = batch;
			char file_name_buffer[257];
			uint32_t num_w_grid_cells = plan->grid_size * plan->grid_size;
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
			
        // Perform 2D FFT on each bound w grid
		sdp_fft_exec(fft, d_w_grid_stack, d_w_grid_stack, status);
		
		if (0) // write out w-images
		{
			sdp_Mem* h_w_image_stack = sdp_mem_create_copy(d_w_grid_stack, SDP_MEM_CPU, status);
			const std::complex<double>* test_image = (const std::complex<double>*)sdp_mem_data_const(h_w_image_stack);
			for (size_t i = 1185039 - 5; i <= 1185039 + 5; i++)
			{			
				printf("test_image[%li] = [%e, %e]\n", i, real(test_image[i]), imag(test_image[i]));
			}
			
			int start_w_grid = batch;
			char file_name_buffer[257];
			uint32_t num_w_grid_cells = plan->grid_size * plan->grid_size;
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
			// const bool do_wstacking = false;
            const void* args[] = {
                sdp_mem_gpu_buffer(dirty_image, status),
                &plan->image_size,
                dbl_vis ? (const void*)&plan->pixel_size : (const void*)&plan->pixel_size_f,
                sdp_mem_gpu_buffer_const(d_w_grid_stack, status),
                &plan->grid_size,
                &grid_start_w,
                &num_w_grids_subset,
                dbl_vis ?
                    (const void*)&plan->inv_w_scale : (const void*)&plan->inv_w_scale_f,
                dbl_vis ?
                    (const void*)&plan->min_plane_w : (const void*)&plan->min_plane_w_f,
				&do_FFT_shift,
				&plan->do_wstacking
            };
            sdp_launch_cuda_kernel(k, num_blocks, num_threads, 0, 0, args, status);
        }
	}	
	
	if (0) // write out dirty_image
	{
		sdp_Mem* h_dirty_image = sdp_mem_create_copy(dirty_image, SDP_MEM_CPU, status);
		const double* test_image = (const double*)sdp_mem_data_const(h_dirty_image);
		for (size_t i = 524288 - 5; i <= 524288 + 5; i++)
		{			
			printf("test_image[%li] = [%e]\n", i, test_image[i]);
		}
		
		char file_name_buffer[257];
		uint32_t num_dirty_image_pixels = plan->image_size * plan->image_size;
		for(int i = 0; i < num_w_grids_batched; i++)
		{
			// build file name, ie: my/folder/path/w_grid_123.bin
			//snprintf(file_name_buffer, 257, "%s/cw_grid_%d.bin", config->data_output_folder, start_w_grid++);
			snprintf(file_name_buffer, 257, "dirty_image.bin");
			printf("Writing image to file: %s ...\n", file_name_buffer);
			FILE *f = fopen(file_name_buffer, "wb");

			fwrite(test_image, sizeof(double), num_dirty_image_pixels, f);
			
			fclose(f);
		}
		
		sdp_mem_free(h_dirty_image);
	}
	
    // Free FFT plan and data.
    sdp_fft_free(fft);	
	
    sdp_mem_free(d_w_grid_stack);

    // Generate Gauss Legendre kernel for convolution correction.
    double *quadrature_kernel, *quadrature_nodes, *quadrature_weights;
    double *conv_corr_kernel;
    quadrature_kernel  = (double*) calloc(QUADRATURE_SUPPORT_BOUND, sizeof(double));
    quadrature_nodes   = (double*) calloc(QUADRATURE_SUPPORT_BOUND, sizeof(double));
    quadrature_weights = (double*) calloc(QUADRATURE_SUPPORT_BOUND, sizeof(double));
    conv_corr_kernel   = (double*) calloc(plan->image_size / 2 + 1, sizeof(double));
    generate_gauss_legendre_conv_kernel(plan->image_size, plan->grid_size, plan->support, plan->beta,
            quadrature_kernel, quadrature_nodes, quadrature_weights,
            conv_corr_kernel);

    // Need to determine normalisation factor for scaling runtime calculated
    // conv correction values for coordinate n (where n = sqrt(1 - l^2 - m^2) - 1)
    //uint32_t p = (uint32_t)(ceil(1.5 * support + 2.0));
    uint32_t p = (uint32_t)(int(1.5 * plan->support + 2.0));
    double conv_corr_norm_factor = 0.0;
    for (uint32_t i = 0; i < p; i++)
        conv_corr_norm_factor += quadrature_kernel[i] * quadrature_weights[i];
    conv_corr_norm_factor *= (double)plan->support;
    const float conv_corr_norm_factor_f = (float)conv_corr_norm_factor;

	if (0) 
	{
		for (size_t i = 0; i < QUADRATURE_SUPPORT_BOUND; i++)
		{			
			// printf("quadrature_weights[%li] = [%.12e]\n", i, quadrature_weights[i]);
			printf("conv_corr_kernel[%li] = [%.12e]\n", i, conv_corr_kernel[i]);
		}
		
		printf("conv_corr_norm_factor = [%.12e]\n", conv_corr_norm_factor);
	}	

    // Convert precision if required.
    void *p_quadrature_kernel, *p_quadrature_nodes, *p_quadrature_weights;
    void *p_conv_corr_kernel;
    sdp_MemType mem_type;
    if (dbl_vis)
    {
        mem_type = SDP_MEM_DOUBLE;
        p_quadrature_kernel = quadrature_kernel;
        p_quadrature_nodes = quadrature_nodes;
        p_quadrature_weights = quadrature_weights;
        p_conv_corr_kernel = conv_corr_kernel;
    }
    else
    {
        // Cast to float.
        mem_type = SDP_MEM_FLOAT;
        p_quadrature_kernel  = calloc(QUADRATURE_SUPPORT_BOUND, sizeof(float));
        p_quadrature_nodes   = calloc(QUADRATURE_SUPPORT_BOUND, sizeof(float));
        p_quadrature_weights = calloc(QUADRATURE_SUPPORT_BOUND, sizeof(float));
        p_conv_corr_kernel   = calloc(plan->image_size / 2 + 1, sizeof(float));
        for (int i = 0; i < QUADRATURE_SUPPORT_BOUND; ++i)
        {
            ((float*)p_quadrature_kernel)[ i] = (float)(quadrature_kernel[ i]);
            ((float*)p_quadrature_nodes)[  i] = (float)(quadrature_nodes[  i]);
            ((float*)p_quadrature_weights)[i] = (float)(quadrature_weights[i]);
        }
        for (int i = 0; i < plan->image_size / 2 + 1; ++i)
            ((float*)p_conv_corr_kernel)[i] = (float)(conv_corr_kernel[i]);
        free(quadrature_kernel);
        free(quadrature_nodes);
        free(quadrature_weights);
        free(conv_corr_kernel);
    }
			
	const int64_t qsb[] = {QUADRATURE_SUPPORT_BOUND};
	const int64_t stride[] = {1};
	const int64_t len[] = {plan->image_size / 2 + 1};
			
	sdp_Mem* h_quadrature_kernel = sdp_mem_create_wrapper(p_quadrature_kernel, 
		mem_type, SDP_MEM_CPU, 1, qsb, stride, status);
	sdp_Mem* h_quadrature_nodes = sdp_mem_create_wrapper(p_quadrature_nodes, 
		mem_type, SDP_MEM_CPU, 1, qsb, stride, status);
	sdp_Mem* h_quadrature_weights = sdp_mem_create_wrapper(p_quadrature_weights, 
		mem_type, SDP_MEM_CPU, 1, qsb, stride, status);
	sdp_Mem* h_conv_corr_kernel = sdp_mem_create_wrapper(p_conv_corr_kernel, 
		mem_type, SDP_MEM_CPU, 1, len, stride, status);

	if (0) 
	{
		const double* test = (const double*)sdp_mem_data_const(h_conv_corr_kernel);
		for (size_t i = 0; i < QUADRATURE_SUPPORT_BOUND; i++)
		{			
			// printf("quadrature_weights[%li] = [%.12e]\n", i, quadrature_weights[i]);
			printf("h_conv_corr_kernel[%li] = [%.12e]\n", i, test[i]);
		}
		
		// printf("conv_corr_norm_factor = [%.12e]\n", conv_corr_norm_factor);
	}

    // Copy arrays to GPU.
	sdp_Mem* d_quadrature_kernel  = sdp_mem_create_copy(h_quadrature_kernel,  SDP_MEM_GPU, status);
	sdp_Mem* d_quadrature_nodes   = sdp_mem_create_copy(h_quadrature_nodes,   SDP_MEM_GPU, status);
	sdp_Mem* d_quadrature_weights = sdp_mem_create_copy(h_quadrature_weights, SDP_MEM_GPU, status);
	sdp_Mem* d_conv_corr_kernel   = sdp_mem_create_copy(h_conv_corr_kernel,   SDP_MEM_GPU, status);
	
    // Perform convolution correction and final scaling on single real plane
    // note: can recycle same block/thread dims as w correction kernel
    {
        const char* k = dbl_vis ?
                "conv_corr_and_scaling<double>" :
                "conv_corr_and_scaling<float>";
        num_threads[0] = std::min(32, (npix_x + 1) / 2);
        num_threads[1] = std::min(32, (npix_y + 1) / 2);
        // Allow extra in negative x quadrants, for asymmetric image centre
        num_blocks[0] = (npix_x / 2 + 1 + num_threads[0] - 1) / num_threads[0];
        num_blocks[1] = (npix_y / 2 + 1 + num_threads[1] - 1) / num_threads[1];
		const bool solving = true;
		// const bool do_wstacking = false;
        const void* args[] = {
            sdp_mem_gpu_buffer(dirty_image, status),
            &plan->image_size,
            dbl_vis ? (const void*)&plan->pixel_size : (const void*)&plan->pixel_size_f,
            &plan->support,
            dbl_vis ?
                (const void*)&conv_corr_norm_factor :
                (const void*)&conv_corr_norm_factor_f,
            sdp_mem_gpu_buffer_const(d_conv_corr_kernel, status),
            dbl_vis ? (const void*)&plan->inv_w_range : (const void*)&plan->inv_w_range_f,
            dbl_vis ? (const void*)&plan->inv_w_scale : (const void*)&plan->inv_w_scale_f,
            sdp_mem_gpu_buffer_const(d_quadrature_kernel, status),
            sdp_mem_gpu_buffer_const(d_quadrature_nodes, status),
            sdp_mem_gpu_buffer_const(d_quadrature_weights, status),
			&solving,
			&plan->do_wstacking
        };
        sdp_launch_cuda_kernel(k, num_blocks, num_threads, 0, 0, args, status);
    }

	if (1) // write out dirty_image
	{
		sdp_Mem* h_dirty_image = sdp_mem_create_copy(dirty_image, SDP_MEM_CPU, status);
		const double* test_image = (const double*)sdp_mem_data_const(h_dirty_image);
		for (size_t i = 524288 - 5; i <= 524288 + 5; i++)
		{			
			// printf("test_image[%li] = [%e]\n", i, test_image[i]);
		}
		
		char file_name_buffer[257];
		uint32_t num_dirty_image_pixels = plan->image_size * plan->image_size;
		for(int i = 0; i < num_w_grids_batched; i++)
		{
			// build file name, ie: my/folder/path/w_grid_123.bin
			//snprintf(file_name_buffer, 257, "%s/cw_grid_%d.bin", config->data_output_folder, start_w_grid++);
			snprintf(file_name_buffer, 257, "dirty_image.bin");
			printf("Writing image to file: %s ...\n", file_name_buffer);
			FILE *f = fopen(file_name_buffer, "wb");

			fwrite(test_image, sizeof(double), num_dirty_image_pixels, f);
			
			fclose(f);
		}
		
		sdp_mem_free(h_dirty_image);
	}
	

    // // Free arrays used for convolution correction.
    // free(p_quadrature_kernel);
    // free(p_quadrature_nodes);
    // free(p_quadrature_weights);
    // free(p_conv_corr_kernel);
    // wrapper.mem_free(h_quadrature_kernel, status);
    // wrapper.mem_free(h_quadrature_nodes, status);
    // wrapper.mem_free(h_quadrature_weights, status);
    // wrapper.mem_free(h_conv_corr_kernel, status);
    // wrapper.mem_free(d_quadrature_kernel, status);
    // wrapper.mem_free(d_quadrature_nodes, status);
    // wrapper.mem_free(d_quadrature_weights, status);
    // wrapper.mem_free(d_conv_corr_kernel, status);
	
	// LogF(verbosityLevel, 2, "\nEnd of ms2dirty_2d()... \n\n");
}

void sdp_gridder_free_plan(sdp_Gridder* plan)
{
    if (!plan) return;
    free(plan->workarea);
    free(plan);
    SDP_LOG_INFO("Destroyed sdp_Gridder");
}
