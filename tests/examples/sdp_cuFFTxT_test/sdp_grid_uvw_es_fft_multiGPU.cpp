/*
 * sdp_grid_uvw_es_fft_multiGPU.cpp
 *
 *  Created on: Feb. 20, 2024
 *      Author: vlad
 */

#ifndef SDP_HAVE_CUDA
#define SDP_HAVE_CUDA 1
#endif

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ska-sdp-func/fourier_transforms/sdp_fft.h"

#include "ska-sdp-func/grid_data/sdp_gridder_uvw_es_fft.h"
#include "ska-sdp-func/grid_data/sdp_gridder_uvw_es_fft_utils.h"

#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "sdp_cuFFTxT.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#ifndef PI
#define PI 3.1415926535897931
#endif

struct sdp_GridderUvwEsFft
{
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
    float beta_f;

    double pixel_size;
    float pixel_size_f;

    double uv_scale;
    float uv_scale_f;

    double min_plane_w;
    double max_plane_w;
    double min_abs_w;
    double max_abs_w;
    int num_total_w_grids;
    double w_scale; // scaling factor for converting w coord to signed w grid index

    double inv_w_scale;
    double inv_w_range; // final scaling factor for scaling dirty image by w grid accumulation

    double conv_corr_norm_factor;

    float inv_w_scale_f;
    float inv_w_range_f;
    float w_scale_f;
    float min_plane_w_f;
    float max_plane_w_f;
    float conv_corr_norm_factor_f;

    // allocated memory
    sdp_Mem* w_grid_stack;
    sdp_Mem* quadrature_kernel;
    sdp_Mem* quadrature_nodes;
    sdp_Mem* quadrature_weights;
    sdp_Mem* conv_corr_kernel;
};


void sdp_gridder_check_buffers(
        const sdp_Mem* uvw,
        const sdp_Mem* freq_hz,  // in Hz
        const sdp_Mem* vis,
        const sdp_Mem* weight,
        const sdp_Mem* dirty_image,
        bool do_degridding,
        sdp_Error* status
)
{
    // check location of parameters (CPU or GPU)
    const sdp_MemLocation location = sdp_mem_location(uvw);
    if (location != sdp_mem_location(freq_hz) ||
            location != sdp_mem_location(vis) ||
            location != sdp_mem_location(weight) ||
            location != sdp_mem_location(dirty_image))
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Memory location mismatch.");
        return;
    }

    // check types of parameters (real or complex)
    if (sdp_mem_is_complex(uvw))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("uvw values must be real.");
        return;
    }
    if (sdp_mem_is_complex(freq_hz))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Frequency values must be real.");
        return;
    }
    if (!sdp_mem_is_complex(vis))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Visibility values must be complex.");
        return;
    }
    if (sdp_mem_is_complex(weight))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Weight values must be real.");
        return;
    }
    if (sdp_mem_is_complex(dirty_image))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Dirty image must be real");
        return;
    }

    // check shapes of parameters
    const int64_t num_vis      = sdp_mem_shape_dim(vis, 0);
    const int64_t num_channels = sdp_mem_shape_dim(vis, 1);

    if (sdp_mem_shape_dim(uvw, 0) != num_vis)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("The number of rows in uvw and vis must match.");
        SDP_LOG_ERROR("uvw is %i by %i",
                sdp_mem_shape_dim(uvw, 0),
                sdp_mem_shape_dim(uvw, 1)
        );
        SDP_LOG_ERROR("vis is %i by %i", num_vis, num_channels);
        return;
    }
    if (sdp_mem_shape_dim(uvw, 1) != 3)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("uvw must be N x 3.");
        SDP_LOG_ERROR("uvw is %i by %i",
                sdp_mem_shape_dim(uvw, 0),
                sdp_mem_shape_dim(uvw, 1)
        );
        return;
    }
    if (sdp_mem_shape_dim(freq_hz, 0) != num_channels)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("The number of channels in vis and freq_hz must match.");
        SDP_LOG_ERROR("freq_hz is %i by %i",
                sdp_mem_shape_dim(freq_hz, 0),
                sdp_mem_shape_dim(freq_hz, 1)
        );
        SDP_LOG_ERROR("vis is %i by %i", num_vis, num_channels);
        return;
    }
    if (sdp_mem_shape_dim(weight, 0) != num_vis ||
            sdp_mem_shape_dim(weight, 1) != num_channels)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("weight and vis must be the same size.");
        SDP_LOG_ERROR("weight is %i by %i",
                sdp_mem_shape_dim(weight, 0),
                sdp_mem_shape_dim(weight, 1)
        );
        SDP_LOG_ERROR("vis is %i by %i", num_vis, num_channels);
        return;
    }

    if (sdp_mem_shape_dim(dirty_image, 0) != sdp_mem_shape_dim(dirty_image, 1))
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Dirty image must be square.");
        SDP_LOG_ERROR("dirty_image is %i by %i",
                sdp_mem_shape_dim(dirty_image, 0),
                sdp_mem_shape_dim(dirty_image, 1)
        );
        return;
    }

    // check precision consistency
    if (sdp_mem_type(uvw) == SDP_MEM_DOUBLE)
    {
        if ((sdp_mem_type(freq_hz) != SDP_MEM_DOUBLE) ||
                (sdp_mem_type(vis) != SDP_MEM_COMPLEX_DOUBLE) ||
                (sdp_mem_type(weight) != SDP_MEM_DOUBLE) ||
                (sdp_mem_type(dirty_image) != SDP_MEM_DOUBLE))
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("All buffers must be the same precision.");
            return;
        }
    }
    else
    {
        if ((sdp_mem_type(freq_hz) != SDP_MEM_FLOAT) ||
                (sdp_mem_type(vis) != SDP_MEM_COMPLEX_FLOAT) ||
                (sdp_mem_type(weight) != SDP_MEM_FLOAT) ||
                (sdp_mem_type(dirty_image) != SDP_MEM_FLOAT))
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("All buffers must be the same precision.");
            return;
        }
    }

    // check contiguity
    if (!sdp_mem_is_c_contiguous(uvw) ||
            !sdp_mem_is_c_contiguous(freq_hz) ||
            !sdp_mem_is_c_contiguous(vis) ||
            !sdp_mem_is_c_contiguous(weight) ||
            !sdp_mem_is_c_contiguous(dirty_image))
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("All input arrays must be C contiguous");
        return;
    }

    // check output is writeable
    if (do_degridding)
    {
        if (sdp_mem_is_read_only(vis))
        {
            *status = SDP_ERR_INVALID_ARGUMENT;
            SDP_LOG_ERROR("Visibility data must be writable.");
            return;
        }
    }
    else
    {
        if (sdp_mem_is_read_only(dirty_image))
        {
            *status = SDP_ERR_INVALID_ARGUMENT;
            SDP_LOG_ERROR("Dirty image must be writable.");
            return;
        }
    }
}


void sdp_gridder_check_parameters(
        const double pixsize_x_rad,
        const double pixsize_y_rad,
        sdp_Error* status
)
{
    if (pixsize_x_rad != pixsize_y_rad)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Only square images supported, so pixsize_x_rad and "
                "pixsize_y_rad must be equal."
        );
        SDP_LOG_ERROR("pixsize_x_rad is %.12e", pixsize_x_rad);
        SDP_LOG_ERROR("pixsize_y_rad is %.12e", pixsize_y_rad);
        return;
    }
}


void sdp_gridder_check_plan(
        sdp_GridderUvwEsFft* plan,
        sdp_Error* status
)
{
    sdp_gridder_check_parameters(plan->pixsize_x_rad,
            plan->pixsize_y_rad, status
    );
}


void sdp_grid_uvw_es_fft_multiGPU(
        sdp_GridderUvwEsFft* plan,
        const sdp_Mem* uvw,
        const sdp_Mem* freq_hz,
        const sdp_Mem* vis,
        const sdp_Mem* weight,
        sdp_Mem* dirty_image,
        sdp_Error* status
)
{
    SDP_LOG_DEBUG("Executing sdp_GridderUvwEsFft...");
    if (*status || !plan) return;

    sdp_gridder_check_plan(plan, status);
    if (*status) return;

    sdp_gridder_check_buffers(
            uvw, freq_hz, vis, weight, dirty_image, false, status
    );
    if (*status) return;

    const int npix_x = (int)sdp_mem_shape_dim(dirty_image, 0);
    const int npix_y = (int)sdp_mem_shape_dim(dirty_image, 1);  // this should be the same as checked by check_params()

    uint64_t num_threads[] = {1, 1, 1}, num_blocks[] = {1, 1, 1};

    const sdp_MemType vis_type = sdp_mem_type(vis);

    const int chunk_size = plan->num_rows;
    const int num_w_grids_batched = 1; // fixed, don't change this!!
    const int coord_type = sdp_mem_type(uvw);
    const int dbl_vis = (vis_type & SDP_MEM_DOUBLE);
    const int dbl_coord = (coord_type & SDP_MEM_DOUBLE);
    int64_t grid_pixel_number = (int64_t)plan->grid_size*(int64_t)plan->grid_size;

    sdp_Mem* w_grid_stack_cpu;

    int64_t dimage_shape[] = {plan->grid_size, plan->grid_size};
    sdp_Mem* dimage_cuFFTxT =
            sdp_mem_create(vis_type, SDP_MEM_CPU, 2, dimage_shape, status);
    sdp_mem_clear_contents(dimage_cuFFTxT, status);

    cudaSetDevice( 0 );


    // Create the FFT plan -- removed for cuFFTxT

    /*
    sdp_Fft* fft = sdp_fft_create(
            plan->w_grid_stack, plan->w_grid_stack, 2, 0, status
    );
    if (*status) return;
	*/

    // Determine how many w grid subset batches to process in total
    const int total_w_grid_batches =
            (plan->num_total_w_grids + num_w_grids_batched - 1) /
            num_w_grids_batched;

    for (int batch = 0; batch < total_w_grid_batches; batch++)
    {
        const int num_w_grids_subset = std::min(
                num_w_grids_batched,
                plan->num_total_w_grids - ((batch * num_w_grids_batched) %
                plan->num_total_w_grids)
        );
        const int grid_start_w = batch * num_w_grids_batched;
        sdp_mem_clear_contents(plan->w_grid_stack, status);
        if (*status) break;

        // Perform gridding on a "chunk" of w grids
        {
            const char* kernel_name = 0;
            if (plan->do_wstacking)
            {
                if (dbl_vis && dbl_coord)
                {
                    kernel_name = "sdp_cuda_nifty_gridder_gridding_3d"
                            "<double, double2, double, double2, double3>";
                }
                else if (!dbl_vis && !dbl_coord)
                {
                    kernel_name = "sdp_cuda_nifty_gridder_gridding_3d"
                            "<float, float2, float, float2, float3>";
                }
            }
            else
            {
                if (dbl_vis && dbl_coord)
                {
                    kernel_name = "sdp_cuda_nifty_gridder_gridding_2d"
                            "<double, double2, double, double2, double3>";
                }
                else if (!dbl_vis && !dbl_coord)
                {
                    kernel_name = "sdp_cuda_nifty_gridder_gridding_2d"
                            "<float, float2, float, float2, float3>";
                }
            }
            if (kernel_name)
            {
                num_threads[0] = 1;
                num_threads[1] = 256;
                num_blocks[0] =
                        (plan->num_chan + num_threads[0] - 1) / num_threads[0];
                num_blocks[1] =
                        (chunk_size + num_threads[1] - 1) / num_threads[1];
                const bool solving = 1;
                const void* args[] = {
                    &chunk_size,
                    &plan->num_chan,
                    sdp_mem_gpu_buffer_const(vis, status),
                    sdp_mem_gpu_buffer_const(weight, status),
                    sdp_mem_gpu_buffer_const(uvw, status),
                    sdp_mem_gpu_buffer_const(freq_hz, status),
                    sdp_mem_gpu_buffer(plan->w_grid_stack, status),
                    &plan->grid_size,
                    &grid_start_w,
                    &num_w_grids_subset,
                    &plan->support,
                    dbl_vis ?
                        (const void*)&plan->beta :
                        (const void*)&plan->beta_f,
                    dbl_coord ?
                        (const void*)&plan->uv_scale :
                        (const void*)&plan->uv_scale_f,
                    dbl_coord ?
                        (const void*)&plan->w_scale :
                        (const void*)&plan->w_scale_f,
                    dbl_coord ?
                        (const void*)&plan->min_plane_w :
                        (const void*)&plan->min_plane_w_f,
                    &solving
                };
                sdp_launch_cuda_kernel(kernel_name,
                        num_blocks, num_threads, 0, 0, args, status
                );
            }
        }

        // Create copy of plan->w_grid_stack on the HOST (CPU)
        w_grid_stack_cpu = sdp_mem_create_copy(plan->w_grid_stack, SDP_MEM_CPU, status);
	cudaDeviceSynchronize();

        // Perform 2D FFT on each bound w grid using cuFFTxT
        int gpu_start = 1;
        sdp_cuFFTxT(w_grid_stack_cpu, dimage_cuFFTxT, gpu_start, status);
        //sdp_fft_exec(fft, plan->w_grid_stack, plan->w_grid_stack, status);

	// Set CUDA device
	cudaDeviceSynchronize();
        cudaSetDevice( 0 );

        // Copy ditry image from image_cuFFTxT from the HOST (CPU) to  plan->w_grid_stack on the DEVICE (GPU)
	//plan->w_grid_stack = sdp_mem_create_copy(dimage_cuFFTxT, SDP_MEM_GPU, status);
        sdp_mem_copy_contents(plan->w_grid_stack, dimage_cuFFTxT, 0, 0, grid_pixel_number, status);
        // Perform phase shift on a "chunk" of planes and sum into single real plane
        {
            const char* kernel_name = dbl_vis ?
                        "apply_w_screen_and_sum<double, double2>" :
                        "apply_w_screen_and_sum<float, float2>";
            num_threads[0] = std::min(32, (npix_x + 1) / 2);
            num_threads[1] = std::min(32, (npix_y + 1) / 2);
            // Allow extra in negative x quadrants, for asymmetric image centre
            num_blocks[0] =
                    (npix_x / 2 + 1 + num_threads[0] - 1) / num_threads[0];
            num_blocks[1] =
                    (npix_y / 2 + 1 + num_threads[1] - 1) / num_threads[1];
            const bool do_FFT_shift = true;
            const void* args[] = {
                sdp_mem_gpu_buffer(dirty_image, status),
                &plan->image_size,
                dbl_vis ?
                    (const void*)&plan->pixel_size :
                    (const void*)&plan->pixel_size_f,
                sdp_mem_gpu_buffer_const(plan->w_grid_stack, status),
                &plan->grid_size,
                &grid_start_w,
                &num_w_grids_subset,
                dbl_vis ?
                    (const void*)&plan->inv_w_scale :
                    (const void*)&plan->inv_w_scale_f,
                dbl_vis ?
                    (const void*)&plan->min_plane_w :
                    (const void*)&plan->min_plane_w_f,
                &do_FFT_shift,
                &plan->do_wstacking
            };
            sdp_launch_cuda_kernel(
                    kernel_name, num_blocks, num_threads, 0, 0, args, status
            );
        }
    }

    // Free FFT plan and data. -- removed for cuFFTxT
    //sdp_fft_free(fft);

    // Perform convolution correction and final scaling on single real plane
    // note: can recycle same block/thread dims as w correction kernel
    {
        const char* kernel_name = dbl_vis ?
                    "conv_corr_and_scaling<double>" :
                    "conv_corr_and_scaling<float>";
        num_threads[0] = std::min(32, (npix_x + 1) / 2);
        num_threads[1] = std::min(32, (npix_y + 1) / 2);
        // Allow extra in negative x quadrants, for asymmetric image centre
        num_blocks[0] = (npix_x / 2 + 1 + num_threads[0] - 1) / num_threads[0];
        num_blocks[1] = (npix_y / 2 + 1 + num_threads[1] - 1) / num_threads[1];
        const bool solving = true;
        const void* args[] = {
            sdp_mem_gpu_buffer(dirty_image, status),
            &plan->image_size,
            dbl_vis ?
                (const void*)&plan->pixel_size :
                (const void*)&plan->pixel_size_f,
            &plan->support,
            dbl_vis ?
                (const void*)&plan->conv_corr_norm_factor :
                (const void*)&plan->conv_corr_norm_factor_f,
            sdp_mem_gpu_buffer_const(plan->conv_corr_kernel, status),
            dbl_vis ?
                (const void*)&plan->inv_w_range :
                (const void*)&plan->inv_w_range_f,
            dbl_vis ?
                (const void*)&plan->inv_w_scale :
                (const void*)&plan->inv_w_scale_f,
            sdp_mem_gpu_buffer_const(plan->quadrature_kernel,  status),
            sdp_mem_gpu_buffer_const(plan->quadrature_nodes,   status),
            sdp_mem_gpu_buffer_const(plan->quadrature_weights, status),
            &solving,
            &plan->do_wstacking
        };
        sdp_launch_cuda_kernel(
                kernel_name, num_blocks, num_threads, 0, 0, args, status
        );
    }

    // Free image_cuFFTxT memory
    sdp_mem_free(dimage_cuFFTxT);

