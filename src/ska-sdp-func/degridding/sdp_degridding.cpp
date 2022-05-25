/* See the LICENSE file at the top-level directory of this distribution. */

#include "ska-sdp-func/degridding/sdp_degridding.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"

#include <math.h>
#include <complex>


void calculate_coordinates(
		int64_t grid_size, //dimension of the image's subgrid grid_size x grid_size x 4?
		int x_stride, // padding in x dimension
		int y_stride, // padding in y dimension
		int kernel_size, // gcf kernel support
		int kernel_stride, // padding of the gcf kernel
		int oversample, // oversampling of the uv kernel
		int wkernel_size, // gcf in w kernel support
		int wkernel_stride, // padding of the gcf w kernel
		int oversample_w, // oversampling of the w kernel
		double theta, //conversion parameter from uv coordinates to xy coordinates x=u*theta
		double wstep, //conversion parameter from w coordinates to z coordinates z=w*wstep 
		double u, // 
		double v, // coordinates of the visibility 
		double w, //
		int *grid_offset, // offset in the image subgrid
		int *sub_offset_x, //
		int *sub_offset_y, // fractional coordinates
		int *sub_offset_z //
	){
	// x coordinate
	double x = theta*u;
	double ox = x*oversample;
	//int iox = lrint(ox);
	int iox = round(ox); // round to nearest
    iox += (grid_size / 2 + 1) * oversample - 1;
    int home_x = iox / oversample;
    int frac_x = oversample - 1 - (iox % oversample);
	
	// y coordinate
	double y = theta*v;
	double oy = y*oversample;
	//int iox = lrint(ox);
	int ioy = round(oy);
    ioy += (grid_size / 2 + 1) * oversample - 1;
    int home_y = ioy / oversample;
    int frac_y = oversample - 1 - (ioy % oversample);
	
	// w coordinate
	double z = 1.0 + w/wstep;
	double oz = z*oversample_w;
	//int iox = lrint(ox);
	int ioz = round(oz);
    ioz += oversample_w - 1;
    //int home_z = ioz / oversample_w;
    int frac_z = oversample_w - 1 - (ioz % oversample_w);
	
    *grid_offset = (home_y-kernel_size/2)*y_stride + (home_x-kernel_size/2)*x_stride;
    *sub_offset_x = kernel_stride * frac_x;
    *sub_offset_y = kernel_stride * frac_y;
    *sub_offset_z = wkernel_stride * frac_z;
    }

static void degridding(
		const int64_t uv_kernel_stride,
		const int64_t w_kernel_stride,
		const int64_t x_size,
		const int64_t y_size,
		const int64_t vis_count,
		const std::complex<double>* grid,
        const double* vis_coordinates,
        const double* uv_kernel,
        const double* w_kernel,
        const int64_t uv_kernel_oversampling,
        const int64_t w_kernel_oversampling,
        const double theta,
        const double wstep, 
        const bool conjugate, 
        std::complex<double>* vis){

int grid_offset, sub_offset_x, sub_offset_y, sub_offset_z;

for(int v = 0; v < vis_count; v++){

		int idx = 3*v;

		float u_vis_coordinate = vis_coordinates[idx];
		float v_vis_coordinate = vis_coordinates[idx+1];
		float w_vis_coordinate = vis_coordinates[idx+2];

		calculate_coordinates(
			x_size, 1, y_size,
			uv_kernel_stride, uv_kernel_stride, uv_kernel_oversampling,
			w_kernel_stride, w_kernel_stride, w_kernel_oversampling,
			theta, wstep, 
			u_vis_coordinate, 
			v_vis_coordinate, 
			w_vis_coordinate,
			&grid_offset, 
			&sub_offset_x, &sub_offset_y, &sub_offset_z
		);

		double vis_r = 0.0, vis_i = 0.0;
		for (int z = 0; z < w_kernel_stride; z++) {
			double visz_r = 0, visz_i = 0;
			for (int y = 0; y < uv_kernel_stride; y++) {
				double visy_r = 0, visy_i = 0;
				for (int x = 0; x < uv_kernel_stride; x++) {
					double grid_r = 0; //
					double grid_i = 0; //
					std::complex<double> temp = grid[z*x_size*y_size + grid_offset + y*y_size + x];
					grid_r = temp.real();
					grid_i = temp.imag();
					visy_r += uv_kernel[sub_offset_x + x] * grid_r;
					visy_i += uv_kernel[sub_offset_x + x] * grid_i;
				}
				visz_r += uv_kernel[sub_offset_y + y] * visy_r;
				visz_i += uv_kernel[sub_offset_y + y] * visy_i;
			}
			vis_r += w_kernel[sub_offset_z + z] * visz_r;
			vis_i += w_kernel[sub_offset_z + z] * visz_i;

		}

		vis[v].real(vis_r);

		if(conjugate) vis[v].imag(-vis_i);
		else vis[v].imag(vis_i);
	}

	// for(int v = 0; v < vis_count; v++){
	// 	std::cout << "\n" << vis[v] << "\n";

	// }

    }


void sdp_degridding(
        const sdp_Mem* grid,
        const sdp_Mem* vis_coordinates,
        const sdp_Mem* uv_kernel,
        const sdp_Mem* w_kernel,
        const int64_t uv_kernel_oversampling,
        const int64_t w_kernel_oversampling,
        const double theta,
        const double wstep, 
        const bool conjugate, 
        sdp_Mem* vis,
        sdp_Error* status)

{
        if (*status) 
        {
                SDP_LOG_INFO("Exit due to error flag set on entry");
                return;
        }

        const sdp_MemLocation location = sdp_mem_location(vis);
        
        if (sdp_mem_is_read_only(vis))
        {
                *status = SDP_ERR_RUNTIME;
                SDP_LOG_ERROR("Output visability must be writable.");
                return;
        }

        if (sdp_mem_location(grid) != location ||
            sdp_mem_location(vis) != location ||
            sdp_mem_location(uv_kernel) != location ||
            sdp_mem_location(w_kernel) != location)
        {
                *status = SDP_ERR_MEM_LOCATION;
                SDP_LOG_ERROR("Memory location mismatch");
                return;
        }

		if (sdp_mem_type(grid) != SDP_MEM_COMPLEX_DOUBLE ||
            sdp_mem_type(vis) != SDP_MEM_COMPLEX_DOUBLE ||
            sdp_mem_type(uv_kernel) != SDP_MEM_DOUBLE ||
            sdp_mem_type(w_kernel) != SDP_MEM_DOUBLE)
        {
                *status = SDP_ERR_DATA_TYPE;
                SDP_LOG_ERROR("Unsuported data type");
                return;
        }


    if (location == SDP_MEM_CPU)
    {

	int64_t uv_kernel_stride = sdp_mem_stride_dim(uv_kernel, 0);
	int64_t w_kernel_stride = sdp_mem_stride_dim(w_kernel, 0);
	
	int64_t vis_count = sdp_mem_shape_dim(vis_coordinates, 1);

	int64_t grid_size = sdp_mem_num_elements(grid);

	int64_t x_size = sqrt((grid_size / sdp_mem_type_size(sdp_mem_type(grid))));
	int64_t y_size = sqrt((grid_size / sdp_mem_type_size(sdp_mem_type(grid))));

	degridding(
		uv_kernel_stride,
		w_kernel_stride,
		x_size,
		y_size,
		vis_count,
		(const std::complex<double>*)sdp_mem_data_const(grid),
        (const double*)sdp_mem_data_const(vis_coordinates),
        (const double*)sdp_mem_data_const(uv_kernel),
        (const double*)sdp_mem_data_const(w_kernel),
        uv_kernel_oversampling,
        w_kernel_oversampling,
        theta,
        wstep, 
        conjugate, 
        (std::complex<double>*)sdp_mem_data(vis));

	// char message[500];
	// sprintf(message, "value is %d", grid_size);
	// SDP_LOG_INFO(message);

	// std::cout << "\n" << grid_size << "\n" << "\n";}}

	// const double* uvw = (const double*)sdp_mem_data_const(vis_coordinates);
	// const double* xyz = (const double*)sdp_mem_data_const(grid);
	
	}

	else if (location == SDP_MEM_GPU)
    {
        SDP_LOG_INFO("GPU not yet implemented");
        return;
    }
}