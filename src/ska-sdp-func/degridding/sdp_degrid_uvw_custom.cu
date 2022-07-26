/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <vector>
#include <stdio.h>

#include "ska-sdp-func/utility/sdp_device_wrapper.h"

#define C_0 299792458.0
#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)
#define INDEX_3D(N3, N2, N1, I3, I2, I1)         (N1 * (N2 * I3 + I2) + I1)
#define INDEX_2D(N2, N1, I2, I1)                 (N1 * I2 + I1)


__device__ __inline__ void calculate_coordinates(
        int64_t grid_size, //dimension of the image's subgrid grid_size x grid_size x 4?
        int x_stride, // padding in x dimension
        int y_stride, // padding in y dimension
        int kernel_size, // gcf kernel support
        int kernel_stride, // padding of the gcf kernel
        int oversample, // oversampling of the uv kernel
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
        int iox = __double2int_rn(ox); // round to nearest
        iox += (grid_size / 2 + 1) * oversample - 1;
        int home_x = iox / oversample;
        int frac_x = oversample - 1 - (iox % oversample);
        
        // y coordinate
        double y = theta*v;
        double oy = y*oversample;
        int ioy = __double2int_rn(oy);
        ioy += (grid_size / 2 + 1) * oversample - 1;
        int home_y = ioy / oversample;
        int frac_y = oversample - 1 - (ioy % oversample);
        
        // w coordinate
        double z = 1.0 + w/wstep;
        double oz = z*oversample_w;
        int ioz = __double2int_rn(oz);
        ioz += oversample_w - 1;
        int frac_z = oversample_w - 1 - (ioz % oversample_w);
        
        *grid_offset = (home_y-kernel_size/2)*y_stride + (home_x-kernel_size/2);
        *sub_offset_x = kernel_stride * frac_x;
        *sub_offset_y = kernel_stride * frac_y;
        *sub_offset_z = wkernel_stride * frac_z;
}


__global__ void degrid_uvw_custom(
    const int64_t uv_kernel_stride_in_elements,
    const int64_t w_kernel_stride_in_elements,
    const int64_t x_size,
    const int64_t y_size,
    const int64_t num_times,
    const int64_t num_baselines,
    const int64_t num_channels,
    const int64_t num_pols,
    const double2* grid,
    const double3* uvw,
    const double* uv_kernel,
    const double* w_kernel,
    const int64_t uv_kernel_oversampling,
    const int64_t w_kernel_oversampling,
    const double theta,
    const double wstep,
    const double channel_start_hz,
    const double channel_step_hz,
    const bool conjugate, 
    double2* vis)
    {

        // Get indices of the output array this thread is working on.
        const int i_baseline = blockDim.x * blockIdx.x + threadIdx.x;
        const int i_channel  = blockDim.y * blockIdx.y + threadIdx.y;
        const int i_time     = blockDim.z * blockIdx.z + threadIdx.z;

        // Bounds check.
        if (i_baseline >= num_baselines ||
                i_channel >= num_channels ||
                i_time >= num_times)
        {
            return;
        }

        // Load uvw-coordinates.
        const double inv_wavelength = (channel_start_hz + i_channel*channel_step_hz) / C_0;
        //const unsigned int i_uvw = INDEX_3D(
        //        num_times, num_baselines, num_channels,
        //        i_time, i_baseline, i_channel);
        //const double3 uvw_vis_coordinates = uvw[i_uvw];
        const unsigned int i_uvw = INDEX_2D(num_times, num_baselines, i_time, i_baseline);
        const double3 uvw_vis_coordinates = uvw[i_uvw];

        int grid_offset, sub_offset_x, sub_offset_y, sub_offset_z;

        calculate_coordinates(
            x_size, 1, y_size,
            uv_kernel_stride_in_elements, uv_kernel_stride_in_elements, uv_kernel_oversampling,
            w_kernel_stride_in_elements, w_kernel_oversampling,
            theta, wstep, 
            uvw_vis_coordinates.x*inv_wavelength, 
            uvw_vis_coordinates.y*inv_wavelength, 
            uvw_vis_coordinates.z*inv_wavelength,
            &grid_offset, 
            &sub_offset_x, &sub_offset_y, &sub_offset_z
        );
        
        // double vis_r = 0.0, vis_i = 0.0;
        double2 visx;
        for (int z = 0; z < w_kernel_stride_in_elements; z++) 
        {
            double visz_r = 0, visz_i = 0;
            for (int y = 0; y < uv_kernel_stride_in_elements; y++) 
            {
                double visy_r = 0, visy_i = 0;
                for (int x = 0; x < uv_kernel_stride_in_elements; x++) 
                {
                    double2 grid_value = grid[z*x_size*y_size + grid_offset + y*y_size + x];

                    visy_r += uv_kernel[sub_offset_x + x] * grid_value.x;
                    visy_i += uv_kernel[sub_offset_x + x] * grid_value.y;
                }
                visz_r += uv_kernel[sub_offset_y + y] * visy_r;
                visz_i += uv_kernel[sub_offset_y + y] * visy_i;
            }
            visx.x += w_kernel[sub_offset_z + z] * visz_r;
            visx.y += w_kernel[sub_offset_z + z] * visz_i;

        }

        double2 temp_result;
        temp_result.x = visx.x;

        if(conjugate)temp_result.y = -visx.y;
        else temp_result.y = visx.y;

        for (int i_pol = 0; i_pol < num_pols; ++i_pol)
            {
                const unsigned int i_out = INDEX_4D(num_times, num_baselines,
                num_channels, num_pols, i_time, i_baseline, i_channel, i_pol);
                vis[i_out] = temp_result;

            } 
}


SDP_CUDA_KERNEL(degrid_uvw_custom)