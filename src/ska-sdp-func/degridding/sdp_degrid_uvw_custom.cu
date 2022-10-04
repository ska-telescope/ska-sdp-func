/* See the LICENSE file at the top-level directory of this distribution. */

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
        int *sub_offset_z, //
        int *grid_coord_x,
        int *grid_coord_y
)
{
    // u or x coordinate
    const double ox = theta * u * oversample;
    int iox = __double2int_rn(ox); // round to nearest
    iox += (grid_size / 2 + 1) * oversample - 1;
    const int home_x = iox / oversample;
    const int frac_x = oversample - 1 - (iox % oversample);

    // v or y coordinate
    const double oy = theta * v * oversample;
    int ioy = __double2int_rn(oy);
    ioy += (grid_size / 2 + 1) * oversample - 1;
    const int home_y = ioy / oversample;
    const int frac_y = oversample - 1 - (ioy % oversample);

    // w or z coordinate
    const double oz = (1.0 + w / wstep) * oversample_w;
    const int ioz = __double2int_rn(oz) + oversample_w - 1;
    const int frac_z = oversample_w - 1 - (ioz % oversample_w);

    // FIXME Why is this NOT multiplied by x_stride? (c.f. CPU version).
    *grid_offset = (home_y - kernel_size / 2) * y_stride + (
            home_x - kernel_size / 2);
    *sub_offset_x = kernel_stride * frac_x;
    *sub_offset_y = kernel_stride * frac_y;
    *sub_offset_z = wkernel_stride * frac_z;
    *grid_coord_x = home_x;
    *grid_coord_y = home_y;
}

template<typename VIS_TYPE2>
__global__ void degrid_uvw_custom(
        const int64_t uv_kernel_stride_in_elements,
        const int64_t w_kernel_stride_in_elements,
        const int64_t x_size,
        const int64_t y_size,
        const int64_t num_times,
        const int64_t num_baselines,
        const int64_t num_channels,
        const int64_t num_pols,
        const VIS_TYPE2 *const __restrict__ grid,
        const double3 *const __restrict__ uvw,
        const double *const __restrict__ uv_kernel,
        const double *const __restrict__ w_kernel,
        const int64_t uv_kernel_oversampling,
        const int64_t w_kernel_oversampling,
        const double theta,
        const double wstep,
        const double channel_start_hz,
        const double channel_step_hz,
        const int32_t conjugate,
        VIS_TYPE2 *__restrict__ vis)
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

    for (int i_pol = 0; i_pol < num_pols; ++i_pol)
    {
        // Get uvw-coordinate scaling.
        const double inv_wavelength = (
                channel_start_hz + i_channel * channel_step_hz) / C_0;
        const unsigned int i_uvw = INDEX_2D(
                num_times, num_baselines, i_time, i_baseline);

        int grid_offset = 0;
        int sub_offset_x = 0, sub_offset_y = 0, sub_offset_z = 0;
        int grid_coord_x = 0, grid_coord_y = 0;
        calculate_coordinates(
                x_size,
                1,
                y_size,
                uv_kernel_stride_in_elements,
                uv_kernel_stride_in_elements,
                uv_kernel_oversampling,
                w_kernel_stride_in_elements,
                w_kernel_oversampling,
                theta,
                wstep,
                inv_wavelength * uvw[i_uvw].x,
                inv_wavelength * uvw[i_uvw].y,
                inv_wavelength * uvw[i_uvw].z,
                &grid_offset,
                &sub_offset_x,
                &sub_offset_y,
                &sub_offset_z,
                &grid_coord_x,
                &grid_coord_y
        );
        
        if(
            grid_coord_x > uv_kernel_stride_in_elements/2 &&
            grid_coord_x < grid_size - v_kernel_stride_in_elements/2 &&
            grid_coord_y > uv_kernel_stride_in_elements/2 &&
            grid_coord_y < grid_size - v_kernel_stride_in_elements/2)
        {

            VIS_TYPE2 vis_local = {0, 0};
            for (int z = 0; z < w_kernel_stride_in_elements; z++)
            {
                VIS_TYPE2 visz = {0, 0};
                for (int y = 0; y < uv_kernel_stride_in_elements; y++)
                {
                    VIS_TYPE2 visy = {0, 0};
                    for (int x = 0; x < uv_kernel_stride_in_elements; x++)
                    {
                            const VIS_TYPE2 grid_value = grid[
                                    z * x_size * y_size + grid_offset + y * y_size + x];
                            visy.x += uv_kernel[sub_offset_x + x] * grid_value.x;
                            visy.y += uv_kernel[sub_offset_x + x] * grid_value.y;
                    }
                    visz.x += uv_kernel[sub_offset_y + y] * visy.x;
                    visz.y += uv_kernel[sub_offset_y + y] * visy.y;
                }
                vis_local.x += w_kernel[sub_offset_z + z] * visz.x;
                vis_local.y += w_kernel[sub_offset_z + z] * visz.y;
            }
            if (conjugate) vis_local.y = -vis_local.y;

            const unsigned int i_out = INDEX_4D(
                    num_times, num_baselines, num_channels, num_pols,
                    i_time, i_baseline, i_channel, i_pol);
            vis[i_out] = vis_local;
        }
    }
}

SDP_CUDA_KERNEL(degrid_uvw_custom<double2>)
