/* See the LICENSE file at the top-level directory of this distribution. */

#include "ska-sdp-func/grid_data/sdp_degrid_uvw_custom.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"

#include <math.h>
#include <complex>
#include <vector>

#define C_0 299792458.0
#define INDEX_3D(N3, N2, N1, I3, I2, I1)         (N1 * (N2 * I3 + I2) + I1)
#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) \
    (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)

using std::complex;


static inline void calculate_coordinates(
        int64_t grid_size, // dimension of the image's subgrid grid_size x grid_size x 4?
        int x_stride, // padding in x dimension
        int y_stride, // padding in y dimension
        int kernel_size, // gcf kernel support
        int kernel_stride, // padding of the gcf kernel
        int oversample, // oversampling of the uv kernel
        int wkernel_stride, // padding of the gcf w kernel
        int oversample_w, // oversampling of the w kernel
        double theta, // conversion parameter from uv coordinates to xy coordinates x=u*theta
        double wstep, // conversion parameter from w coordinates to z coordinates z=w*wstep
        double u, //
        double v, // coordinates of the visibility
        double w, //
        int* grid_offset, // offset in the image subgrid
        int* sub_offset_x, //
        int* sub_offset_y, // fractional coordinates
        int* sub_offset_z, //
        int* grid_coord_x, // grid coordinates
        int* grid_coord_y  //
)
{
    // u or x coordinate
    const double ox = theta * u * oversample;
    const int iox = (int)round(ox) + (grid_size / 2 + 1) * oversample - 1;
    const int home_x = iox / oversample;
    const int frac_x = oversample - 1 - (iox % oversample);

    // v or y coordinate
    const double oy = theta * v * oversample;
    const int ioy = (int)round(oy) + (grid_size / 2 + 1) * oversample - 1;
    const int home_y = ioy / oversample;
    const int frac_y = oversample - 1 - (ioy % oversample);

    // w or z coordinate
    const double oz = (1.0 + w / wstep) * oversample_w;
    const int ioz = (int)round(oz) + oversample_w - 1;
    const int frac_z = oversample_w - 1 - (ioz % oversample_w);

    // FIXME Why is this multiplied by x_stride? (c.f. GPU version).
    *grid_offset = (home_y - kernel_size / 2) * y_stride + (
        home_x - kernel_size / 2) * x_stride;
    *sub_offset_x = kernel_stride * frac_x;
    *sub_offset_y = kernel_stride * frac_y;
    *sub_offset_z = wkernel_stride * frac_z;
    *grid_coord_x = home_x;
    *grid_coord_y = home_y;
}


template<typename VIS_TYPE>
static void degrid_uvw_custom(
        const int64_t uv_kernel_size,
        const int64_t w_kernel_size,
        const int64_t x_size,
        const int64_t y_size,
        const int64_t num_times,
        const int64_t num_baselines,
        const int64_t num_channels,
        const int64_t num_pols,
        const complex<VIS_TYPE>* grid,
        const double* uvw,
        const double* uv_kernel,
        const double* w_kernel,
        const int64_t uv_kernel_oversampling,
        const int64_t w_kernel_oversampling,
        const double theta,
        const double wstep,
        const double channel_start_hz,
        const double channel_step_hz,
        const int32_t conjugate,
        complex<VIS_TYPE>* vis
)
{
    for (int i_time = 0; i_time < num_times; ++i_time)
    {
        for (int i_baseline = 0; i_baseline < num_baselines; ++i_baseline)
        {
            const unsigned int i_uvw = INDEX_3D(
                    num_times, num_baselines, 3,
                    i_time, i_baseline, 0
            );
            const double baseline_u_metres = uvw[i_uvw];
            const double baseline_v_metres = uvw[i_uvw + 1];
            const double baseline_w_metres = uvw[i_uvw + 2];
            for (int i_channel = 0; i_channel < num_channels; ++i_channel)
            {
                // Get uvw-coordinate scaling.
                const double inv_wavelength =
                        (channel_start_hz + i_channel * channel_step_hz) / C_0;

                int grid_offset = 0;
                int sub_offset_x = 0, sub_offset_y = 0, sub_offset_z = 0;
                int grid_coord_x = 0, grid_coord_y = 0;
                calculate_coordinates(
                        x_size,
                        1,
                        y_size,
                        uv_kernel_size,
                        uv_kernel_size,
                        uv_kernel_oversampling,
                        w_kernel_size,
                        w_kernel_oversampling,
                        theta,
                        wstep,
                        inv_wavelength * baseline_u_metres,
                        inv_wavelength * baseline_v_metres,
                        inv_wavelength * baseline_w_metres,
                        &grid_offset,
                        &sub_offset_x,
                        &sub_offset_y,
                        &sub_offset_z,
                        &grid_coord_x,
                        &grid_coord_y
                );

                // Check point is fully within the grid.
                if (!(grid_coord_x > uv_kernel_size / 2 &&
                        grid_coord_x < x_size - uv_kernel_size / 2 &&
                        grid_coord_y > uv_kernel_size / 2 &&
                        grid_coord_y < y_size - uv_kernel_size / 2))
                {
                    continue;
                }

                // De-grid each polarisation.
                for (int i_pol = 0; i_pol < num_pols; ++i_pol)
                {
                    complex<VIS_TYPE> vis_local(0, 0);
                    for (int z = 0; z < w_kernel_size; z++)
                    {
                        complex<VIS_TYPE> visz(0, 0);
                        for (int y = 0; y < uv_kernel_size; y++)
                        {
                            complex<VIS_TYPE> visy(0, 0);
                            for (int x = 0; x < uv_kernel_size; x++)
                            {
                                // FIXME Use polarisation index here.
                                const complex<VIS_TYPE> value = grid[
                                    z * x_size * y_size + grid_offset +
                                    y * y_size + x];
                                visy += uv_kernel[sub_offset_x + x] * value;
                            }
                            visz += uv_kernel[sub_offset_y + y] * visy;
                        }
                        vis_local += w_kernel[sub_offset_z + z] * visz;
                    }
                    if (conjugate) vis_local = std::conj(vis_local);

                    const unsigned int i_out = INDEX_4D(
                            num_times, num_baselines, num_channels, num_pols,
                            i_time, i_baseline, i_channel, i_pol
                    );
                    vis[i_out] = vis_local;
                }
            }
        }
    }
}


void sdp_degrid_uvw_custom(
        const sdp_Mem* grid,
        const sdp_Mem* uvw,
        const sdp_Mem* uv_kernel,
        const sdp_Mem* w_kernel,
        const double theta,
        const double wstep,
        const double channel_start_hz,
        const double channel_step_hz,
        const int32_t conjugate,
        sdp_Mem* vis,
        sdp_Error* status
)
{
    if (*status) return;

    const sdp_MemLocation location = sdp_mem_location(vis);

    if (sdp_mem_is_read_only(vis))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Output visibility must be writable.");
        return;
    }

    if (sdp_mem_location(grid) != location ||
            sdp_mem_location(uvw) != location ||
            sdp_mem_location(uv_kernel) != location ||
            sdp_mem_location(w_kernel) != location ||
            sdp_mem_location(vis) != location)
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Memory location mismatch");
        return;
    }

    if (sdp_mem_type(uvw) != SDP_MEM_DOUBLE ||
            sdp_mem_type(uv_kernel) != SDP_MEM_DOUBLE ||
            sdp_mem_type(w_kernel) != SDP_MEM_DOUBLE)
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsuported data type");
        return;
    }

    if (sdp_mem_type(vis) != SDP_MEM_COMPLEX_DOUBLE)
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Visibility values must be complex doubles");
        return;
    }

    if (sdp_mem_type(grid) != SDP_MEM_COMPLEX_DOUBLE)
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Grid values must be complex doubles");
        return;
    }

    const int64_t uv_kernel_size = sdp_mem_shape_dim(uv_kernel, 1);
    const int64_t uv_kernel_oversampling = sdp_mem_shape_dim(uv_kernel, 0);

    const int64_t w_kernel_size = sdp_mem_shape_dim(w_kernel, 1);
    const int64_t w_kernel_oversampling = sdp_mem_shape_dim(w_kernel, 0);

    const int64_t num_times = sdp_mem_shape_dim(vis, 0);
    const int64_t num_baselines = sdp_mem_shape_dim(vis, 1);
    const int64_t num_channels = sdp_mem_shape_dim(vis, 2);
    const int64_t num_pols = sdp_mem_shape_dim(vis, 3);

    const int64_t x_size = sdp_mem_shape_dim(grid, 2);
    const int64_t y_size = sdp_mem_shape_dim(grid, 3);

    if ((num_pols != 1 && num_pols != 2 && num_pols != 4) ||
            (sdp_mem_shape_dim(grid, 4) != num_pols))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Unsupported number of polarisations, must be 1, 2 or 4");
        return;
    }

    if (num_channels != 1 || sdp_mem_shape_dim(grid, 0) != 1)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Unsupported number of channels, must be 1");
        return;
    }

    if (sdp_mem_shape_dim(grid, 0) != num_channels ||
            sdp_mem_shape_dim(grid, 4) != num_pols)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The grid array must have shape "
                "[num_channels, w, v, u, num_pols] "
                "(expected [%d, w, v, u, %d])",
                num_channels, num_pols
        );
        return;
    }

    if (sdp_mem_shape_dim(uvw, 0) != num_times ||
            sdp_mem_shape_dim(uvw, 1) != num_baselines)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The uvw array must have shape "
                "[num_times, num_baslines, 3] (expected [%d, %d, 3])",
                num_times, num_baselines
        );
        return;
    }

    if (location == SDP_MEM_CPU)
    {
        degrid_uvw_custom(
                uv_kernel_size,
                w_kernel_size,
                x_size,
                y_size,
                num_times,
                num_baselines,
                num_channels,
                num_pols,
                (const complex<double>*)sdp_mem_data_const(grid),
                (const double*)sdp_mem_data_const(uvw),
                (const double*)sdp_mem_data_const(uv_kernel),
                (const double*)sdp_mem_data_const(w_kernel),
                uv_kernel_oversampling,
                w_kernel_oversampling,
                theta,
                wstep,
                channel_start_hz,
                channel_step_hz,
                conjugate,
                (complex<double>*)sdp_mem_data(vis)
        );
    }
    else if (location == SDP_MEM_GPU)
    {
        const uint64_t num_threads[] = {128, 2, 2};
        const uint64_t num_blocks[] = {
            (num_baselines + num_threads[0] - 1) / num_threads[0],
            (num_channels + num_threads[1] - 1) / num_threads[1],
            (num_times + num_threads[2] - 1) / num_threads[2]
        };
        const void* args[] = {
            &uv_kernel_size,
            &w_kernel_size,
            &x_size,
            &y_size,
            &num_times,
            &num_baselines,
            &num_channels,
            &num_pols,
            sdp_mem_gpu_buffer_const(grid, status),
            sdp_mem_gpu_buffer_const(uvw, status),
            sdp_mem_gpu_buffer_const(uv_kernel, status),
            sdp_mem_gpu_buffer_const(w_kernel, status),
            &uv_kernel_oversampling,
            &w_kernel_oversampling,
            &theta,
            &wstep,
            &channel_start_hz,
            &channel_step_hz,
            &conjugate,
            sdp_mem_gpu_buffer(vis, status)
        };
        sdp_launch_cuda_kernel("degrid_uvw_custom<double2>",
                num_blocks, num_threads, 0, 0, args, status
        );
    }
}
