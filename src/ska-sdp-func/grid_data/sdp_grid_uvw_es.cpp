/* See the LICENSE file at the top-level directory of this distribution. */

#include <complex>
#include <cstdlib>

#include "ska-sdp-func/grid_data/sdp_grid_uvw_es.h"
#include "ska-sdp-func/grid_data/sdp_gridder_uvw_es_fft_utils.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"

void sdp_grid_uvw_es(
        const sdp_Mem* uvw,
        const sdp_Mem* vis,
        const sdp_Mem* weights,
        const sdp_Mem* freq_hz,
        int image_size,
        double epsilon,
        double cell_size_rad,
        double w_scale,
        double min_plane_w,
        int sub_grid_start_u,
        int sub_grid_start_v,
        int sub_grid_w,
        sdp_Mem* sub_grid,
        sdp_Error* status
)
{
    if (*status) return;

    // Get full grid size, beta parameter and support size from epsilon.
    int support = 0, full_grid_size = 0;
    double beta = 0.0;
    const sdp_MemType vis_type = sdp_mem_type(vis);
    const int dbl_vis = (vis_type & SDP_MEM_DOUBLE);
    const int vis_prec = (vis_type & SDP_MEM_DOUBLE) ?
                SDP_MEM_DOUBLE : SDP_MEM_FLOAT;
    const sdp_MemType coord_type = sdp_mem_type(uvw);
    const int dbl_coord = (coord_type & SDP_MEM_DOUBLE);
    sdp_calculate_params_from_epsilon(
            epsilon, image_size, vis_prec, full_grid_size, support, beta, status
    );
    const float beta_f = (float) beta;
    const double uv_scale = full_grid_size * cell_size_rad;
    const float uv_scale_f = (float) uv_scale;
    const float w_scale_f = (float) w_scale;
    const float min_plane_w_f = (float) min_plane_w;

    // Data dimensions.
    const int num_rows = (int)sdp_mem_shape_dim(vis, 0);
    const int num_chan = (int)sdp_mem_shape_dim(freq_hz, 0);
    const int sub_grid_size = (int)sdp_mem_shape_dim(sub_grid, 0);

    // Grid visibilities onto this w-plane.
    const char* kernel_name = 0;
    if (dbl_vis && dbl_coord)
    {
        kernel_name = "sdp_grid_uvw_es_cuda_3d"
                "<double, double2, double, double2, double3>";
    }
    else if (!dbl_vis && !dbl_coord)
    {
        kernel_name = "sdp_grid_uvw_es_cuda_3d"
                "<float, float2, float, float2, float3>";
    }
    if (kernel_name)
    {
        uint64_t num_threads[] = {1, 256, 1}, num_blocks[] = {1, 1, 1};
        void* null = 0;
        num_blocks[0] = (num_chan + num_threads[0] - 1) / num_threads[0];
        num_blocks[1] = (num_rows + num_threads[1] - 1) / num_threads[1];
        const void* args[] = {
            &num_rows,
            &num_chan,
            sdp_mem_gpu_buffer_const(vis, status),
            sdp_mem_gpu_buffer_const(weights, status),
            sdp_mem_gpu_buffer_const(uvw, status),
            sdp_mem_gpu_buffer_const(freq_hz, status),
            &support,
            dbl_vis ? (const void*)&beta : (const void*)&beta_f,
            dbl_coord ? (const void*)&uv_scale : (const void*)&uv_scale_f,
            dbl_coord ? (const void*)&w_scale : (const void*)&w_scale_f,
            dbl_coord ? (const void*)&min_plane_w : (const void*)&min_plane_w_f,
            &sub_grid_start_u,
            &sub_grid_start_v,
            &sub_grid_w,
            &full_grid_size,
            &sub_grid_size,
            sdp_mem_gpu_buffer(sub_grid, status),
            &null
        };
        sdp_launch_cuda_kernel(
                kernel_name, num_blocks, num_threads, 0, 0, args, status
        );
    }
}
