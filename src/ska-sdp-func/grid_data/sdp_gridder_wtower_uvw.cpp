/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>
#include <cstdlib>

#include "ska-sdp-func/fourier_transforms/sdp_fft.h"
#include "ska-sdp-func/fourier_transforms/sdp_pswf.h"
#include "ska-sdp-func/grid_data/sdp_gridder_clamp_channels.h"
#include "ska-sdp-func/grid_data/sdp_gridder_utils.h"
#include "ska-sdp-func/grid_data/sdp_gridder_wtower_uvw.h"
#include "ska-sdp-func/math/sdp_math_macros.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"

using std::complex;

struct sdp_GridderWtowerUVW
{
    int image_size;
    int subgrid_size;
    double theta;
    double w_step;
    double shear_u;
    double shear_v;
    int support;
    int vr_size;
    int oversampling;
    int w_support;
    int w_oversampling;
    sdp_PSWF* pswf_n_func;
    sdp_Mem* pswf;
    sdp_Mem* uv_kernel;
    sdp_Mem* uv_kernel_gpu;
    sdp_Mem* w_kernel;
    sdp_Mem* w_kernel_gpu;
    sdp_Mem* w_pattern;
    sdp_Mem* w_pattern_gpu;
};

// Begin anonymous namespace for file-local functions.
namespace {

// Local function to do the degridding.
template<typename UVW_TYPE, typename VIS_TYPE>
void degrid(
        const sdp_GridderWtowerUVW* plan,
        const VIS_TYPE* subgrids,
        int w_plane,
        int subgrid_offset_u,
        int subgrid_offset_v,
        int subgrid_offset_w,
        double freq0_hz,
        double dfreq_hz,
        const sdp_MemViewCpu<const UVW_TYPE, 2>& uvws,
        const sdp_MemViewCpu<const int, 1>& start_chs,
        const sdp_MemViewCpu<const int, 1>& end_chs,
        sdp_MemViewCpu<VIS_TYPE, 2>& vis,
        sdp_Error* status
)
{
    if (*status) return;
    const double* uv_kernel =
            (const double*) sdp_mem_data_const(plan->uv_kernel);
    const double* w_kernel =
            (const double*) sdp_mem_data_const(plan->w_kernel);
    const int64_t num_uvw = uvws.shape[0];
    const int64_t subgrid_size = plan->subgrid_size;
    const int64_t subgrid_square = subgrid_size * subgrid_size;
    const int half_subgrid = plan->subgrid_size / 2;
    const double half_vr_m1 = (plan->vr_size - 1) / 2.0;
    const int half_vr = plan->vr_size / 2;
    const int oversample = plan->oversampling;
    const int w_oversample = plan->w_oversampling;
    const double theta = plan->theta;

    // Loop over rows. Each row contains visibilities for all channels.
    #pragma omp parallel for
    for (int64_t i_row = 0; i_row < num_uvw; ++i_row)
    {
        // Skip if there's no visibility to degrid.
        int64_t start_ch = start_chs(i_row), end_ch = end_chs(i_row);
        if (start_ch >= end_ch)
            continue;

        // Select only visibilities on this w-plane.
        const UVW_TYPE uvw[] = {uvws(i_row, 0), uvws(i_row, 1), uvws(i_row, 2)};
        const double min_w = (w_plane + subgrid_offset_w - 1) * plan->w_step;
        const double max_w = (w_plane + subgrid_offset_w) * plan->w_step;
        sdp_gridder_clamp_channels_inline(
                uvw[2], freq0_hz, dfreq_hz, &start_ch, &end_ch, min_w, max_w
        );
        if (start_ch >= end_ch)
            continue;

        // Scale + shift UVWs.
        const double s_uvw0 = freq0_hz / C_0, s_duvw = dfreq_hz / C_0;
        double uvw0[] = {uvw[0] * s_uvw0, uvw[1] * s_uvw0, uvw[2] * s_uvw0};
        double duvw[] = {uvw[0] * s_duvw, uvw[1] * s_duvw, uvw[2] * s_duvw};
        uvw0[0] -= subgrid_offset_u / plan->theta;
        uvw0[1] -= subgrid_offset_v / plan->theta;
        uvw0[2] -= ((subgrid_offset_w + w_plane) * plan->w_step);

        // Loop over selected channels.
        for (int64_t c = start_ch; c < end_ch; c++)
        {
            const double u = uvw0[0] + c * duvw[0];
            const double v = uvw0[1] + c * duvw[1];
            const double w = uvw0[2] + c * duvw[2];

            // Determine top-left corner of grid region
            // centered approximately on visibility.
            const int iu0 = int(round(theta * u - half_vr_m1)) + half_subgrid;
            const int iv0 = int(round(theta * v - half_vr_m1)) + half_subgrid;
            const int iu_shift = iu0 + half_vr - half_subgrid;
            const int iv_shift = iv0 + half_vr - half_subgrid;

            // Determine which kernel to use.
            int u_off = int(round((u * theta - iu_shift + 1) * oversample));
            int v_off = int(round((v * theta - iv_shift + 1) * oversample));
            int w_off = int(round((w / plan->w_step + 1) * w_oversample));

            // Cater for the negative indexing which is allowed in Python!
            if (u_off < 0) u_off += oversample + 1;
            if (v_off < 0) v_off += oversample + 1;
            if (w_off < 0) w_off += w_oversample + 1;
            u_off *= plan->support;
            v_off *= plan->support;
            w_off *= plan->w_support;

            // Degrid visibility.
            VIS_TYPE local_vis = (VIS_TYPE) 0;
            for (int iw = 0; iw < plan->w_support; ++iw)
            {
                const double kern_w = w_kernel[w_off + iw];
                for (int iu = 0; iu < plan->support; ++iu)
                {
                    const double kern_wu = kern_w * uv_kernel[u_off + iu];
                    for (int iv = 0; iv < plan->support; ++iv)
                    {
                        const double kern_wuv = kern_wu * uv_kernel[v_off + iv];
                        int ix_u = iu0 + iu;
                        int ix_v = iv0 + iv;
                        if (ix_u < 0) ix_u += plan->subgrid_size;
                        if (ix_v < 0) ix_v += plan->subgrid_size;
                        const int64_t idx = (
                            iw * subgrid_square + ix_u * subgrid_size + ix_v
                        );
                        local_vis += ((VIS_TYPE) kern_wuv * subgrids[idx]);
                    }
                }
            }
            vis(i_row, c) = local_vis;
        }
    }
}


// Local function to call the CPU degridding kernel.
void degrid_cpu(
        const sdp_GridderWtowerUVW* plan,
        const sdp_Mem* subgrids,
        int w_plane,
        int subgrid_offset_u,
        int subgrid_offset_v,
        int subgrid_offset_w,
        double freq0_hz,
        double dfreq_hz,
        const sdp_Mem* uvws,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        sdp_Mem* vis,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_MemViewCpu<const int, 1> start_chs_, end_chs_;
    sdp_mem_check_and_view(start_chs, &start_chs_, status);
    sdp_mem_check_and_view(end_chs, &end_chs_, status);
    if (sdp_mem_type(uvws) == SDP_MEM_DOUBLE &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE)
    {
        sdp_MemViewCpu<complex<double>, 2> vis_;
        sdp_MemViewCpu<const double, 2> uvws_;
        sdp_mem_check_and_view(vis, &vis_, status);
        sdp_mem_check_and_view(uvws, &uvws_, status);
        degrid<double, complex<double> >(
                plan, (const complex<double>*) sdp_mem_data_const(subgrids),
                w_plane, subgrid_offset_u, subgrid_offset_v, subgrid_offset_w,
                freq0_hz, dfreq_hz, uvws_, start_chs_, end_chs_, vis_, status
        );
    }
    else if (sdp_mem_type(uvws) == SDP_MEM_FLOAT &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT)
    {
        sdp_MemViewCpu<complex<float>, 2> vis_;
        sdp_MemViewCpu<const float, 2> uvws_;
        sdp_mem_check_and_view(vis, &vis_, status);
        sdp_mem_check_and_view(uvws, &uvws_, status);
        degrid<float, complex<float> >(
                plan, (const complex<float>*) sdp_mem_data_const(subgrids),
                w_plane, subgrid_offset_u, subgrid_offset_v, subgrid_offset_w,
                freq0_hz, dfreq_hz, uvws_, start_chs_, end_chs_, vis_, status
        );
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
    }
}


// Local function to call the GPU degridding kernel.
void degrid_gpu(
        const sdp_GridderWtowerUVW* plan,
        const sdp_Mem* subgrids,
        int w_plane,
        int subgrid_offset_u,
        int subgrid_offset_v,
        int subgrid_offset_w,
        double freq0_hz,
        double dfreq_hz,
        const sdp_Mem* uvws,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        sdp_Mem* vis,
        sdp_Error* status
)
{
    if (*status) return;
    uint64_t num_threads[] = {256, 1, 1}, num_blocks[] = {1, 1, 1};
    const char* kernel_name = 0;
    int is_dbl = 0;
    const int64_t num_rows = sdp_mem_shape_dim(vis, 0);
    sdp_MemViewGpu<const double, 2> uvws_dbl;
    sdp_MemViewGpu<const float, 2> uvws_flt;
    sdp_MemViewGpu<complex<double>, 2> vis_dbl;
    sdp_MemViewGpu<complex<float>, 2> vis_flt;
    sdp_MemViewGpu<const int, 1> start_chs_, end_chs_;
    sdp_mem_check_and_view(start_chs, &start_chs_, status);
    sdp_mem_check_and_view(end_chs, &end_chs_, status);
    if (sdp_mem_type(uvws) == SDP_MEM_DOUBLE &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE)
    {
        is_dbl = 1;
        sdp_mem_check_and_view(uvws, &uvws_dbl, status);
        sdp_mem_check_and_view(vis, &vis_dbl, status);
        kernel_name = "sdp_gridder_wtower_degrid<double, complex<double> >";
    }
    else if (sdp_mem_type(uvws) == SDP_MEM_FLOAT &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT)
    {
        is_dbl = 0;
        sdp_mem_check_and_view(uvws, &uvws_flt, status);
        sdp_mem_check_and_view(vis, &vis_flt, status);
        kernel_name = "sdp_gridder_wtower_degrid<float, complex<float> >";
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
    }
    const void* arg[] = {
        sdp_mem_gpu_buffer_const(subgrids, status),
        &w_plane,
        &subgrid_offset_u,
        &subgrid_offset_v,
        &subgrid_offset_w,
        (const void*) &freq0_hz,
        (const void*) &dfreq_hz,
        is_dbl ? (const void*) &uvws_dbl : (const void*) &uvws_flt,
        (const void*) &start_chs_,
        (const void*) &end_chs_,
        sdp_mem_gpu_buffer_const(plan->uv_kernel_gpu, status),
        sdp_mem_gpu_buffer_const(plan->w_kernel_gpu, status),
        &plan->subgrid_size,
        &plan->vr_size,
        &plan->support,
        &plan->w_support,
        &plan->oversampling,
        &plan->w_oversampling,
        (const void*) &plan->theta,
        (const void*) &plan->w_step,
        is_dbl ? (const void*) &vis_dbl : (const void*) &vis_flt
    };
    num_blocks[0] = (num_rows + num_threads[0] - 1) / num_threads[0];
    sdp_launch_cuda_kernel(kernel_name,
            num_blocks, num_threads, 0, 0, arg, status
    );
}


// Local function to do the gridding.
template<typename UVW_TYPE, typename VIS_TYPE>
void grid(
        const sdp_GridderWtowerUVW* plan,
        VIS_TYPE* subgrids,
        int w_plane,
        int subgrid_offset_u,
        int subgrid_offset_v,
        int subgrid_offset_w,
        double freq0_hz,
        double dfreq_hz,
        const sdp_MemViewCpu<const UVW_TYPE, 2>& uvws,
        const sdp_MemViewCpu<const int, 1>& start_chs,
        const sdp_MemViewCpu<const int, 1>& end_chs,
        const sdp_MemViewCpu<const VIS_TYPE, 2>& vis,
        sdp_Error* status
)
{
    if (*status) return;
    const double* uv_kernel =
            (const double*) sdp_mem_data_const(plan->uv_kernel);
    const double* w_kernel =
            (const double*) sdp_mem_data_const(plan->w_kernel);
    const int64_t num_uvw = uvws.shape[0];
    const int64_t subgrid_size = plan->subgrid_size;
    const int64_t subgrid_square = subgrid_size * subgrid_size;
    const int half_subgrid = plan->subgrid_size / 2;
    const double half_vr_m1 = (plan->vr_size - 1) / 2.0;
    const int half_vr = plan->vr_size / 2;
    const int oversample = plan->oversampling;
    const int w_oversample = plan->w_oversampling;
    const double theta = plan->theta;

    // Loop over rows. Each row contains visibilities for all channels.
    for (int64_t i_row = 0; i_row < num_uvw; ++i_row)
    {
        // Skip if there's no visibility to grid.
        int64_t start_ch = start_chs(i_row), end_ch = end_chs(i_row);
        if (start_ch >= end_ch)
            continue;

        // Select only visibilities on this w-plane.
        const UVW_TYPE uvw[] = {uvws(i_row, 0), uvws(i_row, 1), uvws(i_row, 2)};
        const double min_w = (w_plane + subgrid_offset_w - 1) * plan->w_step;
        const double max_w = (w_plane + subgrid_offset_w) * plan->w_step;
        sdp_gridder_clamp_channels_inline(
                uvw[2], freq0_hz, dfreq_hz, &start_ch, &end_ch, min_w, max_w
        );
        if (start_ch >= end_ch)
            continue;

        // Scale + shift UVWs.
        const double s_uvw0 = freq0_hz / C_0, s_duvw = dfreq_hz / C_0;
        double uvw0[] = {uvw[0] * s_uvw0, uvw[1] * s_uvw0, uvw[2] * s_uvw0};
        double duvw[] = {uvw[0] * s_duvw, uvw[1] * s_duvw, uvw[2] * s_duvw};
        uvw0[0] -= subgrid_offset_u / plan->theta;
        uvw0[1] -= subgrid_offset_v / plan->theta;
        uvw0[2] -= ((subgrid_offset_w + w_plane) * plan->w_step);

        // Loop over selected channels.
        for (int64_t c = start_ch; c < end_ch; c++)
        {
            const double u = uvw0[0] + c * duvw[0];
            const double v = uvw0[1] + c * duvw[1];
            const double w = uvw0[2] + c * duvw[2];

            // Determine top-left corner of grid region
            // centered approximately on visibility.
            const int iu0 = int(round(theta * u - half_vr_m1)) + half_subgrid;
            const int iv0 = int(round(theta * v - half_vr_m1)) + half_subgrid;
            const int iu_shift = iu0 + half_vr - half_subgrid;
            const int iv_shift = iv0 + half_vr - half_subgrid;

            // Determine which kernel to use.
            int u_off = int(round((u * theta - iu_shift + 1) * oversample));
            int v_off = int(round((v * theta - iv_shift + 1) * oversample));
            int w_off = int(round((w / plan->w_step + 1) * w_oversample));
            // Comment from Peter:
            // For future reference - at least on CPU the memory latency for
            // accessing the kernel is the main bottleneck of (de)gridding.
            // This can be mitigated quite well by pre-fetching the next kernel
            // value before starting to (de)grid the current one.

            // Cater for the negative indexing which is allowed in Python!
            // Comment from Peter: The task is to find e.g. iu0 and u_offset
            // such that iu0 + u_offset / oversampling + support / 2
            // most closely approximates u.
            // TODO The proper way of doing this would likely be to calculate
            // u * oversampling - support * (oversampling / 2), round that and
            // take integer division / modulo oversampling of that to obtain
            // iu0 and u_offset respectively.
            if (u_off < 0) u_off += oversample + 1;
            if (v_off < 0) v_off += oversample + 1;
            if (w_off < 0) w_off += w_oversample + 1;
            u_off *= plan->support;
            v_off *= plan->support;
            w_off *= plan->w_support;

            // Grid visibility.
            const complex<double> local_vis = vis(i_row, c);
            for (int iw = 0; iw < plan->w_support; ++iw)
            {
                const double kern_w = w_kernel[w_off + iw];
                for (int iu = 0; iu < plan->support; ++iu)
                {
                    const double kern_wu = kern_w * uv_kernel[u_off + iu];
                    for (int iv = 0; iv < plan->support; ++iv)
                    {
                        const double kern_wuv = kern_wu * uv_kernel[v_off + iv];
                        int ix_u = iu0 + iu;
                        int ix_v = iv0 + iv;
                        if (ix_u < 0) ix_u += plan->subgrid_size;
                        if (ix_v < 0) ix_v += plan->subgrid_size;
                        const int64_t idx = (
                            iw * subgrid_square + ix_u * subgrid_size + ix_v
                        );
                        const complex<double> grid_val = (
                                (complex<double>) kern_wuv * local_vis
                        );
                        subgrids[idx] += (VIS_TYPE) grid_val;
                    }
                }
            }
        }
    }
}


// Local function to call the CPU gridding kernel.
void grid_cpu(
        const sdp_GridderWtowerUVW* plan,
        sdp_Mem* subgrids,
        int w_plane,
        int subgrid_offset_u,
        int subgrid_offset_v,
        int subgrid_offset_w,
        double freq0_hz,
        double dfreq_hz,
        const sdp_Mem* uvws,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        const sdp_Mem* vis,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_MemViewCpu<const int, 1> start_chs_, end_chs_;
    sdp_mem_check_and_view(start_chs, &start_chs_, status);
    sdp_mem_check_and_view(end_chs, &end_chs_, status);
    if (sdp_mem_type(uvws) == SDP_MEM_DOUBLE &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE)
    {
        sdp_MemViewCpu<const complex<double>, 2> vis_;
        sdp_MemViewCpu<const double, 2> uvws_;
        sdp_mem_check_and_view(vis, &vis_, status);
        sdp_mem_check_and_view(uvws, &uvws_, status);
        grid<double, complex<double> >(
                plan, (complex<double>*) sdp_mem_data(subgrids),
                w_plane, subgrid_offset_u, subgrid_offset_v, subgrid_offset_w,
                freq0_hz, dfreq_hz, uvws_, start_chs_, end_chs_, vis_, status
        );
    }
    else if (sdp_mem_type(uvws) == SDP_MEM_FLOAT &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT)
    {
        sdp_MemViewCpu<const complex<float>, 2> vis_;
        sdp_MemViewCpu<const float, 2> uvws_;
        sdp_mem_check_and_view(vis, &vis_, status);
        sdp_mem_check_and_view(uvws, &uvws_, status);
        grid<float, complex<float> >(
                plan, (complex<float>*) sdp_mem_data(subgrids),
                w_plane, subgrid_offset_u, subgrid_offset_v, subgrid_offset_w,
                freq0_hz, dfreq_hz, uvws_, start_chs_, end_chs_, vis_, status
        );
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
    }
}


// Local function to call the GPU gridding kernel.
void grid_gpu(
        const sdp_GridderWtowerUVW* plan,
        sdp_Mem* subgrids,
        int w_plane,
        int subgrid_offset_u,
        int subgrid_offset_v,
        int subgrid_offset_w,
        double freq0_hz,
        double dfreq_hz,
        const sdp_Mem* uvws,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        const sdp_Mem* vis,
        sdp_Error* status
)
{
    if (*status) return;
    uint64_t num_threads[] = {256, 1, 1}, num_blocks[] = {1, 1, 1};
    const char* kernel_name = 0;
    int is_dbl = 0;
    const int64_t num_rows = sdp_mem_shape_dim(vis, 0);
    sdp_MemViewGpu<const double, 2> uvws_dbl;
    sdp_MemViewGpu<const float, 2> uvws_flt;
    sdp_MemViewGpu<const complex<double>, 2> vis_dbl;
    sdp_MemViewGpu<const complex<float>, 2> vis_flt;
    sdp_MemViewGpu<const int, 1> start_chs_, end_chs_;
    sdp_mem_check_and_view(start_chs, &start_chs_, status);
    sdp_mem_check_and_view(end_chs, &end_chs_, status);
    if (sdp_mem_type(uvws) == SDP_MEM_DOUBLE &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE)
    {
        is_dbl = 1;
        sdp_mem_check_and_view(uvws, &uvws_dbl, status);
        sdp_mem_check_and_view(vis, &vis_dbl, status);
        kernel_name = "sdp_gridder_wtower_grid<double, double>";
    }
    else if (sdp_mem_type(uvws) == SDP_MEM_FLOAT &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT)
    {
        is_dbl = 0;
        sdp_mem_check_and_view(uvws, &uvws_flt, status);
        sdp_mem_check_and_view(vis, &vis_flt, status);
        kernel_name = "sdp_gridder_wtower_grid<float, float>";
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
    }
    const void* arg[] = {
        sdp_mem_gpu_buffer(subgrids, status),
        &w_plane,
        &subgrid_offset_u,
        &subgrid_offset_v,
        &subgrid_offset_w,
        (const void*) &freq0_hz,
        (const void*) &dfreq_hz,
        is_dbl ? (const void*) &uvws_dbl : (const void*) &uvws_flt,
        (const void*) &start_chs_,
        (const void*) &end_chs_,
        sdp_mem_gpu_buffer_const(plan->uv_kernel_gpu, status),
        sdp_mem_gpu_buffer_const(plan->w_kernel_gpu, status),
        &plan->subgrid_size,
        &plan->vr_size,
        &plan->support,
        &plan->w_support,
        &plan->oversampling,
        &plan->w_oversampling,
        (const void*) &plan->theta,
        (const void*) &plan->w_step,
        is_dbl ? (const void*) &vis_dbl : (const void*) &vis_flt
    };
    num_blocks[0] = (num_rows + num_threads[0] - 1) / num_threads[0];
    sdp_launch_cuda_kernel(kernel_name,
            num_blocks, num_threads, 0, 0, arg, status
    );
}


// Local function to apply grid correction.
template<typename T>
void grid_corr(
        const sdp_GridderWtowerUVW* plan,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        const double* pswf_l,
        const double* pswf_m,
        sdp_Error* status
)
{
    if (*status) return;

    // Apply portion of shifted PSWF to facet.
    sdp_MemViewCpu<T, 2> facet_;
    sdp_mem_check_and_view(facet, &facet_, status);
    if (*status) return;
    const int num_l = (int) sdp_mem_shape_dim(facet, 0);
    const int num_m = (int) sdp_mem_shape_dim(facet, 1);
    const int image_size = plan->image_size;
    const int half_image_size = image_size / 2;
    const double theta = plan->theta;
    for (int il = 0; il < num_l; ++il)
    {
        const int pl = il + (half_image_size - num_l / 2);
        const double l_ = (
            (pl + facet_offset_l - half_image_size) * theta / image_size
        );
        for (int im = 0; im < num_m; ++im)
        {
            const int pm = im + (half_image_size - num_m / 2);
            const double m_ = (
                (pm + facet_offset_m - half_image_size) * theta / image_size
            );
            const double n_ = lm_to_n(l_, m_, plan->shear_u, plan->shear_v);
            double pswf_n_val = sdp_pswf_evaluate(
                    plan->pswf_n_func, n_ * 2.0 * plan->w_step
            );
            if (pswf_n_val == 0.0) pswf_n_val = 1.0;
            facet_(il, im) /= (T) (pswf_l[pl] * pswf_m[pm] * pswf_n_val);
        }
    }
}

} // End anonymous namespace for file-local functions.


sdp_GridderWtowerUVW* sdp_gridder_wtower_uvw_create(
        int image_size,
        int subgrid_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        int support,
        int oversampling,
        int w_support,
        int w_oversampling,
        sdp_Error* status
)
{
    if (subgrid_size % 2 != 0)
    {
        // If subgrid_size isn't even, the current FFT shift won't be correct.
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR(
                "Subgrid size must be even (value given was %d).", subgrid_size
        );
        return NULL;
    }
    sdp_GridderWtowerUVW* plan = (sdp_GridderWtowerUVW*) calloc(
            1, sizeof(sdp_GridderWtowerUVW)
    );
    plan->image_size = image_size;
    plan->subgrid_size = subgrid_size;
    plan->theta = theta;
    plan->w_step = w_step;
    plan->shear_u = shear_u;
    plan->shear_v = shear_v;
    plan->support = support;
    plan->vr_size = support; // vr_size is the same as support.
    plan->oversampling = oversampling;
    plan->w_support = w_support;
    plan->w_oversampling = w_oversampling;

    // Generate pswf (1D).
    const int64_t pswf_shape[] = {image_size};
    plan->pswf = sdp_mem_create(
            SDP_MEM_DOUBLE, SDP_MEM_CPU, 1, pswf_shape, status
    );
    sdp_generate_pswf(0, support * (M_PI / 2), plan->pswf, status);
    if (image_size % 2 == 0) ((double*) sdp_mem_data(plan->pswf))[0] = 1e-15;

    // Create a function handle to allow pswf_n to be evaluated on-the-fly.
    plan->pswf_n_func = sdp_pswf_create(0, w_support * (M_PI / 2));

    // Generate oversampled convolution kernel (uv_kernel).
    const int64_t uv_kernel_shape[] = {oversampling + 1, plan->vr_size};
    plan->uv_kernel = sdp_mem_create(
            SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, uv_kernel_shape, status
    );
    sdp_gridder_make_pswf_kernel(support, plan->uv_kernel, status);

    // Generate oversampled w convolution kernel (w_kernel).
    const int64_t w_kernel_shape[] = {w_oversampling + 1, w_support};
    plan->w_kernel = sdp_mem_create(
            SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, w_kernel_shape, status
    );
    sdp_gridder_make_pswf_kernel(w_support, plan->w_kernel, status);

    // Generate w_pattern.
    // This is the iDFT of a sole visibility at (0, 0, w) - our plan is roughly
    // to convolve in uvw space by a delta function to move the grid in w.
    const int64_t w_pattern_shape[] = {subgrid_size, subgrid_size};
    sdp_Mem* w_pattern = sdp_mem_create(
            SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, w_pattern_shape, status
    );
    plan->w_pattern = w_pattern;
    sdp_gridder_make_w_pattern(
            subgrid_size, theta, shear_u, shear_v, w_step, w_pattern, status
    );

    return plan;
}


void sdp_gridder_wtower_uvw_degrid(
        sdp_GridderWtowerUVW* plan,
        const sdp_Mem* subgrid_image,
        int subgrid_offset_u,
        int subgrid_offset_v,
        int subgrid_offset_w,
        double freq0_hz,
        double dfreq_hz,
        const sdp_Mem* uvws,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        sdp_Mem* vis,
        sdp_Error* status
)
{
    if (*status) return;
    const sdp_MemLocation loc = sdp_mem_location(vis);
    if (sdp_mem_location(subgrid_image) != loc ||
            sdp_mem_location(uvws) != loc ||
            sdp_mem_location(start_chs) != loc ||
            sdp_mem_location(end_chs) != loc)
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("All arrays must be co-located");
        return;
    }

    sdp_Mem* w_pattern_ptr = plan->w_pattern;
    if (loc == SDP_MEM_GPU)
    {
        // Copy internal arrays to GPU memory as required.
        if (!plan->w_pattern_gpu)
        {
            plan->w_kernel_gpu = sdp_mem_create_copy(
                    plan->w_kernel, loc, status
            );
            plan->uv_kernel_gpu = sdp_mem_create_copy(
                    plan->uv_kernel, loc, status
            );
            plan->w_pattern_gpu = sdp_mem_create_copy(
                    plan->w_pattern, loc, status
            );
        }
        w_pattern_ptr = plan->w_pattern_gpu;
    }

    // Determine w-range.
    double c_min[] = {0, 0, 0}, c_max[] = {0, 0, 0};
    sdp_gridder_uvw_bounds_all(
            uvws, freq0_hz, dfreq_hz, start_chs, end_chs, c_min, c_max, status
    );

    // Get subgrid at first w-plane.
    const double eta = 1e-5;
    const int first_w_plane = (int) floor(c_min[2] / plan->w_step - eta) -
            subgrid_offset_w;
    const int last_w_plane = (int) ceil(c_max[2] / plan->w_step + eta) -
            subgrid_offset_w + 1;

    // First w-plane we need to generate is (support / 2) below the first one
    // with visibilities.
    // Comment from Peter: That is actually a bit of a simplifying
    // assumption I made, and might well overshoot.
    // TODO Might need to check this properly.
    const int64_t subgrid_shape[] = {plan->subgrid_size, plan->subgrid_size};
    sdp_Mem* w_subgrid_image = sdp_mem_create(
            sdp_mem_type(vis), loc, 2, subgrid_shape, status
    );

    // Perform w_subgrid_image = subgrid_image /
    //             plan->w_pattern ** (first_w_plane - plan->w_support // 2)
    const int exponent = first_w_plane - plan->w_support / 2;
    sdp_gridder_scale_inv_array(
            w_subgrid_image, subgrid_image, w_pattern_ptr, exponent, status
    );

    // Create the FFT plan.
    sdp_Fft* fft = sdp_fft_create(
            w_subgrid_image, w_subgrid_image, 2, true, status
    );

    // Create sub-grid stack, with size (w_support, subgrid_size, subgrid_size).
    const int64_t num_elements_sg = plan->subgrid_size * plan->subgrid_size;
    const int64_t subgrids_shape[] = {
        plan->w_support, plan->subgrid_size, plan->subgrid_size
    };
    sdp_Mem* subgrids = sdp_mem_create(
            sdp_mem_type(vis), loc, 3, subgrids_shape, status
    );

    // Get a pointer to the last subgrid in the stack.
    const int64_t slice_offsets[] = {plan->w_support - 1, 0, 0};
    sdp_Mem* last_subgrid_ptr = sdp_mem_create_wrapper_for_slice(
            subgrids, slice_offsets, 2, subgrid_shape, status
    );

    // Fill sub-grid stack.
    for (int i = 0; i < plan->w_support; ++i)
    {
        // Perform subgrids[i] = fft(w_subgrid_image)
        // Copy w_subgrid_image to current sub-grid, and then do FFT in-place.
        const int64_t slice_offsets[] = {i, 0, 0};
        sdp_Mem* current_subgrid_ptr = sdp_mem_create_wrapper_for_slice(
                subgrids, slice_offsets, 2, subgrid_shape, status
        );
        sdp_mem_copy_contents(
                current_subgrid_ptr, w_subgrid_image,
                0, 0, num_elements_sg, status
        );
        sdp_fft_phase(current_subgrid_ptr, status);
        sdp_fft_exec(fft, current_subgrid_ptr, current_subgrid_ptr, status);
        sdp_fft_phase(current_subgrid_ptr, status);
        sdp_mem_free(current_subgrid_ptr);

        // Perform w_subgrid_image /= plan->w_pattern
        sdp_gridder_scale_inv_array(
                w_subgrid_image, w_subgrid_image, w_pattern_ptr, 1, status
        );
    }

    // Loop over w-planes.
    for (int w_plane = first_w_plane; w_plane <= last_w_plane; ++w_plane)
    {
        // Move to next w-plane.
        if (w_plane != first_w_plane)
        {
            // Shift subgrids, add new w-plane.
            // subgrids[:-1] = subgrids[1:]
            for (int i = 0; i < plan->w_support - 1; ++i)
                sdp_mem_copy_contents(
                        subgrids,
                        subgrids,
                        num_elements_sg * i,
                        num_elements_sg * (i + 1),
                        num_elements_sg,
                        status
                );

            // subgrids[-1] = fft(w_subgrid_image)
            // Copy w_subgrid_image to last subgrid, and then do FFT in-place.
            sdp_mem_copy_contents(
                    last_subgrid_ptr, w_subgrid_image,
                    0, 0, num_elements_sg, status
            );
            sdp_fft_phase(last_subgrid_ptr, status);
            sdp_fft_exec(fft, last_subgrid_ptr, last_subgrid_ptr, status);
            sdp_fft_phase(last_subgrid_ptr, status);

            // w_subgrid_image /= plan->w_pattern
            sdp_gridder_scale_inv_array(
                    w_subgrid_image, w_subgrid_image, w_pattern_ptr, 1, status
            );
        }

        if (loc == SDP_MEM_CPU)
        {
            degrid_cpu(
                    plan, subgrids, w_plane, subgrid_offset_u,
                    subgrid_offset_v, subgrid_offset_w, freq0_hz, dfreq_hz,
                    uvws, start_chs, end_chs, vis, status
            );
        }
        else if (loc == SDP_MEM_GPU)
        {
            degrid_gpu(
                    plan, subgrids, w_plane, subgrid_offset_u,
                    subgrid_offset_v, subgrid_offset_w, freq0_hz, dfreq_hz,
                    uvws, start_chs, end_chs, vis, status
            );
        }
    }

    sdp_mem_free(w_subgrid_image);
    sdp_mem_free(subgrids);
    sdp_mem_free(last_subgrid_ptr);
    sdp_fft_free(fft);
}


void sdp_gridder_wtower_uvw_degrid_correct(
        const sdp_GridderWtowerUVW* plan,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        sdp_Error* status
)
{
    if (*status) return;
    if (sdp_mem_num_dims(facet) < 2)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        return;
    }

    // Shift PSWF by facet offsets.
    sdp_Mem* pswf_l = sdp_mem_create_copy(plan->pswf, SDP_MEM_CPU, status);
    sdp_Mem* pswf_m = sdp_mem_create_copy(plan->pswf, SDP_MEM_CPU, status);
    if (*status) return;
    const double* pswf = (const double*) sdp_mem_data_const(plan->pswf);
    double* pswf_l_ = (double*) sdp_mem_data(pswf_l);
    double* pswf_m_ = (double*) sdp_mem_data(pswf_m);
    const int64_t pswf_size = sdp_mem_shape_dim(plan->pswf, 0);
    for (int64_t i = 0; i < pswf_size; ++i)
    {
        int64_t il = i + facet_offset_l;
        int64_t im = i + facet_offset_m;
        if (il >= pswf_size) il -= pswf_size;
        if (im >= pswf_size) im -= pswf_size;
        if (il < 0) il += pswf_size;
        if (im < 0) im += pswf_size;
        pswf_l_[i] = pswf[il];
        pswf_m_[i] = pswf[im];
    }

    // Apply grid correction for the appropriate data type.
    switch (sdp_mem_type(facet))
    {
    case SDP_MEM_DOUBLE:
        grid_corr<double>(
                plan, facet, facet_offset_l, facet_offset_m,
                pswf_l_, pswf_m_, status
        );
        break;
    case SDP_MEM_COMPLEX_DOUBLE:
        grid_corr<complex<double> >(
                plan, facet, facet_offset_l, facet_offset_m,
                pswf_l_, pswf_m_, status
        );
        break;
    case SDP_MEM_FLOAT:
        grid_corr<float>(
                plan, facet, facet_offset_l, facet_offset_m,
                pswf_l_, pswf_m_, status
        );
        break;
    case SDP_MEM_COMPLEX_FLOAT:
        grid_corr<complex<float> >(
                plan, facet, facet_offset_l, facet_offset_m,
                pswf_l_, pswf_m_, status
        );
        break;
    default:
        *status = SDP_ERR_DATA_TYPE;
        break;
    }
    sdp_mem_free(pswf_l);
    sdp_mem_free(pswf_m);
}


void sdp_gridder_wtower_uvw_grid(
        sdp_GridderWtowerUVW* plan,
        const sdp_Mem* vis,
        const sdp_Mem* uvws,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        double freq0_hz,
        double dfreq_hz,
        sdp_Mem* subgrid_image,
        int subgrid_offset_u,
        int subgrid_offset_v,
        int subgrid_offset_w,
        sdp_Error* status
)
{
    if (*status) return;
    const sdp_MemLocation loc = sdp_mem_location(vis);
    if (sdp_mem_location(subgrid_image) != loc ||
            sdp_mem_location(uvws) != loc ||
            sdp_mem_location(start_chs) != loc ||
            sdp_mem_location(end_chs) != loc)
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("All arrays must be co-located");
        return;
    }

    sdp_Mem* w_pattern_ptr = plan->w_pattern;
    if (loc == SDP_MEM_GPU)
    {
        // Copy internal arrays to GPU memory as required.
        if (!plan->w_pattern_gpu)
        {
            plan->w_kernel_gpu = sdp_mem_create_copy(
                    plan->w_kernel, loc, status
            );
            plan->uv_kernel_gpu = sdp_mem_create_copy(
                    plan->uv_kernel, loc, status
            );
            plan->w_pattern_gpu = sdp_mem_create_copy(
                    plan->w_pattern, loc, status
            );
        }
        w_pattern_ptr = plan->w_pattern_gpu;
    }

    // Determine w-range.
    double c_min[] = {0, 0, 0}, c_max[] = {0, 0, 0};
    sdp_gridder_uvw_bounds_all(
            uvws, freq0_hz, dfreq_hz, start_chs, end_chs, c_min, c_max, status
    );

    // Get subgrid at first w-plane.
    const double eta = 1e-5;
    const int first_w_plane = (int) floor(c_min[2] / plan->w_step - eta) -
            subgrid_offset_w;
    const int last_w_plane = (int) ceil(c_max[2] / plan->w_step + eta) -
            subgrid_offset_w + 1;

    // Create subgrid image and subgrids to accumulate on.
    const int64_t num_elements_sg = plan->subgrid_size * plan->subgrid_size;
    const int64_t subgrid_shape[] = {plan->subgrid_size, plan->subgrid_size};
    const int64_t subgrids_shape[] = {
        plan->w_support, plan->subgrid_size, plan->subgrid_size
    };
    sdp_Mem* w_subgrid_image = sdp_mem_create(
            SDP_MEM_COMPLEX_DOUBLE, loc, 2, subgrid_shape, status
    );
    sdp_Mem* subgrids = sdp_mem_create(
            sdp_mem_type(vis), loc, 3, subgrids_shape, status
    );
    sdp_mem_clear_contents(subgrids, status);
    sdp_mem_clear_contents(w_subgrid_image, status);

    // Create the iFFT plan and scratch buffer.
    sdp_Mem* fft_buffer = sdp_mem_create(
            sdp_mem_type(vis), loc, 2, subgrid_shape, status
    );
    sdp_Fft* fft = sdp_fft_create(fft_buffer, fft_buffer, 2, false, status);

    // Loop over w-planes.
    for (int w_plane = first_w_plane; w_plane <= last_w_plane; ++w_plane)
    {
        // Move to next w-plane.
        if (w_plane != first_w_plane)
        {
            // Accumulate zero-th subgrid, shift, clear upper subgrid.
            // w_subgrid_image /= plan->w_pattern
            sdp_gridder_scale_inv_array(
                    w_subgrid_image, w_subgrid_image, w_pattern_ptr, 1, status
            );

            // w_subgrid_image += ifft(subgrids[0])
            sdp_mem_copy_contents(
                    fft_buffer, subgrids, 0, 0, num_elements_sg, status
            );
            sdp_fft_phase(fft_buffer, status);
            sdp_fft_exec(fft, fft_buffer, fft_buffer, status);
            sdp_fft_phase(fft_buffer, status);
            sdp_gridder_accumulate_scaled_arrays(
                    w_subgrid_image, fft_buffer, 0, 0.0, status
            );

            // subgrids[:-1] = subgrids[1:]
            for (int i = 0; i < plan->w_support - 1; ++i)
                sdp_mem_copy_contents(
                        subgrids,
                        subgrids,
                        num_elements_sg * i,
                        num_elements_sg * (i + 1),
                        num_elements_sg,
                        status
                );

            // subgrids[-1] = 0
            sdp_mem_clear_portion(
                    subgrids,
                    num_elements_sg * (plan->w_support - 1),
                    num_elements_sg,
                    status
            );
        }

        if (loc == SDP_MEM_CPU)
        {
            grid_cpu(plan, subgrids, w_plane, subgrid_offset_u,
                    subgrid_offset_v, subgrid_offset_w, freq0_hz, dfreq_hz,
                    uvws, start_chs, end_chs, vis, status
            );
        }
        else if (loc == SDP_MEM_GPU)
        {
            grid_gpu(plan, subgrids, w_plane, subgrid_offset_u,
                    subgrid_offset_v, subgrid_offset_w, freq0_hz, dfreq_hz,
                    uvws, start_chs, end_chs, vis, status
            );
        }
    }

    // Accumulate remaining data from subgrids.
    for (int i = 0; i < plan->w_support; ++i)
    {
        // Perform w_subgrid_image /= plan->w_pattern
        sdp_gridder_scale_inv_array(
                w_subgrid_image, w_subgrid_image, w_pattern_ptr, 1, status
        );

        // Perform w_subgrid_image += ifft(subgrids[i])
        sdp_mem_copy_contents(
                fft_buffer,
                subgrids,
                0,
                num_elements_sg * i,
                num_elements_sg,
                status
        );
        sdp_fft_phase(fft_buffer, status);
        sdp_fft_exec(fft, fft_buffer, fft_buffer, status);
        sdp_fft_phase(fft_buffer, status);
        sdp_gridder_accumulate_scaled_arrays(
                w_subgrid_image, fft_buffer, 0, 0.0, status
        );
    }

    // Return updated subgrid image.
    // Perform subgrid_image += (
    //     w_subgrid_image
    //     * plan->w_pattern ** (last_w_plane + plan->w_support // 2 - 1)
    //     * plan->subgrid_size**2
    // )
    // We don't need to multiply by subgrid_size**2,
    // because the iFFT output is already scaled by this.
    int exponent = last_w_plane + plan->w_support / 2 - 1;
    sdp_gridder_accumulate_scaled_arrays(
            subgrid_image, w_subgrid_image, w_pattern_ptr, exponent, status
    );

    sdp_mem_free(w_subgrid_image);
    sdp_mem_free(subgrids);
    sdp_mem_free(fft_buffer);
    sdp_fft_free(fft);
}


void sdp_gridder_wtower_uvw_grid_correct(
        const sdp_GridderWtowerUVW* plan,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        sdp_Error* status
)
{
    // Grid correction and degrid correction are the same in the notebook.
    sdp_gridder_wtower_uvw_degrid_correct(
            plan, facet, facet_offset_l, facet_offset_m, status
    );
}


void sdp_gridder_wtower_uvw_free(sdp_GridderWtowerUVW* plan)
{
    sdp_pswf_free(plan->pswf_n_func);
    sdp_mem_free(plan->pswf);
    sdp_mem_free(plan->uv_kernel);
    sdp_mem_free(plan->uv_kernel_gpu);
    sdp_mem_free(plan->w_kernel);
    sdp_mem_free(plan->w_kernel_gpu);
    sdp_mem_free(plan->w_pattern);
    sdp_mem_free(plan->w_pattern_gpu);
    free(plan);
}
