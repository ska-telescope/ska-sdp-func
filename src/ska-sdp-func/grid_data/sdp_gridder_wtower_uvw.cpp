/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>
#include <cstdlib>

#include "ska-sdp-func/fourier_transforms/private_pswf.h"
#include "ska-sdp-func/fourier_transforms/sdp_fft.h"
#include "ska-sdp-func/fourier_transforms/sdp_pswf.h"
#include "ska-sdp-func/grid_data/sdp_gridder_clamp_channels.h"
#include "ska-sdp-func/grid_data/sdp_gridder_utils.h"
#include "ska-sdp-func/grid_data/sdp_gridder_wtower_uvw.h"
#include "ska-sdp-func/math/sdp_math_macros.h"
#include "ska-sdp-func/sdp_func_global.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"
#include "ska-sdp-func/utility/sdp_timer.h"

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
    int oversampling;
    int w_support;
    int w_oversampling;
    double tmr_grid_correct[2];
    double tmr_fft[2];
    double tmr_kernel[2];
    double tmr_total[2];
    int num_w_planes[2];
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
        const VIS_TYPE* RESTRICT subgrids,
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
        const sdp_Error* status
)
{
    if (*status) return;
    const double* RESTRICT uv_kernel =
            (const double*) sdp_mem_data_const(plan->uv_kernel);
    const double* RESTRICT w_kernel =
            (const double*) sdp_mem_data_const(plan->w_kernel);
    const int64_t num_uvw = uvws.shape[0];
    const int64_t subgrid_size = plan->subgrid_size;
    const int64_t subgrid_square = subgrid_size * subgrid_size;
    const int half_subgrid = plan->subgrid_size / 2;
    const double half_sup_m1 = (plan->support - 1) / 2.0;
    const int half_sup = plan->support / 2;
    const int oversample = plan->oversampling;
    const int w_oversample = plan->w_oversampling;
    const int support = plan->support;
    const int w_support = plan->w_support;
    const double theta = plan->theta;

    // Loop over rows. Each row contains visibilities for all channels.
    #pragma omp parallel for
    for (int64_t i_row = 0; i_row < num_uvw; ++i_row)
    {
        // Skip if there's no visibility to degrid.
        int64_t start_ch = start_chs(i_row), end_ch = end_chs(i_row);
        if (start_ch >= end_ch) continue;

        // Select only visibilities on this w-plane.
        const UVW_TYPE uvw[] = {uvws(i_row, 0), uvws(i_row, 1), uvws(i_row, 2)};
        const double min_w = (w_plane + subgrid_offset_w - 1) * plan->w_step;
        const double max_w = (w_plane + subgrid_offset_w) * plan->w_step;
        sdp_gridder_clamp_channels_inline(
                uvw[2], freq0_hz, dfreq_hz, &start_ch, &end_ch, min_w, max_w
        );
        if (start_ch >= end_ch) continue;

        // Scale + shift UVWs.
        const double s_uvw0 = freq0_hz / C_0, s_duvw = dfreq_hz / C_0;
        double uvw0[] = {uvw[0] * s_uvw0, uvw[1] * s_uvw0, uvw[2] * s_uvw0};
        double duvw[] = {uvw[0] * s_duvw, uvw[1] * s_duvw, uvw[2] * s_duvw};
        uvw0[0] -= subgrid_offset_u / theta;
        uvw0[1] -= subgrid_offset_v / theta;
        uvw0[2] -= ((subgrid_offset_w + w_plane) * plan->w_step);

        // Bounds check.
        const double u_min = floor(theta * (uvw0[0] + start_ch * duvw[0]));
        const double u_max = ceil(theta * (uvw0[0] + (end_ch - 1) * duvw[0]));
        const double v_min = floor(theta * (uvw0[1] + start_ch * duvw[1]));
        const double v_max = ceil(theta * (uvw0[1] + (end_ch - 1) * duvw[1]));
        if (u_min < -half_subgrid || u_max >= half_subgrid ||
                v_min < -half_subgrid || v_max >= half_subgrid)
        {
            continue;
        }

        // Loop over selected channels.
        for (int64_t c = start_ch; c < end_ch; c++)
        {
            const double u = uvw0[0] + c * duvw[0];
            const double v = uvw0[1] + c * duvw[1];
            const double w = uvw0[2] + c * duvw[2];

            // Determine top-left corner of grid region
            // centered approximately on visibility.
            const int iu0 = int(round(theta * u - half_sup_m1)) + half_subgrid;
            const int iv0 = int(round(theta * v - half_sup_m1)) + half_subgrid;
            const int iu_shift = iu0 + half_sup - half_subgrid;
            const int iv_shift = iv0 + half_sup - half_subgrid;

            // Determine which kernel to use.
            int u_off = int(round((u * theta - iu_shift + 1) * oversample));
            int v_off = int(round((v * theta - iv_shift + 1) * oversample));
            int w_off = int(round((w / plan->w_step + 1) * w_oversample));

            // Cater for the negative indexing which is allowed in Python!
            if (u_off < 0) u_off += oversample + 1;
            if (v_off < 0) v_off += oversample + 1;
            if (w_off < 0) w_off += w_oversample + 1;
            u_off *= support;
            v_off *= support;
            w_off *= w_support;

            // Degrid visibility.
            VIS_TYPE local_vis = (VIS_TYPE) 0;
            for (int iw = 0; iw < w_support; ++iw)
            {
                const double kern_w = w_kernel[w_off + iw];
                for (int iu = 0; iu < support; ++iu)
                {
                    const double kern_wu = kern_w * uv_kernel[u_off + iu];
                    const int ix_u = iu0 + iu;
                    for (int iv = 0; iv < support; ++iv)
                    {
                        const double kern_wuv = kern_wu * uv_kernel[v_off + iv];
                        const int ix_v = iv0 + iv;
                        const int64_t idx = (
                            iw * subgrid_square + ix_u * subgrid_size + ix_v
                        );
                        local_vis += ((VIS_TYPE) kern_wuv * subgrids[idx]);
                    }
                }
            }
            vis(i_row, c) += local_vis;
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
        VIS_TYPE* RESTRICT subgrids,
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
        const sdp_Error* status
)
{
    if (*status) return;
    const double* RESTRICT uv_kernel =
            (const double*) sdp_mem_data_const(plan->uv_kernel);
    const double* RESTRICT w_kernel =
            (const double*) sdp_mem_data_const(plan->w_kernel);
    const int64_t num_uvw = uvws.shape[0];
    const int64_t subgrid_size = plan->subgrid_size;
    const int64_t subgrid_square = subgrid_size * subgrid_size;
    const int half_subgrid = plan->subgrid_size / 2;
    const double half_sup_m1 = (plan->support - 1) / 2.0;
    const int half_sup = plan->support / 2;
    const int oversample = plan->oversampling;
    const int w_oversample = plan->w_oversampling;
    const int support = plan->support;
    const int w_support = plan->w_support;
    const double theta = plan->theta;

    // Loop over rows. Each row contains visibilities for all channels.
    for (int64_t i_row = 0; i_row < num_uvw; ++i_row)
    {
        // Skip if there's no visibility to grid.
        int64_t start_ch = start_chs(i_row), end_ch = end_chs(i_row);
        if (start_ch >= end_ch) continue;

        // Select only visibilities on this w-plane.
        const UVW_TYPE uvw[] = {uvws(i_row, 0), uvws(i_row, 1), uvws(i_row, 2)};
        const double min_w = (w_plane + subgrid_offset_w - 1) * plan->w_step;
        const double max_w = (w_plane + subgrid_offset_w) * plan->w_step;
        sdp_gridder_clamp_channels_inline(
                uvw[2], freq0_hz, dfreq_hz, &start_ch, &end_ch, min_w, max_w
        );
        if (start_ch >= end_ch) continue;

        // Scale + shift UVWs.
        const double s_uvw0 = freq0_hz / C_0, s_duvw = dfreq_hz / C_0;
        double uvw0[] = {uvw[0] * s_uvw0, uvw[1] * s_uvw0, uvw[2] * s_uvw0};
        double duvw[] = {uvw[0] * s_duvw, uvw[1] * s_duvw, uvw[2] * s_duvw};
        uvw0[0] -= subgrid_offset_u / theta;
        uvw0[1] -= subgrid_offset_v / theta;
        uvw0[2] -= ((subgrid_offset_w + w_plane) * plan->w_step);

        // Bounds check.
        const double u_min = floor(theta * (uvw0[0] + start_ch * duvw[0]));
        const double u_max = ceil(theta * (uvw0[0] + (end_ch - 1) * duvw[0]));
        const double v_min = floor(theta * (uvw0[1] + start_ch * duvw[1]));
        const double v_max = ceil(theta * (uvw0[1] + (end_ch - 1) * duvw[1]));
        if (u_min < -half_subgrid || u_max >= half_subgrid ||
                v_min < -half_subgrid || v_max >= half_subgrid)
        {
            continue;
        }

        // Loop over selected channels.
        for (int64_t c = start_ch; c < end_ch; c++)
        {
            const double u = uvw0[0] + c * duvw[0];
            const double v = uvw0[1] + c * duvw[1];
            const double w = uvw0[2] + c * duvw[2];

            // Determine top-left corner of grid region
            // centered approximately on visibility.
            const int iu0 = int(round(theta * u - half_sup_m1)) + half_subgrid;
            const int iv0 = int(round(theta * v - half_sup_m1)) + half_subgrid;
            const int iu_shift = iu0 + half_sup - half_subgrid;
            const int iv_shift = iv0 + half_sup - half_subgrid;

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
            u_off *= support;
            v_off *= support;
            w_off *= w_support;

            // Grid visibility.
            const VIS_TYPE local_vis = vis(i_row, c);
            for (int iw = 0; iw < w_support; ++iw)
            {
                const double kern_w = w_kernel[w_off + iw];
                for (int iu = 0; iu < support; ++iu)
                {
                    const double kern_wu = kern_w * uv_kernel[u_off + iu];
                    const int ix_u = iu0 + iu;
                    for (int iv = 0; iv < support; ++iv)
                    {
                        const double kern_wuv = kern_wu * uv_kernel[v_off + iv];
                        const int ix_v = iv0 + iv;
                        const int64_t idx = (
                            iw * subgrid_square + ix_u * subgrid_size + ix_v
                        );
                        subgrids[idx] += ((VIS_TYPE) kern_wuv * local_vis);
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


// Local function to apply grid correction for the PSWF.
template<typename T>
void grid_corr_pswf(
        int image_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        sdp_Pswf* pswf_lm,
        sdp_Pswf* pswf_n,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        sdp_Error* status
)
{
    if (*status) return;

    // Apply portion of shifted PSWF to facet.
    sdp_MemViewCpu<T, 2> facet_;
    sdp_MemViewCpu<const double, 1> pswf_;
    sdp_mem_check_and_view(facet, &facet_, status);
    sdp_mem_check_and_view(
            sdp_pswf_values(pswf_lm, SDP_MEM_CPU, status), &pswf_, status
    );
    if (*status) return;
    const int num_l = (int) sdp_mem_shape_dim(facet, 0);
    const int num_m = (int) sdp_mem_shape_dim(facet, 1);
    const double* pswf_n_coe = (const double*) sdp_mem_data_const(
            sdp_pswf_coeff(pswf_n, SDP_MEM_CPU, status)
    );
    const double pswf_n_c = sdp_pswf_par_c(pswf_n);
    #pragma omp parallel for
    for (int il = 0; il < num_l; ++il)
    {
        const int pl = il - num_l / 2 + facet_offset_l;
        const double l_ = pl * theta / image_size;
        const double pswf_l = pswf_(pl + image_size / 2);
        for (int im = 0; im < num_m; ++im)
        {
            const int pm = im - num_m / 2 + facet_offset_m;
            const double m_ = pm * theta / image_size;
            const double n_ = lm_to_n(l_, m_, shear_u, shear_v);
            const double pswf_m = pswf_(pm + image_size / 2);
            const double pswf_n_x = fabs(n_ * 2.0 * w_step);
            const double pswf_n = (pswf_n_x < 1.0) ?
                        sdp_pswf_aswfa(0, 0, pswf_n_c, pswf_n_coe, pswf_n_x) :
                        1.0;
            const double scale = 1.0 / (pswf_l * pswf_m * pswf_n);
            facet_(il, im) *= (T) scale;
        }
    }
}


void sdp_gridder_grid_correct_pswf(
        int image_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        int support,
        int w_support,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_Pswf* pswf_n = sdp_pswf_create(0, w_support * (M_PI / 2));
    sdp_Pswf* pswf_lm = sdp_pswf_create(0, support * (M_PI / 2));
    sdp_pswf_generate(pswf_lm, 0, image_size, 1, status);

    // Apply grid correction for the appropriate data type.
    const sdp_MemLocation loc = sdp_mem_location(facet);
    if (loc == SDP_MEM_CPU)
    {
        switch (sdp_mem_type(facet))
        {
        case SDP_MEM_DOUBLE:
            grid_corr_pswf<double>(
                    image_size, theta, w_step, shear_u, shear_v, pswf_lm,
                    pswf_n, facet, facet_offset_l, facet_offset_m, status
            );
            break;
        case SDP_MEM_COMPLEX_DOUBLE:
            grid_corr_pswf<complex<double> >(
                    image_size, theta, w_step, shear_u, shear_v, pswf_lm,
                    pswf_n, facet, facet_offset_l, facet_offset_m, status
            );
            break;
        case SDP_MEM_FLOAT:
            grid_corr_pswf<float>(
                    image_size, theta, w_step, shear_u, shear_v, pswf_lm,
                    pswf_n, facet, facet_offset_l, facet_offset_m, status
            );
            break;
        case SDP_MEM_COMPLEX_FLOAT:
            grid_corr_pswf<complex<float> >(
                    image_size, theta, w_step, shear_u, shear_v, pswf_lm,
                    pswf_n, facet, facet_offset_l, facet_offset_m, status
            );
            break;
        default:
            *status = SDP_ERR_DATA_TYPE;
            break;
        }
    }
    else if (loc == SDP_MEM_GPU)
    {
        uint64_t num_threads[] = {16, 16, 1}, num_blocks[] = {1, 1, 1};
        const char* kernel_name = 0;
        int is_cplx = 0, is_dbl = 0;
        const int num_l = sdp_mem_shape_dim(facet, 0);
        const int num_m = sdp_mem_shape_dim(facet, 1);
        sdp_MemViewGpu<complex<double>, 2> fct_cdbl;
        sdp_MemViewGpu<complex<float>, 2> fct_cflt;
        sdp_MemViewGpu<double, 2> fct_dbl;
        sdp_MemViewGpu<float, 2> fct_flt;
        sdp_MemViewGpu<const double, 1> pswf_;
        sdp_mem_check_and_view(
                sdp_pswf_values(pswf_lm, SDP_MEM_GPU, status), &pswf_, status
        );
        const sdp_Mem* pswf_n_coe = sdp_pswf_coeff(pswf_n, SDP_MEM_GPU, status);
        const double pswf_n_c = sdp_pswf_par_c(pswf_n);
        switch (sdp_mem_type(facet))
        {
        case SDP_MEM_FLOAT:
            is_cplx = 0;
            is_dbl = 0;
            sdp_mem_check_and_view(facet, &fct_flt, status);
            kernel_name = "sdp_gridder_grid_correct_pswf<float>";
            break;
        case SDP_MEM_DOUBLE:
            is_cplx = 0;
            is_dbl = 1;
            sdp_mem_check_and_view(facet, &fct_dbl, status);
            kernel_name = "sdp_gridder_grid_correct_pswf<double>";
            break;
        case SDP_MEM_COMPLEX_FLOAT:
            is_cplx = 1;
            is_dbl = 0;
            sdp_mem_check_and_view(facet, &fct_cflt, status);
            kernel_name = "sdp_gridder_grid_correct_pswf<complex<float> >";
            break;
        case SDP_MEM_COMPLEX_DOUBLE:
            is_cplx = 1;
            is_dbl = 1;
            sdp_mem_check_and_view(facet, &fct_cdbl, status);
            kernel_name = "sdp_gridder_grid_correct_pswf<complex<double> >";
            break;
        default:
            *status = SDP_ERR_DATA_TYPE;
            break;
        }
        const void* arg[] = {
            (const void*) &image_size,
            (const void*) &theta,
            (const void*) &w_step,
            (const void*) &shear_u,
            (const void*) &shear_v,
            (const void*) &pswf_,
            sdp_mem_gpu_buffer_const(pswf_n_coe, status),
            (const void*) &pswf_n_c,
            is_cplx ?
                (is_dbl ? (const void*) &fct_cdbl : (const void*) &fct_cflt) :
                (is_dbl ? (const void*) &fct_dbl : (const void*) &fct_flt),
            (const void*) &facet_offset_l,
            (const void*) &facet_offset_m
        };
        num_blocks[0] = (num_l + num_threads[0] - 1) / num_threads[0];
        num_blocks[1] = (num_m + num_threads[1] - 1) / num_threads[1];
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, 0, 0, arg, status
        );
    }
    sdp_pswf_free(pswf_n);
    sdp_pswf_free(pswf_lm);
}


// Local function to apply w-correction.
template<typename T>
void grid_corr_w_stack(
        int image_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        int w_offset,
        bool inverse,
        sdp_Error* status
)
{
    if (*status || w_offset == 0) return;
    sdp_MemViewCpu<T, 2> facet_;
    sdp_mem_check_and_view(facet, &facet_, status);
    if (*status) return;
    const int num_l = (int) sdp_mem_shape_dim(facet, 0);
    const int num_m = (int) sdp_mem_shape_dim(facet, 1);
    #pragma omp parallel for
    for (int il = 0; il < num_l; ++il)
    {
        const int pl = il - num_l / 2 + facet_offset_l;
        const double l_ = pl * theta / image_size;
        for (int im = 0; im < num_m; ++im)
        {
            const int pm = im - num_m / 2 + facet_offset_m;
            const double m_ = pm * theta / image_size;
            const double n_ = lm_to_n(l_, m_, shear_u, shear_v);
            const double phase = 2.0 * M_PI * w_step * n_;
            complex<double> w = complex<double>(cos(phase), sin(phase));
            w = std::pow(w, w_offset);
            w = !inverse ? 1.0 / w : w;
            facet_(il, im) *= (T) w;
        }
    }
}


void sdp_gridder_grid_correct_w_stack(
        int image_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        int w_offset,
        bool inverse,
        sdp_Error* status
)
{
    if (*status || w_offset == 0) return;
    const sdp_MemLocation loc = sdp_mem_location(facet);
    if (loc == SDP_MEM_CPU)
    {
        if (sdp_mem_type(facet) == SDP_MEM_COMPLEX_DOUBLE)
        {
            grid_corr_w_stack<complex<double> >(
                    image_size, theta, w_step, shear_u, shear_v, facet,
                    facet_offset_l, facet_offset_m, w_offset, inverse, status
            );
        }
        else if (sdp_mem_type(facet) == SDP_MEM_COMPLEX_FLOAT)
        {
            grid_corr_w_stack<complex<float> >(
                    image_size, theta, w_step, shear_u, shear_v, facet,
                    facet_offset_l, facet_offset_m, w_offset, inverse, status
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
        }
    }
    else if (loc == SDP_MEM_GPU)
    {
        uint64_t num_threads[] = {16, 16, 1}, num_blocks[] = {1, 1, 1};
        const char* kernel_name = 0;
        int is_dbl = 0, inverse_int = (int) inverse;
        const int num_l = sdp_mem_shape_dim(facet, 0);
        const int num_m = sdp_mem_shape_dim(facet, 1);
        sdp_MemViewGpu<complex<double>, 2> facet_dbl;
        sdp_MemViewGpu<complex<float>, 2> facet_flt;
        if (sdp_mem_type(facet) == SDP_MEM_COMPLEX_DOUBLE)
        {
            is_dbl = 1;
            sdp_mem_check_and_view(facet, &facet_dbl, status);
            kernel_name = "sdp_gridder_grid_correct_w_stack<complex<double> >";
        }
        else if (sdp_mem_type(facet) == SDP_MEM_COMPLEX_FLOAT)
        {
            is_dbl = 0;
            sdp_mem_check_and_view(facet, &facet_flt, status);
            kernel_name = "sdp_gridder_grid_correct_w_stack<complex<float> >";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
        }
        const void* arg[] = {
            (const void*) &image_size,
            (const void*) &theta,
            (const void*) &w_step,
            (const void*) &shear_u,
            (const void*) &shear_v,
            is_dbl ? (const void*) &facet_dbl : (const void*) &facet_flt,
            (const void*) &facet_offset_l,
            (const void*) &facet_offset_m,
            (const void*) &w_offset,
            (const void*) &inverse_int
        };
        num_blocks[0] = (num_l + num_threads[0] - 1) / num_threads[0];
        num_blocks[1] = (num_m + num_threads[1] - 1) / num_threads[1];
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, 0, 0, arg, status
        );
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
    plan->oversampling = oversampling;
    plan->w_support = w_support;
    plan->w_oversampling = w_oversampling;

    // Generate oversampled convolution kernel (uv_kernel).
    const int64_t uv_kernel_shape[] = {oversampling + 1, plan->support};
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
        SDP_LOG_ERROR("All arrays must be in the same memory space");
        return;
    }

    // Set up timers.
    const sdp_TimerType tmr_type = (
        loc == SDP_MEM_CPU ? SDP_TIMER_NATIVE : SDP_TIMER_CUDA
    );
    sdp_Timer* tmr_degrid = sdp_timer_create(tmr_type);
    sdp_Timer* tmr_total = sdp_timer_create(tmr_type);
    sdp_timer_resume(tmr_total);

    // Copy internal arrays to GPU memory as required.
    sdp_Mem* w_pattern_ptr = plan->w_pattern;
    if (loc == SDP_MEM_GPU)
    {
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
    const int num_w_planes = 1 + last_w_plane - first_w_plane;
    for (int w_plane = first_w_plane; w_plane <= last_w_plane; ++w_plane)
    {
        if (*status) break;
        SDP_LOG_DEBUG("Degridding w-plane %d (%d/%d)",
                w_plane, 1 + w_plane - first_w_plane, num_w_planes
        );

        // Move to next w-plane.
        if (w_plane != first_w_plane)
        {
            // Shift subgrids, add new w-plane.
            // subgrids[:-1] = subgrids[1:]
            for (int i = 0; i < plan->w_support - 1; ++i)
            {
                sdp_mem_copy_contents(
                        subgrids,
                        subgrids,
                        num_elements_sg * i,
                        num_elements_sg * (i + 1),
                        num_elements_sg,
                        status
                );
            }

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

        sdp_timer_resume(tmr_degrid);
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
        sdp_timer_pause(tmr_degrid);
    }

    // Update timers.
    #pragma omp critical
    {
        plan->tmr_kernel[0] += sdp_timer_elapsed(tmr_degrid);
        plan->tmr_fft[0] += sdp_fft_elapsed_time(fft, SDP_FFT_TMR_EXEC);
        plan->tmr_total[0] += sdp_timer_elapsed(tmr_total);
        plan->num_w_planes[0] += num_w_planes;
    }

    sdp_mem_free(w_subgrid_image);
    sdp_mem_free(subgrids);
    sdp_mem_free(last_subgrid_ptr);
    sdp_fft_free(fft);
    sdp_timer_free(tmr_degrid);
    sdp_timer_free(tmr_total);
}


void sdp_gridder_wtower_uvw_degrid_correct(
        sdp_GridderWtowerUVW* plan,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        int w_offset,
        sdp_Error* status
)
{
    const sdp_TimerType tmr_type = (sdp_mem_location(facet) == SDP_MEM_CPU ?
                SDP_TIMER_NATIVE : SDP_TIMER_CUDA
    );
    sdp_Timer* tmr = sdp_timer_create(tmr_type);
    sdp_timer_resume(tmr);
    sdp_gridder_grid_correct_pswf(plan->image_size, plan->theta, plan->w_step,
            plan->shear_u, plan->shear_v, plan->support, plan->w_support,
            facet, facet_offset_l, facet_offset_m, status
    );
    if (sdp_mem_is_complex(facet))
    {
        sdp_gridder_grid_correct_w_stack(plan->image_size, plan->theta,
                plan->w_step, plan->shear_u, plan->shear_v,
                facet, facet_offset_l, facet_offset_m, w_offset, false, status
        );
    }
    #pragma omp critical
    plan->tmr_grid_correct[0] += sdp_timer_elapsed(tmr);
    sdp_timer_free(tmr);
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
        SDP_LOG_ERROR("All arrays must be in the same memory space");
        return;
    }

    // Set up timers.
    const sdp_TimerType tmr_type = (
        loc == SDP_MEM_CPU ? SDP_TIMER_NATIVE : SDP_TIMER_CUDA
    );
    sdp_Timer* tmr_grid = sdp_timer_create(tmr_type);
    sdp_Timer* tmr_total = sdp_timer_create(tmr_type);
    sdp_timer_resume(tmr_total);

    // Copy internal arrays to GPU memory as required.
    sdp_Mem* w_pattern_ptr = plan->w_pattern;
    if (loc == SDP_MEM_GPU)
    {
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
    const int num_w_planes = 1 + last_w_plane - first_w_plane;
    for (int w_plane = first_w_plane; w_plane <= last_w_plane; ++w_plane)
    {
        if (*status) break;
        SDP_LOG_DEBUG("Gridding w-plane %d (%d/%d)",
                w_plane, 1 + w_plane - first_w_plane, num_w_planes
        );

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
            {
                sdp_mem_copy_contents(
                        subgrids,
                        subgrids,
                        num_elements_sg * i,
                        num_elements_sg * (i + 1),
                        num_elements_sg,
                        status
                );
            }

            // subgrids[-1] = 0
            sdp_mem_clear_portion(
                    subgrids,
                    num_elements_sg * (plan->w_support - 1),
                    num_elements_sg,
                    status
            );
        }

        sdp_timer_resume(tmr_grid);
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
        sdp_timer_pause(tmr_grid);
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

    // Update timers.
    #pragma omp critical
    {
        plan->tmr_kernel[1] += sdp_timer_elapsed(tmr_grid);
        plan->tmr_fft[1] += sdp_fft_elapsed_time(fft, SDP_FFT_TMR_EXEC);
        plan->tmr_total[1] += sdp_timer_elapsed(tmr_total);
        plan->num_w_planes[1] += num_w_planes;
    }

    sdp_mem_free(w_subgrid_image);
    sdp_mem_free(subgrids);
    sdp_mem_free(fft_buffer);
    sdp_fft_free(fft);
    sdp_timer_free(tmr_grid);
    sdp_timer_free(tmr_total);
}


void sdp_gridder_wtower_uvw_grid_correct(
        sdp_GridderWtowerUVW* plan,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        int w_offset,
        sdp_Error* status
)
{
    const sdp_TimerType tmr_type = (sdp_mem_location(facet) == SDP_MEM_CPU ?
                SDP_TIMER_NATIVE : SDP_TIMER_CUDA
    );
    sdp_Timer* tmr = sdp_timer_create(tmr_type);
    sdp_timer_resume(tmr);
    sdp_gridder_grid_correct_pswf(plan->image_size, plan->theta, plan->w_step,
            plan->shear_u, plan->shear_v, plan->support, plan->w_support,
            facet, facet_offset_l, facet_offset_m, status
    );
    if (sdp_mem_is_complex(facet))
    {
        sdp_gridder_grid_correct_w_stack(plan->image_size, plan->theta,
                plan->w_step, plan->shear_u, plan->shear_v,
                facet, facet_offset_l, facet_offset_m, w_offset, true, status
        );
    }
    #pragma omp critical
    plan->tmr_grid_correct[1] += sdp_timer_elapsed(tmr);
    sdp_timer_free(tmr);
}


void sdp_gridder_wtower_uvw_free(sdp_GridderWtowerUVW* plan)
{
    sdp_mem_free(plan->uv_kernel);
    sdp_mem_free(plan->uv_kernel_gpu);
    sdp_mem_free(plan->w_kernel);
    sdp_mem_free(plan->w_kernel_gpu);
    sdp_mem_free(plan->w_pattern);
    sdp_mem_free(plan->w_pattern_gpu);
    free(plan);
}


double sdp_gridder_wtower_uvw_elapsed_time(
        const sdp_GridderWtowerUVW* plan,
        sdp_GridderWtowerUVWTimer timer,
        int gridding
)
{
    switch (timer)
    {
    case SDP_WTOWER_TMR_GRID_CORRECT:
        return plan->tmr_grid_correct[gridding];
    case SDP_WTOWER_TMR_FFT:
        return plan->tmr_fft[gridding];
    case SDP_WTOWER_TMR_KERNEL:
        return plan->tmr_kernel[gridding];
    case SDP_WTOWER_TMR_TOTAL:
        return plan->tmr_total[gridding];
    default:
        return 0.0;
    }
}


double sdp_gridder_wtower_uvw_num_w_planes(
        const sdp_GridderWtowerUVW* plan,
        int gridding
)
{
    return (gridding >= 0 && gridding < 2) ? plan->num_w_planes[gridding] : 0;
}
