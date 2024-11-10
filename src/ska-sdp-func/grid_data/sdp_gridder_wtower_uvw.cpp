/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>
#include <cstdlib>

#if defined(AVX2) || defined(AVX512)
#include <immintrin.h>
#endif // AVX2 || AVX512

#include "ska-sdp-func/fourier_transforms/sdp_fft.h"
#include "ska-sdp-func/grid_data/sdp_gridder_clamp_channels.h"
#include "ska-sdp-func/grid_data/sdp_gridder_grid_correct.h"
#include "ska-sdp-func/grid_data/sdp_gridder_utils.h"
#include "ska-sdp-func/grid_data/sdp_gridder_wtower_uvw.h"
#include "ska-sdp-func/math/sdp_math_macros.h"
#include "ska-sdp-func/sdp_func_global.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"
#include "ska-sdp-func/utility/sdp_timer.h"

using std::complex;

struct sdp_GridderWtowerUVW {
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
        int num_w_planes[2];
        sdp_Mem *uv_kernel;
        sdp_Mem *uv_kernel_gpu;
        sdp_Mem *w_kernel;
        sdp_Mem *w_kernel_gpu;
        sdp_Mem *w_pattern;
        sdp_Mem *w_pattern_gpu;
};

// Begin anonymous namespace for file-local functions.
namespace {

// Local function to do the degridding.
template <typename SUBGRID_TYPE, typename UVW_TYPE, typename VIS_TYPE>
void degrid(const sdp_GridderWtowerUVW *plan, const sdp_Mem *subgrids,
            int w_plane, int subgrid_offset_u, int subgrid_offset_v,
            int subgrid_offset_w, double freq0_hz, double dfreq_hz,
            int64_t start_row, int64_t end_row, const sdp_Mem *uvws,
            const sdp_MemViewCpu<const int, 1> &start_chs,
            const sdp_MemViewCpu<const int, 1> &end_chs, sdp_Mem *vis,
            sdp_Error *status) {
    if (*status)
        return;
    sdp_MemViewCpu<const SUBGRID_TYPE, 3> subgrids_;
    sdp_MemViewCpu<const UVW_TYPE, 2> uvws_;
    sdp_MemViewCpu<VIS_TYPE, 2> vis_;
    sdp_mem_check_and_view(subgrids, &subgrids_, status);
    sdp_mem_check_and_view(uvws, &uvws_, status);
    sdp_mem_check_and_view(vis, &vis_, status);
    const double *RESTRICT uv_kernel =
        (const double *)sdp_mem_data_const(plan->uv_kernel);
    const double *RESTRICT w_kernel =
        (const double *)sdp_mem_data_const(plan->w_kernel);
    const int half_subgrid = plan->subgrid_size / 2;
    const int oversample = plan->oversampling;
    const int w_oversample = plan->w_oversampling;
    const int support = plan->support;
    const int w_support = plan->w_support;
    const double theta = plan->theta;
    const double w_step = plan->w_step;
    const double theta_ov = theta * oversample;
    const double w_step_ov = 1.0 / w_step * w_oversample;
    const int half_sg_size_ov = (half_subgrid - support / 2 + 1) * oversample;

// Loop over rows. Each row contains visibilities for all channels.
#pragma omp parallel for schedule(dynamic, 500)
    for (int64_t i_row = start_row; i_row < end_row; ++i_row) {
        // Skip if there's no visibility to degrid.
        int64_t start_ch = start_chs(i_row), end_ch = end_chs(i_row);
        if (start_ch >= end_ch)
            continue;

        // Select only visibilities on this w-plane.
        const UVW_TYPE uvw[] = {uvws_(i_row, 0), uvws_(i_row, 1),
                                uvws_(i_row, 2)};
        const double min_w = (w_plane + subgrid_offset_w - 1) * w_step;
        const double max_w = (w_plane + subgrid_offset_w) * w_step;
        sdp_gridder_clamp_channels_inline(uvw[2], freq0_hz, dfreq_hz, &start_ch,
                                          &end_ch, min_w, max_w);
        if (start_ch >= end_ch)
            continue;

        // Scale + shift UVWs.
        const double s_uvw0 = freq0_hz / C_0, s_duvw = dfreq_hz / C_0;
        double uvw0[] = {uvw[0] * s_uvw0, uvw[1] * s_uvw0, uvw[2] * s_uvw0};
        double duvw[] = {uvw[0] * s_duvw, uvw[1] * s_duvw, uvw[2] * s_duvw};
        uvw0[0] -= subgrid_offset_u / theta;
        uvw0[1] -= subgrid_offset_v / theta;
        uvw0[2] -= ((subgrid_offset_w + w_plane - 1) * w_step);

        // Bounds check.
        const double u_min = floor(theta * (uvw0[0] + start_ch * duvw[0]));
        const double u_max = ceil(theta * (uvw0[0] + (end_ch - 1) * duvw[0]));
        const double v_min = floor(theta * (uvw0[1] + start_ch * duvw[1]));
        const double v_max = ceil(theta * (uvw0[1] + (end_ch - 1) * duvw[1]));
        if (u_min < -half_subgrid || u_max >= half_subgrid ||
            v_min < -half_subgrid || v_max >= half_subgrid) {
            continue;
        }

        // Loop over selected channels.
        for (int64_t c = start_ch; c < end_ch; c++) {
            const double u = uvw0[0] + c * duvw[0];
            const double v = uvw0[1] + c * duvw[1];
            const double w = uvw0[2] + c * duvw[2];

            // Determine top-left corner of grid region centred approximately
            // on visibility in oversampled coordinates (this is subgrid-local,
            // so should be safe for overflows).
            const int iu0_ov = int(round(u * theta_ov)) + half_sg_size_ov;
            const int iv0_ov = int(round(v * theta_ov)) + half_sg_size_ov;
            const int iw0_ov = int(round(w * w_step_ov));
            const int iu0 = iu0_ov / oversample;
            const int iv0 = iv0_ov / oversample;

            // Determine which kernel to use.
            const int u_off = (iu0_ov % oversample) * support;
            const int v_off = (iv0_ov % oversample) * support;
            const int w_off = (iw0_ov % w_oversample) * w_support;

            // Degrid visibility.
            SUBGRID_TYPE local_vis = (SUBGRID_TYPE)0;
            for (int iw = 0; iw < w_support; ++iw) {
                SUBGRID_TYPE local_vis_u = (SUBGRID_TYPE)0;
#pragma GCC ivdep
#pragma GCC unroll(8)
                for (int iu = 0; iu < support; ++iu) {
                    const int ix_u = iu0 + iu;
                    SUBGRID_TYPE local_vis_v = (SUBGRID_TYPE)0;
#pragma GCC ivdep
#pragma GCC unroll(8)
                    for (int iv = 0; iv < support; ++iv) {
                        const int ix_v = iv0 + iv;
                        local_vis_v += ((SUBGRID_TYPE)uv_kernel[v_off + iv] *
                                        subgrids_(iw, ix_u, ix_v));
                    }
                    local_vis_u +=
                        ((SUBGRID_TYPE)uv_kernel[u_off + iu] * local_vis_v);
                }
                local_vis += (SUBGRID_TYPE)w_kernel[w_off + iw] * local_vis_u;
            }
            vis_(i_row, c) += (VIS_TYPE)local_vis;
        }
    }
}

// Local function to call the CPU degridding kernel.
void degrid_cpu(const sdp_GridderWtowerUVW *plan, const sdp_Mem *subgrids,
                int w_plane, int subgrid_offset_u, int subgrid_offset_v,
                int subgrid_offset_w, double freq0_hz, double dfreq_hz,
                int64_t start_row, int64_t end_row, const sdp_Mem *uvws,
                const sdp_Mem *start_chs, const sdp_Mem *end_chs, sdp_Mem *vis,
                sdp_Error *status) {
    if (*status)
        return;
    sdp_MemViewCpu<const int, 1> start_chs_, end_chs_;
    sdp_mem_check_and_view(start_chs, &start_chs_, status);
    sdp_mem_check_and_view(end_chs, &end_chs_, status);
    if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_DOUBLE &&
        sdp_mem_type(uvws) == SDP_MEM_DOUBLE &&
        sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE) {
        degrid<complex<double>, double, complex<double>>(
            plan, subgrids, w_plane, subgrid_offset_u, subgrid_offset_v,
            subgrid_offset_w, freq0_hz, dfreq_hz, start_row, end_row, uvws,
            start_chs_, end_chs_, vis, status);
    } else if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_FLOAT &&
               sdp_mem_type(uvws) == SDP_MEM_DOUBLE &&
               sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT) {
        degrid<complex<float>, double, complex<float>>(
            plan, subgrids, w_plane, subgrid_offset_u, subgrid_offset_v,
            subgrid_offset_w, freq0_hz, dfreq_hz, start_row, end_row, uvws,
            start_chs_, end_chs_, vis, status);
    } else if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_FLOAT &&
               sdp_mem_type(uvws) == SDP_MEM_FLOAT &&
               sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT) {
        degrid<complex<float>, float, complex<float>>(
            plan, subgrids, w_plane, subgrid_offset_u, subgrid_offset_v,
            subgrid_offset_w, freq0_hz, dfreq_hz, start_row, end_row, uvws,
            start_chs_, end_chs_, vis, status);
    } else {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data types: "
                      "subgrids has type %s; uvws has type %s; vis has type %s",
                      sdp_mem_type_name(sdp_mem_type(subgrids)),
                      sdp_mem_type_name(sdp_mem_type(uvws)),
                      sdp_mem_type_name(sdp_mem_type(vis)));
    }
}

// Local function to call the GPU degridding kernel.
void degrid_gpu(const sdp_GridderWtowerUVW *plan, const sdp_Mem *subgrids,
                int w_plane, int subgrid_offset_u, int subgrid_offset_v,
                int subgrid_offset_w, double freq0_hz, double dfreq_hz,
                int64_t start_row, int64_t end_row, const sdp_Mem *uvws,
                const sdp_Mem *start_chs, const sdp_Mem *end_chs, sdp_Mem *vis,
                sdp_Error *status) {
    if (*status)
        return;
    uint64_t num_threads[] = {256, 1, 1}, num_blocks[] = {1, 1, 1};
    const char *kernel_name = 0;
    int is_dbl_uvw = 0;
    int is_dbl_vis = 0;
    const int64_t num_rows = sdp_mem_shape_dim(vis, 0);
    sdp_MemViewGpu<const double, 2> uvws_dbl;
    sdp_MemViewGpu<const float, 2> uvws_flt;
    sdp_MemViewGpu<complex<double>, 2> vis_dbl;
    sdp_MemViewGpu<complex<float>, 2> vis_flt;
    sdp_MemViewGpu<const int, 1> start_chs_, end_chs_;
    sdp_mem_check_and_view(start_chs, &start_chs_, status);
    sdp_mem_check_and_view(end_chs, &end_chs_, status);
    if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_DOUBLE &&
        sdp_mem_type(uvws) == SDP_MEM_DOUBLE &&
        sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE) {
        is_dbl_uvw = 1;
        is_dbl_vis = 1;
        sdp_mem_check_and_view(uvws, &uvws_dbl, status);
        sdp_mem_check_and_view(vis, &vis_dbl, status);
        kernel_name = "sdp_gridder_wtower_degrid"
                      "<complex<double>, double, complex<double> >";
    } else if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_FLOAT &&
               sdp_mem_type(uvws) == SDP_MEM_DOUBLE &&
               sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT) {
        is_dbl_uvw = 1;
        is_dbl_vis = 0;
        sdp_mem_check_and_view(uvws, &uvws_dbl, status);
        sdp_mem_check_and_view(vis, &vis_flt, status);
        kernel_name = "sdp_gridder_wtower_degrid"
                      "<complex<float>, double, complex<float> >";
    } else if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_FLOAT &&
               sdp_mem_type(uvws) == SDP_MEM_FLOAT &&
               sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT) {
        is_dbl_uvw = 0;
        is_dbl_vis = 0;
        sdp_mem_check_and_view(uvws, &uvws_flt, status);
        sdp_mem_check_and_view(vis, &vis_flt, status);
        kernel_name = "sdp_gridder_wtower_degrid"
                      "<complex<float>, float, complex<float> >";
    } else {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data types: "
                      "subgrids has type %s; uvws has type %s; vis has type %s",
                      sdp_mem_type_name(sdp_mem_type(subgrids)),
                      sdp_mem_type_name(sdp_mem_type(uvws)),
                      sdp_mem_type_name(sdp_mem_type(vis)));
    }
    const void *arg[] = {
        sdp_mem_gpu_buffer_const(subgrids, status),
        &w_plane,
        &subgrid_offset_u,
        &subgrid_offset_v,
        &subgrid_offset_w,
        (const void *)&freq0_hz,
        (const void *)&dfreq_hz,
        (const void *)&start_row,
        (const void *)&end_row,
        is_dbl_uvw ? (const void *)&uvws_dbl : (const void *)&uvws_flt,
        (const void *)&start_chs_,
        (const void *)&end_chs_,
        sdp_mem_gpu_buffer_const(plan->uv_kernel_gpu, status),
        sdp_mem_gpu_buffer_const(plan->w_kernel_gpu, status),
        &plan->subgrid_size,
        &plan->support,
        &plan->w_support,
        &plan->oversampling,
        &plan->w_oversampling,
        (const void *)&plan->theta,
        (const void *)&plan->w_step,
        is_dbl_vis ? (const void *)&vis_dbl : (const void *)&vis_flt};
    num_blocks[0] = (num_rows + num_threads[0] - 1) / num_threads[0];
    sdp_launch_cuda_kernel(kernel_name, num_blocks, num_threads, 0, 0, arg,
                           status);
}

// Masked version
template <typename SUBGRID_TYPE, typename UVW_TYPE, typename VIS_TYPE>
void grid_masked(const sdp_GridderWtowerUVW *plan, sdp_Mem *subgrids,
                 int w_plane, int subgrid_offset_u, int subgrid_offset_v,
                 int subgrid_offset_w, double freq0_hz, double dfreq_hz,
                 int64_t start_row, int64_t end_row, const sdp_Mem *uvws,
                 const sdp_MemViewCpu<const int, 1> &start_chs,
                 const sdp_MemViewCpu<const int, 1> &end_chs,
                 const sdp_Mem *vis, sdp_Error *status) {
    if (*status)
        return;
    sdp_MemViewCpu<SUBGRID_TYPE, 3> subgrids_;
    sdp_MemViewCpu<const UVW_TYPE, 2> uvws_;
    sdp_MemViewCpu<const VIS_TYPE, 2> vis_;
    sdp_mem_check_and_view(subgrids, &subgrids_, status);
    sdp_mem_check_and_view(uvws, &uvws_, status);
    sdp_mem_check_and_view(vis, &vis_, status);
    const double *RESTRICT uv_kernel =
        (const double *)sdp_mem_data_const(plan->uv_kernel);
    const double *RESTRICT w_kernel =
        (const double *)sdp_mem_data_const(plan->w_kernel);
    const int half_subgrid = plan->subgrid_size / 2;
    const int oversample = plan->oversampling;
    const int w_oversample = plan->w_oversampling;
    const int support = plan->support;
    const int w_support = plan->w_support;
    const double theta = plan->theta;
    const double w_step = plan->w_step;
    const double theta_ov = theta * oversample;
    const double w_step_ov = 1.0 / w_step * w_oversample;
    const int half_sg_size_ov = (half_subgrid - support / 2 + 1) * oversample;
    const double s_uvw0 = freq0_hz / C_0;
    const double s_duvw = dfreq_hz / C_0;

    struct ValidVis
    {
        int64_t row_idx;
        int64_t start_ch;
        int64_t end_ch;
        int64_t vis_offset; // Offset into packed visibility array
    };

    // Loop over rows. Each row contains visibilities for all channels.
    for (int64_t i_row = start_row; i_row < end_row; ++i_row)
    {
            int64_t row_idx;
            int64_t start_ch;
            int64_t end_ch;
            int64_t vis_offset; // Offset into packed visibility array
    };
    std::vector<ValidVis> valid_vis_info;
    int64_t total_valid_vis_count = 0;

    for (int64_t i_row = 0; i_row < num_uvw; ++i_row) {
        // Select valid visibilities
        int64_t start_ch = start_chs(i_row);
        int64_t end_ch = end_chs(i_row);
        if (start_ch >= end_ch)
            continue;

        // Select only visibilities on this w-plane.
        const UVW_TYPE uvw[] = {uvws_(i_row, 0), uvws_(i_row, 1),
                                uvws_(i_row, 2)};
        const double min_w = (w_plane + subgrid_offset_w - 1) * w_step;
        const double max_w = (w_plane + subgrid_offset_w) * w_step;
        sdp_gridder_clamp_channels_inline(uvw[2], freq0_hz, dfreq_hz, &start_ch,
                                          &end_ch, min_w, max_w);
        if (start_ch >= end_ch)
            continue;

        // Scale + shift UVWs.
        double uvw0[] = {uvw[0] * s_uvw0 - subgrid_offset_u / theta,
                         uvw[1] * s_uvw0 - subgrid_offset_v / theta,
                         uvw[2] * s_uvw0 -
                             ((subgrid_offset_w + w_plane - 1) * w_step)};
        double duvw[] = {uvw[0] * s_duvw, uvw[1] * s_duvw, uvw[2] * s_duvw};

        // Bounds check.
        const double u_min = floor(theta * (uvw0[0] + start_ch * duvw[0]));
        const double u_max = ceil(theta * (uvw0[0] + (end_ch - 1) * duvw[0]));
        const double v_min = floor(theta * (uvw0[1] + start_ch * duvw[1]));
        const double v_max = ceil(theta * (uvw0[1] + (end_ch - 1) * duvw[1]));
        if (u_min < -half_subgrid || u_max >= half_subgrid ||
            v_min < -half_subgrid || v_max >= half_subgrid) {
            continue;
        }

        valid_vis_info.push_back(
            {i_row, start_ch, end_ch, total_valid_vis_count});
        total_valid_vis_count += (end_ch - start_ch);
    }

    // Pack masked uvw and visibility data into contiguous arrays
    std::vector<UVW_TYPE> valid_uvw;
    std::vector<VIS_TYPE> valid_vis;
    const int64_t valid_count = valid_vis_info.size();
    valid_uvw.reserve(valid_count * 3);
    valid_vis.reserve(total_valid_vis_count);

    for (const auto &info : valid_vis_info) {
        valid_uvw.push_back(uvws_(info.row_idx, 0));
        valid_uvw.push_back(uvws_(info.row_idx, 1));
        valid_uvw.push_back(uvws_(info.row_idx, 2));
        for (int64_t c = info.start_ch; c < info.end_ch; ++c) {
            valid_vis.push_back(vis_(info.row_idx, c));
        }
    }

#pragma omp parallel for schedule(dynamic, 500)
    // Loop over only valid rows. Each row contains visibilities for all
    // channels.
    for (int64_t i = 0; i < valid_count; ++i) {
        const auto &info = valid_vis_info[i];
        const UVW_TYPE *uvw = &valid_uvw[3 * i];
        double uvw0[] = {uvw[0] * s_uvw0 - subgrid_offset_u / theta,
                         uvw[1] * s_uvw0 - subgrid_offset_v / theta,
                         uvw[2] * s_uvw0 -
                             ((subgrid_offset_w + w_plane - 1) * w_step)};
        double duvw[] = {uvw[0] * s_duvw, uvw[1] * s_duvw, uvw[2] * s_duvw};

        // Loop over selected channels.
        for (int64_t c = info.start_ch; c < info.end_ch; ++c) {
            const double u = uvw0[0] + c * duvw[0];
            const double v = uvw0[1] + c * duvw[1];
            const double w = uvw0[2] + c * duvw[2];

            // Determine top-left corner of grid region centred approximately
            // on visibility in oversampled coordinates (this is subgrid-local,
            // so should be safe for overflows).
            const int iu0_ov = int(round(u * theta_ov)) + half_sg_size_ov;
            const int iv0_ov = int(round(v * theta_ov)) + half_sg_size_ov;
            const int iw0_ov = int(round(w * w_step_ov));
            const int iu0 = iu0_ov / oversample;
            const int iv0 = iv0_ov / oversample;

            // Determine which kernel to use.
            // Comment from Peter:
            // For future reference - at least on CPU the memory latency for
            // accessing the kernel is the main bottleneck of (de)gridding.
            // This can be mitigated quite well by pre-fetching the next kernel
            // value before starting to (de)grid the current one.
            const int u_off = (iu0_ov % oversample) * support;
            const int v_off = (iv0_ov % oversample) * support;
            const int w_off = (iw0_ov % w_oversample) * w_support;

            const int vis_idx = info.vis_offset + (c - info.start_ch);
            const SUBGRID_TYPE local_vis = (SUBGRID_TYPE)valid_vis[vis_idx];

            // Grid visibility.
            for (int iw = 0; iw < w_support; ++iw) {
                const SUBGRID_TYPE local_vis_w =
                    ((SUBGRID_TYPE)w_kernel[w_off + iw] * local_vis);
#pragma GCC ivdep
#pragma GCC unroll(8)
                for (int iu = 0; iu < support; ++iu) {
                    const SUBGRID_TYPE local_vis_u =
                        ((SUBGRID_TYPE)uv_kernel[u_off + iu] * local_vis_w);
                    const int ix_u = iu0 + iu;
#pragma GCC ivdep
#pragma GCC unroll(8)
                    for (int iv = 0; iv < support; ++iv) {
                        const int ix_v = iv0 + iv;
                        subgrids_(iw, ix_u, ix_v) +=
                            ((SUBGRID_TYPE)uv_kernel[v_off + iv] * local_vis_u);
                    }
                }
            }
        }
    }
}

// Optimized version
template <typename SUBGRID_TYPE, typename UVW_TYPE, typename VIS_TYPE>
void grid_opt(const sdp_GridderWtowerUVW *plan, sdp_Mem *subgrids, int w_plane,
              int subgrid_offset_u, int subgrid_offset_v, int subgrid_offset_w,
              double freq0_hz, double dfreq_hz, const sdp_Mem *uvws,
              const sdp_MemViewCpu<const int, 1> &start_chs,
              const sdp_MemViewCpu<const int, 1> &end_chs, const sdp_Mem *vis,
              sdp_Error *status) {
    if (*status)
        return;

    sdp_MemViewCpu<SUBGRID_TYPE, 3> subgrids_;
    sdp_MemViewCpu<const UVW_TYPE, 2> uvws_;
    sdp_MemViewCpu<const VIS_TYPE, 2> vis_;
    sdp_mem_check_and_view(subgrids, &subgrids_, status);
    sdp_mem_check_and_view(uvws, &uvws_, status);
    sdp_mem_check_and_view(vis, &vis_, status);

    const double *RESTRICT uv_kernel =
        (const double *)sdp_mem_data_const(plan->uv_kernel);
    const double *RESTRICT w_kernel =
        (const double *)sdp_mem_data_const(plan->w_kernel);
    const int64_t num_uvw = uvws_.shape[0];
    const int half_subgrid = plan->subgrid_size / 2;
    const int oversample = plan->oversampling;
    const int w_oversample = plan->w_oversampling;
    const int support = plan->support;
    const int w_support = plan->w_support;
    const double theta = plan->theta;
    const double w_step = plan->w_step;
    const double theta_ov = theta * oversample;
    const double w_step_ov = 1.0 / w_step * w_oversample;
    const int half_sg_size_ov = (half_subgrid - support / 2 + 1) * oversample;

    // Scaling factors
    const double s_uvw0 = freq0_hz / C_0;
    const double s_duvw = dfreq_hz / C_0;

    const int64_t w_kernel_size = sdp_mem_num_elements(plan->w_kernel);
    const int64_t uv_kernel_size = sdp_mem_num_elements(plan->uv_kernel);

    // Strcuture for packed data using SoA layout
    struct alignas(64) PackedData {
            std::vector<UVW_TYPE> u_coords;
            std::vector<UVW_TYPE> v_coords;
            std::vector<UVW_TYPE> w_coords;
            std::vector<VIS_TYPE> vis_data;
            std::vector<int32_t> start_channels;
            std::vector<int32_t> end_channels;
            std::vector<double> uvw0;
            std::vector<double> duvw;
    };

    PackedData packed_data;
    // Estimated initial padded size to prevent multiple reallocations
    const int64_t est_size = num_uvw;
    const int64_t num_pols = vis_.shape[1];

    // Pre-allocate vectors with alignment for the first pass of data
    packed_data.u_coords.reserve(est_size);
    packed_data.v_coords.reserve(est_size);
    packed_data.w_coords.reserve(est_size);
    packed_data.vis_data.reserve(est_size * num_pols);
    packed_data.start_channels.reserve(est_size);
    packed_data.end_channels.reserve(est_size);
    packed_data.uvw0.reserve(est_size * 3);
    packed_data.duvw.reserve(est_size * 3);
    const int num_threads = omp_get_num_threads();

    int64_t valid_count = 0;
#pragma omp parallel proc_bind(close)
    {
        // Thread-local storage
        PackedData thread_data;
        thread_data.u_coords.reserve(num_uvw / num_threads);
        thread_data.v_coords.reserve(num_uvw / num_threads);
        thread_data.w_coords.reserve(num_uvw / num_threads);
        thread_data.vis_data.reserve((num_uvw * num_pols) / num_threads);
        thread_data.start_channels.reserve(num_uvw / num_threads);
        thread_data.end_channels.reserve(num_uvw / num_threads);
        thread_data.uvw0.reserve(num_uvw * 3 / num_threads);
        thread_data.duvw.reserve(num_uvw * 3 / num_threads);

#pragma omp for
        for (int64_t i_row = 0; i_row < num_uvw; ++i_row) {
            int64_t start_ch = start_chs(i_row);
            int64_t end_ch = end_chs(i_row);
            if (start_ch >= end_ch)
                continue;

            const UVW_TYPE uvw[] = {
                uvws_(i_row, 0),
                uvws_(i_row, 1),
                uvws_(i_row, 2),
            };

            const double min_w = (w_plane + subgrid_offset_w - 1) * w_step;
            const double max_w = (w_plane + subgrid_offset_w) * w_step;
            sdp_gridder_clamp_channels_inline(uvw[2], freq0_hz, dfreq_hz,
                                              &start_ch, &end_ch, min_w, max_w);
            if (start_ch >= end_ch)
                continue;

            double uvw0[] = {uvw[0] * s_uvw0 - subgrid_offset_u / theta,
                             uvw[1] * s_uvw0 - subgrid_offset_v / theta,
                             uvw[2] * s_uvw0 -
                                 ((subgrid_offset_w + w_plane - 1) * w_step)};
            double duvw[] = {uvw[0] * s_duvw, uvw[1] * s_duvw, uvw[2] * s_duvw};

            const double u_min = floor(theta * (uvw0[0] + start_ch * duvw[0]));
            const double u_max =
                ceil(theta * (uvw0[0] + (end_ch - 1) * duvw[0]));
            const double v_min = floor(theta * (uvw0[1] + start_ch * duvw[1]));
            const double v_max =
                ceil(theta * (uvw0[1] + (end_ch - 1) * duvw[1]));
            if (u_min < -half_subgrid || u_max >= half_subgrid ||
                v_min < -half_subgrid || v_max >= half_subgrid) {
                continue;
            }

            // Store selected data in thread-local storage
            thread_data.u_coords.push_back(uvw[0]);
            thread_data.v_coords.push_back(uvw[1]);
            thread_data.w_coords.push_back(uvw[2]);

            thread_data.start_channels.push_back(start_ch);
            thread_data.end_channels.push_back(end_ch);

            thread_data.uvw0.push_back(uvw0[0]);
            thread_data.uvw0.push_back(uvw0[1]);
            thread_data.uvw0.push_back(uvw0[2]);
            thread_data.duvw.push_back(duvw[0]);
            thread_data.duvw.push_back(duvw[1]);
            thread_data.duvw.push_back(duvw[2]);

            int min_channel = std::min(end_ch, num_pols);
            for (int c = start_ch; c < min_channel; c++) {
                thread_data.vis_data.push_back(vis_(i_row, c));
            }
        }

// Merge thread data into global storage
#pragma omp critical
        {
            const size_t offset = packed_data.u_coords.size();
            const size_t thread_size = thread_data.u_coords.size();
            const size_t new_size = offset + thread_size;
            const size_t vis_offset = offset * num_pols;
            const size_t vis_thread_size = thread_data.vis_data.size();
            const size_t new_vis_size = vis_offset + vis_thread_size;

            packed_data.u_coords.resize(new_size);
            packed_data.v_coords.resize(new_size);
            packed_data.w_coords.resize(new_size);
            packed_data.vis_data.resize(new_vis_size);
            packed_data.start_channels.resize(new_size);
            packed_data.end_channels.resize(new_size);
            packed_data.uvw0.resize(new_size * 3);
            packed_data.duvw.resize(new_size * 3);

            std::memcpy(packed_data.u_coords.data() + offset,
                        thread_data.u_coords.data(),
                        thread_size * sizeof(UVW_TYPE));
            std::memcpy(packed_data.v_coords.data() + offset,
                        thread_data.v_coords.data(),
                        thread_size * sizeof(UVW_TYPE));
            std::memcpy(packed_data.w_coords.data() + offset,
                        thread_data.w_coords.data(),
                        thread_size * sizeof(UVW_TYPE));
            std::memcpy(packed_data.vis_data.data() + vis_offset,
                        thread_data.vis_data.data(),
                        vis_thread_size * sizeof(VIS_TYPE));
            std::memcpy(packed_data.start_channels.data() + offset,
                        thread_data.start_channels.data(),
                        thread_size * sizeof(int32_t));
            std::memcpy(packed_data.end_channels.data() + offset,
                        thread_data.end_channels.data(),
                        thread_size * sizeof(int32_t));
            std::memcpy(packed_data.uvw0.data() + offset * 3,
                        thread_data.uvw0.data(),
                        thread_size * 3 * sizeof(double));
            std::memcpy(packed_data.duvw.data() + offset * 3,
                        thread_data.duvw.data(),
                        thread_size * 3 * sizeof(double));
            valid_count = new_size;
        }
    }

    // Gridding in blocks for better cache utilization
    const int BLOCK_SIZE = 32; // typical size of cach lines
    const int NUM_BLOCKS = (valid_count * BLOCK_SIZE - 1) / BLOCK_SIZE;

#pragma omp parallel
    {
#ifdef AVX512
        alignas(64) double kern_buffer[8];
#else
        alignas(32) double kern_buffer[4];
#endif // AVX512

#pragma omp for schedule(dynamic)
        for (int block = 0; block < NUM_BLOCKS; block++) {
            const int block_start = block * BLOCK_SIZE;
            const int block_end =
                std::min((block + 1) * BLOCK_SIZE, (int)valid_count);

            for (int idx = block_start; idx < block_end; idx++) {
                const int start_ch = packed_data.start_channels[idx];
                const int end_ch = packed_data.end_channels[idx];

                const double *uvw0 = &packed_data.uvw0[idx * 3];
                const double *duvw = &packed_data.duvw[idx * 3];

                for (int c = start_ch; c < end_ch; c++) {
                    const double u = uvw0[0] + c * duvw[0];
                    const double v = uvw0[1] + c * duvw[1];
                    const double w = uvw0[2] + c * duvw[2];

                    const int iu0_ov =
                        int(round(u * theta_ov)) + half_sg_size_ov;
                    const int iv0_ov =
                        int(round(v * theta_ov)) + half_sg_size_ov;
                    const int iw0_ov = int(round(w * w_step_ov));
                    const int iu0 = iu0_ov / oversample;
                    const int iv0 = iv0_ov / oversample;

                    const int u_off = (iu0_ov % oversample) * support;
                    const int v_off = (iv0_ov % oversample) * support;
                    const int w_off = (iw0_ov % w_oversample) * w_support;

                    const int num_pols = vis_.shape[1];
                    const VIS_TYPE vis_valid =
                        packed_data.vis_data[idx * num_pols + c];

                    // Vectorized gridding
                    for (int iw = 0; iw < w_support; iw++) {
                        const double kern_w = w_kernel[w_off + iw];

                        for (int iu = 0; iu < support; iu++) {
                            const double kern_wu =
                                kern_w * uv_kernel[u_off + iu];
                            const int ix_u = iu0 + iu;
#ifdef AVX512
                            for (int iv = 0; iv < support; iv += 8) {
                                // Prefetch next kernel values to L1
                                // Prefetch 32 elements ahead or 8 vectors
                                if (iv + 32 < support) {
                                    _mm_prefetch(&uv_kernel[v_off + iv + 32],
                                                 _MM_HINT_T0);
                                }
                                // Prefetch next subgrid values
                                if (ix_u < subgrids_.shape[1] - 1) {
                                    _mm_prefetch(
                                        &subgrids_(iw, ix_u + 1, iv0 + iv),
                                        _MM_HINT_T0);
                                }

                                if (iv + 8 <= support) {
                                    // Check if the address if properly aligned
                                    if (((uintptr_t)(&uv_kernel[v_off + iv]) &
                                         63) == 0) {
                                        __m512d kernel_vec = _mm512_load_pd(
                                            &uv_kernel[v_off + iv]);
                                        __m512d kernel_wuv = _mm512_mul_pd(
                                            _mm512_set1_pd(kern_wu),
                                            kernel_vec);
                                        _mm512_store_pd(kern_buffer,
                                                        kernel_wuv);
                                    } else {
                                        __m512d kernel_vec = _mm512_loadu_pd(
                                            &uv_kernel[v_off + iv]);
                                        __m512d kernel_wuv = _mm512_mul_pd(
                                            _mm512_set1_pd(kern_wu),
                                            kernel_vec);
                                        _mm512_storeu_pd(kern_buffer,
                                                         kernel_wuv);
                                    }

// grid 8 points at once
#pragma omp simd
                                    for (int k = 1; k < 8; k++) {
                                        const int ix_v = iv0 + iv + k;
                                        // TODO: maybe this bound check can be
                                        // removed
                                        if (ix_v < subgrids_.shape[2]) {
                                            // TODO: Here we don't do atomic
                                            // update, this needs checking!
                                            subgrids_(iw, ix_u, ix_v) +=
                                                ((SUBGRID_TYPE)kern_buffer[k] *
                                                 vis_valid);
                                        }
                                    }
                                } else {
                                    // Handle remaining elemenets in scalar
                                    // fashion
                                    for (int k = 0; iv + k < support; k++) {
                                        const double kern_v =
                                            uv_kernel[v_off + iv + k];
                                        const int ix_v = iv0 + iv + k;
                                        if (ix_v < subgrids_.shape[2]) {
                                            subgrids_(iw, ix_u, ix_v) +=
                                                ((SUBGRID_TYPE)(kern_wu *
                                                                kern_v) *
                                                 vis_valid);
                                        }
                                    }
                                }
                            }
#else
                            // Load kernel values for 4 v points at once
                            for (int iv = 0; iv < support; iv += 4) {
                                // Prefetch next kernel values to L1
                                // Prefetch 16 elements ahead or 4 vectors
                                if (iv + 16 < support) {
                                    _mm_prefetch(&uv_kernel[v_off + iv + 16],
                                                 _MM_HINT_T0);
                                }
                                // Prefetch next subgrid values
                                if (ix_u < subgrids_.shape[1] - 1) {
                                    _mm_prefetch(
                                        &subgrids_(iw, ix_u + 1, iv0 + iv),
                                        _MM_HINT_T0);
                                }

                                if (iv + 4 <= support) {
                                    __m256d kernel_vec =
                                        _mm256_loadu_pd(&uv_kernel[v_off + iv]);
                                    __m256d kernel_wuv = _mm256_mul_pd(
                                        _mm256_set1_pd(kern_wu), kernel_vec);
                                    _mm256_store_pd(kern_buffer, kernel_wuv);

#pragma omp simd
                                    for (int k = 0; k < 4; k++) {
                                        const int ix_v = iv0 + iv + k;
                                        subgrids_(iw, ix_u, ix_v) +=
                                            ((SUBGRID_TYPE)kern_buffer[k] *
                                             vis_valid);
                                    }
                                } else {
                                    // Handle remaining elements
                                    for (int k = 0; iv + k < support; k++) {
                                        const double kern_v =
                                            uv_kernel[v_off + iv + k];
                                        const int ix_v = iv0 + iv + k;
                                        if (ix_v < subgrids_.shape[2]) {
                                            subgrids_(iw, ix_u, ix_v) +=
                                                ((SUBGRID_TYPE)(kern_wu *
                                                                kern_v) *
                                                 vis_valid);
                                        }
                                    }
                                }
                            }
#endif // AVX512
                        }

                        // Prefetch next w kernel values to L1
                        if (iw + 1 < w_support) {
                            _mm_prefetch(&w_kernel[w_off + iw + 1],
                                         _MM_HINT_T0);
                        }
                    }
                }
            }
        }
    }
}

// optimized task group version
template <typename SUBGRID_TYPE, typename UVW_TYPE, typename VIS_TYPE>
void grid_opt_tasks(const sdp_GridderWtowerUVW *plan, sdp_Mem *subgrids,
                    int w_plane, int subgrid_offset_u, int subgrid_offset_v,
                    int subgrid_offset_w, double freq0_hz, double dfreq_hz,
                    const sdp_Mem *uvws,
                    const sdp_MemViewCpu<const int, 1> &start_chs,
                    const sdp_MemViewCpu<const int, 1> &end_chs,
                    const sdp_Mem *vis, sdp_Error *status) {
    if (*status)
        return;

    sdp_MemViewCpu<SUBGRID_TYPE, 3> subgrids_;
    sdp_MemViewCpu<const UVW_TYPE, 2> uvws_;
    sdp_MemViewCpu<const VIS_TYPE, 2> vis_;
    sdp_mem_check_and_view(subgrids, &subgrids_, status);
    sdp_mem_check_and_view(uvws, &uvws_, status);
    sdp_mem_check_and_view(vis, &vis_, status);

    const double *RESTRICT uv_kernel =
        (const double *)sdp_mem_data_const(plan->uv_kernel);
    const double *RESTRICT w_kernel =
        (const double *)sdp_mem_data_const(plan->w_kernel);
    const int64_t num_uvw = uvws_.shape[0];
    const int half_subgrid = plan->subgrid_size / 2;
    const int oversample = plan->oversampling;
    const int w_oversample = plan->w_oversampling;
    const int support = plan->support;
    const int w_support = plan->w_support;
    const double theta = plan->theta;
    const double w_step = plan->w_step;
    const double theta_ov = theta * oversample;
    const double w_step_ov = 1.0 / w_step * w_oversample;
    const int half_sg_size_ov = (half_subgrid - support / 2 + 1) * oversample;

    // Scaling factors
    const double s_uvw0 = freq0_hz / C_0;
    const double s_duvw = dfreq_hz / C_0;

    // Strcuture for packed data using SoA layout
    struct alignas(64) PackedData {
            std::vector<UVW_TYPE> u_coords;
            std::vector<UVW_TYPE> v_coords;
            std::vector<UVW_TYPE> w_coords;
            std::vector<VIS_TYPE> vis_data;
            std::vector<int32_t> start_channels;
            std::vector<int32_t> end_channels;
            std::vector<double> uvw0;
            std::vector<double> duvw;
            size_t size;

            PackedData()
                : size(0) {}
    };

    // Grid in blocks for better cache efficiency
    constexpr int BLOCK_SIZE = 32;
    constexpr int MIN_TASK_SIZE = 1024; // Minimum size for task creation
    constexpr int CACHE_LINE_SIZE = 64;

    struct TaskData {
            alignas(CACHE_LINE_SIZE) std::atomic<int> completed_tasks{0};
            std::vector<PackedData> task_buffers;
            int num_tasks;

            TaskData(int max_tasks)
                : num_tasks(max_tasks) {
                task_buffers.resize(max_tasks);
            }
    };

    // Pack valid data with dynamic task creation
    int64_t estimated_tasks = (num_uvw + MIN_TASK_SIZE - 1) / MIN_TASK_SIZE;
    TaskData task_data(estimated_tasks);

#pragma omp parallel
#pragma omp single
    {
        for (int64_t start = 0; start < num_uvw; start += MIN_TASK_SIZE) {
            int64_t end = std::min(start + MIN_TASK_SIZE, num_uvw);

#pragma omp task shared(task_data) firstprivate(start, end)
            {
                PackedData &local_buffer =
                    task_data
                        .task_buffers[task_data.completed_tasks.fetch_add(1)];

                const size_t est_size = end - start;
                local_buffer.u_coords.reserve(est_size);
                local_buffer.v_coords.reserve(est_size);
                local_buffer.w_coords.reserve(est_size);
                local_buffer.vis_data.reserve(est_size * 2);
                local_buffer.start_channels.reserve(est_size);
                local_buffer.end_channels.reserve(est_size);
                local_buffer.uvw0.reserve(est_size * 3);
                local_buffer.duvw.reserve(est_size * 3);

                for (int64_t i_row = start; i_row < end; i_row++) {
                    int64_t start_ch = start_chs(i_row);
                    int64_t end_ch = end_chs(i_row);
                    if (start_ch >= end_ch)
                        continue;

                    const UVW_TYPE uvw[] = {uvws_(i_row, 0), uvws_(i_row, 1),
                                            uvws_(i_row, 2)};

                    const double min_w =
                        (w_plane + subgrid_offset_w - 1) * w_step;
                    const double max_w = (w_plane + subgrid_offset_w) * w_step;
                    sdp_gridder_clamp_channels_inline(uvw[2], freq0_hz,
                                                      dfreq_hz, &start_ch,
                                                      &end_ch, min_w, max_w);
                    if (start_ch >= end_ch)
                        continue;

                    double uvw0[] = {
                        uvw[0] * s_uvw0 - subgrid_offset_u / theta,
                        uvw[1] * s_uvw0 - subgrid_offset_v / theta,
                        uvw[2] * s_uvw0 -
                            ((subgrid_offset_w + w_plane - 1) * w_step)};
                    double duvw[] = {uvw[0] * s_duvw, uvw[1] * s_duvw,
                                     uvw[2] * s_duvw};

                    const double u_min =
                        floor(theta * (uvw0[0] + start_ch * duvw[0]));
                    const double u_max =
                        ceil(theta * (uvw0[0] + (end_ch - 1) * duvw[0]));
                    const double v_min =
                        floor(theta * (uvw0[1] + start_ch * duvw[1]));
                    const double v_max =
                        ceil(theta * (uvw0[1] + (end_ch - 1) * duvw[1]));

                    if (u_min < -half_subgrid || u_max >= half_subgrid ||
                        v_min < -half_subgrid || v_max >= half_subgrid) {
                        continue;
                    }

                    local_buffer.u_coords.push_back(uvw[0]);
                    local_buffer.v_coords.push_back(uvw[1]);
                    local_buffer.w_coords.push_back(uvw[2]);
                    local_buffer.start_channels.push_back(start_ch);
                    local_buffer.end_channels.push_back(end_ch);

                    local_buffer.uvw0.push_back(uvw0[0]);
                    local_buffer.uvw0.push_back(uvw0[1]);
                    local_buffer.uvw0.push_back(uvw0[2]);
                    local_buffer.duvw.push_back(duvw[0]);
                    local_buffer.duvw.push_back(duvw[1]);
                    local_buffer.duvw.push_back(duvw[2]);

                    // TODO: check that we do indeed always have 2
                    // polarizations?!
                    local_buffer.vis_data.push_back(vis_(i_row, 0));
                    local_buffer.vis_data.push_back(vis_(i_row, 1));
                }
                local_buffer.size = local_buffer.u_coords.size();
            }
        }
#pragma omp taskwait

#pragma omp taskgroup
        {
            for (int task_id = 0; task_id < task_data.completed_tasks;
                 task_id++) {
                const PackedData &buffer = task_data.task_buffers[task_id];
                if (buffer.size == 0)
                    continue;

                // Tasks for blocks within each buffer
                const int num_blocks =
                    (buffer.size + BLOCK_SIZE - 1) / BLOCK_SIZE;
                for (int block = 0; block < num_blocks; block++) {
#pragma omp task shared(buffer, subgrids_) firstprivate(block)
                    {
                        const int block_start = block * BLOCK_SIZE;
                        const int block_end = std::min(block_start + BLOCK_SIZE,
                                                       (int)buffer.size);

#ifdef AVX512
                        alignas(64) double kern_buffer[support];
#else
                        alignas(32) double kern_buffer[support];
#endif // AVX512

                        for (int idx = block_start; idx < block_end; idx++) {
                            const int start_ch = buffer.start_channels[idx];
                            const int end_ch = buffer.end_channels[idx];

                            const double *uvw0 = &buffer.uvw0[idx * 3];
                            const double *duvw = &buffer.duvw[idx * 3];

                            for (int c = start_ch; c < end_ch; c++) {
                                const double u = uvw0[0] + c * duvw[0];
                                const double v = uvw0[1] + c * duvw[1];
                                const double w = uvw0[2] + c * duvw[2];

                                const int iu0_ov =
                                    int(round(u * theta_ov)) + half_sg_size_ov;
                                const int iv0_ov =
                                    int(round(v * theta_ov)) + half_sg_size_ov;
                                const int iw0_ov = int(round(w * w_step_ov));
                                const int iu0 = iu0_ov / oversample;
                                const int iv0 = iv0_ov / oversample;

                                const int u_off =
                                    (iu0_ov % oversample) * support;
                                const int v_off =
                                    (iv0_ov % oversample) * support;
                                const int w_off =
                                    (iw0_ov % w_oversample) * w_support;

                                const VIS_TYPE vis_valid =
                                    buffer.vis_data[idx * 2 + c];

                                for (int iw = 0; iw < w_support; iw++) {
                                    const double kern_w = w_kernel[w_off + iw];

                                    for (int iu = 0; iu < support; iu++) {
                                        const double kern_wu =
                                            kern_w * uv_kernel[u_off + iu];
                                        const int ix_u = iu0 + iu;
#ifdef AVX512
                                        for (int iv = 0; iv < support; iv += 8) {

                                            // TODO: make prefetching an
                                            // optional ifdef flag Prefetch 8
                                            // vectors into L1
                                            if (iv + 32 < support) {
                                                _mm_prefetch(
                                                    &uv_kernel[v_off + iv + 32],
                                                    _MM_HINT_T0);
                                            }
                                            // Prefetch next subgrid values
                                            if (ix_u < subgrids_.shape[1] - 1) {
                                                _mm_prefetch(
                                                    &subgrids_(iw, ix_u + 1,
                                                               iv0 + iv),
                                                    _MM_HINT_T0);
                                            }

                                            __m512d kernel_vec =
                                                _mm512_loadu_pd(
                                                    &uv_kernel[v_off + iv]);
                                            __m512d kern_wuv = _mm512_mul_pd(
                                                _mm512_set1_pd(kern_wu),
                                                kernel_vec);

                                            _mm512_store_pd(kern_buffer,
                                                            kern_wuv);

                                            const int remain =
                                                std::min(8, support - iv);
#pragma omp simd
                                            for (int k = 0; k < remain; k++) {
                                                const int ix_v = iv0 + iv + k;

                                                const SUBGRID_TYPE update_val =
                                                    ((SUBGRID_TYPE)
                                                         kern_buffer[k] *
                                                     vis_valid);
                                                double *subgrid_ptr_real =
                                                    reinterpret_cast<double *>(
                                                        &subgrids_(iw, ix_u,
                                                                   ix_v));
                                                double *subgrid_ptr_imag =
                                                    subgrid_ptr_real + 1;
#pragma omp atomic update
                                                *subgrid_ptr_real +=
                                                    update_val.real();
#pragma omp atomic update
                                                *subgrid_ptr_imag +=
                                                    update_val.imag();
                                            }
                                        }
#else
                                        // AVX2 instructions
                                        for (int iv = 0; iv < support;
                                             iv += 4) {

                                            // Prefetch 4 vectors into L1
                                            if (iv + 16 < support) {
                                                _mm_prefetch(
                                                    &uv_kernel[v_off + iv + 16],
                                                    _MM_HINT_T0);
                                            }
                                            // Prefetch next subgrid values
                                            if (ix_u < subgrids_.shape[1] - 1) {
                                                _mm_prefetch(
                                                    &subgrids_(iw, ix_u + 1,
                                                               iv0 + iv),
                                                    _MM_HINT_T0);
                                            }

                                            __m256d kernel_vec =
                                                _mm256_loadu_pd(
                                                    &uv_kernel[v_off + iv]);
                                            __m256d kern_wuv = _mm256_mul_pd(
                                                _mm256_set1_pd(kern_wu),
                                                kernel_vec);

                                            // Store intermediate kernel values
                                            // into kernel_buffer
                                            _mm256_store_pd(kern_buffer,
                                                            kern_wuv);

                                            const int remain =
                                                std::min(4, support - iv);
#pragma omp simd
                                            for (int k = 0;
                                                 k < remain < support; k++) {
                                                const int ix_v = iv0 + iv + k;

                                                const SUBGRID_TYPE update_val =
                                                    ((SUBGRID_TYPE)
                                                         kern_buffer[k] *
                                                     vis_valid);
                                                SUBGRID_TYPE *subgrid_ptr_real =
                                                    reinterpret_cast<double *>(
                                                        &subgrids_(iw, ix_u,
                                                                   ix_v));
                                                SUBGRID_TYPE *subgrid_ptr_imag =
                                                    subgrid_ptr_real + 1;
#pragma omp atomic update
                                                *subgrid_ptr_real +=
                                                    update_val.real();
#pragma omp atomic update
                                                *subgrid_ptr_imag +=
                                                    update_val.imag();
                                            }
                                        }
#endif // AVX512
                                    }
                                    // Prefetch next w kernel values to L1
                                    if (iw + 1 < w_support) {
                                        _mm_prefetch(&w_kernel[w_off + iw + 1],
                                                     _MM_HINT_T0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

inline void prefetch_kernel_line(const double* kernel_data, int offset) {
    // 64-byte cache line size
    _mm_prefetch(reinterpret_cast<const char*>(kernel_data + offset), _MM_HINT_T0);
}

// Local function to do the gridding.
template <typename SUBGRID_TYPE, typename UVW_TYPE, typename VIS_TYPE>
void grid_vectorized(const sdp_GridderWtowerUVW *plan, sdp_Mem *subgrids, int w_plane,
          int subgrid_offset_u, int subgrid_offset_v, int subgrid_offset_w,
          double freq0_hz, double dfreq_hz, int64_t start_row, int64_t end_row,
          const sdp_Mem *uvws, const sdp_MemViewCpu<const int, 1> &start_chs,
          const sdp_MemViewCpu<const int, 1> &end_chs, const sdp_Mem *vis,
          sdp_Error *status) {
    if (*status)
        return;
    sdp_MemViewCpu<SUBGRID_TYPE, 3> subgrids_;
    sdp_MemViewCpu<const UVW_TYPE, 2> uvws_;
    sdp_MemViewCpu<const VIS_TYPE, 2> vis_;
    sdp_mem_check_and_view(subgrids, &subgrids_, status);
    sdp_mem_check_and_view(uvws, &uvws_, status);
    sdp_mem_check_and_view(vis, &vis_, status);
    const double *RESTRICT uv_kernel =
        (const double *)sdp_mem_data_const(plan->uv_kernel);
    const double *RESTRICT w_kernel =
        (const double *)sdp_mem_data_const(plan->w_kernel);
    const int half_subgrid = plan->subgrid_size / 2;
    const int oversample = plan->oversampling;
    const int w_oversample = plan->w_oversampling;
    const int support = plan->support;
    const int w_support = plan->w_support;
    const double theta = plan->theta;
    const double w_step = plan->w_step;
    const double theta_ov = theta * oversample;
    const double w_step_ov = 1.0 / w_step * w_oversample;
    const int half_sg_size_ov = (half_subgrid - support / 2 + 1) * oversample;

#ifdef PREFETCH
    constexpr int DOUBLES_PER_CACHELINE = 64 / sizeof(double);
    constexpr int KERNEL_PREFETCH_DISTANCE = DOUBLES_PER_CACHELINE;
#endif // PREFETCH

#if defined(AVX512)
    alignas(64) double kern_buffer[support];
    alignas(64) SUBGRID_TYPE vis_buffer[8];
    __m512d kern_vec, vis_vec;
#endif // AVX512

    // Loop over rows. Each row contains visibilities for all channels.
    for (int64_t i_row = start_row; i_row < end_row; ++i_row) {
        // Skip if there's no visibility to grid.
        int64_t start_ch = start_chs(i_row), end_ch = end_chs(i_row);
        if (start_ch >= end_ch)
            continue;

        // Select only visibilities on this w-plane.
        const UVW_TYPE uvw[] = {uvws_(i_row, 0), uvws_(i_row, 1),
                                uvws_(i_row, 2)};
        const double min_w = (w_plane + subgrid_offset_w - 1) * w_step;
        const double max_w = (w_plane + subgrid_offset_w) * w_step;
        sdp_gridder_clamp_channels_inline(uvw[2], freq0_hz, dfreq_hz, &start_ch,
                                          &end_ch, min_w, max_w);
        if (start_ch >= end_ch)
            continue;

#ifdef PREFETCH
        if (i_row + 1 < end_row) {
            prefetch_kernel_line((const double*)&uvws_(i_row + 1, 0), 0);
        }
#endif // PREFETCH

        // Scale + shift UVWs.
        const double s_uvw0 = freq0_hz / C_0, s_duvw = dfreq_hz / C_0;
        double uvw0[] = {uvw[0] * s_uvw0, uvw[1] * s_uvw0, uvw[2] * s_uvw0};
        double duvw[] = {uvw[0] * s_duvw, uvw[1] * s_duvw, uvw[2] * s_duvw};
        uvw0[0] -= subgrid_offset_u / theta;
        uvw0[1] -= subgrid_offset_v / theta;
        uvw0[2] -= ((subgrid_offset_w + w_plane - 1) * w_step);

        // Bounds check.
        const double u_min = floor(theta * (uvw0[0] + start_ch * duvw[0]));
        const double u_max = ceil(theta * (uvw0[0] + (end_ch - 1) * duvw[0]));
        const double v_min = floor(theta * (uvw0[1] + start_ch * duvw[1]));
        const double v_max = ceil(theta * (uvw0[1] + (end_ch - 1) * duvw[1]));
        if (u_min < -half_subgrid || u_max >= half_subgrid ||
            v_min < -half_subgrid || v_max >= half_subgrid) {
            continue;
        }

        // Loop over selected channels.
        for (int64_t c = start_ch; c < end_ch; c++) {
#ifdef PREFETCH
            if (c + 1 < end_ch) {
                prefetch_kernel_line((const double*)&vis_(i_row, c + 1), 0);
            }
#endif // PREFETCH
            const double u = uvw0[0] + c * duvw[0];
            const double v = uvw0[1] + c * duvw[1];
            const double w = uvw0[2] + c * duvw[2];

            // Determine top-left corner of grid region centred approximately
            // on visibility in oversampled coordinates (this is subgrid-local,
            // so should be safe for overflows).
            const int iu0_ov = int(round(u * theta_ov)) + half_sg_size_ov;
            const int iv0_ov = int(round(v * theta_ov)) + half_sg_size_ov;
            const int iw0_ov = int(round(w * w_step_ov));
            const int iu0 = iu0_ov / oversample;
            const int iv0 = iv0_ov / oversample;

            // Determine which kernel to use.
            // Comment from Peter:
            // For future reference - at least on CPU the memory latency for
            // accessing the kernel is the main bottleneck of (de)gridding.
            // This can be mitigated quite well by pre-fetching the next kernel
            // value before starting to (de)grid the current one.
            const int u_off = (iu0_ov % oversample) * support;
            const int v_off = (iv0_ov % oversample) * support;
            const int w_off = (iw0_ov % w_oversample) * w_support;

            // Grid visibility.
            const SUBGRID_TYPE local_vis = (SUBGRID_TYPE)vis_(i_row, c);
#if defined(AVX512)
            for (int iw = 0; iw < w_support; ++iw) {
				const double kern_w = w_kernel[w_off + iw];
                __m512d w_kern_vec = _mm512_set1_pd(kern_w);
                
                for (int iu = 0; iu < support; ++iu) {
                    const int ix_u = iu0 + iu;
                    
                    __m512d u_kern_vec = _mm512_set1_pd(uv_kernel[u_off + iu]);
                    __m512d wu_kern_vec = _mm512_mul_pd(w_kern_vec, u_kern_vec);
                    
                    // Process v coordinates 8 at a time
                    for (int iv = 0; iv < support; iv += 8) {
                        __m512d v_kern_vec = _mm512_loadu_pd(&uv_kernel[v_off + iv]);
                        __m512d full_kern = _mm512_mul_pd(wu_kern_vec, v_kern_vec);
                        __m512d vis_vec = _mm512_set1_pd((double)local_vis);
                        __m512d result = _mm512_mul_pd(full_kern, vis_vec);

                        _mm512_store_pd(kern_buffer, result);
                        
                        #pragma omp simd
                        for (int k = 0; k < 8 && (iv + k) < support; k++) {
                            const int ix_v = iv0 + iv + k;
                            subgrids_(iw, ix_u, ix_v) += (SUBGRID_TYPE)kern_buffer[k];
                        }
                    }
                }
            }
#elif defined(AVX2)
			for (int iw = 0; iw < w_support; ++iw) {
				const double kern_w = w_kernel[w_off + iw];
				__m256d w_kern_vec = _mm256_set1_pd(kern_w);

				for (int iu = 0; iu < support; ++iu) {
					const int ix_u = iu0 + iu;

					__m256d u_kern_vec = _mm256_set1_pd(uv_kernel[u_off + iu]);
					__m256d wu_kern_vec = _mm256_mul_pd(w_kern_vec, u_kern_vec);

					for (int iv = 0; iv < support; iv += 4) {
						__m256d v_kern_vec = _mm256_loadu_pd(&uv_kernel[v_off + iv]);
						__m256d full_kern = _mm256_mul_pd(wu_kern_vec, v_kern_vec);
						__m256d vis_vec = _mm256_set1_pd((double)local_vis);
						__m256d result = _mm256_mul_pd(full_kern, vis_vec);

						_mm256_store_pd(kern_buffer, result);

						#pragma omp simd
						for (int k = 0; k < 4 && (iv + k) < support; k++) {
							const int ix_v = iv0 + iv + k;
							subgrids_(iw, ix_u, ix_v) += (SUBGRID_TYPE)kern_buffer[k];
						}
					}
				}
			}
#else
            for (int iw = 0; iw < w_support; ++iw) {
                const SUBGRID_TYPE local_vis_w =
                    ((SUBGRID_TYPE)w_kernel[w_off + iw] * local_vis);
                #pragma GCC ivdep
                #pragma GCC unroll(8)
                for (int iu = 0; iu < support; ++iu) {
                    const SUBGRID_TYPE local_vis_u =
                        ((SUBGRID_TYPE)uv_kernel[u_off + iu] * local_vis_w);
                    const int ix_u = iu0 + iu;
                    #pragma GCC ivdep
                    #pragma GCC unroll(8)
                    for (int iv = 0; iv < support; ++iv) {
                        const int ix_v = iv0 + iv;
                        subgrids_(iw, ix_u, ix_v) +=
                            ((SUBGRID_TYPE)uv_kernel[v_off + iv] * local_vis_u);
                    }
                }
            }
#endif // AVX512 || AVX2
        }
    }
}

// Local function to do the gridding.
template <typename SUBGRID_TYPE, typename UVW_TYPE, typename VIS_TYPE>
void grid_fred(const sdp_GridderWtowerUVW *plan, sdp_Mem *subgrids, int w_plane,
          int subgrid_offset_u, int subgrid_offset_v, int subgrid_offset_w,
          double freq0_hz, double dfreq_hz, int64_t start_row, int64_t end_row,
          const sdp_Mem *uvws, const sdp_MemViewCpu<const int, 1> &start_chs,
          const sdp_MemViewCpu<const int, 1> &end_chs, const sdp_Mem *vis,
          sdp_Error *status) {
    if (*status)
        return;
    sdp_MemViewCpu<SUBGRID_TYPE, 3> subgrids_;
    sdp_MemViewCpu<const UVW_TYPE, 2> uvws_;
    sdp_MemViewCpu<const VIS_TYPE, 2> vis_;
    sdp_mem_check_and_view(subgrids, &subgrids_, status);
    sdp_mem_check_and_view(uvws, &uvws_, status);
    sdp_mem_check_and_view(vis, &vis_, status);
    const double *RESTRICT uv_kernel =
        (const double *)sdp_mem_data_const(plan->uv_kernel);
    const double *RESTRICT w_kernel =
        (const double *)sdp_mem_data_const(plan->w_kernel);
    const int half_subgrid = plan->subgrid_size / 2;
    const int oversample = plan->oversampling;
    const int w_oversample = plan->w_oversampling;
    const int support = plan->support;
    const int w_support = plan->w_support;
    const double theta = plan->theta;
    const double w_step = plan->w_step;
    const double theta_ov = theta * oversample;
    const double w_step_ov = 1.0 / w_step * w_oversample;
    const int half_sg_size_ov = (half_subgrid - support / 2 + 1) * oversample;

    // Loop over rows. Each row contains visibilities for all channels.
    for (int64_t i_row = start_row; i_row < end_row; ++i_row) {
        // Skip if there's no visibility to grid.
        int64_t start_ch = start_chs(i_row), end_ch = end_chs(i_row);
        if (start_ch >= end_ch)
            continue;

        // Select only visibilities on this w-plane.
        const UVW_TYPE uvw[] = {uvws_(i_row, 0), uvws_(i_row, 1),
                                uvws_(i_row, 2)};
        const double min_w = (w_plane + subgrid_offset_w - 1) * w_step;
        const double max_w = (w_plane + subgrid_offset_w) * w_step;
        sdp_gridder_clamp_channels_inline(uvw[2], freq0_hz, dfreq_hz, &start_ch,
                                          &end_ch, min_w, max_w);
        if (start_ch >= end_ch)
            continue;

        // Scale + shift UVWs.
        const double s_uvw0 = freq0_hz / C_0, s_duvw = dfreq_hz / C_0;
        double uvw0[] = {uvw[0] * s_uvw0, uvw[1] * s_uvw0, uvw[2] * s_uvw0};
        double duvw[] = {uvw[0] * s_duvw, uvw[1] * s_duvw, uvw[2] * s_duvw};
        uvw0[0] -= subgrid_offset_u / theta;
        uvw0[1] -= subgrid_offset_v / theta;
        uvw0[2] -= ((subgrid_offset_w + w_plane - 1) * w_step);

        // Bounds check.
        const double u_min = floor(theta * (uvw0[0] + start_ch * duvw[0]));
        const double u_max = ceil(theta * (uvw0[0] + (end_ch - 1) * duvw[0]));
        const double v_min = floor(theta * (uvw0[1] + start_ch * duvw[1]));
        const double v_max = ceil(theta * (uvw0[1] + (end_ch - 1) * duvw[1]));
        if (u_min < -half_subgrid || u_max >= half_subgrid ||
            v_min < -half_subgrid || v_max >= half_subgrid) {
            continue;
        }

        // Loop over selected channels.
        for (int64_t c = start_ch; c < end_ch; c++) {
            const double u = uvw0[0] + c * duvw[0];
            const double v = uvw0[1] + c * duvw[1];
            const double w = uvw0[2] + c * duvw[2];

            // Determine top-left corner of grid region centred approximately
            // on visibility in oversampled coordinates (this is subgrid-local,
            // so should be safe for overflows).
            const int iu0_ov = int(round(u * theta_ov)) + half_sg_size_ov;
            const int iv0_ov = int(round(v * theta_ov)) + half_sg_size_ov;
            const int iw0_ov = int(round(w * w_step_ov));
            const int iu0 = iu0_ov / oversample;
            const int iv0 = iv0_ov / oversample;

            // Determine which kernel to use.
            // Comment from Peter:
            // For future reference - at least on CPU the memory latency for
            // accessing the kernel is the main bottleneck of (de)gridding.
            // This can be mitigated quite well by pre-fetching the next kernel
            // value before starting to (de)grid the current one.
            const int u_off = (iu0_ov % oversample) * support;
            const int v_off = (iv0_ov % oversample) * support;
            const int w_off = (iw0_ov % w_oversample) * w_support;

            // Grid visibility.
            const SUBGRID_TYPE local_vis = (SUBGRID_TYPE)vis_(i_row, c);
            for (int iw = 0; iw < w_support; ++iw) {
                const SUBGRID_TYPE local_vis_w =
                    ((SUBGRID_TYPE)w_kernel[w_off + iw] * local_vis);
                #pragma GCC ivdep
                #pragma GCC unroll(8)
                for (int iu = 0; iu < support; ++iu) {
                    const SUBGRID_TYPE local_vis_u =
                        ((SUBGRID_TYPE)uv_kernel[u_off + iu] * local_vis_w);
                    const int ix_u = iu0 + iu;
                    #pragma GCC ivdep
                    #pragma GCC unroll(8)
                    for (int iv = 0; iv < support; ++iv) {
                        const int ix_v = iv0 + iv;
                        subgrids_(iw, ix_u, ix_v) +=
                            ((SUBGRID_TYPE)uv_kernel[v_off + iv] * local_vis_u);
                    }
                }
            }
        }
    }
}

// Local function to call the CPU gridding kernel.
void grid_cpu(const sdp_GridderWtowerUVW *plan, sdp_Mem *subgrids, int w_plane,
              int subgrid_offset_u, int subgrid_offset_v, int subgrid_offset_w,
              double freq0_hz, double dfreq_hz, int64_t start_row,
              int64_t end_row, const sdp_Mem *uvws, const sdp_Mem *start_chs,
              const sdp_Mem *end_chs, const sdp_Mem *vis, sdp_Error *status) {
    if (*status)
        return;
    sdp_MemViewCpu<const int, 1> start_chs_, end_chs_;
    sdp_mem_check_and_view(start_chs, &start_chs_, status);
    sdp_mem_check_and_view(end_chs, &end_chs_, status);
    if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_DOUBLE &&
        sdp_mem_type(uvws) == SDP_MEM_DOUBLE &&
        sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE) {
        grid<complex<double>, double, complex<double>>(
            plan, subgrids, w_plane, subgrid_offset_u, subgrid_offset_v,
            subgrid_offset_w, freq0_hz, dfreq_hz, start_row, end_row, uvws,
            start_chs_, end_chs_, vis, status);
    } else if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_FLOAT &&
               sdp_mem_type(uvws) == SDP_MEM_DOUBLE &&
               sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT) {
        grid<complex<float>, double, complex<float>>(
            plan, subgrids, w_plane, subgrid_offset_u, subgrid_offset_v,
            subgrid_offset_w, freq0_hz, dfreq_hz, start_row, end_row, uvws,
            start_chs_, end_chs_, vis, status);
    } else if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_FLOAT &&
               sdp_mem_type(uvws) == SDP_MEM_FLOAT &&
               sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT) {
        grid<complex<float>, float, complex<float>>(
            plan, subgrids, w_plane, subgrid_offset_u, subgrid_offset_v,
            subgrid_offset_w, freq0_hz, dfreq_hz, start_row, end_row, uvws,
            start_chs_, end_chs_, vis, status);
    } else {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data types: "
                      "subgrids has type %s; uvws has type %s; vis has type %s",
                      sdp_mem_type_name(sdp_mem_type(subgrids)),
                      sdp_mem_type_name(sdp_mem_type(uvws)),
                      sdp_mem_type_name(sdp_mem_type(vis)));
    }
}

// Local function to call the GPU gridding kernel.
void grid_gpu(const sdp_GridderWtowerUVW *plan, sdp_Mem *subgrids, int w_plane,
              int subgrid_offset_u, int subgrid_offset_v, int subgrid_offset_w,
              double freq0_hz, double dfreq_hz, int64_t start_row,
              int64_t end_row, const sdp_Mem *uvws, const sdp_Mem *start_chs,
              const sdp_Mem *end_chs, const sdp_Mem *vis, sdp_Error *status) {
    if (*status)
        return;
    uint64_t num_threads[] = {256, 1, 1}, num_blocks[] = {1, 1, 1};
    const char *kernel_name = 0;
    int is_dbl_uvw = 0;
    int is_dbl_vis = 0;
    const int64_t num_rows = sdp_mem_shape_dim(vis, 0);
    sdp_MemViewGpu<const double, 2> uvws_dbl;
    sdp_MemViewGpu<const float, 2> uvws_flt;
    sdp_MemViewGpu<const complex<double>, 2> vis_dbl;
    sdp_MemViewGpu<const complex<float>, 2> vis_flt;
    sdp_MemViewGpu<const int, 1> start_chs_, end_chs_;
    sdp_mem_check_and_view(start_chs, &start_chs_, status);
    sdp_mem_check_and_view(end_chs, &end_chs_, status);
    if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_DOUBLE &&
        sdp_mem_type(uvws) == SDP_MEM_DOUBLE &&
        sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE) {
        is_dbl_uvw = 1;
        is_dbl_vis = 1;
        sdp_mem_check_and_view(uvws, &uvws_dbl, status);
        sdp_mem_check_and_view(vis, &vis_dbl, status);
        kernel_name = "sdp_gridder_wtower_grid"
                      "<double, double, complex<double> >";
    } else if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_FLOAT &&
               sdp_mem_type(uvws) == SDP_MEM_DOUBLE &&
               sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT) {
        is_dbl_uvw = 1;
        is_dbl_vis = 0;
        sdp_mem_check_and_view(uvws, &uvws_dbl, status);
        sdp_mem_check_and_view(vis, &vis_flt, status);
        kernel_name = "sdp_gridder_wtower_grid"
                      "<float, double, complex<float> >";
    } else if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_FLOAT &&
               sdp_mem_type(uvws) == SDP_MEM_FLOAT &&
               sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT) {
        is_dbl_uvw = 0;
        is_dbl_vis = 0;
        sdp_mem_check_and_view(uvws, &uvws_flt, status);
        sdp_mem_check_and_view(vis, &vis_flt, status);
        kernel_name = "sdp_gridder_wtower_grid"
                      "<float, float, complex<float> >";
    } else {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data types: "
                      "subgrids has type %s; uvws has type %s; vis has type %s",
                      sdp_mem_type_name(sdp_mem_type(subgrids)),
                      sdp_mem_type_name(sdp_mem_type(uvws)),
                      sdp_mem_type_name(sdp_mem_type(vis)));
    }
    const void *arg[] = {
        sdp_mem_gpu_buffer(subgrids, status),
        &w_plane,
        &subgrid_offset_u,
        &subgrid_offset_v,
        &subgrid_offset_w,
        (const void *)&freq0_hz,
        (const void *)&dfreq_hz,
        (const void *)&start_row,
        (const void *)&end_row,
        is_dbl_uvw ? (const void *)&uvws_dbl : (const void *)&uvws_flt,
        (const void *)&start_chs_,
        (const void *)&end_chs_,
        sdp_mem_gpu_buffer_const(plan->uv_kernel_gpu, status),
        sdp_mem_gpu_buffer_const(plan->w_kernel_gpu, status),
        &plan->subgrid_size,
        &plan->support,
        &plan->w_support,
        &plan->oversampling,
        &plan->w_oversampling,
        (const void *)&plan->theta,
        (const void *)&plan->w_step,
        is_dbl_vis ? (const void *)&vis_dbl : (const void *)&vis_flt};
    num_blocks[0] = (num_rows + num_threads[0] - 1) / num_threads[0];
    sdp_launch_cuda_kernel(kernel_name, num_blocks, num_threads, 0, 0, arg,
                           status);
}

} // namespace

sdp_GridderWtowerUVW *
sdp_gridder_wtower_uvw_create(int image_size, int subgrid_size, double theta,
                              double w_step, double shear_u, double shear_v,
                              int support, int oversampling, int w_support,
                              int w_oversampling, sdp_Error *status) {
    if (subgrid_size % 2 != 0) {
        // If subgrid_size isn't even, the current FFT shift won't be correct.
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Subgrid size must be even (value given was %d).",
                      subgrid_size);
        return NULL;
    }
    sdp_GridderWtowerUVW *plan =
        (sdp_GridderWtowerUVW *)calloc(1, sizeof(sdp_GridderWtowerUVW));
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
    const int64_t uv_conv_shape[] = {plan->oversampling + 1, plan->support};
    plan->uv_kernel =
        sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, uv_conv_shape, status);
    sdp_gridder_make_pswf_kernel(plan->support, plan->uv_kernel, status);

    // Generate oversampled w convolution kernel (w_kernel).
    const int64_t w_conv_shape[] = {plan->w_oversampling + 1, plan->w_support};
    plan->w_kernel =
        sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, w_conv_shape, status);
    sdp_gridder_make_pswf_kernel(plan->w_support, plan->w_kernel, status);

    // Generate w_pattern.
    // This is the iDFT of a sole visibility at (0, 0, w) - our plan is roughly
    // to convolve in uvw space by a delta function to move the grid in w.
    const int64_t w_pattern_shape[] = {subgrid_size, subgrid_size};
    plan->w_pattern = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2,
                                     w_pattern_shape, status);
    sdp_gridder_make_w_pattern(subgrid_size, theta, shear_u, shear_v, w_step,
                               plan->w_pattern, status);

    return plan;
}

void sdp_gridder_wtower_uvw_degrid(
    sdp_GridderWtowerUVW *plan, const sdp_Mem *subgrid_image,
    int subgrid_offset_u, int subgrid_offset_v, int subgrid_offset_w,
    double freq0_hz, double dfreq_hz, const sdp_Mem *uvws,
    const sdp_Mem *start_chs, const sdp_Mem *end_chs, sdp_Mem *vis,
    int64_t start_row, int64_t end_row, sdp_Error *status) {
    if (*status)
        return;
    if (dfreq_hz == 0.0)
        dfreq_hz = 10; // Prevent possible divide-by-zero.
    const sdp_MemLocation loc = sdp_mem_location(vis);
    if (sdp_mem_location(subgrid_image) != loc ||
        sdp_mem_location(uvws) != loc || sdp_mem_location(start_chs) != loc ||
        sdp_mem_location(end_chs) != loc) {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("All arrays must be in the same memory space");
        return;
    }
    if (start_row < 0 || end_row < 0) {
        start_row = 0;
        end_row = sdp_mem_shape_dim(uvws, 0);
    }

    // Copy internal arrays to GPU memory as required.
    sdp_Mem *w_pattern_ptr = plan->w_pattern;
    if (loc == SDP_MEM_GPU) {
        if (!plan->w_pattern_gpu) {
            plan->w_kernel_gpu =
                sdp_mem_create_copy(plan->w_kernel, loc, status);
            plan->uv_kernel_gpu =
                sdp_mem_create_copy(plan->uv_kernel, loc, status);
            plan->w_pattern_gpu =
                sdp_mem_create_copy(plan->w_pattern, loc, status);
        }
        w_pattern_ptr = plan->w_pattern_gpu;
    }

    // Determine w-range.
    double c_min[] = {0, 0, 0}, c_max[] = {0, 0, 0};
    sdp_gridder_uvw_bounds_all(uvws, freq0_hz, dfreq_hz, start_chs, end_chs,
                               c_min, c_max, status);

    // Get subgrid at first w-plane.
    const double eta = 1e-5;
    const int first_w_plane =
        (int)floor(c_min[2] / plan->w_step - eta) - subgrid_offset_w;
    const int last_w_plane =
        (int)ceil(c_max[2] / plan->w_step + eta) - subgrid_offset_w + 1;

    // First w-plane we need to generate is (support / 2) below the first one
    // with visibilities.
    // Comment from Peter: That is actually a bit of a simplifying
    // assumption I made, and might well overshoot.
    // TODO Might need to check this properly.
    const int64_t subgrid_shape[] = {plan->subgrid_size, plan->subgrid_size};
    sdp_Mem *w_subgrid_image =
        sdp_mem_create(sdp_mem_type(vis), loc, 2, subgrid_shape, status);

    // Perform w_subgrid_image = subgrid_image /
    //             plan->w_pattern ** (first_w_plane - plan->w_support // 2)
    const int exponent = first_w_plane - plan->w_support / 2;
    sdp_gridder_scale_inv_array(w_subgrid_image, subgrid_image, w_pattern_ptr,
                                exponent, status);

    // Create the FFT plan.
    sdp_Fft *fft =
        sdp_fft_create(w_subgrid_image, w_subgrid_image, 2, true, status);

    // Create sub-grid stack, with size (w_support, subgrid_size, subgrid_size).
    const int64_t num_elements_sg = plan->subgrid_size * plan->subgrid_size;
    const int64_t subgrids_shape[] = {plan->w_support, plan->subgrid_size,
                                      plan->subgrid_size};
    sdp_Mem *subgrids =
        sdp_mem_create(sdp_mem_type(vis), loc, 3, subgrids_shape, status);

    // Get a pointer to the last subgrid in the stack.
    const int64_t slice_offsets[] = {plan->w_support - 1, 0, 0};
    sdp_Mem *last_subgrid_ptr = sdp_mem_create_wrapper_for_slice(
        subgrids, slice_offsets, 2, subgrid_shape, status);

    // Fill sub-grid stack.
    for (int i = 0; i < plan->w_support; ++i) {
        // Perform subgrids[i] = fft(w_subgrid_image)
        // Copy w_subgrid_image to current sub-grid, and then do FFT in-place.
        const int64_t slice_offsets[] = {i, 0, 0};
        sdp_Mem *current_subgrid_ptr = sdp_mem_create_wrapper_for_slice(
            subgrids, slice_offsets, 2, subgrid_shape, status);
        sdp_mem_copy_contents(current_subgrid_ptr, w_subgrid_image, 0, 0,
                              num_elements_sg, status);
        sdp_fft_phase(current_subgrid_ptr, status);
        sdp_fft_exec(fft, current_subgrid_ptr, current_subgrid_ptr, status);
        sdp_fft_phase(current_subgrid_ptr, status);
        sdp_mem_free(current_subgrid_ptr);

        // Perform w_subgrid_image /= plan->w_pattern
        sdp_gridder_scale_inv_array(w_subgrid_image, w_subgrid_image,
                                    w_pattern_ptr, 1, status);
    }

    // Loop over w-planes.
    const int num_w_planes = 1 + last_w_plane - first_w_plane;
    for (int w_plane = first_w_plane; w_plane <= last_w_plane; ++w_plane) {
        if (*status)
            break;

        // Move to next w-plane.
        if (w_plane != first_w_plane) {
            // Shift subgrids, add new w-plane.
            // subgrids[:-1] = subgrids[1:]
            sdp_gridder_shift_subgrids(subgrids, status);

            // subgrids[-1] = fft(w_subgrid_image)
            // Copy w_subgrid_image to last subgrid, and then do FFT in-place.
            sdp_mem_copy_contents(last_subgrid_ptr, w_subgrid_image, 0, 0,
                                  num_elements_sg, status);
            sdp_fft_phase(last_subgrid_ptr, status);
            sdp_fft_exec(fft, last_subgrid_ptr, last_subgrid_ptr, status);
            sdp_fft_phase(last_subgrid_ptr, status);

            // w_subgrid_image /= plan->w_pattern
            sdp_gridder_scale_inv_array(w_subgrid_image, w_subgrid_image,
                                        w_pattern_ptr, 1, status);
        }

        if (loc == SDP_MEM_CPU) {
            degrid_cpu(plan, subgrids, w_plane, subgrid_offset_u,
                       subgrid_offset_v, subgrid_offset_w, freq0_hz, dfreq_hz,
                       start_row, end_row, uvws, start_chs, end_chs, vis,
                       status);
        } else if (loc == SDP_MEM_GPU) {
            degrid_gpu(plan, subgrids, w_plane, subgrid_offset_u,
                       subgrid_offset_v, subgrid_offset_w, freq0_hz, dfreq_hz,
                       start_row, end_row, uvws, start_chs, end_chs, vis,
                       status);
        }
    }

// Update w-plane counter.
#pragma omp atomic
    plan->num_w_planes[0] += num_w_planes;

    sdp_mem_free(w_subgrid_image);
    sdp_mem_free(subgrids);
    sdp_mem_free(last_subgrid_ptr);
    sdp_fft_free(fft);
}

void sdp_gridder_wtower_uvw_degrid_correct(sdp_GridderWtowerUVW *plan,
                                           sdp_Mem *facet, int facet_offset_l,
                                           int facet_offset_m, int w_offset,
                                           sdp_Error *status) {
    sdp_gridder_grid_correct_pswf(plan->image_size, plan->theta, plan->w_step,
                                  plan->shear_u, plan->shear_v, plan->support,
                                  plan->w_support, facet, facet_offset_l,
                                  facet_offset_m, status);
    if (sdp_mem_is_complex(facet)) {
        sdp_gridder_grid_correct_w_stack(
            plan->image_size, plan->theta, plan->w_step, plan->shear_u,
            plan->shear_v, facet, facet_offset_l, facet_offset_m, w_offset,
            false, status);
    }
}

void sdp_gridder_wtower_uvw_grid(sdp_GridderWtowerUVW *plan, const sdp_Mem *vis,
                                 const sdp_Mem *uvws, const sdp_Mem *start_chs,
                                 const sdp_Mem *end_chs, double freq0_hz,
                                 double dfreq_hz, sdp_Mem *subgrid_image,
                                 int subgrid_offset_u, int subgrid_offset_v,
                                 int subgrid_offset_w, int64_t start_row,
                                 int64_t end_row, sdp_Error *status) {
    if (*status)
        return;
    if (dfreq_hz == 0.0)
        dfreq_hz = 10; // Prevent possible divide-by-zero.
    const sdp_MemLocation loc = sdp_mem_location(vis);
    if (sdp_mem_location(subgrid_image) != loc ||
        sdp_mem_location(uvws) != loc || sdp_mem_location(start_chs) != loc ||
        sdp_mem_location(end_chs) != loc) {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("All arrays must be in the same memory space");
        return;
    }
    if (start_row < 0 || end_row < 0) {
        start_row = 0;
        end_row = sdp_mem_shape_dim(uvws, 0);
    }

    // Copy internal arrays to GPU memory as required.
    sdp_Mem *w_pattern_ptr = plan->w_pattern;
    if (loc == SDP_MEM_GPU) {
        if (!plan->w_pattern_gpu) {
            plan->w_kernel_gpu =
                sdp_mem_create_copy(plan->w_kernel, loc, status);
            plan->uv_kernel_gpu =
                sdp_mem_create_copy(plan->uv_kernel, loc, status);
            plan->w_pattern_gpu =
                sdp_mem_create_copy(plan->w_pattern, loc, status);
        }
        w_pattern_ptr = plan->w_pattern_gpu;
    }

    // Determine w-range.
    double c_min[] = {0, 0, 0}, c_max[] = {0, 0, 0};
    sdp_gridder_uvw_bounds_all(uvws, freq0_hz, dfreq_hz, start_chs, end_chs,
                               c_min, c_max, status);

    // Get subgrid at first w-plane.
    const double eta = 1e-5;
    const int first_w_plane =
        (int)floor(c_min[2] / plan->w_step - eta) - subgrid_offset_w;
    const int last_w_plane =
        (int)ceil(c_max[2] / plan->w_step + eta) - subgrid_offset_w + 1;

    // Create subgrid image and subgrids to accumulate on.
    const int64_t num_elements_sg = plan->subgrid_size * plan->subgrid_size;
    const int64_t subgrid_shape[] = {plan->subgrid_size, plan->subgrid_size};
    const int64_t subgrids_shape[] = {plan->w_support, plan->subgrid_size,
                                      plan->subgrid_size};
    sdp_Mem *w_subgrid_image =
        sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, loc, 2, subgrid_shape, status);
    sdp_Mem *subgrids =
        sdp_mem_create(sdp_mem_type(vis), loc, 3, subgrids_shape, status);
    sdp_mem_set_value(subgrids, 0, status);
    sdp_mem_set_value(w_subgrid_image, 0, status);

    // Create the iFFT plan and scratch buffer.
    sdp_Mem *fft_buffer =
        sdp_mem_create(sdp_mem_type(subgrids), loc, 2, subgrid_shape, status);
    sdp_Fft *fft = sdp_fft_create(fft_buffer, fft_buffer, 2, false, status);

    // Loop over w-planes.
    const int num_w_planes = 1 + last_w_plane - first_w_plane;
    for (int w_plane = first_w_plane; w_plane <= last_w_plane; ++w_plane) {
        if (*status)
            break;

        // Move to next w-plane.
        if (w_plane != first_w_plane) {
            // Accumulate zero-th subgrid, shift, clear upper subgrid.
            // w_subgrid_image /= plan->w_pattern
            sdp_gridder_scale_inv_array(w_subgrid_image, w_subgrid_image,
                                        w_pattern_ptr, 1, status);

            // w_subgrid_image += ifft(subgrids[0])
            sdp_mem_copy_contents(fft_buffer, subgrids, 0, 0, num_elements_sg,
                                  status);
            sdp_fft_phase(fft_buffer, status);
            sdp_fft_exec(fft, fft_buffer, fft_buffer, status);
            sdp_fft_phase(fft_buffer, status);
            sdp_gridder_accumulate_scaled_arrays(w_subgrid_image, fft_buffer, 0,
                                                 0.0, status);

            // subgrids[:-1] = subgrids[1:]
            sdp_gridder_shift_subgrids(subgrids, status);

            // subgrids[-1] = 0
            sdp_mem_clear_portion(subgrids,
                                  num_elements_sg * (plan->w_support - 1),
                                  num_elements_sg, status);
        }

        if (loc == SDP_MEM_CPU) {
            grid_cpu(plan, subgrids, w_plane, subgrid_offset_u,
                     subgrid_offset_v, subgrid_offset_w, freq0_hz, dfreq_hz,
                     start_row, end_row, uvws, start_chs, end_chs, vis, status);
        } else if (loc == SDP_MEM_GPU) {
            grid_gpu(plan, subgrids, w_plane, subgrid_offset_u,
                     subgrid_offset_v, subgrid_offset_w, freq0_hz, dfreq_hz,
                     start_row, end_row, uvws, start_chs, end_chs, vis, status);
        }
    }

    // Accumulate remaining data from subgrids.
    for (int i = 0; i < plan->w_support; ++i) {
        // Perform w_subgrid_image /= plan->w_pattern
        sdp_gridder_scale_inv_array(w_subgrid_image, w_subgrid_image,
                                    w_pattern_ptr, 1, status);

        // Perform w_subgrid_image += ifft(subgrids[i])
        sdp_mem_copy_contents(fft_buffer, subgrids, 0, num_elements_sg * i,
                              num_elements_sg, status);
        sdp_fft_phase(fft_buffer, status);
        sdp_fft_exec(fft, fft_buffer, fft_buffer, status);
        sdp_fft_phase(fft_buffer, status);
        sdp_gridder_accumulate_scaled_arrays(w_subgrid_image, fft_buffer, 0,
                                             0.0, status);
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
    sdp_gridder_accumulate_scaled_arrays(subgrid_image, w_subgrid_image,
                                         w_pattern_ptr, exponent, status);

// Update w-plane counter.
#pragma omp atomic
    plan->num_w_planes[1] += num_w_planes;

    sdp_mem_free(w_subgrid_image);
    sdp_mem_free(subgrids);
    sdp_mem_free(fft_buffer);
    sdp_fft_free(fft);
}

void sdp_gridder_wtower_uvw_grid_correct(sdp_GridderWtowerUVW *plan,
                                         sdp_Mem *facet, int facet_offset_l,
                                         int facet_offset_m, int w_offset,
                                         sdp_Error *status) {
    sdp_gridder_grid_correct_pswf(plan->image_size, plan->theta, plan->w_step,
                                  plan->shear_u, plan->shear_v, plan->support,
                                  plan->w_support, facet, facet_offset_l,
                                  facet_offset_m, status);
    if (sdp_mem_is_complex(facet)) {
        sdp_gridder_grid_correct_w_stack(
            plan->image_size, plan->theta, plan->w_step, plan->shear_u,
            plan->shear_v, facet, facet_offset_l, facet_offset_m, w_offset,
            true, status);
    }
}

void sdp_gridder_wtower_uvw_free(sdp_GridderWtowerUVW *plan) {
    if (!plan)
        return;
    sdp_mem_free(plan->uv_kernel);
    sdp_mem_free(plan->uv_kernel_gpu);
    sdp_mem_free(plan->w_kernel);
    sdp_mem_free(plan->w_kernel_gpu);
    sdp_mem_free(plan->w_pattern);
    sdp_mem_free(plan->w_pattern_gpu);
    free(plan);
}

int sdp_gridder_wtower_uvw_num_w_planes(const sdp_GridderWtowerUVW *plan,
                                        int gridding) {
    return (gridding >= 0 && gridding < 2) ? plan->num_w_planes[gridding] : 0;
}

int sdp_gridder_wtower_uvw_image_size(const sdp_GridderWtowerUVW *plan) {
    return plan->image_size;
}

int sdp_gridder_wtower_uvw_oversampling(const sdp_GridderWtowerUVW *plan) {
    return plan->oversampling;
}

double sdp_gridder_wtower_uvw_shear_u(const sdp_GridderWtowerUVW *plan) {
    return plan->shear_u;
}

double sdp_gridder_wtower_uvw_shear_v(const sdp_GridderWtowerUVW *plan) {
    return plan->shear_v;
}

int sdp_gridder_wtower_uvw_subgrid_size(const sdp_GridderWtowerUVW *plan) {
    return plan->subgrid_size;
}

int sdp_gridder_wtower_uvw_support(const sdp_GridderWtowerUVW *plan) {
    return plan->support;
}

double sdp_gridder_wtower_uvw_theta(const sdp_GridderWtowerUVW *plan) {
    return plan->theta;
}

int sdp_gridder_wtower_uvw_w_oversampling(const sdp_GridderWtowerUVW *plan) {
    return plan->w_oversampling;
}

double sdp_gridder_wtower_uvw_w_step(const sdp_GridderWtowerUVW *plan) {
    return plan->w_step;
}

int sdp_gridder_wtower_uvw_w_support(const sdp_GridderWtowerUVW *plan) {
    return plan->w_support;
}
