/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>
#include <cstdlib>

#include "ska-sdp-func/fourier_transforms/sdp_fft.h"
#include "ska-sdp-func/fourier_transforms/sdp_pswf.h"
#include "ska-sdp-func/grid_data/sdp_gridder_utils.h"
#include "ska-sdp-func/grid_data/sdp_gridder_wtower_uvw.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"

using std::complex;

#define C_0 299792458.0

struct sdp_GridderWtowerUVW
{
    int image_size;
    int subgrid_size;
    double theta;
    double shear_u;
    double shear_v;
    int support;
    int vr_size;
    int oversampling;
    double w_step;
    int w_support;
    int w_oversampling;
    sdp_Mem* pswf;
    sdp_Mem* pswf_n;
    sdp_Mem* uv_kernel;
    sdp_Mem* w_kernel;
    sdp_Mem* w_pattern;
};

// Begin anonymous namespace for file-local functions.
namespace {

// Local function to degrid over selected channels.
template<typename VIS_TYPE>
void degrid_channels(
        const sdp_GridderWtowerUVW* plan,
        int64_t row_index,
        int start_ch,
        int end_ch,
        double uvw0[3],
        double duvw[3],
        const sdp_Mem* subgrids,
        sdp_Mem* vis,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_MemViewCpu<VIS_TYPE, 2> vis_;
    sdp_MemViewCpu<const VIS_TYPE, 3> subgrids_;
    sdp_MemViewCpu<const double, 2> uv_k_, w_k_;
    sdp_mem_check_and_view(vis, &vis_, status);
    sdp_mem_check_and_view(subgrids, &subgrids_, status);
    sdp_mem_check_and_view(plan->uv_kernel, &uv_k_, status);
    sdp_mem_check_and_view(plan->w_kernel, &w_k_, status);
    const int half_subgrid = plan->subgrid_size / 2;
    const double half_vr_m1 = (plan->vr_size - 1) / 2.0;
    const int half_vr = plan->vr_size / 2;
    const int oversample = plan->oversampling;
    const int w_oversample = plan->w_oversampling;
    const double theta = plan->theta;
    double u = uvw0[0], v = uvw0[1], w = uvw0[2];
    if (*status) return;

    // Loop over selected channels.
    for (int c = start_ch; c < end_ch; c++)
    {
        // Determine top-left corner of grid region
        // centered approximately on visibility.
        const int iu0 = int(round(theta * u - half_vr_m1)) + half_subgrid;
        const int iv0 = int(round(theta * v - half_vr_m1)) + half_subgrid;
        const int iu_shift = iu0 + half_vr - half_subgrid;
        const int iv_shift = iv0 + half_vr - half_subgrid;

        // Determine which kernel to use.
        int u_offset = int(round((u * theta - iu_shift + 1) * oversample));
        int v_offset = int(round((v * theta - iv_shift + 1) * oversample));
        int w_offset = int(round((w / plan->w_step + 1) * w_oversample));

        // Cater for the negative indexing which is allowed in Python!
        if (u_offset < 0) u_offset += oversample + 1;
        if (v_offset < 0) v_offset += oversample + 1;
        if (w_offset < 0) w_offset += w_oversample + 1;

        // Degrid visibility.
        VIS_TYPE local_vis = (VIS_TYPE) 0;
        for (int iw = 0; iw < plan->w_support; ++iw)
        {
            const double kernel_w = w_k_(w_offset, iw);
            for (int iu = 0; iu < plan->support; ++iu)
            {
                const double kernel_wu = kernel_w * uv_k_(u_offset, iu);
                for (int iv = 0; iv < plan->support; ++iv)
                {
                    const double kernel_wuv = kernel_wu * uv_k_(v_offset, iv);
                    int ix_u = iu0 + iu;
                    int ix_v = iv0 + iv;
                    if (ix_u < 0) ix_u += plan->subgrid_size;
                    if (ix_v < 0) ix_v += plan->subgrid_size;
                    local_vis += ((VIS_TYPE) kernel_wuv *
                            subgrids_(iw, ix_u, ix_v)
                    );
                }
            }
        }
        vis_(row_index, c) = local_vis;

        // Next point.
        u += duvw[0];
        v += duvw[1];
        w += duvw[2];
    }
}


// Local function to do the degridding.
template<typename UVW_TYPE, typename VIS_TYPE>
void degrid(
        const sdp_GridderWtowerUVW* plan,
        const sdp_Mem* subgrids,
        int w_plane,
        int subgrid_offset_u,
        int subgrid_offset_v,
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

    // Loop over rows. Each row contains visibilities for all channels.
    const int64_t num_uvw = sdp_mem_shape_dim(uvws, 0);
    sdp_MemViewCpu<const UVW_TYPE, 2> uvws_;
    sdp_MemViewCpu<const int, 1> start_chs_, end_chs_;
    sdp_mem_check_and_view(uvws, &uvws_, status);
    sdp_mem_check_and_view(start_chs, &start_chs_, status);
    sdp_mem_check_and_view(end_chs, &end_chs_, status);
    if (*status) return;
    #pragma omp parallel for
    for (int64_t i = 0; i < num_uvw; ++i)
    {
        // Skip if there's no visibility to degrid.
        int start_ch = start_chs_(i), end_ch = end_chs_(i);
        if (start_ch >= end_ch)
            continue;

        // Select only visibilities on this w-plane
        // (inlined from clamp_channels).
        const UVW_TYPE uvw[] = {uvws_(i, 0), uvws_(i, 1), uvws_(i, 2)};
        const double w0 = freq0_hz * uvw[2] / C_0;
        const double dw = dfreq_hz * uvw[2] / C_0;
        const double _min = (w_plane - 1) * plan->w_step;
        const double _max = w_plane * plan->w_step;
        const double eta = 1e-3;
        if (w0 > eta)
        {
            start_ch = std::max(start_ch, int(ceil((_min - w0) / dw)));
            end_ch = std::min(end_ch, int(ceil((_max - w0) / dw)));
        }
        else if (w0 < -eta)
        {
            start_ch = std::max(start_ch, int(ceil((_max - w0) / dw)));
            end_ch = std::min(end_ch, int(ceil((_min - w0) / dw)));
        }
        else if (_min > 0 or _max <= 0)
        {
            continue;
        }
        if (start_ch >= end_ch)
            continue;

        // Scale + shift UVWs.
        const double s_uvw0 = (freq0_hz + start_ch * dfreq_hz) / C_0;
        const double s_duvw = dfreq_hz / C_0;
        double uvw0[] = {uvw[0] * s_uvw0, uvw[1] * s_uvw0, uvw[2] * s_uvw0};
        double duvw[] = {uvw[0] * s_duvw, uvw[1] * s_duvw, uvw[2] * s_duvw};
        uvw0[0] -= subgrid_offset_u / plan->theta;
        uvw0[1] -= subgrid_offset_v / plan->theta;
        uvw0[2] -= w_plane * plan->w_step;

        // Degrid visibilities over all selected channels.
        degrid_channels<VIS_TYPE>(
                plan, i, start_ch, end_ch, uvw0, duvw, subgrids, vis, status
        );
    }
}


// Local function to grid over selected channels.
template<typename VIS_TYPE>
void grid_channels(
        const sdp_GridderWtowerUVW* plan,
        int64_t row_index,
        int start_ch,
        int end_ch,
        double uvw0[3],
        double duvw[3],
        sdp_Mem* subgrids,
        const sdp_Mem* vis,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_MemViewCpu<const VIS_TYPE, 2> vis_;
    sdp_MemViewCpu<VIS_TYPE, 3> subgrids_;
    sdp_MemViewCpu<const double, 2> uv_k_, w_k_;
    sdp_mem_check_and_view(vis, &vis_, status);
    sdp_mem_check_and_view(subgrids, &subgrids_, status);
    sdp_mem_check_and_view(plan->uv_kernel, &uv_k_, status);
    sdp_mem_check_and_view(plan->w_kernel, &w_k_, status);
    const int half_subgrid = plan->subgrid_size / 2;
    const double half_vr_m1 = (plan->vr_size - 1) / 2.0;
    const int half_vr = plan->vr_size / 2;
    const int oversample = plan->oversampling;
    const int w_oversample = plan->w_oversampling;
    const double theta = plan->theta;
    double u = uvw0[0], v = uvw0[1], w = uvw0[2];

    // Loop over selected channels.
    for (int c = start_ch; c < end_ch; c++)
    {
        // Determine top-left corner of grid region
        // centered approximately on visibility.
        const int iu0 = int(round(theta * u - half_vr_m1)) + half_subgrid;
        const int iv0 = int(round(theta * v - half_vr_m1)) + half_subgrid;
        const int iu_shift = iu0 + half_vr - half_subgrid;
        const int iv_shift = iv0 + half_vr - half_subgrid;

        // Determine which kernel to use.
        int u_offset = int(round((u * theta - iu_shift + 1) * oversample));
        int v_offset = int(round((v * theta - iv_shift + 1) * oversample));
        int w_offset = int(round((w / plan->w_step + 1) * w_oversample));

        // Cater for the negative indexing which is allowed in Python!
        if (u_offset < 0) u_offset += oversample + 1;
        if (v_offset < 0) v_offset += oversample + 1;
        if (w_offset < 0) w_offset += w_oversample + 1;

        // Grid visibility.
        VIS_TYPE local_vis = vis_(row_index, c);
        for (int iw = 0; iw < plan->w_support; ++iw)
        {
            const double kernel_w = w_k_(w_offset, iw);
            for (int iu = 0; iu < plan->support; ++iu)
            {
                const double kernel_wu = kernel_w * uv_k_(u_offset, iu);
                for (int iv = 0; iv < plan->support; ++iv)
                {
                    const double kernel_wuv = kernel_wu * uv_k_(v_offset, iv);
                    int ix_u = iu0 + iu;
                    int ix_v = iv0 + iv;
                    if (ix_u < 0) ix_u += plan->subgrid_size;
                    if (ix_v < 0) ix_v += plan->subgrid_size;
                    subgrids_(iw, ix_u, ix_v) += ((VIS_TYPE) kernel_wuv *
                            local_vis
                    );
                }
            }
        }

        // Next point.
        u += duvw[0];
        v += duvw[1];
        w += duvw[2];
    }
}


// Local function to do the gridding.
template<typename UVW_TYPE, typename VIS_TYPE>
void grid(
        const sdp_GridderWtowerUVW* plan,
        sdp_Mem* subgrids,
        int w_plane,
        int subgrid_offset_u,
        int subgrid_offset_v,
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

    // Loop over rows. Each row contains visibilities for all channels.
    const int64_t num_uvw = sdp_mem_shape_dim(uvws, 0);
    sdp_MemViewCpu<const UVW_TYPE, 2> uvws_;
    sdp_MemViewCpu<const int, 1> start_chs_, end_chs_;
    sdp_mem_check_and_view(uvws, &uvws_, status);
    sdp_mem_check_and_view(start_chs, &start_chs_, status);
    sdp_mem_check_and_view(end_chs, &end_chs_, status);
    if (*status) return;
    for (int64_t i = 0; i < num_uvw; ++i)
    {
        // Skip if there's no visibility to degrid.
        int start_ch = start_chs_(i), end_ch = end_chs_(i);
        if (start_ch >= end_ch)
            continue;

        // Select only visibilities on this w-plane
        // (inlined from clamp_channels).
        const UVW_TYPE uvw[] = {uvws_(i, 0), uvws_(i, 1), uvws_(i, 2)};
        const double w0 = freq0_hz * uvw[2] / C_0;
        const double dw = dfreq_hz * uvw[2] / C_0;
        const double _min = (w_plane - 1) * plan->w_step;
        const double _max = w_plane * plan->w_step;
        const double eta = 1e-3;
        if (w0 > eta)
        {
            start_ch = std::max(start_ch, int(ceil((_min - w0) / dw)));
            end_ch = std::min(end_ch, int(ceil((_max - w0) / dw)));
        }
        else if (w0 < -eta)
        {
            start_ch = std::max(start_ch, int(ceil((_max - w0) / dw)));
            end_ch = std::min(end_ch, int(ceil((_min - w0) / dw)));
        }
        else if (_min > 0 or _max <= 0)
        {
            continue;
        }
        if (start_ch >= end_ch)
            continue;

        // Scale + shift UVWs.
        const double s_uvw0 = (freq0_hz + start_ch * dfreq_hz) / C_0;
        const double s_duvw = dfreq_hz / C_0;
        double uvw0[] = {uvw[0] * s_uvw0, uvw[1] * s_uvw0, uvw[2] * s_uvw0};
        double duvw[] = {uvw[0] * s_duvw, uvw[1] * s_duvw, uvw[2] * s_duvw};
        uvw0[0] -= subgrid_offset_u / plan->theta;
        uvw0[1] -= subgrid_offset_v / plan->theta;
        uvw0[2] -= w_plane * plan->w_step;

        // Grid visibilities over all selected channels.
        grid_channels<VIS_TYPE>(
                plan, i, start_ch, end_ch, uvw0, duvw, subgrids, vis, status
        );
    }
}


// Local function to apply grid correction.
template<typename T>
void grid_corr(
        const sdp_GridderWtowerUVW* plan,
        sdp_Mem* facet,
        const double* pswf_l,
        const double* pswf_m,
        const double* pswf_n,
        sdp_Error* status
)
{
    if (*status) return;

    // Apply portion of shifted PSWF to facet.
    sdp_MemViewCpu<T, 2> facet_;
    sdp_mem_check_and_view(facet, &facet_, status);
    if (*status) return;
    const int64_t num_l = sdp_mem_shape_dim(facet, 0);
    const int64_t num_m = sdp_mem_shape_dim(facet, 1);
    for (int64_t il = 0; il < num_l; ++il)
    {
        const int64_t pl = il + (plan->image_size / 2 - num_l / 2);
        for (int64_t im = 0; im < num_m; ++im)
        {
            const int64_t pm = im + (plan->image_size / 2 - num_m / 2);
            facet_(il, im) /= (T) (
                pswf_l[pl] * pswf_m[pm] * pswf_n[pl * plan->image_size + pm]
            );
        }
    }
}


// Local function to scale every element in an array by a fixed value.
template<typename T>
void scale_real(sdp_Mem* data, double value, sdp_Error* status)
{
    if (*status) return;
    const int64_t num_elements = sdp_mem_num_elements(data);
    T* data_ = (T*) sdp_mem_data(data);
    for (int64_t i = 0; i < num_elements; ++i)
        data_[i] *= value;
}

} // End anonymous namespace for file-local functions.


sdp_GridderWtowerUVW* sdp_gridder_wtower_uvw_create(
        int image_size,
        int subgrid_size,
        double theta,
        double shear_u,
        double shear_v,
        int support,
        int oversampling,
        double w_step,
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
    plan->shear_u = shear_u;
    plan->shear_v = shear_v;
    plan->support = support;
    plan->vr_size = support; // vr_size is the same as support.
    plan->oversampling = oversampling;
    plan->w_step = w_step;
    plan->w_support = w_support;
    plan->w_oversampling = w_oversampling;

    // Generate pswf (1D).
    const int64_t pswf_shape[] = {image_size};
    plan->pswf = sdp_mem_create(
            SDP_MEM_DOUBLE, SDP_MEM_CPU, 1, pswf_shape, status
    );
    sdp_generate_pswf(0, support * (M_PI / 2), plan->pswf, status);
    if (image_size % 2 == 0) ((double*) sdp_mem_data(plan->pswf))[0] = 1e-15;

    // Generate pswf_n (2D).
    const int64_t pswf_n_shape[] = {image_size, image_size};
    plan->pswf_n = sdp_mem_create(
            SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, pswf_n_shape, status
    );
    sdp_Mem* ns = sdp_mem_create(
            SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, pswf_n_shape, status
    );
    sdp_gridder_image_to_lmn(
            image_size, theta, shear_u, shear_v, 0, 0, ns, status
    );
    scale_real<double>(ns, 2.0 * w_step, status);
    sdp_generate_pswf_at_x(0, w_support * (M_PI / 2), ns, plan->pswf_n, status);
    sdp_mem_free(ns);

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
        const sdp_GridderWtowerUVW* plan,
        const sdp_Mem* subgrid_image,
        int subgrid_offset_u,
        int subgrid_offset_v,
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

    // Determine w-range.
    double c_min[] = {0, 0, 0}, c_max[] = {0, 0, 0};
    sdp_gridder_uvw_bounds_all(
            uvws, freq0_hz, dfreq_hz, start_chs, end_chs, c_min, c_max, status
    );
    const int first_w_plane = (int) floor(c_min[2] / plan->w_step);
    const int last_w_plane = (int) ceil(c_max[2] / plan->w_step);

    // First w-plane we need to generate is (support / 2) below the first one
    // with visibilities.
    const int64_t subgrid_shape[] = {plan->subgrid_size, plan->subgrid_size};
    sdp_Mem* w_subgrid_image = sdp_mem_create(
            sdp_mem_type(vis), SDP_MEM_CPU, 2, subgrid_shape, status
    );

    // Perform w_subgrid_image = subgrid_image /
    //             plan->w_pattern ** (first_w_plane - plan->w_support // 2)
    const double exponent = first_w_plane - plan->w_support / 2;
    sdp_gridder_scale_inv_array(
            w_subgrid_image, subgrid_image, plan->w_pattern, exponent, status
    );

    // Create the FFT plan and scratch buffer.
    sdp_Mem* fft_buffer = sdp_mem_create(
            sdp_mem_type(vis), SDP_MEM_CPU, 2, subgrid_shape, status
    );
    sdp_Fft* fft = sdp_fft_create(fft_buffer, fft_buffer, 2, true, status);

    // Fill sub-grid stack, dimensions (w_support, subgrid_size, subgrid_size).
    const int64_t num_elements_sg = plan->subgrid_size * plan->subgrid_size;
    const int64_t subgrids_shape[] = {
        plan->w_support, plan->subgrid_size, plan->subgrid_size
    };
    sdp_Mem* subgrids = sdp_mem_create(
            sdp_mem_type(vis), SDP_MEM_CPU, 3, subgrids_shape, status
    );
    for (int i = 0; i < plan->w_support; ++i)
    {
        // Perform subgrids[i] = fft(w_subgrid_image)
        sdp_mem_copy_contents(
                fft_buffer, w_subgrid_image, 0, 0, num_elements_sg, status
        );
        sdp_fft_phase(fft_buffer, status);
        sdp_fft_exec(fft, fft_buffer, fft_buffer, status);
        sdp_fft_phase(fft_buffer, status);
        sdp_mem_copy_contents(
                subgrids,
                fft_buffer,
                num_elements_sg * i,
                0,
                num_elements_sg,
                status
        );

        // Perform w_subgrid_image /= plan->w_pattern
        sdp_gridder_scale_inv_array(
                w_subgrid_image, w_subgrid_image, plan->w_pattern, 1, status
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
            sdp_mem_copy_contents(
                    fft_buffer, w_subgrid_image, 0, 0, num_elements_sg, status
            );
            sdp_fft_phase(fft_buffer, status);
            sdp_fft_exec(fft, fft_buffer, fft_buffer, status);
            sdp_fft_phase(fft_buffer, status);
            sdp_mem_copy_contents(
                    subgrids,
                    fft_buffer,
                    num_elements_sg * (plan->w_support - 1),
                    0,
                    num_elements_sg,
                    status
            );

            // w_subgrid_image /= plan->w_pattern
            sdp_gridder_scale_inv_array(
                    w_subgrid_image, w_subgrid_image, plan->w_pattern, 1, status
            );
        }

        if (sdp_mem_type(uvws) == SDP_MEM_DOUBLE &&
                sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE)
        {
            degrid<double, complex<double> >(
                    plan, subgrids, w_plane, subgrid_offset_u, subgrid_offset_v,
                    freq0_hz, dfreq_hz, uvws, start_chs, end_chs, vis, status
            );
        }
        else if (sdp_mem_type(uvws) == SDP_MEM_FLOAT &&
                sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT)
        {
            degrid<float, complex<float> >(
                    plan, subgrids, w_plane, subgrid_offset_u, subgrid_offset_v,
                    freq0_hz, dfreq_hz, uvws, start_chs, end_chs, vis, status
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            break;
        }
    }

    sdp_mem_free(w_subgrid_image);
    sdp_mem_free(subgrids);
    sdp_mem_free(fft_buffer);
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
    sdp_Mem* pswf_n = sdp_mem_create_copy(plan->pswf_n, SDP_MEM_CPU, status);
    if (*status) return;
    const double* pswf = (const double*) sdp_mem_data_const(plan->pswf);
    const double* pswf_n0 = (const double*) sdp_mem_data_const(plan->pswf_n);
    double* pswf_l_ = (double*) sdp_mem_data(pswf_l);
    double* pswf_m_ = (double*) sdp_mem_data(pswf_m);
    double* pswf_n_ = (double*) sdp_mem_data(pswf_n);
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
    for (int64_t j = 0; j < pswf_size; ++j)
    {
        for (int64_t i = 0; i < pswf_size; ++i)
        {
            int64_t il = j + facet_offset_l;
            int64_t im = i + facet_offset_m;
            if (il >= pswf_size) il -= pswf_size;
            if (im >= pswf_size) im -= pswf_size;
            if (il < 0) il += pswf_size;
            if (im < 0) im += pswf_size;
            pswf_n_[j * pswf_size + i] = pswf_n0[il * pswf_size + im];
        }
    }

    // Apply grid correction for the appropriate data type.
    switch (sdp_mem_type(facet))
    {
    case SDP_MEM_DOUBLE:
        grid_corr<double>(
                plan, facet, pswf_l_, pswf_m_, pswf_n_, status
        );
        break;
    case SDP_MEM_COMPLEX_DOUBLE:
        grid_corr<complex<double> >(
                plan, facet, pswf_l_, pswf_m_, pswf_n_, status
        );
        break;
    case SDP_MEM_FLOAT:
        grid_corr<float>(
                plan, facet, pswf_l_, pswf_m_, pswf_n_, status
        );
        break;
    case SDP_MEM_COMPLEX_FLOAT:
        grid_corr<complex<float> >(
                plan, facet, pswf_l_, pswf_m_, pswf_n_, status
        );
        break;
    default:
        *status = SDP_ERR_DATA_TYPE;
        break;
    }
    sdp_mem_free(pswf_l);
    sdp_mem_free(pswf_m);
    sdp_mem_free(pswf_n);
}


void sdp_gridder_wtower_uvw_grid(
        const sdp_GridderWtowerUVW* plan,
        const sdp_Mem* vis,
        const sdp_Mem* uvws,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        double freq0_hz,
        double dfreq_hz,
        sdp_Mem* subgrid_image,
        int subgrid_offset_u,
        int subgrid_offset_v,
        sdp_Error* status
)
{
    if (*status) return;

    // Determine w-range.
    double c_min[] = {0, 0, 0}, c_max[] = {0, 0, 0};
    sdp_gridder_uvw_bounds_all(
            uvws, freq0_hz, dfreq_hz, start_chs, end_chs, c_min, c_max, status
    );
    const int first_w_plane = (int) floor(c_min[2] / plan->w_step);
    const int last_w_plane = (int) ceil(c_max[2] / plan->w_step);

    // Create subgrid image and subgrids to accumulate on.
    const int64_t num_elements_sg = plan->subgrid_size * plan->subgrid_size;
    const int64_t subgrid_shape[] = {plan->subgrid_size, plan->subgrid_size};
    const int64_t subgrids_shape[] = {
        plan->w_support, plan->subgrid_size, plan->subgrid_size
    };
    sdp_Mem* w_subgrid_image = sdp_mem_create(
            sdp_mem_type(vis), SDP_MEM_CPU, 2, subgrid_shape, status
    );
    sdp_Mem* subgrids = sdp_mem_create(
            sdp_mem_type(vis), SDP_MEM_CPU, 3, subgrids_shape, status
    );
    sdp_mem_clear_contents(subgrids, status);
    sdp_mem_clear_contents(w_subgrid_image, status);

    // Create the iFFT plan and scratch buffer.
    sdp_Mem* fft_buffer = sdp_mem_create(
            sdp_mem_type(vis), SDP_MEM_CPU, 2, subgrid_shape, status
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
                    w_subgrid_image, w_subgrid_image, plan->w_pattern, 1, status
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

        if (sdp_mem_type(uvws) == SDP_MEM_DOUBLE &&
                sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE)
        {
            grid<double, complex<double> >(
                    plan, subgrids, w_plane, subgrid_offset_u, subgrid_offset_v,
                    freq0_hz, dfreq_hz, uvws, start_chs, end_chs, vis, status
            );
        }
        else if (sdp_mem_type(uvws) == SDP_MEM_FLOAT &&
                sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT)
        {
            grid<float, complex<float> >(
                    plan, subgrids, w_plane, subgrid_offset_u, subgrid_offset_v,
                    freq0_hz, dfreq_hz, uvws, start_chs, end_chs, vis, status
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            break;
        }
    }

    // Accumulate remaining data from subgrids.
    for (int i = 0; i < plan->w_support; ++i)
    {
        // Perform w_subgrid_image /= plan->w_pattern
        sdp_gridder_scale_inv_array(
                w_subgrid_image, w_subgrid_image, plan->w_pattern, 1, status
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
    double exponent = last_w_plane + plan->w_support / 2 - 1;
    sdp_gridder_accumulate_scaled_arrays(
            subgrid_image, w_subgrid_image, plan->w_pattern, exponent, status
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
    sdp_mem_free(plan->pswf);
    sdp_mem_free(plan->pswf_n);
    sdp_mem_free(plan->uv_kernel);
    sdp_mem_free(plan->w_kernel);
    sdp_mem_free(plan->w_pattern);
    free(plan);
}
