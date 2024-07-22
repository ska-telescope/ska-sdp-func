/* See the LICENSE file at the top-level directory of this distribution. */

#include <thrust/complex.h>

#include "ska-sdp-func/grid_data/sdp_gridder_clamp_channels.h"
#include "ska-sdp-func/utility/sdp_cuda_atomics.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"

using thrust::complex;


template<typename UVW_TYPE, typename VIS_TYPE>
__global__ void sdp_gridder_wtower_degrid(
        const VIS_TYPE* const __restrict__ subgrids, // internal data
        const int w_plane,
        const int subgrid_offset_u,
        const int subgrid_offset_v,
        const int subgrid_offset_w,
        const double freq0_hz,
        const double dfreq_hz,
        const sdp_MemViewGpu<const UVW_TYPE, 2> uvws, // external data
        const sdp_MemViewGpu<const int, 1> start_chs, // external data
        const sdp_MemViewGpu<const int, 1> end_chs, // external data
        const double* const __restrict__ uv_kernel, // internal data
        const double* const __restrict__ w_kernel, // internal data
        const int subgrid_size,
        const int vr_size,
        const int support,
        const int w_support,
        const int oversample,
        const int w_oversample,
        const double theta,
        const double w_step,
        sdp_MemViewGpu<VIS_TYPE, 2> vis // external data
)
{
    const int64_t i_row = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t num_uvw = uvws.shape[0];
    if (i_row >= num_uvw)
        return;

    // Each row contains visibilities for all channels.
    // Skip if there's no visibility to degrid.
    int64_t start_ch = start_chs(i_row), end_ch = end_chs(i_row);
    if (start_ch >= end_ch)
        return;

    // Select only visibilities on this w-plane.
    const UVW_TYPE uvw[] = {uvws(i_row, 0), uvws(i_row, 1), uvws(i_row, 2)};
    const double min_w = (w_plane + subgrid_offset_w - 1) * w_step;
    const double max_w = (w_plane + subgrid_offset_w) * w_step;
    sdp_gridder_clamp_channels_inline(
            uvw[2], freq0_hz, dfreq_hz, &start_ch, &end_ch, min_w, max_w
    );
    if (start_ch >= end_ch)
        return;

    // Scale + shift UVWs.
    const double s_uvw0 = freq0_hz / C_0, s_duvw = dfreq_hz / C_0;
    double uvw0[] = {uvw[0] * s_uvw0, uvw[1] * s_uvw0, uvw[2] * s_uvw0};
    double duvw[] = {uvw[0] * s_duvw, uvw[1] * s_duvw, uvw[2] * s_duvw};
    uvw0[0] -= subgrid_offset_u / theta;
    uvw0[1] -= subgrid_offset_v / theta;
    uvw0[2] -= ((subgrid_offset_w + w_plane) * w_step);

    const int64_t subgrid_square = subgrid_size * subgrid_size;
    const int half_subgrid = subgrid_size / 2;
    const double half_vr_m1 = (vr_size - 1) / 2.0;
    const int half_vr = vr_size / 2;

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
        int w_off = int(round((w / w_step + 1) * w_oversample));

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
                for (int iv = 0; iv < support; ++iv)
                {
                    const double kern_wuv = kern_wu * uv_kernel[v_off + iv];
                    int ix_u = iu0 + iu;
                    int ix_v = iv0 + iv;
                    if (ix_u < 0) ix_u += subgrid_size;
                    if (ix_v < 0) ix_v += subgrid_size;
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


template<typename UVW_TYPE, typename VIS_TYPE>
__global__ void sdp_gridder_wtower_grid(
        VIS_TYPE* __restrict__ subgrids, // internal data
        const int w_plane,
        const int subgrid_offset_u,
        const int subgrid_offset_v,
        const int subgrid_offset_w,
        const double freq0_hz,
        const double dfreq_hz,
        const sdp_MemViewGpu<const UVW_TYPE, 2> uvws, // external data
        const sdp_MemViewGpu<const int, 1> start_chs, // external data
        const sdp_MemViewGpu<const int, 1> end_chs, // external data
        const double* const __restrict__ uv_kernel, // internal data
        const double* const __restrict__ w_kernel, // internal data
        const int subgrid_size,
        const int vr_size,
        const int support,
        const int w_support,
        const int oversample,
        const int w_oversample,
        const double theta,
        const double w_step,
        const sdp_MemViewGpu<const complex<VIS_TYPE>, 2> vis // external data
)
{
    const int64_t i_row = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t num_uvw = uvws.shape[0];
    if (i_row >= num_uvw)
        return;

    // Each row contains visibilities for all channels.
    // Skip if there's no visibility to grid.
    int64_t start_ch = start_chs(i_row), end_ch = end_chs(i_row);
    if (start_ch >= end_ch)
        return;

    // Select only visibilities on this w-plane.
    const UVW_TYPE uvw[] = {uvws(i_row, 0), uvws(i_row, 1), uvws(i_row, 2)};
    const double min_w = (w_plane + subgrid_offset_w - 1) * w_step;
    const double max_w = (w_plane + subgrid_offset_w) * w_step;
    sdp_gridder_clamp_channels_inline(
            uvw[2], freq0_hz, dfreq_hz, &start_ch, &end_ch, min_w, max_w
    );
    if (start_ch >= end_ch)
        return;

    // Scale + shift UVWs.
    const double s_uvw0 = freq0_hz / C_0, s_duvw = dfreq_hz / C_0;
    double uvw0[] = {uvw[0] * s_uvw0, uvw[1] * s_uvw0, uvw[2] * s_uvw0};
    double duvw[] = {uvw[0] * s_duvw, uvw[1] * s_duvw, uvw[2] * s_duvw};
    uvw0[0] -= subgrid_offset_u / theta;
    uvw0[1] -= subgrid_offset_v / theta;
    uvw0[2] -= ((subgrid_offset_w + w_plane) * w_step);

    const int64_t subgrid_square = subgrid_size * subgrid_size;
    const int half_subgrid = subgrid_size / 2;
    const double half_vr_m1 = (vr_size - 1) / 2.0;
    const int half_vr = vr_size / 2;

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
        int w_off = int(round((w / w_step + 1) * w_oversample));

        // Cater for the negative indexing which is allowed in Python!
        if (u_off < 0) u_off += oversample + 1;
        if (v_off < 0) v_off += oversample + 1;
        if (w_off < 0) w_off += w_oversample + 1;
        u_off *= support;
        v_off *= support;
        w_off *= w_support;

        // Grid visibility.
        const complex<double> local_vis = vis(i_row, c);
        for (int iw = 0; iw < w_support; ++iw)
        {
            const double kern_w = w_kernel[w_off + iw];
            for (int iu = 0; iu < support; ++iu)
            {
                const double kern_wu = kern_w * uv_kernel[u_off + iu];
                for (int iv = 0; iv < support; ++iv)
                {
                    const double kern_wuv = kern_wu * uv_kernel[v_off + iv];
                    int ix_u = iu0 + iu;
                    int ix_v = iv0 + iv;
                    if (ix_u < 0) ix_u += subgrid_size;
                    if (ix_v < 0) ix_v += subgrid_size;
                    const int64_t idx = 2 * (
                        iw * subgrid_square + ix_u * subgrid_size + ix_v
                    );
                    const complex<double> grid_val = (
                        (complex<double>) kern_wuv * local_vis
                    );
                    // The atomic adds will be very slow.
                    sdp_atomic_add(&subgrids[idx],     grid_val.real());
                    sdp_atomic_add(&subgrids[idx + 1], grid_val.imag());
                }
            }
        }
    }
}


SDP_CUDA_KERNEL(sdp_gridder_wtower_degrid<double, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_wtower_degrid<float, complex<float> >)
SDP_CUDA_KERNEL(sdp_gridder_wtower_grid<double, double>)
SDP_CUDA_KERNEL(sdp_gridder_wtower_grid<float, float>)
