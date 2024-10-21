/* See the LICENSE file at the top-level directory of this distribution. */

#include "ska-sdp-func/math/sdp_math_macros.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"


template<typename UVW_TYPE>
__global__ void sdp_gridder_clamp_channels_single(
        sdp_MemViewGpu<const UVW_TYPE, 2> uvws,
        const int dim,
        const double freq0_hz,
        const double dfreq_hz,
        sdp_MemViewGpu<const int, 1> start_ch_in,
        sdp_MemViewGpu<const int, 1> end_ch_in,
        const double min_u,
        const double max_u,
        sdp_MemViewGpu<int, 1> start_ch_out,
        sdp_MemViewGpu<int, 1> end_ch_out
)
{
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t num_uvw = uvws.shape[0];
    if (i >= num_uvw) return;
    const double u0 = uvws(i, dim) * (freq0_hz / C_0);
    const double du = uvws(i, dim) * (dfreq_hz / C_0);
    const double rel_min_u = fabs(min_u - u0);
    const double rel_max_u = fabs(max_u - u0);
    const double eta_u = MAX(rel_min_u, rel_max_u) / 2147483645.0;
    if (fabs(du) > eta_u) // Use a safe value for eta.
    {
        const int64_t mins = (int64_t) (ceil((min_u - u0) / du));
        const int64_t maxs = (int64_t) (ceil((max_u - u0) / du));
        const int is_positive = du > 0;
        const int start_ch_ = is_positive ? (int) mins : (int) maxs;
        const int end_ch_ = is_positive ? (int) maxs : (int) mins;
        start_ch_out(i) = MAX(start_ch_in(i), start_ch_);
        end_ch_out(i) = MIN(end_ch_in(i), end_ch_);
    }
    else if (min_u > u0 || max_u <= u0)
    {
        start_ch_out(i) = 0;
        end_ch_out(i) = 0;
    }
    else
    {
        start_ch_out(i) = start_ch_in(i);
        end_ch_out(i) = end_ch_in(i);
    }
    end_ch_out(i) = MAX(end_ch_out(i), start_ch_out(i));
}


template<typename UVW_TYPE>
__global__ void sdp_gridder_clamp_channels_uv(
        sdp_MemViewGpu<const UVW_TYPE, 2> uvws,
        const double freq0_hz,
        const double dfreq_hz,
        sdp_MemViewGpu<const int, 1> start_ch_in,
        sdp_MemViewGpu<const int, 1> end_ch_in,
        const double min_u,
        const double max_u,
        const double min_v,
        const double max_v,
        sdp_MemViewGpu<int, 1> start_ch_out,
        sdp_MemViewGpu<int, 1> end_ch_out
)
{
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t num_uvw = uvws.shape[0];
    if (i >= num_uvw) return;
    const double u0 = uvws(i, 0) * (freq0_hz / C_0);
    const double du = uvws(i, 0) * (dfreq_hz / C_0);
    const double rel_min_u = fabs(min_u - u0);
    const double rel_max_u = fabs(max_u - u0);
    const double eta_u = MAX(rel_min_u, rel_max_u) / 2147483645.0;
    if (fabs(du) > eta_u) // Use a safe value for eta.
    {
        const int64_t mins = (int64_t) (ceil((min_u - u0) / du));
        const int64_t maxs = (int64_t) (ceil((max_u - u0) / du));
        const int is_positive = du > 0;
        const int start_ch_ = is_positive ? (int) mins : (int) maxs;
        const int end_ch_ = is_positive ? (int) maxs : (int) mins;
        start_ch_out(i) = MAX(start_ch_in(i), start_ch_);
        end_ch_out(i) = MIN(end_ch_in(i), end_ch_);
    }
    else if (min_u > u0 || max_u <= u0)
    {
        start_ch_out(i) = 0;
        end_ch_out(i) = 0;
    }
    else
    {
        start_ch_out(i) = start_ch_in(i);
        end_ch_out(i) = end_ch_in(i);
    }
    end_ch_out(i) = MAX(end_ch_out(i), start_ch_out(i));
    if (start_ch_out(i) >= end_ch_out(i)) return;

    const double v0 = uvws(i, 1) * (freq0_hz / C_0);
    const double dv = uvws(i, 1) * (dfreq_hz / C_0);
    const double rel_min_v = fabs(min_v - v0);
    const double rel_max_v = fabs(max_v - v0);
    const double eta_v = MAX(rel_min_v, rel_max_v) / 2147483645.0;
    if (fabs(dv) > eta_v) // Use a safe value for eta.
    {
        const int64_t mins = (int64_t) (ceil((min_v - v0) / dv));
        const int64_t maxs = (int64_t) (ceil((max_v - v0) / dv));
        const int is_positive = dv > 0;
        const int start_ch_ = is_positive ? (int) mins : (int) maxs;
        const int end_ch_ = is_positive ? (int) maxs : (int) mins;
        start_ch_out(i) = MAX(start_ch_in(i), start_ch_);
        end_ch_out(i) = MIN(end_ch_in(i), end_ch_);
    }
    else if (min_v > v0 || max_v <= v0)
    {
        start_ch_out(i) = 0;
        end_ch_out(i) = 0;
    }
    else
    {
        start_ch_out(i) = start_ch_in(i);
        end_ch_out(i) = end_ch_in(i);
    }
    end_ch_out(i) = MAX(end_ch_out(i), start_ch_out(i));
}

SDP_CUDA_KERNEL(sdp_gridder_clamp_channels_single<float>)
SDP_CUDA_KERNEL(sdp_gridder_clamp_channels_single<double>)

SDP_CUDA_KERNEL(sdp_gridder_clamp_channels_uv<float>)
SDP_CUDA_KERNEL(sdp_gridder_clamp_channels_uv<double>)
