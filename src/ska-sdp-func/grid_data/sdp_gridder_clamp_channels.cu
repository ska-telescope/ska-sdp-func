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
        sdp_MemViewGpu<int, 1> start_ch,
        sdp_MemViewGpu<int, 1> end_ch,
        const double min_u,
        const double max_u
)
{
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t num_uvw = uvws.shape[0];
    if (i >= num_uvw) return;
    const double eta = 1e-2;
    const int range_includes_zero = (min_u <= 0 && max_u > 0);
    const double u = uvws(i, dim);
    if (fabs(u) > eta * C_0 / dfreq_hz)
    {
        const double u0 = u * freq0_hz / C_0;
        const double du = u * dfreq_hz / C_0;
        const int64_t mins = (int64_t) (ceil((min_u - u0) / du));
        const int64_t maxs = (int64_t) (ceil((max_u - u0) / du));
        const int is_positive = du > 0;
        const int start_ch_ = is_positive ? (int) mins : (int) maxs;
        const int end_ch_ = is_positive ? (int) maxs : (int) mins;
        start_ch(i) = MAX(start_ch(i), start_ch_);
        end_ch(i) = MIN(end_ch(i), end_ch_);
    }
    else if (!range_includes_zero)
    {
        start_ch(i) = 0;
        end_ch(i) = 0;
    }
    end_ch(i) = MAX(end_ch(i), start_ch(i));
}


template<typename UVW_TYPE>
__global__ void sdp_gridder_clamp_channels_uv(
        sdp_MemViewGpu<const UVW_TYPE, 2> uvws,
        const double freq0_hz,
        const double dfreq_hz,
        sdp_MemViewGpu<int, 1> start_ch,
        sdp_MemViewGpu<int, 1> end_ch,
        const double min_u,
        const double max_u,
        const double min_v,
        const double max_v
)
{
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t num_uvw = uvws.shape[0];
    if (i >= num_uvw) return;
    const double eta = 1e-2;
    const double threshold = eta * C_0 / dfreq_hz;
    const int range_includes_zero[2] = {
        (min_u <= 0 && max_u > 0), (min_v <= 0 && max_v > 0)
    };
    const double u = uvws(i, 0);
    if (fabs(u) > threshold)
    {
        const double u0 = u * freq0_hz / C_0;
        const double du = u * dfreq_hz / C_0;
        const int64_t mins = (int64_t) (ceil((min_u - u0) / du));
        const int64_t maxs = (int64_t) (ceil((max_u - u0) / du));
        const int is_positive = du > 0;
        const int start_ch_ = is_positive ? (int) mins : (int) maxs;
        const int end_ch_ = is_positive ? (int) maxs : (int) mins;
        start_ch(i) = MAX(start_ch(i), start_ch_);
        end_ch(i) = MIN(end_ch(i), end_ch_);
    }
    else if (!range_includes_zero[0])
    {
        start_ch(i) = 0;
        end_ch(i) = 0;
    }
    const double v = uvws(i, 1);
    if (fabs(v) > threshold)
    {
        const double v0 = v * freq0_hz / C_0;
        const double dv = v * dfreq_hz / C_0;
        const int64_t mins = (int64_t) (ceil((min_v - v0) / dv));
        const int64_t maxs = (int64_t) (ceil((max_v - v0) / dv));
        const int is_positive = dv > 0;
        const int start_ch_ = is_positive ? (int) mins : (int) maxs;
        const int end_ch_ = is_positive ? (int) maxs : (int) mins;
        start_ch(i) = MAX(start_ch(i), start_ch_);
        end_ch(i) = MIN(end_ch(i), end_ch_);
    }
    else if (!range_includes_zero[1])
    {
        start_ch(i) = 0;
        end_ch(i) = 0;
    }
    end_ch(i) = MAX(end_ch(i), start_ch(i));
}

SDP_CUDA_KERNEL(sdp_gridder_clamp_channels_single<float>)
SDP_CUDA_KERNEL(sdp_gridder_clamp_channels_single<double>)

SDP_CUDA_KERNEL(sdp_gridder_clamp_channels_uv<float>)
SDP_CUDA_KERNEL(sdp_gridder_clamp_channels_uv<double>)
