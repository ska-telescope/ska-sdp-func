/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>

#include "ska-sdp-func/utility/sdp_device_wrapper.h"

#define C_0 299792458.0
#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) \
    (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)


template<typename COORD_TYPE3>
__global__ void rotate_uvw(
        const int64_t num,
        const double matrix00,
        const double matrix01,
        const double matrix02,
        const double matrix10,
        const double matrix11,
        const double matrix12,
        const double matrix20,
        const double matrix21,
        const double matrix22,
        const COORD_TYPE3* uvw_in,
        COORD_TYPE3* uvw_out
)
{
    const int64_t i_uvw = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_uvw >= num) return;
    const COORD_TYPE3 uvw = uvw_in[i_uvw];
    COORD_TYPE3 uvw_rotated;
    uvw_rotated.x = matrix00 * uvw.x + matrix01 * uvw.y + matrix02 * uvw.z;
    uvw_rotated.y = matrix10 * uvw.x + matrix11 * uvw.y + matrix12 * uvw.z;
    uvw_rotated.z = matrix20 * uvw.x + matrix21 * uvw.y + matrix22 * uvw.z;
    uvw_out[i_uvw] = uvw_rotated;
}

SDP_CUDA_KERNEL(rotate_uvw<double3>)
SDP_CUDA_KERNEL(rotate_uvw<float3>)


template<typename COORD_TYPE3, typename VIS_TYPE2>
__global__ void rotate_vis(
        const int64_t num_times,
        const int64_t num_baselines,
        const int64_t num_channels,
        const int64_t num_pols,
        const double channel_start_hz,
        const double channel_step_hz,
        const double delta_l,
        const double delta_m,
        const double delta_n,
        const COORD_TYPE3* const __restrict__ uvw_metres,
        const VIS_TYPE2* vis_in,
        VIS_TYPE2* vis_out
)
{
    const int64_t i_baseline = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t i_channel  = blockDim.y * blockIdx.y + threadIdx.y;
    const int64_t i_time     = blockDim.z * blockIdx.z + threadIdx.z;
    if (num_pols > 4 ||
            i_baseline >= num_baselines ||
            i_channel >= num_channels ||
            i_time >= num_times)
    {
        return;
    }
    const COORD_TYPE3 uvw = uvw_metres[num_baselines * i_time + i_baseline];
    const double inv_wavelength =
            (channel_start_hz + i_channel * channel_step_hz) / C_0;
    const double phase = 2.0 * M_PI * inv_wavelength *
            (uvw.x * delta_l + uvw.y * delta_m + uvw.z * delta_n);
    double2 phasor;
    sincos(phase, &phasor.y, &phasor.x);
    for (int64_t i_pol = 0; i_pol < num_pols; ++i_pol)
    {
        const int64_t i_vis = INDEX_4D(
                num_times, num_baselines, num_channels, num_pols,
                i_time, i_baseline, i_channel, i_pol
        );
        const VIS_TYPE2 vis = vis_in[i_vis];
        vis_out[i_vis].x = phasor.x * vis.x - phasor.y * vis.y;
        vis_out[i_vis].y = phasor.x * vis.y + phasor.y * vis.x;
    }
}

SDP_CUDA_KERNEL(rotate_vis<double3, double2>)
SDP_CUDA_KERNEL(rotate_vis<double3, float2>)
SDP_CUDA_KERNEL(rotate_vis<float3, float2>)
