/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>

#include "ska-sdp-func/utility/sdp_device_wrapper.h"

#define C_0 299792458.0
#define INDEX_2D(N2, N1, I2, I1)                 (N1 * I2 + I1)
#define INDEX_3D(N3, N2, N1, I3, I2, I1)         (N1 * (N2 * I3 + I2) + I1)
#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)

template<
        typename DIR_TYPE3,
        typename FLUX_TYPE2,
        typename UVW_TYPE3,
        typename VIS_TYPE2
>
__global__ void dft_point_v00(
        const int num_components,
        const int num_pols,
        const int num_channels,
        const int num_baselines,
        const int num_times,
        const DIR_TYPE3 *const __restrict__ source_directions,
        const FLUX_TYPE2 *const __restrict__ source_fluxes,
        const UVW_TYPE3 *const __restrict__ uvw_lambda,
        VIS_TYPE2 *__restrict__ vis
)
{
    // Local (per-thread) visibility. Allow up to 4 polarisations.
    VIS_TYPE2 vis_local[4];
    vis_local[0].x = vis_local[1].x = vis_local[2].x = vis_local[3].x = 0;
    vis_local[0].y = vis_local[1].y = vis_local[2].y = vis_local[3].y = 0;

    // Get indices of the output array this thread is working on.
    const int i_baseline = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_channel  = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_time     = blockDim.z * blockIdx.z + threadIdx.z;

    // Bounds check.
    if (num_pols > 4 ||
            i_baseline >= num_baselines ||
            i_channel >= num_channels ||
            i_time >= num_times)
    {
        return;
    }

    // Load uvw-coordinates.
    const unsigned int i_uvw = INDEX_3D(
            num_times, num_baselines, num_channels,
            i_time, i_baseline, i_channel);
    const UVW_TYPE3 uvw = uvw_lambda[i_uvw];

    // Loop over components and calculate phase for each.
    for (int i_component = 0; i_component < num_components; ++i_component)
    {
        double2 phasor;
        const DIR_TYPE3 dir = source_directions[i_component];
        const double phase = -2.0 * M_PI * (
                dir.x * uvw.x + dir.y * uvw.y + dir.z * uvw.z);
        sincos(phase, &phasor.y, &phasor.x);

        // Multiply by flux in each polarisation and accumulate.
        const unsigned int i_pol_start = INDEX_3D(
                num_components, num_channels, num_pols,
                i_component, i_channel, 0);
        if (num_pols == 1)
        {
            const FLUX_TYPE2 flux = source_fluxes[i_pol_start];
            vis_local[0].x += phasor.x * flux.x - phasor.y * flux.y;
            vis_local[0].y += phasor.x * flux.y + phasor.y * flux.x;
        }
        else if (num_pols == 4)
        {
            #pragma unroll
            for (int i_pol = 0; i_pol < 4; ++i_pol)
            {
                const FLUX_TYPE2 flux = source_fluxes[i_pol_start + i_pol];
                vis_local[i_pol].x += phasor.x * flux.x - phasor.y * flux.y;
                vis_local[i_pol].y += phasor.x * flux.y + phasor.y * flux.x;
            }
        }
    }

    // Write out local visibility.
    for (int i_pol = 0; i_pol < num_pols; ++i_pol)
    {
        const unsigned int i_out = INDEX_4D(num_times, num_baselines,
                num_channels, num_pols, i_time, i_baseline, i_channel, i_pol);
        vis[i_out] = vis_local[i_pol];
    }
}

SDP_CUDA_KERNEL(dft_point_v00<double3, double2, double3, double2>)
SDP_CUDA_KERNEL(dft_point_v00<double3, double2, double3, float2>)

template<
        typename DIR_TYPE3,
        typename FLUX_TYPE2,
        typename UVW_TYPE3,
        typename VIS_TYPE2
>
__global__ void dft_point_v01(
        const int num_components,
        const int num_pols,
        const int num_channels,
        const int num_baselines,
        const int num_times,
        const DIR_TYPE3 *const __restrict__ source_directions,
        const FLUX_TYPE2 *const __restrict__ source_fluxes,
        const UVW_TYPE3 *const __restrict__ uvw_lambda,
        const double channel_start_hz,
        const double channel_step_hz,
        VIS_TYPE2 *__restrict__ vis
)
{
    // Local (per-thread) visibility. Allow up to 4 polarisations.
    VIS_TYPE2 vis_local[4];
    vis_local[0].x = vis_local[1].x = vis_local[2].x = vis_local[3].x = 0;
    vis_local[0].y = vis_local[1].y = vis_local[2].y = vis_local[3].y = 0;

    // Get indices of the output array this thread is working on.
    const int i_baseline = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_channel  = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_time     = blockDim.z * blockIdx.z + threadIdx.z;

    // Bounds check.
    if (num_pols > 4 ||
            i_baseline >= num_baselines ||
            i_channel >= num_channels ||
            i_time >= num_times)
    {
        return;
    }

    // Load uvw-coordinates.
    const unsigned int i_uvw = INDEX_2D(
            num_times, num_baselines,
            i_time, i_baseline);
    const UVW_TYPE3 uvw = uvw_lambda[i_uvw];
    const double inv_wavelength = (
        channel_start_hz + i_channel * channel_step_hz) / C_0;

    // Loop over components and calculate phase for each.
    for (int i_component = 0; i_component < num_components; ++i_component)
    {
        double2 phasor;
        const DIR_TYPE3 dir = source_directions[i_component];
        const double phase = -2.0 * M_PI * inv_wavelength * (
                dir.x * uvw.x + dir.y * uvw.y + dir.z * uvw.z);
        sincos(phase, &phasor.y, &phasor.x);

        // Multiply by flux in each polarisation and accumulate.
        const unsigned int i_pol_start = INDEX_3D(
                num_components, num_channels, num_pols,
                i_component, i_channel, 0);
        if (num_pols == 1)
        {
            const FLUX_TYPE2 flux = source_fluxes[i_pol_start];
            vis_local[0].x += phasor.x * flux.x - phasor.y * flux.y;
            vis_local[0].y += phasor.x * flux.y + phasor.y * flux.x;
        }
        else if (num_pols == 4)
        {
            #pragma unroll
            for (int i_pol = 0; i_pol < 4; ++i_pol)
            {
                const FLUX_TYPE2 flux = source_fluxes[i_pol_start + i_pol];
                vis_local[i_pol].x += phasor.x * flux.x - phasor.y * flux.y;
                vis_local[i_pol].y += phasor.x * flux.y + phasor.y * flux.x;
            }
        }
    }

    // Write out local visibility.
    for (int i_pol = 0; i_pol < num_pols; ++i_pol)
    {
        const unsigned int i_out = INDEX_4D(num_times, num_baselines,
                num_channels, num_pols, i_time, i_baseline, i_channel, i_pol);
        vis[i_out] = vis_local[i_pol];
    }
}

SDP_CUDA_KERNEL(dft_point_v01<double3, double2, double3, double2>)
SDP_CUDA_KERNEL(dft_point_v01<double3, double2, double3, float2>)
