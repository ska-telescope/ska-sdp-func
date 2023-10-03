/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>

#include "ska-sdp-func/utility/sdp_device_wrapper.h"


template<typename FP, typename FP2>
__device__ void sdp_dipole(
        FP kL,
        FP cos_kL,
        FP phi_rad,
        FP sin_theta,
        FP cos_theta,
        FP2& e_theta,
        FP2& e_phi
)
{
    const FP cos_phi = cos(phi_rad);
    const FP denominator =
            (FP)1 + cos_phi * cos_phi * (cos_theta * cos_theta - (FP)1);
    if (denominator == (FP)0)
    {
        FP2 zero;
        zero.x = (FP) 0;
        zero.y = (FP) 0;
        e_theta = e_phi = zero;
    }
    else
    {
        const FP temp = (cos(kL * cos_phi * sin_theta) - cos_kL) / denominator;
        e_theta.x = -cos_phi * cos_theta * temp;
        e_theta.y = (FP) 0;
        e_phi.x = sin(phi_rad) * temp;
        e_phi.y = (FP) 0;
    }
}


template<typename FP, typename FP2>
__global__ void sdp_dipole_pattern(
        const int num_points,
        const FP* theta_rad,
        const FP* phi_rad,
        const FP kL,
        const FP cos_kL,
        const int stride,
        const int e_theta_offset,
        const int e_phi_offset,
        FP2* e_theta,
        FP2* e_phi
)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    sdp_dipole(kL, cos_kL,
            phi_rad[i], (FP)sin(theta_rad[i]), (FP)cos(theta_rad[i]),
            e_theta[i * stride + e_theta_offset],
            e_phi[i * stride + e_phi_offset]
    );
}

SDP_CUDA_KERNEL(sdp_dipole_pattern<float, float2>)
SDP_CUDA_KERNEL(sdp_dipole_pattern<double, double2>)


template<typename FP, typename FP2>
__global__ void sdp_dipole_pattern_scalar(
        const int num_points,
        const FP* theta_rad,
        const FP* phi_rad,
        const FP kL,
        const FP cos_kL,
        const int stride,
        const int offset,
        FP2* pattern
)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    FP amp, phi_;
    const FP sin_theta = sin(theta_rad[i]);
    const FP cos_theta = cos(theta_rad[i]);
    phi_ = phi_rad[i];
    FP2 x_theta, x_phi, y_theta, y_phi;
    sdp_dipole(kL, cos_kL, phi_, sin_theta, cos_theta, x_theta, x_phi);
    phi_ += ((FP) M_PI / (FP)2);
    sdp_dipole(kL, cos_kL, phi_, sin_theta, cos_theta, y_theta, y_phi);
    amp = x_theta.x * x_theta.x + x_phi.x * x_phi.x +
            y_theta.x * y_theta.x + y_phi.x * y_phi.x;
    amp /= (FP)2;
    amp = sqrt(amp);
    pattern[i * stride + offset].x = amp;
    pattern[i * stride + offset].y = (FP) 0;
}

SDP_CUDA_KERNEL(sdp_dipole_pattern_scalar<float, float2>)
SDP_CUDA_KERNEL(sdp_dipole_pattern_scalar<double, double2>)
