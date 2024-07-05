/* See the LICENSE file at the top-level directory of this distribution. */

#include <thrust/complex.h>

#include "ska-sdp-func/math/sdp_math_macros.h"
#include "ska-sdp-func/utility/sdp_cuda_atomics.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"

using thrust::complex;


template<typename OUT_TYPE, typename IN1_TYPE, typename IN2_TYPE>
__global__ void sdp_gridder_accum_scale_array(
        sdp_MemViewGpu<OUT_TYPE, 2> out,
        const sdp_MemViewGpu<const IN1_TYPE, 2> in1,
        const sdp_MemViewGpu<const IN2_TYPE, 2> in2,
        const int exponent,
        const int use_in2
)
{
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= out.shape[0] || j >= out.shape[1]) return;
    if (use_in2)
    {
        if (exponent == 1)
        {
            out(i, j) += (IN2_TYPE) in1(i, j) * in2(i, j);
        }
        else
        {
            out(i, j) += (IN2_TYPE) in1(i, j) * pow(in2(i, j), exponent);
        }
    }
    else
    {
        out(i, j) += in1(i, j);
    }
}


template<typename OUT_TYPE, typename IN1_TYPE, typename IN2_TYPE>
__global__ void sdp_gridder_scale_inv_array(
        sdp_MemViewGpu<OUT_TYPE, 2> out,
        const sdp_MemViewGpu<const IN1_TYPE, 2> in1,
        const sdp_MemViewGpu<const IN2_TYPE, 2> in2,
        const int exponent
)
{
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= out.shape[0] || j >= out.shape[1]) return;
    if (exponent == 1)
    {
        out(i, j) = (IN2_TYPE) in1(i, j) / in2(i, j);
    }
    else
    {
        out(i, j) = (IN2_TYPE) in1(i, j) / pow(in2(i, j), exponent);
    }
}


template<typename UVW_TYPE>
__global__ void sdp_gridder_uvw_bounds_all(
        const sdp_MemViewGpu<const UVW_TYPE, 2> uvws,
        const double freq0_hz,
        const double dfreq_hz,
        const sdp_MemViewGpu<const int, 1> start_chs,
        const sdp_MemViewGpu<const int, 1> end_chs,
        double* uvw_min,
        double* uvw_max
)
{
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t num_uvw = uvws.shape[0];
    if (i >= num_uvw)
        return;
    const int start_ch = start_chs(i), end_ch = end_chs(i);
    if (start_ch >= end_ch)
        return;
    const double uvw[] = {uvws(i, 0), uvws(i, 1), uvws(i, 2)};
    #pragma unroll
    for (int j = 0; j < 3; ++j)
    {
        const double u0 = freq0_hz * uvw[j] / C_0;
        const double du = dfreq_hz * uvw[j] / C_0;
        if (uvw[j] >= 0)
        {
            (void)sdp_atomic_min(&uvw_min[j], u0 + start_ch * du);
            (void)sdp_atomic_max(&uvw_max[j], u0 + (end_ch - 1) * du);
        }
        else
        {
            (void)sdp_atomic_max(&uvw_max[j], u0 + start_ch * du);
            (void)sdp_atomic_min(&uvw_min[j], u0 + (end_ch - 1) * du);
        }
    }
}


// *INDENT-OFF*
SDP_CUDA_KERNEL(sdp_gridder_accum_scale_array<complex<double>, complex<double>, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_accum_scale_array<complex<float>, complex<float>, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_accum_scale_array<complex<double>, complex<float>, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_accum_scale_array<complex<float>, complex<double>, complex<double> >)

SDP_CUDA_KERNEL(sdp_gridder_scale_inv_array<complex<double>, double, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_scale_inv_array<complex<float>, float, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_scale_inv_array<complex<double>, complex<double>, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_scale_inv_array<complex<float>, complex<float>, complex<double> >)

SDP_CUDA_KERNEL(sdp_gridder_uvw_bounds_all<double>)
SDP_CUDA_KERNEL(sdp_gridder_uvw_bounds_all<float>)
// *INDENT-ON*
