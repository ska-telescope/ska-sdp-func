/* See the LICENSE file at the top-level directory of this distribution. */

#include <thrust/complex.h>

#include "ska-sdp-func/math/sdp_math_macros.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"


__device__ __forceinline__ double atomicMin(double* address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while (val < __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if ((ret = atomicCAS((unsigned long long*) address, old,
                __double_as_longlong(val)
                )) == old)
            break;
    }
    return __longlong_as_double(ret);
}


__device__ __forceinline__ double atomicMax(double* address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while (val > __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if ((ret = atomicCAS((unsigned long long*) address, old,
                __double_as_longlong(val)
                )) == old)
            break;
    }
    return __longlong_as_double(ret);
}


template<typename OUT_TYPE, typename IN1_TYPE, typename IN2_TYPE>
__global__ void sdp_gridder_scale_inv_array(
        const int64_t num_elements,
        OUT_TYPE* out,
        const IN1_TYPE* in1,
        const IN2_TYPE* in2,
        const int exponent
)
{
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_elements) return;
    if (exponent == 1)
    {
        out[i] = (IN2_TYPE) in1[i] / in2[i];
    }
    else
    {
        out[i] = (IN2_TYPE) in1[i] / pow(in2[i], exponent);
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
            (void)atomicMin(&uvw_min[j], u0 + start_ch * du);
            (void)atomicMax(&uvw_max[j], u0 + (end_ch - 1) * du);
        }
        else
        {
            (void)atomicMax(&uvw_max[j], u0 + start_ch * du);
            (void)atomicMin(&uvw_min[j], u0 + (end_ch - 1) * du);
        }
    }
}


// *INDENT-OFF*
SDP_CUDA_KERNEL(sdp_gridder_scale_inv_array<thrust::complex<double>, double, thrust::complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_scale_inv_array<thrust::complex<float>, float, thrust::complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_scale_inv_array<thrust::complex<double>, thrust::complex<double>, thrust::complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_scale_inv_array<thrust::complex<float>, thrust::complex<float>, thrust::complex<double> >)

SDP_CUDA_KERNEL(sdp_gridder_uvw_bounds_all<double>)
SDP_CUDA_KERNEL(sdp_gridder_uvw_bounds_all<float>)
// *INDENT-ON*
