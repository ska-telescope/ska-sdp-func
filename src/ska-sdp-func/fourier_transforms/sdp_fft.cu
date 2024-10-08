/* See the LICENSE file at the top-level directory of this distribution. */

#include <thrust/complex.h>

#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"

using thrust::complex;


template<typename T>
__global__ void sdp_fft_norm(sdp_MemViewGpu<T, 2> data, const double factor)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= data.shape[0] || iy >= data.shape[1]) return;
    data(ix, iy) *= factor;
}


template<typename T>
__global__ void sdp_fft_phase(sdp_MemViewGpu<T, 2> data)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= data.shape[0] || iy >= data.shape[1]) return;
    const int x = 1 - (((ix + iy) & 1) << 1);
    data(ix, iy) *= x;
}


SDP_CUDA_KERNEL(sdp_fft_norm<complex<double> >)
SDP_CUDA_KERNEL(sdp_fft_norm<complex<float> >)

SDP_CUDA_KERNEL(sdp_fft_phase<complex<double> >)
SDP_CUDA_KERNEL(sdp_fft_phase<complex<float> >)
