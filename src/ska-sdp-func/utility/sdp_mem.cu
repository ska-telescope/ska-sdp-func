/* See the LICENSE file at the top-level directory of this distribution. */

#include <thrust/complex.h>

#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"

using thrust::complex;


template<typename T>
__global__ void sdp_mem_scale_real(T* mem, int64_t num_elements, double value)
{
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_elements) return;
    mem[i] *= (T) value;
}


template<typename T>
__global__ void sdp_mem_set_value_1d(sdp_MemViewGpu<T, 1> mem, int value)
{
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= mem.shape[0]) return;
    mem(i) = (T) value;
}


template<typename T>
__global__ void sdp_mem_set_value_2d(sdp_MemViewGpu<T, 2> mem, int value)
{
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= mem.shape[0] || j >= mem.shape[1]) return;
    mem(i, j) = (T) value;
}


template<typename T>
__global__ void sdp_mem_set_value_3d(sdp_MemViewGpu<T, 3> mem, int value)
{
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t j = blockDim.y * blockIdx.y + threadIdx.y;
    const int64_t k = blockDim.z * blockIdx.z + threadIdx.z;
    if (i >= mem.shape[0] || j >= mem.shape[1] || k >= mem.shape[2]) return;
    mem(i, j, k) = (T) value;
}


// *INDENT-OFF*
SDP_CUDA_KERNEL(sdp_mem_scale_real<complex<double> >)
SDP_CUDA_KERNEL(sdp_mem_scale_real<complex<float> >)
SDP_CUDA_KERNEL(sdp_mem_scale_real<double>)
SDP_CUDA_KERNEL(sdp_mem_scale_real<float>)

SDP_CUDA_KERNEL(sdp_mem_set_value_1d<complex<double> >)
SDP_CUDA_KERNEL(sdp_mem_set_value_1d<complex<float> >)
SDP_CUDA_KERNEL(sdp_mem_set_value_1d<double>)
SDP_CUDA_KERNEL(sdp_mem_set_value_1d<float>)
SDP_CUDA_KERNEL(sdp_mem_set_value_1d<int>)
SDP_CUDA_KERNEL(sdp_mem_set_value_2d<complex<double> >)
SDP_CUDA_KERNEL(sdp_mem_set_value_2d<complex<float> >)
SDP_CUDA_KERNEL(sdp_mem_set_value_2d<double>)
SDP_CUDA_KERNEL(sdp_mem_set_value_2d<float>)
SDP_CUDA_KERNEL(sdp_mem_set_value_2d<int>)
SDP_CUDA_KERNEL(sdp_mem_set_value_3d<complex<double> >)
SDP_CUDA_KERNEL(sdp_mem_set_value_3d<complex<float> >)
SDP_CUDA_KERNEL(sdp_mem_set_value_3d<double>)
SDP_CUDA_KERNEL(sdp_mem_set_value_3d<float>)
SDP_CUDA_KERNEL(sdp_mem_set_value_3d<int>)
// *INDENT-ON*
