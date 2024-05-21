/* See the LICENSE file at the top-level directory of this distribution. */

#include "ska-sdp-func/utility/sdp_device_wrapper.h"


template<typename T>
__global__ void fft_phase(const int num_x, const int num_y, T* data)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= num_x || iy >= num_y) return;
    const int x = 1 - (((ix + iy) & 1) << 1);
    data[((iy * num_x + ix) << 1)]     *= x;
    data[((iy * num_x + ix) << 1) + 1] *= x;
}

SDP_CUDA_KERNEL(fft_phase<double>)
SDP_CUDA_KERNEL(fft_phase<float>)
