/* See the LICENSE file at the top-level directory of this distribution. */

#include "utility/sdp_device_wrapper.h"

template<typename T>
__global__
void vector_add (
    const int num_elements,
    const T *const __restrict__ input_a,
    const T *const __restrict__ input_b,
    T *__restrict__ output)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        output[i] = input_a[i] + input_b[i];
    }
}

SDP_CUDA_KERNEL(vector_add<float>)
SDP_CUDA_KERNEL(vector_add<double>)
