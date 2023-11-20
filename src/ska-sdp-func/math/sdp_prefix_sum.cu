/* See the LICENSE file at the top-level directory of this distribution. */

#include "ska-sdp-func/utility/sdp_device_wrapper.h"

/* Launch 1 thread block with 512 (or max number of) threads.
 * Shared memory size is num_threads * 2 * sizeof(T). */
template<typename T>
__global__ void sdp_prefix_sum_cuda (
        const int num_elements,
        const T* const __restrict__ in,
        T* out
)
{
    extern __shared__ __align__(64) unsigned char my_smem[];
    T* scratch = reinterpret_cast<T*>(my_smem);
    const int num_loops = (num_elements + blockDim.x) / blockDim.x;
    T running_total = (T)0;
    int idx = threadIdx.x; // Starting value.
    const int t = threadIdx.x + blockDim.x;
    for (int i = 0; i < num_loops; i++)
    {
        T val = (T)0;
        if (idx <= num_elements && idx > 0)
        {
            val = in[idx - 1];
        }
        scratch[threadIdx.x] = (T)0;
        scratch[t] = val;
        for (int j = 1; j < blockDim.x; j <<= 1)
        {
            __syncthreads();
            const T x = scratch[t - j];
            __syncthreads();
            scratch[t] += x;
        }

        // Store results. Note the very last element is the total number!
        __syncthreads();
        if (idx <= num_elements)
        {
            out[idx] = scratch[t] + running_total;
        }
        idx += blockDim.x;
        running_total += scratch[2 * blockDim.x - 1];
    }
}

SDP_CUDA_KERNEL(sdp_prefix_sum_cuda<int>)
