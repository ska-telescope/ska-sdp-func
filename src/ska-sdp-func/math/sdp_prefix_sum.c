/* See the LICENSE file at the top-level directory of this distribution. */

#include "ska-sdp-func/math/sdp_prefix_sum.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"

void sdp_prefix_sum(int num_elements, const sdp_Mem* in,
        sdp_Mem* out, sdp_Error* status)
{
    if (*status) return;
    const sdp_MemType type = sdp_mem_type(in);
    const sdp_MemLocation location = sdp_mem_location(in);
    if (location != sdp_mem_location(out))
    {
        *status = SDP_ERR_MEM_LOCATION;
        return;
    }
    if (type != sdp_mem_type(out))
    {
        *status = SDP_ERR_DATA_TYPE;
        return;
    }
    if (type != SDP_MEM_INT)
    {
        *status = SDP_ERR_DATA_TYPE;
        return;
    }
    if (sdp_mem_num_elements(out) < num_elements + 1)
    {
        /* Last element is total number. */
        *status = SDP_ERR_INVALID_ARGUMENT;
        return;
    }
    if (location == SDP_MEM_CPU)
    {
        int i = 0;
        int sum = 0;
        const int* in_ = (const int*)sdp_mem_data_const(in);
        int* out_ = (int*)sdp_mem_data(out);
        for (i = 0; i < num_elements; ++i)
        {
            int x = in_[i];
            out_[i] = sum;
            sum += x;
        }
        out_[i] = sum;
    }
    else
    {
        const uint64_t num_threads[] = {512, 1, 1}, num_blocks[] = {1, 1, 1};
        const uint64_t shared_mem = 2 * num_threads[0] * sizeof(int);
        const int num = (int) num_elements;
        const void* args[] = {
            &num,
            sdp_mem_gpu_buffer_const(in, status),
            sdp_mem_gpu_buffer(out, status)
        };
        sdp_launch_cuda_kernel(
                "sdp_prefix_sum_cuda<int>", num_blocks, num_threads,
                shared_mem, 0, args, status
        );
    }
}
