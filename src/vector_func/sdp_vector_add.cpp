/* See the LICENSE file at the top-level directory of this distribution. */

#include "logging/sdp_logging.h"
#include "utility/sdp_device_wrapper.h"
#include "vector_func/sdp_vector_add.h"

template<typename T>
void sdp_vector_add(int num_elements, const T* a, const T* b, T* out)
{
    for (int i = 0; i < num_elements; ++i)
    {
        out[i] = a[i] + b[i];
    }
}

void sdp_vector_add(
        int num_elements,
        const sdp_Mem* a,
        const sdp_Mem* b,
        sdp_Mem* out,
        sdp_Error* status)
{
    if (*status) return;
    const sdp_MemType type = sdp_mem_type(out);
    const sdp_MemLocation location = sdp_mem_location(out);
    if (sdp_mem_type(a) != type || sdp_mem_type(b) != type)
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Data type mismatch");
        return;
    }
    if (sdp_mem_location(a) != location || sdp_mem_location(b) != location)
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Memory location mismatch");
        return;
    }
    if (location == SDP_MEM_CPU)
    {
        if (type == SDP_MEM_DOUBLE)
        {
            sdp_vector_add<double>(
                    num_elements,
                    (const double*)sdp_mem_data_const(a),
                    (const double*)sdp_mem_data_const(b),
                    (double*)sdp_mem_data((out)));
        }
        else if (type == SDP_MEM_FLOAT)
        {
            sdp_vector_add<float>(
                    num_elements,
                    (const float*)sdp_mem_data_const(a),
                    (const float*)sdp_mem_data_const(b),
                    (float*)sdp_mem_data((out)));
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type");
        }
    }
    else
    {
        size_t num_threads[] = {256, 1, 1}, num_blocks[] = {1, 1, 1};
        const char* kernel_name = 0;
        if (type == SDP_MEM_DOUBLE)
        {
            kernel_name = "vector_add<double>";
        }
        else if (type == SDP_MEM_FLOAT)
        {
            kernel_name = "vector_add<float>";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type");
        }
        if (kernel_name)
        {
            num_blocks[0] =
                    (num_elements + num_threads[0] - 1) / num_threads[0];
            const void* args[] = {
                    &num_elements,
                    sdp_mem_gpu_buffer_const(a, status),
                    sdp_mem_gpu_buffer_const(b, status),
                    sdp_mem_gpu_buffer(out, status)
            };
            sdp_launch_cuda_kernel(kernel_name,
                    num_blocks, num_threads, 0, 0, args, status);
        }
    }
}
