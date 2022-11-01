/* See the LICENSE file at the top-level directory of this distribution. */

#include "ska-sdp-func/vector/sdp_vector_add.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"


template<typename T>
static void sdp_vector_add(
        int64_t num_elements,
        const T* input_a,
        const T* input_b,
        T* output)
{
    for (int64_t i = 0; i < num_elements; ++i)
    {
        output[i] = input_a[i] + input_b[i];
    }
}


void sdp_vector_add(
        const sdp_Mem* input_a,
        const sdp_Mem* input_b,
        sdp_Mem* output,
        sdp_Error* status)
{
    if (*status) return;
    const sdp_MemType type = sdp_mem_type(output);
    const sdp_MemLocation location = sdp_mem_location(output);
    const int64_t num_elements = sdp_mem_num_elements(output);
    if (sdp_mem_is_read_only(output))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Output vector must be writable.");
        return;
    }
    if (sdp_mem_type(input_a) != type || sdp_mem_type(input_b) != type)
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Data type mismatch");
        return;
    }
    if (sdp_mem_location(input_a) != location ||
            sdp_mem_location(input_b) != location)
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Memory location mismatch");
        return;
    }
    if (sdp_mem_num_elements(input_a) != num_elements ||
            sdp_mem_num_elements(input_b) != num_elements)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("All vectors must have the same number of elements");
        return;
    }
    if (location == SDP_MEM_CPU)
    {
        if (type == SDP_MEM_DOUBLE)
        {
            sdp_vector_add<double>(
                    num_elements,
                    (const double*)sdp_mem_data_const(input_a),
                    (const double*)sdp_mem_data_const(input_b),
                    (double*)sdp_mem_data((output)));
        }
        else if (type == SDP_MEM_FLOAT)
        {
            sdp_vector_add<float>(
                    num_elements,
                    (const float*)sdp_mem_data_const(input_a),
                    (const float*)sdp_mem_data_const(input_b),
                    (float*)sdp_mem_data((output)));
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type");
        }
    }
    else if (location == SDP_MEM_GPU)
    {
        const uint64_t num_threads[] = {256, 1, 1};
        const uint64_t num_blocks[] = {
            (num_elements + num_threads[0] - 1) / num_threads[0], 1, 1
        };
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
        const void* args[] = {
            &num_elements,
            sdp_mem_gpu_buffer_const(input_a, status),
            sdp_mem_gpu_buffer_const(input_b, status),
            sdp_mem_gpu_buffer(output, status)
        };
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, 0, 0, args, status);
    }
}
