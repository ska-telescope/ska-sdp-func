/* See the LICENSE file at the top-level directory of this distribution. */

#include <stdlib.h>
#include <string.h>

#include "mem/sdp_mem.h"
#include "logging/sdp_logging.h"

#ifdef SDP_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

// Private implementation.
struct sdp_Mem
{
    sdp_MemType type; // Enumerated memory element type.
    sdp_MemLocation location; // Enumerated memory address space.
    size_t num_elements; // Number of elements allocated.
    int owner; // True if memory is owned, false if aliased.
    int ref_count; // Reference counter.
    void* data; // Data pointer.
};

sdp_Mem* sdp_mem_create(
        sdp_MemType type,
        sdp_MemLocation location,
        size_t num_elements,
        sdp_Error* status)
{
#ifdef SDP_HAVE_CUDA
    cudaError_t cuda_error = cudaSuccess;
#endif
    sdp_Mem* mem = (sdp_Mem*) calloc(1, sizeof(sdp_Mem));
    mem->owner = 1;
    mem->ref_count = 1;
    mem->type = type;
    mem->location = location;
    if (*status || num_elements == 0) return mem;

    // Get the element size and total amount of memory requested.
    const size_t element_size = sdp_mem_type_size(type);
    if (element_size == 0)
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_CRITICAL("Unknown data type");
        return mem;
    }
    mem->num_elements = num_elements;
    const size_t bytes = num_elements * element_size;

    // Check whether the memory should be on the host or the device.
    switch (location)
    {
    case SDP_MEM_CPU:
        mem->data = calloc(bytes, 1);
        if (!mem->data)
        {
            *status = SDP_ERR_MEM_ALLOC_FAILURE;
            SDP_LOG_CRITICAL("Host memory allocation failure "
                    "(requested %zu bytes)", bytes);
            return mem;
        }
        break;
    case SDP_MEM_GPU:
#ifdef SDP_HAVE_CUDA
        cuda_error = cudaMalloc(&mem->data, bytes);
        if (!mem->data || cuda_error)
        {
            *status = SDP_ERR_MEM_ALLOC_FAILURE;
            SDP_LOG_CRITICAL("GPU memory allocation failure: %s "
                    "(requested %zu bytes)",
                    cudaGetErrorString(cuda_error), bytes);
            if (mem->data)
            {
                cudaFree(mem->data);
                mem->data = 0;
            }
            return mem;
        }
#else
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_CRITICAL("Cannot allocate GPU memory: "
                "The processing function library was compiled without "
                "CUDA support");
#endif
        break;
    default:
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_CRITICAL("Unsupported memory location");
        break;
    };
    return mem;
}

sdp_Mem* sdp_mem_create_from_raw(
        void* ptr,
        sdp_MemType type,
        sdp_MemLocation location,
        size_t num_elements,
        sdp_Error* status)
{
    sdp_Mem* mem = (sdp_Mem*) calloc(1, sizeof(sdp_Mem));
    mem->owner = 0;
    mem->ref_count = 1;
    mem->type = type;
    mem->location = location;
    if (*status || num_elements == 0) return mem;
    mem->num_elements = num_elements;
    mem->data = ptr;
    return mem;
}

sdp_Mem* sdp_mem_create_copy(
        const sdp_Mem* src,
        sdp_MemLocation location,
        sdp_Error* status)
{
    sdp_Mem* mem = sdp_mem_create(src->type, location, src->num_elements, status);
    if (!mem || *status) return mem;
    sdp_mem_copy_contents(mem, src, 0, 0, src->num_elements, status);
    return mem;
}

void sdp_mem_clear_contents(
        sdp_Mem* mem,
        sdp_Error* status)
{
    if (*status || mem->num_elements == 0) return;
    const size_t size = mem->num_elements * sdp_mem_type_size(mem->type);
    if (mem->location == SDP_MEM_CPU)
    {
        memset(mem->data, 0, size);
    }
    else if (mem->location == SDP_MEM_GPU)
    {
#ifdef SDP_HAVE_CUDA
        cudaMemset(mem->data, 0, size);
#else
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("The processing function library was compiled "
                "without CUDA support");
#endif
    }
    else
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_CRITICAL("Unsupported memory location");
    }
}

void sdp_mem_copy_contents(
        sdp_Mem* dst,
        const sdp_Mem* src,
        size_t offset_dst,
        size_t offset_src,
        size_t num_elements,
        sdp_Error* status)
{
#ifdef SDP_HAVE_CUDA
    cudaError_t cuda_error = cudaSuccess;
#endif
    if (*status) return;
    if (src->num_elements == 0 || num_elements == 0) return;
    const size_t element_size = sdp_mem_type_size(src->type);
    const size_t bytes        = element_size * num_elements;
    const size_t start_dst    = element_size * offset_dst;
    const size_t start_src    = element_size * offset_src;
    const int location_src    = src->location;
    const int location_dst    = dst->location;
    const void *p_src = (const void*)((const char*)(src->data) + start_src);
    void* p_dst       = (void*)((char*)(dst->data) + start_dst);

    if (location_src == SDP_MEM_CPU && location_dst == SDP_MEM_CPU)
    {
        memcpy(p_dst, p_src, bytes);
    }
#ifdef SDP_HAVE_CUDA
    else if (location_src == SDP_MEM_CPU && location_dst == SDP_MEM_GPU)
    {
        cuda_error = cudaMemcpy(p_dst, p_src, bytes, cudaMemcpyHostToDevice);
    }
    else if (location_src == SDP_MEM_GPU && location_dst == SDP_MEM_CPU)
    {
        cuda_error = cudaMemcpy(p_dst, p_src, bytes, cudaMemcpyDeviceToHost);
    }
    else if (location_src == SDP_MEM_GPU && location_dst == SDP_MEM_GPU)
    {
        cuda_error = cudaMemcpy(p_dst, p_src, bytes, cudaMemcpyDeviceToDevice);
    }
#endif
    else
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_CRITICAL("Unsupported memory location");
    }
#ifdef SDP_HAVE_CUDA
    if (cuda_error != cudaSuccess)
    {
        *status = SDP_ERR_MEM_COPY_FAILURE;
        SDP_LOG_CRITICAL("cudaMemcpy error: %s",
                cudaGetErrorString(cuda_error));
    }
#endif
}

void* sdp_mem_data(sdp_Mem* mem)
{
    return mem->data;
}

const void* sdp_mem_data_const(const sdp_Mem* mem)
{
    return mem->data;
}

void* sdp_mem_gpu_buffer(sdp_Mem* mem, sdp_Error* status)
{
    if (*status) return 0;
    if (mem->location != SDP_MEM_GPU)
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_CRITICAL("Requested buffer is not in GPU memory");
        return 0;
    }
    return &mem->data;
}

const void* sdp_mem_gpu_buffer_const(const sdp_Mem* mem, sdp_Error* status)
{
    if (*status) return 0;
    if (mem->location != SDP_MEM_GPU)
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_CRITICAL("Requested buffer is not in GPU memory");
        return 0;
    }
    return &mem->data;
}

void sdp_mem_free(sdp_Mem* mem)
{
    if (!mem) return;
    if (--mem->ref_count > 0) return;
    if (mem->owner && mem->data)
    {
        switch (mem->location)
        {
        case SDP_MEM_CPU:
            free(mem->data);
            break;
#ifdef SDP_HAVE_CUDA
        case SDP_MEM_GPU:
            cudaFree(mem->data);
            break;
#endif
        default:
            break;
        };
    }
    free(mem);
}

sdp_MemLocation sdp_mem_location(const sdp_Mem* mem)
{
    return mem->location;
}

size_t sdp_mem_num_elements(const sdp_Mem* mem)
{
    return mem->num_elements;
}

int sdp_mem_ref_count(const sdp_Mem* mem)
{
    return mem->ref_count;
}

void sdp_mem_ref_dec(sdp_Mem* mem)
{
    sdp_mem_free(mem);
}

sdp_Mem* sdp_mem_ref_inc(sdp_Mem* mem)
{
    mem->ref_count++;
    return mem;
}

sdp_MemType sdp_mem_type(const sdp_Mem* mem)
{
    return mem->type;
}

size_t sdp_mem_type_size(sdp_MemType type)
{
    switch (type)
    {
    case SDP_MEM_CHAR:
        return sizeof(char);
    case SDP_MEM_INT:
        return sizeof(int);
    case SDP_MEM_FLOAT:
        return sizeof(float);
    case SDP_MEM_DOUBLE:
        return sizeof(double);
    case SDP_MEM_COMPLEX_FLOAT:
        return 2*sizeof(float);
    case SDP_MEM_COMPLEX_DOUBLE:
        return 2*sizeof(double);
    default:
        return 0;
    }
}
