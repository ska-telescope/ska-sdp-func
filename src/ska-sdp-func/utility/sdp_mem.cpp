/* See the LICENSE file at the top-level directory of this distribution. */

#include <complex>
#include <cstdlib>
#include <cstring>

#include "ska-sdp-func/utility/private_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"

#ifdef SDP_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

using std::complex;

// Private implementation.
struct sdp_Mem
{
    sdp_MemType type; // Enumerated memory element type.
    sdp_MemLocation location; // Enumerated memory address space.
    int32_t is_c_contiguous; // True if strides are C contiguous.
    int32_t is_owner; // True if memory is owned, false if aliased.
    int32_t is_read_only; // True if data should be considered read only.
    int32_t num_dims; // Number of dimensions.
    int64_t num_elements; // Total number of elements allocated.
    int32_t ref_count; // Reference counter.
    int64_t* shape; // Size of each dimension, in number of elements.
    int64_t* stride; // Stride in each dimension, in bytes (Python compatible).
    void* data; // Data pointer.
};


// Begin anonymous namespace for file-local functions.
namespace {

// Local function to perform the memory allocation.
void sdp_mem_alloc(sdp_Mem* mem, sdp_Error* status)
{
    mem->is_owner = 1; // Set flag to indicate ownership, as we're allocating.
    const size_t bytes = mem->num_elements * sdp_mem_type_size(mem->type);
    if (*status || bytes == 0) return;
    if (mem->location == SDP_MEM_CPU)
    {
        mem->data = calloc(bytes, 1);
        if (!mem->data)
        {
            *status = SDP_ERR_MEM_ALLOC_FAILURE;
            SDP_LOG_CRITICAL("Host memory allocation failure "
                    "(requested %zu bytes)", bytes
            );
            return;
        }
    }
    else if (mem->location == SDP_MEM_GPU)
    {
#ifdef SDP_HAVE_CUDA
        const cudaError_t cuda_error = cudaMalloc(&mem->data, bytes);
        if (!mem->data || cuda_error)
        {
            *status = SDP_ERR_MEM_ALLOC_FAILURE;
            SDP_LOG_CRITICAL("GPU memory allocation failure: %s "
                    "(requested %zu bytes)",
                    cudaGetErrorString(cuda_error), bytes
            );
            if (mem->data)
            {
                cudaFree(mem->data);
                mem->data = 0;
            }
            return;
        }
#else
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_CRITICAL("Cannot allocate GPU memory: "
                "The processing function library was compiled without "
                "CUDA support"
        );
#endif
    }
    else
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_CRITICAL("Unsupported memory location");
    }
}


// Multiply all elements of an array by the specified value.
template<typename T>
void scale_real(sdp_Mem* mem, double value)
{
    T* mem_ = (T*) sdp_mem_data(mem);
    const int64_t num_elements = sdp_mem_num_elements(mem);
    #pragma omp parallel for
    for (int64_t i = 0; i < num_elements; ++i)
    {
        mem_[i] *= (T) value;
    }
}


// Set all elements of a 1D array to specified value.
template<typename T>
void set_value_1d(sdp_Mem* mem, int value, sdp_Error* status)
{
    sdp_MemViewCpu<T, 1> mem_;
    sdp_mem_check_and_view(mem, &mem_, status);
    const int shape0 = (int) mem_.shape[0];
    #pragma omp parallel for
    for (int i = 0; i < shape0; ++i)
    {
        mem_(i) = (T) value;
    }
}


// Set all elements of a 2D array to specified value. Much faster than memset.
template<typename T>
void set_value_2d(sdp_Mem* mem, int value, sdp_Error* status)
{
    sdp_MemViewCpu<T, 2> mem_;
    sdp_mem_check_and_view(mem, &mem_, status);
    const int shape0 = (int) mem_.shape[0];
    const int shape1 = (int) mem_.shape[1];
    #pragma omp parallel for
    for (int i = 0; i < shape0; ++i)
    {
        for (int j = 0; j < shape1; ++j)
        {
            mem_(i, j) = (T) value;
        }
    }
}


// Set all elements of a 3D array to specified value. Much faster than memset.
template<typename T>
void set_value_3d(sdp_Mem* mem, int value, sdp_Error* status)
{
    sdp_MemViewCpu<T, 3> mem_;
    sdp_mem_check_and_view(mem, &mem_, status);
    const int shape0 = (int) mem_.shape[0];
    const int shape1 = (int) mem_.shape[1];
    const int shape2 = (int) mem_.shape[2];
    #pragma omp parallel for
    for (int i = 0; i < shape0; ++i)
    {
        for (int j = 0; j < shape1; ++j)
        {
            for (int k = 0; k < shape2; ++k)
            {
                mem_(i, j, k) = (T) value;
            }
        }
    }
}

} // End anonymous namespace for file-local functions.


sdp_Mem* sdp_mem_create(
        sdp_MemType type,
        sdp_MemLocation location,
        int32_t num_dims,
        const int64_t* shape,
        sdp_Error* status
)
{
    sdp_Mem* mem = sdp_mem_create_wrapper(
            0, type, location, num_dims, shape, 0, status
    );
    sdp_mem_alloc(mem, status);
    return mem;
}


sdp_Mem* sdp_mem_create_wrapper(
        void* data,
        sdp_MemType type,
        sdp_MemLocation location,
        int32_t num_dims,
        const int64_t* shape,
        const int64_t* stride,
        sdp_Error* status
)
{
    sdp_Mem* mem = (sdp_Mem*) calloc(1, sizeof(sdp_Mem));
    mem->data = data;
    mem->ref_count = 1;
    mem->type = type;
    mem->location = location;
    mem->num_dims = num_dims;
    if (type == SDP_MEM_VOID) return mem;
    const int64_t element_size = sdp_mem_type_size(type);
    if (element_size <= 0)
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_CRITICAL("Unsupported data type");
        return mem;
    }

    // For Python compatibility, a zero-dimensional tensor is a scalar.
    mem->num_elements = 1;
    if (num_dims == 0) return mem;

    // Store shape (and strides, if given; otherwise compute them).
    mem->shape = (int64_t*) calloc(mem->num_dims, sizeof(int64_t));
    mem->stride = (int64_t*) calloc(mem->num_dims, sizeof(int64_t));
    for (int32_t i = num_dims - 1; i >= 0; --i)
    {
        mem->shape[i] = shape[i];
        mem->stride[i] = stride ? stride[i] : mem->num_elements * element_size;
        mem->num_elements *= shape[i];
    }

    // Check if strides are as expected for a standard contiguous C array.
    mem->is_c_contiguous = 1;
    int64_t num_elements = 1;
    for (int32_t i = num_dims - 1; i >= 0; --i)
    {
        if (sdp_mem_stride_bytes_dim(mem, i) != num_elements * element_size)
        {
            mem->is_c_contiguous = 0;
        }
        num_elements *= shape[i];
    }
    return mem;
}


sdp_Mem* sdp_mem_create_wrapper_for_slice(
        const sdp_Mem* src,
        const int64_t* slice_offsets,
        const int32_t num_dims_slice,
        const int64_t* slice_shape,
        sdp_Error* status
)
{
    if (num_dims_slice > src->num_dims)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_CRITICAL(
                "Number of dimensions in slice (%d) is larger than src (%d)",
                num_dims_slice, src->num_dims
        );
        return 0;
    }
    for (int32_t i = 0; i < src->num_dims; ++i)
    {
        if (slice_offsets[i] >= src->shape[i] || slice_offsets[i] < 0)
        {
            *status = SDP_ERR_INVALID_ARGUMENT;
            SDP_LOG_CRITICAL(
                    "Slice offset in dimension %d is out of bounds: %d (max is %d)",
                    i,
                    slice_offsets[i],
                    src->shape[i]
            );
            return 0;
        }
    }
    for (int32_t i = 0; i < num_dims_slice; ++i)
    {
        const int32_t i_dim_slice = num_dims_slice - 1 - i;
        if (slice_shape[i_dim_slice] > src->shape[src->num_dims - 1 - i])
        {
            *status = SDP_ERR_INVALID_ARGUMENT;
            SDP_LOG_CRITICAL(
                    "Slice shape in dimension %d is too large: %d > %d",
                    i_dim_slice, slice_shape[i_dim_slice],
                    src->shape[src->num_dims - 1 - i]
            );
            return 0;
        }
    }
    int64_t offset = 0; // offset in bytes.
    int64_t* stride = (int64_t*) calloc(
            (uint32_t) num_dims_slice, sizeof(int64_t)
    );
    for (int32_t i = 0; i < src->num_dims; ++i)
    {
        offset += slice_offsets[i] * src->stride[i]; // stride in bytes.
    }
    for (int32_t i = 0; i < num_dims_slice; ++i)
    {
        stride[num_dims_slice - 1 - i] = src->stride[src->num_dims - 1 - i];
    }
    sdp_Mem* mem = sdp_mem_create_wrapper(((char*) (src->data)) + offset,
            src->type, src->location, num_dims_slice, slice_shape, stride,
            status
    );
    free(stride);
    return mem;
}


sdp_Mem* sdp_mem_create_alias(const sdp_Mem* src)
{
    sdp_Error status = SDP_SUCCESS;
    sdp_Mem* mem = sdp_mem_create_wrapper(src->data, src->type, src->location,
            src->num_dims, src->shape, src->stride, &status
    );
    return mem;
}


sdp_Mem* sdp_mem_create_copy(
        const sdp_Mem* src,
        sdp_MemLocation location,
        sdp_Error* status
)
{
    sdp_Mem* mem = sdp_mem_create_wrapper(0, src->type, location,
            src->num_dims, src->shape, src->stride, status
    );
    sdp_mem_alloc(mem, status);
    sdp_mem_copy_contents(mem, src, 0, 0, src->num_elements, status);
    return mem;
}


void sdp_mem_clear_contents(sdp_Mem* mem, sdp_Error* status)
{
    if (*status || !mem || mem->num_elements == 0) return;
    sdp_mem_clear_portion(mem, 0, mem->num_elements, status);
}


void sdp_mem_clear_portion(
        sdp_Mem* mem,
        int64_t start_index,
        int64_t num_elements,
        sdp_Error* status
)
{
    if (*status || !mem || num_elements == 0) return;
    const size_t offset = start_index * sdp_mem_type_size(mem->type);
    const size_t size = num_elements * sdp_mem_type_size(mem->type);
    if (mem->location == SDP_MEM_CPU)
    {
        memset(((char*) (mem->data)) + offset, 0, size);
    }
    else if (mem->location == SDP_MEM_GPU)
    {
#ifdef SDP_HAVE_CUDA
        cudaMemset(((char*) (mem->data)) + offset, 0, size);
#else
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("The processing function library was compiled "
                "without CUDA support"
        );
#endif
    }
    else
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_CRITICAL("Unsupported memory location");
    }
}


sdp_Mem* sdp_mem_convert_precision(
        const sdp_Mem* src,
        sdp_MemType output_type,
        sdp_Error* status
)
{
    if (*status) return 0;
    if (!sdp_mem_is_c_contiguous(src))
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Array must be C contiguous");
        return 0;
    }
    const sdp_MemType input_type = sdp_mem_type(src);
    if (input_type == output_type)
    {
        return sdp_mem_create_copy(src, SDP_MEM_CPU, status);
    }
    sdp_Mem* src_temp = 0;
    const sdp_Mem* src_ptr = src;
    if (sdp_mem_location(src) != SDP_MEM_CPU)
    {
        src_temp = sdp_mem_create_copy(src, SDP_MEM_CPU, status);
        src_ptr = src_temp;
    }
    sdp_Mem* output = sdp_mem_create(
            output_type, SDP_MEM_CPU, src->num_dims, src->shape, status
    );
    const int64_t num_elements = sdp_mem_num_elements(src);
    if (input_type == SDP_MEM_COMPLEX_DOUBLE &&
            output_type == SDP_MEM_COMPLEX_FLOAT)
    {
        const complex<double>* in_ = (const complex<double>*) src_ptr->data;
        complex<float>* out_ = (complex<float>*) output->data;
        for (int64_t i = 0; i < num_elements; ++i)
        {
            out_[i] = (complex<float>) in_[i];
        }
    }
    else if (input_type == SDP_MEM_COMPLEX_FLOAT &&
            output_type == SDP_MEM_COMPLEX_DOUBLE)
    {
        const complex<float>* in_ = (const complex<float>*) src_ptr->data;
        complex<double>* out_ = (complex<double>*) output->data;
        for (int64_t i = 0; i < num_elements; ++i)
        {
            out_[i] = (complex<double>) in_[i];
        }
    }
    else if (input_type == SDP_MEM_DOUBLE &&
            output_type == SDP_MEM_FLOAT)
    {
        const double* in_ = (const double*) src_ptr->data;
        float* out_ = (float*) output->data;
        for (int64_t i = 0; i < num_elements; ++i)
        {
            out_[i] = (float) in_[i];
        }
    }
    else if (input_type == SDP_MEM_FLOAT &&
            output_type == SDP_MEM_DOUBLE)
    {
        const float* in_ = (const float*) src_ptr->data;
        double* out_ = (double*) output->data;
        for (int64_t i = 0; i < num_elements; ++i)
        {
            out_[i] = (double) in_[i];
        }
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data types");
    }
    sdp_mem_free(src_temp);
    return output;
}


void sdp_mem_copy_contents(
        sdp_Mem* dst,
        const sdp_Mem* src,
        int64_t offset_dst,
        int64_t offset_src,
        int64_t num_elements,
        sdp_Error* status
)
{
#ifdef SDP_HAVE_CUDA
    cudaError_t cuda_error = cudaSuccess;
#endif
    if (*status || !dst || !src || !dst->data || !src->data) return;
    if (src->num_elements == 0 || num_elements == 0) return;
    const int64_t element_size = sdp_mem_type_size(src->type);
    const int64_t start_dst   = element_size * offset_dst;
    const int64_t start_src   = element_size * offset_src;
    const size_t bytes        = element_size * num_elements;
    const int location_src    = src->location;
    const int location_dst    = dst->location;
    const void* p_src = (const void*)((const char*)(src->data) + start_src);
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
                cudaGetErrorString(cuda_error)
        );
    }
#endif
}


void sdp_mem_copy_contents_async(
        sdp_Mem* dst,
        const sdp_Mem* src,
        int64_t offset_dst,
        int64_t offset_src,
        int64_t num_elements,
        sdp_CudaStream* stream,
        sdp_Error* status
)
{
#ifdef SDP_HAVE_CUDA
    cudaError_t cuda_error = cudaSuccess;
    cudaStream_t cuda_stream = 0;
    if (stream) cuda_stream = stream->stream;
#endif
    if (*status || !dst || !src || !dst->data || !src->data) return;
    if (src->num_elements == 0 || num_elements == 0) return;
    const int64_t element_size = sdp_mem_type_size(src->type);
    const int64_t start_dst   = element_size * offset_dst;
    const int64_t start_src   = element_size * offset_src;
    const size_t bytes        = element_size * num_elements;
    const int location_src    = src->location;
    const int location_dst    = dst->location;
    const void* p_src = (const void*)((const char*)(src->data) + start_src);
    void* p_dst       = (void*)((char*)(dst->data) + start_dst);

    if (location_src == SDP_MEM_CPU && location_dst == SDP_MEM_CPU)
    {
        memcpy(p_dst, p_src, bytes);
    }
#ifdef SDP_HAVE_CUDA
    else if (location_src == SDP_MEM_CPU && location_dst == SDP_MEM_GPU)
    {
        cuda_error = cudaMemcpyAsync(p_dst, p_src, bytes,
                cudaMemcpyHostToDevice, cuda_stream
        );
    }
    else if (location_src == SDP_MEM_GPU && location_dst == SDP_MEM_CPU)
    {
        cuda_error = cudaMemcpyAsync(p_dst, p_src, bytes,
                cudaMemcpyDeviceToHost, cuda_stream
        );
    }
    else if (location_src == SDP_MEM_GPU && location_dst == SDP_MEM_GPU)
    {
        cuda_error = cudaMemcpyAsync(p_dst, p_src, bytes,
                cudaMemcpyDeviceToDevice, cuda_stream
        );
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
        SDP_LOG_CRITICAL("cudaMemcpyAsync error: %s",
                cudaGetErrorString(cuda_error)
        );
    }
#endif
}


void* sdp_mem_data(sdp_Mem* mem)
{
    return (!mem) ? 0 : mem->data;
}


const void* sdp_mem_data_const(const sdp_Mem* mem)
{
    return (!mem) ? 0 : mem->data;
}


void* sdp_mem_gpu_buffer(sdp_Mem* mem, sdp_Error* status)
{
    if (*status || !mem) return 0;
    if (mem->location != SDP_MEM_GPU && mem->type != SDP_MEM_VOID)
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_CRITICAL("Requested buffer is not in GPU memory");
        return 0;
    }
    return &mem->data;
}


const void* sdp_mem_gpu_buffer_const(const sdp_Mem* mem, sdp_Error* status)
{
    if (*status || !mem) return 0;
    if (mem->location != SDP_MEM_GPU && mem->type != SDP_MEM_VOID)
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
    if (mem->is_owner && mem->data)
    {
        if (mem->location == SDP_MEM_CPU)
        {
            free(mem->data);
        }
        else if (mem->location == SDP_MEM_GPU)
        {
#ifdef SDP_HAVE_CUDA
            cudaFree(mem->data);
#endif
        }
    }
    free(mem->shape);
    free(mem->stride);
    free(mem);
}


int32_t sdp_mem_is_c_contiguous(const sdp_Mem* mem)
{
    return (!mem || !mem->data) ? 0 : mem->is_c_contiguous;
}


int32_t sdp_mem_is_floating_point(const sdp_Mem* mem)
{
    if (!mem || !mem->data) return 0;
    return mem->type == SDP_MEM_FLOAT || mem->type == SDP_MEM_DOUBLE;
}


int32_t sdp_mem_is_complex(const sdp_Mem* mem)
{
    if (!mem || !mem->data) return 0;
    return (mem->type & SDP_MEM_COMPLEX) == SDP_MEM_COMPLEX;
}


int32_t sdp_mem_is_complex4(const sdp_Mem* mem)
{
    if (!sdp_mem_is_complex(mem)) return 0;
    const int32_t nd = mem->num_dims;
    return (nd > 1 && mem->shape[nd - 1] == 4) ||
           (nd > 2 && mem->shape[nd - 1] == 2 && mem->shape[nd - 2] == 2);
}


int32_t sdp_mem_is_double(const sdp_Mem* mem)
{
    if (!mem || !mem->data) return 0;
    return (mem->type & SDP_MEM_DOUBLE) == SDP_MEM_DOUBLE;
}


int32_t sdp_mem_is_matching(
        const sdp_Mem* mem1,
        const sdp_Mem* mem2,
        int32_t check_location
)
{
    if (mem1->type != mem2->type) return 0;
    if (check_location && (mem1->location != mem2->location)) return 0;
    if (mem1->num_dims != mem2->num_dims) return 0;
    for (int32_t i = 0; i < mem1->num_dims; ++i)
    {
        if (mem1->shape[i] != mem2->shape[i]) return 0;
        if (mem1->stride[i] != mem2->stride[i]) return 0;
    }
    return 1;
}


int32_t sdp_mem_is_read_only(const sdp_Mem* mem)
{
    return (!mem || !mem->data) ? 1 : mem->is_read_only;
}


sdp_MemLocation sdp_mem_location(const sdp_Mem* mem)
{
    return (!mem) ? SDP_MEM_CPU : mem->location;
}


int32_t sdp_mem_num_dims(const sdp_Mem* mem)
{
    return (!mem) ? 0 : mem->num_dims;
}


int64_t sdp_mem_num_elements(const sdp_Mem* mem)
{
    return (!mem || !mem->data) ? 0 : mem->num_elements;
}


void sdp_mem_random_fill(sdp_Mem* mem, sdp_Error* status)
{
    if (*status) return;
    if (mem->location != SDP_MEM_CPU)
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Unsupported memory location");
        return;
    }
    int64_t num_elements = mem->num_elements;
    const sdp_MemType precision = (sdp_MemType) (mem->type & 0x0F);
    if (sdp_mem_is_complex(mem)) num_elements *= 2;
    if (precision == SDP_MEM_FLOAT)
    {
        float* data = (float*) mem->data;
        for (int64_t i = 0; i < num_elements; ++i)
        {
            // NOLINTNEXTLINE: rand() is not a problem for our use case.
            data[i] = (float)rand() / (float)RAND_MAX;
        }
    }
    else if (precision == SDP_MEM_DOUBLE)
    {
        double* data = (double*) mem->data;
        for (int64_t i = 0; i < num_elements; ++i)
        {
            // NOLINTNEXTLINE: rand() is not a problem for our use case.
            data[i] = (double)rand() / (double)RAND_MAX;
        }
    }
}


void sdp_mem_ref_dec(sdp_Mem* mem)
{
    sdp_mem_free(mem);
}


sdp_Mem* sdp_mem_ref_inc(sdp_Mem* mem)
{
    if (!mem) return 0;
    mem->ref_count++;
    return mem;
}


void sdp_mem_scale_real(sdp_Mem* mem, double value, sdp_Error* status)
{
    if (*status) return;
    if (!sdp_mem_is_c_contiguous(mem))
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Array must be C contiguous");
        return;
    }
    const sdp_MemLocation loc = sdp_mem_location(mem);
    if (loc == SDP_MEM_CPU)
    {
        switch (sdp_mem_type(mem))
        {
        case SDP_MEM_COMPLEX_DOUBLE:
            scale_real<complex<double> >(mem, value);
            break;
        case SDP_MEM_COMPLEX_FLOAT:
            scale_real<complex<float> >(mem, value);
            break;
        case SDP_MEM_DOUBLE:
            scale_real<double>(mem, value);
            break;
        case SDP_MEM_FLOAT:
            scale_real<float>(mem, value);
            break;
        default:
            SDP_LOG_ERROR("Unsupported data type");
            *status = SDP_ERR_DATA_TYPE;
            return;
        }
    }
    else
    {
        // Call the kernel.
        uint64_t num_threads[] = {256, 1, 1}, num_blocks[] = {1, 1, 1};
        const int64_t num_elements = sdp_mem_num_elements(mem);
        num_blocks[0] = (num_elements + num_threads[0] - 1) / num_threads[0];
        const char* kernel_name = 0;
        switch (sdp_mem_type(mem))
        {
        case SDP_MEM_COMPLEX_DOUBLE:
            kernel_name = "sdp_mem_scale_real<complex<double> >";
            break;
        case SDP_MEM_COMPLEX_FLOAT:
            kernel_name = "sdp_mem_scale_real<complex<float> >";
            break;
        case SDP_MEM_DOUBLE:
            kernel_name = "sdp_mem_scale_real<double>";
            break;
        case SDP_MEM_FLOAT:
            kernel_name = "sdp_mem_scale_real<float>";
            break;
        default:
            SDP_LOG_ERROR("Unsupported data type");
            *status = SDP_ERR_DATA_TYPE;
            return;
        }
        const void* arg[] = {
            sdp_mem_gpu_buffer(mem, status),
            (const void*) &num_elements,
            (const void*) &value
        };
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, 0, 0, arg, status
        );
    }
}


void sdp_mem_set_read_only(sdp_Mem* mem, int32_t value)
{
    if (!mem) return;
    mem->is_read_only = value;
}


void sdp_mem_set_value(sdp_Mem* mem, int value, sdp_Error* status)
{
    if (*status) return;
    const sdp_MemLocation loc = sdp_mem_location(mem);
    if (loc == SDP_MEM_CPU)
    {
        if (sdp_mem_num_dims(mem) == 1)
        {
            switch (sdp_mem_type(mem))
            {
            case SDP_MEM_COMPLEX_DOUBLE:
                set_value_1d<complex<double> >(mem, value, status);
                break;
            case SDP_MEM_COMPLEX_FLOAT:
                set_value_1d<complex<float> >(mem, value, status);
                break;
            case SDP_MEM_DOUBLE:
                set_value_1d<double>(mem, value, status);
                break;
            case SDP_MEM_FLOAT:
                set_value_1d<float>(mem, value, status);
                break;
            case SDP_MEM_INT:
                set_value_1d<int>(mem, value, status);
                break;
            default:
                SDP_LOG_ERROR("Unsupported data type");
                *status = SDP_ERR_DATA_TYPE;
                return;
            }
        }
        else if (sdp_mem_num_dims(mem) == 2)
        {
            switch (sdp_mem_type(mem))
            {
            case SDP_MEM_COMPLEX_DOUBLE:
                set_value_2d<complex<double> >(mem, value, status);
                break;
            case SDP_MEM_COMPLEX_FLOAT:
                set_value_2d<complex<float> >(mem, value, status);
                break;
            case SDP_MEM_DOUBLE:
                set_value_2d<double>(mem, value, status);
                break;
            case SDP_MEM_FLOAT:
                set_value_2d<float>(mem, value, status);
                break;
            case SDP_MEM_INT:
                set_value_2d<int>(mem, value, status);
                break;
            default:
                SDP_LOG_ERROR("Unsupported data type");
                *status = SDP_ERR_DATA_TYPE;
                return;
            }
        }
        else if (sdp_mem_num_dims(mem) == 3)
        {
            switch (sdp_mem_type(mem))
            {
            case SDP_MEM_COMPLEX_DOUBLE:
                set_value_3d<complex<double> >(mem, value, status);
                break;
            case SDP_MEM_COMPLEX_FLOAT:
                set_value_3d<complex<float> >(mem, value, status);
                break;
            case SDP_MEM_DOUBLE:
                set_value_3d<double>(mem, value, status);
                break;
            case SDP_MEM_FLOAT:
                set_value_3d<float>(mem, value, status);
                break;
            case SDP_MEM_INT:
                set_value_3d<int>(mem, value, status);
                break;
            default:
                SDP_LOG_ERROR("Unsupported data type");
                *status = SDP_ERR_DATA_TYPE;
                return;
            }
        }
        else
        {
            *status = SDP_ERR_INVALID_ARGUMENT;
            SDP_LOG_ERROR("Only 1D, 2D or 3D arrays are currently supported");
        }
    }
    else
    {
        // Call the kernel.
        if (sdp_mem_num_dims(mem) == 1)
        {
            uint64_t num_threads[] = {256, 1, 1}, num_blocks[] = {1, 1, 1};
            const int64_t shape0 = sdp_mem_shape_dim(mem, 0);
            num_blocks[0] = (shape0 + num_threads[0] - 1) / num_threads[0];
            switch (sdp_mem_type(mem))
            {
            case SDP_MEM_COMPLEX_DOUBLE:
            {
                sdp_MemViewGpu<complex<double>, 1> mem_;
                sdp_mem_check_and_view(mem, &mem_, status);
                const void* arg[] = {(const void*) &mem_, &value};
                sdp_launch_cuda_kernel("sdp_mem_set_value_1d<complex<double> >",
                        num_blocks, num_threads, 0, 0, arg, status
                );
                break;
            }
            case SDP_MEM_COMPLEX_FLOAT:
            {
                sdp_MemViewGpu<complex<float>, 1> mem_;
                sdp_mem_check_and_view(mem, &mem_, status);
                const void* arg[] = {(const void*) &mem_, &value};
                sdp_launch_cuda_kernel("sdp_mem_set_value_1d<complex<float> >",
                        num_blocks, num_threads, 0, 0, arg, status
                );
                break;
            }
            case SDP_MEM_DOUBLE:
            {
                sdp_MemViewGpu<double, 1> mem_;
                sdp_mem_check_and_view(mem, &mem_, status);
                const void* arg[] = {(const void*) &mem_, &value};
                sdp_launch_cuda_kernel("sdp_mem_set_value_1d<double>",
                        num_blocks, num_threads, 0, 0, arg, status
                );
                break;
            }
            case SDP_MEM_FLOAT:
            {
                sdp_MemViewGpu<float, 1> mem_;
                sdp_mem_check_and_view(mem, &mem_, status);
                const void* arg[] = {(const void*) &mem_, &value};
                sdp_launch_cuda_kernel("sdp_mem_set_value_1d<float>",
                        num_blocks, num_threads, 0, 0, arg, status
                );
                break;
            }
            case SDP_MEM_INT:
            {
                sdp_MemViewGpu<int, 1> mem_;
                sdp_mem_check_and_view(mem, &mem_, status);
                const void* arg[] = {(const void*) &mem_, &value};
                sdp_launch_cuda_kernel("sdp_mem_set_value_1d<int>",
                        num_blocks, num_threads, 0, 0, arg, status
                );
                break;
            }
            default:
            {
                SDP_LOG_ERROR("Unsupported data type");
                *status = SDP_ERR_DATA_TYPE;
                return;
            }
            }
        }
        else if (sdp_mem_num_dims(mem) == 2)
        {
            uint64_t num_threads[] = {16, 16, 1}, num_blocks[] = {1, 1, 1};
            const int64_t shape0 = sdp_mem_shape_dim(mem, 0);
            const int64_t shape1 = sdp_mem_shape_dim(mem, 1);
            num_blocks[0] = (shape0 + num_threads[0] - 1) / num_threads[0];
            num_blocks[1] = (shape1 + num_threads[1] - 1) / num_threads[1];
            switch (sdp_mem_type(mem))
            {
            case SDP_MEM_COMPLEX_DOUBLE:
            {
                sdp_MemViewGpu<complex<double>, 2> mem_;
                sdp_mem_check_and_view(mem, &mem_, status);
                const void* arg[] = {(const void*) &mem_, &value};
                sdp_launch_cuda_kernel("sdp_mem_set_value_2d<complex<double> >",
                        num_blocks, num_threads, 0, 0, arg, status
                );
                break;
            }
            case SDP_MEM_COMPLEX_FLOAT:
            {
                sdp_MemViewGpu<complex<float>, 2> mem_;
                sdp_mem_check_and_view(mem, &mem_, status);
                const void* arg[] = {(const void*) &mem_, &value};
                sdp_launch_cuda_kernel("sdp_mem_set_value_2d<complex<float> >",
                        num_blocks, num_threads, 0, 0, arg, status
                );
                break;
            }
            case SDP_MEM_DOUBLE:
            {
                sdp_MemViewGpu<double, 2> mem_;
                sdp_mem_check_and_view(mem, &mem_, status);
                const void* arg[] = {(const void*) &mem_, &value};
                sdp_launch_cuda_kernel("sdp_mem_set_value_2d<double>",
                        num_blocks, num_threads, 0, 0, arg, status
                );
                break;
            }
            case SDP_MEM_FLOAT:
            {
                sdp_MemViewGpu<float, 2> mem_;
                sdp_mem_check_and_view(mem, &mem_, status);
                const void* arg[] = {(const void*) &mem_, &value};
                sdp_launch_cuda_kernel("sdp_mem_set_value_2d<float>",
                        num_blocks, num_threads, 0, 0, arg, status
                );
                break;
            }
            case SDP_MEM_INT:
            {
                sdp_MemViewGpu<int, 2> mem_;
                sdp_mem_check_and_view(mem, &mem_, status);
                const void* arg[] = {(const void*) &mem_, &value};
                sdp_launch_cuda_kernel("sdp_mem_set_value_2d<int>",
                        num_blocks, num_threads, 0, 0, arg, status
                );
                break;
            }
            default:
            {
                SDP_LOG_ERROR("Unsupported data type");
                *status = SDP_ERR_DATA_TYPE;
                return;
            }
            }
        }
        else if (sdp_mem_num_dims(mem) == 3)
        {
            uint64_t num_threads[] = {16, 8, 4}, num_blocks[] = {1, 1, 1};
            const int64_t shape0 = sdp_mem_shape_dim(mem, 0);
            const int64_t shape1 = sdp_mem_shape_dim(mem, 1);
            const int64_t shape2 = sdp_mem_shape_dim(mem, 2);
            num_blocks[0] = (shape0 + num_threads[0] - 1) / num_threads[0];
            num_blocks[1] = (shape1 + num_threads[1] - 1) / num_threads[1];
            num_blocks[2] = (shape2 + num_threads[2] - 1) / num_threads[2];
            switch (sdp_mem_type(mem))
            {
            case SDP_MEM_COMPLEX_DOUBLE:
            {
                sdp_MemViewGpu<complex<double>, 3> mem_;
                sdp_mem_check_and_view(mem, &mem_, status);
                const void* arg[] = {(const void*) &mem_, &value};
                sdp_launch_cuda_kernel("sdp_mem_set_value_3d<complex<double> >",
                        num_blocks, num_threads, 0, 0, arg, status
                );
                break;
            }
            case SDP_MEM_COMPLEX_FLOAT:
            {
                sdp_MemViewGpu<complex<float>, 3> mem_;
                sdp_mem_check_and_view(mem, &mem_, status);
                const void* arg[] = {(const void*) &mem_, &value};
                sdp_launch_cuda_kernel("sdp_mem_set_value_3d<complex<float> >",
                        num_blocks, num_threads, 0, 0, arg, status
                );
                break;
            }
            case SDP_MEM_DOUBLE:
            {
                sdp_MemViewGpu<double, 3> mem_;
                sdp_mem_check_and_view(mem, &mem_, status);
                const void* arg[] = {(const void*) &mem_, &value};
                sdp_launch_cuda_kernel("sdp_mem_set_value_3d<double>",
                        num_blocks, num_threads, 0, 0, arg, status
                );
                break;
            }
            case SDP_MEM_FLOAT:
            {
                sdp_MemViewGpu<float, 3> mem_;
                sdp_mem_check_and_view(mem, &mem_, status);
                const void* arg[] = {(const void*) &mem_, &value};
                sdp_launch_cuda_kernel("sdp_mem_set_value_3d<float>",
                        num_blocks, num_threads, 0, 0, arg, status
                );
                break;
            }
            case SDP_MEM_INT:
            {
                sdp_MemViewGpu<int, 3> mem_;
                sdp_mem_check_and_view(mem, &mem_, status);
                const void* arg[] = {(const void*) &mem_, &value};
                sdp_launch_cuda_kernel("sdp_mem_set_value_3d<int>",
                        num_blocks, num_threads, 0, 0, arg, status
                );
                break;
            }
            default:
            {
                SDP_LOG_ERROR("Unsupported data type");
                *status = SDP_ERR_DATA_TYPE;
                return;
            }
            }
        }
        else
        {
            *status = SDP_ERR_INVALID_ARGUMENT;
            SDP_LOG_ERROR("Only 1D, 2D or 3D arrays are currently supported");
        }
    }
}


int64_t sdp_mem_shape_dim(const sdp_Mem* mem, int32_t dim)
{
    return (!mem || dim < 0 || dim >= mem->num_dims) ? 0 : mem->shape[dim];
}


int64_t sdp_mem_stride_bytes_dim(const sdp_Mem* mem, int32_t dim)
{
    return (!mem || dim < 0 || dim >= mem->num_dims) ? 0 : mem->stride[dim];
}


int64_t sdp_mem_stride_elements_dim(const sdp_Mem* mem, int32_t dim)
{
    const int64_t type_size = sdp_mem_type_size(mem->type);
    return type_size > 0 ? sdp_mem_stride_bytes_dim(mem, dim) / type_size : 0;
}


sdp_MemType sdp_mem_type(const sdp_Mem* mem)
{
    return (!mem) ? SDP_MEM_VOID : mem->type;
}


int64_t sdp_mem_type_size(sdp_MemType type)
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
        return 2 * sizeof(float);
    case SDP_MEM_COMPLEX_DOUBLE:
        return 2 * sizeof(double);
    default:
        return 0;
    }
}


const char* sdp_mem_location_name(sdp_MemLocation location)
{
    switch (location)
    {
    case SDP_MEM_CPU:
        return "CPU";
    case SDP_MEM_GPU:
        return "GPU";
    default:
        return "???";
    }
}


const char* sdp_mem_type_name(sdp_MemType type)
{
    switch (type)
    {
    case SDP_MEM_VOID:
        return "void";
    case SDP_MEM_CHAR:
        return "char";
    case SDP_MEM_INT:
        return "int";
    case SDP_MEM_FLOAT:
        return "float";
    case SDP_MEM_DOUBLE:
        return "double";
    case SDP_MEM_COMPLEX_FLOAT:
        return "complex float";
    case SDP_MEM_COMPLEX_DOUBLE:
        return "complex double";
    default:
        return "???";
    }
}


void sdp_mem_check_writeable_at(
        const sdp_Mem* mem,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
)
{
    if (*status) return;
    if (sdp_mem_is_read_only(mem))
    {
        sdp_log_message(SDP_LOG_LEVEL_ERROR, stderr, func, file, line,
                "%s: Expected '%s' not to be read-only!", func, expr
        );
        *status = SDP_ERR_INVALID_ARGUMENT;
    }
}


void sdp_mem_check_c_contiguity_at(
        const sdp_Mem* mem,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
)
{
    if (*status) return;
    if (!sdp_mem_is_c_contiguous(mem))
    {
        sdp_log_message(SDP_LOG_LEVEL_ERROR, stderr, func, file, line,
                "%s: Expected '%s' to be C contiguous!", func, expr
        );
        *status = SDP_ERR_INVALID_ARGUMENT;
    }
}


void sdp_mem_check_location_at(
        const sdp_Mem* mem,
        sdp_MemLocation expected_location,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
)
{
    if (*status) return;
    if (sdp_mem_location(mem) != expected_location)
    {
        sdp_log_message(SDP_LOG_LEVEL_ERROR, stderr, func, file, line,
                "%s: Expected '%s' to be in %s memory (found %s)!", func, expr,
                sdp_mem_location_name(expected_location),
                sdp_mem_location_name(sdp_mem_location(mem))
        );
        *status = SDP_ERR_MEM_LOCATION;
    }
}


void sdp_mem_check_num_dims_at(
        const sdp_Mem* mem,
        int64_t expected_num_dims,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
)
{
    if (*status) return;
    if (sdp_mem_num_dims(mem) != expected_num_dims)
    {
        sdp_log_message(SDP_LOG_LEVEL_ERROR, stderr, func, file, line,
                "%s: Expected '%s' to have %d dimension%s (found %d)!",
                func, expr, expected_num_dims,
                (expected_num_dims == 1 ? "" : "s"), sdp_mem_num_dims(mem)
        );
        *status = SDP_ERR_INVALID_ARGUMENT;
    }
}


void sdp_mem_check_dim_size_at(
        const sdp_Mem* mem,
        int32_t dim,
        int64_t size,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
)
{
    if (*status) return;
    if (sdp_mem_shape_dim(mem, dim) != size)
    {
        sdp_log_message(SDP_LOG_LEVEL_ERROR, stderr, func, file, line,
                "%s: Expected '%s' dimension %d to have size %d (found %d)!",
                func, expr, dim, size, sdp_mem_shape_dim(mem, dim)
        );
        *status = SDP_ERR_INVALID_ARGUMENT;
    }
}


void sdp_mem_check_shape_at(
        const sdp_Mem* mem,
        int32_t expected_num_dims,
        const int64_t* expected_shape,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
)
{
    if (*status) return;
    if (sdp_mem_num_dims(mem) != expected_num_dims)
    {
        sdp_log_message(SDP_LOG_LEVEL_ERROR, stderr, func, file, line,
                "%s: Expected '%s' to have %d dimension%s (found %d)!",
                func, expr, expected_num_dims,
                (expected_num_dims == 1 ? "" : "s"), sdp_mem_num_dims(mem)
        );
        *status = SDP_ERR_INVALID_ARGUMENT;
        return;
    }
    for (int32_t dim = 0; dim < expected_num_dims; dim++)
    {
        sdp_mem_check_shape_dim_at(mem, dim, expected_shape[dim], status,
                expr, func, file, line
        );
    }
}


void sdp_mem_check_shape_dim_at(
        const sdp_Mem* mem,
        int32_t dim,
        const int64_t expected_shape,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
)
{
    if (*status) return;
    if (sdp_mem_num_dims(mem) <= dim)
    {
        sdp_log_message(SDP_LOG_LEVEL_ERROR, stderr, func, file, line,
                "%s: Expected '%s' to have at least %d dimension%s (found %d)!",
                func, expr, dim + 1, (dim == 0 ? "" : "s"),
                sdp_mem_num_dims(mem)
        );
        *status = SDP_ERR_INVALID_ARGUMENT;
        return;
    }
    if (sdp_mem_shape_dim(mem, dim) != expected_shape)
    {
        sdp_log_message(SDP_LOG_LEVEL_ERROR, stderr, func, file, line,
                "%s: Expected '%s' dimension %d to have size %d (found %d)!",
                func, expr, dim, expected_shape, sdp_mem_shape_dim(mem, dim)
        );
        *status = SDP_ERR_INVALID_ARGUMENT;
    }
}


void sdp_mem_check_same_shape_at(
        sdp_Mem* mem,
        int32_t dim,
        sdp_Mem* mem2,
        int32_t dim2,
        sdp_Error* status,
        const char* func,
        const char* expr,
        const char* expr2,
        const char* file,
        int line
)
{
    // Skip the check if the memory object has fewer dimension - we
    // assume there will be an sdp_mem_check_num_dims that will
    // already have objected to the wrong dimensionality.
    if (dim < sdp_mem_num_dims(mem) && dim2 < sdp_mem_num_dims(mem2) &&
            sdp_mem_shape_dim(mem, dim) != sdp_mem_shape_dim(mem2, dim2))
    {
        // Same memory object? Reflect in error message
        if (mem == mem2)
        {
            sdp_log_message(SDP_LOG_LEVEL_ERROR, stderr, func, file, line,
                    "%s: '%s' dimensions %d and %d do not"
                    " have the same size (%d != %d)!",
                    func, expr, dim, dim2,
                    sdp_mem_shape_dim(mem, dim), sdp_mem_shape_dim(mem2, dim2)
            );
        }
        else
        {
            sdp_log_message(SDP_LOG_LEVEL_ERROR, stderr, func, file, line,
                    "%s: '%s' dimension %d and '%s' dimension %d do not"
                    " have the same size (%d != %d)!",
                    func, expr, dim, expr2, dim2,
                    sdp_mem_shape_dim(mem, dim), sdp_mem_shape_dim(mem2, dim2)
            );
        }
        *status = SDP_ERR_INVALID_ARGUMENT;
    }
}


void sdp_mem_check_type_at(
        const sdp_Mem* mem,
        sdp_MemType expected_type,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
)
{
    if (*status) return;
    if (sdp_mem_type(mem) != expected_type)
    {
        sdp_log_message(SDP_LOG_LEVEL_ERROR, stderr, func, file, line,
                "%s: Expected '%s' to have type %s (found %s)!", func, expr,
                sdp_mem_type_name(expected_type),
                sdp_mem_type_name(sdp_mem_type(mem))
        );
        *status = SDP_ERR_DATA_TYPE;
    }
}
