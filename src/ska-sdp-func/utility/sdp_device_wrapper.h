/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_DEVICE_WRAPPER_H_
#define SKA_SDP_PROC_FUNC_DEVICE_WRAPPER_H_

/**
 * @file sdp_device_wrapper.h
 */

#include <stdint.h>

#include "ska-sdp-func/utility/sdp_errors.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup Stream_struct
 * @{
 */

/**
 * @struct sdp_CudaStream
 *
 * @brief Wraps a CUDA stream handle.
 */
struct sdp_CudaStream;

/** @} */ /* End group Stream_struct. */

/* Typedefs. */
typedef struct sdp_CudaStream sdp_CudaStream;

/**
 * @defgroup device_func
 * @{
 */

/**
 * @brief Launches a CUDA kernel.
 *
 * The kernel name must have been registered with the processing function
 * library using the ::SDP_CUDA_KERNEL macro.
 *
 * The triple-angle-bracket syntax for launching a kernel is an nvcc extension,
 * so the calling code would need to be compiled with nvcc as well if this
 * was used. This function provides an isolation layer, so that the kernels
 * can be compiled separately when the library is built with CUDA support.
 *
 * @param name Name of the kernel to launch, as provided to ::SDP_CUDA_KERNEL.
 * @param num_blocks Number of thread blocks in 3D.
 * @param num_threads Number of threads per block in 3D.
 * @param shared_mem_bytes Amount of dynamic shared memory required, in bytes.
 * @param stream CUDA stream. A null pointer will use the default stream.
 * @param args Array of pointers to kernel arguments.
 * @param status Error status.
 */
void sdp_launch_cuda_kernel(
        const char* name,
        const uint64_t num_blocks[3],
        const uint64_t num_threads[3],
        uint64_t shared_mem_bytes,
        sdp_CudaStream* stream,
        const void** args,
        sdp_Error* status
);

/**
 * @brief Sets the CUDA device to use.
 *
 * This is a wrapper for cudaSetDevice().
 * It exists to allow the processing function library
 * to be compiled independently of CUDA, if required.
 */
void sdp_cuda_set_device(int device);

/**
 * @brief Creates a CUDA stream.
 *
 * This is a wrapper for cudaStreamCreate().
 * It exists to allow the processing function library
 * to be compiled independently of CUDA, if required.
 */
sdp_CudaStream* sdp_cuda_stream_create();

/**
 * @brief Destroys a CUDA stream.
 *
 * This is a wrapper for cudaStreamDestroy().
 * It exists to allow the processing function library
 * to be compiled independently of CUDA, if required.
 */
void sdp_cuda_stream_destroy(sdp_CudaStream* stream);

/**
 * @brief Synchronize a CUDA stream.
 *
 * Blocks until operations in a CUDA stream complete.
 * This is a wrapper for cudaStreamSynchronize().
 * It exists to allow the processing function library
 * to be compiled independently of CUDA, if required.
 */
void sdp_cuda_stream_synchronize(sdp_CudaStream* stream);

/** @} */ /* End group device_func. */

#ifdef __cplusplus
}
#endif

/* Kernel registrar implementation (for nvcc and C++ only). */

#ifdef __cplusplus

#include <stdlib.h>

/*
 * Note:
 * Don't use any STL containers here - doing so causes problems if using the
 * Intel compiler for host code, and nvcc (which requires GCC) for GPU code,
 * as both compilers link their own versions of the C++ standard library.
 * Use custom containers here instead, and create the kernel map on first use.
 */

struct sdp_CudaKernelRegistrar
{
    struct Pair
    {
        const char* first; // Name of kernel.
        const void* second; // Pointer to kernel.

        Pair(const char* name, const void* ptr) : first(name), second(ptr)
        {
        }
    };
    struct List
    {
        Pair* list_;
        int size_;

        List() : list_(0), size_(0)
        {
        }

        virtual ~List()
        {
            free(list_);
        }

        int size() const
        {
            return size_;
        }

        void push_back(const Pair& value)
        {
            size_++;
            list_ = (Pair*) realloc(list_, size_ * sizeof(Pair));
            list_[size_ - 1].first = value.first;
            list_[size_ - 1].second = value.second;
        }

        const Pair& operator[](int i) const
        {
            return list_[i];
        }
    };


    static List& kernels()
    {
        static List k;
        return k;
    }

    sdp_CudaKernelRegistrar(const char* name, const void* ptr)
    {
        kernels().push_back(Pair(name, ptr));
    }
};

// *INDENT-OFF*

#define M_CAT(A, B) M_CAT_(A, B)
#define M_CAT_(A, B) A##B

/**
 * @brief Registers a CUDA kernel with the processing function library.
 *
 * This allows the kernel to be called without needing to compile host code
 * with nvcc. It should be placed in the same source file as the kernel,
 * after it has been defined.
 *
 * The macro takes a single argument, which is simply the name of the kernel.
 * (It is implemented as a variadic macro to allow for templated kernels
 * that take multiple template parameters, where the commas between type names
 * would otherwise cause problems.)
 */
#define SDP_CUDA_KERNEL(...) \
    static sdp_CudaKernelRegistrar M_CAT(r_, __LINE__)(#__VA_ARGS__, (const void*) & __VA_ARGS__); // NOLINT

// *INDENT-OFF*

#endif /* __cplusplus */

#endif /* include guard */
