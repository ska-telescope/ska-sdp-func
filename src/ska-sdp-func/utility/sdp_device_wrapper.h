/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_DEVICE_WRAPPER_H_
#define SKA_SDP_PROC_FUNC_DEVICE_WRAPPER_H_

/**
 * @file sdp_device_wrapper.h
 */

#include "ska-sdp-func/utility/sdp_errors.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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
 * @param stream CUDA stream (currently a placeholder).
 * @param args Array of pointers to kernel arguments.
 * @param status Error status.
 */
void sdp_launch_cuda_kernel(
        const char* name,
        const uint64_t num_blocks[3],
        const uint64_t num_threads[3],
        uint64_t shared_mem_bytes,
        void* stream,
        const void** args,
        sdp_Error* status);

/** @} */ /* End group device_func. */

#ifdef __cplusplus
}
#endif

/* Kernel registrar implementation (for nvcc and C++ only). */

#ifdef __cplusplus

#include <map>
#include <string>

struct sdp_CudaKernelRegistrar
{
    static std::map<std::string, const void*>& kernels()
    {
        static std::map<std::string, const void*> kernels_;
        return kernels_;
    }
    sdp_CudaKernelRegistrar(const char* name, const void* ptr)
    {
        kernels().insert(std::make_pair(std::string(name), ptr));
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
