/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef SDP_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include "utility/sdp_device_wrapper.h"
#include "utility/sdp_logging.h"

void sdp_launch_cuda_kernel(
        const char* name,
        const uint64_t num_blocks[3],
        const uint64_t num_threads[3],
        uint64_t shared_mem_bytes,
        void* stream,
        const void** args,
        sdp_Error* status)
{
    if (*status) return;
    if (!name)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        return;
    }
#ifdef SDP_HAVE_CUDA
    std::map<std::string, const void*>::const_iterator iter =
            sdp_CudaKernelRegistrar::kernels().find(std::string(name));
    if (iter != sdp_CudaKernelRegistrar::kernels().end())
    {
        dim3 num_blocks_(num_blocks[0], num_blocks[1], num_blocks[2]);
        dim3 num_threads_(num_threads[0], num_threads[1], num_threads[2]);
        // TODO Stream is currently only a placeholder...
        (void)stream;
        cudaError_t cuda_error = cudaLaunchKernel(iter->second,
                num_blocks_, num_threads_, const_cast<void**>(args),
                (size_t) shared_mem_bytes, 0);
        SDP_LOG_DEBUG("Running CUDA kernel '%s'", name);
        if (cuda_error != cudaSuccess)
        {
            *status = SDP_ERR_RUNTIME;
            SDP_LOG_ERROR("Kernel '%s' launch failure (CUDA error: %s).",
                    name, cudaGetErrorString(cuda_error));
        }
    }
    else
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Kernel '%s' has not been registered with "
                "the processing function library.", name);
    }
#else
    (void)num_blocks;
    (void)num_threads;
    (void)shared_mem_bytes;
    (void)stream;
    (void)args;
    *status = SDP_ERR_RUNTIME;
    SDP_LOG_ERROR("Unable to run kernel '%s': "
            "The processing function library was not compiled "
            "with CUDA support.", name);
#endif
}
