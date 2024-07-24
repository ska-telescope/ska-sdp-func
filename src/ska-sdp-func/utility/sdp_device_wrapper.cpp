/* See the LICENSE file at the top-level directory of this distribution. */

#include <map>
#include <string>

#ifdef SDP_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include "ska-sdp-func/utility/private_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"

static std::map<std::string, const void*> cuda_kernels_;


void sdp_launch_cuda_kernel(
        const char* name,
        const uint64_t num_blocks[3],
        const uint64_t num_threads[3],
        uint64_t shared_mem_bytes,
        sdp_CudaStream* stream,
        const void** args,
        sdp_Error* status
)
{
    if (*status) return;
    if (!name)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        return;
    }
#ifdef SDP_HAVE_CUDA
    if (cuda_kernels_.empty())
    {
        const sdp_CudaKernelRegistrar::List& kernels =
                sdp_CudaKernelRegistrar::kernels();
        for (int i = 0; i < kernels.size(); ++i)
        {
            std::string key = std::string(kernels[i].first);
            cuda_kernels_.insert(make_pair(key, kernels[i].second));
        }
    }
    std::map<std::string, const void*>::const_iterator iter =
            cuda_kernels_.find(std::string(name));
    if (iter != cuda_kernels_.end())
    {
        dim3 num_blocks_(num_blocks[0], num_blocks[1], num_blocks[2]);
        dim3 num_threads_(num_threads[0], num_threads[1], num_threads[2]);
        cudaStream_t cuda_stream = 0;
        if (stream)
        {
            cuda_stream = stream->stream;
        }
        cudaError_t cuda_error = cudaLaunchKernel(iter->second,
                num_blocks_, num_threads_, const_cast<void**>(args),
                (size_t) shared_mem_bytes, cuda_stream
        );
        SDP_LOG_DEBUG("Running CUDA kernel '%s'", name);
        if (cuda_error != cudaSuccess)
        {
            *status = SDP_ERR_RUNTIME;
            SDP_LOG_ERROR("Kernel '%s' launch failure (CUDA error: %s).",
                    name, cudaGetErrorString(cuda_error)
            );
        }
    }
    else
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Kernel '%s' has not been registered with "
                "the processing function library.", name
        );
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
            "with CUDA support.", name
    );
#endif
}


void sdp_cuda_set_device(int device)
{
#ifdef SDP_HAVE_CUDA
    cudaSetDevice(device);
#endif
}


sdp_CudaStream* sdp_cuda_stream_create()
{
    sdp_CudaStream* strm = (sdp_CudaStream*)calloc(1, sizeof(sdp_CudaStream));
#ifdef SDP_HAVE_CUDA
    cudaStreamCreate(&strm->stream);
#endif
    return strm;
}


void sdp_cuda_stream_destroy(sdp_CudaStream* stream)
{
#ifdef SDP_HAVE_CUDA
    cudaStreamDestroy(stream->stream);
#endif
    free(stream);
}
