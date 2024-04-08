/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PRIVATE_DEVICE_WRAPPER_H_
#define SKA_SDP_PRIVATE_DEVICE_WRAPPER_H_

#ifdef SDP_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct sdp_CudaStream
{
#ifdef SDP_HAVE_CUDA
    cudaStream_t stream;
#else
    int dummy; /* Avoid an empty struct. */
#endif
};

#ifdef __cplusplus
}
#endif

#endif /* include guard */
