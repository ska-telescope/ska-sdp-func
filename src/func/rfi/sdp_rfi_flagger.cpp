/* See the LICENSE file at the top-level directory of this distribution. */

#include "func/rfi/sdp_rfi_flagger.h"
#include "utility/sdp_device_wrapper.h"
#include "utility/sdp_logging.h"
static void check_params(
	       	const sdp_Mem* vis,
	       	const sdp_Mem* sequence,
	       	const sdp_Mem* thresholds,
		const sdp_MemLocation vis_location,
		const sdp_MemLocation sequence_location,
		const sdp_MemLocation thresholds_location,
		const sdp_MemLocation flags_location,
		sdp_Mem*  flags,
        	sdp_Error* status)
{
    if (*status) return;

    if (sdp_mem_is_read_only(flags))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("output flags must be writable.");
        return;
    }

    if (!sdp_mem_is_c_contiguous(vis) || !sdp_mem_is_c_contiguous(sequence) || !sdp_mem_is_c_contiguous(thresholds))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("All arrays must be C contiguous");
        return;
    }
    
    if (vis_location != SDP_MEM_GPU)
    {

        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("vis data must be in GPU memory");
        return;
    }

    if (sequence_location != SDP_MEM_GPU)
    {

        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("sequences must be in GPU memory");
        return;
    }

    if (thresholds_location != SDP_MEM_GPU)
    {

        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("thresholds must be in GPU memory");
        return;
    }

    if (flags_location != SDP_MEM_GPU)
    {

        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("flags must be in GPU memory");
        return;
    }

}
void sdp_rfi_flagger(
	       	const sdp_Mem* vis,
	       	const sdp_Mem* sequence,
	       	const sdp_Mem* thresholds,
		sdp_Mem*  flags,
        	sdp_Error* status)
{


    const sdp_MemType type = sdp_mem_type(vis);
    const sdp_MemLocation location = sdp_mem_location(vis);
    const sdp_MemLocation sequence_location = sdp_mem_location(sequence);
    const sdp_MemLocation thresholds_location = sdp_mem_location(thresholds);
    const sdp_MemLocation flags_location = sdp_mem_location(flags);
    


    check_params(vis,sequence,thresholds,location,sequence_location,thresholds_location,flags_location,flags,status);	
    if (*status) return;

    const uint64_t num_baselines  = 21;
    const uint64_t num_times      = (uint64_t)(sdp_mem_shape_dim(vis, 0))*num_baselines;
    const uint64_t num_channels   = (uint64_t)sdp_mem_shape_dim(vis, 2);
    const uint64_t seqlen 	     = (uint64_t)sdp_mem_shape_dim(sequence, 0);
    

    const uint64_t num_threads[] = {256, 1, 1};
    const uint64_t num_blocks[] = {
            (num_times + num_threads[0] - 1) / num_threads[0], 1, 1
    };
    const char* kernel_name = 0;
    if (type == SDP_MEM_DOUBLE)
    {
        kernel_name = "rfi_flagger<double>";
    }

    else if (type == SDP_MEM_FLOAT)
    {
        kernel_name = "rfi_flagger<float>";
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data type");
    }
    const void* args[] = {
            &num_times,
    	&num_channels,
    	&seqlen,
    	sdp_mem_gpu_buffer_const(sequence,status),
            sdp_mem_gpu_buffer_const(vis, status),
            sdp_mem_gpu_buffer_const(thresholds, status),
    	sdp_mem_gpu_buffer(flags,status),
    };
    
    sdp_launch_cuda_kernel(kernel_name,num_blocks, num_threads, 0, 0, args, status);
}
