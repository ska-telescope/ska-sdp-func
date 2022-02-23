/* See the LICENSE file at the top-level directory of this distribution. */

#include "func/rfi/sdp_rfi_flagger.h"
#include "utility/sdp_device_wrapper.h"
#include "utility/sdp_logging.h"
static void check_params(
	       	const sdp_Mem* vis,
	       	const sdp_Mem* sequence,
	       	const sdp_Mem* thresholds,
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
    
}
void sdp_rfi_flagger(
	       	const sdp_Mem* vis,
	       	const sdp_Mem* sequence,
	       	const sdp_Mem* thresholds,
		sdp_Mem*  flags,
        	sdp_Error* status)
{

    check_params(vis,sequence,thresholds,flags,status);	
    const sdp_MemType type = sdp_mem_type(vis);
    const sdp_MemLocation location = sdp_mem_location(vis);
    if (*status) return;

    const int num_times      = (int)(sdp_mem_shape_dim(vis, 0))*21;
    const int num_channels   = (int)sdp_mem_shape_dim(vis, 2);
    const int seqlen 	     = (int)sdp_mem_shape_dim(sequence, 0);
    
    printf("Num time:%d\tNum_channels:%d\n",num_times,num_channels);

    if (location == SDP_MEM_GPU)
    {
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
}
