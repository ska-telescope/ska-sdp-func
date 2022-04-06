/* See the LICENSE file at the top-level directory of this distribution. */

#include <complex>

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
    
    if ( ( sdp_mem_location(vis) == SDP_MEM_GPU && sdp_mem_location(sequence)!=SDP_MEM_GPU && sdp_mem_location(thresholds)!=SDP_MEM_GPU && sdp_mem_location(flags) == SDP_MEM_GPU ) ||  (sdp_mem_location(vis) == SDP_MEM_CPU && sdp_mem_location(sequence) != SDP_MEM_CPU && sdp_mem_location(thresholds)!=SDP_MEM_CPU && sdp_mem_location(flags) == SDP_MEM_CPU ) )
    {

        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("All array's must be in the same memory location.");
        return;
    }
}

void write_flags_to_the_slice_array(
				const uint64_t num_freqs, 
				const uint64_t num_baselines, 
				const uint64_t num_polarisation, 
				const uint64_t num_time,
                                const int baseline_id, 
				const int polarisation_id, 
				int* flags_on_the_block, 
				int* flags
)
{

    for (uint64_t i = 0; i < num_time; i++){
        for (uint64_t j = 0; j < num_freqs; j++){
            flags[i * num_freqs * num_baselines * num_polarisation + baseline_id * num_freqs * num_polarisation
            + j * num_polarisation + polarisation_id] = flags_on_the_block[i * num_freqs + j];
        }
    }

}


template<typename TCPU>
void sum_threshold_on_block(
		const TCPU* thresholds, 
		const uint64_t seqlen, 
		const int* sequence_lengths, 
		TCPU *block,
		const uint64_t num_freqs,
		const uint64_t num_time,
		int *flags_on_block, 
		const int freq_or_time
)
{
    float current_threshold = 0;
    float sum = 0;
    uint64_t current_seqlen = 0;
    if (!freq_or_time){
        for (uint64_t k = 0; k < seqlen; k++){
            current_seqlen = sequence_lengths[k];
            current_threshold = thresholds[k] * current_seqlen;
            for (uint64_t j = 0; j < num_freqs; j++){
                for (uint64_t i = 0; i < num_time - current_seqlen; i++){
                    sum = 0;
                    for (uint64_t m = 0; m < current_seqlen; m++){


                        sum = sum + block[(i + m) * num_freqs + j];
                    }
                 if (sum > current_threshold){
                        for (uint64_t m = 0; m < current_seqlen; m++){
                            flags_on_block[(i + m) * num_freqs + j] = 1;
                        }
                    } 
                }
            }

        }
    } else if (freq_or_time){
        for (uint64_t k = 0; k < seqlen; k++){
            current_seqlen = sequence_lengths[k];
            current_threshold = thresholds[k] * current_seqlen;
            for (uint64_t i = 0; i < num_time; i++){
                for (uint64_t j = 0; j < num_freqs - current_seqlen; j++){
                    sum = 0;
                    for (uint64_t m = 0; m < current_seqlen; m++){
                        if (flags_on_block[i * num_freqs + j + m] == 1){
                            block[i * num_freqs + j + m] = thresholds[k];
                        }
                        sum = sum + block[i * num_freqs + j + m];
                    }

                    if (sum > current_threshold){
                        for (uint64_t m = 0; m < current_seqlen; m++){
                            flags_on_block[i  * num_freqs + j + m] = 1;
                        }
                    }
                }
            }

        }
    }

}

template<typename TCPU>
void  rfi_flagger(
		const uint64_t num_time,
		const uint64_t num_freqs,
		const uint64_t seqlen,
		const uint64_t num_polarisations,
		const uint64_t num_baselines,
		const int* sequence_lengths,
		const std::complex<TCPU>* const __restrict__ visibilities,
		const TCPU*const __restrict__ thresholds,
		int*  flags
)
{

    TCPU *block = new TCPU[num_time * num_freqs];
    int* flags_on_the_block = new int[num_time * num_freqs];
    for (uint64_t m = 0; m < num_baselines; m++){
        for (uint64_t k = 0; k < num_polarisations; k++){

            for (uint64_t i = 0; i < num_freqs * num_time; i++){
                flags_on_the_block[i] = 0;
            }
            for (uint64_t i = 0; i < num_time; i++){
                for (uint64_t j = 0; j < num_freqs; j++){

		    std::complex<TCPU>temp =  visibilities[i * num_freqs * num_polarisations
                                            * num_baselines + m * num_freqs * num_polarisations + j * num_polarisations + k];
                    block[i * num_freqs + j] =  std::abs(temp);

	  	}
            }

            sum_threshold_on_block(thresholds, seqlen, sequence_lengths, block, num_freqs, num_time, flags_on_the_block, false);
            write_flags_to_the_slice_array(num_freqs, num_baselines, num_polarisations, num_time, m, k, flags_on_the_block, flags);

        }
    }
    delete[] block;
    delete[] flags_on_the_block;
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

    check_params(vis,sequence,thresholds,flags,status);	
    if (*status) return;




    const uint64_t num_baselines  = (uint64_t)(sdp_mem_shape_dim(vis, 1));
    const uint64_t num_times      = (uint64_t)(sdp_mem_shape_dim(vis, 0)/num_baselines);
    const uint64_t num_channels   = (uint64_t)sdp_mem_shape_dim(vis, 2);
    const uint64_t num_polarisations = (uint64_t)(sdp_mem_shape_dim(vis, 3));
    const uint64_t seqlen 	     = (uint64_t)sdp_mem_shape_dim(sequence, 0);
    
    if (location == SDP_MEM_GPU){

    	const uint64_t num_threads[] = {256, 1, 1};
    	const uint64_t num_blocks[] = {
		num_polarisations,
            	(num_times + num_threads[0] - 1) / num_threads[0], 
	       	1
    	};
    	const char* kernel_name = 0;
    	if (type == SDP_MEM_COMPLEX_DOUBLE)
    	{
		kernel_name = "rfi_flagger<double,double2>";
    	}
   	else if (type == SDP_MEM_COMPLEX_FLOAT)
    	{
        	kernel_name = "rfi_flagger<float,float2>";
    	}
    	else
    	{
        	*status = SDP_ERR_DATA_TYPE;
        	SDP_LOG_ERROR("Unsupported data type");
    	}
    	const void* args[] = {
        	&num_times,
		&num_baselines,
		&num_polarisations,
    		&num_channels,
    		&seqlen,
    		sdp_mem_gpu_buffer_const(sequence,status),
            	sdp_mem_gpu_buffer_const(vis, status),
            	sdp_mem_gpu_buffer_const(thresholds, status),
    		sdp_mem_gpu_buffer(flags,status),
    	};
    
    	sdp_launch_cuda_kernel(kernel_name,num_blocks, num_threads, 0, 0, args, status);
    }
    else if (location == SDP_MEM_CPU){
	 	if (type == SDP_MEM_COMPLEX_FLOAT){

			rfi_flagger(
				num_times,
				num_channels,
				seqlen,
				num_polarisations,
				num_baselines,
				(const int*)sdp_mem_data_const(sequence),
	       			(const std::complex<float>*)sdp_mem_data_const(vis),
				(const float*)sdp_mem_data_const(thresholds),
				(int*)sdp_mem_data(flags)
			);
		}
		else if (type == SDP_MEM_COMPLEX_DOUBLE){

			rfi_flagger(
				num_times,
				num_channels,
				seqlen,
				num_polarisations,
				num_baselines,
				(const int*)sdp_mem_data_const(sequence),
	       			(const std::complex<double>*)sdp_mem_data_const(vis),
				(const double*)sdp_mem_data_const(thresholds),
				(int*)sdp_mem_data(flags)
			);

		}
		else
		{
        		*status = SDP_ERR_DATA_TYPE;
        		SDP_LOG_ERROR("Unsupported data type");
		}

	   
    }
    else
    {

        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Unknown memory location for visibility data.");
        return;
    }
}
