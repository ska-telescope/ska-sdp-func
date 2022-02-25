/* See the LICENSE file at the top-level directory of this distribution. */
  
#include "utility/sdp_device_wrapper.h"

template<typename T>
__global__ void rfi_flagger(const  int num_time,
		const  int num_freqs,
		const int seqlen,
		const int* sequence_lengths,
	       	const T* const __restrict__ spectrogram,
		const T*const __restrict__ thresholds,
		int*  flags)
{
	
	float current_threshold = 0;
   	float tmp_sum=0.0;
    	uint64_t did=0;
	
    	did=blockIdx.x *num_freqs + threadIdx.x; 

	__shared__ float block[256];
	__shared__ int s_flags[256];

	if(threadIdx.x<num_freqs)
	{
		block[threadIdx.x]=spectrogram[did];
		s_flags[threadIdx.x]=0;
		__syncthreads();
		current_threshold=thresholds[0] * sequence_lengths[0];

		if(block[threadIdx.x]>current_threshold)
			s_flags[threadIdx.x]=1;
		__syncthreads();
        	for (int k = 1; k < seqlen; k++)
		{
            			current_threshold = thresholds[k] * sequence_lengths[k];
				if(threadIdx.x+sequence_lengths[k]<num_freqs)
				{
					tmp_sum=block[threadIdx.x]+block[threadIdx.x+ (int)sequence_lengths[k]/2 ];
				}
				
				__syncthreads();
				
				if(threadIdx.x+sequence_lengths[k]<num_freqs)
				{
					block[threadIdx.x]=tmp_sum;
					tmp_sum=0.0;
				}
				__syncthreads();
				if(block[threadIdx.x]>current_threshold)
				{
					for(int m=threadIdx.x;m<threadIdx.x+sequence_lengths[k];m++)
					{
						s_flags[m]=1;
					}
				}
				
		}


		__syncthreads();
		flags[did]=s_flags[threadIdx.x];
		__syncthreads();
		
	}
}

SDP_CUDA_KERNEL(rfi_flagger<float>)
	
