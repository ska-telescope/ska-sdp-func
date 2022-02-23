/* See the LICENSE file at the top-level directory of this distribution. */
  
#include "utility/sdp_device_wrapper.h"

template<typename T>
__global__ void rfi_flagger(const  int num_time,const  int num_freqs,
		const int seqlen,
		const int* sequence_lengths,
	       	const T* const __restrict__ spectrogram,
		const T*const __restrict__ thresholds,
		int*  flags)
{
	
	float current_threshold = 0;
   	float tmp_sum=0.0;
    	int tid=0,did=0;
	
    	did=blockIdx.x *num_freqs + threadIdx.x; 
	tid=threadIdx.x;
	__shared__ float block[512];
	__shared__ int s_flags[512];

	if(tid<num_freqs)
	{
		block[tid]=spectrogram[did];
		s_flags[tid]=0;
		__syncthreads();
		current_threshold=thresholds[0] * sequence_lengths[0];

		if(block[tid]>current_threshold)
			s_flags[tid]=1;
		__syncthreads();
        	for (int k = 1; k < seqlen; k++)
		{
            			current_threshold = thresholds[k] * sequence_lengths[k];
				if(tid+sequence_lengths[k]<num_freqs)
				{
					tmp_sum=block[tid]+block[tid+ (int)sequence_lengths[k]/2 ];
				}
				
				__syncthreads();
				
				if(tid+sequence_lengths[k]<num_freqs)
				{
					block[tid]=tmp_sum;
					tmp_sum=0.0;
				}
				__syncthreads();
				if(block[tid]>current_threshold)
				{
					for(int m=tid;m<tid+sequence_lengths[k];m++)
					{
						s_flags[m]=1;
					}
				}
				
		}


		__syncthreads();
		flags[did]=s_flags[tid];
		__syncthreads();
		
	}
}

SDP_CUDA_KERNEL(rfi_flagger<float>)
	
