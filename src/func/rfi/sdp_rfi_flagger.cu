/* See the LICENSE file at the top-level directory of this distribution. */
  
#include "utility/sdp_device_wrapper.h"


/*__global__
void rfi_flagger (
    const int64_t num_elements,
    const T *const __restrict__ input_a,
    const T *const __restrict__ input_b,
    T *__restrict__ output)
{
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements)
    {
        output[i] = input_a[i] + input_b[i];
    }
}
*/

//function to reorganise the threasholds 
__device__ void write_flags_to_the_slice_array_device(const int num_freqs, const int num_baselines, const int num_polarisation,const int num_time,
                                    int baseline_id, int polarisation_id, int* flags_on_the_block, int* flags){


    int i=blockIdx.x *blockDim.x + threadIdx.x; 
    if(i<num_time)
    {
        for (int j = 0; j < num_freqs; j++){
            flags[i * num_freqs * num_baselines * num_polarisation + baseline_id * num_freqs * num_polarisation
            + j * num_polarisation + polarisation_id] = flags_on_the_block[i * num_freqs + j];
        }
    }

}

//function computs the sumthresholds and flags the block of the data if thresholding condition is not met
__device__ void sum_threshold_on_block_device(const float* thresholds, const int seqlen, const int* sequence_lengths, float *block, const int num_freqs,const int num_time,int *flags_on_block, int freq_or_time)
{

    float current_threshold = 0;
    int current_seqlen = 0;
    float sum = 0;

    int i=0;

    i=blockIdx.x *blockDim.x + threadIdx.x; 
    //if(i<num_time)
    {		    
    if (!freq_or_time){

        for (int k = 0; k < seqlen; k++){
            current_seqlen = sequence_lengths[k];
            current_threshold = thresholds[k] * current_seqlen;

	   if(i<num_time-current_seqlen)
	    {


                for (int j = 0; j < num_freqs; j++)
		{
                    sum = 0;
                    for (int m = 0; m < current_seqlen; m++){

                        sum = sum + block[(i + m) * num_freqs + j];
                    }
                    
		    if (sum > current_threshold)
		    {
                        for (int m = 0; m < current_seqlen; m++){

                            flags_on_block[(i + m) * num_freqs + j] = 1;
                        }
                    }
                }
            }

        }
    } else if (freq_or_time){


        for (int k = 0; k < seqlen; k++)
	{

		
            	current_seqlen = sequence_lengths[k];
            	current_threshold = thresholds[k] * current_seqlen;
            	for (int i = 0; i < num_time; i++)
		{
                	for (int j = 0; j < num_freqs - current_seqlen; j++)
			{
                		sum = 0;
                    		for (int m = 0; m < current_seqlen; m++)
				{
                        		if (flags_on_block[i * num_freqs + j + m] == 1)
					{
                            			block[i * num_freqs + j + m] = thresholds[k];
                        		}

                        sum = sum + block[i * num_freqs + j + m];
                    }

                    if (sum > current_threshold)
		    {
                        for (int m = 0; m < current_seqlen; m++)
			{
                            flags_on_block[i  * num_freqs + j + m] = 1;
                        }
                    }
                }
            }

        }
    }
    }
}

template<typename T>
__global__ void rfi_flagger(const  int num_time,const  int num_freqs, const int num_baselines,const int num_polarisations,
		const int seqlen,
		const int* sequence_lengths,
	       	const T* const __restrict__ spectrogram,
		const T*const __restrict__ thresholds,
		T*  block,
		int*  flags,
		int* flags_on_the_block)
{
	
	int i=blockIdx.x *blockDim.x + threadIdx.x; 
	if(i<num_time)
	{



    		for (int m = 0; m < num_baselines; m++)
		{
        		for (int k = 0; k < num_polarisations; k++)
			{
				
				//copy data for each baseline and polarisation to a temp array and use that array for further processing. 
				//This steps mainly reorganise the data in the format required for processing. The data read from the 
				//casa table is oragnised differently and cannot be used directly for processing.
            			//for (int i = 0; i < num_time; i++)
	    			{
                			for (int j = 0; j < num_freqs; j++)
					{
                				block[k*num_baselines*num_freqs*num_time+m*num_freqs*num_time+i * num_freqs + j] = spectrogram[i* num_baselines * num_freqs * num_polarisations + m * num_freqs * num_polarisations + j * num_polarisations + k];
                			}
            			}
			}
		}
	}
		__syncthreads();

	if(i<num_time)
	{
    		for (int m = 0; m < num_baselines; m++)
		{
        		for (int k = 0; k < num_polarisations; k++)
			{

				//call device function to compute the sum threashold and flag the sequences which are above the threshold.
            			sum_threshold_on_block_device(thresholds, seqlen, sequence_lengths, &block[k*num_baselines*num_freqs*num_time+m*num_freqs*num_time], num_freqs, num_time, &flags_on_the_block[k*num_baselines*num_freqs*num_time+m*num_freqs*num_time], false);  
			}
		}


	}
	__syncthreads();

	for (int m = 0; m < num_baselines; m++)
        {
                        for (int k = 0; k < num_polarisations; k++)
                        {

				write_flags_to_the_slice_array_device(num_freqs, num_baselines, num_polarisations, num_time, m, k, &flags_on_the_block[k*num_baselines*num_freqs*num_time+m*num_freqs*num_time], flags);
			}
	}
	__syncthreads();
}

SDP_CUDA_KERNEL(rfi_flagger<float>)
	
