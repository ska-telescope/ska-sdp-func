// Copyright 2021 High Performance Computing Research Laboratory, Auckland University of Technology (AUT)

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/*****************************************************************************
 * Msmfsfunctionsdevice.cu
 * Andrew Ensor
 * CUDA device functions for the MSMFS cleaning algorithm
 *****************************************************************************/

#include "Msmfsfunctionsdevice.h"

/*****************************************************************************
 * Calculates the (L1-normalised) gaussian function at a specified point x_squared
 * Note uses the 2D gaussian normalisation 1/((PRECISION)TWO_PI*variance) in place of 1D 1/sqrt((PRECISION)TWO_PI*variance)
 *****************************************************************************/
template<typename PRECISION>
__device__ PRECISION normalised_gaussian(PRECISION x_squared, PRECISION variance)
{
    if (variance > 0)
        return 1/((PRECISION)TWO_PI*variance)*exp((PRECISION)(-0.5)*x_squared/variance);
    else
       return (x_squared>0) ? 0.0 : 1.0;
}


/*****************************************************************************
 * Calculates the (Lmax-normalised) gaussian function at a specified point x_squared
 * Note this can be used in combination with normalised_gaussian to avoid normalising twice
 *****************************************************************************/
template<typename PRECISION>
__device__ PRECISION gaussian(PRECISION x_squared, PRECISION variance)
{
    if (variance > 0)
        return exp((PRECISION)(-0.5)*x_squared/variance);
    else
       return (x_squared>0) ? 0.0 : 1.0;
}


/*****************************************************************************
 * Device function that performs the one-dimensional horizontal convolution for
 * convolve_horizontal and double_convolve_horizontal
 *****************************************************************************/
template<typename PRECISION>
__device__ void convolve_horizontal_device
    (
    PRECISION *image_device, // input flat array containing image
    PRECISION *convolved_device, // output flat array containing convolved image except at left and right border
    const unsigned int convolved_width, // width of resulting horizontally convolved image
    const unsigned int convolved_height, // height of image_device and resulting horizontally convolved image
    const unsigned int image_border, // left and right border regions of image_device
    const unsigned int convolution_support, // convolution half support
    const PRECISION variance // variance used for calculating gaussian kernel
    )
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i<convolved_width && j<convolved_height)
    {
        unsigned int image_width = convolved_width + 2*image_border;
        unsigned int image_index = image_width*j + (image_border+i);
        // exploit the horizontal symmetry of the gaussian function
        PRECISION convolution_sum = image_device[image_index]*normalised_gaussian((PRECISION)0, variance);
        for (int x=1; x<=(int)convolution_support; x++)
        {
            PRECISION left_image_pixel = (image_border+i>=x) ? image_device[image_index-x] : 0; // pad with 0 outside image_border
            PRECISION right_image_pixel = (image_border+i+x<image_width) ? image_device[image_index+x] : 0; // pad with 0 outside image_border
            convolution_sum += (left_image_pixel+right_image_pixel)*normalised_gaussian((PRECISION)(x*x), variance);
        }
        convolved_device[convolved_width*j+i] = convolution_sum;
    }
}


/*****************************************************************************
 * Performs a one-dimensional horizontal convolution at each non-border column of image
 * Note this presumes image_device has width convolved_width+2*image_border
 * so there are additional image_border columns in image_device at left and right
 * and presumes image_device has same height convolved_height as convolved_device
 * Parallelised so each CUDA thread processes a single pixel convolution result
 *****************************************************************************/
template<typename PRECISION>
__global__ void convolve_horizontal
    (
    PRECISION *image_device, // input flat array containing image
    PRECISION *convolved_device, // output flat array containing convolved image except at left and right border
    const unsigned int convolved_width, // width of resulting horizontally convolved image
    const unsigned int convolved_height, // height of image_device and resulting horizontally convolved image
    const unsigned int image_border, // left and right border regions of image_device
    const unsigned int *convolution_support_device, // supports required for each gaussian kernel shape for convolution
    const PRECISION *variances_device, // variances used for calculating gaussian kernel, one for each of num_scales
    const unsigned int scale_index
    )
{
    convolve_horizontal_device(image_device, convolved_device, convolved_width, convolved_height,
        image_border, convolution_support_device[scale_index], variances_device[scale_index]);
}

template __global__ void convolve_horizontal<float>(float*, float*, const unsigned int, const unsigned int, const unsigned int, const unsigned int*, const float*, const unsigned int);
template __global__ void convolve_horizontal<double>(double*, double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int*, const double*, const unsigned int);


/*****************************************************************************
 * Same as convolve_horizontal but accepts two variances and uses their sum as the variance for convolution
 * Note first scale index presumed to be known on host so passed by value
 * and second scale index presumed on device so passed by reference
 * Parallelised so each CUDA thread processes a single pixel convolution result
 *****************************************************************************/
template<typename PRECISION>
__global__ void double_convolve_horizontal
    (
    PRECISION *image_device, // input flat array containing image
    PRECISION *convolved_device, // output flat array containing convolved image except at left and right border
    const unsigned int convolved_width, // width of resulting horizontally convolved image
    const unsigned int convolved_height, // height of image_device and resulting horizontally convolved image
    const unsigned int image_border, // left and right border regions of image_device
    const unsigned int *double_convolution_support_device, // supports required for each gaussian kernel shape for double convolution
    const unsigned int num_scales,
    const PRECISION *variances_device, // variances used for calculating gaussian kernel, one for each of num_scales
    const unsigned int scale_index1,
    const unsigned int *scale_index2_device // second scale index already on device so passed by reference
    )
{
    convolve_horizontal_device(image_device, convolved_device, convolved_width, convolved_height,
        image_border, double_convolution_support_device[num_scales*scale_index1+(*scale_index2_device)],
        variances_device[scale_index1]+variances_device[*scale_index2_device]);
}

template __global__ void double_convolve_horizontal<float>(float*, float*, const unsigned int, const unsigned int, const unsigned int, const unsigned int*, const unsigned int, const float*, const unsigned int, const unsigned int*);
template __global__ void double_convolve_horizontal<double>(double*, double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int*, const unsigned int, const double*, const unsigned int, const unsigned int*);


/*****************************************************************************
 * Device function that performs the one-dimensional vertical convolution for
 * convolve_vertical and double_convolve_vertical
 *****************************************************************************/
template<typename PRECISION>
__device__ void convolve_vertical_device
    (
    PRECISION *horiz_convolved_device, // input flat array containing horizontally convolved image
    PRECISION *convolved_device, // output flat array containing vertically convolved image except at top and bottom border
    const unsigned int convolved_width, // width of image_device and resulting vertically convolved image
    const unsigned int convolved_height, // height of resulting vertically convolved image
    const unsigned int image_border, // top and bottom border regions of horiz_convolved_device
    const unsigned int convolution_support, // convolution half support
    const PRECISION variance // variance used for calculating gaussian kernel
    )
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i<convolved_width && j<convolved_height)
    {
        unsigned int image_width = convolved_width;
        unsigned int image_height = convolved_height + 2*image_border;
        unsigned int image_index = image_width*(image_border+j) + i;
        // exploit the vertical symmetry of the gaussian function
        PRECISION convolution_sum = horiz_convolved_device[image_index]*gaussian((PRECISION)0, variance);
        for (int y=1; y<=(int)convolution_support; y++)
        {
            PRECISION upper_image_pixel = (image_border+j>=y) ? horiz_convolved_device[image_index-image_width*y] : 0; // pad with 0 outside image_border
            PRECISION lower_image_pixel = (image_border+j+y<image_height) ? horiz_convolved_device[image_index+image_width*y] : 0; // pad with 0 outside image_border
            convolution_sum += (upper_image_pixel+lower_image_pixel)*gaussian((PRECISION)(y*y), variance);
        }
        convolved_device[convolved_width*j+i] = convolution_sum;
    }
}


/*****************************************************************************
 * Performs a one-dimensional vertical convolution at each non-border row of image
 * using an unnormalised gaussian as presumes horiz_convolved_device has already had a normalised gaussian applied
 * Note this presumes horiz_convolved_device has same width convolved_width as convolved_device
 * and presumes horiz_convolved_device has height convolved_height+2*image_border
 * with additional image_border rows at top and bottom
 * Parallelised so each CUDA thread processes a single pixel convolution result
 *****************************************************************************/
template<typename PRECISION>
__global__ void convolve_vertical
    (
    PRECISION *horiz_convolved_device, // input flat array containing horizontally convolved image
    PRECISION *convolved_device, // output flat array containing vertically convolved image except at top and bottom border
    const unsigned int convolved_width, // width of image_device and resulting vertically convolved image
    const unsigned int convolved_height, // height of resulting vertically convolved image
    const unsigned int image_border, // top and bottom border regions of horiz_convolved_device
    const unsigned int *convolution_support_device, // supports required for each gaussian kernel shape for convolution
    const PRECISION *variances_device, // variances used for calculating gaussian kernel, one for each of num_scales
    const unsigned int scale_index
    )
{
    convolve_vertical_device(horiz_convolved_device, convolved_device, convolved_width, convolved_height,
        image_border, convolution_support_device[scale_index], variances_device[scale_index]);
}

template __global__ void convolve_vertical<float>(float*, float*, const unsigned int, const unsigned int, const unsigned int, const unsigned int*, const float*, const unsigned int);
template __global__ void convolve_vertical<double>(double*, double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int*, const double*, const unsigned int);


/*****************************************************************************
 * Same as convolve_vertical but accepts two variances and uses their sum as the variance for convolution
 * Note first scale index presumed to be known on host so passed by value
 * and second scale index presumed on device so passed by reference
 * Parallelised so each CUDA thread processes a single pixel convolution result
 *****************************************************************************/
template<typename PRECISION>
__global__ void double_convolve_vertical
    (
    PRECISION *horiz_convolved_device, // input flat array containing horizontally convolved image
    PRECISION *convolved_device, // output flat array containing vertically convolved image except at top and bottom border
    const unsigned int convolved_width, // width of image_device and resulting vertically convolved image
    const unsigned int convolved_height, // height of resulting vertically convolved image
    const unsigned int image_border, // top and bottom border regions of horiz_convolved_device
    const unsigned int *double_convolution_support_device, // supports required for each gaussian kernel shape for double convolution
    const unsigned int num_scales,
    const PRECISION *variances_device, // variances used for calculating gaussian kernel, one for each of num_scales
    const unsigned int scale_index1,
    const unsigned int *scale_index2_device // second scale index already on device so passed by reference
    )
{
    convolve_vertical_device(horiz_convolved_device, convolved_device, convolved_width, convolved_height,
        image_border, double_convolution_support_device[num_scales*scale_index1+(*scale_index2_device)],
        variances_device[scale_index1]+variances_device[*scale_index2_device]);
}

template __global__ void double_convolve_vertical<float>(float*, float*, const unsigned int, const unsigned int, const unsigned int, const unsigned int*, const unsigned int, const float*, const unsigned int, const unsigned int*);
template __global__ void double_convolve_vertical<double>(double*, double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int*, const unsigned int, const double*, const unsigned int, const unsigned int*);


/*****************************************************************************
 * Calculates each entry for the hessian matrices by performing a two-dimensional convolution at
 * the centre of one psf at one scale to get one of the entries
 * Note this presumes the psf is at least as large as the convolution support
 * Parallelised so each CUDA thread processes a single convolution
 *****************************************************************************/
template<typename PRECISION>
__global__ void calculate_hessian_entries
    (
    PRECISION *psf_moment_images_device, // input flat array containing input Taylor coefficient psf images to be convolved
    const unsigned int psf_moment_size, // one dimensional size of psf, assumed square
    const unsigned int num_psf,
    PRECISION *hessian_entries_device, // output flat array to hold entries for the hessian matrices
    const unsigned int num_scales,
    PRECISION *variances_host, // variances used for calculating gaussian kernel, one for each of num_scales
    unsigned int *double_convolution_support_device // supports required for each gaussian kernel shape for double convolution
    )
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int num_entries_calculate = num_scales*num_psf;
    if (i<num_entries_calculate)
    {
        unsigned int psf_index = (unsigned int)i / num_scales;
        unsigned int scale_index = (unsigned int)i % num_scales;
        PRECISION variance = 2*variances_host[scale_index]; // note factor of 2 as hessian uses doubly convolved kernels
        // calculate a suitable convolution support based on the requested convolution accuracy
        unsigned int convolution_support = double_convolution_support_device[num_scales*scale_index+scale_index];
        // determine the x,y centre of the correct psf image
        unsigned int image_index = psf_moment_size*psf_moment_size*psf_index
            + (psf_moment_size+1)*(psf_moment_size/2);
        // note the following nested loops have not exploited the symmetry of the gaussian function
        // which could be used to improve performance of the convolution here
        PRECISION convolution_sum = 0.0;
        unsigned int convolution_bound = min((psf_moment_size-1)/2, convolution_support);
        for (int y=-(int)convolution_bound; y<=(int)convolution_bound; y++)
        {
            for (int x=-(int)convolution_bound; x<=(int)convolution_bound; x++)
            {
                convolution_sum
                    += psf_moment_images_device[image_index+psf_moment_size*y+x]*normalised_gaussian((PRECISION)(x*x+y*y), variance);
            }
        }
        hessian_entries_device[i] = convolution_sum;
    }
}

template __global__ void calculate_hessian_entries<float>(float*, const unsigned int, const unsigned int, float*, const unsigned int, float*, unsigned int*);
template __global__ void calculate_hessian_entries<double>(double*, const unsigned int, const unsigned int, double*, const unsigned int, double*, unsigned int*);


/*****************************************************************************
 * Populates the hessian matrices with the entries given by calculate_hessian_entries
 * The (t,t') entry of s-th hessian matrix uses the s-th scale convolution of the t+t'-th psf
 * Note this presumes hessian_entries_device has entries arranged with all scales for each of
 * the num_psf psf, where num_psf is presumed to be 2*num_taylor-1
 * Note this presumes hessian_matrix_device has entries arranged row by row for each of
 * the num_scales hessian matrices 
 * Parallelised so each CUDA thread processes a single entry of one of the hessian matrices
 *****************************************************************************/
template<typename PRECISION>
__global__ void populate_hessian_matrices
    (
    PRECISION *hessian_entries_device, // input flat array containing the num_scales*num_psf entries
    const unsigned int num_scales,
    const unsigned int num_taylor,
    PRECISION *hessian_matrix_device // output flat array containing the num_scales hessian matrices, each of size num_taylor*num_taylor
    )
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int hessian_matrix_size = num_taylor*num_taylor;
    const unsigned int num_matrix_entries = hessian_matrix_size*num_scales;
    if (i < num_matrix_entries)
    {
        unsigned int scale_index = (unsigned int)i / hessian_matrix_size;
        unsigned int matrix_index = (unsigned int)i % hessian_matrix_size;
        unsigned int row_index = matrix_index / num_taylor;
        unsigned int col_index = matrix_index % num_taylor;
        unsigned int psf_index = row_index + col_index;
        hessian_matrix_device[i] = hessian_entries_device[num_scales*psf_index+scale_index];
    }
}

template __global__ void populate_hessian_matrices<float>(float*, const unsigned int, const unsigned int, float*);
template __global__ void populate_hessian_matrices<double>(double*, const unsigned int, const unsigned int, double*);


/*****************************************************************************
 * Populates the principal solution image with the maximum principal solution with bias applied,
 * across all scales at each point for t=0
 * Note this presumes smpsol_max_device and smpsol_scale_device have size scale_moment_size*scale_moment_size
 * Parallelised so each CUDA thread processes a single point of principal solution across all scales
 *****************************************************************************/
template<typename PRECISION>
__global__ void calculate_principal_solution_max_scale
    (
    PRECISION *scale_moment_residuals_device, // input flat array that holds all convolved scale moment residuals
    const unsigned int num_scales,
    const unsigned int num_taylor,
    const unsigned int scale_moment_size, // one dimensional size of scale moment residuals, assumed square
    PRECISION *inverse_hessian_matrices_device, // input flat array that holds all inverse hessian matrices
    const PRECISION* scale_bias, // bias multiplicative factor to favour cleaning with smaller scales
    PRECISION *smpsol_max_device, // output entries for the smpsol that has abs max at each point
    unsigned int* smpsol_scale_device // output scale indices for the abs max smpsol at each point
    )
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i<scale_moment_size && j<scale_moment_size)
    {
        unsigned int point_index = scale_moment_size*j + i;
        PRECISION smpsol_max = (PRECISION)0.0;
        unsigned int smpsol_scale = 0;
        for (unsigned int scale_index=0; scale_index<num_scales; scale_index++)
        {
            // calculate the entry for the principal solution at this scale and t=0 as a sum of products
            PRECISION smpsol_value = (PRECISION)0.0;
            for (unsigned int taylor_index=0; taylor_index<num_taylor; taylor_index++)
            {
                unsigned int inverse_hessian_index = num_taylor*num_taylor*scale_index + taylor_index*num_taylor + 0;
                unsigned int scale_moment_residuals_index
                    = scale_moment_size*scale_moment_size*(taylor_index*num_scales+scale_index) + point_index;
                smpsol_value
                    += inverse_hessian_matrices_device[inverse_hessian_index] * scale_moment_residuals_device[scale_moment_residuals_index];
            }
            // apply bias for this scale to the principal solution
            smpsol_value = abs(smpsol_value*scale_bias[scale_index]);
            // take the largest absolute value of the principal solution across all scales once scale bias applied
            if (smpsol_value > smpsol_max)
            {
                smpsol_max = smpsol_value;
                smpsol_scale = scale_index;
            }
        }  
        smpsol_max_device[point_index] = smpsol_max;
        smpsol_scale_device[point_index] = smpsol_scale;
    }
}

template __global__ void calculate_principal_solution_max_scale<float>(float*, const unsigned int, const unsigned int, const unsigned int, float*, const float*, float*, unsigned int*);
template __global__ void calculate_principal_solution_max_scale<double>(double*, const unsigned int, const unsigned int, const unsigned int, double*, const double*, double*, unsigned int*);


/*****************************************************************************
 * Find the larger of the array entry held at address with the value at next_entry
 * and atomically places the larger of the two into address
 * Avoids race conditions where threads from another warp (or block)
 * concurrently also tries to change the array entry at the same address
 * Returns the old value that was at the address 
 *****************************************************************************/
__device__ Array_entry atomic_max_entry(Array_entry* address, Array_entry next_entry)
{
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull; // old array entry before the max check
    unsigned long long assumed;
    do
    {
        assumed = old; // assume the old array entry is not being concurrently modified by another thread
        float assumed_value = ((Array_entry*)&assumed)->value;
        if (next_entry.value > assumed_value)
        {
            // atomically try to update value at address if old hasn't yet been changed (by another thread)
            old = atomicCAS(address_as_ull, assumed, *(unsigned long long*)&next_entry);
        }
    }
    while (assumed != old); // repeat attempted update if assumed did get changed by another thread
    return *(Array_entry*)(&old);
}


/*****************************************************************************
 * Finds the maximum value and its index location in a one-dimensional array
 * Loosely follows the reduction approach described in
 * https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
 * with grid-strided loops as described in
 * https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
 * Parallelised so each CUDA thread processes all array entries that are separated
 * by the same grid stride
 *****************************************************************************/
#define FOUR_BYTE_FULL_MASK 0xffffffff

template<typename PRECISION>
__global__ void find_max_entry_grid_strided_reduction
    (
    PRECISION *input_array_device, // input 1D array for which the location of maximum is to be found
    unsigned int input_num_values, // number of values in the input array (might be many times number cuda threads)
    Array_entry* max_entry_device // output array entry where the maximum is found in input_array_device
    )
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= input_num_values)
        return;
    const unsigned int grid_stride = blockDim.x*gridDim.x; // total number of cuda threads
    // first have this thread find the minimum across all array entries spaced apart by grid_stride
    Array_entry maximum_entry =
    {
        .value = (float)input_array_device[i], // typecast to float to facilitate 32-bit shift down op
        .index = (uint32_t)i
    };
    for (uint32_t next_index=(uint32_t)(i+grid_stride); next_index<input_num_values; next_index+=grid_stride)
    {
        float next_value = (float)input_array_device[next_index];
        if (next_value > maximum_entry.value)
        {
            maximum_entry.value = next_value;
            maximum_entry.index = next_index;
        }
    }

    // allocate shared memory for each of the (upto 1024/32) warps in this thread block to store max for each warp
    static __shared__ Array_entry block_maxima[32]; // presumed at most 32 warps in this block (max block size 1024 and warp size 32)

    // next have the warp for this thread find the maximum among all its (32) threads
    int thread_lane = threadIdx.x % warpSize; // thread lane within this warp
    int warp_identifier = threadIdx.x / warpSize; // unique identifier for this warp within this block
    for (int offset = warpSize/2; offset>0; offset/=2)
    {
        // get the maximum_value and maximum_index found by another thread in this warp
        // TODO recheck carefully this works when last warp extends beyond end, technically shuffle has undefined behaviour
        float next_value = __shfl_down_sync(FOUR_BYTE_FULL_MASK, maximum_entry.value, offset); // 32 bit shuffle down
        uint32_t next_index = __shfl_down_sync(FOUR_BYTE_FULL_MASK, maximum_entry.index, offset); // 32 bit shuffle down
        if (next_value > maximum_entry.value)
        {
            maximum_entry.value = next_value;
            maximum_entry.index = next_index;
        }
    }
    // note the first thread in this warp now has the maximum entry for this block
    // so have it store the maximum it found in the shared memory
    if (thread_lane == 0)
    {
        block_maxima[warp_identifier] = maximum_entry;
    }
    __syncthreads(); // synchronize all threads in this thread block to ensure each warp has updated block_maxima

    // have threads in the first warp within this block find the maximum in block_maxima across this block of threads
    if (warp_identifier == 0)
    {
        // have this thread in first warp responsible for holding up to one of the block_maxima entries
        // presumes the warp has at least num_warps_in_block threads
        int num_warps_in_block = blockDim.x/warpSize; // presumes this is within size of block_maxima
        if (thread_lane < num_warps_in_block)
        {
            maximum_entry = block_maxima[thread_lane];
        } // else this thread can just hold its old (now non-maximum) maximum_entry
        for (int offset = warpSize/2; offset>0; offset/=2)
        {
            // get the maximum_value and maximum_index held by another thread in this warp
            float next_value = __shfl_down_sync(FOUR_BYTE_FULL_MASK, maximum_entry.value, offset); // 32 bit shuffle down
            uint32_t next_index = __shfl_down_sync(FOUR_BYTE_FULL_MASK, maximum_entry.index, offset); // 32 bit shuffle down
            if (next_value > maximum_entry.value)
            {
                maximum_entry.value = next_value;
                maximum_entry.index = next_index;
            }      
        }
        // note the first thread in this (first) warp now has the maximum entry for this block
    }

    // finally have the first thread in this block compare its maximum with that of other blocks
    if (threadIdx.x == 0)
    {
        atomic_max_entry(max_entry_device, maximum_entry);
    }
}

template __global__ void find_max_entry_grid_strided_reduction<float>(float*, unsigned int, Array_entry*);
template __global__ void find_max_entry_grid_strided_reduction<double>(double*, unsigned int, Array_entry*);


/*****************************************************************************
 * Calculates the principal solution at the peak, known as mval in the MSMFS algorithm
 * Note this presumes mval_device has size num_taylor
 * Parallelised so each CUDA thread processes a single taylor term value of principal solution at peak
 *****************************************************************************/
template<typename PRECISION>
__global__ void calculate_principal_solution_at_peak
    (
    PRECISION *scale_moment_residuals_device, // input flat array that holds all convolved scale moment residuals
    const unsigned int num_scales,
    const unsigned int num_taylor,
    const unsigned int scale_moment_size, // one dimensional size of scale moment residuals, assumed square
    PRECISION *inverse_hessian_matrices_device, // input flat array that holds all inverse hessian matrices
    unsigned int *smpsol_scale_device, // input flat array that holds scale indices for the abs max smpsol at each point
    unsigned int *smpsol_max_index_device, // index in smpsol_scale_device and in one of the scale_moment_residuals_device at the peak
    PRECISION* peak_point_smpsol_device, // output principal solution at the peak
    unsigned int *peak_point_scale_device, // output scale index at the peak
    unsigned int *peak_point_index_device // output array offset index of point at the peak
    )
{
    int i = blockIdx.x*blockDim.x + threadIdx.x; // taylor term for which mval is being evaluated
    if (i<num_taylor)
    {
        unsigned int point_index = *smpsol_max_index_device;
        unsigned int scale_index = smpsol_scale_device[point_index];
        PRECISION mval = (PRECISION)0.0;
        for (unsigned int taylor_index=0; taylor_index<num_taylor; taylor_index++)
        {
            unsigned int inverse_hessian_index = num_taylor*num_taylor*scale_index + taylor_index*num_taylor + i;
            unsigned int scale_moment_residuals_index
                = scale_moment_size*scale_moment_size*(taylor_index*num_scales+scale_index) + point_index;
            mval += inverse_hessian_matrices_device[inverse_hessian_index] * scale_moment_residuals_device[scale_moment_residuals_index];
        }
        peak_point_smpsol_device[i] = mval;
        // have first thread also output the scale and array offset index at the peak
        if (i == 0)
        {
            *peak_point_scale_device = scale_index;
            *peak_point_index_device = point_index;
        }
    }
}

template __global__ void calculate_principal_solution_at_peak<float>(float*, const unsigned int, const unsigned int, const unsigned int, float*, unsigned int*, unsigned int*, float*, unsigned int*, unsigned int*);
template __global__ void calculate_principal_solution_at_peak<double>(double*, const unsigned int, const unsigned int, const unsigned int, double*, unsigned int*, unsigned int*, double*, unsigned int*, unsigned int*);


/*****************************************************************************
 * Subtracts the (doubly) convolved psf scaled by the peak point and clean_loop_gain
 * from the scale_moment_residual_device centred at peak_point_index
 * Note this presumes scale_moment_residual_device has size scale_moment_size*scale_moment_size
 * and psf_convolved_device has size psf_convolved_size*psf_convolved_size
 * and 0<=peak_point_index<scale_moment_size*scale_moment_size
 * Parallelised so each CUDA thread processes the scaled subtraction at a single point of scale_moment_residual_device
 *****************************************************************************/
template<typename PRECISION>
__global__ void subtract_psf_convolved_from_scale_moment_residual
    (
    PRECISION *scale_moment_residual_device, // inout flat array that holds one convolved scale moment residual
    const unsigned int scale_moment_size, // one dimensional size of scale moment residuals, assumed square
    PRECISION *psf_convolved_device, // input flat array containing one (doubly) convolved psf image
    const unsigned int psf_convolved_size, // one dimensional size of convolved psf, assumed square
    PRECISION *peak_point_smpsol_device, // principal solution at the peak
    unsigned int taylor_index,
    unsigned int *peak_point_index_device, // input array offset index of point at the peak
    PRECISION clean_loop_gain // loop gain fraction of peak point to clean from the peak each minor cycle
    )
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i<psf_convolved_size && j<psf_convolved_size)
    {
        int peak_point_x = (int)((*peak_point_index_device) % scale_moment_size);
        int peak_point_y = (int)((*peak_point_index_device) / scale_moment_size);
        unsigned int psf_convolved_half_size = psf_convolved_size / 2;
        int point_x = (int)peak_point_x + (i - psf_convolved_half_size);
        int point_y = (int)peak_point_y + (j - psf_convolved_half_size);
        if (point_x>=0 && point_x<scale_moment_size && point_y>=0 && point_y<scale_moment_size)
        {
            PRECISION peak_removal_value = peak_point_smpsol_device[taylor_index] * psf_convolved_device[psf_convolved_size*j+i]
                * clean_loop_gain;
            scale_moment_residual_device[scale_moment_size*point_y+point_x] -= peak_removal_value;
        }
    }
}

template __global__ void subtract_psf_convolved_from_scale_moment_residual<float>(float*, const unsigned int, float*, const unsigned int, float*, unsigned int, unsigned int*, float);
template __global__ void subtract_psf_convolved_from_scale_moment_residual<double>(double*, const unsigned int, double*, const unsigned int, double*, unsigned int, unsigned int*, double);


/*****************************************************************************
 * Adds the specified source to the scale moment model, either by adjusting the inensity of an
 * existing source already in the model or else by adding it as a new source appended to gaussian_sources_device
 * Uses grid-strided loops as described in
 * https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
 * Parallelised so each CUDA thread processes all sources in the model that are separated
 * by the same grid stride
 * For convenient presumes there is only one block of threads in the cuda grid so no need to check across multiple blocks
 *****************************************************************************/
template<typename PRECISION>
__global__ void add_source_to_model_grid_strided_reduction
    (
    Gaussian_source<PRECISION> *gaussian_sources_device, // inout list of sources that have distinct scales/positions
    unsigned int *num_gaussian_sources_device, // inout current number of sources in gaussian_sources_device
    PRECISION *peak_point_smpsol_device, // input principal solution at the peak
    const unsigned int num_taylor,
    PRECISION *variances_device, // variances used for calculating gaussian kernel, one for each of num_scales
    unsigned int *peak_point_scale_device, // input scale index at the peak
    unsigned int *peak_point_index_device, // input array offset index of point at the peak
    PRECISION clean_loop_gain, // loop gain fraction of peak point to clean from the peak each minor cycle
    bool *is_existing_source_device // output flag whether the source added was found already in model
    )
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=(int)*num_gaussian_sources_device && (*num_gaussian_sources_device>0 || i!=0)) // note special case if no sources in model
        return;
    const unsigned int grid_stride = blockDim.x*gridDim.x; // total number of cuda threads, presumed gridDim.x==1
    // first have this thread search whether gaussian_sources_device already contains a source at this scale and this position
    // at all source model entries spaced apart by grid_stride
    for (uint32_t next_index=(uint32_t)i; next_index<(uint32_t)*num_gaussian_sources_device; next_index+=(uint32_t)grid_stride)
    {
        Gaussian_source<PRECISION> next_source = (Gaussian_source<PRECISION>)gaussian_sources_device[next_index];
        if (next_source.index==*peak_point_index_device && next_source.variance==variances_device[*peak_point_scale_device])
        {
            // a source with this position and scale already exists in the scale moment model
            // so the existing source will be updated
            // note each scale/position should be unique in gaussian_sources_device so at most one thread has match
            for (unsigned int taylor_index=0; taylor_index<num_taylor; taylor_index++)
            {
                PRECISION source_intensity = peak_point_smpsol_device[taylor_index] * clean_loop_gain;
                gaussian_sources_device[next_index].intensities[taylor_index] += source_intensity;
            }
            *is_existing_source_device = true;
        }
    }

    __syncthreads(); // synchronize all threads in this thread block to ensure each thread finished search

    // finally have the first thread in this block append the source to gaussian_sources_device if it wasn't found
    if (threadIdx.x==0 && !*is_existing_source_device)
    {
        // this is a new source so append to the end of scale moment model as a new source
        gaussian_sources_device[*num_gaussian_sources_device].index = *peak_point_index_device;
        gaussian_sources_device[*num_gaussian_sources_device].variance = variances_device[*peak_point_scale_device];
        // update the existing source with the intensity of the new source
        for (unsigned int taylor_index=0; taylor_index<num_taylor; taylor_index++)
        {
            PRECISION source_intensity = peak_point_smpsol_device[taylor_index] * clean_loop_gain;
            gaussian_sources_device[*num_gaussian_sources_device].intensities[taylor_index] = source_intensity;
        }
        *num_gaussian_sources_device = *num_gaussian_sources_device+1;
    }
}

template __global__ void add_source_to_model_grid_strided_reduction<float>(Gaussian_source<float>*, unsigned int*, float*, const unsigned int, float*, unsigned int*, unsigned int*, float, bool*);
template __global__ void add_source_to_model_grid_strided_reduction<double>(Gaussian_source<double>*, unsigned int*, double*, const unsigned int, double*, unsigned int*, unsigned int*, double, bool*);
