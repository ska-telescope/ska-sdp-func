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
 * Msmfssimpletest.cu
 * Andrew Ensor
 * C with C++ templates/CUDA code for preparing simple test dirty moment images and psf images
 *
 *****************************************************************************/

#include "Msmfssimpletest.h"

/*****************************************************************************
 * Temporary utility function that places a gaussian source with specified
 * amplitude, centre and size on the image
 *****************************************************************************/
template<typename PRECISION>
void add_source_to_image(PRECISION *image, unsigned int image_size,
    PRECISION amplitude, PRECISION variance,
    unsigned int x_centre, unsigned int y_centre, unsigned int half_source_size)
{
    unsigned int x_min = (unsigned int)max((int)(x_centre-half_source_size), 0);
    unsigned int x_max = (unsigned int)min((int)(x_centre+half_source_size), (int)(image_size-1));
    unsigned int y_min = (unsigned int)max((int)(y_centre-half_source_size), 0);
    unsigned int y_max = (unsigned int)min((int)(y_centre+half_source_size), (int)(image_size-1));
    for (unsigned int y=y_min; y<=y_max; y++)
    {
        for (unsigned int x=x_min; x<=x_max; x++)
        {
            PRECISION r_squared = (PRECISION)((x-x_centre)*(x-x_centre)+(y-y_centre)*(y-y_centre));
            unsigned int index = image_size*y + x;
            if (variance > 0)
            {
                image[index] += amplitude/((PRECISION)TWO_PI*variance)*exp((PRECISION)(-0.5)*r_squared/variance);
            }
            else
            {
                image[index] += (r_squared>0) ? 0.0 : amplitude;
            }
        }
    }
}

template void add_source_to_image<float>(float*, unsigned int, float, float, unsigned int, unsigned int, unsigned int);
template void add_source_to_image<double>(double*, unsigned int, double, double, unsigned int, unsigned int, unsigned int);


/*****************************************************************************
 * Msmfs function that allocates and clears the data structure that will
 * hold all the dirty moment images on the device
 * Note should be paired with a later call to free_simple_dirty_image
 *****************************************************************************/
template<typename PRECISION>
PRECISION* allocate_simple_dirty_image
    (const unsigned int dirty_moment_size, unsigned int num_taylor)
{
    PRECISION *dirty_moment_images_device;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&dirty_moment_images_device, dirty_moment_size*dirty_moment_size*num_taylor*sizeof(PRECISION)));
    CUDA_CHECK_RETURN(cudaMemset(dirty_moment_images_device, 0, dirty_moment_size*dirty_moment_size*num_taylor*sizeof(PRECISION))); // clear the image to zero
    return dirty_moment_images_device;
}

template float* allocate_simple_dirty_image<float>(const unsigned int dirty_moment_size, unsigned int num_taylor);
template double* allocate_simple_dirty_image<double>(const unsigned int dirty_moment_size, unsigned int num_taylor);


/*****************************************************************************
 * Temporary utility function that adds some sources to dirty_moment_images_device for testing
 *****************************************************************************/
template<typename PRECISION>
void calculate_simple_dirty_image
    (
    PRECISION *dirty_moment_images_device, unsigned int num_taylor, unsigned int dirty_moment_size
    )
{
    PRECISION *dirty_moment_images_host; // flat array containing input Taylor coefficient dirty images to be convolved
    CUDA_CHECK_RETURN(cudaHostAlloc(&dirty_moment_images_host, dirty_moment_size*dirty_moment_size*num_taylor*sizeof(PRECISION), 0));
    memset(dirty_moment_images_host, 0, dirty_moment_size*dirty_moment_size*num_taylor*sizeof(PRECISION));
    // add a few sources to taylor term 0
    add_source_to_image(dirty_moment_images_host, dirty_moment_size, (PRECISION)10.0, (PRECISION)1.0,
        dirty_moment_size/2, dirty_moment_size/2, 10);
    add_source_to_image(dirty_moment_images_host, dirty_moment_size, (PRECISION)2.0, (PRECISION)0.0,
        dirty_moment_size/2-4, dirty_moment_size/2-4, 0);
    add_source_to_image(dirty_moment_images_host, dirty_moment_size, (PRECISION)5.0, (PRECISION)4.0,
        dirty_moment_size/2+20, dirty_moment_size/2, 20);
    // copy the test input image to the device
    CUDA_CHECK_RETURN(cudaMemcpy(dirty_moment_images_device, dirty_moment_images_host, dirty_moment_size*dirty_moment_size*num_taylor*sizeof(PRECISION), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaFreeHost(dirty_moment_images_host));
}

template void calculate_simple_dirty_image<float>(float*, unsigned int, unsigned int);
template void calculate_simple_dirty_image<double>(double*, unsigned int, unsigned int);


/*****************************************************************************
 * Msmfs function that deallocates device data structure that was used to
 * hold all the dirty moment images on the device
 * Note should be paired with an earlier call to allocate_simple_dirty_image
 *****************************************************************************/
template<typename PRECISION>
void free_simple_dirty_image(PRECISION* dirty_moment_images_device)
{
    CUDA_CHECK_RETURN(cudaFree(dirty_moment_images_device));
}

template void free_simple_dirty_image<float>(float*);
template void free_simple_dirty_image<double>(double*);


/*****************************************************************************
 * Msmfs function that allocates and clears the data structure that will
 * hold all the psf moment images on the device
 * Note should be paired with a later call to free_simple_psf_image
 *****************************************************************************/
template<typename PRECISION>
PRECISION* allocate_simple_psf_image
    (const unsigned int psf_moment_size, unsigned int num_psf)
{
    PRECISION *psf_moment_images_device;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&psf_moment_images_device, psf_moment_size*psf_moment_size*num_psf*sizeof(PRECISION)));
    CUDA_CHECK_RETURN(cudaMemset(psf_moment_images_device, 0, psf_moment_size*psf_moment_size*num_psf*sizeof(PRECISION))); // clear the image to zero
    return psf_moment_images_device;
}

template float* allocate_simple_psf_image<float>(const unsigned int, unsigned int);
template double* allocate_simple_psf_image<double>(const unsigned int, unsigned int);


/*****************************************************************************
 * Temporary utility function that create a simple test input paraboloid psf
 * with specified radius and dropoff amplitude between successive taylor terms
 *****************************************************************************/
template<typename PRECISION>
void calculate_simple_psf_image
    (
    PRECISION *psf_moment_images_device, unsigned int num_psf, unsigned int psf_moment_size
    )
{
    const PRECISION psf_max_radius = (PRECISION)1.5; // radius of psf
    const PRECISION psf_moment_dropoff = (PRECISION)0.004; // amplitude dropoff with even taylor terms of the psf
    PRECISION *psf_moment_images_host; // flat array containing input Taylor coefficient psf images
    CUDA_CHECK_RETURN(cudaHostAlloc(&psf_moment_images_host, psf_moment_size*psf_moment_size*num_psf*sizeof(PRECISION), 0));
    memset(psf_moment_images_host, 0, psf_moment_size*psf_moment_size*num_psf*sizeof(PRECISION));
    // add a narrow paraboloid at centre of the even moment psf terms
    unsigned int psf_centre = psf_moment_size/2;
    for (unsigned int taylor_index=0; taylor_index<num_psf; taylor_index+=2) // only using even moments for initial testing
    {
        PRECISION central_peak = (PRECISION)pow(psf_moment_dropoff,sqrt((PRECISION)taylor_index)); // note MSMFS won't work with strick power law on PSF peaks
        for (unsigned int y=0; y<psf_moment_size; y++)
        {
            for (unsigned int x=0; x<psf_moment_size; x++)
            {
                PRECISION radius = (PRECISION)sqrt((x-psf_centre)*(x-psf_centre)+(y-psf_centre)*(y-psf_centre));
                psf_moment_images_host[psf_moment_size*psf_moment_size*taylor_index +
                    psf_moment_size*y + x] = max(central_peak*(1-(radius/psf_max_radius)*(radius/psf_max_radius)),(PRECISION)0);
            }
        }
    }
    // copy the test input psf to the device
    CUDA_CHECK_RETURN(cudaMemcpy(psf_moment_images_device, psf_moment_images_host, psf_moment_size*psf_moment_size*num_psf*sizeof(PRECISION), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaFreeHost(psf_moment_images_host));
}

template void calculate_simple_psf_image<float>(float*, unsigned int, unsigned int);
template void calculate_simple_psf_image<double>(double*, unsigned int, unsigned int);


/*****************************************************************************
 * Msmfs function that deallocates device data structure that was used to
 * hold all the psf moment images on the device
 * Note should be paired with an earlier call to allocate_simple_psf_image
 *****************************************************************************/
template<typename PRECISION>
void free_simple_psf_image(PRECISION *psf_moment_images_device)
{
    CUDA_CHECK_RETURN(cudaFree(psf_moment_images_device));
}

template void free_simple_psf_image<float>(float*);
template void free_simple_psf_image<double>(double*);

