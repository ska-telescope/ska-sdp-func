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

/**
 * @file Msmfssimpletest.h
 * @author Andrew Ensor
 * @brief C with C++ templates/CUDA code for preparing simple test dirty moment images and psf images.
 */

#ifndef MSMFS_SIMPLE_TEST_H
#define MSMFS_SIMPLE_TEST_H

#include "Msmfsfunctionshost.h"

/**
 * @brief Msmfs function that allocates and clears the data structure that will
 * hold all the dirty moment images on the device.
 *
 * Note should be paired with a later call to free_simple_dirty_image.
 * @param dirty_moment_size One dimensional size of image, assumed square.
 * @param num_taylor Number of Taylor moments.
 */
template<typename PRECISION>
PRECISION* allocate_dirty_image
    (const unsigned int dirty_moment_size, unsigned int num_taylor);

/**
 * @brief Temporary utility function that adds some sources to dirty_moment_images_device for testing.
 *
 * @param dirty_moment_images_device Dirty moment images held on device.
 * @param dirty_moment_size One dimensional size of image, assumed square.
 * @param num_taylor Number of Taylor moments.
 */
template<typename PRECISION>
void calculate_simple_dirty_image
    (
    PRECISION *dirty_moment_images_device, unsigned int dirty_moment_size, unsigned int num_taylor
    );

/**
 * @brief Msmfs function that deallocates device data structure that was used to
 * hold all the dirty moment images on the device.
 *
 * Note should be paired with an earlier call to allocate_simple_dirty_image.
 * @param dirty_moment_images_device Dirty moment images held on device.
 */
template<typename PRECISION>
void free_dirty_image(PRECISION* dirty_moment_images_device);

/**
 * @brief Msmfs function that allocates and clears the data structure that will
 * hold all the psf moment images on the device.
 * 
 * Note should be paired with a later call to free_simple_dirty_image.
 * @param psf_moment_size One dimensional size of psf, assumed square.
 * @param num_psf Number of psf (determined by the number of Taylor terms)
 */
template<typename PRECISION>
PRECISION* allocate_psf_image
    (const unsigned int psf_moment_size, unsigned int num_psf);

/**
 * @brief Temporary utility function that create a simple test input paraboloid psf
 * with specified radius and dropoff amplitude between successive taylor terms.
 * 
 * @param psf_moment_images_device Psf moment images held on device.
 * @param psf_moment_size One dimensional size of psf, assumed square.
 * @param num_psf Number of psf (determined by the number of Taylor terms).
 */
template<typename PRECISION>
void calculate_simple_psf_image
    (
    PRECISION *psf_moment_images_device, unsigned int psf_moment_size, unsigned int num_psf
    );

/**
 * @brief Msmfs function that deallocates device data structure that was used to
 * hold all the psf moment images on the device.
 * 
 * Note should be paired with an earlier call to allocate_simple_psf_image
 * @param psf_moment_images_device Psf moment images held on device.
 */
template<typename PRECISION>
void free_psf_image(PRECISION *psf_moment_images_device);

#endif /* include guard */
