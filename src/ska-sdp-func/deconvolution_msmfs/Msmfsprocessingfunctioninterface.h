/**
 * @file Msmfsprocessingfunctioninterface.h
 * @author Andrew Ensor
 * @brief C with C++ templates/CUDA program providing an SDP processing function interface steps for the MSMFS cleaning algorithm
 */

#ifndef MSMFS_PROCESSING_FUNCTIONS_INTERFACE_H
#define MSMFS_PROCESSING_FUNCTIONS_INTERFACE_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup deconvolution_msmfs_func
 * @{
 */

/**
 * @brief C (untemplated) version of the function which allocates and clears the data structure that will
 * hold all the dirty moment images on the device.
 *
 * @param dirty_moment_size One dimensional size of image, assumed square.
 * @param num_taylor Number of Taylor moments.
 * @param mem_type The data type to use for the dirty moment images, either SDP_MEM_FLOAT or SDP_MEM_DOUBLE.
 * @return The allocated sdp_Mem data structure on the device.
 */
sdp_Mem *sdp_msmfs_allocate_dirty_image
    (const unsigned int dirty_moment_size, unsigned int num_taylor, sdp_MemType mem_type);

/**
 * @brief C (untemplated) version of the function which adds some sources to dirty_moment_images_device for testing
 *
 * @param dirty_moment_images The allocated sdp_Mem data structure on the device.
 * @param dirty_moment_size One dimensional size of image, assumed square.
 * @param num_taylor Number of Taylor moments.
 */
void sdp_msmfs_calculate_simple_dirty_image
    (
    sdp_Mem *dirty_moment_images, unsigned int dirty_moment_size, unsigned int num_taylor
    );

/**
 * @brief C (untemplated) version of the function which deallocates device data structure that was used to
 * hold all the dirty moment images on the device
 *
 * @param dirty_moment_images The allocated sdp_Mem data structure on the device.
 */
void sdp_msmfs_free_dirty_image(sdp_Mem *dirty_moment_images);

/**
 * @brief C (untemplated) version of the function which allocates and clears the data structure that will
 * hold all the psf moment images on the device.
 *
 * @param psf_moment_size One dimensional size of psf, assumed square.
 * @param num_psf Number of psf (determined by the number of Taylor terms).
 * @param mem_type The data type to use for the psf moment images, either SDP_MEM_FLOAT or SDP_MEM_DOUBLE.
 * @return The allocated sdp_Mem data structure on the device.
 */
sdp_Mem *sdp_msmfs_allocate_psf_image
    (const unsigned int psf_moment_size, unsigned int num_psf, sdp_MemType mem_type);

/**
 * @brief C (untemplated) version of the function which create a simple test input paraboloid psf
 * with specified radius and dropoff amplitude between successive taylor terms.
 *
 * @param psf_moment_images The allocated sdp_Mem data structure on the device.
 * @param psf_moment_size One dimensional size of psf, assumed square.
 * @param num_psf Number of psf (determined by the number of Taylor terms).
 */
void sdp_msmfs_calculate_simple_psf_image
    (
    sdp_Mem *psf_moment_images, unsigned int psf_moment_size, unsigned int num_psf
    );

/**
 * @brief C (untemplated) version of the function which deallocates device data structure that was used to
 * hold all the psf moment images on the device
 *
 * @param psf_moment_images The allocated sdp_Mem data structure on the device.
 */
void sdp_msmfs_free_psf_image(sdp_Mem *psf_moment_images);

/**
 * @brief C (untemplated) version of the function which performs the entire msmfs deconvolution using sdp_Mem handles
 * 
 * dirty_moment_images presumed to have size num_taylor*dirty_moment_size*dirty_moment_size.
 * scale_moment_residuals presumed to have size num_taylor*num_scales*scale_moment_size*scale_moment_size.
 * and both assumed to be centred around origin with dirty_moment_images having sufficient border for convolutions.
 * @param dirty_moment_images Flat sdp_Mem array containing input Taylor coefficient dirty images to be convolved.
 * @param psf_moment_images Flat sdp_Mem array containing input Taylor coefficient psf images to be convolved.
 * @param dirty_moment_size One dimensional size of image, assumed square.
 * @param num_scales Number of scales to use in msmfs cleaning.
 * @param num_taylor Number of Taylor moments.
 * @param psf_moment_size One dimensional size of psf, assumed square.
 * @param image_border Border around dirty moment images and psfs to clip when using convolved images or convolved psfs.
 * @param convolution_accuracy Fraction of peak accuracy used to determine supports for convolution kernels.
 * @param clean_loop_gain Loop gain fraction of peak point to clean from the peak each minor cycle.
 * @param max_gaussian_sources_host Upper bound on the number of gaussian sources the list data structure will hold.
 * @param scale_bias_factor Bias multiplicative factor to favour cleaning with smaller scales.
 * @param clean_threshold Set clean_threshold to 0 to disable checking whether source to clean below cutoff threshold.
 * @param num_gaussian_sources_host Output pointer to unsigned int holding number of distinct sources cleaned.
 * @param gausian_source_position Output sdp_Mem array containing (x,y) int2 position of each gaussian source.
 * @param gaussian_source_variance Output sdp_Mem array containing variance of each gaussian source.
 * @param gaussian_source_taylor_intensities Output sdp_Mem array containing intensity for each taylor term of each gaussian source.
 */
void sdp_msmfs_perform
    (
    sdp_Mem *dirty_moment_images,
    sdp_Mem *psf_moment_images,
    const unsigned int dirty_moment_size,
    const unsigned int num_scales,
    const unsigned int num_taylor,
    const unsigned int psf_moment_size,
    const unsigned int image_border,
    const double convolution_accuracy,
    const double clean_loop_gain,
    const unsigned int max_gaussian_sources_host,
    const double scale_bias_factor,
    const double clean_threshold,
    unsigned int *num_gaussian_sources_host,
    sdp_Mem *gausian_source_position,
    sdp_Mem *gaussian_source_variance,
    sdp_Mem *gaussian_source_taylor_intensities
    );

/** @} */ /* End group deconvolution_msmfs_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */