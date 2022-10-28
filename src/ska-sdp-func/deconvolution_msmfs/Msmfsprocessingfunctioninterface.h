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
#include <math.h>

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

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
 */
void perform_msmfs
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
    const double clean_threshold
    );

#ifdef __cplusplus
}
#endif

#endif /* include guard */