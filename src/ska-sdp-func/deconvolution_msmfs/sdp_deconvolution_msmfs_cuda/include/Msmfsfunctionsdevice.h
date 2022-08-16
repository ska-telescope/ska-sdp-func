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
 * @file Msmfsfunctionsdevice.h
 * @author Andrew Ensor
 * @brief CUDA device functions for the MSMFS cleaning algorithm.
 */

#ifndef MSMFS_FUNCTIONS_DEVICE_H
#define MSMFS_FUNCTIONS_DEVICE_H

#include <stdint.h>

#include "Msmfscommon.h"

/**
 * @brief Performs a one-dimensional horizontal convolution at each non-border column of image.
 *
 * Note this presumes image_device has width convolved_width+2*image_border
 * so there are additional image_border columns in image_device at left and right
 * and presumes image_device has same height convolved_height as convolved_device.
 * Parallelised so each CUDA thread processes a single pixel convolution result.
 * @param image_device Input flat array containing image.
 * @param convolved_device Output flat array containing convolved image except at left and right border.
 * @param convolved_width Width of resulting horizontally convolved image.
 * @param convolved_height Height of image_device and resulting horizontally convolved image.
 * @param image_border Left and right border regions of image_device.
 * @param convolution_support_device Supports required for each gaussian kernel shape for convolution.
 * @param variances_device Variances used for calculating gaussian kernel, one for each of num_scales.
 * @param scale_index Scale index determining convolution support and variance to use in convolution.
 */
template<typename PRECISION>
__global__ void convolve_horizontal
    (
    PRECISION *image_device,
    PRECISION *convolved_device,
    const unsigned int convolved_width,
    const unsigned int convolved_height,
    const unsigned int image_border,
    const unsigned int *convolution_support_device,
    const PRECISION *variances_device,
    const unsigned int scale_index
    );

/**
 * @brief Same as convolve_horizontal but accepts two variances and uses their sum as the variance for convolution.
 * 
 * Note first scale index presumed to be known on host so passed by value
 * and second scale index presumed on device so passed by reference.
 * Parallelised so each CUDA thread processes a single pixel convolution result.
 * @param image_device Input flat array containing image.
 * @param convolved_device Output flat array containing convolved image except at left and right border.
 * @param convolved_width Width of resulting horizontally convolved image.
 * @param convolved_height Height of image_device and resulting horizontally convolved image.
 * @param image_border Left and right border regions of image_device.
 * @param double_convolution_support_device Supports required for each gaussian kernel shape for double convolution.
 * @param num_scales Number of scales that can be chosen between for the convolution.
 * @param variances_device Variances used for calculating gaussian kernel, one for each of num_scales.
 * @param scale_index1 Index of first scale.
 * @param scale_index2_device Index of second scale index already on device so passed by reference.
 */
template<typename PRECISION>
__global__ void double_convolve_horizontal
    (
    PRECISION *image_device,
    PRECISION *convolved_device,
    const unsigned int convolved_width,
    const unsigned int convolved_height,
    const unsigned int image_border,
    const unsigned int *double_convolution_support_device,
    const unsigned int num_scales,
    const PRECISION *variances_device,
    const unsigned int scale_index1,
    const unsigned int *scale_index2_device
    );

/**
 * @brief Performs a one-dimensional vertical convolution at each non-border row of image
 * using an unnormalised gaussian as presumes horiz_convolved_device has already had a normalised gaussian applied.
 * 
 * Note this presumes horiz_convolved_device has same width convolved_width as convolved_device
 * and presumes horiz_convolved_device has height convolved_height+2*image_border
 * with additional image_border rows at top and bottom.
 * Parallelised so each CUDA thread processes a single pixel convolution result.
 * @param horiz_convolved_device Input flat array containing horizontally convolved image.
 * @param convolved_device Output flat array containing vertically convolved image except at top and bottom border.
 * @param convolved_width Width of image_device and resulting vertically convolved image.
 * @param convolved_height Height of resulting vertically convolved image.
 * @param image_border Top and bottom border regions of horiz_convolved_device.
 * @param convolution_support_device Supports required for each gaussian kernel shape for convolution.
 * @param variances_device Variances used for calculating gaussian kernel, one for each of num_scales.
 * @param scale_index Scale index determining convolution support and variance to use in convolution.
 */
template<typename PRECISION>
__global__ void convolve_vertical
    (
    PRECISION *horiz_convolved_device,
    PRECISION *convolved_device,
    const unsigned int convolved_width,
    const unsigned int convolved_height,
    const unsigned int image_border,
    const unsigned int *convolution_support_device,
    const PRECISION *variances_device,
    const unsigned int scale_index
    );

/**
 * @brief Same as convolve_vertical but accepts two variances and uses their sum as the variance for convolution.
 * 
 * Note first scale index presumed to be known on host so passed by value
 * and second scale index presumed on device so passed by reference.
 * Parallelised so each CUDA thread processes a single pixel convolution result.
 * @param horiz_convolved_device Input flat array containing horizontally convolved image.
 * @param convolved_device Output flat array containing vertically convolved image except at top and bottom border.
 * @param convolved_width Width of image_device and resulting vertically convolved image.
 * @param convolved_height Height of resulting vertically convolved image.
 * @param image_border Top and bottom border regions of horiz_convolved_device.
 * @param double_convolution_support_device Supports required for each gaussian kernel shape for double convolution.
 * @param num_scales Number of scales that can be chosen between for the convolution.
 * @param variances_device Variances used for calculating gaussian kernel, one for each of num_scales.
 * @param scale_index1 Index of first scale.
 * @param scale_index2_device Second scale index already on device so passed by reference.
 */
template<typename PRECISION>
__global__ void double_convolve_vertical
    (
    PRECISION *horiz_convolved_device,
    PRECISION *convolved_device,
    const unsigned int convolved_width,
    const unsigned int convolved_height,
    const unsigned int image_border,
    const unsigned int *double_convolution_support_device,
    const unsigned int num_scales,
    const PRECISION *variances_device,
    const unsigned int scale_index1,
    const unsigned int *scale_index2_device
    );

/**
 * @brief Calculates each entry for the hessian matrices by performing a two-dimensional convolution at
 * the centre of one psf at one scale to get one of the entries.
 * 
 * Note this presumes the psf is at least as large as the convolution support.
 * Parallelised so each CUDA thread processes a single convolution.
 * @param psf_moment_images_device Input flat array containing input Taylor coefficient psf images to be convolved.
 * @param psf_moment_size One dimensional size of psf, assumed square.
 * @param num_psf Number of psf (determined by the number of Taylor terms).
 * @param hessian_entries_device Output flat array to hold entries for the hessian matrices.
 * @param num_scales Number of scales to use in msmfs cleaning.
 * @param variances_host Variances used for calculating gaussian kernel, one for each of num_scales.
 * @param double_convolution_support_device Supports required for each gaussian kernel shape for double convolution.
 */
template<typename PRECISION>
__global__ void calculate_hessian_entries
    (
    PRECISION *psf_moment_images_device,
    const unsigned int psf_moment_size,
    const unsigned int num_psf,
    PRECISION *hessian_entries_device,
    const unsigned int num_scales,
    PRECISION *variances_host,
    unsigned int *double_convolution_support_device
    );

/**
 * @brief Populates the hessian matrices with the entries given by calculate_hessian_entries.
 *
 * The (t,t') entry of s-th hessian matrix uses the s-th scale convolution of the t+t'-th psf
 * Note this presumes hessian_entries_device has entries arranged with all scales for each of
 * the num_psf psf, where num_psf is presumed to be 2*num_taylor-1.
 * Note this presumes hessian_matrix_device has entries arranged row by row for each of
 * the num_scales hessian matrices.
 * Parallelised so each CUDA thread processes a single entry of one of the hessian matrices.
 * @param hessian_entries_device Input flat array containing the num_scales*num_psf entries.
 * @param num_scales Number of scales to use in msmfs cleaning.
 * @param num_taylor Number of Taylor moments.
 * @param hessian_matrix_device Output flat array containing the num_scales hessian matrices, each of size num_taylor*num_taylor.
 */
template<typename PRECISION>
__global__ void populate_hessian_matrices
    (
    PRECISION *hessian_entries_device,
    const unsigned int num_scales,
    const unsigned int num_taylor,
    PRECISION *hessian_matrix_device
    );

/**
 * @brief Populates the principal solution image with the maximum principal solution with bias applied,
 * across all scales at each point for t=0.
 * 
 * Note this presumes smpsol_max_device and smpsol_scale_device have size scale_moment_size*scale_moment_size.
 * Parallelised so each CUDA thread processes a single point of principal solution across all scales.
 * @param scale_moment_residuals_device Input flat array that holds all convolved scale moment residuals.
 * @param num_scales Number of scales to use in msmfs cleaning.
 * @param num_taylor Number of Taylor moments.
 * @param scale_moment_size One dimensional size of scale moment residuals, assumed square.
 * @param inverse_hessian_matrices_device Input flat array that holds all inverse hessian matrices.
 * @param scale_bias Bias multiplicative factor to favour cleaning with smaller scales.
 * @param smpsol_max_device Output entries for the smpsol that has abs max at each point.
 * @param smpsol_scale_deviceOutput scale indices for the abs max smpsol at each point.
 */
template<typename PRECISION>
__global__ void calculate_principal_solution_max_scale
    (
    PRECISION *scale_moment_residuals_device,
    const unsigned int num_scales,
    const unsigned int num_taylor,
    const unsigned int scale_moment_size,
    PRECISION *inverse_hessian_matrices_device,
    const PRECISION* scale_bias,
    PRECISION *smpsol_max_device,
    unsigned int* smpsol_scale_device
    );

/**
 * @struct 32-bit struct that holds a float value and its index location in a 1D array
 * @var Array_entry::value
 * Member 'value' array entry stored as a 32-bit float so struct will fit as 64 bits.
 * @var Array_entry::index
 * Member 'index' index of the entry in the array.
 */
struct Array_entry
{
    float value; // array entry stored as a 32-bit float so struct will fit as 64 bits
    uint32_t index; // index of the entry in the array
};

/**
 * @brief Finds the maximum value and its index location in a one-dimensional array.
 *
 * Loosely follows the reduction approach described in
 * https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
 * with grid-strided loops as described in
 * https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
 * Parallelised so each CUDA thread processes all array entries that are separated
 * by the same grid stride.
 * @param input_array_device Input 1D array for which the location of maximum is to be found.
 * @param input_num_values Number of values in the input array (might be many times number cuda threads).
 * @param max_entry_device Output array entry where the maximum is found in input_array_device.
 */
template<typename PRECISION>
__global__ void find_max_entry_grid_strided_reduction
    (
    PRECISION *input_array_device,
    unsigned int input_num_values,
    Array_entry* max_entry_device
    );

/**
 * @brief Calculates the principal solution at the peak, known as mval in the MSMFS algorithm.
 *
 * Note this presumes mval_device has size num_taylor.
 * Parallelised so each CUDA thread processes a single taylor term value of principal solution at peak.
 * @param scale_moment_residuals_device Input flat array that holds all convolved scale moment residuals.
 * @param num_scales Number of scales to use in msmfs cleaning.
 * @param num_taylor Number of Taylor moments.
 * @param scale_moment_size One dimensional size of scale moment residuals, assumed square.
 * @param inverse_hessian_matrices_device Input flat array that holds all inverse hessian matrices.
 * @param smpsol_scale_device Input flat array that holds scale indices for the abs max smpsol at each point.
 * @param smpsol_max_index_device Index in smpsol_scale_device and in one of the scale_moment_residuals_device at the peak.
 * @param peak_point_smpsol_device Output principal solution at the peak.
 * @param peak_point_scale_device Output scale index at the peak.
 * @param peak_point_index_deviceOutput array offset index of point at the peak.
 */
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
    );

/**
 * @brief Subtracts the (doubly) convolved psf scaled by the peak point and clean_loop_gain
 * from the scale_moment_residual_device centred at peak_point_index.
 * 
 * Note this presumes scale_moment_residual_device has size scale_moment_size*scale_moment_size
 * and psf_convolved_device has size psf_convolved_size*psf_convolved_size
 * and 0<=peak_point_index<scale_moment_size*scale_moment_size.
 * Parallelised so each CUDA thread processes the scaled subtraction at a single point of scale_moment_residual_device.
 * @param scale_moment_residual_device Inout flat array that holds one convolved scale moment residual.
 * @param scale_moment_size One dimensional size of scale moment residuals, assumed square.
 * @param psf_convolved_device Input flat array containing one (doubly) convolved psf image.
 * @param psf_convolved_size One dimensional size of convolved psf, assumed square.
 * @param peak_point_smpsol_device Principal solution at the peak.
 * @param taylor_index Taylor index at the peak.
 * @param peak_point_index_device Input array offset index of point at the peak.
 * @param clean_loop_gain Loop gain fraction of peak point to clean from the peak each minor cycle.
 */
template<typename PRECISION>
__global__ void subtract_psf_convolved_from_scale_moment_residual
    (
    PRECISION *scale_moment_residual_device,
    const unsigned int scale_moment_size,
    PRECISION *psf_convolved_device,
    const unsigned int psf_convolved_size,
    PRECISION *peak_point_smpsol_device,
    unsigned int taylor_index,
    unsigned int *peak_point_index_device,
    PRECISION clean_loop_gain
    );

/**
 * @brief Adds the specified source to the scale moment model, either by adjusting the inensity of an
 * existing source already in the model or else by adding it as a new source appended to gaussian_sources_device.
 * 
 * Uses grid-strided loops as described in
 * https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
 * Parallelised so each CUDA thread processes all sources in the model that are separated
 * by the same grid stride.
 * For convenience presumes there is only one block of threads in the cuda grid so no need to check across multiple blocks.
 * @param gaussian_sources_device Inout list of sources that have distinct scales/positions.
 * @param num_gaussian_sources_device Inout current number of sources in gaussian_sources_device.
 * @param peak_point_smpsol_device Input principal solution at the peak.
 * @param num_taylor Number of Taylor moments.
 * @param variances_device Variances used for calculating gaussian kernel, one for each of num_scales.
 * @param peak_point_scale_device Input scale index at the peak.
 * @param peak_point_index_device Input array offset index of point at the peak.
 * @param clean_loop_gain Loop gain fraction of peak point to clean from the peak each minor cycle.
 * @param is_existing_source_device Output flag whether the source added was found already in model.
 */
template<typename PRECISION>
__global__ void add_source_to_model_grid_strided_reduction
    (
    Gaussian_source<PRECISION> *gaussian_sources_device,
    unsigned int *num_gaussian_sources_device,
    PRECISION *peak_point_smpsol_device,
    const unsigned int num_taylor,
    PRECISION *variances_device,
    unsigned int *peak_point_scale_device,
    unsigned int *peak_point_index_device,
    PRECISION clean_loop_gain,
    bool *is_existing_source_device
    );

#endif /* include guard */