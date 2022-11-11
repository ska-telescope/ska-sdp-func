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
 * @file Msmfsfunctionshost.h
 * @author Andrew Ensor
 * @brief C with C++ templates/CUDA device functions for the MSMFS cleaning algorithm.
 */

#ifndef MSMFS_FUNCTIONS_HOST_H
#define MSMFS_FUNCTIONS_HOST_H

#include <stdint.h>
#include <cusolverDn.h>

#include "Msmfscommon.h"
#include "Msmfslogger.h"
#include "Msmfsfunctionsdevice.h"

#define CUDA_CHECK_RETURN(value) check_cuda_error_aux(__FILE__, __LINE__, #value, value)

/**
 * Checks the Cuda status and logs any errors
 * @return any synchronous or asynchronous cuda error.
 */
cudaError_t checkCudaStatus();

/**
 * @brief Checks the Cuda status return value from a cuda call.
 * @param file File in which cuda error status reported.
 * @param line Line in file in which cuda error status reported.
 * @param statement Statement which resulted in the cuda error status.
 * @param err Cuda error.
 */
void check_cuda_error_aux(const char *file, unsigned line, const char *statement, cudaError_t err);

/**
 * @brief Calculates the Cuda configs for block size in 1D and 2D, and number of available cuda threads.
 * @param cuda_block_size Output cuda block size for one-dimensional kernels giving number of cuda threads per block.
 * @param cuda_block_size_2D Output cuda block size for two-dimensional kernels giving number of cuda threads per block.
 * @param cuda_num_threads Output number of cuda threads available which is used for grid-strided kernel.
 */
void calculate_cuda_configs(int *cuda_block_size, dim3 *cuda_block_size_2D, int *cuda_num_threads);

/**
 * @brief Msmfs function that allocates and prepares the shape configurations for cleaning
 * by taking the variances at each clean scale and calculating the
 * scale bias to use at that scale, the convolution support to use at that scale
 * and the doubly convolved support to use for each combination of scales.
 * 
 * Note should be paired with a later call to free_shape_configurations.
 * @param variances_host Input array of shape variances, if NULL then variances calculated to be 0, 1, 4, 16, 64, ...
 * @param num_scales Number of scales to use in msmfs cleaning.
 * @param convolution_accuracy Fraction of peak accuracy used to determine supports for convolution kernels.
 * @param scale_bias_factor Bias multiplicative factor to favour cleaning with smaller scales.
 */
template<typename PRECISION>
Gaussian_shape_configurations<PRECISION> allocate_shape_configurations
    (
    PRECISION *variances_host,
    const unsigned int num_scales,
    const PRECISION convolution_accuracy,
    const PRECISION scale_bias_factor
    );

/**
 * @brief Msmfs function that deallocates shape configurations, frees each pointer in the parameter struct.
 * 
 * Note should be paired with an earlier call to allocate_shape_configurations.
 * @param shape_configs Shape configurations for cleaning that will be freed.
 */
template<typename PRECISION>
void free_shape_configurations(Gaussian_shape_configurations<PRECISION> shape_configs);

/**
 * @brief Msmfs function that allocates and clears the data structures that will
 * hold all the scale moment residuals on the device, one for each combination
 * of scale and taylor term.
 * 
 * Note should be paired with a later call to free_scale_moment_residuals.
 * @param scale_moment_size One dimensional size of scale moment residuals, assumed square.
 * @param num_scales Number of scales to use in msmfs cleaning.
 * @param num_taylor Number of Taylor moments.
 */
template<typename PRECISION>
PRECISION* allocate_scale_moment_residuals
    (const unsigned int scale_moment_size, const unsigned int num_scales, unsigned int num_taylor);

/**
 * @brief Temporary utility function that displays the scale moment residuals on std out
 * for all combinations of scale and taylor moment.
 * @param scale_moment_residuals_device Flat array that holds scale moment residuals.
 * @param num_scales Number of scales to use in msmfs cleaning.
 * @param num_taylor Number of Taylor moments.
 * @param scale_moment_size One dimensional size of scale moment residuals, assumed square.
 */
template<typename PRECISION>
void display_scale_moment_residuals
    (
    PRECISION *scale_moment_residuals_device,
    const unsigned int num_scales,
    const unsigned int num_taylor,
    const unsigned int scale_moment_size
    );

/**
 * @brief Msmfs function that deallocates device data structures that were used
 * hold all the scale moment residuals on the device, one for each combination
 * of scale and taylor term.
 * 
 * Note should be paired with an earlier call to allocate_scale_moment_residuals.
 * @param scale_moment_residuals_device Flat array that holds scale moment residuals.
 */
template<typename PRECISION>
void free_scale_moment_residuals(PRECISION* scale_moment_residuals_device);

/**
 * @brief Msmfs function that calculates the scale convolutions of moment residuals on device.
 * 
 * dirty_moment_images_device presumed to have size num_taylor*dirty_moment_size*dirty_moment_size.
 * scale_moment_residuals_device presumed to have size num_taylor*num_scales*scale_moment_size*scale_moment_size.
 * and both assumed to be centred around origin with dirty_moment_images having sufficient border for convolutions.
 * variances presumed to have size num_scales.
 * convolution_support_device presumed to have size num_scales.
 * @param dirty_moment_images_device Flat array containing input Taylor coefficient dirty images to be convolved.
 * @param dirty_moment_size One dimensional size of image, assumed square.
 * @param num_taylor Number of Taylor moments.
 * @param scale_moment_residuals_device Output flat array that will be populated with all convolved scale moment residuals.
 * @param scale_moment_size One dimensional size of scale moment residuals, assumed square.
 * @param num_scales Number of scales to use in msmfs cleaning.
 * @param variances_device Variances used for calculating gaussian kernel, one for each of num_scales.
 * @param convolution_support_device Supports required for each gaussian kernel shape for convolution.
 * @param cuda_block_size_2D Cuda block size for two-dimensional kernels giving number of cuda threads per block.
 */
template<typename PRECISION>
void calculate_scale_moment_residuals
    (
    PRECISION *dirty_moment_images_device,
    const unsigned int dirty_moment_size,
    const unsigned int num_taylor,
    PRECISION *scale_moment_residuals_device, // 
    const unsigned int scale_moment_size, // one dimensional size of scale moment residuals, assumed square
    const unsigned int num_scales,
    PRECISION *variances_device, // variances used for calculating gaussian kernel, one for each of num_scales
    unsigned int *convolution_support_device, // supports required for each gaussian kernel shape for convolution
    dim3 cuda_block_size_2D
    );

/**
 * @brief Msmfs function that allocates and clears the data structures that will
 * hold all the inverse hessian matrices on the device, one for each scale
 * and each matrix of size num_taylor x num_taylor.
 * 
 * Note should be paired with a later call to free_inverse_hessian_matrices.
 * @param num_scales Number of scales to use in msmfs cleaning.
 * @param num_taylor Number of Taylor moments.
 */
template<typename PRECISION>
PRECISION* allocate_inverse_hessian_matrices(const unsigned int num_scales, unsigned int num_taylor);

/**
 * @brief Temporary utility function that displays entries calculated for inverse hessian matrices.
 * @param inverse_hessian_matrices_device Holds all the inverse hessian matrices on the device, one for each scale.
 * @param num_scales Number of scales to use in msmfs cleaning.
 * @param num_taylor Number of Taylor moments.
 */
template<typename PRECISION>
void display_inverse_hessian_matrices
    (
    PRECISION *inverse_hessian_matrices_device,
    unsigned int num_scales,
    unsigned int num_taylor
    );

/**
 * @brief Msmfs function that deallocates device data structures that were used
 * hold all the inverse hessian matrices on the device, one for each scale
 * and each matrix of size num_taylor x num_taylor.
 * 
 * Note should be paired with an earlier call to allocate_inverse_hessian_matrices.
 * @param inverse_hessian_matrices_device Holds all the inverse hessian matrices on the device, one for each scale.
 */
template<typename PRECISION>
void free_inverse_hessian_matrices(PRECISION *inverse_hessian_matrices_device);

/**
 * @brief Msmfs function that calculates the inverse of each scale-dependent moment hessian matrices on device.
 * 
 * psf_moment_images_device presumed to have size num_psf*psf_moment_size*psf_moment_size
 * and assumed to be centred around origin with psf_moment_images_device having sufficient size for convolution at its centre
 * and assumed that num_psf = 2*num_taylor - 1.
 * inverse_hessian_matrices_device presumed to have size num_scales*num_taylor*num_taylor.
 * variances presumed to have size num_scales.
 * double_convolution_support_device presumed to have size num_scales*num_scales.
 * Note check_cusolver_info can be turned off to avoid waiting for transfer of potential error code.
 * @param psf_moment_images_device Flat array containing input Taylor coefficient psf images to be convolved.
 * @param psf_moment_size One dimensional size of psf, assumed square.
 * @param num_psf Number of psf (determined by the number of Taylor terms).
 * @param inverse_hessian_matrices_device Output flat array that will be populated with all inverse hessian matrices.
 * @param num_scales Number of scales to use in msmfs cleaning.
 * @param num_taylor Number of Taylor moments.
 * @param variances_device Variances used for calculating gaussian kernel, one for each of num_scales.
 * @param double_convolution_support_device Supports required for each gaussian kernel shape for double convolution.
 * @param check_cusolver_info Whether to explicitly check for cusolver errors during matrix inversion.
 * @param cuda_block_size Cuda block size for one-dimensional kernels giving number of cuda threads per block.
 */
template<typename PRECISION>
void calculate_inverse_hessian_matrices
    (
    PRECISION *psf_moment_images_device, // flat array containing input Taylor coefficient psf images to be convolved
    const unsigned int psf_moment_size, // one dimensional size of psf, assumed square
    const unsigned int num_psf,
    PRECISION *inverse_hessian_matrices_device, // output flat array that will be populated with all inverse hessian matrices
    const unsigned int num_scales,
    const unsigned int num_taylor,
    PRECISION *variances_device, // variances used for calculating gaussian kernel, one for each of num_scales
    unsigned int *double_convolution_support_device, // supports required for each gaussian kernel shape for double convolution
    bool check_cusolver_info, // whether to explicitly check for cusolver errors during matrix inversion
    int cuda_block_size
    );

/**
 * @brief Msmfs function that allocates device data structures that are used
 * to hold the Gaussian_source that get found, and the number of them.
 * 
 * Note should be paired with a later call to free_gaussian_source_list.
 * @param max_gaussian_sources_host Upper bound on the number of gaussian sources the list data structure will hold.
 */
template<typename PRECISION>
Gaussian_source_list<PRECISION> allocate_gaussian_source_list(const unsigned int max_gaussian_sources_host);

/**
 * @brief Temporary utility function that copies the gaussian sources found during cleaning from device to host.
 * @param gaussian_sources_device Input Gaussian sources to display.
 * @param num_gaussian_sources_device Input number of gaussian sources to display.
 * @param max_gaussian_sources_host Input upper bound on the number of gaussian sources.
 * @param num_gaussian_sources_host Output that will hold the number of Gaussian sources on host.
 * @param gaussian_sources_host Output that will hold the Gaussian sources list on host.
 */
template<typename PRECISION>
void copy_gaussian_source_list_to_host
    (
    Gaussian_source<PRECISION> *gaussian_sources_device,
    unsigned int *num_gaussian_sources_device,
    unsigned int max_gaussian_sources_host,
    unsigned int *num_gaussian_sources_host,
    Gaussian_source<PRECISION> *gaussian_sources_host
    );

/**
 * @brief Temporary utility function that displays the gaussian sources found during cleaning.
 * @param gaussian_sources_device Gaussian sources to display.
 * @param num_gaussian_sources_device Number of gaussian sources to display.
 * @param max_gaussian_sources_host Upper bound on the number of gaussian sources.
 * @param dirty_moment_size One dimensional size of image, assumed square.
 * @param image_border Border around dirty moment images and psfs to clip when using convolved images or convolved psfs.
 * @param num_taylor Number of Taylor moments.
 */
template<typename PRECISION>
void display_gaussian_source_list
    (
    Gaussian_source<PRECISION> *gaussian_sources_device,
    unsigned int *num_gaussian_sources_device,
    unsigned int max_gaussian_sources_host,
    unsigned int dirty_moment_size,
    unsigned int image_border,
    unsigned int num_taylor
    );

/**
 * @brief Msmfs function that deallocates device data structures that were used
 * to hold the Gaussian_source that got found, and the number of them.
 * 
 * Note should be paired with an earlier call to allocate_gaussian_source_list.
 * @param gaussian_source_list Gaussian source list data structure to free.
 */
template<typename PRECISION>
void free_gaussian_source_list(Gaussian_source_list<PRECISION> gaussian_source_list);

/**
 * @brief Msmfs function that allocates device data structures that are used
 * during cleaning minor cycles but which are unlikely to be of interest afterwards.
 * 
 * Returns a Cleaning_device_data_structure holding pointers to all the
 * allocated data structures.
 * Note should be paired with a later call to free_device_data_structures
 * and is called before the clean minor cycle loop commences.
 * @param num_taylor Number of Taylor moments.
 * @param scale_moment_size One dimensional size of scale moment residuals, assumed square.
 * @param larger_psf_convolved_buffer Whether sufficient device memory to hold (double) convolution of psf images.
 * @param psf_moment_size One dimensional size of psf, assumed square.
 * @param psf_convolved_size One dimensional size of psf_convolved_images_device.
 * @param num_psf Number of psf (determined by the number of Taylor terms).
 */
template<typename PRECISION>
Cleaning_device_data_structures<PRECISION> allocate_device_data_structures
    (
    const unsigned int num_taylor,
    const unsigned int scale_moment_size, // one dimensional size of scale moment residuals, assumed square
    bool larger_psf_convolved_buffer, // whether sufficient device memory to hold (double) convolution of psf images
    const unsigned int psf_moment_size, // one dimensional size of psf, assumed square
    const unsigned int psf_convolved_size, // one dimensional size of psf_convolved_images_device
    const unsigned int num_psf
    );

/**
 * @brief Msmfs function that deallocates device data structures that were used
 * during cleaning minor cycles, frees each pointer in the parameter struct.
 * 
 * Note should be paired with an earlier call to allocated_device_data_structures
 * and is called after the clean minor cycle loop completes.
 */
template<typename PRECISION>
void free_device_data_structures(Cleaning_device_data_structures<PRECISION> working_data);

/**
 * @brief Msmfs function that performs the major cycles for msmfs cleaning.
 * @param max_clean_cycles Maximum number of clean cycles to perform.
 * @param min_clean_cycles Minimum number of cycles to perform before checking clean threshold.
 * @param clean_threshold Set clean_threshold to 0 to disable checking whether source to clean below cutoff threshold.
 * @param clean_loop_gain Loop gain fraction of peak point to clean from the peak each minor cycle.
 * @param scale_moment_residuals_device Inout flat array that holds scale moment residuals.
 * @param num_scales Number of scales to use in msmfs cleaning.
 * @param num_taylor Number of Taylor moments.
 * @param scale_moment_size One dimensional size of scale moment residuals, assumed square.
 * @param psf_moment_images_device Flat array containing input Taylor coefficient psf images to be convolved.
 * @param num_psf Number of psf (determined by the number of Taylor terms).
 * @param psf_moment_size One dimensional size of psf, assumed square.
 * @param psf_convolved_size One dimensional size of psf_convolved_images_device.
 * @param larger_psf_convolved_buffer Whether psf_convolved_device holds num_psf or just 1 (doubly) convolved psf image.
 * @param inverse_hessian_matrices_device Flat array that holds all inverse hessian matrices.
 * @param variances_device Variances used for calculating gaussian kernel, one for each of num_scales.
 * @param scale_bias_device Bias multiplicative factor to favour cleaning with smaller scales.
 * @param double_convolution_support_device Supports required for each gaussian kernel shape for double convolution.
 * @param peak_point_smpsol_device Output principal solution at the peak for each taylor term.
 * @param peak_point_scale_device Output scale at the peak.
 * @param peak_point_index_device Output array offset index of point at the peak.
 * @param smpsol_max_device Temporary array reused in find_principal_solution_at_peak.
 * @param smpsol_scale_device Temporary array reused in find_principal_solution_at_peak.
 * @param psf_convolved_images_device Reused buffer holding either 2*num_taylor-1 or else 1 (double) convolutions.
 * @param horiz_convolved_device Reused buffer partially convolved horiz_convolved_device not typically square as border only trimmed on left and right sides.
 * @param is_existing_source_device Output flag whether the source added was found already in model.
 * @param gaussian_sources_device Output sources that have distinct scales/positions (duplicates get merged).
 * @param num_gaussian_sources_device Output number of sources found in gaussian_sources_device.
 * @param cuda_block_size Cuda block size for one-dimensional kernels giving number of cuda threads per block.
 * @param cuda_block_size_2D Cuda block size for two-dimensional kernels giving number of cuda threads per block.
 * @param cuda_num_threads Number of cuda threads available which is used for grid-strided kernel.
 * @return The number of major cycles actually performed.
 */
template<typename PRECISION>
int perform_major_cycles
    (
    const unsigned int max_clean_cycles, // maximum number of clean cycles to perform
    const unsigned int min_clean_cycles, // minimum number of cycles to perform before checking clean threshold
    const PRECISION clean_threshold, // set clean_threshold to 0 to disable checking whether source to clean below cutoff threshold
    const PRECISION clean_loop_gain, // loop gain fraction of peak point to clean from the peak each minor cycle
    PRECISION *scale_moment_residuals_device, // inout flat array that holds scale moment residuals
    const unsigned int num_scales,
    const unsigned int num_taylor,
    const unsigned int scale_moment_size, // one dimensional size of scale moment residuals, assumed square

    PRECISION *psf_moment_images_device,
    const unsigned int num_psf,
    const unsigned int psf_moment_size, // one dimensional size of psf, assumed square
    const unsigned int psf_convolved_size, // one dimensional size of psf_convolved_images_device
    bool larger_psf_convolved_buffer, // whether psf_convolved_device holds num_psf or just 1 (doubly) convolved psf image
    
    PRECISION *inverse_hessian_matrices_device, // flat array that holds all inverse hessian matrices

    PRECISION *variances_device, // variances used for calculating gaussian kernel, one for each of num_scales
    const PRECISION *scale_bias_device, // bias multiplicative factor to favour cleaning with smaller scales
    unsigned int *double_convolution_support_device, // supports required for each gaussian kernel shape for double convolution

    PRECISION *peak_point_smpsol_device, // output principal solution at the peak for each taylor term
    unsigned int *peak_point_scale_device, // output scale at the peak
    unsigned int *peak_point_index_device, // output array offset index of point at the peak
    PRECISION *smpsol_max_device, // temporary array reused in find_principal_solution_at_peak
    unsigned int *smpsol_scale_device, // temporary array reused in find_principal_solution_at_peak
    PRECISION *psf_convolved_images_device,
    PRECISION *horiz_convolved_device,
    bool *is_existing_source_device, // output flag whether the source added was found already in model
            
    Gaussian_source<PRECISION> *gaussian_sources_device, // output sources that have distinct scales/positions (duplicates get merged)
    unsigned int *num_gaussian_sources_device, // output number of sources found in gaussian_sources_device

    int cuda_block_size,
    dim3 cuda_block_size_2D,
    int cuda_num_threads
    );

#endif /* include guard */