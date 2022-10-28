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
 * @file Msmfscommon.h
 * @author Andrew Ensor
 * @brief C/CUDA common header for MSMFS cleaning algorithm.
 */

#ifndef MSMFS_COMMON_H
#define MSMFS_COMMON_H

const double TWO_PI = 2.0*3.141592653589793238463;

/**
 * @struct Gaussian_shape_configurations
 * @brief Configuration struct of arrays for each the Gaussian shapes used during MSMFS.
 * @var Gaussian_shape_configurations::variances_device
 * Member 'variances_device' variance (st.dev squared) of each Gaussian shape, or 0 for point shape.
 * @var Gaussian_shape_configurations::scale_bias_device
 * Member 'scale_bias_device' multiplicative bias to apply to favour cleaning with smaller scales.
 * @var Gaussian_shape_configurations::convolution_support_device
 * Member 'convolution_support_device' support to use for convolution with each Gaussian shape.
 * @var Gaussian_shape_configurations::double_convolution_support_device
 * Member 'double_convolution_support_device' support to use for (double) convolved Gaussian shapes.
 */
template<typename PRECISION>
struct Gaussian_shape_configurations
{
    PRECISION *variances_device;
    PRECISION *scale_bias_device;
    unsigned int *convolution_support_device;
    unsigned int *double_convolution_support_device;
};

#define MAX_TAYLOR_MOMENTS 6 /* compile-time upper limit on the number of possible taylor moments */

/**
 * @struct Gaussian_source
 * @brief Represents a single gaussian shaped source in the scale_moment_model.
 * 
 * Note the (x,y) position of source in dirty image is given by
 * x = (index % (dirty_moment_size-2*image_border)) + image_border
 * y = (index / (dirty_moment_size-2*image_border)) + image_border
 * Note the intensities for a gaussian source are taken at its central peak
 * and variance 0 represents a point source.
 * @var Gaussian_source::index
 * Member 'index' index of this source in the dirty_moment_image flat array once image_border clipped.
 * @var Gaussian_source::variance
 * Member 'variance' variance of this gaussian source.
 * @var Gaussian_source::intensities
 * Member 'intensities' intensity at peak of source (once loop­_gain applied in cleaning minor cycle) at each of num_taylor moments.
 */
template<typename PRECISION>
struct Gaussian_source
{
    unsigned int index;
    PRECISION variance; 
    PRECISION intensities[MAX_TAYLOR_MOMENTS];
};

/**
 * @struct Gaussian_source_list
 * @brief Represents a list of Gaussian_source found during cleaning minor cycles of MSMFS.
 * @var Gaussian_source_list::gaussian_sources_device
 * Member 'gaussian_sources_device' gaussian sources that have been found.
 * @var Gaussian_source_list::num_gaussian_sources_device
 * Member 'num_gaussian_sources_device' number of gaussian sources that have been found.
 */
template<typename PRECISION>
struct Gaussian_source_list
{
    Gaussian_source<PRECISION> *gaussian_sources_device;
    unsigned int *num_gaussian_sources_device;
};

/**
 * @struct Cleaning_device_data_structures
 * @brief Data structures that are required internally while cleaning minor cycles of MSMFS.
 * @var Cleaning_device_data_structures::peak_point_smpsol_device
 * Member 'peak_point_smpsol_device' output principal solution at the peak for each taylor term.
 * @var Cleaning_device_data_structures::peak_point_scale_device
 * Member 'peak_point_scale_device' output scale at the peak.
 * @var Cleaning_device_data_structures::peak_point_index_device
 * Member 'peak_point_index_device' output array offset index of point at the peak.
 * @var Cleaning_device_data_structures::smpsol_max_device
 * Member 'smpsol_max_device' temporary array reused in find_principal_solution_at_peak.
 * @var Cleaning_device_data_structures::smpsol_scale_device
 * Member 'smpsol_scale_device' temporary array reused in find_principal_solution_at_peak.
 * @var Cleaning_device_data_structures::psf_convolved_images_device
 * Member 'psf_convolved_images_device' reused buffer holding either 2*num_taylor-1 or else 1 (double) convolutions.
 * @var Cleaning_device_data_structures::horiz_convolved_device
 * Member 'horiz_convolved_device' reused buffer partially convolved horiz_convolved_device not typically square as border only trimmed on left and right sides.
 * @var Cleaning_device_data_structures::is_existing_source_device
 * Member 'is_existing_source_device' out flag whether the source added was found already in model.
 */
template<typename PRECISION>
struct Cleaning_device_data_structures
{
    PRECISION *peak_point_smpsol_device;
    unsigned int *peak_point_scale_device;
    unsigned int *peak_point_index_device;
    PRECISION *smpsol_max_device;
    unsigned int *smpsol_scale_device;
    PRECISION *psf_convolved_images_device;
    PRECISION *horiz_convolved_device;
    bool *is_existing_source_device;
};

#endif /* include guard */