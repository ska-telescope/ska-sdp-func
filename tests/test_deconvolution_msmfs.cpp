// Copyright 2022 High Performance Computing Research Laboratory, Auckland University of Technology (AUT)

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
 * @file test_deconvolution_msmfs.cpp
 * @author Andrew Ensor
 * @brief C++ code providing a simple test for deconvolution_msmfs.
 */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ska-sdp-func/deconvolution_msmfs/Msmfssimpletest.h"
#include "ska-sdp-func/deconvolution_msmfs/Msmfsprocessingfunctioninterface.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"

using std::complex;

static void run_and_check(
        const char* test_name,
        bool expect_pass,
        bool read_only_output,
        sdp_MemType input_type,
        sdp_MemType output_type,
        sdp_MemLocation input_location,
        sdp_MemLocation output_location,
        sdp_Error* status
)
{
    // specify msmsf key configuration parameters
    const unsigned int dirty_moment_size = 8192; // one dimensional size of image, assumed square
    const unsigned int num_scales = 6; // number of scales to use in msmfs cleaning
    unsigned int num_taylor = 3; // number of taylor moments to use in msmfs cleaning
    if (num_taylor > MAX_TAYLOR_MOMENTS)
    {
        printf("Number of Taylor moments was set at %u but will be capped at %u, change MAX_TAYLOR_MOMENTS to adjust",
            num_taylor, MAX_TAYLOR_MOMENTS);
        num_taylor = MAX_TAYLOR_MOMENTS;
    }
    const unsigned int psf_moment_size = dirty_moment_size/4; // one dimensional size of psf, assumed square
    const unsigned int image_border = 0; // border around dirty moment images and psfs to clip when using convolved images or convolved psfs
    const double convolution_accuracy = 1.2E-3; // fraction of peak accuracy used to determine supports for convolution kernels
    const double clean_loop_gain = 0.35; // loop gain fraction of peak point to clean from the peak each minor cycle
    const unsigned int max_gaussian_sources_host = 200; // maximum number of gaussian sources to find during cleaning (so bounds number clean minor cycles)
    const double scale_bias_factor = 0.6; // 0.6 is typical bias multiplicative factor to favour cleaning with smaller scales
    const double clean_threshold = 0.001; // fractional threshold at which to stop cleaning (or non-positive to disable threshold check)

    // Call the function to test.
    SDP_LOG_INFO("Running test: %s", test_name);

    if (input_type==SDP_MEM_FLOAT || input_type==SDP_MEM_DOUBLE)
    {
        // create a simple test input image
        sdp_Mem *dirty_moment_images = sdp_msmfs_allocate_dirty_image(dirty_moment_size, num_taylor, input_type);
        sdp_msmfs_calculate_simple_dirty_image(dirty_moment_images, dirty_moment_size, num_taylor);

        // create a simple test input psf
        sdp_Mem *psf_moment_images = sdp_msmfs_allocate_psf_image(psf_moment_size, 2*num_taylor-1, input_type);
        sdp_msmfs_calculate_simple_psf_image(psf_moment_images , psf_moment_size, 2*num_taylor-1);

        // create the sdp_Mem data structures that will hold the output cleaned sources
        unsigned int *num_gaussian_sources_host = NULL;
        num_gaussian_sources_host = (unsigned int *)malloc(sizeof(unsigned int));
        memset(num_gaussian_sources_host, 0, sizeof(unsigned int));
        sdp_Error *status = NULL;
        const int64_t gaussian_source_position_shape[] = {max_gaussian_sources_host, 2};
        sdp_Mem *gaussian_source_position = sdp_mem_create(SDP_MEM_INT, SDP_MEM_CPU, 2, gaussian_source_position_shape, status);
        sdp_mem_clear_contents(gaussian_source_position, status);
        const int64_t gaussian_source_variance_shape[] = {max_gaussian_sources_host};
        sdp_Mem *gaussian_source_variance = sdp_mem_create(input_type, SDP_MEM_CPU, 1, gaussian_source_variance_shape, status);
        sdp_mem_clear_contents(gaussian_source_variance, status);
        const int64_t gaussian_source_taylor_intensities_shape[] = {max_gaussian_sources_host, num_taylor};
        sdp_Mem *gaussian_source_taylor_intensities = sdp_mem_create(input_type, SDP_MEM_CPU, 2, gaussian_source_taylor_intensities_shape, status);
        sdp_mem_clear_contents(gaussian_source_taylor_intensities, status);

        sdp_msmfs_perform(
            dirty_moment_images, psf_moment_images,
            dirty_moment_size, num_scales, num_taylor, psf_moment_size, image_border,
            convolution_accuracy, clean_loop_gain, max_gaussian_sources_host,
            scale_bias_factor, clean_threshold,
            num_gaussian_sources_host, gaussian_source_position, gaussian_source_variance, gaussian_source_taylor_intensities);

        // clean up the allocated sdp_Mem data structures
        sdp_mem_ref_dec(gaussian_source_taylor_intensities);
        sdp_mem_ref_dec(gaussian_source_variance);
        sdp_mem_ref_dec(gaussian_source_position);
        free(num_gaussian_sources_host);

        // clean up simple test input image and simple test input psf
        sdp_msmfs_free_psf_image(psf_moment_images);
        sdp_msmfs_free_dirty_image(dirty_moment_images);
    }

    if (expect_pass)
    {
        // TODO: check output once returned as sdp_Mem
    }
}

int main()
{
#ifdef SDP_HAVE_CUDA
    // Happy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU, double precision", true, false,
                SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status);
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU, single precision", true, false,
                SDP_MEM_COMPLEX_FLOAT, SDP_MEM_COMPLEX_FLOAT,
                SDP_MEM_GPU, SDP_MEM_GPU, &status);
        assert(status == SDP_SUCCESS);
    }

    // Unhappy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Read-only output", false, true,
                SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status);
        assert(status != SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Inconsistent data types", false, false,
                SDP_MEM_COMPLEX_FLOAT, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status);
        assert(status == SDP_ERR_DATA_TYPE);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Inconsistent locations", false, false,
                SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_GPU, &status);
        assert(status == SDP_ERR_MEM_LOCATION);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Wrong data type", false, false,
                SDP_MEM_DOUBLE, SDP_MEM_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status);
        assert(status == SDP_ERR_DATA_TYPE);
    }
#endif
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, double precision", false, false,
                SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status);
        assert(status == SDP_ERR_MEM_LOCATION);
    }
    return 0;
}
