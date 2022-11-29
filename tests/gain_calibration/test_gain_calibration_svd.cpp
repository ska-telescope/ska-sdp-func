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
 * @file test_gain_calibration_svd.cpp
 * @author Andrew Ensor
 * @brief C++ code providing a simple test for gain_calibration_svd.
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

#include "ska-sdp-func/gain_calibration_svd/Gaincalsimpletest.h"
#include "ska-sdp-func/gain_calibration_svd/Gaincalprocessingfunctioninterface.h"
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
    // create some simple sample visibilities and gains for testing
    const unsigned int num_receivers = 10;
    const unsigned int num_baselines = num_receivers*(num_receivers-1)/2;
    const unsigned int max_calibration_cycles = 10;

    // Call the function to test.
    SDP_LOG_INFO("Running test: %s", test_name);

    if ((input_type==SDP_MEM_COMPLEX_FLOAT || input_type==SDP_MEM_COMPLEX_DOUBLE) && output_type==input_type
        && input_location==SDP_MEM_GPU && output_location==SDP_MEM_GPU)
    {
        sdp_Mem *vis_predicted_host = sdp_gaincal_allocate_visibilities_host(num_baselines, input_type);
        sdp_gaincal_generate_sample_visibilities_host(vis_predicted_host, num_baselines);
        sdp_Mem *receiver_pairs_host = sdp_gaincal_allocate_receiver_pairs_host(num_baselines);
        sdp_gaincal_generate_sample_receiver_pairs_host(receiver_pairs_host, num_baselines, num_receivers);
        sdp_Mem *vis_predicted_device = sdp_gaincal_allocate_visibilities_device(num_baselines, input_type);
        sdp_Mem *vis_measured_device = sdp_gaincal_allocate_visibilities_device(num_baselines, input_type);

        sdp_Error *status = NULL;
        const int64_t gains_shape[] = {num_receivers};
        sdp_Mem *actual_gains_host = NULL;
        if (input_type == SDP_MEM_COMPLEX_FLOAT)
        {
            float2 actual_gains_host_array[num_receivers];
            for (unsigned int receiver=0; receiver<num_receivers; receiver++)
            {
                float amplitude = (float)(1.0+sdp_gaincal_get_random_gaussian_float()*0.1);
                float phase = sdp_gaincal_get_random_gaussian_float()*(float)0.1;
                actual_gains_host_array[receiver].x = (float)(amplitude * cos(phase));
                actual_gains_host_array[receiver].y = (float)(amplitude * sin(phase));
            }
            actual_gains_host = sdp_mem_create_wrapper(actual_gains_host_array, input_type, SDP_MEM_CPU, 1, gains_shape, 0, status);
        }
        else
        {
            double2 actual_gains_host_array[num_receivers];
            for (unsigned int receiver=0; receiver<num_receivers; receiver++)
            {
                double amplitude = 1.0+sdp_gaincal_get_random_gaussian_double()*0.1;
                double phase = sdp_gaincal_get_random_gaussian_double()*0.1;
                actual_gains_host_array[receiver].x = amplitude * cos(phase);
                actual_gains_host_array[receiver].y = amplitude * sin(phase);
            }
            actual_gains_host = sdp_mem_create_wrapper(actual_gains_host_array, input_type, SDP_MEM_CPU, 1, gains_shape, 0, status);
        }

        sdp_gaincal_calculate_measured_and_predicted_visibilities_device
            (vis_predicted_host, receiver_pairs_host, num_baselines, actual_gains_host,
            num_receivers, vis_measured_device, vis_predicted_device);
        sdp_Mem *receiver_pairs_device = sdp_gaincal_allocate_receiver_pairs_device(num_baselines);
        sdp_gaincal_set_receiver_pairs_device(receiver_pairs_host, receiver_pairs_device, num_baselines);
        sdp_Mem *gains_device = sdp_gaincal_allocate_gains_device(num_receivers, input_type);

        sdp_gaincal_perform(vis_measured_device, vis_predicted_device, receiver_pairs_device, num_receivers, num_baselines,
            max_calibration_cycles, gains_device);

        if (expect_pass && !*status)
        {
            sdp_gaincal_display_gains_actual_and_calculated(actual_gains_host, gains_device, num_receivers);
            // TODO perform a check on how close the actual and measured gains are to each other
            assert(1);
        }

        // clean up the allocated sdp_Mem data structures
        sdp_gaincal_free_gains_device(gains_device);
        sdp_gaincal_free_receiver_pairs_device(receiver_pairs_device);
        sdp_gaincal_free_visibilities_device(vis_measured_device);
        sdp_gaincal_free_visibilities_device(vis_predicted_device);
        sdp_gaincal_free_receiver_pairs_host(receiver_pairs_host);
        sdp_gaincal_free_visibilities_host(vis_predicted_host);
    }
    else if ((input_type!=SDP_MEM_COMPLEX_FLOAT && input_type!=SDP_MEM_COMPLEX_DOUBLE) || output_type!=input_type)
    {
        *status = SDP_ERR_DATA_TYPE;
    }
    else if (input_location!=SDP_MEM_GPU || output_location!=SDP_MEM_GPU)
    {
        *status = SDP_ERR_MEM_LOCATION;
    }
}

int main()
{
#ifdef SDP_HAVE_CUDA
    // Happy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU, single precision", true, false,
                SDP_MEM_COMPLEX_FLOAT, SDP_MEM_COMPLEX_FLOAT,
                SDP_MEM_GPU, SDP_MEM_GPU, &status);
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU, double precision", true, false,
                SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status);
        assert(status == SDP_SUCCESS);
    }

    // Unhappy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Inconsistent data types", false, false,
                SDP_MEM_COMPLEX_FLOAT, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status);
        assert(status == SDP_ERR_DATA_TYPE);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Invalid data types", false, false,
                SDP_MEM_FLOAT, SDP_MEM_FLOAT,
                SDP_MEM_GPU, SDP_MEM_GPU, &status);
        assert(status == SDP_ERR_DATA_TYPE);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Inconsistent locations", false, false,
                SDP_MEM_COMPLEX_FLOAT, SDP_MEM_COMPLEX_FLOAT,
                SDP_MEM_CPU, SDP_MEM_GPU, &status);
        assert(status == SDP_ERR_MEM_LOCATION);
    }
#endif
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, double precision invalid locations", false, false,
                SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status);
        assert(status == SDP_ERR_MEM_LOCATION);
    }
    return 0;
}
