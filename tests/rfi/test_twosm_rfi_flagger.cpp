/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "ska-sdp-func/rfi/sdp_twosm_rfi_flagger.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"


static void check_results(
        const char* test_name,
        const int* flags,
        const int* predicted_flags,
        uint64_t num_elements,
        const sdp_Error* status
)
{
    if (*status)
    {
        SDP_LOG_ERROR("%s: Test failed (error signalled)", test_name);
        return;
    }
    uint64_t wrong_flags = 0;
    int counter0 = 0;
    int counter1 = 0;
    for (uint64_t f = 0; f < num_elements; f++)
    {
        if (flags[f] == 1)
        {
            counter0++;
        }
        if (predicted_flags[f] == 1)
        {
            counter1++;
        }
        if (flags[f] != predicted_flags[f])
        {
            wrong_flags++;
            std::cout << "wrong pos: " << f << std::endl;
        }
    }
    std::cout << "Number of wrong flags: " << wrong_flags << std::endl;
    std::cout << "Number of flags: " << counter0 << std::endl;
    std::cout << "Number of predicted flags: " << counter1 << std::endl;
    assert(wrong_flags == 0);
    SDP_LOG_INFO("%s: Test passed", test_name);
}


template<typename input_type>
static void threshold_calc(
        input_type* thresholds
)
{
    thresholds[0] = 0.015;
    thresholds[1] = 0.015;
}


static double rnd_norm()
{
    // NOLINTNEXTLINE: rand() is not a problem for our use case.
    return (double) rand() / (double) RAND_MAX;
}


template<typename input_type>
static void data_preparation(
        std::complex<input_type>* visibilities,
        int* predicted_flags,
        input_type* thresholds,
        uint64_t num_antennas,
        uint64_t num_timesamples,
        uint64_t num_channels,
        uint64_t num_pols,
        int num_rfi_spikes
)
{
    uint64_t num_baselines = num_antennas * (num_antennas + 1) / 2;
    uint64_t timesample_block = num_channels * num_pols * num_baselines;
    uint64_t baseline_block = num_channels * num_pols;
    uint64_t frequency_block = num_pols;

    double threshold = thresholds[0];
    for (int s = 0; s < num_rfi_spikes; s++)
    {
        uint64_t time = 5 + (uint64_t) (rnd_norm() * (num_timesamples - 5));
        uint64_t freq = (uint64_t) (rnd_norm() * (num_channels - 1));
        // std::cout << time << "    " << freq << std::endl;
        for (uint64_t b = 0; b < num_baselines; b++)
        {
            for (uint64_t p = 0; p < num_pols; p++)
            {
                uint64_t pos = time * timesample_block + b * baseline_block +
                        freq * frequency_block + p;
                std::complex<input_type> ctemp(threshold + 0.01, 0);
                // std::cout << pos << "   " << time << "   " << freq << std::endl;
                visibilities[pos] = ctemp;
                predicted_flags[pos] = 1;
            }
        }
    }
}


static void run_and_check(
        const char* test_name,
        bool expect_pass,
        bool read_only_output,
        sdp_MemType visibilities_type,
        sdp_MemType threshold_type,
        sdp_MemType antennas_type,
        sdp_MemType flags_type,
        sdp_MemLocation visibilities_location,
        sdp_MemLocation thresholds_location,
        sdp_MemLocation antennas_location,
        sdp_MemLocation flags_location,
        sdp_Error* status
)
{
    // Generate some test data.
    const uint64_t num_timesamples     = 1000;
    const uint64_t num_antennas        = 6;
    const uint64_t num_baselines       = 21;
    const uint64_t num_channels        = 200;
    const uint64_t num_pols            = 4;
    int num_rfi_spikes = 14;

    int64_t visibilities_shape[] = {
        num_timesamples, num_baselines, num_channels, num_pols
    };
    int64_t antennas_shape[] = {num_antennas};
    int64_t threshold_shape[] = {2};
    sdp_Mem* visibilities = sdp_mem_create(
            visibilities_type, SDP_MEM_CPU, 4, visibilities_shape, status
    );
    sdp_Mem* antennas = sdp_mem_create(
            antennas_type, SDP_MEM_CPU, 1, antennas_shape, status
    );
    sdp_Mem* predicted_flags = sdp_mem_create(
            SDP_MEM_INT, SDP_MEM_CPU, 4, visibilities_shape, status
    );
    sdp_Mem* flags = sdp_mem_create(
            flags_type, SDP_MEM_CPU, 4, visibilities_shape, status
    );
    sdp_Mem* thresholds = sdp_mem_create(
            threshold_type, SDP_MEM_CPU, 1, threshold_shape, status
    );

    sdp_mem_clear_contents(visibilities, status);
    sdp_mem_clear_contents(thresholds, status);
    sdp_mem_clear_contents(predicted_flags, status);
    sdp_mem_clear_contents(flags, status);

    // Prepare data for the test
    // Thresholds
    if (threshold_type == SDP_MEM_FLOAT)
    {
        threshold_calc((float*) sdp_mem_data(thresholds));
    }
    else if (threshold_type == SDP_MEM_DOUBLE)
    {
        threshold_calc((double*) sdp_mem_data(thresholds));
    }
    // Visibilities and predicted flags
    if (visibilities_type == SDP_MEM_COMPLEX_FLOAT &&
            threshold_type == SDP_MEM_FLOAT)
    {
        data_preparation(
                (std::complex<float>*) sdp_mem_data(visibilities),
                (int*) sdp_mem_data(predicted_flags),
                (float*) sdp_mem_data(thresholds),
                num_antennas,
                num_timesamples,
                num_channels,
                num_pols,
                num_rfi_spikes
        );
    }
    else if (visibilities_type == SDP_MEM_COMPLEX_DOUBLE &&
            threshold_type == SDP_MEM_DOUBLE)
    {
        data_preparation(
                (std::complex<double>*) sdp_mem_data(visibilities),
                (int*) sdp_mem_data(predicted_flags),
                (double*) sdp_mem_data(thresholds),
                num_antennas,
                num_timesamples,
                num_channels,
                num_pols,
                num_rfi_spikes
        );
    }

    // Copy inputs to specified location.
    sdp_Mem* visibilities_in = sdp_mem_create_copy(
            visibilities, visibilities_location, status
    );
    sdp_Mem* thresholds_in = sdp_mem_create_copy(
            thresholds, thresholds_location, status
    );
    sdp_Mem* antennas_in = sdp_mem_create_copy(
            antennas, antennas_location, status
    );
    sdp_Mem* flags_in = sdp_mem_create_copy(flags, flags_location, status);
    sdp_mem_set_read_only(flags_in, read_only_output);
    const uint64_t num_elems = (uint64_t) sdp_mem_num_elements(visibilities);
    sdp_mem_ref_dec(visibilities);
    sdp_mem_ref_dec(thresholds);
    sdp_mem_ref_dec(flags);

    // Call the function to test.
    SDP_LOG_INFO("Running test: %s", test_name);
    sdp_twosm_algo_flagger(visibilities_in,
            thresholds_in,
            antennas_in,
            flags_in,
            status
    );
    sdp_mem_ref_dec(visibilities_in);
    sdp_mem_ref_dec(thresholds_in);

    // Copy the output for checking.
    sdp_Mem* flags_out = sdp_mem_create_copy(flags_in, SDP_MEM_CPU, status);
    sdp_mem_ref_dec(flags_in);

    // Check output only if test is expected to pass.
    if (expect_pass)
    {
        check_results(
                test_name,
                (int*) sdp_mem_data(flags_out),
                (int*) sdp_mem_data(predicted_flags),
                num_elems,
                status
        );
    }
    sdp_mem_ref_dec(flags_out);
    sdp_mem_ref_dec(predicted_flags);
}


int main()
{
    // Happy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, double precision", true, false,
                SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_DOUBLE, SDP_MEM_INT,
                SDP_MEM_INT, SDP_MEM_CPU, SDP_MEM_CPU, SDP_MEM_CPU, SDP_MEM_CPU,
                &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, single precision", true, false,
                SDP_MEM_COMPLEX_FLOAT, SDP_MEM_FLOAT, SDP_MEM_INT, SDP_MEM_INT,
                SDP_MEM_CPU, SDP_MEM_CPU, SDP_MEM_CPU, SDP_MEM_CPU,
                &status
        );
        assert(status == SDP_SUCCESS);
    }

    // Unhappy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Read-only output", false, true,
                SDP_MEM_COMPLEX_FLOAT, SDP_MEM_FLOAT, SDP_MEM_INT, SDP_MEM_INT,
                SDP_MEM_CPU, SDP_MEM_CPU, SDP_MEM_CPU, SDP_MEM_CPU,
                &status
        );
        assert(status != SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Wrong flags type", false, false,
                SDP_MEM_COMPLEX_FLOAT, SDP_MEM_FLOAT, SDP_MEM_INT,
                SDP_MEM_FLOAT, SDP_MEM_CPU, SDP_MEM_CPU, SDP_MEM_CPU,
                SDP_MEM_CPU, &status
        );
        assert(status == SDP_ERR_DATA_TYPE);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Wrong visibility type", false, false,
                SDP_MEM_FLOAT, SDP_MEM_FLOAT, SDP_MEM_INT, SDP_MEM_INT,
                SDP_MEM_CPU, SDP_MEM_CPU, SDP_MEM_CPU, SDP_MEM_CPU,
                &status
        );
        assert(status == SDP_ERR_DATA_TYPE);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Wrong threshold type", false, false,
                SDP_MEM_COMPLEX_FLOAT, SDP_MEM_DOUBLE, SDP_MEM_INT, SDP_MEM_INT,
                SDP_MEM_CPU, SDP_MEM_CPU, SDP_MEM_CPU, SDP_MEM_CPU,
                &status
        );
        assert(status == SDP_ERR_DATA_TYPE);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Wrong antennas type", false, false,
                SDP_MEM_COMPLEX_FLOAT, SDP_MEM_DOUBLE, SDP_MEM_FLOAT,
                SDP_MEM_INT, SDP_MEM_CPU, SDP_MEM_CPU, SDP_MEM_CPU, SDP_MEM_CPU,
                &status
        );
        assert(status == SDP_ERR_DATA_TYPE);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Unsupported GPU location", false, false,
                SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_DOUBLE, SDP_MEM_INT,
                SDP_MEM_INT, SDP_MEM_GPU, SDP_MEM_GPU, SDP_MEM_GPU, SDP_MEM_CPU,
                &status
        );
        assert(status == SDP_ERR_MEM_LOCATION);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Wrong flag location", false, false,
                SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_DOUBLE, SDP_MEM_INT,
                SDP_MEM_INT, SDP_MEM_CPU, SDP_MEM_CPU, SDP_MEM_GPU, SDP_MEM_CPU,
                &status
        );
        assert(status == SDP_ERR_MEM_LOCATION);
    }

    return 0;
}
