/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ska-sdp-func/rfi_flagger/sdp_rfi_flagger.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"

#define INDEX_3D(N3, N2, N1, I3, I2, I1)         (N1 * (N2 * I3 + I2) + I1)
#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)

using std::complex;

static void check_results(
        const char* test_name,
        const sdp_Error* status)
{
    if (*status) {
        SDP_LOG_ERROR("%s: Test failed (error signalled)", test_name);
        return;
    }
    SDP_LOG_INFO("%s: Test passed", test_name);
}

static void run_and_check(
        const char* test_name,
        bool expect_pass,
        bool read_only_output,
        sdp_MemType visibilities_type,
        sdp_MemType threshold_type,
        sdp_MemType flags_type,
        sdp_MemLocation visibilities_location,
        sdp_MemLocation thresholds_location,
        sdp_MemLocation flags_location,
        sdp_Error* output_status
)
{
    sdp_Error status = SDP_SUCCESS;
    // Generate some test data.
    const uint64_t num_timesamples     = 100;
    const uint64_t num_baselines       = 15;
    const uint64_t num_channels        = 128;
    const uint64_t num_polarisations   = 4;
    const uint64_t max_sequence_length = 32;
    const uint64_t num_sequence_el     = (uint64_t) (log(max_sequence_length)/log(2));
    
    int64_t visibilities_shape[] = {num_timesamples, num_baselines, num_channels, num_polarisations};
    int64_t threshold_shape[] = {num_sequence_el};
    sdp_Mem* visibilities = sdp_mem_create(visibilities_type, SDP_MEM_CPU, 4, visibilities_shape, &status);
    sdp_Mem* flags = sdp_mem_create(flags_type, SDP_MEM_CPU, 4, visibilities_shape, &status);
    sdp_Mem* thresholds = sdp_mem_create(threshold_type, SDP_MEM_CPU, 1, threshold_shape, &status);
    
    sdp_mem_random_fill(visibilities, &status);
    sdp_mem_random_fill(thresholds, &status);
    sdp_mem_clear_contents(flags, &status);

    // Copy inputs to specified location.
    sdp_Mem* visibilities_in = sdp_mem_create_copy(visibilities, visibilities_location, &status);
    sdp_Mem* thresholds_in = sdp_mem_create_copy(thresholds, thresholds_location, &status);
    sdp_Mem* flags_in = sdp_mem_create_copy(flags, flags_location, &status);
    sdp_mem_set_read_only(flags_in, read_only_output);
    sdp_mem_ref_dec(visibilities);
    sdp_mem_ref_dec(thresholds);
    sdp_mem_ref_dec(flags);

    // Call the function to test.
    SDP_LOG_INFO("Running test: %s", test_name);
    sdp_sum_threshold_rfi_flagger(visibilities_in, thresholds_in, flags_in, max_sequence_length, output_status);
    sdp_mem_ref_dec(visibilities_in);
    sdp_mem_ref_dec(thresholds_in);

    // Copy the output for checking.
    sdp_Mem* flags_out = sdp_mem_create_copy(flags_in, SDP_MEM_CPU, &status);
    sdp_mem_ref_dec(flags_in);

    // Check output only if test is expected to pass.
    if (expect_pass) {
        check_results(test_name, output_status);
    }
    sdp_mem_ref_dec(flags_out);
}

int main()
{
    // Happy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, double precision", true, false,
                SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_DOUBLE, SDP_MEM_INT,
                SDP_MEM_CPU, SDP_MEM_CPU, SDP_MEM_CPU, 
                &status);
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, single precision", true, false,
                SDP_MEM_COMPLEX_FLOAT, SDP_MEM_FLOAT, SDP_MEM_INT,
                SDP_MEM_CPU, SDP_MEM_CPU, SDP_MEM_CPU, 
                &status);
        assert(status == SDP_SUCCESS);
    }

    // Unhappy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Read-only output", false, true,
                SDP_MEM_COMPLEX_FLOAT, SDP_MEM_FLOAT, SDP_MEM_INT,
                SDP_MEM_CPU, SDP_MEM_CPU, SDP_MEM_CPU, 
                &status);
        assert(status != SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Wrong flags type", false, false,
                SDP_MEM_COMPLEX_FLOAT, SDP_MEM_FLOAT, SDP_MEM_FLOAT,
                SDP_MEM_CPU, SDP_MEM_CPU, SDP_MEM_CPU, 
                &status);
        assert(status == SDP_ERR_DATA_TYPE);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Wrong visibility type", false, false,
                SDP_MEM_FLOAT, SDP_MEM_FLOAT, SDP_MEM_INT,
                SDP_MEM_CPU, SDP_MEM_CPU, SDP_MEM_CPU, 
                &status);
        assert(status == SDP_ERR_DATA_TYPE);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Wrong threshold type", false, false,
                SDP_MEM_COMPLEX_FLOAT, SDP_MEM_DOUBLE, SDP_MEM_INT,
                SDP_MEM_CPU, SDP_MEM_CPU, SDP_MEM_CPU, 
                &status);
        assert(status == SDP_ERR_DATA_TYPE);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Unsupported GPU location", false, false,
                SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_DOUBLE, SDP_MEM_INT,
                SDP_MEM_GPU, SDP_MEM_GPU, SDP_MEM_GPU, 
                &status);
        assert(status == SDP_ERR_MEM_LOCATION);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Wrong flag location", false, false,
                SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_DOUBLE, SDP_MEM_INT,
                SDP_MEM_CPU, SDP_MEM_CPU, SDP_MEM_GPU, 
                &status);
        assert(status == SDP_ERR_MEM_LOCATION);
    }

    return 0;
}

