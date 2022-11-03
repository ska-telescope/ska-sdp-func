/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ska-sdp-func/function_example_a/sdp_function_example_a.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"


static void run_and_check(
        const char* test_name,
        bool expect_pass,
        bool read_only_output,
        bool too_small,
        sdp_MemType output_type,
        sdp_MemLocation output_location,
        int par_a,
        int par_b,
        float par_c,
        sdp_Error* status
)
{
    // Generate some test data.
    int num_dims = 1;
    int64_t output_num_elements = too_small ? 1 : par_a * par_b;
    sdp_Mem* output = sdp_mem_create(output_type, output_location,
            num_dims, &output_num_elements, status);
    sdp_mem_clear_contents(output, status);
    sdp_mem_set_read_only(output, read_only_output);

    // Call the function to test.
    SDP_LOG_INFO("Running test: %s", test_name);
    sdp_FunctionExampleA* func_a = sdp_function_example_a_create_plan(
            par_a, par_b, par_c, status);
    sdp_function_example_a_exec(func_a, output, status);
    sdp_function_example_a_free_plan(func_a);

    // Check output only if test is expected to pass.
    if (expect_pass)
    {
        // Pretending to check the results...
    }
    sdp_mem_ref_dec(output);
}


int main()
{
    // Happy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Single precision", true, false, false,
                SDP_MEM_FLOAT, SDP_MEM_CPU, 5, 10, 0.1f, &status);
        assert(status == SDP_SUCCESS);
    }

    // Unhappy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Double precision (not supported)", false, false, false,
                SDP_MEM_DOUBLE, SDP_MEM_CPU, 5, 10, 0.1f, &status);
        assert(status == SDP_ERR_DATA_TYPE);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Output too small", false, false, true,
                SDP_MEM_FLOAT, SDP_MEM_CPU, 5, 10, 0.1f, &status);
        assert(status != SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Read-only output", false, true, false,
                SDP_MEM_FLOAT, SDP_MEM_CPU, 5, 10, 0.1f, &status);
        assert(status != SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Invalid argument value", false, false, false,
                SDP_MEM_FLOAT, SDP_MEM_CPU, 10, 10, 0.1f, &status);
        assert(status == SDP_ERR_INVALID_ARGUMENT);
    }
#ifdef SDP_HAVE_CUDA
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU version (not implemented)", false, false, false,
                SDP_MEM_FLOAT, SDP_MEM_GPU, 5, 10, 0.1f, &status);
        assert(status == SDP_ERR_MEM_LOCATION);
    }
#endif
    return 0;
}
