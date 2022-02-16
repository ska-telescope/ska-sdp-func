/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "func/function_example_a/sdp_function_example_a.h"
#include "utility/sdp_logging.h"
#include "utility/sdp_mem.h"

static void run_and_check(
        const char* test_name,
        bool expect_pass,
        bool read_only_output,
        sdp_MemType output_type,
        sdp_MemLocation output_location,
        int par_a,
        int par_b,
		float par_c,
        sdp_Error* status
)
{
    // Generate some test data.
	int nDims = 1;
	int64_t output_num_elements = par_a*par_b;
    sdp_Mem* output = sdp_mem_create(output_type, output_location, nDims, &output_num_elements, status);
    sdp_mem_clear_contents(output, status);
    sdp_mem_set_read_only(output, read_only_output);

    // Call the function to test.
    SDP_LOG_INFO("Running test: %s", test_name);
    sdp_FunctionExampleA* func_a_plan = sdp_function_example_a_create_plan(par_a, par_b, par_c, status);
    sdp_function_example_a_exec(func_a_plan, output, status);
    sdp_function_example_a_free_plan(func_a_plan);

    sdp_mem_ref_dec(output);

    // Check output only if test is expected to pass.
    if (expect_pass) {
		// Pretending to check the results...
    }
}

int main()
{
    // Happy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, double precision", true, false,
                SDP_MEM_FLOAT, SDP_MEM_CPU, 5, 10, 0.1f, &status);
        assert(status == SDP_SUCCESS);
    }
#ifdef SDP_HAVE_CUDA

#endif

    // Unhappy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, double precision", false, false,
                SDP_MEM_DOUBLE, SDP_MEM_CPU, 5, 10, 0.1f, &status);
        assert(status == SDP_ERR_INVALID_ARGUMENT);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, single precision, bad argument", false, false,
                SDP_MEM_FLOAT, SDP_MEM_CPU, 10, 10, 0.1f, &status);
        assert(status == SDP_ERR_DATA_TYPE);
    }
    return 0;
}
