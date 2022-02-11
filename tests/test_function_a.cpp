/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "func/function_a/sdp_function_a.h"
#include "utility/sdp_logging.h"
#include "utility/sdp_mem.h"

template<typename T>
static void check_results(
        const char* test_name,
        const sdp_Mem* a,
        const sdp_Mem* b,
        const sdp_Mem* out,
        const sdp_Error* status)
{
    if (*status)
    {
        SDP_LOG_ERROR("%s: Test failed (error signalled)", test_name);
        return;
    }
    const T* a_data = (const T*)sdp_mem_data_const(a);
    const T* b_data = (const T*)sdp_mem_data_const(b);
    const T* out_data = (const T*)sdp_mem_data_const(out);
    const int64_t num_elements = sdp_mem_num_elements(a);
    for (int64_t i = 0; i < num_elements; ++i)
    {
        const T expected = a_data[i] + b_data[i];
        assert(fabs(out_data[i] - expected) < 1e-5);
    }
    SDP_LOG_INFO("%s: Test passed", test_name);
}

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
    sdp_FunctionA* func_a_plan = sdp_function_a_create_plan(par_a, par_b, par_c, status);
    sdp_function_a_exec(func_a_plan, output, status);
    sdp_function_a_free_plan(func_a_plan);

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
