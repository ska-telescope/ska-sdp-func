/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ska-sdp-func/vector/sdp_vector_add.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"


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
        sdp_MemType input_type,
        sdp_MemType output_type,
        sdp_MemLocation input_location,
        sdp_MemLocation output_location,
        int64_t input_num_elements,
        int64_t output_num_elements,
        sdp_Error* status)
{
    // Generate some test data.
    sdp_Mem* a = sdp_mem_create(
            input_type, SDP_MEM_CPU, 1, &input_num_elements, status);
    sdp_Mem* b = sdp_mem_create(
            input_type, SDP_MEM_CPU, 1, &input_num_elements, status);
    sdp_Mem* out = sdp_mem_create(
            output_type, output_location, 1, &output_num_elements, status);
    sdp_mem_random_fill(a, status);
    sdp_mem_random_fill(b, status);
    sdp_mem_clear_contents(out, status);
    sdp_mem_set_read_only(out, read_only_output);

    // Copy inputs to specified location.
    sdp_Mem* a_in = sdp_mem_create_copy(a, input_location, status);
    sdp_Mem* b_in = sdp_mem_create_copy(b, input_location, status);

    // Call the function to test.
    SDP_LOG_INFO("Running test: %s", test_name);
    sdp_vector_add(a_in, b_in, out, status);
    sdp_mem_ref_dec(a_in);
    sdp_mem_ref_dec(b_in);

    // Copy the output for checking.
    sdp_Mem* out2 = sdp_mem_create_copy(out, SDP_MEM_CPU, status);
    sdp_mem_ref_dec(out);

    // Check output only if test is expected to pass.
    if (expect_pass)
    {
        if (output_type == SDP_MEM_DOUBLE)
        {
            check_results<double>(test_name, a, b, out2, status);
        }
        else
        {
            check_results<float>(test_name, a, b, out2, status);
        }
    }
    sdp_mem_ref_dec(a);
    sdp_mem_ref_dec(b);
    sdp_mem_ref_dec(out2);
}


int main()
{
    // Happy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, double precision", true, false,
                SDP_MEM_DOUBLE, SDP_MEM_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, 10, 10, &status);
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, single precision", true, false,
                SDP_MEM_FLOAT, SDP_MEM_FLOAT,
                SDP_MEM_CPU, SDP_MEM_CPU, 10, 10, &status);
        assert(status == SDP_SUCCESS);
    }
#ifdef SDP_HAVE_CUDA
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU, double precision", true, false,
                SDP_MEM_DOUBLE, SDP_MEM_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, 10, 10, &status);
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU, single precision", true, false,
                SDP_MEM_FLOAT, SDP_MEM_FLOAT,
                SDP_MEM_GPU, SDP_MEM_GPU, 10, 10, &status);
        assert(status == SDP_SUCCESS);
    }
#endif

    // Unhappy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Read-only output", false, true,
                SDP_MEM_FLOAT, SDP_MEM_FLOAT,
                SDP_MEM_CPU, SDP_MEM_CPU, 10, 10, &status);
        assert(status != SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Type mismatch", false, false,
                SDP_MEM_FLOAT, SDP_MEM_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, 10, 10, &status);
        assert(status == SDP_ERR_DATA_TYPE);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Dimension mismatch", false, false,
                SDP_MEM_DOUBLE, SDP_MEM_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, 100, 10, &status);
        assert(status != SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Unsupported data type", false, false,
                SDP_MEM_CHAR, SDP_MEM_CHAR,
                SDP_MEM_CPU, SDP_MEM_CPU, 10, 10, &status);
        assert(status == SDP_ERR_DATA_TYPE);
    }
#ifdef SDP_HAVE_CUDA
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Memory location mismatch", false, false,
                SDP_MEM_FLOAT, SDP_MEM_FLOAT,
                SDP_MEM_CPU, SDP_MEM_GPU, 10, 10, &status);
        assert(status == SDP_ERR_MEM_LOCATION);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Unsupported data type", false, false,
                SDP_MEM_CHAR, SDP_MEM_CHAR,
                SDP_MEM_GPU, SDP_MEM_GPU, 10, 10, &status);
        assert(status == SDP_ERR_DATA_TYPE);
    }
#endif
    return 0;
}
