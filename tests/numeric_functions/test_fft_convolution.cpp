/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "ska-sdp-func/numeric_functions/sdp_fft_convolution.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"

#include <cassert>


static void run_and_check(
        const char* test_name,
        bool read_only_output,
        sdp_MemType input_type,
        sdp_MemType output_type,
        sdp_MemLocation input_location,
        sdp_MemLocation output_location,
        sdp_Error* status
)
{
    // settings
    int64_t in1_dim = 128;
    int64_t in2_dim = 256;

    const int64_t in1_shape[] = {in1_dim, in1_dim};
    const int64_t in2_shape[] = {in2_dim, in2_dim};

    // create test data
    sdp_Mem* in1 =
            sdp_mem_create(input_type, SDP_MEM_CPU, 2, in1_shape, status);
    sdp_mem_random_fill(in1, status);

    sdp_Mem* in2 =
            sdp_mem_create(input_type, SDP_MEM_CPU, 2, in2_shape, status);
    sdp_mem_random_fill(in2, status);

    // create output
    sdp_Mem* out = sdp_mem_create(output_type,
            output_location,
            2,
            in1_shape,
            status
    );
    sdp_mem_clear_contents(out, status);
    sdp_mem_set_read_only(out, read_only_output);

    // Copy inputs to specified location.
    sdp_Mem* in1_copy = sdp_mem_create_copy(in1, input_location, status);
    sdp_Mem* in2_copy = sdp_mem_create_copy(in2, input_location, status);

    // call function to test
    // this test only checks the interface
    // the correctness of the algorithm is checked in test_fft_convolution.py against scipy
    SDP_LOG_INFO("Running test: %s", test_name);
    sdp_fft_convolution(
            in1_copy,
            in2_copy,
            out,
            status
    );

    // release memory
    sdp_mem_ref_dec(in1);
    sdp_mem_ref_dec(in2);
    sdp_mem_ref_dec(out);
    sdp_mem_ref_dec(in1_copy);
    sdp_mem_ref_dec(in2_copy);
}


int main()
{
    SDP_LOG_INFO("start of test:");

    // happy paths
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, double precision", false,
                SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, single precision", false,
                SDP_MEM_COMPLEX_FLOAT, SDP_MEM_COMPLEX_FLOAT,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_SUCCESS);
    }

    // unhappy paths
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, Type mis-match", false,
                SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_FLOAT,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_ERR_DATA_TYPE);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, read-only output", true,
                SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_ERR_RUNTIME);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, non-complex input", false,
                SDP_MEM_DOUBLE, SDP_MEM_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_ERR_DATA_TYPE);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Unsuported data type", false,
                SDP_MEM_CHAR, SDP_MEM_CHAR,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_ERR_DATA_TYPE);
    }

    #ifdef SDP_HAVE_CUDA
    // Happy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU, double precision", false,
                SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU, single precision", false,
                SDP_MEM_COMPLEX_FLOAT, SDP_MEM_COMPLEX_FLOAT,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_SUCCESS);
    }

    // Unhappy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("memory location mis-match", false,
                SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_ERR_MEM_LOCATION);
    }

#endif

    return 0;
}
