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

#include "ska-sdp-func/fft/sdp_fft.h"
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
        sdp_Error* status)
{
    // Generate some test data.
    const int num_dims = 2;
    const int num_points = 256;
    int64_t* data_shape = (int64_t*) calloc(num_dims, sizeof(int64_t));
    for (int i = 0; i < num_dims; ++i)
    {
        data_shape[i] = num_points;
    }
    sdp_Mem* input_cpu = sdp_mem_create(
            input_type, SDP_MEM_CPU, num_dims, data_shape, status);
    sdp_Mem* output = sdp_mem_create(
            output_type, output_location, num_dims, data_shape, status);
    free(data_shape);
    sdp_mem_clear_contents(output, status);
    sdp_mem_set_read_only(output, read_only_output);
    const int num_elements = (int) sdp_mem_num_elements(input_cpu);
    if (input_type == SDP_MEM_COMPLEX_DOUBLE)
    {
        complex<double>* ptr = (complex<double>*)sdp_mem_data(input_cpu);
        for (int i = 0; i < num_elements; ++i)
        {
            ptr[i] = complex<double>(1.0, 0.0);
        }
    }
    else if (input_type == SDP_MEM_COMPLEX_FLOAT)
    {
        complex<float>* ptr = (complex<float>*)sdp_mem_data(input_cpu);
        for (int i = 0; i < num_elements; ++i)
        {
            ptr[i] = complex<float>(1.0f, 0.0f);
        }
    }

    // Copy inputs to specified location.
    sdp_Mem* input = sdp_mem_create_copy(input_cpu, input_location, status);
    sdp_mem_ref_dec(input_cpu);

    // Call the function to test.
    SDP_LOG_INFO("Running test: %s", test_name);
    sdp_Fft* fft = sdp_fft_create(input, output, num_dims, 1, status);
    sdp_fft_exec(fft, input, output, status);
    sdp_fft_free(fft);
    sdp_mem_ref_dec(input);

    // Copy the output for checking.
    sdp_Mem* output_cpu = sdp_mem_create_copy(output, SDP_MEM_CPU, status);
    sdp_mem_ref_dec(output);

    // Check output only if test is expected to pass.
    if (expect_pass && !*status)
    {
        if (output_type == SDP_MEM_COMPLEX_DOUBLE)
        {
            complex<double>* ptr = (complex<double>*)sdp_mem_data(output_cpu);
            assert(std::real(ptr[0]) == (double) num_elements);
            for (int i = 1; i < num_elements; ++i)
            {
                assert(std::real(ptr[i]) == (double) 0);
            }
        }
        else if (output_type == SDP_MEM_COMPLEX_FLOAT)
        {
            complex<float>* ptr = (complex<float>*)sdp_mem_data(output_cpu);
            assert(std::real(ptr[0]) == (float) num_elements);
            for (int i = 1; i < num_elements; ++i)
            {
                assert(std::real(ptr[i]) == (float) 0);
            }
        }
    }
    sdp_mem_ref_dec(output_cpu);
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

    // Happy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, float precision", true, false,
                SDP_MEM_COMPLEX_FLOAT, SDP_MEM_COMPLEX_FLOAT,
                SDP_MEM_CPU, SDP_MEM_CPU, &status);
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, double precision", true, false,
                SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status);
        assert(status == SDP_SUCCESS);
    }
    return 0;
}
