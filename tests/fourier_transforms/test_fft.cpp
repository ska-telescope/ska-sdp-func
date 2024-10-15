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

#include "ska-sdp-func/fourier_transforms/sdp_fft.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"

using std::complex;

const int PATTERN_LENGTH = 4;

const int NUM_POINTS = 256;


static void run_and_check(
        const char* test_name,
        const complex<double>* pattern_x,
        const complex<double>* pattern_y,
        int check_point_x,
        int check_point_y,
        bool forward,
        bool expect_pass,
        bool read_only_output,
        sdp_MemType input_type,
        sdp_MemType output_type,
        sdp_MemLocation input_location,
        sdp_MemLocation output_location,
        sdp_Error* status
)
{
    // Generate some test data.
    const int num_dims = 2;
    int64_t* data_shape = (int64_t*) calloc(num_dims, sizeof(int64_t));
    for (int i = 0; i < num_dims; ++i)
    {
        data_shape[i] = NUM_POINTS;
    }
    sdp_Mem* input_cpu = sdp_mem_create(
            input_type, SDP_MEM_CPU, num_dims, data_shape, status
    );
    sdp_Mem* output = sdp_mem_create(
            output_type, output_location, num_dims, data_shape, status
    );
    free(data_shape);
    sdp_mem_clear_contents(output, status);
    sdp_mem_set_read_only(output, read_only_output);
    const int num_elements = (int) sdp_mem_num_elements(input_cpu);
    if (input_type == SDP_MEM_COMPLEX_DOUBLE)
    {
        complex<double>* ptr = (complex<double>*)sdp_mem_data(input_cpu);
        for (int i = 0; i < num_elements; ++i)
        {
            int y = i / NUM_POINTS;
            ptr[i] = pattern_x[i % PATTERN_LENGTH] *
                    pattern_y[y % PATTERN_LENGTH];
        }
    }
    else if (input_type == SDP_MEM_COMPLEX_FLOAT)
    {
        complex<float>* ptr = (complex<float>*)sdp_mem_data(input_cpu);
        for (int i = 0; i < num_elements; ++i)
        {
            int y = i / NUM_POINTS;
            ptr[i] = complex<float>(pattern_x[i % PATTERN_LENGTH] *
                    pattern_y[y % PATTERN_LENGTH]
            );
        }
    }

    // Copy inputs to specified location.
    sdp_Mem* input = sdp_mem_create_copy(input_cpu, input_location, status);
    sdp_mem_ref_dec(input_cpu);

    // Determine position of point, possibly correcting for forward vs backward
    if (!forward && check_point_x)
        check_point_x = NUM_POINTS - check_point_x;
    if (!forward && check_point_y)
        check_point_y = NUM_POINTS - check_point_y;
    const int check_point = check_point_x + check_point_y * NUM_POINTS;

    // Call the function to test.
    SDP_LOG_INFO("Running test: %s, %s (point at %d)",
            test_name,
            forward ? "forward" : "backward",
            check_point
    );
    sdp_Fft* fft = sdp_fft_create(input, output, num_dims, forward, status);
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
            assert(std::real(ptr[check_point]) == (double) num_elements);
            for (int i = 0; i < num_elements; ++i)
            {
                if (i != check_point)
                {
                    assert(std::real(ptr[i]) < (double) 5e-12);
                }
                assert(std::imag(ptr[i]) < (double) 5e-12);
            }
        }
        else if (output_type == SDP_MEM_COMPLEX_FLOAT)
        {
            complex<float>* ptr = (complex<float>*)sdp_mem_data(output_cpu);
            assert(std::real(ptr[check_point]) == (float) num_elements);
            for (int i = 0; i < num_elements; ++i)
            {
                if (i != check_point)
                {
                    assert(std::real(ptr[i]) < (float) 5e-12);
                }
                assert(std::imag(ptr[i]) < (float) 5e-12);
            }
        }
    }
    sdp_mem_ref_dec(output_cpu);
}


int main()
{
    const complex<double> PATTERNS[][PATTERN_LENGTH] =
    {
        {1, 1, 1, 1},
        {1, -1, 1, -1},
        {1, complex<double> {0, 1.}, -1, complex<double> {0, -1.}},
        {1, complex<double> {0, -1.}, -1, complex<double> {0, 1.}}
    };
    const int PATTERN_POINT[] =
    {
        0, NUM_POINTS / 2, NUM_POINTS / 4, 3 * NUM_POINTS / 4
    };

#ifdef SDP_HAVE_CUDA
    // GPU Happy paths.
    for (int forward = 0; forward < 2; forward++)
        for (int pattern_x = 0; pattern_x < 4; pattern_x++)
            for (int pattern_y = 0; pattern_y < 4; pattern_y++)
            {
                sdp_Error status = SDP_SUCCESS;
                run_and_check("GPU, double precision",
                        PATTERNS[pattern_x], PATTERNS[pattern_y],
                        PATTERN_POINT[pattern_x], PATTERN_POINT[pattern_y],
                        forward, true, false,
                        SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                        SDP_MEM_GPU, SDP_MEM_GPU, &status
                );
                assert(status == SDP_SUCCESS);
            }
    for (int forward = 0; forward < 2; forward++)
        for (int pattern_x = 0; pattern_x < 4; pattern_x++)
            for (int pattern_y = 0; pattern_y < 4; pattern_y++)
            {
                sdp_Error status = SDP_SUCCESS;
                run_and_check("GPU, single precision",
                        PATTERNS[pattern_x], PATTERNS[pattern_y],
                        PATTERN_POINT[pattern_x], PATTERN_POINT[pattern_y],
                        forward, true, false,
                        SDP_MEM_COMPLEX_FLOAT, SDP_MEM_COMPLEX_FLOAT,
                        SDP_MEM_GPU, SDP_MEM_GPU, &status
                );
                assert(status == SDP_SUCCESS);
            }

    // Unhappy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Read-only output",
                PATTERNS[0], PATTERNS[0], 0, 0,
                true, false, true,
                SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status != SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Inconsistent data types",
                PATTERNS[0], PATTERNS[0], 0, 0,
                true, false, false,
                SDP_MEM_COMPLEX_FLOAT, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_ERR_DATA_TYPE);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Inconsistent locations",
                PATTERNS[0], PATTERNS[0], 0, 0,
                true, false, false,
                SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_ERR_MEM_LOCATION);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Wrong data type",
                PATTERNS[0], PATTERNS[0], 0, 0,
                true, false, false,
                SDP_MEM_DOUBLE, SDP_MEM_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_ERR_DATA_TYPE);
    }
#endif

    // CPU Happy paths.
    for (int forward = 0; forward < 2; forward++)
        for (int pattern_x = 0; pattern_x < 4; pattern_x++)
            for (int pattern_y = 0; pattern_y < 4; pattern_y++)
            {
                sdp_Error status = SDP_SUCCESS;
                run_and_check("CPU, single precision",
                        PATTERNS[pattern_x], PATTERNS[pattern_y],
                        PATTERN_POINT[pattern_x], PATTERN_POINT[pattern_y],
                        forward, true, false,
                        SDP_MEM_COMPLEX_FLOAT, SDP_MEM_COMPLEX_FLOAT,
                        SDP_MEM_CPU, SDP_MEM_CPU, &status
                );
                assert(status == SDP_SUCCESS);
            }
    for (int forward = 0; forward < 2; forward++)
        for (int pattern_x = 0; pattern_x < 4; pattern_x++)
            for (int pattern_y = 0; pattern_y < 4; pattern_y++)
            {
                sdp_Error status = SDP_SUCCESS;
                run_and_check("CPU, double precision",
                        PATTERNS[pattern_x], PATTERNS[pattern_y],
                        PATTERN_POINT[pattern_x], PATTERN_POINT[pattern_y],
                        forward, true, false,
                        SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
                        SDP_MEM_CPU, SDP_MEM_CPU, &status
                );
                assert(status == SDP_SUCCESS);
            }
    return 0;
}
