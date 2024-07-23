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


static void run_and_check(
        const char* test_name,
        bool expect_pass,
        bool read_only_output,
        sdp_MemType input_type,
        sdp_MemType output_type,
        // sdp_MemLocation input_location,
        // sdp_MemLocation output_location,
        sdp_Error* status
)
{
    // Generate some test data.
    const int num_dims = 2;
    const int num_points = 256;
    int64_t* data_shape = (int64_t*) calloc(num_dims, sizeof(int64_t));
    for (int i = 0; i < num_dims; ++i)
    {
        data_shape[i] = num_points;
    }
    sdp_Mem* input = sdp_mem_create(
            input_type, SDP_MEM_CPU, num_dims, data_shape, status
    );
    sdp_Mem* output = sdp_mem_create(
            output_type, SDP_MEM_CPU, num_dims, data_shape, status
    );
    free(data_shape);
    sdp_mem_clear_contents(output, status);
    sdp_mem_set_read_only(output, read_only_output);
    const int num_elements = (int) sdp_mem_num_elements(input);
    if (input_type == SDP_MEM_COMPLEX_DOUBLE)
    {
        complex<double>* ptr = (complex<double>*)sdp_mem_data(input);
        for (int i = 0; i < num_elements; ++i)
        {
            ptr[i] = complex<double>(1.0, 0.0);
        }
    }
    else if (input_type == SDP_MEM_COMPLEX_FLOAT)
    {
        complex<float>* ptr = (complex<float>*)sdp_mem_data(input);
        for (int i = 0; i < num_elements; ++i)
        {
            ptr[i] = complex<float>(1.0f, 0.0f);
        }
    }
#ifdef SDP_HAVE_CUDA
    const int num_streams = 4;
    const int batch_size = 32;
    int64_t data_1d_shape[] = {num_points* batch_size* num_streams};
    sdp_MemType data_1d_type = input_type;

    // status[0] = SDP_SUCCESS;
    sdp_Mem* idata_1d =
            sdp_mem_create(data_1d_type,
            SDP_MEM_GPU,
            1,
            data_1d_shape,
            status
            );
    sdp_mem_clear_contents(idata_1d, status);

    sdp_Mem* odata_1d =
            sdp_mem_create(data_1d_type,
            SDP_MEM_GPU,
            1,
            data_1d_shape,
            status
            );
    sdp_mem_clear_contents(odata_1d, status);

    int is_forward = 1;

    // Call the function to test.
    SDP_LOG_INFO("Running test: %s", test_name);
    sdp_Fft_extended* fft_extended = sdp_fft_extended_create(
            idata_1d,
            odata_1d,
            1,
            is_forward,
            num_streams,
            batch_size,
            status
    );

    sdp_fft_extended_exec(
            fft_extended,
            input,  // Input grid
            output, // Output dirty image
            status
    );

    sdp_fft_extended_free(
            fft_extended,
            status
    );
#endif
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
                // SDP_MEM_GPU, SDP_MEM_GPU,
                &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU, single precision", true, false,
                SDP_MEM_COMPLEX_FLOAT, SDP_MEM_COMPLEX_FLOAT,
                // SDP_MEM_GPU, SDP_MEM_GPU,
                &status
        );
        assert(status == SDP_SUCCESS);
    }
#endif

    return 0;
}
