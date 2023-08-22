/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif


#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"
#include "ska-sdp-func/visibility/sdp_weighting.h"

#define INDEX_3D(N3, N2, N1, I3, I2, I1)         (N1 * (N2 * I3 + I2) + I1)
#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) \
    (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)


static void create_test_data_single(
        double* freq_ptr,
        double* uvw_ptr,
        float* input_weights_ptr,
        float* weights_grid_uv_ptr
)
{
    freq_ptr[0] = 1e9; freq_ptr[1] = 1.1e9; freq_ptr[2] = 1.2e9;

    double uvw_values[1][3][3] = {{{2, 3, 5}, {7, 11, 13}, {17, 19, 23}}};
    for (int k = 0; k < 3; k++)
    {
        for (int i = 0; i < 3; i++)
        {
            const unsigned int i_val = INDEX_3D(1, 3, 3, 0, k, i);
            uvw_ptr[i_val] = uvw_values[0][k][i];
        }
    }

    float input_weights_values[3] = {10, 31, 21};
    for (int k = 0; k < 3; k++)
    {
        for (int i = 0; i < 3; i++)
        {
            const unsigned int i_val = INDEX_4D(1, 3, 3, 1, 0, k, i, 0);
            input_weights_ptr[i_val] = input_weights_values[i];
        }
    }

    for (int i = 0; i < 9; i++)
    {
        weights_grid_uv_ptr[i] = 0;
    }
}


static void create_test_data_double(
        double* freq_ptr,
        double* uvw_ptr,
        double* input_weights_ptr,
        double* weights_grid_uv_ptr
)
{
    freq_ptr[0] = 1e9; freq_ptr[1] = 1.1e9; freq_ptr[2] = 1.2e9;

    double uvw_values[1][3][3] = {{{2, 3, 5}, {7, 11, 13}, {17, 19, 23}}};
    for (int k = 0; k < 3; k++)
    {
        for (int i = 0; i < 3; i++)
        {
            const unsigned int i_val = INDEX_3D(1, 3, 3, 0, k, i);
            uvw_ptr[i_val] = uvw_values[0][k][i];
        }
    }

    double input_weights_values[3] = {10, 31, 21};
    for (int k = 0; k < 3; k++)
    {
        for (int i = 0; i < 3; i++)
        {
            const unsigned int i_val = INDEX_4D(1, 3, 3, 1, 0, k, i, 0);
            input_weights_ptr[i_val] = input_weights_values[i];
        }
    }

    for (int i = 0; i < 9; i++)
    {
        weights_grid_uv_ptr[i] = 0;
    }
}


static void run_and_check_uniform(
        const char* test_name,
        bool read_only_output,
        sdp_MemType uvw_type,
        sdp_MemType weight_type,
        sdp_MemType freqs_type,
        sdp_MemLocation output_location,
        sdp_Error* status
)
{
    // Generate some test data for CPU.
    int64_t freqs_shape[] = {3};
    sdp_Mem* freqs = sdp_mem_create(
            freqs_type, SDP_MEM_CPU, 1, freqs_shape, status
    );
    double* freqs_ptr = (double*) sdp_mem_data(freqs);

    sdp_mem_clear_contents(freqs, status);

    int64_t uvw_shape[] = {1, 3, 3};
    sdp_Mem* uvw = sdp_mem_create(
            uvw_type, SDP_MEM_CPU, 3, uvw_shape, status
    );

    double* uvw_ptr = (double*) sdp_mem_data(uvw);

    sdp_mem_clear_contents(uvw, status);

    int64_t weight_shape[] = {1, 3, 3, 1};
    sdp_Mem* input_weights = sdp_mem_create(
            weight_type, SDP_MEM_CPU, 4, weight_shape, status
    );

    double* input_weights_ptr = (double*)sdp_mem_data(input_weights);
    sdp_mem_clear_contents(input_weights, status);

    sdp_Mem* output_weights = sdp_mem_create(
            weight_type, SDP_MEM_CPU, 4, weight_shape, status
    );

    sdp_mem_clear_contents(output_weights, status);

    int64_t weights_grid_uv_shape[] = {3, 3, 1};
    sdp_Mem* weights_grid_uv = sdp_mem_create(
            weight_type, SDP_MEM_CPU, 3, weights_grid_uv_shape, status
    );

    double* weights_grid_uv_ptr = (double*)sdp_mem_data(weights_grid_uv);

    sdp_mem_clear_contents(weights_grid_uv, status);

    if (weight_type == SDP_MEM_FLOAT)
    {
        float* input_weights_ptr = (float*)sdp_mem_data(input_weights);
        sdp_mem_clear_contents(input_weights, status);

        float* weights_grid_uv_ptr = (float*)sdp_mem_data(weights_grid_uv);
        sdp_mem_clear_contents(weights_grid_uv, status);

        create_test_data_single(freqs_ptr,
                uvw_ptr,
                input_weights_ptr,
                weights_grid_uv_ptr
        );
    }
    else
    {
        create_test_data_double(freqs_ptr,
                uvw_ptr,
                input_weights_ptr,
                weights_grid_uv_ptr
        );
    }

    sdp_Mem* uvw_cp = sdp_mem_create_copy(uvw, output_location, status);
    sdp_mem_ref_dec(uvw);
    sdp_Mem* freqs_cp = sdp_mem_create_copy(freqs, output_location, status);
    sdp_mem_ref_dec(freqs);
    sdp_Mem* weights_grid_uv_cp = sdp_mem_create_copy(weights_grid_uv,
            output_location,
            status
    );
    sdp_mem_ref_dec(weights_grid_uv);
    sdp_Mem* input_weights_cp = sdp_mem_create_copy(input_weights,
            output_location,
            status
    );
    sdp_mem_ref_dec(input_weights);
    sdp_Mem* output_weights_cp = sdp_mem_create_copy(output_weights,
            output_location,
            status
    );
    sdp_mem_ref_dec(output_weights);
    sdp_mem_set_read_only(output_weights_cp, read_only_output);

    double max_abs_uv = 16011.076569511299;

    // Call the function to test.
    SDP_LOG_INFO("Running test: %s", test_name);
    sdp_weighting_uniform(uvw_cp,
            freqs_cp,
            max_abs_uv,
            weights_grid_uv_cp,
            input_weights_cp,
            output_weights_cp,
            status
    );

    sdp_mem_ref_dec(uvw_cp);
    sdp_mem_ref_dec(freqs_cp);
    sdp_mem_ref_dec(weights_grid_uv_cp);
    sdp_mem_ref_dec(input_weights_cp);
    sdp_mem_ref_dec(output_weights_cp);
}


static void run_and_check_briggs(
        const char* test_name,
        bool read_only_output,
        sdp_MemType uvw_type,
        sdp_MemType weight_type,
        sdp_MemType freqs_type,
        sdp_MemLocation output_location,
        sdp_Error* status
)
{
    // Generate some test data for CPU.
    int64_t freqs_shape[] = {3};
    sdp_Mem* freqs = sdp_mem_create(
            freqs_type, SDP_MEM_CPU, 1, freqs_shape, status
    );
    double* freqs_ptr = (double*) sdp_mem_data(freqs);

    sdp_mem_clear_contents(freqs, status);

    int64_t uvw_shape[] = {1, 3, 3};
    sdp_Mem* uvw = sdp_mem_create(
            uvw_type, SDP_MEM_CPU, 3, uvw_shape, status
    );

    double* uvw_ptr = (double*) sdp_mem_data(uvw);

    sdp_mem_clear_contents(uvw, status);

    int64_t weight_shape[] = {1, 3, 3, 1};
    sdp_Mem* input_weights = sdp_mem_create(
            weight_type, SDP_MEM_CPU, 4, weight_shape, status
    );

    double* input_weights_ptr = (double*)sdp_mem_data(input_weights);
    sdp_mem_clear_contents(input_weights, status);

    sdp_Mem* output_weights = sdp_mem_create(
            weight_type, SDP_MEM_CPU, 4, weight_shape, status
    );

    sdp_mem_clear_contents(output_weights, status);

    int64_t weights_grid_uv_shape[] = {3, 3, 1};
    sdp_Mem* weights_grid_uv = sdp_mem_create(
            weight_type, SDP_MEM_CPU, 3, weights_grid_uv_shape, status
    );

    double* weights_grid_uv_ptr = (double*)sdp_mem_data(weights_grid_uv);

    sdp_mem_clear_contents(weights_grid_uv, status);

    if (weight_type == SDP_MEM_FLOAT)
    {
        float* input_weights_ptr = (float*)sdp_mem_data(input_weights);
        sdp_mem_clear_contents(input_weights, status);

        float* weights_grid_uv_ptr = (float*)sdp_mem_data(weights_grid_uv);
        sdp_mem_clear_contents(weights_grid_uv, status);

        create_test_data_single(freqs_ptr,
                uvw_ptr,
                input_weights_ptr,
                weights_grid_uv_ptr
        );
    }
    else
    {
        create_test_data_double(freqs_ptr,
                uvw_ptr,
                input_weights_ptr,
                weights_grid_uv_ptr
        );
    }

    sdp_Mem* uvw_cp = sdp_mem_create_copy(uvw, output_location, status);
    sdp_mem_ref_dec(uvw);
    sdp_Mem* freqs_cp = sdp_mem_create_copy(freqs, output_location, status);
    sdp_mem_ref_dec(freqs);
    sdp_Mem* weights_grid_uv_cp = sdp_mem_create_copy(weights_grid_uv,
            output_location,
            status
    );
    sdp_mem_ref_dec(weights_grid_uv);
    sdp_Mem* input_weights_cp = sdp_mem_create_copy(input_weights,
            output_location,
            status
    );
    sdp_mem_ref_dec(input_weights);
    sdp_Mem* output_weights_cp = sdp_mem_create_copy(output_weights,
            output_location,
            status
    );
    sdp_mem_ref_dec(output_weights);
    sdp_mem_set_read_only(output_weights_cp, read_only_output);

    double max_abs_uv = 16011.076569511299;
    double robust_param = -2.0;

    // Call the function to test.
    SDP_LOG_INFO("Running test: %s", test_name);
    sdp_weighting_briggs(uvw_cp,
            freqs_cp,
            max_abs_uv,
            robust_param,
            weights_grid_uv_cp,
            input_weights_cp,
            output_weights_cp,
            status
    );

    sdp_mem_ref_dec(uvw_cp);
    sdp_mem_ref_dec(freqs_cp);
    sdp_mem_ref_dec(weights_grid_uv_cp);
    sdp_mem_ref_dec(input_weights_cp);
    sdp_mem_ref_dec(output_weights_cp);
}


int main()
{
    // Check for Happy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_uniform("Single precision",
                false,
                SDP_MEM_DOUBLE,
                SDP_MEM_FLOAT,
                SDP_MEM_DOUBLE,
                SDP_MEM_CPU,
                &status
        );
        assert(status == SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_briggs("Single precision",
                false,
                SDP_MEM_DOUBLE,
                SDP_MEM_FLOAT,
                SDP_MEM_DOUBLE,
                SDP_MEM_CPU,
                &status
        );
        assert(status == SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_uniform("Double precision",
                false,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_CPU,
                &status
        );
        assert(status == SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_briggs("Double precision",
                false,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_CPU,
                &status
        );
        assert(status == SDP_SUCCESS);
    }

    // //Check for unhappy paths.

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_uniform("Read-only output",
                true,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_CPU,
                &status
        );
        assert(status != SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_briggs("Read-only output",
                true,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_CPU,
                &status
        );
        assert(status != SDP_SUCCESS);
    }

    // {
    //     sdp_Error status = SDP_SUCCESS;
    //     run_and_check_uniform("Type mismatch",  false,
    //         SDP_MEM_FLOAT, SDP_MEM_DOUBLE, SDP_MEM_FLOAT, SDP_MEM_CPU, &status
    //     );
    //     assert(status == SDP_ERR_DATA_TYPE);
    // }

//     {
//         sdp_Error status = SDP_SUCCESS;
//         run_and_check_briggs("Type mismatch",  false,
//             SDP_MEM_FLOAT, SDP_MEM_DOUBLE, SDP_MEM_FLOAT, SDP_MEM_CPU, &status
//         );
//         assert(status == SDP_ERR_DATA_TYPE);
//     }

    // {
    //     sdp_Error status = SDP_SUCCESS;
    //     run_and_check_uniform("Unsupported data type", false,
    //         SDP_MEM_INT, SDP_MEM_INT, SDP_MEM_INT, SDP_MEM_CPU, &status
    //     );
    //     assert(status == SDP_ERR_DATA_TYPE);
    // }

#ifdef SDP_HAVE_CUDA

    // check for happy paths.

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_uniform("Single precision",
                false,
                SDP_MEM_DOUBLE,
                SDP_MEM_FLOAT,
                SDP_MEM_DOUBLE,
                SDP_MEM_GPU,
                &status
        );
        assert(status == SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_briggs("Single precision",
                false,
                SDP_MEM_DOUBLE,
                SDP_MEM_FLOAT,
                SDP_MEM_DOUBLE,
                SDP_MEM_GPU,
                &status
        );
        assert(status == SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_uniform("Double precision",
                false,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_GPU,
                &status
        );
        assert(status == SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check_briggs("Double precision",
                false,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_GPU,
                &status
        );
        assert(status == SDP_SUCCESS);
    }

#endif
    return 0;
}
