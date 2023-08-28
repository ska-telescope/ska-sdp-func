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

#include "ska-sdp-func/station_beam/sdp_station.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"

#define C_0 299792458.0

using std::complex;


template<typename FP, int NUM_POL>
void check_results(
        const char* test_name,
        const FP wavenumber,
        const int num_in,
        const complex<FP>* const __restrict__ weights_in,
        const FP* const __restrict__ x_in,
        const FP* const __restrict__ y_in,
        const FP* const __restrict__ z_in,
        const int idx_offset_out,
        const int num_out,
        const FP* const __restrict__ x_out,
        const FP* const __restrict__ y_out,
        const FP* const __restrict__ z_out,
        const int* const __restrict__ data_idx,
        const complex<FP>* const __restrict__ data,
        const int idx_offset_output,
        const complex<FP>* __restrict__ output,
        const FP norm_factor,
        const int eval_x,
        const int eval_y,
        const sdp_Error* status
)
{
    if (*status)
    {
        SDP_LOG_ERROR("%s: Test failed (error signalled)", test_name);
        return;
    }
    for (int i_out = 0; i_out < num_out; i_out++)
    {
        complex<FP> out[NUM_POL];
        for (int k = 0; k < NUM_POL; ++k)
        {
            out[k] = complex<FP>(0, 0);
        }
        const FP xo = wavenumber * x_out[i_out + idx_offset_out];
        const FP yo = wavenumber * y_out[i_out + idx_offset_out];
        const FP zo = z_out ?
                    wavenumber * z_out[i_out + idx_offset_out] : (FP) 0;
        if (data)
        {
            for (int i = 0; i < num_in; ++i)
            {
                const double phase = xo * x_in[i] + yo * y_in[i] + zo * z_in[i];
                const complex<FP> weighted_phasor =
                        complex<FP>(cos(phase), sin(phase)) * weights_in[i];
                const int i_in = NUM_POL * (
                    (data_idx ? data_idx[i] : i) * num_out + i_out
                );
                if (NUM_POL == 1)
                {
                    out[0] += weighted_phasor * data[i_in];
                }
                else if (NUM_POL == 4)
                {
                    if (eval_x)
                    {
                        out[0] += weighted_phasor * data[i_in + 0];
                        out[1] += weighted_phasor * data[i_in + 1];
                    }
                    if (eval_y)
                    {
                        out[2] += weighted_phasor * data[i_in + 2];
                        out[3] += weighted_phasor * data[i_in + 3];
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < num_in; ++i)
            {
                const double phase = xo * x_in[i] + yo * y_in[i] + zo * z_in[i];
                const complex<FP> weighted_phasor =
                        complex<FP>(cos(phase), sin(phase)) * weights_in[i];
                if (NUM_POL == 1)
                {
                    out[0] += weighted_phasor;
                }
                else if (NUM_POL == 4)
                {
                    if (eval_x)
                    {
                        out[0] += weighted_phasor;
                        out[1] += weighted_phasor;
                    }
                    if (eval_y)
                    {
                        out[2] += weighted_phasor;
                        out[3] += weighted_phasor;
                    }
                }
            }
        }

        // Check output data.
        const int i_out_offset_scaled = NUM_POL * (i_out + idx_offset_output);
        for (int k = 0; k < NUM_POL; ++k)
        {
            out[k] = out[k] * norm_factor - output[i_out_offset_scaled + k];
        }
        if (NUM_POL == 1)
        {
            assert(fabs(real(out[0])) < 1e-5);
            assert(fabs(imag(out[0])) < 1e-5);
        }
        else if (NUM_POL == 4)
        {
            if (eval_x)
            {
                assert(fabs(real(out[0])) < 1e-5);
                assert(fabs(imag(out[0])) < 1e-5);
                assert(fabs(real(out[1])) < 1e-5);
                assert(fabs(imag(out[1])) < 1e-5);
            }
            if (eval_y)
            {
                assert(fabs(real(out[2])) < 1e-5);
                assert(fabs(imag(out[2])) < 1e-5);
                assert(fabs(real(out[3])) < 1e-5);
                assert(fabs(imag(out[3])) < 1e-5);
            }
        }
    }
    SDP_LOG_INFO("%s: Test passed", test_name);
}


template<int NUM_POL>
static void run_and_check(
        const char* test_name,
        bool expect_pass,
        sdp_MemType data_type,
        sdp_MemLocation in_location,
        sdp_MemLocation out_location,
        sdp_Error* status
)
{
    // Generate test data and copy to specified location.
    const int64_t num_ant = 256;
    const int64_t num_dir = 64;
    const double freq_hz = 100e6;
    const double wavenumber = 2.0 * M_PI * C_0 / freq_hz;
    const double norm_factor = 1.0 / num_ant;

    // Coordinates.
    sdp_Mem* i_c[] = {0, 0, 0};
    sdp_Mem* o_c[] = {0, 0, 0};
    sdp_Mem* in_i_c[] = {0, 0, 0};
    sdp_Mem *in_o_c[] = {0, 0, 0};
    for (int i = 0; i < 3; ++i)
    {
        i_c[i] = sdp_mem_create(data_type, SDP_MEM_CPU, 1, &num_ant, status);
        o_c[i] = sdp_mem_create(data_type, SDP_MEM_CPU, 1, &num_dir, status);
        sdp_mem_random_fill(i_c[i], status);
        sdp_mem_random_fill(o_c[i], status);
        in_i_c[i] = sdp_mem_create_copy(i_c[i], in_location, status);
        in_o_c[i] = sdp_mem_create_copy(o_c[i], in_location, status);
    }

    // Element weights.
    const sdp_MemType cplx = (sdp_MemType)(data_type | SDP_MEM_COMPLEX);
    sdp_Mem* weights = sdp_mem_create(cplx, SDP_MEM_CPU, 1, &num_ant, status);
    sdp_mem_random_fill(weights, status);
    sdp_Mem* in_weights = sdp_mem_create_copy(weights, in_location, status);

    // Element beam indices.
    sdp_Mem* el_beam_index = sdp_mem_create(
            SDP_MEM_INT, SDP_MEM_CPU, 1, &num_ant, status
    );
    sdp_mem_clear_contents(el_beam_index, status);
    sdp_Mem* in_el_beam_index = sdp_mem_create_copy(
            el_beam_index, in_location, status
    );

    // Element beam data.
    int64_t data_shape[] = {num_dir, 1, 1};
    if (NUM_POL != 1) data_shape[1] = data_shape[2] = 2;
    sdp_Mem* el_beam = sdp_mem_create(cplx, SDP_MEM_CPU, 3, data_shape, status);
    sdp_mem_random_fill(el_beam, status);
    sdp_Mem* in_el_beam = sdp_mem_create_copy(el_beam, in_location, status);
    const int64_t empty_shape[] = {0};
    sdp_Mem* el_beam_empty = sdp_mem_create(
            cplx, in_location, 1, empty_shape, status
    );

    // Call the function to test.
    SDP_LOG_INFO("Running test: %s", test_name);
    for (int i = 0; i < 2; ++i)
    {
        sdp_Mem* st_beam = sdp_mem_create(
                cplx, out_location, 3, data_shape, status
        );
        sdp_station_beam_aperture_array(wavenumber, in_weights,
                in_i_c[0], in_i_c[1], in_i_c[2],
                0, num_dir, in_o_c[0], in_o_c[1], in_o_c[2],
                in_el_beam_index, i == 0 ? el_beam_empty : in_el_beam,
                0, st_beam, 1, 1, 1, status
        );

        // Copy the output for checking.
        sdp_Mem* out_st_beam = sdp_mem_create_copy(
                st_beam, SDP_MEM_CPU, status
        );

        // Check output only if test is expected to pass.
        if (expect_pass)
        {
            if (data_type == SDP_MEM_DOUBLE)
            {
                check_results<double, NUM_POL>(
                        test_name,
                        wavenumber,
                        num_ant,
                        (complex<double>*)sdp_mem_data(weights),
                        (double*)sdp_mem_data(i_c[0]),
                        (double*)sdp_mem_data(i_c[1]),
                        (double*)sdp_mem_data(i_c[2]),
                        0,
                        num_dir,
                        (double*)sdp_mem_data(o_c[0]),
                        (double*)sdp_mem_data(o_c[1]),
                        (double*)sdp_mem_data(o_c[2]),
                        (int*)sdp_mem_data(el_beam_index),
                        i == 0 ? 0 : (complex<double>*)sdp_mem_data(el_beam),
                        0,
                        (complex<double>*)sdp_mem_data(out_st_beam),
                        norm_factor,
                        1,
                        1,
                        status
                );
            }
            else
            {
                check_results<float, NUM_POL>(
                        test_name,
                        wavenumber,
                        num_ant,
                        (complex<float>*)sdp_mem_data(weights),
                        (float*)sdp_mem_data(i_c[0]),
                        (float*)sdp_mem_data(i_c[1]),
                        (float*)sdp_mem_data(i_c[2]),
                        0,
                        num_dir,
                        (float*)sdp_mem_data(o_c[0]),
                        (float*)sdp_mem_data(o_c[1]),
                        (float*)sdp_mem_data(o_c[2]),
                        (int*)sdp_mem_data(el_beam_index),
                        i == 0 ? 0 : (complex<float>*)sdp_mem_data(el_beam),
                        0,
                        (complex<float>*)sdp_mem_data(out_st_beam),
                        norm_factor,
                        1,
                        1,
                        status
                );
            }
        }
        sdp_mem_ref_dec(st_beam);
        sdp_mem_ref_dec(out_st_beam);
    }
    for (int i = 0; i < 3; ++i)
    {
        sdp_mem_ref_dec(i_c[i]);
        sdp_mem_ref_dec(o_c[i]);
        sdp_mem_ref_dec(in_i_c[i]);
        sdp_mem_ref_dec(in_o_c[i]);
    }
    sdp_mem_ref_dec(weights);
    sdp_mem_ref_dec(in_weights);
    sdp_mem_ref_dec(el_beam_index);
    sdp_mem_ref_dec(in_el_beam_index);
    sdp_mem_ref_dec(el_beam);
    sdp_mem_ref_dec(in_el_beam);
    sdp_mem_ref_dec(el_beam_empty);
}


int main()
{
    // Happy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check<1>("CPU, double precision, 1 pol", true,
                SDP_MEM_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check<1>("CPU, single precision, 1 pol", true,
                SDP_MEM_FLOAT,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check<4>("CPU, double precision, 4 pols", true,
                SDP_MEM_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check<4>("CPU, single precision, 4 pols", true,
                SDP_MEM_FLOAT,
                SDP_MEM_CPU, SDP_MEM_CPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
#ifdef SDP_HAVE_CUDA
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check<1>("GPU, double precision, 1 pol", true,
                SDP_MEM_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check<1>("GPU, single precision, 1 pol", true,
                SDP_MEM_FLOAT,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check<4>("GPU, double precision, 4 pols", true,
                SDP_MEM_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check<4>("GPU, single precision, 4 pols", true,
                SDP_MEM_FLOAT,
                SDP_MEM_GPU, SDP_MEM_GPU, &status
        );
        assert(status == SDP_SUCCESS);
    }
#endif

    return 0;
}
