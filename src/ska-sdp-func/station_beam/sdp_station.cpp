/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>
#include <cstdlib>

#include "ska-sdp-func/station_beam/sdp_station.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"

using std::complex;

#if 0

// These are not currently used.
// Still wondering whether or not they're needed.

struct sdp_Station
{
    sdp_Mem* coords[3];
};


sdp_Station* sdp_station_create(
        sdp_MemType type,
        sdp_MemLocation location,
        int num_elements,
        sdp_Error* status
)
{
    if (*status) return NULL;
    sdp_Station* model = (sdp_Station*) calloc(1, sizeof(sdp_Station));
    const int64_t shape[] = {(int64_t) num_elements};
    for (int i = 0 ; i < 3; ++i)
    {
        model->coords[i] = sdp_mem_create(type, location, 1, shape, status);
    }
    return model;
}


void sdp_station_free(sdp_Station* model)
{
    if (!model) return;
    for (int i = 0 ; i < 3; ++i)
    {
        sdp_mem_free(model->coords[i]);
    }
}

#endif


template<typename FP, int NUM_POL>
void sdp_station_beam_dft(
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
        complex<FP>* __restrict__ output,
        const FP norm_factor,
        const int eval_x,
        const int eval_y
)
{
    #pragma omp parallel for
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
        const int i_out_offset_scaled = NUM_POL * (i_out + idx_offset_output);
        if (NUM_POL == 1)
        {
            output[i_out_offset_scaled] = out[0] * norm_factor;
        }
        else if (NUM_POL == 4)
        {
            if (eval_x)
            {
                output[i_out_offset_scaled + 0] = out[0] * norm_factor;
                output[i_out_offset_scaled + 1] = out[1] * norm_factor;
            }
            if (eval_y)
            {
                output[i_out_offset_scaled + 2] = out[2] * norm_factor;
                output[i_out_offset_scaled + 3] = out[3] * norm_factor;
            }
        }
    }
}


void sdp_station_beam_aperture_array(
        const double wavenumber,
        const sdp_Mem* element_weights,
        const sdp_Mem* element_x,
        const sdp_Mem* element_y,
        const sdp_Mem* element_z,
        const int index_offset_points,
        const int num_points,
        const sdp_Mem* point_x,
        const sdp_Mem* point_y,
        const sdp_Mem* point_z,
        const sdp_Mem* element_beam_index,
        const sdp_Mem* element_beam,
        const int index_offset_station_beam,
        sdp_Mem* station_beam,
        const int normalise,
        const int eval_x,
        const int eval_y,
        sdp_Error* status
)
{
    if (*status) return;
    const sdp_MemLocation location = sdp_mem_location(station_beam);
    const sdp_MemType precision = sdp_mem_type(element_x);
    const int num_elements = sdp_mem_num_elements(element_x);
    const float wavenumber_f = (float) wavenumber;
    const double norm_factor = normalise ? 1.0 / num_elements : 1.0;
    const float norm_factor_f = (float) norm_factor;
    if (!sdp_mem_is_complex(element_weights))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Input element_weights array must be complex");
        return;
    }
    if (sdp_mem_data_const(element_beam) &&
            !sdp_mem_is_complex(element_beam))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Input data array must be complex");
        return;
    }
    if (sdp_mem_data_const(element_beam_index) &&
            sdp_mem_type(element_beam_index) != SDP_MEM_INT)
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Input data index array must be integer");
        return;
    }
    if (!sdp_mem_is_complex(station_beam))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Output beam array must be complex");
        return;
    }
    const int is_complex4 = sdp_mem_is_complex4(station_beam);
    if (location == SDP_MEM_CPU)
    {
        if (precision == SDP_MEM_FLOAT && !is_complex4)
        {
            sdp_station_beam_dft<float, 1>(
                    wavenumber_f,
                    num_elements,
                    (const complex<float>*)sdp_mem_data_const(element_weights),
                    (const float*)sdp_mem_data_const(element_x),
                    (const float*)sdp_mem_data_const(element_y),
                    (const float*)sdp_mem_data_const(element_z),
                    index_offset_points,
                    num_points,
                    (const float*)sdp_mem_data_const(point_x),
                    (const float*)sdp_mem_data_const(point_y),
                    (const float*)sdp_mem_data_const(point_z),
                    (const int*)sdp_mem_data_const(element_beam_index),
                    (const complex<float>*)sdp_mem_data_const(element_beam),
                    index_offset_station_beam,
                    (complex<float>*)sdp_mem_data(station_beam),
                    norm_factor_f,
                    eval_x,
                    eval_y
            );
        }
        else if (precision == SDP_MEM_DOUBLE && !is_complex4)
        {
            sdp_station_beam_dft<double, 1>(
                    wavenumber,
                    num_elements,
                    (const complex<double>*)sdp_mem_data_const(element_weights),
                    (const double*)sdp_mem_data_const(element_x),
                    (const double*)sdp_mem_data_const(element_y),
                    (const double*)sdp_mem_data_const(element_z),
                    index_offset_points,
                    num_points,
                    (const double*)sdp_mem_data_const(point_x),
                    (const double*)sdp_mem_data_const(point_y),
                    (const double*)sdp_mem_data_const(point_z),
                    (const int*)sdp_mem_data_const(element_beam_index),
                    (const complex<double>*)sdp_mem_data_const(element_beam),
                    index_offset_station_beam,
                    (complex<double>*)sdp_mem_data(station_beam),
                    norm_factor,
                    eval_x,
                    eval_y
            );
        }
        else if (precision == SDP_MEM_FLOAT && is_complex4)
        {
            sdp_station_beam_dft<float, 4>(
                    wavenumber_f,
                    num_elements,
                    (const complex<float>*)sdp_mem_data_const(element_weights),
                    (const float*)sdp_mem_data_const(element_x),
                    (const float*)sdp_mem_data_const(element_y),
                    (const float*)sdp_mem_data_const(element_z),
                    index_offset_points,
                    num_points,
                    (const float*)sdp_mem_data_const(point_x),
                    (const float*)sdp_mem_data_const(point_y),
                    (const float*)sdp_mem_data_const(point_z),
                    (const int*)sdp_mem_data_const(element_beam_index),
                    (const complex<float>*)sdp_mem_data_const(element_beam),
                    index_offset_station_beam,
                    (complex<float>*)sdp_mem_data(station_beam),
                    norm_factor_f,
                    eval_x,
                    eval_y
            );
        }
        else if (precision == SDP_MEM_DOUBLE && is_complex4)
        {
            sdp_station_beam_dft<double, 4>(
                    wavenumber,
                    num_elements,
                    (const complex<double>*)sdp_mem_data_const(element_weights),
                    (const double*)sdp_mem_data_const(element_x),
                    (const double*)sdp_mem_data_const(element_y),
                    (const double*)sdp_mem_data_const(element_z),
                    index_offset_points,
                    num_points,
                    (const double*)sdp_mem_data_const(point_x),
                    (const double*)sdp_mem_data_const(point_y),
                    (const double*)sdp_mem_data_const(point_z),
                    (const int*)sdp_mem_data_const(element_beam_index),
                    (const complex<double>*)sdp_mem_data_const(element_beam),
                    index_offset_station_beam,
                    (complex<double>*)sdp_mem_data(station_beam),
                    norm_factor,
                    eval_x,
                    eval_y
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
            return;
        }
    }
    else if (location == SDP_MEM_GPU)
    {
        int is_dbl = 0, max_in_chunk = 0;
        const uint64_t num_threads[] = {256, 1, 1};
        const uint64_t num_blocks[] = {
            (num_points + num_threads[0] - 1) / num_threads[0], 1, 1
        };
        const char* kernel_name = 0;
        if (precision == SDP_MEM_FLOAT && !is_complex4)
        {
            is_dbl = 0;
            max_in_chunk = 512;
            kernel_name = "sdp_station_beam_dft<float, float2, 1>";
        }
        else if (precision == SDP_MEM_DOUBLE && !is_complex4)
        {
            is_dbl = 1;
            max_in_chunk = 352;
            kernel_name = "sdp_station_beam_dft<double, double2, 1>";
        }
        else if (precision == SDP_MEM_FLOAT && is_complex4)
        {
            is_dbl = 0;
            max_in_chunk = 512;
            kernel_name = "sdp_station_beam_dft<float, float2, 4>";
        }
        else if (precision == SDP_MEM_DOUBLE && is_complex4)
        {
            is_dbl = 1;
            max_in_chunk = 352;
            kernel_name = "sdp_station_beam_dft<double, double2, 4>";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
            return;
        }
        const uint64_t shared_mem_bytes = max_in_chunk * (
            5 * sdp_mem_type_size(precision) + sizeof(int)
        );
        const void* args[] = {
            is_dbl ? (const void*)&wavenumber : (const void*)&wavenumber_f,
            &num_elements,
            sdp_mem_gpu_buffer_const(element_weights, status),
            sdp_mem_gpu_buffer_const(element_x, status),
            sdp_mem_gpu_buffer_const(element_y, status),
            sdp_mem_gpu_buffer_const(element_z, status),
            &index_offset_points,
            &num_points,
            sdp_mem_gpu_buffer_const(point_x, status),
            sdp_mem_gpu_buffer_const(point_y, status),
            sdp_mem_gpu_buffer_const(point_z, status),
            sdp_mem_gpu_buffer_const(element_beam_index, status),
            sdp_mem_gpu_buffer_const(element_beam, status),
            &index_offset_station_beam,
            sdp_mem_gpu_buffer(station_beam, status),
            is_dbl ? (const void*)&norm_factor : (const void*)&norm_factor_f,
            &eval_x,
            &eval_y,
            &max_in_chunk
        };
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, shared_mem_bytes, 0, args, status
        );
    }
}
