/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>
#include <cstdlib>

#include "ska-sdp-func/station_beam/sdp_station.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"

using std::complex;

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


template<typename FP>
void sdp_station_beam_dft_scalar(
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
        const int idx_offset_data,
        complex<FP>* __restrict__ output,
        const FP norm_factor
)
{
    #pragma omp parallel for
    for (int i_out = 0; i_out < num_out; i_out++)
    {
        complex<FP> out(0, 0);
        const FP xo = wavenumber * x_out[i_out + idx_offset_out];
        const FP yo = wavenumber * y_out[i_out + idx_offset_out];
        const FP zo = wavenumber * z_out[i_out + idx_offset_out];
        if (data)
        {
            for (int i = 0; i < num_in; ++i)
            {
                const double phase = xo * x_in[i] + yo * y_in[i] + zo * z_in[i];
                const double cos_phase = cos(phase), sin_phase = sin(phase);
                const complex<FP> phasor(cos_phase, sin_phase);
                const int i_in = (data_idx ? data_idx[i] : i) * num_out + i_out;
                out += phasor * weights_in[i] * data[i_in];
            }
        }
        else
        {
            for (int i = 0; i < num_in; ++i)
            {
                const double phase = xo * x_in[i] + yo * y_in[i] + zo * z_in[i];
                const double cos_phase = cos(phase), sin_phase = sin(phase);
                const complex<FP> phasor(cos_phase, sin_phase);
                out += phasor * weights_in[i];
            }
        }
        output[i_out + idx_offset_data] = out * norm_factor;
    }
}

void sdp_station_beam_array_factor(
        const double wavenumber,
        const sdp_Mem* element_weights,
        const sdp_Mem* element_x,
        const sdp_Mem* element_y,
        const sdp_Mem* element_z,
        int index_offset_points,
        int num_points,
        const sdp_Mem* point_x,
        const sdp_Mem* point_y,
        const sdp_Mem* point_z,
        const sdp_Mem* data_index,
        const sdp_Mem* data,
        int index_offset_beam,
        sdp_Mem* beam,
        int normalise,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_MemLocation location = sdp_mem_location(beam);
    sdp_MemType precision = sdp_mem_type(element_x);
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
    if (sdp_mem_data_const(data) && !sdp_mem_is_complex(data))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Input data array must be complex");
        return;
    }
    if (sdp_mem_data_const(data_index) && sdp_mem_type(data) != SDP_MEM_INT)
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Input data index array must be integer");
        return;
    }
    if (!sdp_mem_is_complex(beam))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Output beam array must be complex");
        return;
    }
    if (location == SDP_MEM_CPU)
    {
        if (precision == SDP_MEM_FLOAT)
        {
            sdp_station_beam_dft_scalar<float>(
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
                (const int*)sdp_mem_data_const(data_index),
                (const complex<float>*)sdp_mem_data_const(data),
                index_offset_beam,
                (complex<float>*)sdp_mem_data(beam),
                norm_factor_f
            );
        }
        else if (precision == SDP_MEM_DOUBLE)
        {
            sdp_station_beam_dft_scalar<double>(
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
                (const int*)sdp_mem_data_const(data_index),
                (const complex<double>*)sdp_mem_data_const(data),
                index_offset_beam,
                (complex<double>*)sdp_mem_data(beam),
                norm_factor
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
        if (precision == SDP_MEM_FLOAT)
        {
            is_dbl = 0;
            max_in_chunk = 512;
            kernel_name = "sdp_station_beam_dft_scalar<float, float2>";
        }
        else if (precision == SDP_MEM_DOUBLE)
        {
            is_dbl = 1;
            max_in_chunk = 352;
            kernel_name = "sdp_station_beam_dft_scalar<double, double2>";
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
            sdp_mem_gpu_buffer_const(data_index, status),
            sdp_mem_gpu_buffer_const(data, status),
            &index_offset_beam,
            sdp_mem_gpu_buffer(beam, status),
            is_dbl ? (const void*)&norm_factor : (const void*)&norm_factor_f,
            &max_in_chunk
        };
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, shared_mem_bytes, 0, args, status
        );
    }
}
