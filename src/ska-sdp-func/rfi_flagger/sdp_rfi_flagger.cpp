/* See the LICENSE file at the top-level directory of this distribution. */

#include <complex>
#include <cstring>

#include "ska-sdp-func/rfi_flagger/sdp_rfi_flagger.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"


static void check_params(
        const sdp_Mem* vis,
        const sdp_Mem* thresholds,
        sdp_Mem* flags,
        sdp_Error* status
)
{
    if (*status) return;
    if (sdp_mem_is_read_only(flags))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Output flags must be writable.");
        return;
    }
    if (!sdp_mem_is_c_contiguous(vis) ||
            !sdp_mem_is_c_contiguous(thresholds) ||
            !sdp_mem_is_c_contiguous(flags))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("All arrays must be C contiguous.");
        return;
    }
    if (sdp_mem_num_dims(vis) != 4 || sdp_mem_num_dims(flags) != 4)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Visibility and flags arrays must be 4D.");
        return;
    }
    if (!sdp_mem_is_complex(vis))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Visibilities must be complex.");
        return;
    }
    if (sdp_mem_type(flags) != SDP_MEM_INT)
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Flags must be integers.");
        return;
    }
    if (sdp_mem_location(vis) != sdp_mem_location(thresholds) ||
            sdp_mem_location(vis) != sdp_mem_location(flags))
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("All arrays must be in the same memory location.");
        return;
    }
}


template<typename FP>
static void sum_threshold_on_block(
        const FP* thresholds,
        const uint64_t seqlen,
        const int* sequence_lengths,
        FP* block,
        const uint64_t num_channels,
        const uint64_t num_timesamples,
        int* flags_on_block
)
{
    for (uint64_t k = 0; k < seqlen; k++)
    {
        const uint64_t current_seqlen = sequence_lengths[k];
        const FP current_threshold = thresholds[k] * current_seqlen;
        for (uint64_t j = 0; j < num_channels; j++)
        {
            for (uint64_t i = 0; i < num_timesamples - current_seqlen; i++)
            {
                FP sum = (FP) 0;
                for (uint64_t m = 0; m < current_seqlen; m++)
                {
                    if (flags_on_block[(i + m) * num_channels + j] == 1)
                    {
                        block[(i + m) * num_channels + j] = thresholds[k];
                    }
                    sum = sum + block[(i + m) * num_channels + j];
                }
                if (sum > current_threshold)
                {
                    for (uint64_t m = 0; m < current_seqlen; m++)
                    {
                        flags_on_block[(i + m) * num_channels + j] = 1;
                    }
                }
            }
        }
    }
}


template<typename FP>
static void sum_threshold_rfi_flagger(
        int* flags,
        const std::complex<FP>* visibilities,
        const FP* thresholds,
        const uint64_t num_timesamples,
        const uint64_t num_baselines,
        const uint64_t num_channels,
        const uint64_t num_pols,
        const uint64_t max_sequence_length
)
{
    uint64_t timesample_block = num_channels * num_pols * num_baselines;
    uint64_t baseline_block = num_channels * num_pols;
    uint64_t channel_block = num_pols;
    int num_sequence_elements = (int) (log(max_sequence_length) / log(2)) + 1;
    int* sequence_lengths = new int[num_sequence_elements];
    for (int f = 0; f < num_sequence_elements; f++)
    {
        sequence_lengths[f] = (1 << f);
    }

    FP* block = new FP[num_timesamples * num_channels];
    int* flags_on_the_block = new int[num_timesamples * num_channels];
    for (uint64_t b = 0; b < num_baselines; b++)
    {
        for (uint64_t p = 0; p < num_pols; p++)
        {
            memset(flags_on_the_block, 0,
                    num_timesamples * num_channels * sizeof(int)
            );
            for (uint64_t t = 0; t < num_timesamples; t++)
            {
                for (uint64_t c = 0; c < num_channels; c++)
                {
                    uint64_t pos = t * timesample_block +
                            b * baseline_block + c * channel_block + p;
                    std::complex<FP> temp = visibilities[pos];
                    block[t * num_channels + c] = std::abs(temp);
                }
            }

            sum_threshold_on_block(thresholds, num_sequence_elements,
                    sequence_lengths, block, num_channels, num_timesamples,
                    flags_on_the_block
            );

            for (uint64_t t = 0; t < num_timesamples; t++)
            {
                for (uint64_t c = 0; c < num_channels; c++)
                {
                    uint64_t pos = t * timesample_block +
                            b * baseline_block + c * channel_block + p;
                    flags[pos] = flags_on_the_block[t * num_channels + c];
                }
            }
        }
    }

    delete[] block;
    delete[] flags_on_the_block;
    delete[] sequence_lengths;
}


void sdp_sum_threshold_rfi_flagger(
        const sdp_Mem* vis,
        const sdp_Mem* thresholds,
        sdp_Mem* flags,
        const int64_t max_sequence_length,
        sdp_Error* status
)
{
    check_params(vis, thresholds, flags, status);
    if (*status) return;

    const uint64_t num_timesamples = (uint64_t) sdp_mem_shape_dim(vis, 0);
    const uint64_t num_baselines = (uint64_t) sdp_mem_shape_dim(vis, 1);
    const uint64_t num_channels = (uint64_t) sdp_mem_shape_dim(vis, 2);
    const uint64_t num_pols = (uint64_t) sdp_mem_shape_dim(vis, 3);

    if (sdp_mem_location(vis) == SDP_MEM_CPU)
    {
        if (sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT &&
                sdp_mem_type(thresholds) == SDP_MEM_FLOAT &&
                sdp_mem_type(flags) == SDP_MEM_INT)
        {
            sum_threshold_rfi_flagger(
                    (int*) sdp_mem_data(flags),
                    (const std::complex<float>*) sdp_mem_data_const(vis),
                    (const float*) sdp_mem_data_const(thresholds),
                    num_timesamples,
                    num_baselines,
                    num_channels,
                    num_pols,
                    max_sequence_length
            );
        }
        else if (sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(thresholds) == SDP_MEM_DOUBLE &&
                sdp_mem_type(flags) == SDP_MEM_INT)
        {
            sum_threshold_rfi_flagger(
                    (int*) sdp_mem_data(flags),
                    (const std::complex<double>*) sdp_mem_data_const(vis),
                    (const double*) sdp_mem_data_const(thresholds),
                    num_timesamples,
                    num_baselines,
                    num_channels,
                    num_pols,
                    max_sequence_length
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s): visibilities and "
                    "thresholds arrays must have the same precision."
            );
        }
    }
    else
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Unsupported memory location for visibility data.");
        return;
    }
}
