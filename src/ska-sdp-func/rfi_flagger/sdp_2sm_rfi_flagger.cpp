/* See the LICENSE file at the top-level directory of this distribution. */

#include <complex>
#include <cstring>

#include "ska-sdp-func/rfi_flagger/sdp_2sm_rfi_flagger.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"

static void check_params(
        const sdp_Mem* vis,
        const sdp_Mem* thresholds,
        sdp_Mem* flags,
        sdp_Error* status)
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
static void twosm_rfi_flagger(
        int* flags,
        const std::complex<FP>* visibilities,
        const FP* thresholds,
        const uint64_t num_timesamples,
        const uint64_t num_baselines,
        const uint64_t num_channels,
        const uint64_t num_pols)
{
    uint64_t timesample_block = num_channels * num_pols * num_baselines;
    uint64_t baseline_block = num_channels * num_pols;
    uint64_t channel_block = num_pols;


    float dv_between_minustwo_and_one = 0;
    float dv_between_minusthree_and_two = 0;
    float second_dv = 0;
    float predicted_value = 0;

    for (uint16_t b = 0; b < num_baselines; b++){
        for (uint16_t c = 0; c < num_channels; c++){
            for (uint16_t t = 3; t < num_timesamples; t++){
                uint16_t pos_current = t * timesample_block + b * baseline_block + c * channel_block;
                uint16_t pos_minusone = (t - 1) * timesample_block + b * baseline_block + c * channel_block;
                uint16_t pos_minustwo = (t - 2) * timesample_block + b * baseline_block + c * channel_block;
                uint16_t pos_minusthree = (t - 3) * timesample_block + b * baseline_block + c * channel_block;
                dv_between_minustwo_and_one = std::abs(visibilities[pos_minusone]) - std::abs(visibilities[pos_minustwo]);
                dv_between_minusthree_and_two = std::abs(visibilities[pos_minustwo]) - std::abs(visibilities[pos_minusthree]);
                second_dv = dv_between_minustwo_and_one - dv_between_minusthree_and_two;
                
            }
        }
    }

}


void sdp_twosm_rfi_flagger(
        const sdp_Mem* vis,
        const sdp_Mem* thresholds,
        sdp_Mem* flags,
        sdp_Error* status)
{
    check_params(vis, thresholds, flags, status);
    if (*status) return;

    const uint64_t num_timesamples   = (uint64_t) sdp_mem_shape_dim(vis, 0);
    const uint64_t num_baselines     = (uint64_t) sdp_mem_shape_dim(vis, 1);
    const uint64_t num_channels      = (uint64_t) sdp_mem_shape_dim(vis, 2);
    const uint64_t num_pols          = (uint64_t) sdp_mem_shape_dim(vis, 3);

    if (sdp_mem_location(vis) == SDP_MEM_CPU)
    {
        if (sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT &&
            sdp_mem_type(thresholds) == SDP_MEM_FLOAT &&
            sdp_mem_type(flags) == SDP_MEM_INT)
        {
            twosm_rfi_flagger(
                    (int*) sdp_mem_data(flags),
                    (const std::complex<float>*) sdp_mem_data_const(vis),
                    (const float*) sdp_mem_data_const(thresholds),
                    num_timesamples,
                    num_baselines,
                    num_channels,
                    num_pols
            );
        }
        else if (sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE &&
                 sdp_mem_type(thresholds) == SDP_MEM_DOUBLE &&
                 sdp_mem_type(flags) == SDP_MEM_INT)
        {
            twosm_rfi_flagger(
                    (int*) sdp_mem_data(flags),
                    (const std::complex<double>*) sdp_mem_data_const(vis),
                    (const double*) sdp_mem_data_const(thresholds),
                    num_timesamples,
                    num_baselines,
                    num_channels,
                    num_pols
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s): visibilities and "
                          "thresholds arrays must have the same precision.");
        }
    }
    else
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Unsupported memory location for visibility data.");
        return;
    }
}
