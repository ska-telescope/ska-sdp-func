/* See the LICENSE file at the top-level directory of this distribution. */

#include <complex>
#include "src/ska-sdp-func/twosm_rfi_flagger/sdp_twosm_rfi_flagger.h"#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include <iostream>

static void check_params(
        const sdp_Mem* vis,
        const sdp_Mem* thresholds,
        const sdp_Mem* antennas,
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
        !sdp_mem_is_c_contiguous(antennas)||
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
        sdp_mem_location(vis) != sdp_mem_location(flags) ||
        sdp_mem_location(vis) != sdp_mem_location(antennas))
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
        const int* antennas,
        const uint64_t num_timesamples,
        const uint64_t num_antennas,
        const uint64_t num_channels,
        const uint64_t num_pols)
{
    uint64_t num_baselines = num_antennas * (num_antennas + 1)/2;
    uint64_t timesample_block = num_channels * num_pols * num_baselines;
    uint64_t baseline_block = num_channels * num_pols;
    uint64_t channel_block = num_pols;
    uint64_t num_elements = num_timesamples * num_baselines * num_channels * num_pols;

    double dv_between_cur_one = 0;
    double dv_ratio = 0;
    double tol_margin_2sm_time = thresholds[0];
    double tol_margin_2sm_freq = thresholds[1];

    for (uint64_t a = 0; a < num_antennas; a++){
        uint64_t b = antennas[a];
        for (uint64_t c = 0; c < num_channels; c++){
            for (uint64_t t = 1; t < num_timesamples; t++) {
                uint64_t pos_current = t * timesample_block + b * baseline_block + c * channel_block;
                uint64_t pos_minusone = (t - 1) * timesample_block + b * baseline_block + c * channel_block;
                double vis0 = std::abs(visibilities[pos_current]);
                double vis1 = std::abs(visibilities[pos_minusone]);
                dv_between_cur_one = vis0 - vis1;
                dv_ratio =  dv_between_cur_one/vis1;
                if (flags[pos_current] != 1){
                    bool cnd0 = dv_ratio > tol_margin_2sm_time;
                    bool cnd1 = dv_ratio < (-1) * tol_margin_2sm_time;
                    bool cnd2 = dv_ratio > (-1) * tol_margin_2sm_time;
                    bool cnd3 = dv_ratio < tol_margin_2sm_time;
                    bool cnd4 = dv_ratio < 0;
                    bool cnd5 = dv_ratio > 0;
                    bool cnd6 = flags[pos_minusone] == 1;
                    bool cnd7 = flags[pos_minusone] == 0;

                    if (cnd0 || (((cnd2 && cnd4) || (cnd3 && cnd5)) && cnd6)){
                        for (uint64_t b = 0; b < num_baselines; b++){
                            for (uint64_t p = 0; p < num_pols; p++){
                                uint64_t pos = t * timesample_block + b * baseline_block + c * channel_block + p;
                                flags[pos] = 1;
                            }
                        }
                    }
                    if (cnd1 && (cnd7 == 0)){
                        uint64_t i = 0;
                        uint64_t pos = pos_minusone;
                        while (flags[pos] == 0 && t > i){
                            for (uint64_t b = 0; b < num_baselines; b++){
                                for (uint64_t p = 0; p < num_pols; p++){
                                    uint64_t pos = t * timesample_block + b * baseline_block + c * channel_block + p;
                                    flags[pos] = 1;
                                }
                            }
                            i = i + 1;
                            pos = (t - i) * timesample_block + b * baseline_block + c * channel_block;
                        }
                    }

                }

            }
        }
        for (uint64_t t = 0; t < num_timesamples; t++){
            for (uint64_t c = 1; c < num_channels; c++) {
                uint64_t pos_current = t * timesample_block + b * baseline_block + c * channel_block;
                uint64_t pos_minusone = t * timesample_block + b * baseline_block + (c - 1) * channel_block;
                double vis0 = std::abs(visibilities[pos_current]);
                double vis1 = std::abs(visibilities[pos_minusone]);
                dv_between_cur_one = vis0 - vis1;
                dv_ratio =  dv_between_cur_one/vis1;
                if (flags[pos_current] != 1){
                    bool cnd0 = dv_ratio > tol_margin_2sm_freq;
                    bool cnd1 = dv_ratio < (-1) * tol_margin_2sm_freq;
                    bool cnd2 = dv_ratio > (-1) * tol_margin_2sm_freq;
                    bool cnd3 = dv_ratio < tol_margin_2sm_freq;
                    bool cnd4 = dv_ratio < 0;
                    bool cnd5 = dv_ratio > 0;
                    bool cnd6 = flags[pos_minusone] == 2;
                    bool cnd7 = flags[pos_minusone] == 0 || flags[pos_minusone] == 1;
                    if (cnd0 || (((cnd2 && cnd4) || (cnd3 && cnd5)) && cnd6)){
                        for (uint64_t b = 0; b < num_baselines; b++){
                            for (uint64_t p = 0; p < num_pols; p++){
                                uint64_t pos = t * timesample_block + b * baseline_block + c * channel_block + p;
                                flags[pos] = 2;
                            }
                        }
                    }
                    if (cnd1 && (cnd7 == 0)){
                        uint64_t i = 0;
                        uint64_t pos = pos_minusone;
                        while ((flags[pos] == 0 || flags[pos] == 1) && t > i){
                            for (uint64_t b = 0; b < num_baselines; b++){
                                for (uint64_t p = 0; p < num_pols; p++){
                                    uint64_t pos = t * timesample_block + b * baseline_block + c * channel_block + p;
                                    flags[pos] = 2;
                                }
                            }
                            i = i + 1;
                            pos = t * timesample_block + b * baseline_block + (c - 1) * channel_block;
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < num_elements; i++){
        if (flags[i] > 0){
            flags[i] = 1;
        }
    }
}


void sdp_twosm_algo_flagger(
        const sdp_Mem* vis,
        const sdp_Mem* thresholds,
        const sdp_Mem* antennas,
        sdp_Mem* flags,
        sdp_Error* status)
{
    check_params(vis, thresholds, antennas, flags, status);
    if (*status) return;

    const uint64_t num_timesamples   = (uint64_t) sdp_mem_shape_dim(vis, 0);
    const uint64_t num_channels      = (uint64_t) sdp_mem_shape_dim(vis, 2);
    const uint64_t num_pols          = (uint64_t) sdp_mem_shape_dim(vis, 3);
    const uint64_t num_antennas      = (uint64_t) sdp_mem_shape_dim(antennas, 0);


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
                    (const int*) sdp_mem_data_const(antennas),
                    num_timesamples,
                    num_antennas,
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
                    (const int*) sdp_mem_data_const(antennas),
                    num_timesamples,
                    num_antennas,
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
