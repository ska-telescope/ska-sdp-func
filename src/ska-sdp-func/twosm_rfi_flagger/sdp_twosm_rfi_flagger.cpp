/* See the LICENSE file at the top-level directory of this distribution. */

#include <complex>
#include "src/ska-sdp-func/twosm_rfi_flagger/sdp_twosm_rfi_flagger.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include <iostream>

using namespace std;

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
    if (sdp_mem_num_dims(vis) != 2 || sdp_mem_num_dims(flags) != 2)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Visibility and flags arrays must be 2D.");
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


int compare (const void * a, const void * b){
    if (*(double*)a > *(double*)b)
        return 1;
    else if (*(double*)a < *(double*)b)
        return -1;
    else
        return 0;
}

double quantile(double* arr, double q, int n){
    qsort(arr, n, sizeof(double), compare);
    int cutpoint = round(q * n);
    return arr[cutpoint];
}

template<typename AA, typename VL>
void filler(AA* arr, VL val, int length){
    for (int i = 0; i < length; i++){
        arr[i] = val;
    }
}


template<typename FP>
static void twosm_rfi_flagger(
        int* flags,
        const std::complex<FP>* visibilities,
        const FP* thresholds,
        const uint64_t num_timesamples,
        const uint64_t num_channels)
{
    double what_quantile_for_changes = thresholds[0];
    double what_quantile_for_vis = thresholds[1];
    int sampling_step = thresholds[2];
    double q_for_vis = 0;

    int num_samples = num_channels/sampling_step;
    double *samples = new double[num_samples];
    filler(samples, 0, num_samples);

    for (uint64_t t = 0; t < num_timesamples; t++){
        cout << " Helllllooooo!!" << endl;
        for (uint64_t s = 0; s < num_samples; s++){
            samples[s] = abs(visibilities[t * num_channels + s * sampling_step]);
        }
        q_for_vis = quantile(samples, what_quantile_for_vis, num_samples);
        cout << t << "    " << q_for_vis << endl;
        for (uint64_t c = 0; c < num_channels; c++){
            double vis0 = abs(visibilities[t * num_channels + c]);
            if (vis0 > q_for_vis){
                flags[t * num_channels + c] = 1;
            }
        }
    }






    /*
    double dv_between_cur_one = 0;
    double dv_ratio = 0;
    double tol_margin_2sm_time = thresholds[0];
    double tol_margin_2sm_freq = thresholds[1];
    uint64_t num_elements = num_timesamples * num_channels;

    for (uint64_t c = 0; c < num_channels; c++){
        for (uint64_t t = 1; t < num_timesamples; t++) {
            uint64_t pos_current = t * num_channels + c; // current position
            uint64_t pos_minusone = (t - 1) * num_channels + c; //position at t-1
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

                //detecting uphill transition ('good' to 'bad') or remaining in the bad position
                if (cnd0 || (((cnd2 && cnd4) || (cnd3 && cnd5)) && cnd6)){
                    uint64_t pos = t * num_channels + c;
                    flags[pos] = 1;
                }
                // downhill transition detection and back-tracking (if we detect a big downhill we know that we
                // have been in the 'bad' state, so we flag the previous samples)
                if (cnd1 && (cnd7 == 0)){
                    uint64_t i = 0;
                    uint64_t pos = pos_minusone;
                    while (flags[pos] == 0 && t > i){
                        flags[pos] = 1;
                        i = i + 1;
                        pos = (t - i) * num_channels + c;
                    }
                }

            }

        }
    }
    // two-state machine algorithm applied to frequency dimension
    // we register the flag with number 2 in here to make calculating the union
    // of the flags in frequency and time directions easier.
    for (uint64_t t = 0; t < num_timesamples; t++){
        for (uint64_t c = 1; c < num_channels; c++) {
            uint64_t pos_current = t * num_channels + c;
            uint64_t pos_minusone = t * num_channels + c - 1;
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
                //detecting uphill transition ('good' to 'bad') or remaining in the bad position
                if (cnd0 || (((cnd2 && cnd4) || (cnd3 && cnd5)) && cnd6)){
                   flags[pos_current] = 2;
                }
                // downhill transition detection and back-tracking (if we detect a big downhill we know that we
                // have been in the 'bad' state, so we flag the previous samples)
                if (cnd1 && (cnd7 == 0)){
                    uint64_t i = 0;
                    uint64_t pos = pos_minusone;
                    while ((flags[pos] == 0 || flags[pos] == 1) && t > i){
                        flags[pos] = 2;
                        i = i + 1;
                        pos = t * num_channels + (c - i) ;
                    }
                }
            }
        }
    }
    // calculating the union (replace all 2's and 1's with 1)
    for (uint64_t i = 0; i < num_elements; i++){
        if (flags[i] > 0){
            flags[i] = 1;
        }
    }
    */
}


void sdp_twosm_algo_flagger(
        const sdp_Mem* vis,
        const sdp_Mem* thresholds,
        sdp_Mem* flags,
        sdp_Error* status)
{
    check_params(vis, thresholds, flags, status);
    if (*status) return;

    const uint64_t num_timesamples   = (uint64_t) sdp_mem_shape_dim(vis, 0);
    const uint64_t num_channels      = (uint64_t) sdp_mem_shape_dim(vis, 1);

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
                    num_channels
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
                    num_channels
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
