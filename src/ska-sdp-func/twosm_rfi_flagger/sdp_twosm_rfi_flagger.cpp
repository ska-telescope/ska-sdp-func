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

double extrapolate(double mrec, int mrecloc, double minone,
                  int minoneloc, double mintwo, int mintwoloc, uint64_t current_time){
    double extrapolated_val;
    if (mrecloc != -1 && minoneloc == -1 && mintwoloc == -1){
        extrapolated_val = mrec;
    }
    if (mrecloc != -1 && minoneloc != -1 && mintwoloc == -1){
        extrapolated_val = mrec + ((mrec - minone)/(mrecloc - minoneloc)) * (current_time - mrecloc);
    }
    if (mrecloc != -1 && minoneloc != -1 && mintwoloc != -1){
        double firstdev1 = (mrec - minone)/(mrecloc - minoneloc);
        double firstdev2 = (minone - mintwo)/(minoneloc - mintwoloc);
        double seconddev = firstdev1 - firstdev2;
        double predicted_firstdev = firstdev1 + seconddev;
        extrapolated_val = mrec + (current_time - mrecloc) * predicted_firstdev;
    }

    return extrapolated_val;
}


template<typename AA, typename VL>
void filler(AA* arr, VL val, int length){
    for (int i = 0; i < length; i++){
        arr[i] = val;
    }
}

/*double moving_first_dev(double* arr, int window){
   double val1 = 0;
   double val2 = 0;
   for (int j = 0; j < window; j++){
       val1 = val1 + arr[j];
   }
    for (int k = 0; k < window; k++){
        val2 = val2 + arr[k + 1];
    }
    return val2 - val1;
}

double moving_second_dev(double* arr, int window){
    double *for_first_dev1 = new double[window];
    double *for_first_dev2 = new double[window];
    for (int i = 0; i < window; i++){
        for_first_dev1[i] = arr[i];
    }
    for (int k = 0; k < window; k++){
        for_first_dev2[k] = arr[k + 1];
    }
    double first_dev1 =

}*/



template<typename FP>
static void twosm_rfi_flagger(
        int* flags,
        const std::complex<FP>* visibilities,
        const FP* thresholds,
        const uint64_t num_timesamples,
        const uint64_t num_channels)
{
    double quant_ft_low = 0;
    double quant_ft_high = 0;
    double quant_td = 0;

    double *most_rec = new double[num_channels];
    int *most_rec_loc = new int[num_channels];
    double *minus_one = new double[num_channels];
    int *minus_one_loc = new int[num_channels];
    double *minus_two = new double[num_channels];
    int *minus_two_loc = new int[num_channels];
    double *transit_score = new double[num_channels];

    filler(most_rec, 0, num_channels);
    filler(most_rec_loc, -1, num_channels);
    filler(minus_one, 0, num_channels);
    filler(minus_one_loc, -1, num_channels);
    filler(minus_two, 0, num_channels);
    filler(minus_two_loc, -1, num_channels);
    filler(transit_score, 0, num_channels);


    for (uint64_t t = 0; t < num_timesamples; t++) {
        if (t == 0) {
            double *first_time = new double[num_channels - 1];
            for (uint64_t k = 0; k < num_channels - 1; k++) {
                first_time[k] = abs(visibilities[t * num_channels + k]);
            }
            quant_ft_high = quantile(first_time, 0.9, num_channels - 1);
            quant_ft_low = quantile(first_time, 0.1, num_channels - 1);
            delete[] first_time;
            for (uint64_t c = 0; c < num_channels; c++){
                double vis0 = abs(visibilities[t * num_channels + c]);
                if (vis0 < quant_ft_low || vis0 > quant_ft_high){
                    flags[t * num_channels + c] = 1;
                }else{
                    most_rec_loc[t * num_channels + c] = 0;
                    most_rec[t * num_channels + c] = vis0;
                }
            }
        }

        if (t == 1) {
            double *time_diffs = new double[num_channels - 1];
            for (uint64_t m = 0; m < num_channels; m++) {
                time_diffs[m] = abs(
                        abs(visibilities[t * num_channels + m] - abs(visibilities[(t - 1) * num_channels + m])));
            }
            quant_td = quantile(time_diffs, 0.9, num_channels);
            delete[] time_diffs;
            for (int c = 0; c < num_channels; c++){
                double vis1 = abs(visibilities[t * num_channels + c]);
                double vis0 = abs(visibilities[(t - 1) * num_channels + c]);
                if (vis1 > quant_ft_high || vis1 < quant_ft_low || abs(vis1 - vis0) > quant_td){
                    flags[t * num_channels + c] = 1;
                }else{
                    if (most_rec_loc[c] == -1){
                        most_rec_loc[c] = 0;
                        most_rec[c] = vis1;
                    } else{
                        most_rec_loc[c] = 1;
                        most_rec[c] = vis1;
                        minus_one[c] = vis0;
                        minus_one_loc[c] = 0;
                    }

                }
                transit_score[c] = abs(vis1 - vis0);
            }
        }
        if (t > 1){
            for (int c = 0; c < num_channels; c++){
                double vis1 = abs(visibilities[t * num_channels + c]);
                double vis0 = abs(visibilities[(t - 1) * num_channels + c]);
                transit_score[c] = abs(vis1 - vis0) + thresholds[2] * transit_score[c];
                if (flags[(t - 1) * num_channels + c] == 1){
                    double m0 = most_rec[c];
                    int l0 = most_rec_loc[c];
                    double m1 = minus_one[c];
                    int l1 = minus_one_loc[c];
                    double m2 = minus_two[c];
                    int l2 = minus_two_loc[c];
                    double extpl = extrapolate(m0, l0, m1, l1, m2, l2, t);
                    bool cnd0 = !(vis1 > quant_ft_high || vis1< quant_ft_low);
                    bool cnd1 = transit_score[c] < thresholds[0] * quant_td;
                    bool cnd2 = abs(vis1 - extpl) < thresholds[1] * quant_td;
                    if (cnd0 && cnd1 && cnd2){
                        flags[t * num_channels + c] = 1;
                    } else{
                        minus_two[c] = minus_one[c];
                        minus_two_loc[c] = minus_one_loc[c];
                        minus_one[c] = most_rec[c];
                        minus_one_loc[c] = most_rec_loc[c];
                        most_rec[c] = vis1;
                        most_rec_loc[c] = t;
                    }
                }
            }
        }
    }



    /*for (uint64_t c = 0; c < num_channels; c++){
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
    }*/
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
