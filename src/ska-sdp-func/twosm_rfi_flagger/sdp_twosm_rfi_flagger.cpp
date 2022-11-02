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
    double extrapolated_val = 0;
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

double transit_sc_threshold_calc(double stat, double alpha, double beta, int time){
    double val = beta * stat;
    double threshold = 0;
    for (int t = 0; t < time; t++){
        threshold = threshold + val * pow(alpha, t);
    }
    return threshold;
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
                    bool cnd3 = l0 == -1 && l1 == -1 && l2 == -1;

                    if ((cnd0 && cnd1 && cnd2 && !cnd3) || (cnd0 && cnd1 && cnd3)){
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
                if (c < num_channels - 1){
                    if (flags[t * num_channels + c] == 1 && transit_score[c + 1] > thresholds[3] * quant_td){
                        flags[c + 1] = 1;
                    }
                }
            }
        }

    }



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