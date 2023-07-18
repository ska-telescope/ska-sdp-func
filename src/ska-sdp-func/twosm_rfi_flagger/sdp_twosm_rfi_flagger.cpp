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

double distance(double x0, double y0, double x1, double y1, double x2, double y2){
    double top = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1));
    double bottom = sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));
    double dist = top/bottom;
    return dist;
}

double knee(double* arr, double termination, int n){
    int place = -1;
    double interval = 1;
    double central = 0.5;
    double left = 0.25;
    double right = 0.75;
    int central_pos = round(central * n);
    int left_pos = round(left * n);
    int right_pos = round(right * n);

    double x1 = arr[0];
    double y1 = 1/n;
    double x2 = arr[n-1];
    double y2 = 1;

    while (interval > termination){
        double dist_left = distance(arr[left_pos], left, x1, y1, x2, y2);
        double dist_right = distance(arr[right_pos], right, x1, y1, x2, y2);
        if (dist_left > dist_right){
            interval = interval/2;
            central = left;
            left = left - interval/2;
            right = left + interval/2;
            left_pos = round(left * n);
            right_pos = round(right * n);
        } else{
            interval = interval/2;
            central = right;
            left = right - interval/2;
            right = right + interval/2;
            left_pos = round(left * n);
            right_pos = round(right * n);
        }
    }
    cout << "This is the quantile: " << central << endl;
    return central;
}

double auto_quantile(double* arr, double termination, int n){
    qsort(arr, n, sizeof(double), compare);
    int cutpoint = knee(arr, termination, n);
    return arr[cutpoint];
}


template<typename FP>
static void twosm_rfi_flagger(
        int* flags,
        const std::complex<FP>* visibilities,
        const FP* thresholds,
        const uint64_t num_timesamples,
        const uint64_t num_channels)
{
    cout << " Helllllooooo!!" << endl;

    double what_quantile_for_changes = thresholds[0];
    double what_quantile_for_vis = thresholds[1];
    int sampling_step = thresholds[2];
    double alpha = thresholds[3];
    int window = thresholds[4];
    int settlement = thresholds[5];
    double termination = thresholds[6];
    double q_for_vis = 0;
    double q_for_ts = 0;

    int num_samples = num_channels/sampling_step;
    double *samples = new double[num_samples];
    double *transit_score = new double[num_channels];
    double *transit_samples = new double[num_samples];
    filler(samples, 0, num_samples);
    filler(transit_score, 0, num_channels);

    for (uint64_t t = 0; t < num_timesamples; t++){
        for (uint64_t s = 0; s < num_samples; s++){
            samples[s] = abs(visibilities[t * num_channels + s * sampling_step]);
        }
     //   q_for_vis = quantile(samples, what_quantile_for_vis, num_samples);
        q_for_vis = auto_quantile(samples, termination, num_samples);
        for (uint64_t c = 0; c < num_channels; c++) {
            double vis1 = abs(visibilities[t * num_channels + c]);
            if (vis1 > q_for_vis) {
                flags[t * num_channels + c] = 1;
                if (window > 0){
                    for (int w = 0; w < window; w++){
                        if (c-w-1 > 0){
                            flags[t * num_channels + c - w - 1] = 1;
                        }
                        if (c+w+1 < num_channels){
                            flags[t * num_channels + c + w + 1] = 1;
                        }
                    }
                }
            }
        }

        if (t > 0){
            for (uint64_t c = 0; c < num_channels; c++){
                double vis0 = abs(visibilities[(t - 1) * num_channels + c]);
                double vis1 = abs(visibilities[t * num_channels + c]);
                double rate_of_change = abs(vis1 - vis0);
                if (t == 1){
                    transit_score[c] = rate_of_change;
                } else{
                    transit_score[c] = alpha * rate_of_change + (1 - alpha) * transit_score[c];
                }
            }

            for (uint64_t s = 0; s < num_samples; s++){
                transit_samples[s] = transit_score[s * sampling_step];
            }
          //  q_for_ts = quantile(transit_samples, what_quantile_for_changes, num_samples);
            q_for_ts = quantile(transit_samples, termination, num_samples);
            for (uint64_t c = 0; c < num_channels; c++){
                if (transit_score[c] > q_for_ts){
                    flags[t * num_channels + c] = 1;
                    flags[(t - 1) * num_channels + c] = 1;
                    if (window > 0){
                        for (int w = 0; w < window; w++){
                            if (c-w-1 > 0){
                                flags[t * num_channels + c - w - 1] = 1;
                            }
                            if (c+w+1 < num_channels){
                                flags[t * num_channels + c + w + 1] = 1;
                            }
                        }
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