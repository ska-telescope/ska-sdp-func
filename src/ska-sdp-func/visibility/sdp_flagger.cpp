/* See the LICENSE file at the top-level directory of this distribution. */

#include <complex>
#include <math.h>
#include "src/ska-sdp-func/visibility/sdp_flagger.h"


using namespace std;


static void check_params_dynamic(
        const sdp_Mem* vis,
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

    if (sdp_mem_location(vis) != sdp_mem_location(flags))
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("All arrays must be in the same memory location.");
        return;
    }
}


static int compare(const void* a, const void* b)
{
    const double* da = static_cast<const double*>(a);
    const double* db = static_cast<const double*>(b);

    if (*da > *db)
        return 1;
    else if (*da < *db)
        return -1;
    else
        return 0;
}


template<typename AA, typename VL>
static void filler(AA* arr, VL val, int length)
{
    for (int i = 0; i < length; i++)
    {
        arr[i] = val;
    }
}


static double median_calc(double* arr, int n)
{
    int mid = int(round(0.5 * n));
    double median = arr[mid];
    return median;
}


static double median_dev_calc(double* arr, int n, double median)
{
    int mid = int(round(0.5 * n));
    double* devs = new double[n];
    for (int i = 0; i < n; i++)
    {
        devs[i] = abs(arr[i] - median);
    }
    qsort(devs, n, sizeof(double), compare);
    double medidev = devs[mid];
    delete[] devs;
    return medidev;
}


static double modified_zscore(double median, double mediandev, double val)
{
    double zscore = 0;
    if (mediandev == 0 && val == median)
    {
        zscore = 0;
    }
    else if (mediandev == 0 && val != median)
    {
        zscore = 10000000;
    }
    else
    {
        zscore = 0.6795 * (val - median) / mediandev;
    }
    return zscore;
}


template<typename FP>
static void flagger_dynamic_threshold(
        const std::complex<FP>* visibilities,
        int* flags,
        const double alpha,
        const double threshold_magnitudes,
        const double threshold_variations,
        const double threshold_broadband,
        const int sampling_step,
        const int window,
        const int window_median_history,
        const int num_timesamples,
        const int num_baselines,
        const int num_channels,
        const int num_pols
)
{
#pragma omp parallel

    {
        int num_samples = num_channels / sampling_step;
        double* samples = new double[num_samples];
        double* transit_score = new double[num_channels];
        double* transit_samples = new double[num_samples];
        double* median_history = new double[num_timesamples];

        filler(samples, 0, num_samples);
        filler(transit_score, 0, num_channels);
        filler(transit_samples, 0, num_samples);
        filler(median_history, 0, num_timesamples);

        #pragma omp for
        for (int b = 0; b < num_baselines; b++)
        {
            for (int p = 0; p < num_pols; p++)
            {
                for (int t = 0; t < num_timesamples; t++)
                {
                    int situation = 0;
                    int time_block = num_baselines * num_channels * num_pols;
                    int baseline_block = num_channels * num_pols;
                    int baseline_pos = t * time_block + b * baseline_block;

                    int medwindow = std::min(t + 1, window_median_history);
                    double* medarray = new double[medwindow];

                    // method 1 only operating on absolute values
                    // and method 3 for broadband detection:

                    for (int s = 0; s < num_samples; s++)
                    {
                        int pos = baseline_pos + (s * sampling_step) *
                                num_pols + p;
                        samples[s] = abs(visibilities[pos]);
                    }
                    qsort(samples, num_samples, sizeof(double), compare);

                    double median = median_calc(samples, num_samples);
                    double mediandev = median_dev_calc(samples,
                            num_samples,
                            median
                    );

                    median_history[t] = median;

                    for (int tt = 0; tt < medwindow; tt++)
                    {
                        medarray[tt] = median_history[t - tt];
                    }

                    qsort(medarray, medwindow, sizeof(double), compare);

                    double medmed = median_calc(medarray, medwindow);

                    double medmeddev = median_dev_calc(medarray,
                            medwindow,
                            medmed
                    );
                    double zscore_med = modified_zscore(medmed,
                            medmeddev,
                            median
                    );

                    if ((zscore_med > threshold_broadband ||
                            zscore_med < -threshold_broadband) && t != 0)
                    {
                        situation = 1;
                    }

                    for (int c = 0; c < num_channels; c++)
                    {
                        int pos = baseline_pos + c * num_pols + p;
                        double vis1 = abs(visibilities[pos]);
                        double zscore_mags = modified_zscore(median,
                                mediandev,
                                vis1
                        );

                        if (zscore_mags > threshold_magnitudes ||
                                zscore_mags < -threshold_magnitudes ||
                                situation == 1)
                        {
                            flags[pos] = 1;
                            if (window > 0)
                            {
                                for (int w = 0; w < window; w++)
                                {
                                    if (c - w - 1 > 0)
                                    {
                                        pos = baseline_pos + (c - w - 1) *
                                                num_pols + p;
                                        flags[pos] = 1;
                                    }
                                    if (c + w + 1 < num_channels)
                                    {
                                        pos = baseline_pos + (c + w + 1) *
                                                num_pols + p;
                                        flags[pos] = 1;
                                    }
                                }
                            }
                        }
                    }

                    delete[] medarray;

                    // method 2 operating on rate of changes (fluctuations):

                    if (t > 0)
                    {
                        int baseline_pos_minus_one = (t - 1) * time_block + b *
                                baseline_block;
                        for (int c = 0; c < num_channels; c++)
                        {
                            int pos0 = baseline_pos + c * num_pols + p;
                            int pos1 = (t - 1) * time_block + b *
                                    baseline_block + c * num_pols + p;
                            double vis0 = abs(visibilities[pos0]);
                            double vis1 = abs(visibilities[pos1]);
                            double rate_of_change = abs(vis1 - vis0);

                            if (t == 1)
                            {
                                transit_score[c] = rate_of_change;
                            }
                            else
                            {
                                transit_score[c] = alpha * rate_of_change +
                                        (1 - alpha) * transit_score[c];
                            }
                        }

                        for (int s = 0; s < num_samples; s++)
                        {
                            transit_samples[s] =
                                    abs(transit_score[s * sampling_step]);
                        }

                        qsort(transit_samples,
                                num_samples,
                                sizeof(double),
                                compare
                        );

                        double medianvar = median_calc(transit_samples,
                                num_samples
                        );
                        double mediandevvar = median_dev_calc(transit_samples,
                                num_samples,
                                median
                        );

                        for (int c = 0; c < num_channels; c++)
                        {
                            int pos = baseline_pos + c * num_pols + p;
                            int pos_minus_one = baseline_pos_minus_one + c *
                                    num_pols + p;
                            double ts = abs(transit_score[c]);
                            double zscore_vars = modified_zscore(medianvar,
                                    mediandevvar,
                                    ts
                            );

                            if (zscore_vars > threshold_variations ||
                                    zscore_vars < -threshold_variations)
                            {
                                flags[pos] = 1;
                                flags[pos_minus_one] = 1;
                                if (window > 0)
                                {
                                    for (int w = 0; w < window; w++)
                                    {
                                        if (c - w - 1 > 0)
                                        {
                                            pos = baseline_pos + (c - w - 1) *
                                                    num_pols + p;
                                            pos_minus_one =
                                                    baseline_pos_minus_one +
                                                    (c - w - 1) * num_pols + p;
                                            flags[pos] = 1;
                                            flags[pos_minus_one] = 1;
                                        }
                                        if (c + w + 1 < num_channels)
                                        {
                                            pos = baseline_pos + (c + w + 1) *
                                                    num_pols + p;
                                            pos_minus_one =
                                                    baseline_pos_minus_one +
                                                    (c + w + 1) * num_pols + p;
                                            flags[pos] = 1;
                                            flags[pos_minus_one] = 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        delete[] transit_score;
        delete[] samples;
        delete[] transit_samples;
        delete[] median_history;
    }
}


void sdp_flagger_dynamic_threshold(
        const sdp_Mem* vis,
        sdp_Mem* flags,
        const double alpha,
        const double threshold_magnitudes,
        const double threshold_variations,
        const double threshold_broadband,
        const int sampling_step,
        const int window,
        const int window_median_history,
        sdp_Error* status
)
{
    check_params_dynamic(vis, flags, status);
    if (*status) return;

    const int num_timesamples   = (int) sdp_mem_shape_dim(vis, 0);
    const int num_baselines   = (int) sdp_mem_shape_dim(vis, 1);
    const int num_channels      = (int) sdp_mem_shape_dim(vis, 2);
    const int num_pols   = (int) sdp_mem_shape_dim(vis, 3);

    if (sdp_mem_location(vis) == SDP_MEM_CPU)
    {
        if (sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT &&
                sdp_mem_type(flags) == SDP_MEM_INT)
        {
            flagger_dynamic_threshold(
                    (const std::complex<float>*) sdp_mem_data_const(vis),
                    (int*) sdp_mem_data(flags),
                    alpha,
                    threshold_magnitudes,
                    threshold_variations,
                    threshold_broadband,
                    sampling_step,
                    window,
                    window_median_history,
                    num_timesamples,
                    num_baselines,
                    num_channels,
                    num_pols
            );
        }
        else if (sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(flags) == SDP_MEM_INT)
        {
            flagger_dynamic_threshold(
                    (const std::complex<double>*) sdp_mem_data_const(vis),
                    (int*) sdp_mem_data(flags),
                    alpha,
                    threshold_magnitudes,
                    threshold_variations,
                    threshold_broadband,
                    sampling_step,
                    window,
                    window_median_history,
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
