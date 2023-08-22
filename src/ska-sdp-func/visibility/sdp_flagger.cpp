/* See the LICENSE file at the top-level directory of this distribution. */

#include <complex>
#include "src/ska-sdp-func/visibility/sdp_flagger.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include <iostream>
#include <iomanip>
#include "omp.h"


using namespace std;

static void check_params(
        const sdp_Mem* vis,
        const sdp_Mem* parameters,
        sdp_Mem* flags,
        const sdp_Mem* antennas,
        const sdp_Mem* baseline1,
        const sdp_Mem* baseline2,
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
        !sdp_mem_is_c_contiguous(parameters) ||
        !sdp_mem_is_c_contiguous(flags) ||
        !sdp_mem_is_c_contiguous(antennas) ||
        !sdp_mem_is_c_contiguous(baseline1) ||
        !sdp_mem_is_c_contiguous(baseline2))
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

    if (sdp_mem_location(vis) != sdp_mem_location(parameters) ||
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
    double interval = 0.5;
    double central = 0.5;
    double left = 0.25;
    double right = 0.75;
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
    return central;
}

int my_baseline_ids(int antenna_id, const int* baseline1, const int* baseline2, int* my_ids, int nbaselines){
    int where_myself = 0;
    int k= 0;
    for (int i = 0; i < nbaselines; i++){
        if (baseline1[i] == antenna_id || baseline2[i] == antenna_id){
            my_ids[k] = i;
            k++;
        }
    }  
    return where_myself;
}





template<typename FP>
static void flagger_fixed_threshold(
        const std::complex<FP>* visibilities,
        const FP* parameters,
        int* flags,
        const int* antennas,
        const int* baseline1,
        const int* baseline2,
        const uint64_t num_timesamples,
        const uint64_t num_baselines,
        const uint64_t num_channels,
        const uint64_t num_pols,
        const uint64_t num_antennas){
    double start; 
    double end;
    start = omp_get_wtime(); 
    double what_quantile_for_vis = parameters[0];
    double what_quantile_for_changes = parameters[1];
    int sampling_step = parameters[2];
    double alpha = parameters[3];
    int window = parameters[4];
    

#pragma omp parallel  shared(parameters, flags, visibilities, antennas, baseline1, baseline2) 
    {

      double q_for_vis = 0;
      double q_for_ts = 0;
   
      int num_samples = num_channels / sampling_step;
      double *samples = new double[num_samples];
      double *transit_score = new double[num_channels];
      double *transit_samples = new double[num_samples];
      int *my_ids = new int[num_antennas];

      filler(samples, 0, num_samples);
      filler(transit_score, 0, num_channels);
      filler(transit_samples, 0, num_samples);
      filler(my_ids, 0, num_antennas);
    
      int time_block = num_baselines * num_channels * num_pols;
      int baseline_block = num_channels * num_pols;
    

      
        #pragma omp for 
        for (int a = 0; a < num_antennas; a++){
            int current_antenna = antennas[a];
            int b = my_baseline_ids(current_antenna, baseline1, baseline2, my_ids, num_baselines);
              
            for (int p = 0; p < num_pols; p++){
                int k = 0;
                for (int t = 0; t < num_timesamples; t++){
                    int time_pos = t * time_block;
                    int baseline_pos = t * time_block + b * baseline_block;
                                         
                    
                    // method 1 only operating on absolute values:

                    //calculating the threshold by sorting the sampled channels and find the value of the given percentile
                    int sum_pos = 0;
                    double sum_samples = 0;
                    for (int s = 0; s < num_samples; s++) {
                        int pos = baseline_pos + (s * sampling_step) * num_pols + p; 
                        samples[s] = abs(visibilities[pos]);
                    }
                    
                  
                    qsort(samples, num_samples, sizeof(double), compare);
                    q_for_vis = samples[int(round(num_samples * what_quantile_for_vis))];                   
                                     
                    for (int c = 0; c < num_channels; c++){
                        int pos = baseline_pos + c * num_pols + p;
                        double vis1 = abs(visibilities[pos]);

                        if (vis1 > q_for_vis){ 
                            for (int bb = 0; bb < num_antennas; bb++){
                                int bs = my_ids[bb];
                                int baseline_pos_for_bs = time_pos + bs * baseline_block;
                                pos = baseline_pos_for_bs + c * num_pols + p;
                                flags[pos] = 1;

                                if (window > 0) {
                                    for (int w = 0; w < window; w++) {
                                        if (c - w - 1 > 0) {
                                            int pos = baseline_pos_for_bs + (c - w - 1) * num_pols + p;
                                            flags[pos] = 1;
                                        }
                                        if (c + w + 1 < num_channels) {
                                            int pos = baseline_pos_for_bs + (c + w + 1) * num_pols + p;
                                           flags[pos] = 1;
                                        }
                                    }
                                }
  
                            }
                        }
                   
                    }

                 
                 
                    // method 2 operating on rate of changes (fluctuations):
                    if (t > 0){
                        for (uint64_t c = 0; c < num_channels; c++) {
                            int pos0 = baseline_pos + c * num_pols + p;
                            int pos1 = (t - 1) * time_block + b * baseline_block + c * num_pols + p;
                            double vis0 = abs(visibilities[pos0]);
                            double vis1 = abs(visibilities[pos1]);
                            double rate_of_change = abs(vis1 - vis0);
                           
                            if (t == 1) {
                                transit_score[c] = rate_of_change;
                            } else {
                                transit_score[c] = alpha * rate_of_change + (1 - alpha) * transit_score[c];
                            }
                        }

                        for (uint64_t s = 0; s < num_samples; s++) {
                            transit_samples[s] = abs(transit_score[s * sampling_step]);   
                        }

                        qsort(transit_samples, num_samples, sizeof(double), compare);
                        q_for_ts = transit_samples[int(round(num_samples * what_quantile_for_changes))];
                                 
                        for (uint64_t c = 0; c < num_channels; c++){
                            int pos = baseline_pos + c * num_pols + p;
                            double ts = abs(transit_score[c]);
                
                            if (ts > q_for_ts) {
                                for (int bb = 0; bb < num_antennas; bb++) {
                                    int bs = my_ids[bb];
                                    int baseline_pos_for_bs = time_pos + bs * baseline_block;
                                    pos = baseline_pos_for_bs + c * num_pols + p;
                                    flags[pos] = 1;
                                     if (window > 0) {
                                       for (int w = 0; w < window; w++) {
                                          if (c - w - 1 > 0) {
                                              int pos = baseline_pos_for_bs + (c - w - 1) * num_pols + p;
                                              flags[pos] = 1;
                                          }
                                          if (c + w + 1 < num_channels) {
                                              int pos = baseline_pos_for_bs + (c + w + 1) * num_pols + p;
                                              flags[pos] = 1;
                                       }
                                    }
                                 }

                            }
                         }
                      }

                  }
               }
            }
        }

    delete transit_score;
    delete samples;
    delete transit_samples;
    delete my_ids;
   }
end = omp_get_wtime(); 
printf("Work took %f seconds\n", end - start);
}









template<typename FP>
static void flagger_dynamic_threshold(
        const std::complex<FP>* visibilities,
        const FP* parameters,
        int* flags,
        const int* antennas,
        const int* baseline1,
        const int* baseline2,
        const uint64_t num_timesamples,
        const uint64_t num_baselines,
        const uint64_t num_channels,
        const uint64_t num_pols,
        const uint64_t num_antennas){
    double start; 
    double end;
    start = omp_get_wtime(); 
    double alpha = parameters[0];
    double beta = parameters[1];
    double termination = parameters[2];
    int sampling_step = parameters[3];
    int window = parameters[4];
 
    int my_thread;

#pragma omp parallel  shared(parameters, flags, visibilities, antennas, baseline1, baseline2)
   {

    double q_for_vis = 0;
    double q_for_ts = 0;
    double median_for_vis = 0;
    double median_for_ts = 0;

    int num_samples = num_channels / sampling_step;
    double *samples = new double[num_samples];
    double *transit_score = new double[num_channels];
    double *transit_samples = new double[num_samples];
    int *my_ids = new int[num_antennas];

    filler(samples, 0, num_samples);
    filler(transit_score, 0, num_channels);
    filler(transit_samples, 0, num_samples);
    filler(my_ids, 0, num_antennas);

        #pragma omp for
        for (int a = 0; a < num_antennas; a++){
            int current_antenna = antennas[a];
            int b = my_baseline_ids(current_antenna, baseline1, baseline2, my_ids, num_baselines);
        
            for (int p = 0; p < num_pols; p++){
                for (uint64_t t = 0; t < num_timesamples; t++){
     
                    int time_block = num_baselines * num_channels * num_pols;
                    int baseline_block = num_channels * num_pols;
                    int time_pos = t * time_block;
                    int baseline_pos = t * time_block + b * baseline_block;
                    // method 1 only operating on absolute values:
                   
                    //calculating the threshold by sorting the sampled channels and find the value of the given percentile
                    for (uint64_t s = 0; s < num_samples; s++) {
                        int pos = baseline_pos + (s * sampling_step) * num_pols + p;
                        samples[s] = abs(visibilities[pos]);
                    }
                    qsort(samples, num_samples, sizeof(double), compare);
                    q_for_vis = samples[int(round(num_samples * knee(samples, termination, num_samples)))];
                    median_for_vis = samples[int(round(0.5 * num_samples))];
//                    cout << q_for_vis << "    " << median_for_vis << endl;  
                                     
                    for (uint64_t c = 0; c < num_channels; c++){
                        int pos = baseline_pos + c * num_pols + p;
                        double vis1 = abs(visibilities[pos]);
                        if (vis1 > q_for_vis || vis1 > median_for_vis + (1 - beta) * (q_for_vis - median_for_vis)){  
                            
                            for (int bb = 0; bb < num_antennas; bb++) {
                                    int bs = my_ids[bb];
                                    int baseline_pos_for_bs = time_pos + bs * baseline_block;
                                    pos = baseline_pos_for_bs + c * num_pols + p;
                                    flags[pos] = 1;
                       
                                    if (window > 0) {
                                        for (int w = 0; w < window; w++) {
                                            if (c - w - 1 > 0) {
                                                int pos = baseline_pos_for_bs + (c - w - 1) * num_pols + p;
                                                flags[pos] = 1;
                                            } 
                                            if (c + w + 1 < num_channels) {
                                                int pos = baseline_pos_for_bs + (c + w + 1) * num_pols + p;          
                                                flags[pos] = 1;
                                            }

                                        }
                                    }

                              }

                         }

                    }


                    // method 2 operating on rate of changes (fluctuations):
                    if (t > 0){
                        for (uint64_t c = 0; c < num_channels; c++) {
                            int pos0 = baseline_pos + c * num_pols + p;
                            int pos1 = (t - 1) * time_block + b * baseline_block + c * num_pols + p;
                            double vis0 = abs(visibilities[pos0]);
                            double vis1 = abs(visibilities[pos1]);
                            double rate_of_change = abs(vis1 - vis0);

                            if (t == 1) {
                                transit_score[c] = rate_of_change;
                            } else {
                                transit_score[c] = alpha * rate_of_change + (1 - alpha) * transit_score[c];
                            }
                        }

                        for (uint64_t s = 0; s < num_samples; s++) {
                            transit_samples[s] = abs(transit_score[s * sampling_step]);
                        }

                        qsort(transit_samples, num_samples, sizeof(double), compare);
                        q_for_ts = transit_samples[int(round(num_samples * knee(samples, termination, num_samples)))];
                        median_for_ts = transit_samples[int(round(0.5 * num_samples))];
//                        cout << q_for_ts << "    " << median_for_ts << endl;

                        for (uint64_t c = 0; c < num_channels; c++){
                            int pos = baseline_pos + c * num_pols + p;
                            double ts = abs(transit_score[c]);
                      
                            if (ts > q_for_ts || ts > median_for_ts + (1 - beta) * (q_for_ts - median_for_ts)) {
                                for (int bb = 0; bb < num_antennas; bb++) {
                                    int bs = my_ids[bb];                                    
                                    int baseline_pos_for_bs = time_pos + bs * baseline_block;
                                    pos = baseline_pos_for_bs + c * num_pols + p;
                                    flags[pos] = 1;
                                    if (window > 0) {
                                        for (int w = 0; w < window; w++) {
                                            if (c - w - 1 > 0) {
                                                int pos = baseline_pos_for_bs + (c - w - 1) * num_pols + p;
                                                flags[pos] = 1;
                                            }
                                            if (c + w + 1 < num_channels) {
                                                int pos = baseline_pos_for_bs + (c + w + 1) * num_pols + p;
                                                flags[pos] = 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                    }
                }
            }
        }
    delete transit_score;
    delete samples;
    delete transit_samples;
    delete my_ids;
    }
end = omp_get_wtime(); 
printf("Work took %f seconds\n", end - start);
}















void sdp_flagger_fixed_threshold(
        const sdp_Mem* vis,
        const sdp_Mem* parameters,
        sdp_Mem* flags,
        const sdp_Mem* antennas,
        const sdp_Mem* baseline1,
        const sdp_Mem* baseline2,
        sdp_Error* status){
    check_params(vis, parameters, flags, antennas, baseline1, baseline2, status);
    if (*status) return;

    const uint64_t num_timesamples   = (uint64_t) sdp_mem_shape_dim(vis, 0);
    const uint64_t num_baselines   = (uint64_t) sdp_mem_shape_dim(vis, 1);
    const uint64_t num_channels      = (uint64_t) sdp_mem_shape_dim(vis, 2);
    const uint64_t num_pols   = (uint64_t) sdp_mem_shape_dim(vis, 3);
    const uint64_t num_antennas = (uint64_t) sdp_mem_shape_dim(antennas, 0);

    if (sdp_mem_location(vis) == SDP_MEM_CPU)
    {
        if (sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT &&
            sdp_mem_type(parameters) == SDP_MEM_FLOAT &&
            sdp_mem_type(flags) == SDP_MEM_INT &&
            sdp_mem_type(antennas) == SDP_MEM_INT &&
            sdp_mem_type(baseline1) == SDP_MEM_INT &&
            sdp_mem_type(baseline2) == SDP_MEM_INT)
        {
            flagger_fixed_threshold(
                    (const std::complex<float>*) sdp_mem_data_const(vis),
                    (const float*) sdp_mem_data_const(parameters),
                    (int*) sdp_mem_data(flags),
                    (const int*) sdp_mem_data_const(antennas),
                    (const int*) sdp_mem_data_const(baseline1),
                    (const int*) sdp_mem_data_const(baseline2),
                    num_timesamples,
                    num_baselines,
                    num_channels,
                    num_pols,
                    num_antennas
            );
        }
        else if (sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE &&
                 sdp_mem_type(parameters) == SDP_MEM_DOUBLE &&
                 sdp_mem_type(flags) == SDP_MEM_INT &&
                 sdp_mem_type(antennas) == SDP_MEM_INT &&
                 sdp_mem_type(baseline1) == SDP_MEM_INT &&
                 sdp_mem_type(baseline1) == SDP_MEM_INT)
        {
            flagger_fixed_threshold(
                    (const std::complex<double>*) sdp_mem_data_const(vis),
                    (const double*) sdp_mem_data_const(parameters),
                    (int*) sdp_mem_data(flags),
                    (const int*) sdp_mem_data_const(antennas),
                    (const int*) sdp_mem_data_const(baseline1),
                    (const int*) sdp_mem_data_const(baseline2),
                    num_timesamples,
                    num_baselines,
                    num_channels,
                    num_pols,
                    num_antennas
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





void sdp_flagger_dynamic_threshold(
        const sdp_Mem* vis,
        const sdp_Mem* parameters,
        sdp_Mem* flags,
        const sdp_Mem* antennas,
        const sdp_Mem* baseline1,
        const sdp_Mem* baseline2,
        sdp_Error* status){
    check_params(vis, parameters, flags, antennas, baseline1, baseline2, status);
    if (*status) return;

    const uint64_t num_timesamples   = (uint64_t) sdp_mem_shape_dim(vis, 0);
    const uint64_t num_baselines   = (uint64_t) sdp_mem_shape_dim(vis, 1);
    const uint64_t num_channels      = (uint64_t) sdp_mem_shape_dim(vis, 2);
    const uint64_t num_pols   = (uint64_t) sdp_mem_shape_dim(vis, 3);
    const uint64_t num_antennas = (uint64_t) sdp_mem_shape_dim(antennas, 0);

    if (sdp_mem_location(vis) == SDP_MEM_CPU)
    {
        if (sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT &&
            sdp_mem_type(parameters) == SDP_MEM_FLOAT &&
            sdp_mem_type(flags) == SDP_MEM_INT &&
            sdp_mem_type(antennas) == SDP_MEM_INT &&
            sdp_mem_type(baseline1) == SDP_MEM_INT &&
            sdp_mem_type(baseline2) == SDP_MEM_INT)
        {
            flagger_dynamic_threshold(
                    (const std::complex<float>*) sdp_mem_data_const(vis),
                    (const float*) sdp_mem_data_const(parameters),
                    (int*) sdp_mem_data(flags),
                    (const int*) sdp_mem_data_const(antennas),
                    (const int*) sdp_mem_data_const(baseline1),
                    (const int*) sdp_mem_data_const(baseline2),
                    num_timesamples,
                    num_baselines,
                    num_channels,
                    num_pols,
                    num_antennas
            );
        }
        else if (sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE &&
                 sdp_mem_type(parameters) == SDP_MEM_DOUBLE &&
                 sdp_mem_type(flags) == SDP_MEM_INT &&
                 sdp_mem_type(antennas) == SDP_MEM_INT &&
                 sdp_mem_type(baseline1) == SDP_MEM_INT &&
                 sdp_mem_type(baseline1) == SDP_MEM_INT)
        {
            flagger_dynamic_threshold(
                    (const std::complex<double>*) sdp_mem_data_const(vis),
                    (const double*) sdp_mem_data_const(parameters),
                    (int*) sdp_mem_data(flags),
                    (const int*) sdp_mem_data_const(antennas),
                    (const int*) sdp_mem_data_const(baseline1),
                    (const int*) sdp_mem_data_const(baseline2),
                    num_timesamples,
                    num_baselines,
                    num_channels,
                    num_pols,
                    num_antennas
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

