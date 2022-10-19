/*
 * Gaincalprocessingfunctiontest.cpp
 * Andrew Ensor
 * C++ program for testing the SDP processing function interface steps for the Gain Calibration algorithm
 * Note apart from convenience calls of allocate_visibilities_host, agenerate_sample_visibilities_host,
 * allocate_receiver_pairs_host, generate_sample_receiver_pairs_host, allocate_visibilities_device,
 * calculate_measured_and_predicted_visibilities_device, allocate_receiver_pairs_device,
 * set_receiver_pairs_device, allocate_gains_device, free_gains_device, free_receiver_pairs_device,
 * free_visibilities_device, free_receiver_pairs_host, free_visibilities_host
 * this code would also be compilable as C
*/

#include "Gaincalprocessingfunctiontest.h"
#include "Gaincalsimpletest.h" // C++ interface used for convenience during testing

#define GAINCAL_PRECISION_SINGLE 1

// use Knuth's method to find a pseudo-gaussian random number with mean 0 and standard deviation 1
double get_random_gaussian()
{
    double v1, v2, s;
    do
    {
        v1 = 2.0 * ((double) rand()/RAND_MAX) - 1;
        v2 = 2.0 * ((double) rand()/RAND_MAX) - 1;
        s = v1*v1 + v2*v2;
    }
    while (s>=1.0);
    if (s == 0.0)
        return 0.0;
    else
        return v1 * sqrt(-2.0*log(s)/s);
}

/**********************************************************************
 * Main method to execute
 **********************************************************************/
int main(int argc, char *argv[])
{
    printf("Gain calibration processing function interface test starting");
    #ifdef GAINCAL_PRECISION_SINGLE
        printf(" using single precision\n");
        #define PRECISION float
        #define PRECISION2 float2
        #define VIS_PRECISION float
        #define VIS_PRECISION2 float2
        #define SDP_MEM_PRECISION SDP_MEM_COMPLEX_FLOAT
    #else
        printf(" using double precision\n");
        #define PRECISION double
        #define PRECISION2 double2
        #define VIS_PRECISION double
        #define VIS_PRECISION2 double2
        #define SDP_MEM_PRECISION SDP_MEM_COMPLEX_DOUBLE
    #endif

    setLogLevel(LOG_DEBUG);
    // note call srand with seed to have different random gains

    // create some simple sample visibilities and gains for testing
    const unsigned int num_receivers = 10;
    const unsigned int num_baselines = num_receivers*(num_receivers-1)/2;
    const unsigned int max_calibration_cycles = 10;

    VIS_PRECISION2 *vis_predicted_host = allocate_visibilities_host<VIS_PRECISION2>(num_baselines);
    generate_sample_visibilities_host<VIS_PRECISION2>(vis_predicted_host, num_baselines);
    uint2 *receiver_pairs_host = allocate_receiver_pairs_host(num_baselines);
    generate_sample_receiver_pairs_host(receiver_pairs_host, num_baselines, num_receivers);
    VIS_PRECISION2 *vis_predicted_device = allocate_visibilities_device<VIS_PRECISION2>(num_baselines);
    VIS_PRECISION2 *vis_measured_device = allocate_visibilities_device<VIS_PRECISION2>(num_baselines);
    PRECISION2 actual_gains_host[num_receivers];
    for (int receiver=0; receiver<num_receivers; receiver++)
    {
        PRECISION amplitude = (PRECISION)(1.0+get_random_gaussian()*0.1);
        PRECISION phase = get_random_gaussian()*0.1;
        actual_gains_host[receiver].x = amplitude * cos(phase);
        actual_gains_host[receiver].y = amplitude * sin(phase);
    }
    calculate_measured_and_predicted_visibilities_device<VIS_PRECISION2, VIS_PRECISION, PRECISION2>
        (vis_predicted_host, receiver_pairs_host, num_baselines, actual_gains_host,
        num_receivers, vis_measured_device, vis_predicted_device);
    uint2 *receiver_pairs_device = allocate_receiver_pairs_device(num_baselines);
    set_receiver_pairs_device(receiver_pairs_host, receiver_pairs_device, num_baselines);
    PRECISION2 *gains_device = allocate_gains_device<PRECISION2>(num_receivers);

    // wrap the simple measured visibilities, predicted visibilities, receiver pairs and gains as sdp_Mem
    sdp_Error *status;
    const int64_t vis_shape[] = {num_baselines};
    sdp_Mem *vis_measured = sdp_mem_create_wrapper(vis_measured_device, SDP_MEM_PRECISION, SDP_MEM_GPU, 1, vis_shape, 0, status);
    sdp_Mem *vis_predicted = sdp_mem_create_wrapper(vis_predicted_device, SDP_MEM_PRECISION, SDP_MEM_GPU, 1, vis_shape, 0, status);
    const int64_t receiver_pairs_shape[] = {num_baselines, 2};
    sdp_Mem *receiver_pairs = sdp_mem_create_wrapper(receiver_pairs_device, SDP_MEM_INT, SDP_MEM_GPU, 2, receiver_pairs_shape, 0, status);
    const int64_t gains_shape[] = {num_receivers};
    sdp_Mem *gains = sdp_mem_create_wrapper(gains_device, SDP_MEM_PRECISION, SDP_MEM_GPU, 1, gains_shape, 0, status);

    perform_gaincalibration(vis_measured, vis_predicted, receiver_pairs, num_receivers, num_baselines,
        max_calibration_cycles, gains);

    display_gains_actual_and_calculated<PRECISION2>(actual_gains_host, gains_device, num_receivers);

    free_gains_device<PRECISION2>(gains_device);
    free_receiver_pairs_device(receiver_pairs_device);
    free_visibilities_device<VIS_PRECISION2>(vis_measured_device);
    free_visibilities_device<VIS_PRECISION2>(vis_predicted_device);
    free_receiver_pairs_host(receiver_pairs_host);
    free_visibilities_host<VIS_PRECISION2>(vis_predicted_host);
    printf("Gain calibration test ending\n");

    checkCudaStatus();
    return 0;
}