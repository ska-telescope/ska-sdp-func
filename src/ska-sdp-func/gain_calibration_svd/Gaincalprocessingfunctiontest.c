/*
 * Gaincalprocessingfunctiontest.c
 * Andrew Ensor
 * C program for testing the SDP processing function interface steps for the Gain Calibration algorithm
*/

#include "Gaincalprocessingfunctiontest.h"

#define GAINCAL_PRECISION_SINGLE 1

typedef struct float2
{
    float x;
    float y;
} float2;
typedef struct double2
{
    float x;
    float y;
} double2;


/**********************************************************************
 * Main method to execute the interface test
 **********************************************************************/
int gain_calibration_interface_test()
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

    // note call srand with seed to have different random gains

    // create some simple sample visibilities and gains for testing
    const unsigned int num_receivers = 10;
    const unsigned int num_baselines = num_receivers*(num_receivers-1)/2;
    const unsigned int max_calibration_cycles = 10;

    sdp_Mem *vis_predicted_host = sdp_gaincal_allocate_visibilities_host(num_baselines, SDP_MEM_PRECISION);
    sdp_gaincal_generate_sample_visibilities_host(vis_predicted_host, num_baselines);
    sdp_Mem *receiver_pairs_host = sdp_gaincal_allocate_receiver_pairs_host(num_baselines);
    sdp_gaincal_generate_sample_receiver_pairs_host(receiver_pairs_host, num_baselines, num_receivers);
    sdp_Mem *vis_predicted_device = sdp_gaincal_allocate_visibilities_device(num_baselines, SDP_MEM_PRECISION);
    sdp_Mem *vis_measured_device = sdp_gaincal_allocate_visibilities_device(num_baselines, SDP_MEM_PRECISION);

    PRECISION2 actual_gains_host_array[num_receivers];
    for (unsigned int receiver=0; receiver<num_receivers; receiver++)
    {
        #ifdef GAINCAL_PRECISION_SINGLE
            PRECISION amplitude = (PRECISION)(1.0+sdp_gaincal_get_random_gaussian_float()*0.1);
            PRECISION phase = sdp_gaincal_get_random_gaussian_float()*(PRECISION)0.1;
        #else
            PRECISION amplitude = (PRECISION)(1.0+sdp_gaincal_get_random_gaussian_double()*0.1);
            PRECISION phase = sdp_gaincal_get_random_gaussian_double()*(PRECISION)0.1;
        #endif
        actual_gains_host_array[receiver].x = amplitude * cos(phase);
        actual_gains_host_array[receiver].y = amplitude * sin(phase);
    }
    sdp_Error *status = NULL;
    const int64_t gains_shape[] = {num_receivers};
    sdp_Mem *actual_gains_host = sdp_mem_create_wrapper(actual_gains_host_array, SDP_MEM_PRECISION, SDP_MEM_CPU, 1, gains_shape, 0, status);

    sdp_gaincal_calculate_measured_and_predicted_visibilities_device
        (vis_predicted_host, receiver_pairs_host, num_baselines, actual_gains_host,
        num_receivers, vis_measured_device, vis_predicted_device);
    sdp_Mem *receiver_pairs_device = sdp_gaincal_allocate_receiver_pairs_device(num_baselines);
    sdp_gaincal_set_receiver_pairs_device(receiver_pairs_host, receiver_pairs_device, num_baselines);
    sdp_Mem *gains_device = sdp_gaincal_allocate_gains_device(num_receivers, SDP_MEM_PRECISION);

    sdp_gaincal_perform(vis_measured_device, vis_predicted_device, receiver_pairs_device, num_receivers, num_baselines,
        max_calibration_cycles, gains_device);

    sdp_gaincal_display_gains_actual_and_calculated(actual_gains_host, gains_device, num_receivers);

    sdp_gaincal_free_gains_device(gains_device);
    sdp_gaincal_free_receiver_pairs_device(receiver_pairs_device);
    sdp_gaincal_free_visibilities_device(vis_measured_device);
    sdp_gaincal_free_visibilities_device(vis_predicted_device);
    sdp_gaincal_free_receiver_pairs_host(receiver_pairs_host);
    sdp_gaincal_free_visibilities_host(vis_predicted_host);
    printf("Gain calibration test ending\n");

    return 0;
}