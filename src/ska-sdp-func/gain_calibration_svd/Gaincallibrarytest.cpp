/*
 * Gaincallibrarytest.cpp
 * Andrew Ensor
 * C with C++ templates/CUDA program for testing steps of the Gain Calibration algorithm
*/

#include "Gaincallibrarytest.h"

#define GAINCAL_PRECISION_SINGLE 1

/**********************************************************************
 * Main method to execute
 **********************************************************************/
int gain_calibration_library_test()
{
    printf("Gain calibration library test starting");
    #ifdef GAINCAL_PRECISION_SINGLE
        printf(" using single precision\n");
        #define PRECISION float
        #define PRECISION2 float2
        #define VIS_PRECISION float
        #define VIS_PRECISION2 float2
    #else
        printf(" using double precision\n");
        #define PRECISION double
        #define PRECISION2 double2
        #define VIS_PRECISION double
        #define VIS_PRECISION2 double2
    #endif

    setLogLevel(LOG_DEBUG);
    // note call srand with seed to have different random gains

    // create some simple sample visibilities and gains for testing
    const unsigned int num_receivers = 10;
    const unsigned int num_baselines = num_receivers*(num_receivers-1)/2;
    VIS_PRECISION2 *vis_predicted_host = allocate_visibilities_host<VIS_PRECISION2>(num_baselines);
    generate_sample_visibilities_host<VIS_PRECISION2>(vis_predicted_host, num_baselines);
    uint2 *receiver_pairs_host = allocate_receiver_pairs_host(num_baselines);
    generate_sample_receiver_pairs_host(receiver_pairs_host, num_baselines, num_receivers);
    VIS_PRECISION2 *vis_predicted_device = allocate_visibilities_device<VIS_PRECISION2>(num_baselines);
    VIS_PRECISION2 *vis_measured_device = allocate_visibilities_device<VIS_PRECISION2>(num_baselines);
    PRECISION2 actual_gains_host[num_receivers];
    for (unsigned int receiver=0; receiver<num_receivers; receiver++)
    {
        PRECISION amplitude = (PRECISION)(1.0+get_random_gaussian<PRECISION>()*0.1);
        PRECISION phase = get_random_gaussian<PRECISION>()*(PRECISION)0.1;
        actual_gains_host[receiver].x = amplitude * cos(phase);
        actual_gains_host[receiver].y = amplitude * sin(phase);
    }
    calculate_measured_and_predicted_visibilities_device<VIS_PRECISION2, VIS_PRECISION, PRECISION2>
        (vis_predicted_host, receiver_pairs_host, num_baselines, actual_gains_host,
        num_receivers, vis_measured_device, vis_predicted_device);
    uint2 *receiver_pairs_device = allocate_receiver_pairs_device(num_baselines);
    set_receiver_pairs_device(receiver_pairs_host, receiver_pairs_device, num_baselines);
    PRECISION2 *gains_device = allocate_gains_device<PRECISION2>(num_receivers);

    // calculate suitable cuda block size in 1D and number of available cuda threads
    int cuda_block_size;
    int cuda_num_threads;
    calculate_cuda_configs(&cuda_block_size, &cuda_num_threads);

    Jacobian_SVD_matrices<PRECISION> jacobian_svd_matrices = allocate_jacobian_svd_matrices<PRECISION>(num_receivers);

    perform_gain_calibration<VIS_PRECISION2, PRECISION2, PRECISION>
        (vis_measured_device, vis_predicted_device, receiver_pairs_device,
        num_receivers, num_baselines, 10, jacobian_svd_matrices, gains_device, true, cuda_block_size);

    display_gains_actual_and_calculated<PRECISION2>(actual_gains_host, gains_device, num_receivers);

    free_jacobian_svd_matrices(jacobian_svd_matrices);

    free_gains_device<PRECISION2>(gains_device);
    free_receiver_pairs_device(receiver_pairs_device);
    free_visibilities_device<VIS_PRECISION2>(vis_measured_device);
    free_visibilities_device<VIS_PRECISION2>(vis_predicted_device);
    free_receiver_pairs_host(receiver_pairs_host);
    free_visibilities_host<VIS_PRECISION2>(vis_predicted_host);
    printf("Gain calibration library test ending\n");
    return 0;
}