/*
 * Gaincalprocessingfunctioninterface.cpp
 * Andrew Ensor
 * C with C++ templates/CUDA program providing an SDP processing function interface steps for the Gain Calibration algorithm
*/

#include "Gaincalprocessingfunctioninterface.h"
#include "Gaincallogger.h"
#include "Gaincalfunctionshost.h"
#include "Gaincalsimpletest.h"


/**********************************************************************
 * C (untemplated) version of the function which allocates and clears the data structures that will
 * hold the visibility data set on the host.
 **********************************************************************/
sdp_Mem *sdp_gaincal_allocate_visibilities_host
    (const unsigned int num_baselines, sdp_MemType mem_type)
{
    sdp_Mem *visibilities = NULL;
    if (mem_type == SDP_MEM_COMPLEX_FLOAT)
    {
        sdp_Error *status = NULL;
        float2 *visibilities_host = allocate_visibilities_host<float2>(num_baselines);
        const int64_t visibilities_shape[] = {num_baselines};
        visibilities = sdp_mem_create_wrapper(visibilities_host, SDP_MEM_COMPLEX_FLOAT, SDP_MEM_CPU, 1, visibilities_shape, 0, status);
    }
    else if (mem_type == SDP_MEM_COMPLEX_DOUBLE)
    {
        sdp_Error *status = NULL;
        double2 *visibilities_host = allocate_visibilities_host<double2>(num_baselines);
        const int64_t visibilities_shape[] = {num_baselines};
        visibilities = sdp_mem_create_wrapper(visibilities_host, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 1, visibilities_shape, 0, status);
    }
    else
    {
        logger(LOG_CRIT, "Visibilities on host must be either SDP_MEM_COMPLEX_FLOAT or else SDP_MEM_COMPLEX_DOUBLE");
    }
    return visibilities;
}


/*****************************************************************************
 * C (untemplated) version of the function which adds some sources to dirty_moment_images_device for testing
 *****************************************************************************/
void sdp_gaincal_generate_sample_visibilities_host
    (
    sdp_Mem *visibilities, const unsigned int num_baselines
    )
{
    if (sdp_mem_location(visibilities) != SDP_MEM_CPU)
    {
        logger(LOG_CRIT, "Visibilities must be on host");
    }
    else if (sdp_mem_type(visibilities) == SDP_MEM_COMPLEX_FLOAT)
    {
        generate_sample_visibilities_host<float2>((float2 *)sdp_mem_data(visibilities), num_baselines);
    }
    else if (sdp_mem_type(visibilities) == SDP_MEM_COMPLEX_DOUBLE)
    {
        generate_sample_visibilities_host<double2>((double2 *)sdp_mem_data(visibilities), num_baselines);
    }
    else
    {
        logger(LOG_CRIT, "Visibilities on host must be either SDP_MEM_COMPLEX_FLOAT or else SDP_MEM_COMPLEX_DOUBLE");
    }
}


/*****************************************************************************
 * C (untemplated) version of the function which deallocates device data structure that was used to
 * hold the visibilities on the host
 *****************************************************************************/
void sdp_gaincal_free_visibilities_host(sdp_Mem *visibilities)
{
    if (sdp_mem_type(visibilities) == SDP_MEM_COMPLEX_FLOAT)
    {
        free_visibilities_host<float2>((float2 *)sdp_mem_data(visibilities));
    }
    else if (sdp_mem_type(visibilities) == SDP_MEM_COMPLEX_DOUBLE)
    {
        free_visibilities_host<double2>((double2 *)sdp_mem_data(visibilities));
    }
    else
    {
        logger(LOG_CRIT, "Visibilities on host must be either SDP_MEM_COMPLEX_FLOAT or else SDP_MEM_COMPLEX_DOUBLE");
    }
}


/**********************************************************************
 * C (untemplated) version of the function which allocates and clears the data structures that will
 * hold the receiver pairs for each baseline on the host.
 **********************************************************************/
sdp_Mem *sdp_gaincal_allocate_receiver_pairs_host
    (const unsigned int num_baselines)
{
    sdp_Mem *receiver_pairs = NULL;
    sdp_Error *status = NULL;
    uint2 *receiver_pairs_host = allocate_receiver_pairs_host(num_baselines);
    const int64_t receiver_pairs_shape[] = {num_baselines, 2};
    receiver_pairs = sdp_mem_create_wrapper(receiver_pairs_host, SDP_MEM_INT, SDP_MEM_CPU, 2, receiver_pairs_shape, 0, status);
    return receiver_pairs;
}


/*****************************************************************************
 * C (untemplated) version of the function which generates receiver pairs on the host
 *****************************************************************************/
void sdp_gaincal_generate_sample_receiver_pairs_host
    (
    sdp_Mem *receiver_pairs,
    const unsigned int num_baselines, // number of baselines
    const unsigned int num_receivers // number of receivers
    )
{
    generate_sample_receiver_pairs_host((uint2 *)sdp_mem_data(receiver_pairs), num_baselines, num_receivers);
}


/*****************************************************************************
 * C (untemplated) version of the function which deallocates device data structure that was used to
 * hold the receiver pairs for each baseline on the host.
 *****************************************************************************/
void sdp_gaincal_free_receiver_pairs_host(sdp_Mem *receiver_pairs)
{
    if (sdp_mem_type(receiver_pairs) == SDP_MEM_INT)
    {
        free_receiver_pairs_host((uint2 *)sdp_mem_data(receiver_pairs));
    }
    else
    {
        logger(LOG_CRIT, "Receiver pairs on host must be of type SDP_MEM_INT");
    }
}


/**********************************************************************
 * C (untemplated) version of the function which allocates and clears the data structures that will
 * hold the visibility data set on the device.
 **********************************************************************/
sdp_Mem *sdp_gaincal_allocate_visibilities_device
    (const unsigned int num_baselines, sdp_MemType mem_type)
{
    sdp_Mem *visibilities = NULL;
    if (mem_type == SDP_MEM_COMPLEX_FLOAT)
    {
        sdp_Error *status = NULL;
        float2 *visibilities_device = allocate_visibilities_device<float2>(num_baselines);
        const int64_t visibilities_shape[] = {num_baselines};
        visibilities = sdp_mem_create_wrapper(visibilities_device, SDP_MEM_COMPLEX_FLOAT, SDP_MEM_GPU, 1, visibilities_shape, 0, status);
    }
    else if (mem_type == SDP_MEM_COMPLEX_DOUBLE)
    {
        sdp_Error *status = NULL;
        double2 *visibilities_device = allocate_visibilities_device<double2>(num_baselines);
        const int64_t visibilities_shape[] = {num_baselines};
        visibilities = sdp_mem_create_wrapper(visibilities_device, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_GPU, 1, visibilities_shape, 0, status);
    }
    else
    {
        logger(LOG_CRIT, "Visibilities on device must be either SDP_MEM_COMPLEX_FLOAT or else SDP_MEM_COMPLEX_DOUBLE");
    }
    return visibilities;
}


/*****************************************************************************
 * Temporary utility function that calculates some measured and predicted visibilities for given gains
 *****************************************************************************/
void sdp_gaincal_calculate_measured_and_predicted_visibilities_device
    (
    sdp_Mem *vis_predicted_host, // input sdp_Mem of predicted visibilities
    sdp_Mem *receiver_pairs_host, // input sdp_Mem giving receiver pair for each baseline
    const unsigned int num_baselines, // number of baselines
    sdp_Mem *actual_gains_host, // actual complex gains for each receiver
    const unsigned int num_receivers, // number of receivers
    sdp_Mem *vis_measured_device, // output sdp_Mem of measured visibilities
    sdp_Mem *vis_predicted_device // output sdp_Mem of predicted visibilities
    )
{
    if (sdp_mem_location(vis_predicted_host)!=SDP_MEM_CPU || sdp_mem_location(vis_predicted_device)!=SDP_MEM_GPU)
    {
        logger(LOG_CRIT, "Predicted visibilities must be on host and on device");
    }
    else if (sdp_mem_location(vis_measured_device)!=SDP_MEM_GPU)
    {
        logger(LOG_CRIT, "Measured visibilities must be on device");
    }
    else if (sdp_mem_location(actual_gains_host)!=SDP_MEM_CPU)
    {
        logger(LOG_CRIT, "Actual gains must be on host");
    }
    else if (sdp_mem_type(vis_predicted_host)==SDP_MEM_COMPLEX_FLOAT
        && sdp_mem_type(vis_measured_device)==SDP_MEM_COMPLEX_FLOAT
        && sdp_mem_type(vis_predicted_device)==SDP_MEM_COMPLEX_FLOAT)
    {
        calculate_measured_and_predicted_visibilities_device<float2, float, float2>
            ((float2 *)sdp_mem_data(vis_predicted_host), (uint2 *)sdp_mem_data(receiver_pairs_host),
            num_baselines, (float2 *)sdp_mem_data(actual_gains_host),
            num_receivers, (float2 *)sdp_mem_data(vis_measured_device), (float2 *)sdp_mem_data(vis_predicted_device));
    }
    else if (sdp_mem_type(vis_predicted_host)==SDP_MEM_COMPLEX_DOUBLE
        && sdp_mem_type(vis_measured_device)==SDP_MEM_COMPLEX_DOUBLE
        && sdp_mem_type(vis_predicted_device)==SDP_MEM_COMPLEX_DOUBLE)
    {
        calculate_measured_and_predicted_visibilities_device<double2, double, double2>
            ((double2 *)sdp_mem_data(vis_predicted_host), (uint2 *)sdp_mem_data(receiver_pairs_host),
            num_baselines, (double2 *)sdp_mem_data(actual_gains_host),
            num_receivers, (double2 *)sdp_mem_data(vis_measured_device), (double2 *)sdp_mem_data(vis_predicted_device));
    }
    else
    {
        logger(LOG_CRIT, "Predicted and measured visibilities must all be either SDP_MEM_COMPLEX_FLOAT or else SDP_MEM_COMPLEX_DOUBLE");
    }
}


/*****************************************************************************
 * C (untemplated) version of the function which deallocates device data structure that was used to
 * hold the visibilities on the device
 *****************************************************************************/
void sdp_gaincal_free_visibilities_device(sdp_Mem *visibilities)
{
    if (sdp_mem_type(visibilities) == SDP_MEM_COMPLEX_FLOAT)
    {
        free_visibilities_device<float2>((float2 *)sdp_mem_data(visibilities));
    }
    else if (sdp_mem_type(visibilities) == SDP_MEM_COMPLEX_DOUBLE)
    {
        free_visibilities_device<double2>((double2 *)sdp_mem_data(visibilities));
    }
    else
    {
        logger(LOG_CRIT, "Visibilities on device must be either SDP_MEM_COMPLEX_FLOAT or else SDP_MEM_COMPLEX_DOUBLE");
    }
}


/**********************************************************************
 * C (untemplated) version of the function which allocates and clears the data structures that will
 * hold the receiver pairs for each baseline on the device.
 **********************************************************************/
sdp_Mem *sdp_gaincal_allocate_receiver_pairs_device
    (const unsigned int num_baselines)
{
    sdp_Mem *receiver_pairs = NULL;
    sdp_Error *status = NULL;
    uint2 *receiver_pairs_device = allocate_receiver_pairs_device(num_baselines);
    const int64_t receiver_pairs_shape[] = {num_baselines, 2};
    receiver_pairs = sdp_mem_create_wrapper(receiver_pairs_device, SDP_MEM_INT, SDP_MEM_GPU, 2, receiver_pairs_shape, 0, status);
    return receiver_pairs;
}


/*****************************************************************************
 * C (untemplated) version of the function which copies receiver pairs for each baseline from host to the device.
 *****************************************************************************/
void sdp_gaincal_set_receiver_pairs_device
    (
    sdp_Mem *receiver_pairs_host,
    sdp_Mem *receiver_pairs_device,
    const unsigned int num_baselines // number of baselines
    )
{
    if (sdp_mem_location(receiver_pairs_host)!=SDP_MEM_CPU || sdp_mem_location(receiver_pairs_device)!=SDP_MEM_GPU)
    {
        logger(LOG_CRIT, "Receiver pairs must be on both host and on device");
    }
    else if (sdp_mem_type(receiver_pairs_host)!=SDP_MEM_INT || sdp_mem_type(receiver_pairs_device)!=SDP_MEM_INT)
    {
        logger(LOG_CRIT, "Receiver pairs on host and device must be of type SDP_MEM_INT");
    }
    else
    {
        set_receiver_pairs_device((uint2 *)sdp_mem_data(receiver_pairs_host),
            (uint2 *)sdp_mem_data(receiver_pairs_device), num_baselines);
    }
}


/*****************************************************************************
 * C (untemplated) version of the function which deallocates device data structure that was used to
 * hold the receiver pairs for each baseline on the device.
 *****************************************************************************/
void sdp_gaincal_free_receiver_pairs_device(sdp_Mem *receiver_pairs)
{
    if (sdp_mem_type(receiver_pairs) == SDP_MEM_INT)
    {
        free_receiver_pairs_device((uint2 *)sdp_mem_data(receiver_pairs));
    }
    else
    {
        logger(LOG_CRIT, "Receiver pairs on device must be of type SDP_MEM_INT");
    }
}


/**********************************************************************
 * C (untemplated) version of the function which allocates and clears the data structures that will
 * hold the calculated gains on the device and initialises each gain to 1+0i.
 **********************************************************************/
sdp_Mem *sdp_gaincal_allocate_gains_device
    (const unsigned int num_receivers, sdp_MemType mem_type)
{
    sdp_Mem *gains = NULL;
    if (mem_type == SDP_MEM_COMPLEX_FLOAT)
    {
        sdp_Error *status = NULL;
        float2 *gains_device = allocate_gains_device<float2>(num_receivers);
        const int64_t gains_shape[] = {num_receivers};
        gains = sdp_mem_create_wrapper(gains_device, SDP_MEM_COMPLEX_FLOAT, SDP_MEM_GPU, 1, gains_shape, 0, status);
    }
    else if (mem_type == SDP_MEM_COMPLEX_DOUBLE)
    {
        sdp_Error *status = NULL;
        double2 *gains_device = allocate_gains_device<double2>(num_receivers);
        const int64_t gains_shape[] = {num_receivers};
        gains = sdp_mem_create_wrapper(gains_device, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_GPU, 1, gains_shape, 0, status);
    }
    else
    {
        logger(LOG_CRIT, "Gains on device must be either SDP_MEM_COMPLEX_FLOAT or else SDP_MEM_COMPLEX_DOUBLE");
    }
    return gains;
}


/**********************************************************************
 * C (untemplated) version of the utility function which generates pseudo-random numbers
 * Uses Knuth's method to find a pseudo-gaussian random number with mean 0 and standard deviation 1
 **********************************************************************/
float sdp_gaincal_get_random_gaussian_float()
{
    return get_random_gaussian<float>();
}

double sdp_gaincal_get_random_gaussian_double()
{
    return get_random_gaussian<double>();
}


/**********************************************************************
 * C (untemplated) version of the function which performs the entire gain calibration using sdp_Mem handles
 **********************************************************************/
void sdp_gaincal_perform
    (
    sdp_Mem *vis_measured_device, // input data of measured visibilities
    sdp_Mem *vis_predicted_device, // input data of preducted visibilities
    sdp_Mem *receiver_pairs_device, // input data giving receiver pair for each baseline
    const unsigned int num_receivers, // number of receivers
    const unsigned int num_baselines, // number of baselines
    const unsigned int max_calibration_cycles, // maximum number of calibration cycles to perform
    sdp_Mem *gains_device // output data of calculated complex gains
    )
{
    if (sdp_mem_location(vis_measured_device)!=SDP_MEM_GPU || sdp_mem_location(vis_predicted_device)!=SDP_MEM_GPU
        || sdp_mem_location(receiver_pairs_device)!=SDP_MEM_GPU || sdp_mem_location(gains_device)!=SDP_MEM_GPU)
    {
        logger(LOG_CRIT, "Measured visibilities, predicted visibilities, receiver pairs, and gains must all be held in GPU device memory");
    }
    else if (sdp_mem_type(receiver_pairs_device)!=SDP_MEM_INT)
    {
        logger(LOG_CRIT, "Receiver pairs must be pairs of (unsigned) integer values");
    }
    else if (sdp_mem_type(vis_measured_device)==SDP_MEM_COMPLEX_FLOAT && sdp_mem_type(vis_predicted_device)==SDP_MEM_COMPLEX_FLOAT
        && sdp_mem_type(gains_device)==SDP_MEM_COMPLEX_FLOAT)
    {
        // calculate suitable cuda block size in 1D and number of available cuda threads
        int cuda_block_size;
        int cuda_num_threads;
        calculate_cuda_configs(&cuda_block_size, &cuda_num_threads);

        Jacobian_SVD_matrices<float> jacobian_svd_matrices = allocate_jacobian_svd_matrices<float>(num_receivers);

        perform_gain_calibration<float2, float2, float>
            ((float2 *)sdp_mem_data(vis_measured_device), (float2 *)sdp_mem_data(vis_predicted_device),
            (uint2 *)sdp_mem_data(receiver_pairs_device), num_receivers, num_baselines, max_calibration_cycles,
            jacobian_svd_matrices, (float2 *)sdp_mem_data(gains_device), true, cuda_block_size);

        free_jacobian_svd_matrices(jacobian_svd_matrices);
    }
    else if (sdp_mem_type(vis_measured_device)==SDP_MEM_COMPLEX_DOUBLE && sdp_mem_type(vis_predicted_device)==SDP_MEM_COMPLEX_DOUBLE
        && sdp_mem_type(gains_device)==SDP_MEM_COMPLEX_DOUBLE)
    {
        // calculate suitable cuda block size in 1D and number of available cuda threads
        int cuda_block_size;
        int cuda_num_threads;
        calculate_cuda_configs(&cuda_block_size, &cuda_num_threads);

        Jacobian_SVD_matrices<double> jacobian_svd_matrices = allocate_jacobian_svd_matrices<double>(num_receivers);

        perform_gain_calibration<double2, double2, double>
            ((double2 *)sdp_mem_data(vis_measured_device), (double2 *)sdp_mem_data(vis_predicted_device),
            (uint2 *)sdp_mem_data(receiver_pairs_device), num_receivers, num_baselines, max_calibration_cycles,
            jacobian_svd_matrices, (double2 *)sdp_mem_data(gains_device), true, cuda_block_size);

        free_jacobian_svd_matrices(jacobian_svd_matrices);
    }
    else
    {
        logger(LOG_CRIT, "Measured and predicted visibilities and gains must all be either SDP_MEM_COMPLEX_FLOAT or else SDP_MEM_COMPLEX_DOUBLE");
    }
}


/*****************************************************************************
 * C (untemplated) version of the function which displays the actual and calculated gains with
 * all the calculated gains rotated so receiver 0 has zero phase.
 *****************************************************************************/
void sdp_gaincal_display_gains_actual_and_calculated
    (
    sdp_Mem *actual_gains_host, // actual complex gains for each receiver
    sdp_Mem *gains_device, // calculated gains
    const unsigned int num_receivers
    )
{
    if (sdp_mem_location(actual_gains_host)!=SDP_MEM_CPU || sdp_mem_location(gains_device)!=SDP_MEM_GPU)
    {
        logger(LOG_CRIT, "Actual gains must be location on host and device gains must all be held in GPU device memory");
    }
    else if (sdp_mem_type(actual_gains_host)==SDP_MEM_COMPLEX_FLOAT && sdp_mem_type(gains_device)==SDP_MEM_COMPLEX_FLOAT)
    {
        display_gains_actual_and_calculated<float2>((float2 *)sdp_mem_data(actual_gains_host),
            (float2 *)sdp_mem_data(gains_device), num_receivers);
    }
    else if (sdp_mem_type(actual_gains_host)==SDP_MEM_COMPLEX_DOUBLE && sdp_mem_type(gains_device)==SDP_MEM_COMPLEX_DOUBLE)
    {
        display_gains_actual_and_calculated<double2>((double2 *)sdp_mem_data(actual_gains_host),
            (double2 *)sdp_mem_data(gains_device), num_receivers);
    }
    else
    {
        logger(LOG_CRIT, "Actual and device gains must both be either SDP_MEM_COMPLEX_FLOAT or else SDP_MEM_COMPLEX_DOUBLE");
    }
}


/*****************************************************************************
 * C (untemplated) version of the function which deallocates device data structure that was used to
 * hold the calculated gains on the device.
 *****************************************************************************/
void sdp_gaincal_free_gains_device(sdp_Mem *gains)
{
    if (sdp_mem_type(gains) == SDP_MEM_COMPLEX_FLOAT)
    {
        free_gains_device<float2>((float2 *)sdp_mem_data(gains));
    }
    else if (sdp_mem_type(gains) == SDP_MEM_COMPLEX_DOUBLE)
    {
        free_gains_device<double2>((double2 *)sdp_mem_data(gains));
    }
    else
    {
        logger(LOG_CRIT, "Gains on device must be either SDP_MEM_COMPLEX_FLOAT or else SDP_MEM_COMPLEX_DOUBLE");
    }
}


