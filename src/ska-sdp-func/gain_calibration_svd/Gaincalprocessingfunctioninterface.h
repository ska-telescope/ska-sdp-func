/**
 * @file Gaincalprocessingfunctioninterface.h
 * @author Andrew Ensor
 * @brief C with C++ templates/CUDA program providing an SDP processing function interface steps for the Gain Calibration algorithm
 */

#ifndef GAINCAL_PROCESSING_FUNCTIONS_INTERFACE_H
#define GAINCAL_PROCESSING_FUNCTIONS_INTERFACE_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief C (untemplated) version of the function which allocates and clears the data structures that will
 * hold the visibility data set on the host.
 * 
 * @param num_baselines Number of baselines.
 * @param mem_type The data type to use for the visibilities, either SDP_MEM_COMPLEX_FLOAT or SDP_MEM_COMPLEX_DOUBLE.
 * @return The allocated sdp_Mem data structure on the host.
 */
sdp_Mem *sdp_gaincal_allocate_visibilities_host
    (const unsigned int num_baselines, sdp_MemType mem_type);

/**
 * @brief C (untemplated) version of the function which allocates and clears the data structures that will
 * hold the visibility data set on the host.
 *
 * @param visibilities The allocated sdp_Mem data structure on the host.
 * @param num_baselines Number of baselines.
 */
void sdp_gaincal_generate_sample_visibilities_host
    (sdp_Mem *visibilities, const unsigned int num_baselines);

/**
 * @brief C (untemplated) version of the function which deallocates device data structure that was used to
 * hold the visibilities on the host
 *
 * @param visibilities The allocated sdp_Mem data structure on the host.
 */
void sdp_gaincal_free_visibilities_host(sdp_Mem *visibilities);

/**
 * @brief C (untemplated) version of the function which allocates and clears the data structures that will
 * hold the receiver pairs for each baseline on the host.
 * 
 * @param num_baselines Number of baselines.
 * @return The allocated sdp_Mem data structure on the host.
 */
sdp_Mem *sdp_gaincal_allocate_receiver_pairs_host
    (const unsigned int num_baselines);

/**
 * @brief C (untemplated) version of the function which generates receiver pairs on the host.
 *
 * @param receiver_pairs The allocated sdp_Mem data structure on the host.
 * @param num_baselines Number of baselines.
 * @param num_receivers Number of receivers.
 */
void sdp_gaincal_generate_sample_receiver_pairs_host
    (
    sdp_Mem *receiver_pairs,
    const unsigned int num_baselines,
    const unsigned int num_receivers
    );

/**
 * @brief C (untemplated) version of the function which deallocates device data structure that was used to
 * hold the receiver pairs for each baseline on the host.
 *
 * @param receiver_pairs The allocated sdp_Mem data structure on the host.
 */
void sdp_gaincal_free_receiver_pairs_host(sdp_Mem *receiver_pairs);

/**
 * @brief C (untemplated) version of the function which allocates and clears the data structures that will
 * hold the visibility data set on the device.
 * 
 * @param num_baselines Number of baselines.
 * @param mem_type The data type to use for the visibilities, either SDP_MEM_COMPLEX_FLOAT or SDP_MEM_COMPLEX_DOUBLE.
 * @return The allocated sdp_Mem data structure on the device.
 */
sdp_Mem *sdp_gaincal_allocate_visibilities_device
    (const unsigned int num_baselines, sdp_MemType mem_type);

/**
 * @brief C (untemplated) version of the function which calculates some measured and predicted visibilities
 * for given gains
 *
 * @param vis_predicted_host The input sdp_Mem of predicted visibilities.
 * @param receiver_pairs_host The input sdp_Mem giving receiver pair for each baseline.
 * @param num_baselines Number of baselines.
 * @param actual_gains_host The actual complex gains for each receiver.
 * @param num_receivers Number of receivers.
 * @param vis_measured_device Output sdp_Mem of measured visibilities.
 * @param vis_predicted_device Output sdp_Mem of predicted visibilities.
 */
void sdp_gaincal_calculate_measured_and_predicted_visibilities_device
    (
    sdp_Mem *vis_predicted_host,
    sdp_Mem *receiver_pairs_host,
    const unsigned int num_baselines,
    sdp_Mem *actual_gains_host,
    const unsigned int num_receivers,
    sdp_Mem *vis_measured_device,
    sdp_Mem *vis_predicted_device
    );

/**
 * @brief C (untemplated) version of the function which deallocates device data structure that was used to
 * hold the visibilities on the device
 *
 * @param visibilities The allocated sdp_Mem data structure on the device.
 */
void sdp_gaincal_free_visibilities_device(sdp_Mem *visibilities);

/**
 * @brief C (untemplated) version of the function which allocates and clears the data structures that will
 * hold the receiver pairs for each baseline on the device.
 * 
 * @param num_baselines Number of baselines.
 * @return The allocated sdp_Mem data structure on the device.
 */
sdp_Mem *sdp_gaincal_allocate_receiver_pairs_device
    (const unsigned int num_baselines);

/**
 * @brief C (untemplated) version of the function which copies receiver pairs for each baseline from host to the device.
 *
 * @param receiver_pairs_host The allocated sdp_Mem data structure on the host.
 * @param receiver_pairs_device The allocated sdp_Mem data structure on the device.
 * @param num_baselines Number of baselines.
 */
void sdp_gaincal_set_receiver_pairs_device
    (
    sdp_Mem *receiver_pairs_host,
    sdp_Mem *receiver_pairs_device,
    const unsigned int num_baselines
    );

/**
 * @brief C (untemplated) version of the function which deallocates device data structure that was used to
 * hold the receiver pairs for each baseline on the device.
 *
 * @param receiver_pairs The allocated sdp_Mem data structure on the device.
 */
void sdp_gaincal_free_receiver_pairs_device(sdp_Mem *receiver_pairs);

/**
 * @brief C (untemplated) version of the function which allocates and clears the data structures that will
 * hold the calculated gains on the device and initialises each gain to 1+0i.
 * 
 * @param num_receivers Number of receivers.
 * @param mem_type The data type to use for the gains, either SDP_MEM_COMPLEX_FLOAT or SDP_MEM_COMPLEX_DOUBLE.
 * @return The allocated sdp_Mem data structure on the device.
 */
sdp_Mem *sdp_gaincal_allocate_gains_device
    (const unsigned int num_receivers, sdp_MemType mem_type);

/**
 * @brief C (untemplated) version of the utility function which generates pseudo-random numbers
 * Uses Knuth's method to find a pseudo-gaussian random number with mean 0 and standard deviation 1.
 *
 * @return A pseudo-random number.
 */
float sdp_gaincal_get_random_gaussian_float();
double sdp_gaincal_get_random_gaussian_double();

/**
 * @brief C (untemplated) version of the function which performs the gain calibration using sdp_Mem handles.
 * 
 * @param vis_measured_device Input data of measured visibilities.
 * @param vis_predicted_device Input data of preducted visibilities.
 * @param receiver_pairs_device Input array giving receiver pair for each baseline.
 * @param num_receivers Number of receivers.
 * @param num_baselines Number of baselines.
 * @param max_calibration_cycles Maximum number of calibration cycles to perform.
 * @param gains_device Output data of calculated complex gains.
 */
void sdp_gaincal_perform
    (
    sdp_Mem *vis_measured_device,
    sdp_Mem *vis_predicted_device,
    sdp_Mem *receiver_pairs_device,
    const unsigned int num_receivers,
    const unsigned int num_baselines,
    const unsigned int max_calibration_cycles,
    sdp_Mem *gains_device
    );

/**
 * @brief C (untemplated) version of the function which displays the actual and calculated gains with
 * all the calculated gains rotated so receiver 0 has zero phase.
 * 
 * @param actual_gains_host The actual sdp_Mem data structure complex gains for each receiver on the host.
 * @param gains_device The calculated sdp_Mem data structure gains on the device.
 * @param num_receivers Number of receivers.
 */
void sdp_gaincal_display_gains_actual_and_calculated
    (
    sdp_Mem *actual_gains_host,
    sdp_Mem *gains_device,
    const unsigned int num_receivers
    );

/**
 * @brief C (untemplated) version of the function which deallocates device data structure that was used to
 * hold the calculated gains on the device.
 *
 * @param gains The allocated sdp_Mem data structure on the device.
 */
void sdp_gaincal_free_gains_device(sdp_Mem *gains);


#ifdef __cplusplus
}
#endif

#endif /* include guard */