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
 * @brief C (untemplated) version of the function which performs the gain calibration using sdp_Mem handles
 * 
 * @param vis_measured_device Input data of measured visibilities.
 * @param vis_predicted_device Input data of preducted visibilities.
 * @param receiver_pairs_device Input array giving receiver pair for each baseline.
 * @param num_receivers Number of receivers.
 * @param num_baselines Number of baselines.
 * @param max_calibration_cycles Maximum number of calibration cycles to perform.
 * @param gains_device Output data of calculated complex gains.
 */
void perform_gaincalibration
    (
    sdp_Mem *vis_measured_device,
    sdp_Mem *vis_predicted_device,
    sdp_Mem *receiver_pairs_device,
    const unsigned int num_receivers,
    const unsigned int num_baselines,
    const unsigned int max_calibration_cycles,
    sdp_Mem *gains_device
    );

#ifdef __cplusplus
}
#endif

#endif /* include guard */