// Copyright 2022 High Performance Computing Research Laboratory, Auckland University of Technology (AUT)

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/**
 * @file Gaincalsimpletest.h
 * @author Andrew Ensor
 * @brief C with C++ templates/CUDA code for preparing simple test measured and predicted visibility data sets
 */

#ifndef GAINCAL_SIMPLE_TEST_H
#define GAINCAL_SIMPLE_TEST_H

#include "Gaincalfunctionshost.h"

/**
 * @brief Gain calibration function that allocates and clears the data structures that will
 * hold the visibility data set on the host.
 *
 * Note should be paired with a later call to free_visibilities_host
 * @param num_baselines Number of baselines which gives the number of visibilities.
 * @return The allocated visibility data array on the host.
 */
template<typename VIS_PRECISION2>
VIS_PRECISION2* allocate_visibilities_host
    (const unsigned int num_baselines);

/**
 * @brief Gain calibration function that generates some simple sample visibilities on the host.
 *
 * @param visibilities_host Output array of visibilities.
 * @param num_baselines Number of baselines which gives the number of visibilities.
 */
template<typename VIS_PRECISION2>
void generate_sample_visibilities_host
    (
    VIS_PRECISION2 *visibilities_host, // output array of predicted visibilities
    const unsigned int num_baselines // number of baselines
    );

/**
 * @brief Gain calibration function that deallocates host data structure that was used to
 * hold a visibility data set on the host.
 *
 * Note should be paired with an earlier call to allocate_visibilities_host.
 * @param visibilities_host Input array of visibilities.
 */
template<typename VIS_PRECISION2>
void free_visibilities_host(VIS_PRECISION2 *visibilities_host);

/**
 * @brief Gain calibration function that allocates and clears the data structure that will
 * hold the receiver pairs for each baseline on the host.
 *
 * Note should be paired with a later call to free_receiver_pairs_host
 * @param num_baselines Number of baselines which gives the number of visibilities.
 * @return The allocated receiver pair array on the host.
 */
uint2* allocate_receiver_pairs_host
    (
    const unsigned int num_baselines // number of baselines
    );

/**
 * @brief Gain calibration function that generates some simple sample visibilities on the host
 *
 * @param receiver_pairs_host Output array giving receiver pair for each baseline.
 * @param num_baselines Number of baselines.
 * @param num_receivers Number of receivers.
 */
void generate_sample_receiver_pairs_host
    (
    uint2 *receiver_pairs_host, // output array giving receiver pair for each baseline
    const unsigned int num_baselines, // number of baselines
    const unsigned int num_receivers // number of receivers
    );

/**
 * @brief Gain calibration function that deallocates host data structure that was used to
 * hold the receiver pairs for each baseline on the host.
 *
 * Note should be paired with an earlier call to allocate_receiver_pairs_host
 * @param receiver_pairs_host Input array giving receiver pair for each baseline.
 */
void free_receiver_pairs_host(uint2 *receiver_pairs_host);

/**
 * @brief Gain calibration function that allocates and clears the data structure that will
 * hold a visibility data set on the device.
 *
 * Note should be paired with a later call to free_visibilities_device.
 * @param num_baselines Number of baselines which gives the number of visibilities.
 * @return The allocated visibility data array on the device.
 */
template<typename VIS_PRECISION2>
VIS_PRECISION2* allocate_visibilities_device
    (const unsigned int num_baselines);

/**
 * @brief Temporary utility function that calculates some measured and predicted visibilities for given gains.
 *
 * @param vis_predicted_host Input array of preducted visibilities.
 * @param receiver_pairs_host Input array giving receiver pair for each baseline.
 * @param num_baselines Number of baselines.
 * @param actual_gains_host Input array giving actual complex gains for each receiver.
 * @param num_receivers Number of receivers.
 * @param vis_measured_device Output array of measured visibilities.
 * @param vis_predicted_device Output array of preducted visibilities.
 */
template<typename VIS_PRECISION2, typename VIS_PRECISION, typename PRECISION2>
void calculate_measured_and_predicted_visibilities_device
    (
    VIS_PRECISION2 *vis_predicted_host, // input array of predicted visibilities
    uint2 *receiver_pairs_host, // input array giving receiver pair for each baseline
    const unsigned int num_baselines, // number of baselines
    PRECISION2 *actual_gains_host, // actual complex gains for each receiver
    const unsigned int num_receivers, // number of receivers
    VIS_PRECISION2 *vis_measured_device, // output array of measured visibilities
    VIS_PRECISION2 *vis_predicted_device // output array of preducted visibilities
    );

/**
 * @brief Gain calibration function that deallocates device data structure that was used to
 * hold a visibility data set on the device.
 *
 * Note should be paired with an earlier call to allocate_visibilities_device.
 * @param visibilities_device Visibility dataset held on device.
 */
template<typename VIS_PRECISION2>
void free_visibilities_device(VIS_PRECISION2 *visibilities_device);

/**
 * @brief Gain calibration function that allocates and clears the data structure that will
 * hold the receiver pairs for each baseline on the device.
 *
 * Note should be paired with a later call to free_receiver_pairs_device.
 * @param num_baselines Number of baselines.
 */
uint2* allocate_receiver_pairs_device(const unsigned int num_baselines);

/**
 * @brief Gain calibration function that copies receiver pairs for each baseline from host to the device.
 *
 * @param receiver_pairs_host Input array giving receiver pair for each baseline.
 * @param receiver_pairs_device Inout array giving receiver pair for each baseline.
 * @param num_baselines Number of baselines.
 */
void set_receiver_pairs_device
    (
    uint2 *receiver_pairs_host, // input array of receiver pairs for each baseline
    uint2 *receiver_pairs_device, // inout array of receiver pairs for each baseline
    const unsigned int num_baselines // number of baselines
    );

/**
 * @brief Gain calibration function that deallocates device data structure that was used to
 * hold the receiver pairs for each baseline on the device.
 *
 * Note should be paired with an earlier call to allocate_receiver_pairs_device.
 * @param visibilities_device Visibility dataset held on device.
 */
void free_receiver_pairs_device(uint2 *receiver_pairs_device);

/**
 * @brief Gain calibration function that allocates and clears the data structure that will
 * hold the calculated gains on the device.
 *
 * Note should be paired with a later call to free_gains_device.
 * @param num_receivers Number of receivers.
 * @return The allocated gain data array on the device.
 */
template<typename PRECISION2>
PRECISION2* allocate_gains_device
    (const unsigned int num_receivers);

/**
 * @brief Gain calibration function that displays the actual and calculated gains with
 * all the calculated gains rotated so receiver 0 has zero phase.
 *
 * @param actual_gains_host Input array giving actual complex gains for each receiver.
 * @param gains_device Input array giving calculated complex gains for each receiver.
 * @param num_receivers Number of receivers.
 */
template<typename PRECISION2>
void display_gains_actual_and_calculated
    (
    PRECISION2 *actual_gains_host, // actual complex gains for each receiver
    PRECISION2 *gains_device, // calculated gains
    const unsigned int num_receivers
    );

/**
 * @brief Gain calibration function that deallocates device data structure that was used to
 * hold the calculated gains on the device.
 *
 * Note should be paired with an earlier call to allocate_gains_device.
 * @param gains_device Gain dataset held on device.
 */
template<typename PRECISION2>
void free_gains_device(PRECISION2 *gains_device);




#endif /* include guard */
