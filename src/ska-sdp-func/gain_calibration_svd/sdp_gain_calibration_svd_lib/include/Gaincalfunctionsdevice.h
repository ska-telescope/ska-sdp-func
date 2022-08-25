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
 * @file Gaincalfunctionsdevice.h
 * @author Andrew Ensor
 * @brief CUDA device functions for the Gain calibration algorithm.
 */

#ifndef GAINCAL_FUNCTIONS_DEVICE_H
#define GAINCAL_FUNCTIONS_DEVICE_H

#include <stdint.h>
#include <cuda_fp16.h>

#define half __half
#define half2 __half2

/*****************************************************************************
 * @brief Resets the gain for each receiver to default 1+0i
 *
 * Parallelised so each CUDA thread resets a single receiver gain
 * @param gains complex gain for each receiver
 * @param num_receivers total number of receivers

 *****************************************************************************/
template<typename PRECISION2>
__global__ void reset_gains
    (
    PRECISION2 *gains, // complex gain for each receiver
    const unsigned int num_receivers // total number of receivers
    );


/*****************************************************************************
 * @brief Updates the gain calibrations though one gain calibration cycle
 *
 * Note it is presumed there are num_baselines measured and predicted visibilities
 * Parallelised so each CUDA thread handles a single baseline/visibility
 * @param vis_measured_array In array of measured visibilities
 * @param vis_predicted_array In array of preducted visibilities
 * @param gains_array Out array of calculated complex gains
 * @param receiver_pairs In array giving receiver pair for each baseline
 * @param A_array Inout 4N*4N Jacobian^transpose * Jacobian matrix 
 * @param Q_array Inout 4N*1 Jacobian^transpose * Residuals matrix
 * @param num_recievers Number of receivers
 * @param num_baselines Number of baselines
 *****************************************************************************/
template<typename VIS_PRECISION2, typename PRECISION2, typename PRECISION>
__global__ void update_gain_calibration
    (
    const VIS_PRECISION2 *vis_measured_array, // in array of measured visibilities
    const VIS_PRECISION2 *vis_predicted_array, // in array of preducted visibilities
    const PRECISION2 *gains_array, // out array of calculated complex gains
    const uint2 *receiver_pairs, // in array giving receiver pair for each baseline
    PRECISION *A_array, // inout 4N*4N Jacobian^transpose * Jacobian matrix 
    PRECISION *Q_array, // inout 4N*1 Jacobian^transpose * Residuals matrix
    const unsigned int num_recievers, // number of receivers
    const unsigned int num_baselines // number of baselines
    );


#endif /* include guard */