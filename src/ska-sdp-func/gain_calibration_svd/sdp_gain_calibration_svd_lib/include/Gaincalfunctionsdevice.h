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
 * @param gains Output complex gain for each receiver
 * @param num_receivers total number of receivers

 *****************************************************************************/
template<typename PRECISION2>
__global__ void reset_gains
    (
    PRECISION2 *gains,
    const unsigned int num_receivers
    );


/*****************************************************************************
 * @brief Updates the gain calibrations though one gain calibration cycle
 *
 * Note it is presumed there are num_baselines measured and predicted visibilities
 * Parallelised so each CUDA thread handles a single baseline/visibility
 * @param vis_measured_device Input array of measured visibilities
 * @param vis_predicted_device Input array of preducted visibilities
 * @param gains_device Output array of calculated complex gains
 * @param receiver_pairs_device Input array giving receiver pair for each baseline
 * @param jacobtjacob Inout 4N*4N Jacobian^transpose * Jacobian matrix 
 * @param jacobtresidual Inout 4N*1 Jacobian^transpose * Residuals matrix
 * @param num_recievers Number of receivers
 * @param num_baselines Number of baselines
 *****************************************************************************/
template<typename VIS_PRECISION2, typename PRECISION2, typename PRECISION>
__global__ void update_gain_calibration
    (
    const VIS_PRECISION2 *vis_measured_device,
    const VIS_PRECISION2 *vis_predicted_device,
    const PRECISION2 *gains_device,
    const uint2 *receiver_pairs_device,
    PRECISION *jacobtjacob,
    PRECISION *jacobtresidual,
    const unsigned int num_recievers,
    const unsigned int num_baselines
    );


/*****************************************************************************
 * @brief Calculates the matrix product diagonalS inverse * unitaryU transpose * jacobtresidual
 *
 * Parallelised so each CUDA thread handles one of the 2N entries in the product
 * where N is the number of receivers
 * @param diagonalS Input 2N*1 array for the diagonal matrix S in the SVD of A
 * @param unitaryU Input 2N*2N array for the unitary matrix U in the SVD of A
 * @param jacobtresidual Input 2Nx1 array Q for the 4N*1 Jacobian^transpose * Residuals matrix
 * @param diagonalSUQ Output 2Nx1 array for the resulting SUQ product
 * @param num_entries Total number 2N of entries in the matrix product
 *****************************************************************************/
template<typename PRECISION>
__global__ void calculate_suq_product
    (
    const PRECISION *diagonalS,
    const PRECISION *unitaryU,
    const PRECISION *jacobtresidual,
    PRECISION *diagonalSUQ,
    const unsigned int num_entries
    );


/*****************************************************************************
 * @brief Calculates the matrix Delta as the product of unitaryV transpose * diagonalSUQ
 * and adds it to the gains
 *
 * Parallelised so each CUDA thread handles the two gains for one of the N antennas
 * @param unitaryV Input array for the unitary V matrix in the SVD of A
 * @param diagonalSUQ Input 2Nx1 array for the SUQ matrix product
 * @param gains_device Inout array of antenna gains
 * @param num_recievers Number N of receivers
 * @param num_entries Total number 2N of gain entries
 *****************************************************************************/
template<typename PRECISION, typename PRECISION2>
__global__ void calculate_delta_update_gains
    (
    const PRECISION *unitaryV, // input array for the unitary V matrix in the SVD of A
    const PRECISION *diagonalSUQ, // input 2Nx1 array for the SUQ matrix product
    PRECISION2 *gains_device, // inout array of antenna gains
    const unsigned int num_recievers, // number N of receivers
    const unsigned int num_entries // total number 2N of gain entries
    );


#endif /* include guard */