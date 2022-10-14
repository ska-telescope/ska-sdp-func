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
 * @file Gaincalfunctionshost.h
 * @author Andrew Ensor
 * @brief C with C++ templates/CUDA device functions for the Gain calibration algorithm.
 */

#ifndef GAINCAL_FUNCTIONS_HOST_H
#define GAINCAL_FUNCTIONS_HOST_H

#include <stdint.h>
#include <cusolverDn.h>

#include "Gaincallogger.h"
#include "Gaincalfunctionsdevice.h"

#define CUDA_CHECK_RETURN(value) check_cuda_error_aux(__FILE__, __LINE__, #value, value)

/**
 * Checks the Cuda status and logs any errors
 * @return any synchronous or asynchronous cuda error.
 */
cudaError_t checkCudaStatus();

/**
 * @brief Checks the Cuda status return value from a cuda call.
 * @param file File in which cuda error status reported.
 * @param line Line in file in which cuda error status reported.
 * @param statement Statement which resulted in the cuda error status.
 * @param err Cuda error.
 */
void check_cuda_error_aux(const char *file, unsigned line, const char *statement, cudaError_t err);

/**
 * @brief Calculates the Cuda configs for block size in 1D and 2D, and number of available cuda threads.
 * @param cuda_block_size Output cuda block size for one-dimensional kernels giving number of cuda threads per block.
 * @param cuda_num_threads Output number of cuda threads available which is used for grid-strided kernel.
 */
void calculate_cuda_configs(int *cuda_block_size, int *cuda_num_threads);

/**
 * @struct Jacobian_SVD_matrices
 * @brief Configuration struct of arrays on device for each the matrices required for solving J^T*J*Delta=J^T*R.
 * @var Jacobian_matrices::jacobtjacob
 * Member 'jacobtjacob' 2Nx2N Jacobian^transpose * Jacobian matrix.
 * @var Jacobian_matrices::jacobtresidual
 * Member 'jacobtresidual' 2Nx1 Jacobian^transpose * Residuals matrix.
 * @var Jacobian_matrices::diagonalS
 * Member 'diagonalS' 2N array for the diagonal matrix S in the SVD of Jacobian^transpose * Jacobian matrix.
 * @var Jacobian_matrices::unitaryU
 * Member 'unitaryU' 2Nx2N array for the unitary matrix U in the SVD of Jacobian^transpose * Jacobian matrix.
 * @var Jacobian_matrices::unitaryV
 * Member 'unitaryV' 2Nx2N array for the unitary matrix V in the SVD of Jacobian^transpose * Jacobian matrix.
 * @var Jacobian_matrices::diagonalSUQ
 * Member 'productSUJR' 2Nx1 array for the product of diagonalS * unitaryU * jacobtresidual.
 */
template<typename PRECISION>
struct Jacobian_SVD_matrices
{
    PRECISION *jacobtjacob;
    PRECISION *jacobtresidual;
    PRECISION *diagonalS;
    PRECISION *unitaryU;
    PRECISION *unitaryV;
    PRECISION *productSUJR;
};

/**
 * @brief Gain calibration function that allocates the Jacobian SVD matrices on device.
 *
 * Note should be paired with a later call to free_jacobian_svd_matrices.
 * @param num_receivers Number of receivers.
 */
template<typename PRECISION>
Jacobian_SVD_matrices<PRECISION> allocate_jacobian_svd_matrices
    (
    const unsigned int num_receivers // number of receivers
    );

/**
 * @brief Gain calibration function that deallocates the Jacobian SVD matrices on device, frees each pointer in the parameter struct.
 * 
 * Note should be paired with an earlier call to allocate_jacobian_svd_matrices.
 * @param jacobian_svd_matrices Jacobian SVD matrices that will be freed.
 */
template<typename PRECISION>
void free_jacobian_svd_matrices(Jacobian_SVD_matrices<PRECISION> jacobian_svd_matrices);

/**
 * @brief Gain calibration function that performs the gain calibration using an SVD solver.
 *
 * @param vis_measured_device Input array of measured visibilities.
 * @param vis_predicted_device Input array of preducted visibilities.
 * @param receiver_pairs_device Input array giving receiver pair for each baseline.
 * @param num_receivers Number of receivers.
 * @param num_baselines Number of baselines.
 * @param max_calibration_cycles Maximum number of calibration cycles to perform.
 * @param jacobian_svd_matrices Input preallocated Jacobian SVD matrices.
 * @param gains_device Output array of calculated complex gains.
 * @param check_cusolver_info Whether to explicitly check for cusolver errors during SVD.
 * @param cuda_block_size Cuda block size.
 */
template<typename VIS_PRECISION2, typename PRECISION2, typename PRECISION>
void perform_gain_calibration
    (
    const VIS_PRECISION2 *vis_measured_device, // input array of measured visibilities
    const VIS_PRECISION2 *vis_predicted_device, // input array of preducted visibilities
    const uint2 *receiver_pairs_device, // input array giving receiver pair for each baseline
    const unsigned int num_receivers, // number of receivers
    const unsigned int num_baselines, // number of baselines
    const unsigned int max_calibration_cycles, // maximum number of calibration cycles to perform
    Jacobian_SVD_matrices<PRECISION> jacobian_svd_matrices, // input preallocated Jacobian SVD matrices
    PRECISION2 *gains_device, // output array of calculated complex gains
    bool check_cusolver_info, // whether to explicitly check for cusolver errors during SVD
    int cuda_block_size
    );


#endif /* include guard */