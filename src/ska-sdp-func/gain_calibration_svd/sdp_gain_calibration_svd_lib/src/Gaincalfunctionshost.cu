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

/*****************************************************************************
 * Gaincalfunctionshost.cu
 * Andrew Ensor
 * C with C++ templates/CUDA host functions for the Gain calibration algorithm
 *****************************************************************************/

#include "Gaincalfunctionshost.h"

/*****************************************************************************
 * Checks the Cuda status and logs any errors
 *****************************************************************************/
cudaError_t checkCudaStatus()
{
    cudaError_t errorSynch = cudaGetLastError();
    cudaError_t errorAsync = cudaDeviceSynchronize(); // blocks host thread until all previously issued CUDA commands have completed
    if (errorSynch != cudaSuccess)
        logger(LOG_ERR, "Cuda synchronous error %s", cudaGetErrorString(errorSynch));
    if (errorAsync != cudaSuccess)
        logger(LOG_ERR, "Cuda asynchronous error %s", cudaGetErrorString(errorAsync));
    if (errorSynch != cudaSuccess)
        return errorSynch;
    else
        return errorAsync;
}

/*****************************************************************************
 * Checks the Cuda status return value from a cuda call
 *****************************************************************************/
void check_cuda_error_aux(const char *file, unsigned line, const char *statement, cudaError_t err)
{
    if (err != cudaSuccess)
    {
        printf("Cuda error %s returned %s at %s : %u\n", statement, cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}


/*****************************************************************************
 * Gain calibration function that allocates the Jacobian SVD matrices on device
 * Note should be paired with a later call to free_jacobian_svd_matrices
 *****************************************************************************/
template<typename PRECISION>
Jacobian_SVD_matrices<PRECISION> allocate_jacobian_svd_matrices
    (
    const unsigned int num_receivers // number of receivers
    )
{
    Jacobian_SVD_matrices<PRECISION> jacobian_svd_matrices;

    PRECISION *jacobtjacob;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&jacobtjacob, 2*num_receivers*2*num_receivers*sizeof(PRECISION)));
    PRECISION *jacobtresidual;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&jacobtresidual, 2*num_receivers*sizeof(PRECISION)));
    PRECISION *diagonalS;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&diagonalS, 2*num_receivers*sizeof(PRECISION)));
    PRECISION *unitaryU;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&unitaryU, 2*num_receivers*2*num_receivers*sizeof(PRECISION)));
    PRECISION *unitaryV;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&unitaryV, 2*num_receivers*2*num_receivers*sizeof(PRECISION)));
    PRECISION *productSUJR;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&productSUJR, 2*num_receivers*sizeof(PRECISION)));

    jacobian_svd_matrices.jacobtjacob = jacobtjacob;
    jacobian_svd_matrices.jacobtresidual = jacobtresidual;
    jacobian_svd_matrices.diagonalS = diagonalS;
    jacobian_svd_matrices.unitaryU = unitaryU;
    jacobian_svd_matrices.unitaryV = unitaryV;
    jacobian_svd_matrices.productSUJR = productSUJR;
    return jacobian_svd_matrices;
}

template Jacobian_SVD_matrices<float> allocate_jacobian_svd_matrices<float>(const unsigned int);
template Jacobian_SVD_matrices<double> allocate_jacobian_svd_matrices<double>(const unsigned int);


/*****************************************************************************
 * Msmfs function that deallocates shape configurations on device, frees each pointer in the parameter struct
 * Note should be paired with an earlier call to allocate_shape_configurations
 *****************************************************************************/
template<typename PRECISION>
void free_jacobian_svd_matrices(Jacobian_SVD_matrices<PRECISION> jacobian_svd_matrices)
{
    CUDA_CHECK_RETURN(cudaFree(jacobian_svd_matrices.productSUJR));
    CUDA_CHECK_RETURN(cudaFree(jacobian_svd_matrices.unitaryV));
    CUDA_CHECK_RETURN(cudaFree(jacobian_svd_matrices.unitaryU));
    CUDA_CHECK_RETURN(cudaFree(jacobian_svd_matrices.diagonalS));
    CUDA_CHECK_RETURN(cudaFree(jacobian_svd_matrices.jacobtresidual));
    CUDA_CHECK_RETURN(cudaFree(jacobian_svd_matrices.jacobtjacob));
}

template void free_jacobian_svd_matrices<float>(Jacobian_SVD_matrices<float>);
template void free_jacobian_svd_matrices<double>(Jacobian_SVD_matrices<double>);


/*****************************************************************************
 * Calculates the Cuda configs for block size in 1D, and number of available cuda threads
 *****************************************************************************/
void calculate_cuda_configs(int *cuda_block_size, int *cuda_num_threads)
{
    // calculate suitable cuda blockSize and cuda gridSize
    // following https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
    int min_cuda_grid_size;
    CUDA_CHECK_RETURN(cudaOccupancyMaxPotentialBlockSize(&min_cuda_grid_size, cuda_block_size,
        update_gain_calibration<float2, float2, float>, 0, 0)); // NOTE THIS HAS NOT BEEN MADE TEMPLATE PRECISION
    logger(LOG_DEBUG,
        "Cuda minimum 1D grid size available is %d"
        " and using 1D block size %d",
        min_cuda_grid_size, *cuda_block_size);

    // calculate the number of available cuda threads for performing any grid-strided kernels
    int cuda_device;
    CUDA_CHECK_RETURN(cudaGetDevice(&cuda_device));
    int cuda_num_multiprocessors, cuda_warp_size;
    CUDA_CHECK_RETURN(cudaDeviceGetAttribute(&cuda_num_multiprocessors, cudaDevAttrMultiProcessorCount, cuda_device));
    CUDA_CHECK_RETURN(cudaDeviceGetAttribute(&cuda_warp_size, cudaDevAttrWarpSize, cuda_device));
    *cuda_num_threads = cuda_num_multiprocessors*cuda_warp_size;
    logger(LOG_DEBUG,
        "Cuda device has %d multiprocessors and %d threads per warp",
        cuda_num_multiprocessors, cuda_warp_size);
    checkCudaStatus();
}


/*****************************************************************************
 * Gain calibration functions that uses template specialisations for handling cuSolver routines
 * separately in float and double precision
 *****************************************************************************/
template<typename PRECISION> cusolverStatus_t CUSOLVERDNGESVD_BUFFERSIZE
    (cusolverDnHandle_t handle, int m, int n, int *Lwork)
{
    PRECISION::unimplemented_function; // only allow specialisations of this function 
}
template<> cusolverStatus_t CUSOLVERDNGESVD_BUFFERSIZE<float>
    (cusolverDnHandle_t handle, int m, int n, int *Lwork)
{
    return cusolverDnSgesvd_bufferSize(handle, m, n, Lwork);
}
template<> cusolverStatus_t CUSOLVERDNGESVD_BUFFERSIZE<double>
    (cusolverDnHandle_t handle, int m, int n, int *Lwork)
{
    return cusolverDnDgesvd_bufferSize(handle, m, n, Lwork);
}

template<typename PRECISION> cusolverStatus_t CUSOLVERDNGESVD
    (cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, PRECISION *A, int lda,
    PRECISION *S, PRECISION *U, int ldu, PRECISION *VT, int ldvt, PRECISION *work, int lwork, PRECISION *rwork, int *devInfo)
{
    PRECISION::unimplemented_function; // only allow specialisations of this function 
}
template<> cusolverStatus_t CUSOLVERDNGESVD<float>
    (cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, float *A, int lda,
    float *S, float *U, int ldu, float *VT, int ldvt, float *work, int lwork, float *rwork, int *devInfo)
{
    return cusolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, devInfo);
}
template<> cusolverStatus_t CUSOLVERDNGESVD<double>
    (cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, double *A, int lda,
    double *S, double *U, int ldu, double *VT, int ldvt, double *work, int lwork, double *rwork, int *devInfo)
{
    return cusolverDnDgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, devInfo);
}


/*****************************************************************************
 * Gain calibration function that performs the gain calibration using an SVD solver
 *****************************************************************************/
template<typename VIS_PRECISION2, typename PRECISION2, typename PRECISION>
void perform_gain_calibration
    (
    const VIS_PRECISION2 *vis_measured_device, // input array of measured visibilities
    const VIS_PRECISION2 *vis_predicted_device, // input array of preducted visibilities
    const uint2 *receiver_pairs_device, // input array giving receiver pair for each baseline
    const unsigned int num_receivers, // number of receivers
    const unsigned int num_baselines, // number of baselines
    const unsigned int max_calibration_cycles, // maximum number of calibration cycles to perform
    Jacobian_SVD_matrices<PRECISION> jacobian_svd_matrices, // preallocated Jacobian SVD matrices
    const PRECISION2 *gains_device, // output array of calculated complex gains
    bool check_cusolver_info, // whether to explicitly check for cusolver errors during SVD
    int cuda_block_size
    )
{
    // set up timer
    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);
    float execution_time = 0; // milliseconds
    float summed_time = 0; // milliseconds

    int cuda_grid_num_baselines = (num_baselines+cuda_block_size-1)/cuda_block_size;
    int cuda_grid_num_columns = (2*num_receivers+cuda_block_size-1)/cuda_block_size;
    int cuda_grid_num_receivers = (num_receivers+cuda_block_size-1)/cuda_block_size;

    // note cusolver uses column-major matrices rather than the row-major matrices used here but
    // as all the matrices (and inverses) are symmetric they do not need to be rearranged
    cusolverDnHandle_t cusolver = NULL;
    cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolver);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
    {
        logger(LOG_CRIT, "Unable to create a cuSolver handle");
    }
    // allocate buffer space on device for cusolver gesvd working
    int space_required = 0;
    cusolver_status = CUSOLVERDNGESVD_BUFFERSIZE<PRECISION>(cusolver, 2*num_receivers, 2*num_receivers, &space_required);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
    {
        logger(LOG_CRIT, "Unable to determine space required for cusolver SVD");
    }
    PRECISION *working_space_device;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&working_space_device, space_required*sizeof(PRECISION)));
    // allocate int for device info on the device
    int *cusolver_info_device;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&cusolver_info_device, sizeof(int)));
    CUDA_CHECK_RETURN(cudaMemset(cusolver_info_device, 0, sizeof(int))); // clear the device info to zero

    // perform the gain calibration cycles
    for (unsigned int calibration_cycle=0; calibration_cycle<max_calibration_cycles; calibration_cycle++)
    {
        logger(LOG_INFO, "Performing calibration cycle %u", calibration_cycle);

        // clear the Jacobian SVD matrices for this calibration iteration
        CUDA_CHECK_RETURN(cudaMemset(jacobian_svd_matrices.jacobtjacob, 0, 2*num_receivers*2*num_receivers*sizeof(PRECISION)));
        CUDA_CHECK_RETURN(cudaMemset(jacobian_svd_matrices.jacobtresidual, 0, 2*num_receivers*sizeof(PRECISION)));
        CUDA_CHECK_RETURN(cudaMemset(jacobian_svd_matrices.diagonalS, 0, 2*num_receivers*sizeof(PRECISION)));
        CUDA_CHECK_RETURN(cudaMemset(jacobian_svd_matrices.unitaryU, 0, 2*num_receivers*2*num_receivers*sizeof(PRECISION)));
        CUDA_CHECK_RETURN(cudaMemset(jacobian_svd_matrices.unitaryV, 0, 2*num_receivers*2*num_receivers*sizeof(PRECISION)));
        CUDA_CHECK_RETURN(cudaMemset(jacobian_svd_matrices.productSUJR, 0, 2*num_receivers*sizeof(PRECISION)));

        CUDA_CHECK_RETURN(cudaMemset(working_space_device, 0, space_required*sizeof(PRECISION)));

        update_gain_calibration<<<cuda_grid_num_baselines, cuda_block_size>>>
            (vis_measured_device, vis_predicted_device, gains_device, receiver_pairs_device,
            jacobian_svd_matrices.jacobtjacob, jacobian_svd_matrices.jacobtresidual,
            num_receivers, num_baselines);

        checkCudaStatus();

        // perform SVD on jacobtjacob
        cusolver_status = CUSOLVERDNGESVD<PRECISION>(cusolver, 'A', 'A', 2*num_receivers, 2*num_receivers,
            jacobian_svd_matrices.jacobtjacob, 2*num_receivers,
            jacobian_svd_matrices.diagonalS, jacobian_svd_matrices.unitaryU,
            2*num_receivers, jacobian_svd_matrices.unitaryV, 2*num_receivers,
            working_space_device, space_required, NULL, cusolver_info_device);
        if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
        {
            logger(LOG_CRIT, "SVD unsuccessful for gain calibration cycle %u", calibration_cycle);
        }
        if (check_cusolver_info)
        {
            // copy the cusolver device info value back to host to optionally check for errors
            int cusolver_info_host = 0;
            CUDA_CHECK_RETURN(cudaMemcpy(&cusolver_info_host, cusolver_info_device, sizeof(int), cudaMemcpyDeviceToHost));
            if (cusolver_info_host != 0)
            {
                logger(LOG_WARNING, "SVD for gain calibration cycle %u returned device info %d",
                    calibration_cycle, cusolver_info_host);
            }
        }
        if (cusolver_status == CUSOLVER_STATUS_SUCCESS)
        {
            calculate_product_sujr<<<cuda_grid_num_columns, cuda_block_size>>>
                (jacobian_svd_matrices.diagonalS, jacobian_svd_matrices.unitaryU,
                jacobian_svd_matrices.jacobtresidual, jacobian_svd_matrices.productSUJR,
                2*num_receivers);

            checkCudaStatus();

            calculate_delta_update_gains<<<cuda_grid_num_receivers, cuda_block_size>>>
                (jacobian_svd_matrices.unitaryV, jacobian_svd_matrices.productSUJR,
                gains_device, num_receivers, 2*num_receivers);

            checkCudaStatus();
        }

        cudaEventRecord(stop_time);
        cudaEventSynchronize(stop_time);
        cudaEventElapsedTime(&execution_time, start_time, stop_time);
        logger(LOG_DEBUG, "Execution time for calibration cycle %d is %.3fms", calibration_cycle, execution_time);
        summed_time += execution_time;

        checkCudaStatus();
    }
    CUDA_CHECK_RETURN(cudaFree(cusolver_info_device));
    CUDA_CHECK_RETURN(cudaFree(working_space_device)); // discard working_space_device as no longer needed 
    if (cusolver != NULL)
        cusolverDnDestroy(cusolver);
}

template void perform_gain_calibration<half2, float2, float>
    (const half2*, const half2*, const uint2*, const unsigned int, const unsigned int, const unsigned int, Jacobian_SVD_matrices<float>, const float2*, bool, int);
template void perform_gain_calibration<float2, float2, float>
    (const float2*, const float2*, const uint2*, const unsigned int, const unsigned int, const unsigned int, Jacobian_SVD_matrices<float>, const float2*, bool, int);
template void perform_gain_calibration<float2, double2, double>
    (const float2*, const float2*, const uint2*, const unsigned int, const unsigned int, const unsigned int, Jacobian_SVD_matrices<double>, const double2*, bool, int);

