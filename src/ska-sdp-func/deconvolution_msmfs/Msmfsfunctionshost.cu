// Copyright 2021 High Performance Computing Research Laboratory, Auckland University of Technology (AUT)

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
 * Msmfsfunctionshost.cu
 * Andrew Ensor
 * C with C++ templates/CUDA host functions for the MSMFS cleaning algorithm
 *****************************************************************************/

#include "Msmfsfunctionshost.h"

/*****************************************************************************
 * Checks the Cuda status and logs any errors
 *****************************************************************************/
cudaError_t checkCudaStatus()
{
    cudaError_t errorSynch = cudaGetLastError();
    cudaError_t errorAsync = cudaDeviceSynchronize(); // blocks host thread until all previously issued CUDA commands have completed
    if (errorSynch != cudaSuccess)
        sdp_logger(LOG_ERR, "Cuda synchronous error %s", cudaGetErrorString(errorSynch));
    if (errorAsync != cudaSuccess)
        sdp_logger(LOG_ERR, "Cuda asynchronous error %s", cudaGetErrorString(errorAsync));
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
 * Calculates the Cuda configs for block size in 1D and 2D, and number of available cuda threads
 *****************************************************************************/
void calculate_cuda_configs(int *cuda_block_size, dim3 *cuda_block_size_2D, int *cuda_num_threads)
{
    // calculate suitable cuda blockSize and cuda gridSize
    // following https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
    int min_cuda_grid_size;
    CUDA_CHECK_RETURN(cudaOccupancyMaxPotentialBlockSize(&min_cuda_grid_size, cuda_block_size,
        calculate_principal_solution_at_peak<float>, 0, 0)); // NOTE THIS HAS NOT BEEN MADE TEMPLATE PRECISION
    int cuda_block_size_sqrt = floor(sqrt(*cuda_block_size));
    (*cuda_block_size_2D).x = cuda_block_size_sqrt;
    (*cuda_block_size_2D).y = cuda_block_size_sqrt;
    (*cuda_block_size_2D).z = 1;
    sdp_logger(LOG_DEBUG,
        "Cuda minimum 1D grid size available is %d"
        " and using 1D block size %d and 2D block size (%d,%d)",
        min_cuda_grid_size, *cuda_block_size,
        (*cuda_block_size_2D).x, (*cuda_block_size_2D).y);

    // calculate the number of available cuda threads for performing any grid-strided kernels
    int cuda_device;
    CUDA_CHECK_RETURN(cudaGetDevice(&cuda_device));
    int cuda_num_multiprocessors, cuda_warp_size;
    CUDA_CHECK_RETURN(cudaDeviceGetAttribute(&cuda_num_multiprocessors, cudaDevAttrMultiProcessorCount, cuda_device));
    CUDA_CHECK_RETURN(cudaDeviceGetAttribute(&cuda_warp_size, cudaDevAttrWarpSize, cuda_device));
    *cuda_num_threads = cuda_num_multiprocessors*cuda_warp_size;
    sdp_logger(LOG_DEBUG,
        "Cuda device has %d multiprocessors and %d threads per warp",
        cuda_num_multiprocessors, cuda_warp_size);
    checkCudaStatus();
}


/*****************************************************************************
 * Msmfs function that allocates and prepares the shape configurations for cleaning
 * by taking the variances at each clean scale and calculating the
 * scale bias to use at that scale, the convolution support to use at that scale
 * and the doubly convolved support to use for each combination of scales
 * Note should be paired with a later call to free_shape_configurations
 *****************************************************************************/
template<typename PRECISION>
Gaussian_shape_configurations<PRECISION> allocate_shape_configurations
    (
    PRECISION *variances_host, // input array of shape variances, if NULL then variances calculated to be 0, 1, 4, 16, 64, ...
    const unsigned int num_scales, // number of scales to use in msmfs cleaning
    const PRECISION convolution_accuracy, // fraction of peak accuracy used to determine supports for convolution kernels
    const PRECISION scale_bias_factor // bias multiplicative factor to favour cleaning with smaller scales
    )
{
    Gaussian_shape_configurations<PRECISION> shape_configs;

    PRECISION *variances_device;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&variances_device, num_scales*sizeof(PRECISION)));
    PRECISION *scale_bias_device;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&scale_bias_device, num_scales*sizeof(PRECISION)));
    unsigned int *convolution_support_device;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&convolution_support_device, num_scales*sizeof(unsigned int)));
    unsigned int *double_convolution_support_device;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&double_convolution_support_device, num_scales*num_scales*sizeof(unsigned int)));

    const bool use_default_variances = (variances_host==NULL); // whether variances have been specified as parameter
    if (use_default_variances)
    {
        CUDA_CHECK_RETURN(cudaHostAlloc(&variances_host, num_scales*sizeof(PRECISION), 0));
    }
    memset(variances_host, 0, num_scales*sizeof(PRECISION));
    variances_host[0] = (PRECISION)0.0; // point cleaning at first scale
    // find the largest variance which is used in calculating the scale bias
    PRECISION variances_max = variances_host[0];
    PRECISION next_scale_stdev = (PRECISION)1.0;
    for (unsigned int scale_index=1; scale_index<num_scales; scale_index++)
    {
        variances_host[scale_index] = next_scale_stdev*next_scale_stdev;
        next_scale_stdev *= (PRECISION)2.0; // double standard deviation of each scale from previous
        if (variances_host[scale_index] > variances_max)
           variances_max = variances_host[scale_index];
    }
    // copy the variances to the device
    CUDA_CHECK_RETURN(cudaMemset(variances_device, 0, num_scales*sizeof(PRECISION)));
    CUDA_CHECK_RETURN(cudaMemcpy(variances_device, variances_host, num_scales*sizeof(PRECISION), cudaMemcpyHostToDevice));

    // create the scale multiplicative bias for each scale to favour cleaning with smaller scales
    PRECISION *scale_bias_host;
    CUDA_CHECK_RETURN(cudaHostAlloc(&scale_bias_host, num_scales*sizeof(PRECISION), 0));
    memset(scale_bias_host, 0, num_scales*sizeof(PRECISION));
    for (unsigned int scale_index=0; scale_index<num_scales; scale_index++)
    {
        // calculate scale_bias for this scale depending on a fraction of the support at the largest scale
        // note the support is proportional to the square root of the variance for a gaussian kernel
        scale_bias_host[scale_index] = (PRECISION)1.0
            - scale_bias_factor*sqrt(variances_host[scale_index]/variances_max);
        sdp_logger(LOG_DEBUG,
            "Kernel index %u has variance %f and scale bias %f",
            scale_index, variances_host[scale_index], scale_bias_host[scale_index]);
    }
    // copy the scale biases to the device
    CUDA_CHECK_RETURN(cudaMemset(scale_bias_device, 0, num_scales*sizeof(PRECISION)));
    CUDA_CHECK_RETURN(cudaMemcpy(scale_bias_device, scale_bias_host, num_scales*sizeof(PRECISION), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaFreeHost(scale_bias_host)); scale_bias_host = NULL;

    // calculate the convolution (half) supports for single and for double convolutions
    unsigned int *convolution_support_host; // array containing the supports required for each gaussian kernel shape for convolution
    CUDA_CHECK_RETURN(cudaHostAlloc(&convolution_support_host, num_scales*sizeof(unsigned int), 0));
    memset(convolution_support_host, 0, num_scales*sizeof(unsigned int));
    unsigned int *double_convolution_support_host; // array containing the supports required for each gaussian kernel shape for double convolution
    CUDA_CHECK_RETURN(cudaHostAlloc(&double_convolution_support_host, num_scales*num_scales*sizeof(unsigned int), 0)); // note actually only half of entries are unique
    memset(convolution_support_host, 0, num_scales*num_scales*sizeof(unsigned int));
    for (unsigned int scale_index=0; scale_index<num_scales; scale_index++)
    {
        convolution_support_host[scale_index] = (unsigned int)floor(sqrt(-2*variances_host[scale_index]*log(convolution_accuracy)));        
        sdp_logger(LOG_DEBUG,
            "Kernel index %u will use convolution support %u",
            scale_index, convolution_support_host[scale_index]);
        for (unsigned int scale_index2=0; scale_index2<num_scales; scale_index2++)
        {
            double_convolution_support_host[num_scales*scale_index+scale_index2]
                = (unsigned int)floor(sqrt(-2*(variances_host[scale_index]+variances_host[scale_index2])*log(convolution_accuracy)));        
        }
    }
    // copy the convolution (half) supports to the device
    CUDA_CHECK_RETURN(cudaMemset(convolution_support_device, 0, num_scales*sizeof(unsigned int)));
    CUDA_CHECK_RETURN(cudaMemcpy(convolution_support_device, convolution_support_host, num_scales*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemset(double_convolution_support_device, 0, num_scales*num_scales*sizeof(unsigned int)));
    CUDA_CHECK_RETURN(cudaMemcpy(double_convolution_support_device, double_convolution_support_host, num_scales*num_scales*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaFreeHost(double_convolution_support_host)); double_convolution_support_host = NULL;
    CUDA_CHECK_RETURN(cudaFreeHost(convolution_support_host)); convolution_support_host = NULL;
    if (use_default_variances)
    {
        CUDA_CHECK_RETURN(cudaFreeHost(variances_host)); variances_host = NULL;
    }

    shape_configs.variances_device = variances_device;
    shape_configs.scale_bias_device = scale_bias_device;
    shape_configs.convolution_support_device = convolution_support_device;
    shape_configs.double_convolution_support_device = double_convolution_support_device;
    return shape_configs;
}

template Gaussian_shape_configurations<float> allocate_shape_configurations<float>(float*, const unsigned int, const float, const float);
template Gaussian_shape_configurations<double> allocate_shape_configurations<double>(double*, const unsigned int, const double, const double);


/*****************************************************************************
 * Msmfs function that deallocates shape configurations, frees each pointer in the parameter struct
 * Note should be paired with an earlier call to allocate_shape_configurations
 *****************************************************************************/
template<typename PRECISION>
void free_shape_configurations(Gaussian_shape_configurations<PRECISION> shape_configs)
{
    CUDA_CHECK_RETURN(cudaFree(shape_configs.double_convolution_support_device));
    CUDA_CHECK_RETURN(cudaFree(shape_configs.convolution_support_device));
    CUDA_CHECK_RETURN(cudaFree(shape_configs.scale_bias_device));
    CUDA_CHECK_RETURN(cudaFree(shape_configs.variances_device));
}

template void free_shape_configurations<float>(Gaussian_shape_configurations<float>);
template void free_shape_configurations<double>(Gaussian_shape_configurations<double>);


/*****************************************************************************
 * Msmfs function that allocates and clears the data structures that will
 * hold all the scale moment residuals on the device, one for each combination
 * of scale and taylor term
 * Note should be paired with a later call to free_scale_moment_residuals
 *****************************************************************************/
template<typename PRECISION>
PRECISION* allocate_scale_moment_residuals
    (const unsigned int scale_moment_size, const unsigned int num_scales, unsigned int num_taylor)
{
    // allocate device memory for the scale moment residuals that get calculated from the dirty images
    PRECISION *scale_moment_residuals_device;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&scale_moment_residuals_device,
        scale_moment_size*scale_moment_size*num_scales*num_taylor*sizeof(PRECISION)));
    CUDA_CHECK_RETURN(cudaMemset(scale_moment_residuals_device, 0,
        scale_moment_size*scale_moment_size*num_scales*num_taylor*sizeof(PRECISION))); // clear the image to zero
    return scale_moment_residuals_device;
}

template float* allocate_scale_moment_residuals<float>(const unsigned int, const unsigned int, unsigned int);
template double* allocate_scale_moment_residuals<double>(const unsigned int, const unsigned int, unsigned int);


/*****************************************************************************
 * Temporary utility function that displays the scale moment residuals on std out
 * for all combinations of scale and taylor moment
 *****************************************************************************/
template<typename PRECISION>
void display_scale_moment_residuals
    (
    PRECISION *scale_moment_residuals_device, // flat array that holds scale moment residuals
    const unsigned int num_scales,
    const unsigned int num_taylor,
    const unsigned int scale_moment_size // one dimensional size of scale moment residuals, assumed square
    )
{
    // copy the scale moment residuals back to host
    PRECISION *scale_moment_residuals_host;
    CUDA_CHECK_RETURN(cudaHostAlloc(&scale_moment_residuals_host, scale_moment_size*scale_moment_size*num_scales*num_taylor*sizeof(PRECISION), 0));
    memset(scale_moment_residuals_host, 0, scale_moment_size*scale_moment_size*num_scales*num_taylor*sizeof(PRECISION));
    CUDA_CHECK_RETURN(cudaMemcpy(scale_moment_residuals_host, scale_moment_residuals_device, scale_moment_size*scale_moment_size*num_scales*num_taylor*sizeof(PRECISION), cudaMemcpyDeviceToHost));
    // display central region of each scale moment residual image
    for (unsigned int taylor_output=0; taylor_output<num_taylor; taylor_output++)
    {
        for (unsigned int scale_output=0; scale_output<num_scales; scale_output++)
        {
            printf("Scale moment residual initial image around centre for taylor index %u at scale index %u:\n",
                taylor_output, scale_output);
            const int image_output_least_xy = -5;
            const int image_output_largest_xy = 5;
            for (int y=image_output_least_xy; y<=image_output_largest_xy; y++)
            {
                printf("Row %5d:", y);
                for (int x=image_output_least_xy; x<=image_output_largest_xy; x++)
                {
                    int index = (y+scale_moment_size/2)*scale_moment_size + (x+scale_moment_size/2);
                    index += scale_moment_size*scale_moment_size*(num_scales*taylor_output+scale_output);
                    printf("%+9.4lf ", scale_moment_residuals_host[index]);
                }
                printf("\n");
            }
        }
    }
    CUDA_CHECK_RETURN(cudaFreeHost(scale_moment_residuals_host));
}

template void display_scale_moment_residuals<float>(float*, const unsigned int, const unsigned int, const unsigned int);
template void display_scale_moment_residuals<double>(double*, const unsigned int, const unsigned int, const unsigned int);


/*****************************************************************************
 * Msmfs function that deallocates device data structures that were used to
 * hold all the scale moment residuals on the device, one for each combination
 * of scale and taylor term
 * Note should be paired with an earlier call to allocate_scale_moment_residuals
 *****************************************************************************/
template<typename PRECISION>
void free_scale_moment_residuals(PRECISION* scale_moment_residuals_device)
{
    CUDA_CHECK_RETURN(cudaFree(scale_moment_residuals_device));
}

template void free_scale_moment_residuals<float>(float*);
template void free_scale_moment_residuals<double>(double*);


/*****************************************************************************
 * Msmfs function that calculates the scale convolutions of moment residuals on device
 * dirty_moment_images_device presumed to have size num_taylor*dirty_moment_size*dirty_moment_size
 * scale_moment_residuals_device presumed to have size num_taylor*num_scales*scale_moment_size*scale_moment_size
 * and both assumed to be centred around origin with dirty_moment_images having sufficient border for convolutions
 * variances presumed to have size num_scales
 * convolution_support_device presumed to have size num_scales
 *****************************************************************************/
template<typename PRECISION>
void calculate_scale_moment_residuals
    (
    PRECISION *dirty_moment_images_device, // flat array containing input Taylor coefficient dirty images to be convolved
    const unsigned int dirty_moment_size, // one dimensional size of image, assumed square
    const unsigned int num_taylor,
    PRECISION *scale_moment_residuals_device, // output flat array that will be populated with all convolved scale moment residuals
    const unsigned int scale_moment_size, // one dimensional size of scale moment residuals, assumed square
    const unsigned int num_scales,
    PRECISION *variances_device, // variances used for calculating gaussian kernel, one for each of num_scales
    unsigned int *convolution_support_device, // supports required for each gaussian kernel shape for convolution
    dim3 cuda_block_size_2D
    )
{
    // set up timer
    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);
    float execution_time = 0; // milliseconds
    cudaEventRecord(start_time);

    int cuda_grid_dirty_moment_size = (dirty_moment_size+cuda_block_size_2D.y-1)/cuda_block_size_2D.y;
    int cuda_grid_scale_moment_size = (scale_moment_size+cuda_block_size_2D.x-1)/cuda_block_size_2D.x;
    dim3 cuda_grid_horiz_convolve_2D(cuda_grid_scale_moment_size,cuda_grid_dirty_moment_size,1);  // create a thread per image pixel to be convolved
    dim3 cuda_grid_vert_convolve_2D(cuda_grid_scale_moment_size,cuda_grid_scale_moment_size,1);  // create a thread per image pixel to be convolved
    unsigned int image_border = (dirty_moment_size-scale_moment_size)/2;
    // create a temporary array reused on device to hold partially convolved images
    // note the partially convolved images are not square as border only trimmed on left and right sides
    PRECISION *horiz_convolved_device;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&horiz_convolved_device, scale_moment_size*dirty_moment_size*sizeof(PRECISION)));
    for (unsigned int taylor_index=0; taylor_index<num_taylor; taylor_index++)
    {
        // calculate scale convolutions for one taylor_index moment residual
        PRECISION *image_device = &dirty_moment_images_device[taylor_index*dirty_moment_size*dirty_moment_size];
        for (unsigned int scale_index=0; scale_index<num_scales; scale_index++)
        {
            // calculate one scale convolution for one taylor_index moment residual
            PRECISION *scale_moment_residual_device = &scale_moment_residuals_device[(taylor_index*num_scales+scale_index)*scale_moment_size*scale_moment_size];
            // calculate a suitable convolution support based on the requested convolution accuracy
            // perform the 2D convolution as two 1D convolutions
            convolve_horizontal<<<cuda_grid_horiz_convolve_2D, cuda_block_size_2D>>>
                (image_device, horiz_convolved_device, scale_moment_size, dirty_moment_size,
                image_border, convolution_support_device, variances_device, scale_index);
            convolve_vertical<<<cuda_grid_vert_convolve_2D, cuda_block_size_2D>>>
                (horiz_convolved_device, scale_moment_residual_device, scale_moment_size, scale_moment_size,
                image_border, convolution_support_device, variances_device, scale_index);
        }
    }
    CUDA_CHECK_RETURN(cudaFree(horiz_convolved_device));

    // stop timer
    cudaEventRecord(stop_time);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&execution_time, start_time, stop_time);
    sdp_logger(LOG_DEBUG, "Execution time for calculating scale moment residuals is %.3fms", execution_time);
}

template void calculate_scale_moment_residuals<float>(float*, const unsigned int, const unsigned int, float*, const unsigned int, const unsigned int, float*, unsigned int*, dim3);
template void calculate_scale_moment_residuals<double>(double*, const unsigned int, const unsigned int, double*, const unsigned int, const unsigned int, double*, unsigned int*, dim3);


/*****************************************************************************
 * Msmfs function that allocates and clears the data structures that will
 * hold all the inverse hessian matrices on the device, one for each scale
 * and each matrix of size num_taylor x num_taylor
 * Note should be paired with a later call to free_inverse_hessian_matrices
 *****************************************************************************/
template<typename PRECISION>
PRECISION* allocate_inverse_hessian_matrices(const unsigned int num_scales, unsigned int num_taylor)
{
    // allocate device memory for the inverse hessian matrices
    PRECISION *inverse_hessian_matrices_device;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&inverse_hessian_matrices_device, num_taylor*num_taylor*num_scales*sizeof(PRECISION)));
    CUDA_CHECK_RETURN(cudaMemset(inverse_hessian_matrices_device, 0, num_taylor*num_taylor*num_scales*sizeof(PRECISION))); // clear the matrices to zero
    return inverse_hessian_matrices_device;
}

template float* allocate_inverse_hessian_matrices<float>(const unsigned int, unsigned int);
template double* allocate_inverse_hessian_matrices<double>(const unsigned int, unsigned int);


/*****************************************************************************
 * Temporary utility function that displays entries calculated for inverse hessian matrices
 *****************************************************************************/
template<typename PRECISION>
void display_inverse_hessian_matrices
    (
    PRECISION *inverse_hessian_matrices_device,
    unsigned int num_scales,
    unsigned int num_taylor
    )
{
    // copy the inverse hessian matrix result back to host
    PRECISION *inverse_hessian_matrices_host;
    CUDA_CHECK_RETURN(cudaHostAlloc(&inverse_hessian_matrices_host, num_taylor*num_taylor*num_scales*sizeof(PRECISION), 0));
    memset(inverse_hessian_matrices_host, 0, num_taylor*num_taylor*num_scales*sizeof(PRECISION));
    CUDA_CHECK_RETURN(cudaMemcpy(inverse_hessian_matrices_host, inverse_hessian_matrices_device, num_taylor*num_taylor*num_scales*sizeof(PRECISION), cudaMemcpyDeviceToHost));
    // display entries of the inverse of each scale-dependent hessian matrices
    for (unsigned int inverse_scale_index=0; inverse_scale_index<num_scales; inverse_scale_index++)
    {
        printf("Inverse hessian matrix for scale index %d:\n", inverse_scale_index);
        for (unsigned int y=0; y<num_taylor; y++)
        {
            printf("Row %5d:", y);
            for (unsigned int x=0; x<num_taylor; x++)
            {
                int index = num_taylor*num_taylor*inverse_scale_index + y*num_taylor + x;
                printf("%+9.4lf ", inverse_hessian_matrices_host[index]);
            }
            printf("\n");
        }
    }
    CUDA_CHECK_RETURN(cudaFreeHost(inverse_hessian_matrices_host));
}

template void display_inverse_hessian_matrices<float>(float*, unsigned int, unsigned int);
template void display_inverse_hessian_matrices<double>(double*, unsigned int, unsigned int);


/*****************************************************************************
 * Msmfs function that deallocates device data structures that were used
 * hold all the inverse hessian matrices on the device, one for each scale
 * and each matrix of size num_taylor x num_taylor
 * Note should be paired with an earlier call to allocate_inverse_hessian_matrices
 *****************************************************************************/
template<typename PRECISION>
void free_inverse_hessian_matrices(PRECISION *inverse_hessian_matrices_device)
{
    CUDA_CHECK_RETURN(cudaFree(inverse_hessian_matrices_device));
}

template void free_inverse_hessian_matrices<float>(float*);
template void free_inverse_hessian_matrices<double>(double*);


/*****************************************************************************
 * Msmfs functions that uses template specialisations for handling cuSolver routines
 * separately in float and double precision
 *****************************************************************************/
template<typename PRECISION> cusolverStatus_t CUSOLVERDNGETRF_BUFFERSIZE
    (cusolverDnHandle_t handle, int m, int n, PRECISION *A, int lda, int *Lwork)
{
    PRECISION::unimplemented_function; // only allow specialisations of this function
    return CUSOLVER_STATUS_EXECUTION_FAILED;
}

template<> cusolverStatus_t CUSOLVERDNGETRF_BUFFERSIZE<float>
    (cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *Lwork)
{
    return cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}
template<> cusolverStatus_t CUSOLVERDNGETRF_BUFFERSIZE<double>
    (cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *Lwork)
{
    return cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

template<typename PRECISION> cusolverStatus_t CUSOLVERDNGETRF
    (cusolverDnHandle_t handle, int m, int n, PRECISION *A, int lda, PRECISION *workspace, int *devIpiv, int *devInfo)
{
    PRECISION::unimplemented_function; // only allow specialisations of this function 
    return CUSOLVER_STATUS_EXECUTION_FAILED;
}
template<> cusolverStatus_t CUSOLVERDNGETRF<float>
    (cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *workspace, int *devIpiv, int *devInfo)
{
    return cusolverDnSgetrf(handle, m, n, A, lda, workspace, devIpiv, devInfo);
}
template<> cusolverStatus_t CUSOLVERDNGETRF<double>
    (cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *workspace, int *devIpiv, int *devInfo)
{
    return cusolverDnDgetrf(handle, m, n, A, lda, workspace, devIpiv, devInfo);
}

template<typename PRECISION> cusolverStatus_t CUSOLVERDNGETRS
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const PRECISION *A, int lda, const int *devIpiv, PRECISION *B, int ldb, int *devInfo)
{
    PRECISION::unimplemented_function; // only allow specialisations of this function 
    return CUSOLVER_STATUS_EXECUTION_FAILED;
}
template<> cusolverStatus_t CUSOLVERDNGETRS<float>
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float *A, int lda, const int *devIpiv, float *B, int ldb, int *devInfo)
{
    return cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}
template<> cusolverStatus_t CUSOLVERDNGETRS<double>
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double *A, int lda, const int *devIpiv, double *B, int ldb, int *devInfo)
{
    return cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}


/*****************************************************************************
 * Msmfs function that calculates the inverse of each scale-dependent moment hessian matrices on device
 * psf_moment_images_device presumed to have size num_psf*psf_moment_size*psf_moment_size
 * and assumed to be centred around origin with psf_moment_images_device having sufficient size for convolution at its centre
 * and assumed that num_psf = 2*num_taylor - 1
 * inverse_hessian_matrices_device presumed to have size num_scales*num_taylor*num_taylor
 * variances presumed to have size num_scales
 * double_convolution_support_device presumed to have size num_scales*num_scales
 * Note check_cusolver_info can be turned off to avoid waiting for transfer of potential error code
 *****************************************************************************/
template<typename PRECISION>
void calculate_inverse_hessian_matrices
    (
    PRECISION *psf_moment_images_device, // flat array containing input Taylor coefficient psf images to be convolved
    const unsigned int psf_moment_size, // one dimensional size of psf, assumed square
    const unsigned int num_psf,
    PRECISION *inverse_hessian_matrices_device, // output flat array that will be populated with all inverse hessian matrices
    const unsigned int num_scales,
    const unsigned int num_taylor,
    PRECISION *variances_device, // variances used for calculating gaussian kernel, one for each of num_scales
    unsigned int *double_convolution_support_device, // supports required for each gaussian kernel shape for double convolution
    bool check_cusolver_info, // whether to explicitly check for cusolver errors during matrix inversion
    int cuda_block_size
    )
{
    // set up timer
    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);
    float execution_time = 0; // milliseconds
    cudaEventRecord(start_time);

    // allocate temporary device memory to hold entries for the hessian matrices
    PRECISION *hessian_entries_device;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&hessian_entries_device, num_scales*num_psf*sizeof(PRECISION)));
    CUDA_CHECK_RETURN(cudaMemset(hessian_entries_device, 0, num_scales*num_psf*sizeof(PRECISION))); // clear the temp array to zero

    // calculate the required entries for the scale-dependent moment hessian matrices on the device
    // entries are obtained as convolutions of all scales with the centre of each psf taylor term
    const unsigned int num_entries_calculate = num_scales*num_psf;
    int cuda_grid_num_entries_calculate = (num_entries_calculate+cuda_block_size-1)/cuda_block_size;
    calculate_hessian_entries<<<cuda_grid_num_entries_calculate, cuda_block_size>>>
        (psf_moment_images_device, psf_moment_size, num_psf,
        hessian_entries_device, num_scales,
        variances_device, double_convolution_support_device);

    // allocate device memory for the inverse hessian matrices
    PRECISION *hessian_matrices_device;
    const unsigned int num_matrix_entries = num_taylor*num_taylor*num_scales;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&hessian_matrices_device, num_matrix_entries*sizeof(PRECISION)));
    CUDA_CHECK_RETURN(cudaMemset(hessian_matrices_device, 0, num_matrix_entries*sizeof(PRECISION))); // clear the matrices to zero

    // populate each hessian matrix with the calculated hessian entries
    int cuda_grid_num_matrix_entries = (num_matrix_entries+cuda_block_size-1)/cuda_block_size;
    populate_hessian_matrices<<<cuda_grid_num_matrix_entries, cuda_block_size>>>
        (hessian_entries_device, num_scales, num_taylor, hessian_matrices_device);
    CUDA_CHECK_RETURN(cudaFree(hessian_entries_device)); // free the array of entries as hessian metrices now populated

    // create an identity matrix used when solving to find inverse of each hessian matrix
    // note this identity only depends on num_taylor and so does not actually need to be regenerated
    // each time calculate_inverse_hessian_matrices gets called
    PRECISION *identity_matrix_host;
    CUDA_CHECK_RETURN(cudaHostAlloc(&identity_matrix_host, num_taylor*num_taylor*sizeof(PRECISION), 0));
    memset(identity_matrix_host, 0, num_taylor*num_taylor*sizeof(PRECISION)); // clear the matrix to zero
    for (unsigned int i=0; i<num_taylor; i++)
    {
        unsigned int diagonal_index = num_taylor*i + i;
        identity_matrix_host[diagonal_index] = (PRECISION)1.0; // put 1 down the diagonal 
    }
    // transfer the identity matrix to the device
    PRECISION *identity_matrix_device;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&identity_matrix_device, num_taylor*num_taylor*sizeof(PRECISION)));
    CUDA_CHECK_RETURN(cudaMemcpy(identity_matrix_device, identity_matrix_host, num_taylor*num_taylor*sizeof(PRECISION), cudaMemcpyHostToDevice));
    // free identity matrix on host as no longer needed
    CUDA_CHECK_RETURN(cudaFreeHost(identity_matrix_host));

    // calculate the inverse of each hessian matrix using cuSolver dense matrix LU decomposition
    // note cusolver uses column-major matrices rather than the row-major matrices used here but
    // as all the hessian matrices (and inverses) are symmetric they do not need to be rearranged
    cusolverDnHandle_t cusolver = NULL;
    cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolver);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
    {
        sdp_logger(LOG_CRIT, "Unable to create a cuSolver handle");
    }
    // create an int[] on device to represent the row pivots for permutation matrix
    int *pivot_rows_device;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&pivot_rows_device, num_taylor*sizeof(int)));
    CUDA_CHECK_RETURN(cudaMemset(pivot_rows_device, 0, num_taylor*sizeof(int))); // clear the pivot rows to zero
    // allocate buffer space on device for cusolver getrf working
    int space_required = 0;
    CUSOLVERDNGETRF_BUFFERSIZE<PRECISION>(cusolver, num_taylor, num_taylor,
        hessian_matrices_device, num_taylor, &space_required);
    PRECISION *working_space_device;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&working_space_device, space_required*sizeof(PRECISION)));
    // allocate int for device info on the device
    int *cusolver_info_device;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&cusolver_info_device, sizeof(int)));
    CUDA_CHECK_RETURN(cudaMemset(cusolver_info_device, 0, sizeof(int))); // clear the device info to zero
    // iterate through each hessian matrix to find its inverse
    for (unsigned int hessian_index=0; hessian_index<num_scales; hessian_index++)
    {
        PRECISION *hessian_matrix_device = &hessian_matrices_device[num_taylor*num_taylor*hessian_index];
        // perform the LU factorization for the hessian matrix
        cusolver_status = CUSOLVERDNGETRF<PRECISION>(cusolver, num_taylor, num_taylor,
            hessian_matrix_device, num_taylor,
            working_space_device, pivot_rows_device, cusolver_info_device);
        if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
        {
            sdp_logger(LOG_CRIT, "LU factorization unsuccessful for hessian matrix for scale index %u", hessian_index);
        }
        if (check_cusolver_info)
        {
            // copy the cusolver device info value back to host to optionally check for errors
            int cusolver_info_host = 0;
            CUDA_CHECK_RETURN(cudaMemcpy(&cusolver_info_host, cusolver_info_device, sizeof(int), cudaMemcpyDeviceToHost));
            if (cusolver_info_host != 0)
            {
                sdp_logger(LOG_WARNING, "LU factorization for hessian matrix for scale index %u returned device info %d",
                    hessian_index, cusolver_info_host);
            }
        }
        // copy the identity matrix I into the inverse hessian matrix for solving to find inverse
        PRECISION *inverse_hessian_device = &inverse_hessian_matrices_device[num_taylor*num_taylor*hessian_index];
        CUDA_CHECK_RETURN(cudaMemcpy(inverse_hessian_device, identity_matrix_device, num_taylor*num_taylor*sizeof(PRECISION), cudaMemcpyDeviceToDevice));
        // solve to find the inverse of the hessian matrix
        cusolver_status = CUSOLVERDNGETRS<PRECISION>(cusolver, CUBLAS_OP_N, num_taylor, num_taylor,
            hessian_matrix_device, num_taylor, pivot_rows_device,
            inverse_hessian_device, num_taylor, cusolver_info_device);
        if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
        {
            sdp_logger(LOG_CRIT, "LU inversion unsuccessful for hessian matrix for scale index ", hessian_index);
        }
        if (check_cusolver_info)
        {
            // copy the cusolver device info value back to host to optionally check for errors
            int cusolver_info_host = 0;
            CUDA_CHECK_RETURN(cudaMemcpy(&cusolver_info_host, cusolver_info_device, sizeof(int), cudaMemcpyDeviceToHost));
            if (cusolver_info_host != 0)
            {
                sdp_logger(LOG_WARNING, "LU inversion for hessian matrix for scale index %u returned device info %d",
                    hessian_index, cusolver_info_host);
            }
        }
    }
    // clean up resources used during cusolve
    CUDA_CHECK_RETURN(cudaFree(cusolver_info_device));
    CUDA_CHECK_RETURN(cudaFree(working_space_device));
    CUDA_CHECK_RETURN(cudaFree(pivot_rows_device));
    if (cusolver != NULL)
        cusolverDnDestroy(cusolver);

    // free identity matrix on device as no longer needed
    CUDA_CHECK_RETURN(cudaFree(identity_matrix_device));

    CUDA_CHECK_RETURN(cudaFree(hessian_matrices_device)); // discard hessian matrices as now have found their inverses

    // stop timer
    cudaEventRecord(stop_time);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&execution_time, start_time, stop_time);
    sdp_logger(LOG_DEBUG, "Execution time for calculating inverse hessian matrices is %.3fms", execution_time);

}

template void calculate_inverse_hessian_matrices<float>(float*, const unsigned int, const unsigned int, float*, const unsigned int, const unsigned int, float*, unsigned int*, bool, int);
template void calculate_inverse_hessian_matrices<double>(double*, const unsigned int, const unsigned int, double*, const unsigned int, const unsigned int, double*, unsigned int*, bool, int);


/*****************************************************************************
 * Msmfs function that allocates device data structures that are used
 * to hold the Gaussian_source that get found, and the number of them
 * Note should be paired with a later call to free_gaussian_source_list
 *****************************************************************************/
template<typename PRECISION>
Gaussian_source_list<PRECISION> allocate_gaussian_source_list(const unsigned int max_gaussian_sources_host)
{
    Gaussian_source<PRECISION> *gaussian_sources_device; // will hold sources that have distinct scales/positions (duplicates get merged)
    CUDA_CHECK_RETURN(cudaMalloc((void**)&gaussian_sources_device, max_gaussian_sources_host*sizeof(Gaussian_source<PRECISION>)));
    CUDA_CHECK_RETURN(cudaMemset(gaussian_sources_device, 0, max_gaussian_sources_host*sizeof(Gaussian_source<PRECISION>))); // clear the list of sources
    unsigned int *num_gaussian_sources_device;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&num_gaussian_sources_device, sizeof(unsigned int)));
    CUDA_CHECK_RETURN(cudaMemset(num_gaussian_sources_device, 0, sizeof(unsigned int))); // initially 0 sources

    Gaussian_source_list<PRECISION> gaussian_sources_list;
    gaussian_sources_list.gaussian_sources_device = gaussian_sources_device;
    gaussian_sources_list.num_gaussian_sources_device = num_gaussian_sources_device;
    return gaussian_sources_list;
}

template Gaussian_source_list<float> allocate_gaussian_source_list<float>(const unsigned int);
template Gaussian_source_list<double> allocate_gaussian_source_list<double>(const unsigned int);


/*****************************************************************************
 * Temporary utility function that copies the gaussian sources found during cleaning from device to the host
 *****************************************************************************/
template<typename PRECISION>
void copy_gaussian_source_list_to_host
    (
    Gaussian_source<PRECISION> *gaussian_sources_device,
    unsigned int *num_gaussian_sources_device,
    unsigned int max_gaussian_sources_host,
    unsigned int *num_gaussian_sources_host,
    Gaussian_source<PRECISION> *gaussian_sources_host
    )
{
    memset(gaussian_sources_host, 0, max_gaussian_sources_host*sizeof(Gaussian_source<PRECISION>)); // clear the list of sources
    CUDA_CHECK_RETURN(cudaMemcpy(gaussian_sources_host, gaussian_sources_device, max_gaussian_sources_host*sizeof(Gaussian_source<PRECISION>), cudaMemcpyDeviceToHost));
    memset(num_gaussian_sources_host, 0, sizeof(unsigned int));
    CUDA_CHECK_RETURN(cudaMemcpy(num_gaussian_sources_host, num_gaussian_sources_device, sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

template void copy_gaussian_source_list_to_host<float>(Gaussian_source<float>*, unsigned int*, unsigned int, unsigned int*, Gaussian_source<float>*);
template void copy_gaussian_source_list_to_host<double>(Gaussian_source<double>*, unsigned int*, unsigned int, unsigned int*, Gaussian_source<double>*);


/*****************************************************************************
 * Temporary utility function that displays the gaussian sources found during cleaning
 *****************************************************************************/
template<typename PRECISION>
void display_gaussian_source_list
    (
    Gaussian_source<PRECISION> *gaussian_sources_device,
    unsigned int *num_gaussian_sources_device,
    unsigned int max_gaussian_sources_host,
    unsigned int dirty_moment_size,
    unsigned int image_border,
    unsigned int num_taylor
    )
{
    unsigned int *num_gaussian_sources_host;
    CUDA_CHECK_RETURN(cudaHostAlloc(&num_gaussian_sources_host, sizeof(unsigned int), 0));
    Gaussian_source<PRECISION> *gaussian_sources_host; // sources that have distinct scales/positions (duplicates get merged)
    CUDA_CHECK_RETURN(cudaHostAlloc(&gaussian_sources_host, max_gaussian_sources_host*sizeof(Gaussian_source<PRECISION>), 0));
    copy_gaussian_source_list_to_host<PRECISION>
        (gaussian_sources_device, num_gaussian_sources_device, max_gaussian_sources_host, num_gaussian_sources_host, gaussian_sources_host);
    printf("In total %u distinct sources were discovered during cleaning\n", *num_gaussian_sources_host);
    // display each distinct source that has been found
    for (unsigned int source_index=0; source_index<*num_gaussian_sources_host; source_index++)
    {
        Gaussian_source<PRECISION> source = (Gaussian_source<PRECISION>)gaussian_sources_host[source_index];
        unsigned int x_pos = (source.index % (dirty_moment_size-2*image_border)) + image_border;
        unsigned int y_pos = (source.index / (dirty_moment_size-2*image_border)) + image_border; 
        printf("Source %3u has scale variance %8.2lf and position (%5u,%5u) with taylor term intensities: ",
            source_index, source.variance, x_pos, y_pos);
        for (unsigned int taylor_index=0; taylor_index<num_taylor; taylor_index++)
        {
            printf("%+12lf ", source.intensities[taylor_index]);
        }
        printf("\n");
    }
    CUDA_CHECK_RETURN(cudaFreeHost(gaussian_sources_host));
    CUDA_CHECK_RETURN(cudaFreeHost(num_gaussian_sources_host));
}

template void display_gaussian_source_list<float>(Gaussian_source<float>*, unsigned int*, unsigned int, unsigned int, unsigned int, unsigned int);
template void display_gaussian_source_list<double>(Gaussian_source<double>*, unsigned int*, unsigned int, unsigned int, unsigned int, unsigned int);


/*****************************************************************************
 * Msmfs function that deallocates device data structures that were used
 * to hold the Gaussian_source that got found, and the number of them
 * Note should be paired with an earlier call to allocate_gaussian_source_list
 *****************************************************************************/
template<typename PRECISION>
void free_gaussian_source_list(Gaussian_source_list<PRECISION> gaussian_source_list)
{
    // clean up the source model
    CUDA_CHECK_RETURN(cudaFree(gaussian_source_list.num_gaussian_sources_device));
    CUDA_CHECK_RETURN(cudaFree(gaussian_source_list.gaussian_sources_device));
}

template void free_gaussian_source_list<float>(Gaussian_source_list<float>);
template void free_gaussian_source_list<double>(Gaussian_source_list<double>);


/*****************************************************************************
 * Msmfs function that allocates device data structures that are used
 * during cleaning minor cycles but which are unlikely to be of interest afterwards
 * Returns a Cleaning_device_data_structure holding pointers to all the
 * allocated data structures
 * Note should be paired with a later call to free_device_data_structures
 * and is called before the clean minor cycle loop commences
 *****************************************************************************/
template<typename PRECISION>
Cleaning_device_data_structures<PRECISION> allocate_device_data_structures
    (
    const unsigned int num_taylor,
    const unsigned int scale_moment_size, // one dimensional size of scale moment residuals, assumed square
    bool larger_psf_convolved_buffer, // whether sufficient device memory to hold (double) convolution of psf images
    const unsigned int psf_moment_size, // one dimensional size of psf, assumed square
    const unsigned int psf_convolved_size, // one dimensional size of psf_convolved_images_device
    const unsigned int num_psf
    )
{
    PRECISION *peak_point_smpsol_device; // output principal solution at the peak for each taylor term
    unsigned int *peak_point_scale_device; // output scale at the peak
    unsigned int *peak_point_index_device; // output array offset index of point at the peak
    CUDA_CHECK_RETURN(cudaMalloc((void**)&peak_point_smpsol_device, num_taylor*sizeof(PRECISION)));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&peak_point_scale_device, sizeof(unsigned int)));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&peak_point_index_device, sizeof(unsigned int)));
    PRECISION *smpsol_max_device; // temporary array reused in find_principal_solution_at_peak
    unsigned int *smpsol_scale_device; // temporary array reused in find_principal_solution_at_peak
    CUDA_CHECK_RETURN(cudaMalloc((void**)&smpsol_max_device, scale_moment_size*scale_moment_size*sizeof(PRECISION)));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&smpsol_scale_device, scale_moment_size*scale_moment_size*sizeof(unsigned int)));
    // state whether subtract_peak_from_scale_moment_residuals should either reuse convolved psf
    // so that subtract_peak_from_scale_moment_residuals only needs to calculate
    // 2*num_taylor-1 (double) convolutions for each scale during each minor cycle
    // or instead calculate all on the fly to reduce memory usage
    // so that subtract_peak_from_scale_moment_residuals must calculate
    // num_taylor*num_taylor (double) convolutions for each scale on the fly
    // Note that if there were sufficient memory for
    // psf_convolved_size*psf_convolved_size*num_psf*(2*num_scales-1)
    // then all (doubly) convolved psf could be precalculated at all scales once for all minor cycles
    PRECISION *psf_convolved_images_device;
    if (larger_psf_convolved_buffer)
    {
        // allocate sufficient device memory to hold a (double) convolution of each of the psf images
        // note this means subtract_peak_from_scale_moment_residuals only needs to calculate
        // 2*num_taylor-1 (double) convolutions for each scale during each minor cycle
        // rather than num_taylor*num_taylor (double) convolutions for each scale
        CUDA_CHECK_RETURN(cudaMalloc((void**)&psf_convolved_images_device, psf_convolved_size*psf_convolved_size*num_psf*sizeof(PRECISION)));
    }
    else
    {
        // only allocate sufficient device memory to hold a (double) convolution of one psf images
        // note this means subtract_peak_from_scale_moment_residuals will recalculate each
        // double convolution between 0 and num_taylor times at each scale
        CUDA_CHECK_RETURN(cudaMalloc((void**)&psf_convolved_images_device, psf_convolved_size*psf_convolved_size*sizeof(PRECISION)));
    }
    // note partially convolved horiz_convolved_device is not typically square as border only trimmed on left and right sides
    PRECISION *horiz_convolved_device;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&horiz_convolved_device, psf_convolved_size*psf_moment_size*sizeof(PRECISION)));
    bool *is_existing_source_device; // out flag whether the source added was found already in model
    CUDA_CHECK_RETURN(cudaMalloc((void**)&is_existing_source_device, sizeof(bool)));

    checkCudaStatus();

    Cleaning_device_data_structures<PRECISION> working_data;
    working_data.peak_point_smpsol_device = peak_point_smpsol_device;
    working_data.peak_point_scale_device = peak_point_scale_device;
    working_data.peak_point_index_device = peak_point_index_device;
    working_data.smpsol_max_device = smpsol_max_device;
    working_data.smpsol_scale_device = smpsol_scale_device;
    working_data.psf_convolved_images_device = psf_convolved_images_device;
    working_data.horiz_convolved_device = horiz_convolved_device;
    working_data.is_existing_source_device = is_existing_source_device;
    return working_data;
}

template Cleaning_device_data_structures<float> allocate_device_data_structures<float>(const unsigned int, const unsigned int, bool, const unsigned int, const unsigned int, const unsigned int);
template Cleaning_device_data_structures<double> allocate_device_data_structures<double>(const unsigned int, const unsigned int, bool, const unsigned int, const unsigned int, const unsigned int);


/*****************************************************************************
 * Msmfs function that deallocates device data structures that were used
 * during cleaning minor cycles, frees each pointer in the parameter struct
 * Note should be paired with an earlier call to allocated_device_data_structures
 * and is called after the clean minor cycle loop completes
 *****************************************************************************/
template<typename PRECISION>
void free_device_data_structures(Cleaning_device_data_structures<PRECISION> working_data)
{
    CUDA_CHECK_RETURN(cudaFree(working_data.is_existing_source_device));
    CUDA_CHECK_RETURN(cudaFree(working_data.horiz_convolved_device));
    CUDA_CHECK_RETURN(cudaFree(working_data.psf_convolved_images_device));
    CUDA_CHECK_RETURN(cudaFree(working_data.smpsol_scale_device)); // discard reused smpsol_scale_device as no longer needed
    CUDA_CHECK_RETURN(cudaFree(working_data.smpsol_max_device)); // discard reused smpsol_scale_device as no longer needed
    CUDA_CHECK_RETURN(cudaFree(working_data.peak_point_index_device));
    CUDA_CHECK_RETURN(cudaFree(working_data.peak_point_scale_device));
    CUDA_CHECK_RETURN(cudaFree(working_data.peak_point_smpsol_device));
}

template void free_device_data_structures<float>(Cleaning_device_data_structures<float>);
template void free_device_data_structures<double>(Cleaning_device_data_structures<double>);


/*****************************************************************************
 * Msmfs function that calculates the principal solution smpsol across all scales at t=0 on device
 * and finds the largest absolute value at each point across all the scales
 * scale_moment_residuals_device presumed to have size scale_moment_size*scale_moment_size*num_scales*num_taylor
 * inverse_hessian_matrices_device presumed to have size num_scales*num_taylor*num_taylor
 *****************************************************************************/
template<typename PRECISION>
void find_principal_solution_at_peak
    (
    PRECISION *scale_moment_residuals_device, // flat array that holds all convolved scale moment residuals
    const unsigned int num_scales,
    const unsigned int num_taylor,
    const unsigned int scale_moment_size, // one dimensional size of scale moment residuals, assumed square
    PRECISION *inverse_hessian_matrices_device, // flat array that holds all inverse hessian matrices
    const PRECISION* scale_bias_device, // bias multiplicative factor to favour cleaning with smaller scales
    PRECISION* peak_point_smpsol_device, // output principal solution at the peak
    unsigned int* peak_point_scale_device, // output scale at the peak
    unsigned int* peak_point_index_device, // output array offset index of point at the peak
    PRECISION *smpsol_max_device, // temporary array reused in find_principal_solution_at_peak
    unsigned int* smpsol_scale_device, // temporary array reused in find_principal_solution_at_peak
    dim3 cuda_block_size_2D,
    int cuda_block_size,
    int cuda_num_threads // number of cuda threads available which is used for grid-strided kernel
    )
{
    // clear temporary device memory that will hold entries for the maximum smpsol and its scale at each point
    // note smpsol has scale bias applied to it in calculate_principal_solution_max_scale
    CUDA_CHECK_RETURN(cudaMemset(smpsol_max_device, 0, scale_moment_size*scale_moment_size*sizeof(PRECISION))); // clear the reused temp array to zero
    CUDA_CHECK_RETURN(cudaMemset(smpsol_scale_device, 0, scale_moment_size*scale_moment_size*sizeof(unsigned int))); // clear the reused temp array to zero

    // calculate the maximum principal solution with bias applied across all scales at each point on the device
    int cuda_grid_scale_moment_size = (scale_moment_size+cuda_block_size_2D.x-1)/cuda_block_size_2D.x;
    dim3 cuda_grid_scale_moment_size_2D(cuda_grid_scale_moment_size,cuda_grid_scale_moment_size,1);  // create a thread per point to be calculated
    calculate_principal_solution_max_scale<<<cuda_grid_scale_moment_size_2D, cuda_block_size_2D>>>
        (scale_moment_residuals_device, num_scales, num_taylor, scale_moment_size,
        inverse_hessian_matrices_device, scale_bias_device, smpsol_max_device, smpsol_scale_device);

    // find the location of the entry in smp_max_entry_device that has maximum biased value
    // by using a one-dimensional reduction with a grid-strided cuda kernel
    Array_entry *smpsol_max_entry_device; // note could also reuse this data structure each minor cycle
    CUDA_CHECK_RETURN(cudaMalloc((void**)&smpsol_max_entry_device, sizeof(Array_entry)));
    CUDA_CHECK_RETURN(cudaMemset(smpsol_max_entry_device, 0, sizeof(Array_entry)));
    int cuda_grid_size_strided = cuda_num_threads;
    find_max_entry_grid_strided_reduction<<<cuda_grid_size_strided,cuda_block_size>>>
        (smpsol_max_device, scale_moment_size*scale_moment_size, smpsol_max_entry_device);

    // determine the principal solution, its scale and position at the peak smpsol point
    CUDA_CHECK_RETURN(cudaMemset(peak_point_smpsol_device, 0, num_taylor*sizeof(PRECISION)));
    CUDA_CHECK_RETURN(cudaMemset(peak_point_scale_device, 0, sizeof(unsigned int)));
    CUDA_CHECK_RETURN(cudaMemset(peak_point_index_device, 0, sizeof(unsigned int)));
    int cuda_grid_mval_size = (num_taylor+cuda_block_size-1)/cuda_block_size;
    calculate_principal_solution_at_peak<<<cuda_grid_mval_size,cuda_block_size>>>
        (scale_moment_residuals_device, num_scales, num_taylor, scale_moment_size,
        inverse_hessian_matrices_device,
        smpsol_scale_device, (unsigned int*)&(smpsol_max_entry_device->index),
        peak_point_smpsol_device, peak_point_scale_device, peak_point_index_device);

    CUDA_CHECK_RETURN(cudaFree(smpsol_max_entry_device)); // discard smpsol_max_entry_device as no longer needed 
}

template void find_principal_solution_at_peak<float>(float*, const unsigned int, const unsigned int, const unsigned int, float*, const float*, float*, unsigned int*, unsigned int*, float*, unsigned int*, dim3, int, int);
template void find_principal_solution_at_peak<double>(double*, const unsigned int, const unsigned int, const unsigned int, double*, const double*, double*, unsigned int*, unsigned int*, double*, unsigned int*, dim3, int, int);


/*****************************************************************************
 * Msmfs function that subtracts the scale-dependent moment psf from the
 * scale convolutions of moment residuals at the peak on device
 * Note only the portion of the psf within a border region given by the convolution size is subtracted 
 * Note scale_moment_residuals_device presumed to have size num_taylor*num_scales*scale_moment_size*scale_moment_size
 * psf_moment_images_device presumed to have size num_psf*psf_moment_size*psf_moment_size
 * and assumed to be centred around origin with psf_moment_images_device having sufficient size for convolution at its centre
 * and assumed that num_psf = 2*num_taylor - 1
 * variances presumed to have size num_scales
 * double_convolution_support_device presumed to have size num_scales*num_scales
 * if larger_convolved_psf_buffer true then psf_convolved_device presumed to have size psf_convolved_size*psf_convolved_size*num_psf
 * if larger_convolved_psf_buffer false then psf_convolved_device presumed to have size psf_convolved_size*psf_convolved_size
 * horiz_convolved_device presumed to have width psf_convolved_size and height psf_moment_size
 *****************************************************************************/
template<typename PRECISION>
void subtract_peak_from_scale_moment_residuals
    (
    PRECISION *scale_moment_residuals_device, // inout flat array that holds scale moment residuals
    const unsigned int num_scales,
    const unsigned int num_taylor,
    const unsigned int scale_moment_size, // one dimensional size of scale moment residuals, assumed square
    PRECISION *psf_moment_images_device, // flat array containing input Taylor coefficient psf images to be convolved
    const unsigned int num_psf,
    const unsigned int psf_moment_size, // one dimensional size of psf, assumed square
    PRECISION *variances_device, // variances used for calculating gaussian kernel, one for each of num_scales
    unsigned int *double_convolution_support_device, // supports required for each gaussian kernel shape for double convolution
    PRECISION *peak_point_smpsol_device, // input principal solution at the peak
    unsigned int *peak_point_scale_device, // input scale index at the peak
    unsigned int *peak_point_index_device, // input array offset index of point at the peak
    PRECISION clean_loop_gain, // loop gain fraction of peak point to clean from the peak each minor cycle
    PRECISION *psf_convolved_images_device, // reused buffer that holds (doubly) convolved psf images so they can be reused here
    bool larger_psf_convolved_buffer, // whether psf_convolved_device holds num_psf or just 1 (doubly) convolved psf image
    unsigned int psf_convolved_size, // one dimensional size of psf_convolved_images_device
    PRECISION *horiz_convolved_device, // reused partially convolved psf image
    dim3 cuda_block_size_2D
    )
{
    int cuda_grid_psf_moment_size = (psf_moment_size+cuda_block_size_2D.y-1)/cuda_block_size_2D.y;
    for (unsigned int scale_index=0; scale_index<num_scales; scale_index++)
    {
        // calculate a suitable convolution support based on the requested convolution accuracy
        unsigned int image_border = (psf_moment_size-psf_convolved_size)/2;
        int cuda_grid_psf_convolved_size = (psf_convolved_size+cuda_block_size_2D.x-1)/cuda_block_size_2D.x;
        dim3 cuda_grid_horiz_convolve_2D(cuda_grid_psf_convolved_size,cuda_grid_psf_moment_size,1);  // create a thread per image pixel to be convolved
        dim3 cuda_grid_vert_convolve_2D(cuda_grid_psf_convolved_size,cuda_grid_psf_convolved_size,1);  // create a thread per image pixel to be convolved
        if (larger_psf_convolved_buffer)
        {
            // precalculate the (doubly) convolved psf images for this scale
            CUDA_CHECK_RETURN(cudaMemset(psf_convolved_images_device, 0, psf_convolved_size*psf_convolved_size*num_psf*sizeof(PRECISION))); // clear all the images to zero
            for (unsigned int taylor_index_sum=0; taylor_index_sum<num_psf; taylor_index_sum++)
            {
                // calculate one (doubly) convolved psf image on the fly
                PRECISION *psf_device = &psf_moment_images_device[taylor_index_sum*psf_moment_size*psf_moment_size];
                PRECISION *psf_convolved_device = &psf_convolved_images_device[taylor_index_sum*psf_convolved_size*psf_convolved_size];
                // perform the 2D convolution as two 1D convolutions
                double_convolve_horizontal<<<cuda_grid_horiz_convolve_2D, cuda_block_size_2D>>>
                    (psf_device, horiz_convolved_device, psf_convolved_size, psf_moment_size, image_border,
                    double_convolution_support_device, num_scales, variances_device, scale_index, peak_point_scale_device);
                double_convolve_vertical<<<cuda_grid_vert_convolve_2D, cuda_block_size_2D>>>
                    (horiz_convolved_device, psf_convolved_device, psf_convolved_size, psf_convolved_size, image_border,
                    double_convolution_support_device, num_scales, variances_device, scale_index, peak_point_scale_device);
            }
        }
        for (unsigned int taylor_index=0; taylor_index<num_taylor; taylor_index++)
        {
            // calculate one scale convolution for one taylor_index moment residual
            PRECISION *scale_moment_residual_device = &scale_moment_residuals_device[(taylor_index*num_scales+scale_index)*scale_moment_size*scale_moment_size];
            for (unsigned int taylor_index_two=0; taylor_index_two<num_taylor; taylor_index_two++)
            {
                unsigned int taylor_index_sum = taylor_index + taylor_index_two;
                PRECISION *psf_convolved_device = NULL;
                // determine which psf to doubly convolve using a single gaussian with summed variance
                if (larger_psf_convolved_buffer)
                {
                    // lookup the appropriate precalculated (doubly) convolved psf image
                    psf_convolved_device = &psf_convolved_images_device[taylor_index_sum*psf_convolved_size*psf_convolved_size];
                }
                else
                {
                    // calculate one (doubly) convolved psf image on the fly
                    PRECISION *psf_device = &psf_moment_images_device[taylor_index_sum*psf_moment_size*psf_moment_size];
                    CUDA_CHECK_RETURN(cudaMemset(psf_convolved_images_device, 0, psf_convolved_size*psf_convolved_size*sizeof(PRECISION))); // clear the image to zero
                    psf_convolved_device = psf_convolved_images_device; // use the single psf image buffer
                    // perform the 2D convolution as two 1D convolutions
                    double_convolve_horizontal<<<cuda_grid_horiz_convolve_2D, cuda_block_size_2D>>>
                        (psf_device, horiz_convolved_device, psf_convolved_size, psf_moment_size, image_border,
                        double_convolution_support_device, num_scales, variances_device, scale_index, peak_point_scale_device);
                    double_convolve_vertical<<<cuda_grid_vert_convolve_2D, cuda_block_size_2D>>>
                        (horiz_convolved_device, psf_convolved_device, psf_convolved_size, psf_convolved_size, image_border,
                        double_convolution_support_device, num_scales, variances_device, scale_index, peak_point_scale_device);
                }
                // subtract psf_convolved_device scaled by peak_point_smpsol_host[taylor_index_two] and clean_loop_gain
                // from scale_moment_residual_device centred at peak_point_index
                dim3 cuda_grid_subtract_psf_2D(cuda_grid_psf_convolved_size,cuda_grid_psf_convolved_size,1);  // create a thread per psf pixel to be subtracted
                subtract_psf_convolved_from_scale_moment_residual<<<cuda_grid_subtract_psf_2D, cuda_block_size_2D>>>
                    (scale_moment_residual_device, scale_moment_size,
                    psf_convolved_device, psf_convolved_size,
                    peak_point_smpsol_device, taylor_index_two, peak_point_index_device, clean_loop_gain);
            }
        }
    }
}

template void subtract_peak_from_scale_moment_residuals<float>(float*, const unsigned int, const unsigned int, const unsigned int, float*, const unsigned int, const unsigned int, float*, unsigned int*, float*, unsigned int*, unsigned int*, float, float*, bool, unsigned int, float*, dim3);
template void subtract_peak_from_scale_moment_residuals<double>(double*, const unsigned int, const unsigned int, const unsigned int, double*, const unsigned int, const unsigned int, double*, unsigned int*, double*, unsigned int*, unsigned int*, double, double*, bool, unsigned int, double*, dim3);


/*****************************************************************************
 * Msmfs function that updates the scale moment model of gaussian sources
 * Note that gaussian_sources_device presumed to currently have num_gaussian_sources_device sources already
 * and presumed to have sufficient size to allow potentially adding an additional source
 * although if additional source has the same scale and index as one already in list
 * then they are considered duplicates and so get merged by adding together their intensities
 * peak_point_smpsol_device presumed to have size num_taylor
 *****************************************************************************/
template<typename PRECISION>
void update_scale_moment_model
    (
    Gaussian_source<PRECISION> *gaussian_sources_device, // inout list of sources that have distinct scales/positions
    unsigned int *num_gaussian_sources_device, // inout current number of sources in gaussian_sources_device
    PRECISION *peak_point_smpsol_device, // input principal solution at the peak
    const unsigned int num_taylor,
    PRECISION *variances_device, // variances used for calculating gaussian kernel, one for each of num_scales
    unsigned int *peak_point_scale_device, // input scale index at the peak
    unsigned int *peak_point_index_device, // input array offset index of point at the peak
    PRECISION clean_loop_gain, // loop gain fraction of peak point to clean from the peak each minor cycle
    bool *is_existing_source_device, // out flag whether the source added was found already in model
    int cuda_block_size
    )
{
    // execute kernel using single block with each thread in that block using a grid-strided loop
    CUDA_CHECK_RETURN(cudaMemset(is_existing_source_device, 0, sizeof(bool))); // clear the flag to false
    add_source_to_model_grid_strided_reduction<<<1, cuda_block_size>>>
        (gaussian_sources_device, num_gaussian_sources_device, peak_point_smpsol_device, num_taylor,
        variances_device, peak_point_scale_device, peak_point_index_device, clean_loop_gain,
        is_existing_source_device);
}

template void update_scale_moment_model<float>(Gaussian_source<float>*, unsigned int*, float*, const unsigned int, float*, unsigned int*, unsigned int*, float, bool*, int);
template void update_scale_moment_model<double>(Gaussian_source<double>*, unsigned int*, double*, const unsigned int, double*, unsigned int*, unsigned int*, double, bool*, int);


/*****************************************************************************
 * Temporary utility function that displays the principal solution at the peak found
 * this clean minor cycle
 *****************************************************************************/
template<typename PRECISION>
void display_principal_solution_at_peak
    (
    PRECISION *peak_point_smpsol_device,
    unsigned int *peak_point_scale_device,
    unsigned int *peak_point_index_device,
    unsigned int scale_moment_size,
    unsigned int num_taylor
    )
{
    // obtain the principal solution at the peak on the host
    PRECISION *peak_point_smpsol_host;
    unsigned int *peak_point_scale_host;
    unsigned int *peak_point_index_host;
    CUDA_CHECK_RETURN(cudaHostAlloc(&peak_point_smpsol_host, num_taylor*sizeof(PRECISION), 0));
    CUDA_CHECK_RETURN(cudaHostAlloc(&peak_point_scale_host, sizeof(unsigned int), 0));
    CUDA_CHECK_RETURN(cudaHostAlloc(&peak_point_index_host, sizeof(unsigned int), 0));
    CUDA_CHECK_RETURN(cudaMemcpy(peak_point_smpsol_host, peak_point_smpsol_device, num_taylor*sizeof(PRECISION), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(peak_point_scale_host, peak_point_scale_device, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(peak_point_index_host, peak_point_index_device, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    // display the peak found during this minor cycle
    printf("Peak found at (%5u,%5u) with scale index %3u",
        *peak_point_index_host%scale_moment_size, *peak_point_index_host/scale_moment_size, *peak_point_scale_host);
    printf("  has principal solution at peak: ");
    for (unsigned int taylor_index=0; taylor_index<num_taylor; taylor_index++)
    {
        printf("%+9.4lf ", peak_point_smpsol_host[taylor_index]);
    }
    printf("\n");
    CUDA_CHECK_RETURN(cudaFreeHost(peak_point_index_host));
    CUDA_CHECK_RETURN(cudaFreeHost(peak_point_scale_host));
    CUDA_CHECK_RETURN(cudaFreeHost(peak_point_smpsol_host));
}

template void display_principal_solution_at_peak<float>(float*, unsigned int*, unsigned int*, unsigned int, unsigned int);
template void display_principal_solution_at_peak<double>(double*, unsigned int*, unsigned int*, unsigned int, unsigned int);


/*****************************************************************************
 * Msmfs function that performs the major cycles for msmfs cleaning
 * Returns the number of major cycles actually performed
 *****************************************************************************/
template<typename PRECISION>
int perform_major_cycles
    (
    const unsigned int max_clean_cycles, // amximum number of clean cycles to perform
    const unsigned int min_clean_cycles, // minimum number of cycles to perform before checking clean threshold
    const PRECISION clean_threshold, // set clean_threshold to 0 to disable checking whether source to clean below cutoff threshold
    const PRECISION clean_loop_gain, // loop gain fraction of peak point to clean from the peak each minor cycle
    PRECISION *scale_moment_residuals_device, // inout flat array that holds scale moment residuals
    const unsigned int num_scales,
    const unsigned int num_taylor,
    const unsigned int scale_moment_size, // one dimensional size of scale moment residuals, assumed square

    PRECISION *psf_moment_images_device,
    const unsigned int num_psf,
    const unsigned int psf_moment_size, // one dimensional size of psf, assumed square
    const unsigned int psf_convolved_size, // one dimensional size of psf_convolved_images_device
    bool larger_psf_convolved_buffer, // whether psf_convolved_device holds num_psf or just 1 (doubly) convolved psf image
    
    PRECISION *inverse_hessian_matrices_device, // flat array that holds all inverse hessian matrices

    PRECISION *variances_device, // variances used for calculating gaussian kernel, one for each of num_scales
    const PRECISION *scale_bias_device, // bias multiplicative factor to favour cleaning with smaller scales
    unsigned int *double_convolution_support_device, // supports required for each gaussian kernel shape for double convolution

    PRECISION *peak_point_smpsol_device, // output principal solution at the peak for each taylor term
    unsigned int *peak_point_scale_device, // output scale at the peak
    unsigned int *peak_point_index_device, // output array offset index of point at the peak
    PRECISION *smpsol_max_device, // temporary array reused in find_principal_solution_at_peak
    unsigned int *smpsol_scale_device, // temporary array reused in find_principal_solution_at_peak
    PRECISION *psf_convolved_images_device,
    PRECISION *horiz_convolved_device,
    bool *is_existing_source_device, // out flag whether the source added was found already in model
            
    Gaussian_source<PRECISION> *gaussian_sources_device, // output sources that have distinct scales/positions (duplicates get merged)
    unsigned int *num_gaussian_sources_device, // output number of sources found in gaussian_sources_device

    int cuda_block_size,
    dim3 cuda_block_size_2D,
    int cuda_num_threads
    )
{
    // set up timer
    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);
    float execution_time = 0; // milliseconds
    float summed_time = 0; // milliseconds

    unsigned int clean_cycle = 0;
    bool clean_threshold_reached = false;
    while (clean_cycle<max_clean_cycles && !clean_threshold_reached)
    {
        if (max_clean_cycles<10 || clean_cycle%10==0)
        {
            sdp_logger(LOG_INFO, "Commencing clean cycle minor iteration %d", clean_cycle);
        }
        else
        {
            sdp_logger(LOG_DEBUG, "Commencing clean cycle minor iteration %d", clean_cycle);
        }

        // calculate the principal solution at t=0
        cudaEventRecord(start_time);
        find_principal_solution_at_peak
            (scale_moment_residuals_device, num_scales, num_taylor, scale_moment_size,
            inverse_hessian_matrices_device, scale_bias_device,
            peak_point_smpsol_device, peak_point_scale_device,
            peak_point_index_device,
            smpsol_max_device, smpsol_scale_device,
            cuda_block_size_2D, cuda_block_size, cuda_num_threads);
        cudaEventRecord(stop_time);
        cudaEventSynchronize(stop_time);
        cudaEventElapsedTime(&execution_time, start_time, stop_time);
        sdp_logger(LOG_DEBUG, "Execution time for finding principal solution at peak across all scales at t=0 is %.3fms", execution_time);
        summed_time += execution_time;

        checkCudaStatus();

        // ***** temporary code to display the principal solution at the peak *****
//        display_principal_solution_at_peak(peak_point_smpsol_device, peak_point_scale_device, peak_point_index_device, scale_moment_size, num_taylor);
        // ***** end of temporary code *****

        if (clean_cycle>=min_clean_cycles && clean_threshold>0)
        {
            // TODO COPY peak_point_smpsol_device AND AT INDEX 0 TO DEVICE AND CHECK CLEAN THRESHOLD
            // clean_threshold_reached = true;
        }

        if (!clean_threshold_reached)
        {
            // subtract num_taylor psf centered at peak from each of the scale moment residuals
            cudaEventRecord(start_time);
            subtract_peak_from_scale_moment_residuals
                (scale_moment_residuals_device, num_scales, num_taylor, scale_moment_size,
                psf_moment_images_device, num_psf, psf_moment_size,
                variances_device, double_convolution_support_device,
                peak_point_smpsol_device, peak_point_scale_device,
                peak_point_index_device,
                clean_loop_gain,
                psf_convolved_images_device, larger_psf_convolved_buffer,
                psf_convolved_size, horiz_convolved_device,
                cuda_block_size_2D);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&execution_time, start_time, stop_time);
            sdp_logger(LOG_DEBUG, "Execution time for subtracting peak from each of the scale moment residuals is %.3fms", execution_time);
            summed_time += execution_time;

            checkCudaStatus();

            // update gaussian_sources_device and potentially too num_gaussian_sources_device with the additional discovered source
            cudaEventRecord(start_time);
            update_scale_moment_model(gaussian_sources_device, num_gaussian_sources_device,
                peak_point_smpsol_device, num_taylor, variances_device,
                peak_point_scale_device, peak_point_index_device,
                clean_loop_gain, is_existing_source_device, cuda_block_size);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&execution_time, start_time, stop_time);
            sdp_logger(LOG_DEBUG, "Execution time for updating the source model is %.3fms", execution_time);
            summed_time += execution_time;

            checkCudaStatus();

            clean_cycle++;
        }
    }
    sdp_logger(LOG_NOTICE, "Cleaning completed after completing %u clean minor cycles in time %.3fms", clean_cycle, summed_time);
    // end of minor cycles
    return clean_cycle;
}

template int perform_major_cycles<float>(const unsigned int, const unsigned int, const float, const float, float*, const unsigned int, const unsigned int,
    const unsigned int, float*, const unsigned int, const unsigned int, const unsigned int, bool, float*, float*,
    const float*, unsigned int*, float*, unsigned int*, unsigned int*, float*, unsigned int*, float*, float*, bool*, Gaussian_source<float>*, unsigned int*, int, dim3, int);
template int perform_major_cycles<double>(const unsigned int, const unsigned int, const double, const double, double*, const unsigned int, const unsigned int,
    const unsigned int, double*, const unsigned int, const unsigned int, const unsigned int, bool, double*, double*,
    const double*, unsigned int*, double*, unsigned int*, unsigned int*, double*, unsigned int*, double*, double*, bool*, Gaussian_source<double>*, unsigned int*, int, dim3, int);

