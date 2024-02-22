/*
 * sdp_cuFFTxT.cu
 *
 *  Created on: 6 Feb 2024
 *      Author: vlad
 */


#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>
#include <cufftXt.h>

#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"

typedef struct sdp_Double2
{
    double x;
    double y;
} sdp_Double2;


void gpuAssert(cufftResult code)
{
    if (code != CUFFT_SUCCESS)
    {
        fprintf(stderr, "GPUassert: CUFFT error \n");
        exit(code);
    }
}


void gpuError(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(code));
        exit(code);
    }
}


/*
 * Retrieve device IDs for all CUDA devices in the current system.
 */
int getAllGpus(int** gpus, int gpu_start)
{
    int i;
    int nGpus;

    gpuError(cudaGetDeviceCount(&nGpus));
    nGpus -= gpu_start;

    *gpus = (int*)malloc(sizeof(int) * nGpus);

    for (i = 0; i < nGpus; i++)
    {
        (*gpus)[i] = i + gpu_start;
    }

    return nGpus;
}


void sdp_cuFFTxT(
        sdp_Mem* grid_sim1,
        sdp_Mem* image_out1,
        int gpu_start,
        sdp_Error* status
)
{
    int64_t i;
    int64_t grid_size;

    grid_size = sdp_mem_shape_dim(grid_sim1, 0);

    // TODO: How to assing sdp_Double2* pointer to cufftDoubleComplex* in this case?
    // cufftDoubleComplex *grid_sim_FFT = (sdp_Double2*) sdp_mem_data(grid_sim);

    cufftDoubleComplex* grid_sim =
            (cufftDoubleComplex*)malloc(
            sizeof(cufftDoubleComplex) * grid_size * grid_size
            );
    cufftDoubleComplex* image_out =
            (cufftDoubleComplex*)malloc(
            sizeof(cufftDoubleComplex) * grid_size * grid_size
            );

    sdp_Double2* grid_out = (sdp_Double2*) sdp_mem_data(grid_sim1);
    for (i = 0; i < grid_size * grid_size; i++)
    {
        grid_sim[i].x = grid_out[i].x;
        grid_sim[i].y = grid_out[i].y;
    }

    // cuFFtxT initialization
    int* gpus;
    size_t* workSize;
    cufftHandle plan = 0;
    cudaLibXtDesc* dComplexSamples;

    if (gpu_start < 0)
        gpu_start = 0;

    int nGPUs = getAllGpus(&gpus, gpu_start);
    SDP_LOG_INFO("System nGPUs = %d\n", nGPUs);
    nGPUs = nGPUs > 4 ? 4 : nGPUs;

    workSize = (size_t*)malloc(sizeof(size_t) * nGPUs);
    SDP_LOG_INFO("Using nGPUs = %d\n", nGPUs);
    SDP_LOG_INFO("%d x %d FFT test\n", grid_size, grid_size);

    for (int ii; ii < nGPUs; ii++)
        SDP_LOG_INFO("Using GPU %d, CUDA_DEVICE=%d\n", ii, gpus[ii]);

    // Setup the cuFFT Multi-GPU plan
    gpuAssert(cufftCreate(&plan));
    gpuAssert(cufftXtSetGPUs(plan, nGPUs, gpus));
    gpuAssert(cufftMakePlan2d(plan, grid_size, grid_size, CUFFT_Z2Z, workSize));

    // Get system time
    clock_t t;
    t = clock();

    // Allocate memory across multiple GPUs and transfer the inputs into it
    gpuAssert(cufftXtMalloc(plan, &dComplexSamples, CUFFT_XT_FORMAT_INPLACE));
    gpuAssert(cufftXtMemcpy(plan, dComplexSamples, grid_sim,
            CUFFT_COPY_HOST_TO_DEVICE
    )
    );

    // Execute a complex-to-complex 2D FFT across multiple GPUs
    gpuAssert(cufftXtExecDescriptorZ2Z(plan, dComplexSamples, dComplexSamples,
            CUFFT_INVERSE
    )
    );

    // Retrieve the results from multiple GPUs into host memory
    gpuAssert(cufftXtMemcpy(plan, image_out, dComplexSamples,
            CUFFT_COPY_DEVICE_TO_HOST
    )
    );

    // Get elapsed time for CUDA-related processes
    t = clock() - t;
    double time_taken = ((double)t) / CLOCKS_PER_SEC; // calculate the elapsed time
    SDP_LOG_INFO("CUDA part of the program took %f seconds to execute\n",
            time_taken
    );

    void* image_out2 = (void*)sdp_mem_data(image_out1);
    sdp_Double2* temp = (sdp_Double2*)image_out2;
    for (i = 0; i < grid_size * grid_size; i++)
    {
        temp[i].x = image_out[i].x;
        temp[i].y = image_out[i].y;
    }

    free(grid_sim);
    free(image_out);

    gpuAssert(cufftXtFree(dComplexSamples));
    gpuAssert(cufftDestroy(plan));
}
