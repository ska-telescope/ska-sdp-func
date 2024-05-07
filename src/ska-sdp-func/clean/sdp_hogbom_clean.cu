/* See the LICENSE file at the top-level directory of this distribution. */
#define __CUDA_NO_BFLOAT16_CONVERSIONS__
#define __CUDA_NO_BFLOAT16_OPERATORS__

#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/numeric_functions/sdp_fft_convolution.h"
#include "ska-sdp-func/utility/sdp_mem.h"

#include <cuComplex.h>
#include <cuda_bf16.h>

// #define INDEX_2D(N2, N1, I2, I1)    (N1 * I2 + I1)

// create a copy of a complex number using only its real part
template<typename CT, typename T>
__global__ void create_copy_real(
    const CT* in,
    int64_t size,
    T* out){

    }

template<>
__global__ void create_copy_real<cuDoubleComplex, double>(
    const cuDoubleComplex* in,
    int64_t size,
    double* out
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        out[i] = cuCreal(in[i]);
    }
}

template<>
__global__ void create_copy_real<cuFloatComplex, float>(
    const cuFloatComplex* in,
    int64_t size,
    float* out
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        out[i] = cuCrealf(in[i]);
    }
}

SDP_CUDA_KERNEL(create_copy_real<cuDoubleComplex, double>);
SDP_CUDA_KERNEL(create_copy_real<cuFloatComplex, float>);


// create the CLEAN beam
template<typename T, typename CT>
__global__ void create_cbeam(
    const T sigma_X,
    const T sigma_Y,
    const T rotation,
    int16_t psf_dim,
    CT* cbeam
) {

}

template<>
__global__ void create_cbeam<double, cuDoubleComplex>(
    const double sigma_X,
    const double sigma_Y,
    const double rotation,
    int16_t cbeam_dim,
    cuDoubleComplex* cbeam
) {
    // Fit a Gaussian to the main lobe of the PSF based on the parameters passed

    double A = 1;
    double x0 = 0;
    double y0 = 0;

    // Check if the number of rows and columns is odd
    if (cbeam_dim % 2 == 1) {
        x0 = cbeam_dim / 2;
        y0 = cbeam_dim / 2;
    } else {
        x0 = cbeam_dim / 2 - 1;
        y0 = cbeam_dim / 2 - 1;
    }

    // double sigma_X = cbeam_details[0];
    // double sigma_Y = cbeam_details[1];
    double theta = (M_PI / 180) * rotation;

    double a = pow(cos(theta), 2) / (2 * pow(sigma_X, 2)) + pow(sin(theta), 2) / (2 * pow(sigma_Y, 2));
    double b = sin(2 * theta) / (4 * pow(sigma_X, 2)) - sin(2 * theta) / (4 * pow(sigma_Y, 2));
    double c = pow(sin(theta), 2) / (2 * pow(sigma_X, 2)) + pow(cos(theta), 2) / (2 * pow(sigma_Y, 2));

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int size = cbeam_dim * cbeam_dim;

    if (i < size){ 
        int x = i / cbeam_dim;
        int y = i % cbeam_dim;

        double component = A * exp(-(a * pow(x - x0, 2) + 2 * b * (x - x0) * (y - y0) + c * pow(y - y0, 2)));
        cbeam[i] = make_cuDoubleComplex(component, 0);
    }
}

template<>
__global__ void create_cbeam<float, cuFloatComplex>(
    const float sigma_X,
    const float sigma_Y,
    const float rotation,
    int16_t cbeam_dim,
    cuFloatComplex* cbeam
) {
    // Fit a Gaussian to the main lobe of the PSF based on the parameters passed

    float A = 1;
    float x0 = 0;
    float y0 = 0;

    // Check if the number of rows and columns is odd
    if (cbeam_dim % 2 == 1) {
        x0 = cbeam_dim / 2;
        y0 = cbeam_dim / 2;
    } else {
        x0 = cbeam_dim / 2 - 1;
        y0 = cbeam_dim / 2 - 1;
    }
    // float sigma_X = cbeam_details[0];
    // float sigma_Y = cbeam_details[1];
    float theta = (M_PI / 180) * rotation;

    float a = pow(cos(theta), 2) / (2 * pow(sigma_X, 2)) + pow(sin(theta), 2) / (2 * pow(sigma_Y, 2));
    float b = sin(2 * theta) / (4 * pow(sigma_X, 2)) - sin(2 * theta) / (4 * pow(sigma_Y, 2));
    float c = pow(sin(theta), 2) / (2 * pow(sigma_X, 2)) + pow(cos(theta), 2) / (2 * pow(sigma_Y, 2));

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int size = cbeam_dim * cbeam_dim;

    if (i < size){ 
        int x = i / cbeam_dim;
        int y = i % cbeam_dim;

        float component = A * exp(-(a * pow(x - x0, 2) + 2 * b * (x - x0) * (y - y0) + c * pow(y - y0, 2)));
        cbeam[i] = make_cuFloatComplex(component, 0);
    }
}

SDP_CUDA_KERNEL(create_cbeam<double, cuDoubleComplex>);
SDP_CUDA_KERNEL(create_cbeam<float, cuFloatComplex>);


template<typename T, typename I>
__device__ void warpReduce(
            volatile T* max_values,
            volatile I* max_indices,
            int tid)
{
}

template<>
__device__ void warpReduce<double, int>(
            volatile double* max_values,
            volatile int* max_indices,
            int tid)
{
    double shfl_val;
    int shfl_idx; 

    // unsigned mask = __ballot_sync(0xffffffff, tid < 32); // warpSize is 32

    for (int offset = 16; offset > 0; offset >>=1) {

        shfl_val = __shfl_down_sync(0xFFFFFFFF, max_values[tid], offset);
        shfl_idx = __shfl_down_sync(0xFFFFFFFF, max_indices[tid], offset);
        __syncwarp();

        if (shfl_val > max_values[tid]) {
            max_values[tid] = shfl_val;
            max_indices[tid] = shfl_idx;
        }
    }


    // if (max_values[tid] < max_values[tid + 32]) {
    //             max_values[tid] = max_values[tid + 32];
    //             max_indices[tid] = max_indices[tid + 32];
    //     }
    // if (max_values[tid] < max_values[tid + 16]) {
    //             max_values[tid] = max_values[tid + 16];
    //             max_indices[tid] = max_indices[tid + 16];
    //     } 
    // if (max_values[tid] < max_values[tid + 8]) {
    //             max_values[tid] = max_values[tid + 8];
    //             max_indices[tid] = max_indices[tid + 8];
    //     } 
    // if (max_values[tid] < max_values[tid + 4]) {
    //             max_values[tid] = max_values[tid + 4];
    //             max_indices[tid] = max_indices[tid + 4];
    //     }
    // if (max_values[tid] < max_values[tid + 2]) {
    //             max_values[tid] = max_values[tid + 2];
    //             max_indices[tid] = max_indices[tid + 2];
    //     } 
    // if (max_values[tid] < max_values[tid + 1]) {
    //             max_values[tid] = max_values[tid + 1];
    //             max_indices[tid] = max_indices[tid + 1];
    //     }      
}

template<>
__device__ void warpReduce<float, int>(
            volatile float* max_values,
            volatile int* max_indices,
            int tid)
{
    float shfl_val;
    int shfl_idx;

    // unsigned mask = __ballot_sync(0xffffffff, tid < 32); // warpSize is 32

    for (int offset = 16; offset > 0; offset >>=1) {

        shfl_val = __shfl_down_sync(0xFFFFFFFF, max_values[tid], offset);
        shfl_idx = __shfl_down_sync(0xFFFFFFFF, max_indices[tid], offset);
        __syncwarp();

        if (shfl_val > max_values[tid]) {
            max_values[tid] = shfl_val;
            max_indices[tid] = shfl_idx;
        }
    }

    // if (max_values[tid] < max_values[tid + 32]) {
    //             max_values[tid] = max_values[tid + 32];
    //             max_indices[tid] = max_indices[tid + 32];
    //     }
    // if (max_values[tid] < max_values[tid + 16]) {
    //             max_values[tid] = max_values[tid + 16];
    //             max_indices[tid] = max_indices[tid + 16];
    //     } 
    // if (max_values[tid] < max_values[tid + 8]) {
    //             max_values[tid] = max_values[tid + 8];
    //             max_indices[tid] = max_indices[tid + 8];
    //     } 
    // if (max_values[tid] < max_values[tid + 4]) {
    //             max_values[tid] = max_values[tid + 4];
    //             max_indices[tid] = max_indices[tid + 4];
    //     }
    // if (max_values[tid] < max_values[tid + 2]) {
    //             max_values[tid] = max_values[tid + 2];
    //             max_indices[tid] = max_indices[tid + 2];
    //     } 
    // if (max_values[tid] < max_values[tid + 1]) {
    //             max_values[tid] = max_values[tid + 1];
    //             max_indices[tid] = max_indices[tid + 1];
    //     }      
}

template<>
__device__ void warpReduce<__nv_bfloat162, int2>(
            volatile __nv_bfloat162* max_values,
            volatile int2* max_indices,
            int tid)
{
    __nv_bfloat162 shfl_val;
    int2 shfl_idx;

    // unsigned mask = __ballot_sync(0xffffffff, tid < 32); // warpSize is 32

    for (int offset = 16; offset > 0; offset >>=1) {

        shfl_val.x = __shfl_down_sync(0xFFFFFFFF, max_values[tid].x, offset);
        shfl_val.y = __shfl_down_sync(0xFFFFFFFF, max_values[tid].y, offset);
        shfl_idx.x = __shfl_down_sync(0xFFFFFFFF, max_indices[tid].x, offset);
        shfl_idx.y = __shfl_down_sync(0xFFFFFFFF, max_indices[tid].y, offset);

        __syncwarp();

        if (__hlt(max_values[tid].x, shfl_val.x)) {
            max_values[tid].x = shfl_val.x;
            max_indices[tid].x = shfl_idx.x;
        }
        if (__hlt(max_values[tid].y, shfl_val.y)) {
            max_values[tid].y = shfl_val.y;
            max_indices[tid].y = shfl_idx.y;
        }
    }

    // if (__hlt(max_values[tid].x, max_values[tid + 32].x)) {
    //     max_values[tid].x = max_values[tid + 32].x;
    //     max_indices[tid].x = max_indices[tid + 32].x;
    // }
    // if (__hlt(max_values[tid].y, max_values[tid + 32].y)) {
    //     max_values[tid].y = max_values[tid + 32].y;
    //     max_indices[tid].y = max_indices[tid + 32].y;
    // }

    // if (__hlt(max_values[tid].x, max_values[tid + 16].x)) {
    //     max_values[tid].x = max_values[tid + 16].x;
    //     max_indices[tid].x = max_indices[tid + 16].x;
    // }
    // if (__hlt(max_values[tid].y, max_values[tid + 16].y)) {
    //     max_values[tid].y = max_values[tid + 16].y;
    //     max_indices[tid].y = max_indices[tid + 16].y;
    // }

    // if (__hlt(max_values[tid].x, max_values[tid + 8].x)) {
    //     max_values[tid].x = max_values[tid + 8].x;
    //     max_indices[tid].x = max_indices[tid + 8].x;
    // }
    // if (__hlt(max_values[tid].y, max_values[tid + 8].y)) {
    //     max_values[tid].y = max_values[tid + 8].y;
    //     max_indices[tid].y = max_indices[tid + 8].y;
    // }

    // if (__hlt(max_values[tid].x, max_values[tid + 4].x)) {
    //     max_values[tid].x = max_values[tid + 4].x;
    //     max_indices[tid].x = max_indices[tid + 4].x;
    // }
    // if (__hlt(max_values[tid].y, max_values[tid + 4].y)) {
    //     max_values[tid].y = max_values[tid + 4].y;
    //     max_indices[tid].y = max_indices[tid + 4].y;
    // }

    // if (__hlt(max_values[tid].x, max_values[tid + 2].x)) {
    //     max_values[tid].x = max_values[tid + 2].x;
    //     max_indices[tid].x = max_indices[tid + 2].x;
    // }
    // if (__hlt(max_values[tid].y, max_values[tid + 2].y)) {
    //     max_values[tid].y = max_values[tid + 2].y;
    //     max_indices[tid].y = max_indices[tid + 2].y;
    // }
    
    // if (__hlt(max_values[tid].x, max_values[tid + 1].x)) {
    //     max_values[tid].x = max_values[tid + 1].x;
    //     max_indices[tid].x = max_indices[tid + 1].x;
    // }
    // if (__hlt(max_values[tid].y, max_values[tid + 1].y)) {
    //     max_values[tid].y = max_values[tid + 1].y;
    //     max_indices[tid].y = max_indices[tid + 1].y;
    // }

}

// SDP_CUDA_KERNEL(warpReduce)


// find the maximum value in a list using reduction
template<typename T, typename I>
__global__ void find_maximum_value(
            const T *input,
            I *index_in,
            T *output,
            I *index_out,
            const int elements_per_thread,
            const int num_elements,
            bool init_idx)
{    

}

template<>
__global__ void find_maximum_value<double, int>(
            const double *input,
            int *index_in,
            double *output,
            int *index_out,
            const int elements_per_thread,
            const int num_elements,
            bool init_idx)
{
    __shared__ double max_values[256];
    __shared__ int max_indices[256];

    // thread index
    int64_t tid = threadIdx.x;
    // index of block of values being worked on by this thread
    int64_t super_idx = blockIdx.x * elements_per_thread * blockDim.x;

    // initialise max_values to zero
    max_values[tid] = (double)0.0;

    if(super_idx + tid < num_elements){
        // load input to shared mem
        max_values[tid] = input[super_idx + tid];
        // check if idx was initalised in a previous kernel
        max_indices[tid] = (init_idx == true) ? index_in[super_idx + tid] : super_idx + tid;
    }

    #pragma unroll
    for (int i = 1; i < elements_per_thread; i++){
        // current index being compared to value in shared mem
        int64_t curr_idx = super_idx + i * blockDim.x + tid;
        if (curr_idx < num_elements)
        {
            double val = max_values[tid];
            double next_val = input[curr_idx];
            if (next_val > val){
                max_values[tid] = next_val;
                max_indices[tid] = (init_idx == true) ? index_in[curr_idx] : curr_idx;
            }
        }
    }
    __syncthreads();

    #pragma unroll 
    for (int s = blockDim.x / 2; s > 16; s >>= 1){
        if (tid < s){
            if (max_values[tid] < max_values[tid + s]) {
                max_values[tid] = max_values[tid + s];
                max_indices[tid] = max_indices[tid + s];
            }
        }
        __syncthreads();
    }

    // if (tid < 128) {
    //         if (max_values[tid] < max_values[tid + 128]) {
    //             max_values[tid] = max_values[tid + 128];
    //             max_indices[tid] = max_indices[tid + 128];
    //         }
    //     }
    // __syncthreads();

    // if (tid < 64) {
    //         if (max_values[tid] < max_values[tid + 64]) {
    //             max_values[tid] = max_values[tid + 64];
    //             max_indices[tid] = max_indices[tid + 64];
    //         }
    //     }
    // __syncthreads();

    if (tid < 16) warpReduce<double, int>(max_values, max_indices, tid);

    // Write the final result to output
    if (tid == 0) {
        output[blockIdx.x] = max_values[0];
        index_out[blockIdx.x] = max_indices[0];
    }
    
}

template<>
__global__ void find_maximum_value<float, int>(
            const float *input,
            int *index_in,
            float *output,
            int *index_out,
            const int elements_per_thread,
            const int num_elements,
            bool init_idx)
{
    __shared__ float max_values[256];
    __shared__ int max_indices[256];


    // thread index
    int64_t tid = threadIdx.x;
    // index of block of values being worked on by this thread
    int64_t super_idx = blockIdx.x * elements_per_thread * blockDim.x;
    
    // initialise max_values to zero
    max_values[tid] = 0.0f;

    if(super_idx + tid < num_elements){
        // load input to shared mem
        max_values[tid] = input[super_idx + tid];
        // check if idx was initalised in a previous kernel
        max_indices[tid] = (init_idx == true) ? index_in[super_idx + tid] : super_idx + tid;
    }

    #pragma unroll
    for (int i = 1; i < elements_per_thread; i++){
        // current index being compared to value in shared mem
        int64_t curr_idx = super_idx + i * blockDim.x + tid;
        if (curr_idx < num_elements)
        {
            float val = max_values[tid];
            float next_val = input[curr_idx];
            if (next_val > val){
                max_values[tid] = next_val;
                max_indices[tid] = (init_idx == true) ? index_in[curr_idx] : curr_idx;
            }
        }
    }
    __syncthreads();

    #pragma unroll 
    for (int s = blockDim.x / 2; s > 16; s >>= 1){
        if (tid < s){
            if (max_values[tid] < max_values[tid + s]) {
                max_values[tid] = max_values[tid + s];
                max_indices[tid] = max_indices[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid < 16) warpReduce<float, int>(max_values, max_indices, tid);

    // Write the final result to output
    if (tid == 0) {
        output[blockIdx.x] = max_values[0];
        index_out[blockIdx.x] = max_indices[0];
    }
}

template<>
__global__ void find_maximum_value<__nv_bfloat162, int2>(
            const __nv_bfloat162 *input,
            int2 *index_in,
            __nv_bfloat162 *output,
            int2 *index_out,
            const int elements_per_thread,
            const int num_elements,
            bool init_idx)
{
    __shared__ __nv_bfloat162 max_values[256];
    __shared__ int2 max_indices[256];

    // thread index
    int64_t tid = threadIdx.x;
    // index of block of values being worked on by this thread
    int64_t super_idx = blockIdx.x * elements_per_thread * blockDim.x;

    // initialise max_values to zero
    max_values[tid] =  __floats2bfloat162_rn (0.0f, 0.0f);

    if(super_idx + tid < num_elements){
        // load input to shared mem
        max_values[tid] = input[super_idx + tid];
        // check if idx was initalised in a previous kernel
        max_indices[tid].x = (init_idx == true) ? index_in[super_idx + tid].x : 2 * (super_idx + tid);
        max_indices[tid].y = (init_idx == true) ? index_in[super_idx + tid].y : 2 * (super_idx + tid) + 1;
    }

    #pragma unroll
    for (int i = 1; i < elements_per_thread; i++){
        // current index being compared to value in shared mem
        int64_t curr_idx = super_idx + i * blockDim.x + tid;
        if (curr_idx < num_elements){
            __nv_bfloat162 val = max_values[tid];
            __nv_bfloat162 next_val = input[curr_idx];
            if (__hlt(val.x, next_val.x)){
                max_values[tid].x = next_val.x;
                max_indices[tid].x = (init_idx == true) ? index_in[curr_idx].x : 2 * curr_idx;
            }

            if (__hlt(val.y, next_val.y)){
                max_values[tid].y = next_val.y;
                max_indices[tid].y = (init_idx == true) ? index_in[curr_idx].y : 2 * curr_idx + 1;
            }
        }
    }
    __syncthreads();

    #pragma unroll 
    for (int s = blockDim.x / 2; s > 16; s >>= 1){
        if (tid < s){
            if (__hlt(max_values[tid].x, max_values[tid + s].x)) {
                max_values[tid].x = max_values[tid + s].x;
                max_indices[tid].x = max_indices[tid + s].x;
            }

            if (__hlt(max_values[tid].y, max_values[tid + s].y)) {
                max_values[tid].y = max_values[tid + s].y;
                max_indices[tid].y = max_indices[tid + s].y;
            }
        }
        __syncthreads();
    }

    // if (tid < 128) {
    //     if (__hlt(max_values[tid].x, max_values[tid + 128].x)) {
    //         max_values[tid].x = max_values[tid + 128].x;
    //         max_indices[tid].x = max_indices[tid + 128].x;
    //     }

    //     if (__hlt(max_values[tid].y, max_values[tid + 128].y)) {
    //         max_values[tid].y = max_values[tid + 128].y;
    //         max_indices[tid].y = max_indices[tid + 128].y;
    //     }
    // }
    // __syncthreads();

    // if (tid < 64) {
    //     if (__hlt(max_values[tid].x, max_values[tid + 64].x)) {
    //         max_values[tid].x = max_values[tid + 64].x;
    //         max_indices[tid].x = max_indices[tid + 64].x;
    //     }

    //     if (__hlt(max_values[tid].y, max_values[tid + 64].y)) {
    //         max_values[tid].y = max_values[tid + 64].y;
    //         max_indices[tid].y = max_indices[tid + 64].y;
    //     }
    // }
    // __syncthreads();

    if (tid < 16) warpReduce<__nv_bfloat162, int2>(max_values, max_indices, tid);

    // Write the final result to output
    if (tid == 0) {
        output[blockIdx.x] = max_values[0];
        index_out[blockIdx.x] = max_indices[0];
    } 
}

SDP_CUDA_KERNEL(find_maximum_value<double, int>);
SDP_CUDA_KERNEL(find_maximum_value<float, int>);
SDP_CUDA_KERNEL(find_maximum_value<__nv_bfloat162, int2>);


__global__ void overall_maximum_value_bfloat(
            const __nv_bfloat162 *in,
            int2 *index_in,
            __nv_bfloat16 *out,
            int *index_out)
{

    if(__hgt(in[0].y, in[0].x)){
        *out = in[0].x;
        *index_out = index_in[0].x;
    }  
    else{
        *out = in[0].y;
        *index_out = index_in[0].y;
    }  
    
    // if(__hgt(in[0].x, in[0].y)){
    //     *out = in[0].x;
    //     *index_out = index_in[0].x * 2;
    // }
    // else{
    //     *out = in[0].y;
    //     *index_out = index_in[0].y * 2 + 1;
    // }

}

SDP_CUDA_KERNEL(overall_maximum_value_bfloat);


// add a component to the CLEAN component list
template<typename T, typename T2>
__global__ void add_clean_comp(
            T2* clean_comp,
            int* max_idx_flat,
            T* loop_gain,
            T* highest_value,
            T* threshold,
            int* thresh_reached
){

}

template<>
__global__ void add_clean_comp<double, double>(
            double* clean_comp,
            int* max_idx_flat,
            double* loop_gain,
            double* highest_value,
            double* threshold,
            int* thresh_reached
){
    // check threshold
    if (highest_value[0] > threshold[0] && *thresh_reached == 0){
        
        // Add fraction of maximum to clean components list
        double inter = __dmul_rn(loop_gain[0], highest_value[0]);
        clean_comp[max_idx_flat[0]] = __dadd_rn(clean_comp[max_idx_flat[0]], inter);
        // clean_comp[max_idx_flat[0]] = clean_comp[max_idx_flat[0]] + (loop_gain[0] * highest_value[0]);

    }
    // if threshold reached, set flag
    else{
        *thresh_reached = 1;
    }

}

template<>
__global__ void add_clean_comp<float, float>(
            float* clean_comp,
            int* max_idx_flat,
            float* loop_gain,
            float* highest_value,
            float* threshold,
            int* thresh_reached
){
    // check threshold
    if (highest_value[0] > threshold[0] && *thresh_reached == 0){
        
        // Add fraction of maximum to clean components list
        float inter = __fmul_rn(loop_gain[0], highest_value[0]);
        clean_comp[max_idx_flat[0]] = __fadd_rn(clean_comp[max_idx_flat[0]], inter);
        // clean_comp[max_idx_flat[0]] = clean_comp[max_idx_flat[0]] + (loop_gain[0] * highest_value[0]);

    }
    // if threshold reached, set flag
    else{
        *thresh_reached = 1;
    }

}

template<>
__global__ void add_clean_comp<__nv_bfloat16, __nv_bfloat162>(
            __nv_bfloat162* clean_comp,
            int* max_idx_flat,
            __nv_bfloat16* loop_gain,
            __nv_bfloat16* highest_value,
            __nv_bfloat16* threshold,
            int* thresh_reached
){
    // check threshold
    if (__hgt(highest_value[0], threshold[0]) && *thresh_reached == 0){

        int bfloat_idx = *max_idx_flat / 2;
        
        // Add fraction of maximum to clean components list
        __nv_bfloat16 inter = __hmul(loop_gain[0], highest_value[0]);

        if (*max_idx_flat % 2 == 0){
            clean_comp[bfloat_idx].x = __hadd(clean_comp[bfloat_idx].x, inter);
        }
        else{
            clean_comp[bfloat_idx].y = __hadd(clean_comp[bfloat_idx].y, inter);
        }
    }
    // if threshold reached, set flag
    else{
        *thresh_reached = 1;
    }

}

SDP_CUDA_KERNEL(add_clean_comp<double, double>);
SDP_CUDA_KERNEL(add_clean_comp<float, float>);
SDP_CUDA_KERNEL(add_clean_comp<__nv_bfloat16, __nv_bfloat162>);


// subtract the psf from the residual image
template<typename T, typename T2>
__global__ void subtract_psf(
            int64_t dirty_img_dim, 
            int64_t psf_dim, 
            T* loop_gain, 
            int* max_idx_flat, 
            T* highest_value, 
            const int elements_per_thread,
            const T2* psf, 
            T2* residual,
            T* threshold
) {
            
}

template<>
__global__ void subtract_psf<double, double>(
            int64_t dirty_img_dim, 
            int64_t psf_dim, 
            double* loop_gain, 
            int* max_idx_flat, 
            double* highest_value, 
            const int elements_per_thread,
            const double* psf, 
            double* residual,
            double* threshold) {

    // check threshold
    if (highest_value[0] > threshold[0]){
        int64_t dirty_img_size = dirty_img_dim * dirty_img_dim;
        // int64_t psf_size = psf_dim * psf_dim;

        // get x and y from flat index
        int max_idx_x = max_idx_flat[0] / dirty_img_dim;
        int max_idx_y = max_idx_flat[0] % dirty_img_dim;

        // Identify start position of PSF window to subtract from residual
        int64_t psf_x_start = dirty_img_dim - max_idx_x;
        int64_t psf_y_start = dirty_img_dim - max_idx_y;

        // int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

        // thread index
        int64_t tid = threadIdx.x;
        // index of block of values being worked on by this thread
        int64_t super_idx = blockIdx.x * elements_per_thread * blockDim.x;

        #pragma unroll
        for (int i = 0; i < elements_per_thread; i++){
            int64_t curr_idx = super_idx + i * blockDim.x + tid;

            // check thread is in bounds
            if (curr_idx < dirty_img_size){

                // Compute the x and y coordinates in the dirty image
                int64_t x_dirty = curr_idx / dirty_img_dim;
                int64_t y_dirty = curr_idx % dirty_img_dim;

                // Compute the x and y coordinates in the psf
                int64_t x_psf = x_dirty + psf_x_start;
                int64_t y_psf = y_dirty + psf_y_start;

                // // get flat index for dirty image
                // int64_t dirty_img_flat_idx = x_dirty * dirty_img_dim + y_dirty;

                // get flat index for psf
                int64_t psf_flat_idx = x_psf * psf_dim + y_psf;

                // Subtract the PSF contribution from the residual
                double inter = __dmul_rn(loop_gain[0], highest_value[0]);
                inter = __dmul_rn(inter, psf[psf_flat_idx]);
                residual[curr_idx] =  __dsub_rn(residual[curr_idx],inter);
                
                // residual[i] = residual[i] - (loop_gain[0] * highest_value[0] * psf[psf_flat_idx]);
            }
        }
    }
    else{
        return;
    }
}


template<>
__global__ void subtract_psf<float, float>(
            int64_t dirty_img_dim, 
            int64_t psf_dim, 
            float* loop_gain, 
            int* max_idx_flat, 
            float* highest_value, 
            const int elements_per_thread,
            const float* psf, 
            float* residual,
            float* threshold) {

        // check threshold
    if (highest_value[0] > threshold[0]){

        int64_t dirty_img_size = dirty_img_dim * dirty_img_dim;
        // int64_t psf_size = psf_dim * psf_dim;

        // get x and y from flat index
        int max_idx_x = max_idx_flat[0] / dirty_img_dim;
        int max_idx_y = max_idx_flat[0] % dirty_img_dim;

        // Identify start position of PSF window to subtract from residual
        int64_t psf_x_start = dirty_img_dim - max_idx_x;
        int64_t psf_y_start = dirty_img_dim - max_idx_y;

        // int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

        // thread index
        int64_t tid = threadIdx.x;
        // index of block of values being worked on by this thread
        int64_t super_idx = blockIdx.x * elements_per_thread * blockDim.x;

        #pragma unroll
        for (int i = 0; i < elements_per_thread; i++){
            int64_t curr_idx = super_idx + i * blockDim.x + tid;

            // check thread is in bounds
            if (curr_idx < dirty_img_size){

                // Compute the x and y coordinates in the dirty image
                int64_t x_dirty = curr_idx / dirty_img_dim;
                int64_t y_dirty = curr_idx % dirty_img_dim;

                // Compute the x and y coordinates in the psf
                int64_t x_psf = x_dirty + psf_x_start;
                int64_t y_psf = y_dirty + psf_y_start;

                // // get flat index for dirty image
                // int64_t dirty_img_flat_idx = x_dirty * dirty_img_dim + y_dirty;

                // get flat index for psf
                int64_t psf_flat_idx = x_psf * psf_dim + y_psf;

                // Subtract the PSF contribution from the residual
                float inter = __fmul_rn(loop_gain[0], highest_value[0]);
                inter = __fmul_rn(inter, psf[psf_flat_idx]);
                residual[curr_idx] =  __fsub_rn(residual[curr_idx],inter);
                
                // residual[i] = residual[i] - (loop_gain[0] * highest_value[0] * psf[psf_flat_idx]);
            }
        }
    }
    else{
        return;
    }
}

template<>
__global__ void subtract_psf<__nv_bfloat16, __nv_bfloat162>(
            int64_t dirty_img_dim, 
            int64_t psf_dim, 
            __nv_bfloat16* loop_gain, 
            int* max_idx_flat, 
            __nv_bfloat16* highest_value, 
            const int elements_per_thread,
            const __nv_bfloat162* psf, 
            __nv_bfloat162* residual,
            __nv_bfloat16* threshold) {

    // check threshold
    if (__hgt(highest_value[0], threshold[0])){

        int64_t dirty_img_size = dirty_img_dim * dirty_img_dim;
        // int64_t psf_size = psf_dim * psf_dim;

        // get x and y from flat index
        int max_idx_x = max_idx_flat[0] / dirty_img_dim;
        int max_idx_y = max_idx_flat[0] % dirty_img_dim;

        // Identify start position of PSF window to subtract from residual
        int64_t psf_x_start = dirty_img_dim - max_idx_x;
        int64_t psf_y_start = dirty_img_dim - max_idx_y;

        // int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

        // thread index
        int64_t tid = threadIdx.x;
        // index of block of values being worked on by this thread
        int64_t super_idx = blockIdx.x * elements_per_thread * blockDim.x;

        #pragma unroll
        for (int i = 0; i < elements_per_thread; i++){
            int64_t curr_idx = super_idx + i * blockDim.x + tid;

            // check thread is in bounds
            if (curr_idx < dirty_img_size){

                // Compute the x and y coordinates in the dirty image
                int64_t x_dirty = curr_idx / dirty_img_dim;
                int64_t y_dirty = curr_idx % dirty_img_dim;

                // Compute the x and y coordinates in the psf
                int64_t x_psf = x_dirty + psf_x_start;
                int64_t y_psf = y_dirty + psf_y_start;

                // // get flat index for dirty image
                // int64_t dirty_img_flat_idx = x_dirty * dirty_img_dim + y_dirty;

                // get flat index for psf
                int64_t psf_flat_idx = x_psf * psf_dim + y_psf;

                // find index in bfloat terms
                int64_t bfloat_psf_idx = psf_flat_idx / 2;
                int64_t bfloat_residual_idx = curr_idx / 2;

                // Subtract the PSF contribution from the residual
                __nv_bfloat16 inter = __hmul(*loop_gain, highest_value[0]);
                
                // if psf_flat_index / 2 is even then we have .x, if odd then .y
                if (psf_flat_idx % 2 == 0){
                    inter = __hmul(inter, psf[bfloat_psf_idx].x);
                }
                else{
                    inter = __hmul(inter, psf[bfloat_psf_idx].y);
                }
                
                if (curr_idx % 2 == 0){
                    residual[bfloat_residual_idx].x =  __hsub(residual[bfloat_residual_idx].x,inter);
                }
                else
                {
                    residual[bfloat_residual_idx].y =  __hsub(residual[bfloat_residual_idx].y,inter);
                }
                
                // residual[i] = residual[i] - (loop_gain[0] * highest_value[0] * psf[psf_flat_idx]);
            }
        }
    }
    else{
        return;
    }
}

SDP_CUDA_KERNEL(subtract_psf<double, double>);
SDP_CUDA_KERNEL(subtract_psf<float, float>);
SDP_CUDA_KERNEL(subtract_psf<__nv_bfloat16, __nv_bfloat162>);


// add the final residual image to the skymodel
template<typename T>
__global__ void add_residual(
            T* in,
            int64_t size,
            T* out
){
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (size)){
        out[i] = out[i] + in[i];
    }
}

SDP_CUDA_KERNEL(add_residual<double>);
SDP_CUDA_KERNEL(add_residual<float>);


// create a copy of a real value as a complex value with imaginary part set to 0
template<typename T, typename CT>
__global__ void create_copy_complex(
    const T* in,
    int64_t size,
    CT* out
) {

}

template<>
__global__ void create_copy_complex<double, cuDoubleComplex>(
    const double* in,
    int64_t size,
    cuDoubleComplex* out
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size){
        out[i] = make_cuDoubleComplex(in[i], 0);
    }
}

template<>
__global__ void create_copy_complex<float, cuFloatComplex>(
    const float* in,
    int64_t size,
    cuFloatComplex* out
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size){
        out[i] = make_cuFloatComplex(in[i], 0);
    }
}

SDP_CUDA_KERNEL(create_copy_complex<double, cuDoubleComplex>);
SDP_CUDA_KERNEL(create_copy_complex<float, cuFloatComplex>);


// convert a number from bfloat162 to float or double precision
template<typename T>
__global__ void convert_from_bfloat(
    const __nv_bfloat162* in,
    int64_t size_out,
    T* out
){

}

template<>
__global__ void convert_from_bfloat<float>(
    const __nv_bfloat162* in,
    int64_t size_out,
    float* out
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i * 2 < size_out) { // Check if the first element of the pair is within the array
        out[2*i] = __low2float(in[i]);

        if (2*i + 1 < size_out){ // Check if the second element of the pair is within the array
            out[2*i + 1] = __high2float(in[i]);
        }
    }
}

template<>
__global__ void convert_from_bfloat<double>(
    const __nv_bfloat162* in,
    int64_t size_out,
    double* out
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i * 2 < size_out) { // Check if the first element of the pair is within the array
        out[2*i] = (double)__low2float(in[i]);

        if (2*i + 1 < size_out){ // Check if the second element of the pair is within the array
            out[2*i + 1] = (double)__high2float(in[i]);
        }
    }
}

SDP_CUDA_KERNEL(convert_from_bfloat<double>);
SDP_CUDA_KERNEL(convert_from_bfloat<float>);




// cpnvert a double or single precision number to bfloat162
template<typename T>
__global__ void convert_to_bfloat(
    const T* in,
    int64_t size,
    __nv_bfloat162* out
){

}

template<>
__global__ void convert_to_bfloat<float>(
    const float* in,
    int64_t size,
    __nv_bfloat162* out
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i * 2 < size) { // Check if the first element of the pair is within the array
        if (2*i + 1 < size){ // Check if the second element of the pair is within the array
            out[i] = __floats2bfloat162_rn(in[2*i], in[2*i + 1]);
        }
        else{
            out[i] = __floats2bfloat162_rn(in[2*i], 0.0f);
        }
    }
}

template<>
__global__ void convert_to_bfloat<double>(
    const double* in,
    int64_t size,
    __nv_bfloat162* out
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i * 2 < size) { // Check if the first element of the pair is within the array
        __nv_bfloat16 first = __double2bfloat16(in[2*i]);
        __nv_bfloat16 second;

        if (2*i + 1 < size){ // Check if the second element of the pair is within the array
            second = __double2bfloat16(in[2*i + 1]);
        }
        else{
            second = __double2bfloat16(0.0) ;
        }
        
        out[i] = __halves2bfloat162(first, second);
    }
}

SDP_CUDA_KERNEL(convert_to_bfloat<double>);
SDP_CUDA_KERNEL(convert_to_bfloat<float>);


// copy one value to gpu, can convert to bfloat from float or double precision
// needed to convert loop gain and threshold to bfloat16 for correct use in maths with bfloat16 data 
template<typename T, typename OT>
__global__ void copy_var_gpu(
    T in,
    OT* out
){

}

template<>
__global__ void copy_var_gpu<double, double>(
    double in,
    double* out
){
    out[0] = in;
}

template<>
__global__ void copy_var_gpu<float, float>(
    float in,
    float* out
){
    out[0] = in;
}

template<>
__global__ void copy_var_gpu<double, __nv_bfloat16>(
    double in,
    __nv_bfloat16* out
){
    out[0] = __double2bfloat16(in);
}

template<>
__global__ void copy_var_gpu<float, __nv_bfloat16>(
    float in,
    __nv_bfloat16* out
){
    out[0] = __float2bfloat16(in);
}

SDP_CUDA_KERNEL(copy_var_gpu<double, double>);
SDP_CUDA_KERNEL(copy_var_gpu<float, float>);
SDP_CUDA_KERNEL(copy_var_gpu<double, __nv_bfloat16>);
SDP_CUDA_KERNEL(copy_var_gpu<float, __nv_bfloat16>);


__global__ void bf16_2_double(
    __nv_bfloat16* in,
    double* out
){
    *out = (double)__bfloat162float(*in);
}

SDP_CUDA_KERNEL(bf16_2_double)
// // max finding atomic experiment
// __device__ __forceinline__ void my_atomic_max(double* addr, double value)
// {

//     unsigned long long int* laddr = (unsigned long long int*)(addr);
//     unsigned long long int assumed, old_ = *laddr;
//     do
//     {
//         assumed = old_;
//         old_ = atomicCAS(laddr,
//                 assumed,
//                 __double_as_longlong(max(value,
//                 __longlong_as_double(assumed)))
//                 );
//     }
//     while (assumed != old_);

// }


// typedef union  {
//   double floats[2];                 // floats[0] = lowest
//   int ints[2];                     // ints[1] = lowIdx
//   unsigned long long int ulong;    // for atomic update
// } my_atomics;

// __device__ my_atomics test;

// __device__ unsigned long long int my_atomicMin(unsigned long long int* address, double val1, int val2)
// {
//     my_atomics loc, loctest;
//     loc.floats[0] = val1;
//     loc.ints[1] = val2;
//     loctest.ulong = *address;
//     while (loctest.floats[0] <  val1) 
//       loctest.ulong = atomicCAS(address, loctest.ulong,  loc.ulong);
//     return loctest.ulong;
// }



// __global__ void find_maximum_value_atomic(
//             double* residual, 
//             int64_t dirty_img_size,
//             double* highest_value, 
//             int* max_idx_flat){

//     __shared__ double shared_values[256];
//     __shared__ int shared_indices[256];

//     int tid = threadIdx.x;
//     int i = blockIdx.x * blockDim.x + threadIdx.x;

//     // Load input elements into shared memory
//     shared_values[tid] = residual[i];
//     shared_indices[tid] = i;

//     __syncthreads();

//     // Perform reduction
//     for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
//         if (tid < stride) {
//             if (shared_values[tid + stride] > shared_values[tid]) {
//                 shared_values[tid] = shared_values[tid + stride];
//                 shared_indices[tid] = shared_indices[tid + stride];
//             }
//         }
//         __syncthreads();
//     }

//     // Update global maximum value and index
//     if (tid == 0) {

//         // my_atomic_max(&highest_value[0],shared_values[0]);

//         my_atomicMin(&(test.ulong), highest_value[0],max_idx_flat[0]);
//     }

//     highest_value[0] = *test.floats;
//     // highest_value[1] = *test.ints;


//     // // Write the final result to output
//     // if (tid == 0) {
//     //     output[blockIdx.x] = max_values[0];
//     //     index[blockIdx.x] = max_indices[0];
//     // }


// }
// SDP_CUDA_KERNEL(find_maximum_value_atomic)








// previous code
// __device__ __forceinline__ void my_atomic_max(double* addr, double value)
// {

//     unsigned long long int* laddr = (unsigned long long int*)(addr);
//     unsigned long long int assumed, old_ = *laddr;
//     do
//     {
//         assumed = old_;
//         old_ = atomicCAS(laddr,
//                 assumed,
//                 __double_as_longlong(max(value,
//                 __longlong_as_double(assumed)))
//                 );
//     }
//     while (assumed != old_);

// }


// __device__ void find_maximum_value(
//             double* residual, 
//             int64_t dirty_img_size,
//             double* highest_value, 
//             int* max_idx_flat,
//             double* skymodel){

//     __shared__ double shared_values[256];
//     __shared__ int shared_indices[256];

//     int tid = threadIdx.x;
//     int i = blockIdx.x * blockDim.x + threadIdx.x;

//     // Load input elements into shared memory
//     shared_values[tid] = residual[i];
//     shared_indices[tid] = i;

//     __syncthreads();

//     // Perform reduction
//     for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
//         if (tid < stride) {
//             if (shared_values[tid + stride] > shared_values[tid]) {
//                 shared_values[tid] = shared_values[tid + stride];
//                 shared_indices[tid] = shared_indices[tid + stride];
//             }
//         }
//         __syncthreads();
//     }

//     // Update global maximum value and index
//     if (tid == 0) {
//         // if (shared_values[0] > *highest_value){
//         //     *highest_value = shared_values[0];
//         //     *max_idx_flat = shared_indices[0];

//         // }

//         skymodel[blockIdx.x] = shared_values[0];
//         // my_atomic_max(highest_value,shared_values[0]);
//         // *max_idx_flat = shared_indices[0];

//     }


// }

// __device__ void subtract_psf(
//             int64_t dirty_img_dim, 
//             int64_t psf_dim, 
//             int64_t psf_x_start, 
//             int64_t psf_y_start, 
//             double loop_gain, 
//             double highest_value, 
//             const double* psf, 
//             double* residual,
//             double* skymodel) {

//     int64_t dirty_img_size = dirty_img_dim * dirty_img_dim;
//     int64_t psf_size = psf_dim * psf_dim;

//     int i = blockIdx.x * blockDim.x + threadIdx.x;

//     if (i < dirty_img_size){

//         // Compute the x and y coordinates in the dirty image
//         int64_t x_dirty = i / dirty_img_dim;
//         int64_t y_dirty = i % dirty_img_dim;

//         // Compute the x and y coordinates in the psf
//         int64_t x_psf = x_dirty + psf_x_start;
//         int64_t y_psf = y_dirty + psf_y_start;

//         // // get flat index for dirty image
//         // int64_t dirty_img_flat_idx = x_dirty * dirty_img_dim + y_dirty;

//         // get flat index for psf
//         int64_t psf_flat_idx = x_psf * psf_dim + y_psf;

//         // Subtract the PSF contribution from the residual
//         residual[i] -= (loop_gain * highest_value * psf[psf_flat_idx]);

//         // skymodel[i] = psf[psf_flat_idx];

//     }
// }


// __device__ void create_copy_complex(
//     const double* in,
//     int64_t size,
//     cuDoubleComplex* out
// ) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;

//     if (i < size){
//         out[i] = make_cuDoubleComplex(in[i], 0);
//     }
// }

// __device__ void copy_to_skymodel(
//     const double* in,
//     int64_t size,
//     double* out
// ) {
//     for (int i = 0; i < size; i++){
//         out[i] = in[i];
//     }
// }


// __global__ void Hogbom_clean(
//             const double* psf,
//             const double loop_gain,
//             const double threshold,
//             const double cycle_limit,
//             double* residual,
//             const int64_t dirty_img_dim,
//             const int64_t psf_dim,
//             double* clean_comp,
//             cuDoubleComplex* clean_comp_complex,
//             double* skymodel
// ){
//     // calculate useful shapes and sizes
//     int64_t dirty_img_size = dirty_img_dim * dirty_img_dim;
//     int64_t psf_size = psf_dim * psf_dim;

//     // set up some loop variables
//     int cur_cycle = 0;
//     bool stop = 0;

//     // CLEAN loop executes while the stop conditions (threshold and cycle limit) are not met
//     while (cur_cycle < 1 && !stop) {
//         // Find index and value of the maximum value in residual
//         double highest_value = 0.0;
//         int max_idx_flat = 0;

//         find_maximum_value(residual, dirty_img_size, &highest_value, &max_idx_flat, clean_comp);
//         copy_to_skymodel(clean_comp, dirty_img_size, skymodel);
//         // find_maximum_value(clean_comp, dirty_img_size, &highest_value, &max_idx_flat, skymodel);


//         // Check maximum value against threshold
//         if (highest_value < threshold) {
//             stop = true;
//             break;
//         }

//         // int max_idx_flat_test = 256 *1024 + 256;
//         // skymodel[0] = (double)max_idx_flat;
//         // skymodel[1] = (double)max_idx_flat_test;

//         // get x and y from flat index
//         int max_idx_x = max_idx_flat / dirty_img_dim;
//         int max_idx_y = max_idx_flat % dirty_img_dim;

//         // highest_value = 10.0;

//         // skymodel[2] = (double)max_idx_x;
//         // skymodel[3] = (double)max_idx_y;
//         // skymodel[4] = highest_value;

//         // Add fraction of maximum to clean components list
//         clean_comp[max_idx_flat] += (loop_gain * highest_value);

//         // Identify start position of PSF window to subtract from residual
//         int64_t psf_x_start = dirty_img_dim - max_idx_x;
//         int64_t psf_y_start = dirty_img_dim - max_idx_y;

//         subtract_psf(dirty_img_dim, psf_dim, psf_x_start, psf_y_start, loop_gain, highest_value, psf, residual,skymodel);

//         cur_cycle += 1;
//     }

//     // copy_to_skymodel(clean_comp, dirty_img_size, skymodel);

//     // Convolution code only works with complex input, so make clean components and clean beam complex
//     create_copy_complex(clean_comp, dirty_img_size, clean_comp_complex);

// }

// SDP_CUDA_KERNEL(Hogbom_clean);


