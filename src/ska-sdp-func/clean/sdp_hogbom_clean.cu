/* See the LICENSE file at the top-level directory of this distribution. */

#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/numeric_functions/sdp_fft_convolution.h"
#include "ska-sdp-func/utility/sdp_mem.h"

#include <cuComplex.h>
#include <cuda_bf16.h>

// #define INDEX_2D(N2, N1, I2, I1)    (N1 * I2 + I1)

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


template<typename T>
__global__ void create_cbeam(
    const double* cbeam_details,
    int16_t psf_dim,
    T* cbeam
) {

}

template<>
__global__ void create_cbeam<cuDoubleComplex>(
    const double* cbeam_details,
    int16_t psf_dim,
    cuDoubleComplex* cbeam
) {
    // Fit a Gaussian to the main lobe of the PSF based on the parameters passed

    double A = 1;
    double x0 = (psf_dim / 2);
    double y0 = (psf_dim / 2);
    double sigma_X = cbeam_details[0];
    double sigma_Y = cbeam_details[1];
    double theta = (M_PI / 180) * cbeam_details[2];

    double a = pow(cos(theta), 2) / (2 * pow(sigma_X, 2)) + pow(sin(theta), 2) / (2 * pow(sigma_Y, 2));
    double b = sin(2 * theta) / (4 * pow(sigma_X, 2)) - sin(2 * theta) / (4 * pow(sigma_Y, 2));
    double c = pow(sin(theta), 2) / (2 * pow(sigma_X, 2)) + pow(cos(theta), 2) / (2 * pow(sigma_Y, 2));

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int size = psf_dim * psf_dim;

    if (i < size){ 
        int x = i / psf_dim;
        int y = i % psf_dim;

        double component = A * exp(-(a * pow(x - x0, 2) + 2 * b * (x - x0) * (y - y0) + c * pow(y - y0, 2)));
        cbeam[i] = make_cuDoubleComplex(component, 0);
    }
}

template<>
__global__ void create_cbeam<cuFloatComplex>(
    const double* cbeam_details,
    int16_t psf_dim,
    cuFloatComplex* cbeam
) {
    // Fit a Gaussian to the main lobe of the PSF based on the parameters passed

    double A = 1;
    double x0 = (psf_dim / 2);
    double y0 = (psf_dim / 2);
    double sigma_X = cbeam_details[0];
    double sigma_Y = cbeam_details[1];
    double theta = (M_PI / 180) * cbeam_details[2];

    double a = pow(cos(theta), 2) / (2 * pow(sigma_X, 2)) + pow(sin(theta), 2) / (2 * pow(sigma_Y, 2));
    double b = sin(2 * theta) / (4 * pow(sigma_X, 2)) - sin(2 * theta) / (4 * pow(sigma_Y, 2));
    double c = pow(sin(theta), 2) / (2 * pow(sigma_X, 2)) + pow(cos(theta), 2) / (2 * pow(sigma_Y, 2));

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int size = psf_dim * psf_dim;

    if (i < size){ 
        int x = i / psf_dim;
        int y = i % psf_dim;

        double component = A * exp(-(a * pow(x - x0, 2) + 2 * b * (x - x0) * (y - y0) + c * pow(y - y0, 2)));
        cbeam[i] = make_cuFloatComplex(component, 0);
    }
}

SDP_CUDA_KERNEL(create_cbeam<cuDoubleComplex>);
SDP_CUDA_KERNEL(create_cbeam<cuFloatComplex>);


template<typename T>
__global__ void find_maximum_value(
            const T *input,
            int *index_in,
            T *output,
            int *index_out,
            bool init_idx)
{
    __shared__ T max_values[256];
    __shared__ int max_indices[256];

    int64_t tid = threadIdx.x;
    int64_t i = blockIdx.x * (blockDim.x) + threadIdx.x;

    // Load input elements into shared memory
    max_values[tid] = input[i];
    // if index array has already been initialised then load it
    if(init_idx == true){
        max_indices[tid] = index_in[i];

    }
    // if it hasn't, initialise it.
    else{

        max_indices[tid] = i;

    }
    __syncthreads();


    // Perform reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (max_values[tid] < max_values[tid + stride]) {
                max_values[tid] = max_values[tid + stride];
                max_indices[tid] = max_indices[tid + stride];
            }
        }
        __syncthreads();
    }

    // Write the final result to output
    if (tid == 0) {
        output[blockIdx.x] = max_values[0];
        index_out[blockIdx.x] = max_indices[0];
    }

}

SDP_CUDA_KERNEL(find_maximum_value<double>);
SDP_CUDA_KERNEL(find_maximum_value<float>);
SDP_CUDA_KERNEL(find_maximum_value<__nv_bfloat16>);


template<typename T>
__global__ void add_clean_comp(
            T* clean_comp,
            int* max_idx_flat,
            double loop_gain,
            T* highest_value,
            double threshold
){
    // check threshold
    if (highest_value[0] > threshold){
        
    // Add fraction of maximum to clean components list
    clean_comp[max_idx_flat[0]] += (loop_gain * highest_value[0]);
    }

}

SDP_CUDA_KERNEL(add_clean_comp<double>);
SDP_CUDA_KERNEL(add_clean_comp<float>);
// SDP_CUDA_KERNEL(add_clean_comp<__nv_bfloat16>);


template<typename T>
__global__ void subtract_psf(
            int64_t dirty_img_dim, 
            int64_t psf_dim, 
            double loop_gain, 
            int* max_idx_flat, 
            T* highest_value, 
            const T* psf, 
            T* residual,
            T* clean_comp,
            T* skymodel,
            double threshold) {

    // check threshold
    if (highest_value[0] > threshold){

        int64_t dirty_img_size = dirty_img_dim * dirty_img_dim;
        // int64_t psf_size = psf_dim * psf_dim;

        // get x and y from flat index
        int max_idx_x = max_idx_flat[0] / dirty_img_dim;
        int max_idx_y = max_idx_flat[0] % dirty_img_dim;

        // Identify start position of PSF window to subtract from residual
        int64_t psf_x_start = dirty_img_dim - max_idx_x;
        int64_t psf_y_start = dirty_img_dim - max_idx_y;

        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < dirty_img_size){

            // Compute the x and y coordinates in the dirty image
            int64_t x_dirty = i / dirty_img_dim;
            int64_t y_dirty = i % dirty_img_dim;

            // Compute the x and y coordinates in the psf
            int64_t x_psf = x_dirty + psf_x_start;
            int64_t y_psf = y_dirty + psf_y_start;

            // // get flat index for dirty image
            // int64_t dirty_img_flat_idx = x_dirty * dirty_img_dim + y_dirty;

            // get flat index for psf
            int64_t psf_flat_idx = x_psf * psf_dim + y_psf;

            // Subtract the PSF contribution from the residual
            residual[i] -= (loop_gain * highest_value[0] * psf[psf_flat_idx]);
        }

    }
}

SDP_CUDA_KERNEL(subtract_psf<double>);
SDP_CUDA_KERNEL(subtract_psf<float>);
// SDP_CUDA_KERNEL(subtract_psf<__nv_bfloat16>);


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


template<typename T>
__global__ void convert_from_bfloat(
    const __nv_bfloat16* in,
    int64_t size,
    T* out
){

}

template<>
__global__ void convert_from_bfloat<float>(
    const __nv_bfloat16* in,
    int64_t size,
    float* out
){
    
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size){
        out[i] = __bfloat162float(in[i]);
         
    }
}

template<>
__global__ void convert_from_bfloat<double>(
    const __nv_bfloat16* in,
    int64_t size,
    double* out
){
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size){
        out[i] = in[i];
         
    }
}

SDP_CUDA_KERNEL(convert_from_bfloat<double>);
SDP_CUDA_KERNEL(convert_from_bfloat<float>);


template<typename T>
__global__ void convert_to_bfloat(
    const T* in,
    int64_t size,
    __nv_bfloat16* out
){

}

template<>
__global__ void convert_to_bfloat<double>(
    const double* in,
    int64_t size,
    __nv_bfloat16* out
){
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size){
        out[i] = __double2bfloat16(in[i]);
         
    }

}

template<>
__global__ void convert_to_bfloat<float>(
    const float* in,
    int64_t size,
    __nv_bfloat16* out
){
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size){
        out[i] = __float2bfloat16(in[i]);
        
    }

}

SDP_CUDA_KERNEL(convert_to_bfloat<double>);
SDP_CUDA_KERNEL(convert_to_bfloat<float>);






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


