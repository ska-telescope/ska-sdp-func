/* See the LICENSE file at the top-level directory of this distribution. */

#define NUM_THREADS 256

#include "ska-sdp-func/numeric_functions/sdp_fft_convolution.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_mem.h"

#include <cuComplex.h>


// #define INDEX_2D(N2, N1, I2, I1)    (N1 * I2 + I1)


// create a copy of a complex number using only its real part
template<typename CT, typename T>
__global__ void create_copy_real(
        const CT* in,
        int64_t size,
        T* out
)
{
}


template<>
__global__ void create_copy_real<cuDoubleComplex, double>(
        const cuDoubleComplex* in,
        int64_t size,
        double* out
)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
    {
        out[i] = cuCreal(in[i]);
    }
}


template<>
__global__ void create_copy_real<cuFloatComplex, float>(
        const cuFloatComplex* in,
        int64_t size,
        float* out
)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
    {
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
)
{
}


template<>
__global__ void create_cbeam<double, cuDoubleComplex>(
        const double sigma_X,
        const double sigma_Y,
        const double rotation,
        int16_t cbeam_dim,
        cuDoubleComplex* cbeam
)
{
    // Fit a Gaussian to the main lobe of the PSF based on the parameters passed

    double A = 1;
    double x0 = 0;
    double y0 = 0;

    // Check if the number of rows and columns is odd
    if (cbeam_dim % 2 == 1)
    {
        x0 = cbeam_dim / 2;
        y0 = cbeam_dim / 2;
    }
    else
    {
        x0 = cbeam_dim / 2 - 1;
        y0 = cbeam_dim / 2 - 1;
    }

    double theta = (M_PI / 180) * rotation;

    double a =
            pow(cos(theta),
            2
            ) /
            (2 * pow(sigma_X, 2)) + pow(sin(theta), 2) / (2 * pow(sigma_Y, 2));
    double b = sin(2 * theta) /
            (4 * pow(sigma_X, 2)) - sin(2 * theta) / (4 * pow(sigma_Y, 2));
    double c =
            pow(sin(theta),
            2
            ) /
            (2 * pow(sigma_X, 2)) + pow(cos(theta), 2) / (2 * pow(sigma_Y, 2));

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int size = cbeam_dim * cbeam_dim;

    if (i < size)
    {
        int x = i / cbeam_dim;
        int y = i % cbeam_dim;

        double component = A *
                exp(-(a *
                pow(x - x0,
                2
                ) + 2 * b * (x - x0) * (y - y0) + c * pow(y - y0, 2))
                );
        cbeam[i] = make_cuDoubleComplex(component, 0.0);
    }
}


template<>
__global__ void create_cbeam<float, cuFloatComplex>(
        const float sigma_X,
        const float sigma_Y,
        const float rotation,
        int16_t cbeam_dim,
        cuFloatComplex* cbeam
)
{
    // Fit a Gaussian to the main lobe of the PSF based on the parameters passed

    float A = 1;
    float x0 = 0;
    float y0 = 0;

    // Check if the number of rows and columns is odd
    if (cbeam_dim % 2 == 1)
    {
        x0 = cbeam_dim / 2;
        y0 = cbeam_dim / 2;
    }
    else
    {
        x0 = cbeam_dim / 2 - 1;
        y0 = cbeam_dim / 2 - 1;
    }

    float theta = (M_PI / 180) * rotation;

    float a =
            pow(cos(theta),
            2
            ) /
            (2 * pow(sigma_X, 2)) + pow(sin(theta), 2) / (2 * pow(sigma_Y, 2));
    float b = sin(2 * theta) /
            (4 * pow(sigma_X, 2)) - sin(2 * theta) / (4 * pow(sigma_Y, 2));
    float c =
            pow(sin(theta),
            2
            ) /
            (2 * pow(sigma_X, 2)) + pow(cos(theta), 2) / (2 * pow(sigma_Y, 2));

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int size = cbeam_dim * cbeam_dim;

    if (i < size)
    {
        int x = i / cbeam_dim;
        int y = i % cbeam_dim;

        float component = A *
                exp(-(a *
                pow(x - x0,
                2
                ) + 2 * b * (x - x0) * (y - y0) + c * pow(y - y0, 2))
                );
        cbeam[i] = make_cuFloatComplex(component, 0.0);
    }
}

SDP_CUDA_KERNEL(create_cbeam<double, cuDoubleComplex>);
SDP_CUDA_KERNEL(create_cbeam<float, cuFloatComplex>);


template<typename T, typename I>
__device__ void warpReduce(
        volatile T* max_values,
        volatile I* max_indices,
        int tid
)
{
}


template<>
__device__ void warpReduce<double, int>(
        volatile double* max_values,
        volatile int* max_indices,
        int tid
)
{
    double shfl_val;
    int shfl_idx;

    // unsigned mask = __ballot_sync(0xffffffff, tid < 32); // warpSize is 32

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        shfl_val = __shfl_down_sync(0xFFFFFFFF, max_values[tid], offset);
        shfl_idx = __shfl_down_sync(0xFFFFFFFF, max_indices[tid], offset);
        __syncwarp();

        if (shfl_val > max_values[tid])
        {
            max_values[tid] = shfl_val;
            max_indices[tid] = shfl_idx;
        }
    }
}


template<>
__device__ void warpReduce<float, int>(
        volatile float* max_values,
        volatile int* max_indices,
        int tid
)
{
    float shfl_val;
    int shfl_idx;

    // unsigned mask = __ballot_sync(0xffffffff, tid < 32); // warpSize is 32

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        shfl_val = __shfl_down_sync(0xFFFFFFFF, max_values[tid], offset);
        shfl_idx = __shfl_down_sync(0xFFFFFFFF, max_indices[tid], offset);
        __syncwarp();

        if (shfl_val > max_values[tid])
        {
            max_values[tid] = shfl_val;
            max_indices[tid] = shfl_idx;
        }
    }
}


// find the maximum value in a list using reduction
template<typename T, typename I>
__global__ void find_maximum_value(
        const T* input,
        I* index_in,
        T* output,
        I* index_out,
        const int num_elements,
        bool init_idx
)
{
}


template<>
__global__ void find_maximum_value<double, int>(
        const double* input,
        int* index_in,
        double* output,
        int* index_out,
        const int num_elements,
        bool init_idx
)
{
    __shared__ double max_values[256];
    __shared__ int max_indices[256];

    int64_t tid = threadIdx.x;
    int64_t i = blockIdx.x * (blockDim.x) + threadIdx.x;

    // initialise max_values to lowest possible value
    max_values[tid] = -__DBL_MAX__;
    max_indices[tid] = -1;

    // Load input elements into shared memory
    max_values[tid] = input[i];
    // if index array has already been initialised then load it
    if (init_idx == true)
    {
        max_indices[tid] = index_in[i];
    }
    // if it hasn't, initialise it.
    else
    {
        max_indices[tid] = i;
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            if (max_values[tid] < max_values[tid + stride])
            {
                max_values[tid] = max_values[tid + stride];
                max_indices[tid] = max_indices[tid + stride];
            }
        }
        __syncthreads();
    }

    // Write the final result to output
    if (tid == 0)
    {
        output[blockIdx.x] = max_values[0];
        index_out[blockIdx.x] = max_indices[0];
    }
}


template<>
__global__ void find_maximum_value<float, int>(
        const float* input,
        int* index_in,
        float* output,
        int* index_out,
        const int num_elements,
        bool init_idx
)
{
    __shared__ double max_values[256];
    __shared__ int max_indices[256];

    int64_t tid = threadIdx.x;
    int64_t i = blockIdx.x * (blockDim.x) + threadIdx.x;

    // initialise max_values to lowest possible value
    max_values[tid] = -__FLT_MAX__;
    max_indices[tid] = -1;

    // Load input elements into shared memory
    max_values[tid] = input[i];
    // if index array has already been initialised then load it
    if (init_idx == true)
    {
        max_indices[tid] = index_in[i];
    }
    // if it hasn't, initialise it.
    else
    {
        max_indices[tid] = i;
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            if (max_values[tid] < max_values[tid + stride])
            {
                max_values[tid] = max_values[tid + stride];
                max_indices[tid] = max_indices[tid + stride];
            }
        }
        __syncthreads();
    }

    // Write the final result to output
    if (tid == 0)
    {
        output[blockIdx.x] = max_values[0];
        index_out[blockIdx.x] = max_indices[0];
    }
}

SDP_CUDA_KERNEL(find_maximum_value<double, int>);
SDP_CUDA_KERNEL(find_maximum_value<float, int>);


// add a component to the CLEAN component list
template<typename T, typename T2>
__global__ void add_clean_comp(
        T2* clean_comp,
        int* max_idx_flat,
        T* loop_gain,
        T* highest_value,
        T* threshold,
        int* thresh_reached
)
{
}


template<>
__global__ void add_clean_comp<double, double>(
        double* clean_comp,
        int* max_idx_flat,
        double* loop_gain,
        double* highest_value,
        double* threshold,
        int* thresh_reached
)
{
    // check threshold
    if (highest_value[0] > threshold[0] && *thresh_reached == 0)
    {
        // Add fraction of maximum to clean components list
        double inter = __dmul_rn(loop_gain[0], highest_value[0]);
        clean_comp[max_idx_flat[0]] = __dadd_rn(clean_comp[max_idx_flat[0]],
                inter
        );
        // clean_comp[max_idx_flat[0]] = clean_comp[max_idx_flat[0]] + (loop_gain[0] * highest_value[0]);
    }
    // if threshold reached, set flag
    else
    {
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
)
{
    // check threshold
    if (*highest_value > *threshold && *thresh_reached == 0)
    {
        // Add fraction of maximum to clean components list
        float inter = __fmul_rn(*loop_gain, *highest_value);
        clean_comp[*max_idx_flat] = __fadd_rn(clean_comp[*max_idx_flat], inter);
        // clean_comp[max_idx_flat[0]] = clean_comp[max_idx_flat[0]] + (loop_gain[0] * highest_value[0]);
    }
    // if threshold reached, set flag
    else
    {
        *thresh_reached = 1;
    }
}

SDP_CUDA_KERNEL(add_clean_comp<double, double>);
SDP_CUDA_KERNEL(add_clean_comp<float, float>);


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
)
{
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
        double* threshold
)
{
    // check threshold
    if (highest_value[0] > threshold[0])
    {
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
        for (int i = 0; i < elements_per_thread; i++)
        {
            int64_t curr_idx = super_idx + i * blockDim.x + tid;

            // check thread is in bounds
            if (curr_idx < dirty_img_size)
            {
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
                residual[curr_idx] =  __dsub_rn(residual[curr_idx], inter);

                // residual[i] = residual[i] - (loop_gain[0] * highest_value[0] * psf[psf_flat_idx]);
            }
        }
    }
    else
    {
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
        float* threshold
)
{
    // check threshold
    if (highest_value[0] > threshold[0])
    {
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
        for (int i = 0; i < elements_per_thread; i++)
        {
            int64_t curr_idx = super_idx + i * blockDim.x + tid;

            // check thread is in bounds
            if (curr_idx < dirty_img_size)
            {
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
                residual[curr_idx] =  __fsub_rn(residual[curr_idx], inter);

                // residual[i] = residual[i] - (loop_gain[0] * highest_value[0] * psf[psf_flat_idx]);
            }
        }
    }
    else
    {
        return;
    }
}

SDP_CUDA_KERNEL(subtract_psf<double, double>);
SDP_CUDA_KERNEL(subtract_psf<float, float>);


// add the final residual image to the skymodel
template<typename T>
__global__ void add_residual(
        T* in,
        int64_t size,
        T* out
)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (size))
    {
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
)
{
}


template<>
__global__ void create_copy_complex<double, cuDoubleComplex>(
        const double* in,
        int64_t size,
        cuDoubleComplex* out
)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
    {
        out[i] = make_cuDoubleComplex(in[i], 0.0);
    }
}


template<>
__global__ void create_copy_complex<float, cuFloatComplex>(
        const float* in,
        int64_t size,
        cuFloatComplex* out
)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
    {
        out[i] = make_cuFloatComplex(in[i], 0);
    }
}

SDP_CUDA_KERNEL(create_copy_complex<double, cuDoubleComplex>);
SDP_CUDA_KERNEL(create_copy_complex<float, cuFloatComplex>);


// copy one value to gpu, can convert to bfloat from float or double precision
// needed to convert loop gain and threshold to bfloat16 for correct use in maths with bfloat16 data
template<typename T, typename OT>
__global__ void copy_var_gpu(
        T in,
        OT* out
)
{
}


template<>
__global__ void copy_var_gpu<double, double>(
        double in,
        double* out
)
{
    out[0] = in;
}


template<>
__global__ void copy_var_gpu<float, float>(
        float in,
        float* out
)
{
    out[0] = in;
}

SDP_CUDA_KERNEL(copy_var_gpu<double, double>);
SDP_CUDA_KERNEL(copy_var_gpu<float, float>);
