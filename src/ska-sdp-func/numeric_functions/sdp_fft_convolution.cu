/* See the LICENSE file at the top-level directory of this distribution. */

#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/fourier_transforms/sdp_fft.h"
#include"ska-sdp-func/utility/sdp_mem.h"

#include <cuComplex.h>

template<typename T>
__global__ void pad_2D_gpu(
        const T *data,
        T *padded_data,
        int64_t rows,
        int64_t cols,
        int64_t pad_rows,
        int64_t pad_cols,
        int64_t padded_cols) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols) {
        padded_data[(i+pad_rows)*padded_cols + (j+pad_cols)] = data[i*cols + j];
    }
}

SDP_CUDA_KERNEL(pad_2D_gpu<cuDoubleComplex>);
SDP_CUDA_KERNEL(pad_2D_gpu<cuFloatComplex>);

template<typename T>
__global__ void complex_multiply(
        const T* in1,
        const T* in2,
        T* out,
        int64_t size) {

}

template<>
__global__ void complex_multiply<cuDoubleComplex>(
        const cuDoubleComplex* in1,
        const cuDoubleComplex* in2,
        cuDoubleComplex* out,
        int64_t size) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        out[i] = cuCmul(in1[i], in2[i]);
    }
}

template<>
__global__ void complex_multiply<cuFloatComplex>(
        const cuFloatComplex* in1,
        const cuFloatComplex* in2,
        cuFloatComplex* out,
        int64_t size) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        out[i] = cuCmulf(in1[i], in2[i]);
    }
}

SDP_CUDA_KERNEL(complex_multiply<cuDoubleComplex>);
SDP_CUDA_KERNEL(complex_multiply<cuFloatComplex>);

template<typename T>
__global__ void fft_normalise_gpu(
        T* fft_in,
        int64_t size){

}

template<>
__global__ void fft_normalise_gpu<cuDoubleComplex>(
        cuDoubleComplex* fft_in,
        int64_t size){

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    cuDoubleComplex normalise = make_cuDoubleComplex(size,0);

    if (i < size) {
        fft_in[i] = cuCdiv(fft_in[i], normalise);
    }
}

template<>
__global__ void fft_normalise_gpu<cuFloatComplex>(
        cuFloatComplex* fft_in,
        int64_t size){

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    cuFloatComplex normalise = make_cuFloatComplex(size,0);

    if (i < size) {
        fft_in[i] = cuCdivf(fft_in[i], normalise);
    }
}

SDP_CUDA_KERNEL(fft_normalise_gpu<cuDoubleComplex>);
SDP_CUDA_KERNEL(fft_normalise_gpu<cuFloatComplex>);

template<typename T>
__global__ void fft_shift_2D_gpu(
        T *data,
        T *shifted_data,
        int64_t rows,
        int64_t cols,
        int64_t half_rows,
        int64_t half_cols) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < half_rows && j < half_cols) {
            // bottom right to top left
            shifted_data[i*cols + j] = data[(i+half_rows)*cols + j + half_cols];
            // top right to bottom left
            shifted_data[(i+half_rows)*cols + j] = data[i*cols + j + half_cols];
            // top left to bottom right
            shifted_data[(i+half_rows)*cols + j + half_cols] = data[i*cols + j];
            // bottom left to top right
            shifted_data[i*cols + j + half_cols] = data[(i+half_rows)*cols + j];
    }
}
SDP_CUDA_KERNEL(fft_shift_2D_gpu<cuDoubleComplex>);
SDP_CUDA_KERNEL(fft_shift_2D_gpu<cuFloatComplex>);

template<typename T>
__global__ void remove_padding_2D_gpu(
        T *padded_data,
        T *data,
        int64_t rows,
        int64_t cols,
        int64_t pad_rows,
        int64_t pad_cols,
        int64_t original_rows,
        int64_t original_cols) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < original_rows && j < original_cols) {
        data[i*original_cols + j] = padded_data[(i+(pad_rows-1))*cols + (j+(pad_cols-1))];
    }
}

SDP_CUDA_KERNEL(remove_padding_2D_gpu<cuDoubleComplex>);
SDP_CUDA_KERNEL(remove_padding_2D_gpu<cuFloatComplex>);