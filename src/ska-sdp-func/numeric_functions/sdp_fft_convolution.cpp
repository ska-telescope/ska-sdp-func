/* See the LICENSE file at the top-level directory of this distribution. 

Convolution using the convolution theorem. Inputs are assumed to be square.
returned convolution is the same size as in1, similar to scipy.signal.convolve "same" mode
*/

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <complex>

#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/fourier_transforms/sdp_fft.h"
#include "ska-sdp-func/utility/sdp_mem.h"
#include "ska-sdp-func/numeric_functions/sdp_fft_convolution.h"

using std::complex;

template<typename T>
inline void fft_shift_2D(
        complex<T> *data,
        int64_t rows,
        int64_t cols) {

    int64_t i, j;
    int64_t half_rows = rows / 2;
    int64_t half_cols = cols / 2;
    complex<T> tmp;

    // shift rows
    for (i = 0; i < half_rows; i++) {
        for (j = 0; j < cols; j++) {
            tmp = data[i*cols + j];
            data[i*cols + j] = data[(i+half_rows)*cols + j];
            data[(i+half_rows)*cols + j] = tmp;
        }
    }

    // shift columns
    for (i = 0; i < rows; i++) {
        for (j = 0; j < half_cols; j++) {
            tmp = data[i*cols + j];
            data[i*cols + j] = data[i*cols + j+half_cols];
            data[i*cols + j+half_cols] = tmp;
        }
    }
}

template<typename T>
inline void pad_2D(
        const complex<T> *data,
        complex<T> *padded_data,
        int64_t rows,
        int64_t cols,
        int64_t pad_rows,
        int64_t pad_cols) {

    int64_t i, j; 
    // int64_t padded_rows = rows + 2*pad_rows;
    int64_t padded_cols = cols + 2*pad_cols;


    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            padded_data[(i+pad_rows)*padded_cols + (j+pad_cols)] = data[i*cols + j];
        }
    }
}

template<typename T>
inline void remove_padding_2D(
        complex<T> *padded_data,
        complex<T> *data,
        int64_t rows,
        int64_t cols,
        int64_t pad_rows,
        int64_t pad_cols) {

    int64_t i, j;
    int64_t original_rows = rows - 2*pad_rows;
    int64_t original_cols = cols - 2*pad_cols;

    for (i = 0; i < original_rows; i++) {
        for (j = 0; j < original_cols; j++) {
            data[i*original_cols + j] = padded_data[(i+(pad_rows-1))*cols + (j+(pad_cols-1))];
        }
    }
}

template<typename T>
inline void fft_normalise(
        complex<T>* fft_in,
        int64_t size){

            // double normalise = (double) 1/size;
            complex<T> normalise = complex<T>(size,0);

            for(int i = 0; i < size; i++){
                fft_in[i] = fft_in[i] / normalise;
            }
        }

template<typename T>
static void fft_convolution(
        const complex<T>* in1,
        const complex<T>* in2,
        const int64_t in1_dim,
        const int64_t in2_dim,
        const sdp_MemType data_type,
        complex<T>* out,
        sdp_Error* status
){

        // pad images
        // calculate minimum length for padding of each dim
        // m + n -1 
        int64_t pad_dim = in1_dim + in2_dim - 1;

        // make sure padded image is a power of 2
        while (ceil(log2(pad_dim)) != floor(log2(pad_dim))){

            pad_dim += 1;
        }

        int64_t pad_shape[] = {pad_dim, pad_dim};
        int64_t pad_size = pad_dim * pad_dim;

        // create memory to save padded input in1
        sdp_Mem* in1_pad_mem = sdp_mem_create(data_type, SDP_MEM_CPU, 2, pad_shape, status);
        complex<T>* in1_pad_ptr = (complex<T>*)sdp_mem_data(in1_pad_mem);
        sdp_mem_clear_contents(in1_pad_mem, status);

        // calculate the number of extra columns and rows need to reach padded lenth
        int64_t extra_in1 = (pad_dim - in1_dim)/2;

        // pad in1
        pad_2D<T>(in1, in1_pad_ptr, in1_dim, in1_dim, extra_in1, extra_in1);

        // create memory to save padded input in2
        sdp_Mem* in2_pad_mem = sdp_mem_create(data_type, SDP_MEM_CPU, 2, pad_shape, status);
        complex<T>* in2_pad_ptr = (complex<T>*)sdp_mem_data(in2_pad_mem);
        sdp_mem_clear_contents(in2_pad_mem, status);

        // calculate the number of extra columns and rows need to reach padded lenth
        int64_t extra_in2 = (pad_dim - in2_dim)/2;

        // pad in2
        pad_2D<T>(in2, in2_pad_ptr, in2_dim, in2_dim, extra_in2, extra_in2);

        // create variables for FFT results
        sdp_Mem* in1_fft_result_mem = sdp_mem_create(data_type, SDP_MEM_CPU, 2, pad_shape, status);
        complex<T>* in1_fft_result_ptr = (complex<T>*)sdp_mem_data(in1_fft_result_mem);
        sdp_mem_clear_contents(in1_fft_result_mem, status);
        sdp_Mem* in2_fft_result_mem = sdp_mem_create(data_type, SDP_MEM_CPU, 2, pad_shape, status);
        complex<T>* in2_fft_result_ptr = (complex<T>*)sdp_mem_data(in2_fft_result_mem);
        sdp_mem_clear_contents(in2_fft_result_mem, status);

        // get FFT of padded in1
        sdp_Fft *in1_fft_plan = sdp_fft_create(in1_pad_mem, in1_fft_result_mem, 2, 1, status);
        sdp_fft_exec(in1_fft_plan, in1_pad_mem, in1_fft_result_mem, status);
        sdp_fft_free(in1_fft_plan);

        // get FFT of padded in2
        sdp_Fft *in2_fft_plan = sdp_fft_create(in2_pad_mem, in2_fft_result_mem, 2, 1, status);
        sdp_fft_exec(in2_fft_plan, in2_pad_mem, in2_fft_result_mem, status);
        sdp_fft_free(in2_fft_plan);

        // create variables for frequency domain multiplication result
        sdp_Mem* multiply_mem = sdp_mem_create(data_type, SDP_MEM_CPU, 2, pad_shape, status);
        complex<T>* multiply_ptr = (complex<T>*)sdp_mem_data(multiply_mem);
        sdp_mem_clear_contents(multiply_mem, status);

        // multiply FFTs together
        for (int i = 0; i < pad_size; i++){
            
            multiply_ptr[i] = in1_fft_result_ptr[i] * in2_fft_result_ptr[i];

        }

        // inverse FFT of result
        sdp_Mem* multiply_ifft_result_mem = sdp_mem_create(data_type, SDP_MEM_CPU, 2, pad_shape, status);
        complex<T>* multiply_ifft_result_ptr = (complex<T>*)sdp_mem_data(multiply_ifft_result_mem);
        
        sdp_Fft *result_ifft_plan = sdp_fft_create(multiply_mem, multiply_ifft_result_mem,2,0,status);
        sdp_fft_exec(result_ifft_plan,multiply_mem,multiply_ifft_result_mem,status);
        sdp_fft_free(result_ifft_plan);
        fft_normalise<T>(multiply_ifft_result_ptr, pad_size);

        // shift the result to the center of the image
        fft_shift_2D<T>(multiply_ifft_result_ptr, pad_dim, pad_dim);

        // remove padding from the convolved result
        remove_padding_2D<T>(multiply_ifft_result_ptr, out, pad_dim, pad_dim, extra_in1, extra_in1);

        sdp_mem_ref_dec(in1_pad_mem);
        sdp_mem_ref_dec(in2_pad_mem);
        sdp_mem_ref_dec(in1_fft_result_mem);
        sdp_mem_ref_dec(in2_fft_result_mem);
        sdp_mem_ref_dec(multiply_mem);
        sdp_mem_ref_dec(multiply_ifft_result_mem);
}

void sdp_fft_convolution(
        const sdp_Mem* in1,
        const sdp_Mem* in2,
        sdp_Mem* out,
        sdp_Error* status
){
    if (*status) return;

    const int64_t in1_dim = sdp_mem_shape_dim(in1, 0);
    const int64_t in2_dim = sdp_mem_shape_dim(in2, 0);

    const sdp_MemType data_type = sdp_mem_type(in1);

    if (sdp_mem_is_read_only(out)){
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Output is not writable");
        return;
    }

    if (sdp_mem_location(in1) != sdp_mem_location(out)){
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Memory location mismatch");
        return;
    }

    if (sdp_mem_type(in1) != sdp_mem_type(out)){
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Both input and output must be of the same data type");
        return;
    }

    if (!sdp_mem_is_complex(in1)){
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Input data must be complex");
        return;
    }

    if (sdp_mem_location(in1) == SDP_MEM_CPU)
    {
        if (sdp_mem_type(in1) == SDP_MEM_COMPLEX_DOUBLE){
            fft_convolution<double>(
                (const complex<double>*)sdp_mem_data_const(in1),
                (const complex<double>*)sdp_mem_data_const(in2),
                in1_dim,
                in2_dim,
                data_type,
                (complex<double>*)sdp_mem_data(out),
                status
            );
        }

        if (sdp_mem_type(in1) == SDP_MEM_COMPLEX_FLOAT){
            fft_convolution<float>(
                (const complex<float>*)sdp_mem_data_const(in1),
                (const complex<float>*)sdp_mem_data_const(in2),
                in1_dim,
                in2_dim,
                data_type,
                (complex<float>*)sdp_mem_data(out),
                status
            );
        }
    }
}