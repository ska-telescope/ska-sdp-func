/* See the LICENSE file at the top-level directory of this distribution.

CLEAN algorithm taken from:
A.R. Thompson, J.M. Moran, and G.W. Swenson Jr., Interferometry and Synthesis
in Radio Astronomy, Astronomy and Astrophysics Library, 2017, page 552
DOI 10.1007/978-3-319-44431-4_11

and

Deconvolution Tutorial, December 1996, T. Cornwell and A.H. Bridle, page 6
https://www.researchgate.net/publication/2336887_Deconvolution_Tutorial

*/
#define PIXELS_PER_THREAD 1
#define NUM_THREADS 256

#include "ska-sdp-func/clean/sdp_hogbom_clean.h"
#include "ska-sdp-func/numeric_functions/sdp_fft_convolution.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"

#include <cmath>
#include <complex>
#include <stdlib.h>
#include <time.h>

using std::complex;

#define INDEX_2D(N2, N1, I2, I1)    (N1 * I2 + I1)


template<typename T>
inline void sdp_create_cbeam(
        const T* cbeam_details,
        int16_t cbeam_dim,
        complex<T>* cbeam
)
{
    // fit a guassian to the main lobe of the psf

    T A = 1;
    int16_t x0 = 0;
    int16_t y0 = 0;

    x0 = cbeam_dim / 2;
    y0 = cbeam_dim / 2;

    T sigma_X = cbeam_details[0];
    T sigma_Y = cbeam_details[1];
    T theta = (M_PI / 180) * cbeam_details[2];

    T a =
            pow(cos(theta),
            2
            ) /
            (2 * pow(sigma_X, 2)) + pow(sin(theta), 2) / (2 * pow(sigma_Y, 2));
    T b = sin(2 * theta) /
            (4 * pow(sigma_X, 2)) - sin(2 * theta) / (4 * pow(sigma_Y, 2));
    T c =
            pow(sin(theta),
            2
            ) /
            (2 * pow(sigma_X, 2)) + pow(cos(theta), 2) / (2 * pow(sigma_Y, 2));

    for (int x = 0; x < cbeam_dim; x++)
    {
        for (int y = 0; y < cbeam_dim; y++)
        {
            const unsigned int i_cbeam = INDEX_2D(cbeam_dim, cbeam_dim, x, y);
            T component = A *
                    exp(-(a *
                    pow(x - x0,
                    2
                    ) + 2 * b * (x - x0) * (y - y0) + c * pow(y - y0, 2))
                    );
            cbeam[i_cbeam] = complex<T>(component, 0.0);
        }
    }
}


template<typename T>
inline void sdp_create_copy_complex(
        const T* in,
        int64_t size,
        complex<T>* out
)
{
    // creates a copy of an array to a complex array
    for (int i = 0; i < size; i++)
    {
        out[i] = complex<T>(in[i], 0.0);
    }
}


template<typename T>
inline void sdp_create_copy_real(
        const complex<T>* in,
        int64_t size,
        T* out
)
{
    // creates a copy of a complex double array to a double array
    for (int i = 0; i < size; i++)
    {
        out[i] = std::real(in[i]);
    }
}


template<typename T>
static void hogbom_clean(
        const T* psf,
        const T* cbeam_details,
        const T loop_gain,
        const T threshold,
        const int cycle_limit,
        const int64_t dirty_img_dim,
        const int64_t psf_dim,
        const sdp_MemType data_type,
        T* clean_model,
        T* residual,
        T* skymodel,
        sdp_Error* status
)
{
    // calculate useful shapes and sizes
    int64_t dirty_img_size = dirty_img_dim * dirty_img_dim;
    // int64_t psf_size = psf_dim * psf_dim;

    int64_t dirty_img_shape[] = {dirty_img_dim, dirty_img_dim};
    // int64_t psf_shape[] = {psf_dim, psf_dim};

    int64_t cbeam_dim = (int64_t)cbeam_details[3];
    int64_t cbeam_shape[] = {cbeam_dim, cbeam_dim};

    // choose correct complex data type
    sdp_MemType complex_data_type = SDP_MEM_VOID;

    if (data_type == SDP_MEM_DOUBLE)
    {
        complex_data_type = SDP_MEM_COMPLEX_DOUBLE;
    }
    else if (data_type == SDP_MEM_FLOAT)
    {
        complex_data_type = SDP_MEM_COMPLEX_FLOAT;
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data type");
    }

    // Create intermediate data arrays
    sdp_Mem* cbeam_mem = sdp_mem_create(complex_data_type,
            SDP_MEM_CPU,
            2,
            cbeam_shape,
            status
    );
    complex<T>* cbeam_ptr = (complex<T>*)sdp_mem_data(cbeam_mem);
    sdp_Mem* clean_comp_complex_mem = sdp_mem_create(complex_data_type,
            SDP_MEM_CPU,
            2,
            dirty_img_shape,
            status
    );
    complex<T>* clean_comp_complex_ptr = (complex<T>*)sdp_mem_data(
            clean_comp_complex_mem
    );
    sdp_mem_clear_contents(clean_comp_complex_mem, status);

    // set up some loop variables
    int cur_cycle = 0;
    bool stop = 0;

    // create CLEAN Beam
    sdp_mem_clear_contents(cbeam_mem, status);
    sdp_create_cbeam<T>(cbeam_details, cbeam_details[3], cbeam_ptr);

    // CLEAN loop executes while the stop conditions (threshold and cycle limit) are not met
    while (cur_cycle < cycle_limit)
    {
        // Find index and value of the maximum value in residual
        double highest_value = residual[0];
        int max_idx_flat = 0;

        for (int i = 0; i < dirty_img_size; i++)
        {
            if (residual[i] > highest_value)
            {
                highest_value = residual[i];
                max_idx_flat = i;
            }
        }

        // check maximum value against threshold
        if (residual[max_idx_flat] < threshold)
        {
            break;
        }

        // unravel x and y from flat index
        int max_idx_x = 0;
        int max_idx_y = 0;

        max_idx_x = max_idx_flat / dirty_img_dim;
        max_idx_y = max_idx_flat % dirty_img_dim;

        // add fraction of maximum to clean components list
        clean_comp_complex_ptr[max_idx_flat] += complex<T>(
                loop_gain * highest_value,
                0
        );

        // identify psf window to subtract from residual
        int64_t psf_x_start = 0, psf_x_end = 0, psf_y_start = 0, psf_y_end = 0;

        psf_x_start = dirty_img_dim - max_idx_x;
        psf_x_end = psf_x_start + dirty_img_dim;
        psf_y_start = dirty_img_dim - max_idx_y;
        psf_y_end = psf_y_start + dirty_img_dim;

        for (int x = psf_x_start, i = 0; x < psf_x_end; x++, i++)
        {
            for (int y = psf_y_start, j = 0; y < psf_y_end; y++, j++)
            {
                const unsigned int i_psf = INDEX_2D(psf_dim, psf_dim, x, y);
                const unsigned int i_res = INDEX_2D(dirty_img_dim,
                        dirty_img_dim,
                        i,
                        j
                );
                residual[i_res] -= (loop_gain * highest_value * psf[i_psf]);
            }
        }

        cur_cycle += 1;
    }

    // convolve clean components with clean beam
    sdp_Mem* convolution_result_mem = sdp_mem_create(complex_data_type,
            SDP_MEM_CPU,
            2,
            dirty_img_shape,
            status
    );
    complex<T>* convolution_result_ptr = (complex<T>*)sdp_mem_data(
            convolution_result_mem
    );

    sdp_fft_convolution(clean_comp_complex_mem,
            cbeam_mem,
            convolution_result_mem,
            status
    );

    // complex result to real for the output
    sdp_create_copy_real<T>(convolution_result_ptr, dirty_img_size, skymodel);

    // add remaining residual
    for (int i = 0; i < dirty_img_size; i++)
    {
        skymodel[i] = skymodel[i] + residual[i];
    }

    // convert residual to real for output
    sdp_create_copy_real<T>(clean_comp_complex_ptr, dirty_img_size,
            clean_model
    );

    // free memory
    sdp_mem_ref_dec(cbeam_mem);
    sdp_mem_ref_dec(clean_comp_complex_mem);
    // sdp_mem_ref_dec(residual_complex_mem);
    sdp_mem_ref_dec(convolution_result_mem);
}


template<typename T>
void sdp_hogbom_clean_gpu(
        const sdp_Mem* dirty_img,
        const sdp_Mem* psf,
        const sdp_Mem* cbeam_details,
        const T loop_gain,
        const T threshold,
        const int cycle_limit,
        const int64_t dirty_img_dim,
        const int64_t psf_dim,
        const sdp_MemType data_type,
        sdp_Mem* clean_model,
        sdp_Mem* residual,
        sdp_Mem* skymodel,
        sdp_Error* status
)
{
    // set cbeam shape and size
    const T* cbeam_details_ptr = (const T*)sdp_mem_data_const(cbeam_details);

    int64_t cbeam_dim = (int64_t)cbeam_details_ptr[3];
    int64_t cbeam_shape[] = {cbeam_dim, cbeam_dim};

    // calculate useful shapes and sizes
    int64_t dirty_img_size = dirty_img_dim * dirty_img_dim;
    int64_t psf_size = psf_dim * psf_dim;
    int64_t variable_size = 1;
    int64_t cbeam_size = cbeam_dim * cbeam_dim;

    int64_t dirty_img_shape[] = {dirty_img_dim, dirty_img_dim};
    int64_t psf_shape[] = {psf_dim, psf_dim};
    int64_t variable_shape[] = {variable_size};

    // variable to hold name of kernel to be launched
    const char* kernel_name = 0;

    // select correct complex variable to use
    sdp_MemType complex_data_type = SDP_MEM_VOID;

    if (data_type == SDP_MEM_DOUBLE)
    {
        complex_data_type = SDP_MEM_COMPLEX_DOUBLE;
    }
    else if (data_type == SDP_MEM_FLOAT)
    {
        complex_data_type = SDP_MEM_COMPLEX_FLOAT;
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data type");
    }

    // Create intermediate data arrays
    // to hold loop gain, threshold flag and threshold value in GPU memory
    sdp_Mem* loop_gain_mem = sdp_mem_create(data_type,
            SDP_MEM_GPU,
            1,
            variable_shape,
            status
    );
    sdp_Mem* threshold_mem = sdp_mem_create(data_type,
            SDP_MEM_GPU,
            1,
            variable_shape,
            status
    );
    sdp_Mem* thresh_reached_gpu_mem = sdp_mem_create(SDP_MEM_INT,
            SDP_MEM_GPU,
            1,
            variable_shape,
            status
    );
    sdp_mem_clear_contents(thresh_reached_gpu_mem, status);
    sdp_Mem* thresh_reached_mem = sdp_mem_create(SDP_MEM_INT,
            SDP_MEM_CPU,
            1,
            variable_shape,
            status
    );
    int* thresh_reached_ptr = (int*)sdp_mem_data(thresh_reached_mem);
    sdp_mem_clear_contents(thresh_reached_mem, status);

    // for CLEAN beam
    sdp_Mem* cbeam_complex_mem = sdp_mem_create(complex_data_type,
            SDP_MEM_GPU,
            2,
            cbeam_shape,
            status
    );
    sdp_mem_clear_contents(cbeam_complex_mem, status);

    // for CLEAN components
    sdp_Mem* clean_comp_complex_mem = sdp_mem_create(complex_data_type,
            SDP_MEM_GPU,
            2,
            dirty_img_shape,
            status
    );
    // sdp_mem_clear_contents(clean_model, status);
    sdp_mem_clear_contents(clean_comp_complex_mem, status);

    // for psf working space
    sdp_Mem* working_psf_mem = sdp_mem_create(data_type,
            SDP_MEM_GPU,
            2,
            psf_shape,
            status
    );
    sdp_mem_clear_contents(working_psf_mem, status);

    // copy variables to GPU
    sdp_mem_copy_contents(residual, dirty_img, 0, 0, dirty_img_size, status);
    sdp_mem_copy_contents(working_psf_mem, psf, 0, 0, psf_size, status);

    // set kernel names
    const char* kernel_name_var_copy = 0;

    if (data_type == SDP_MEM_DOUBLE)
    {
        kernel_name_var_copy = "copy_var_gpu<double, double>";
    }
    else if (data_type == SDP_MEM_FLOAT)
    {
        kernel_name_var_copy = "copy_var_gpu<float, float>";
    }
    // catch unsupported data types
    else
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data type");
    }

    uint64_t num_threads_copy_gpu[] = {1, 1, 1};
    uint64_t num_blocks_copy_gpu[] = {1, 1, 1};

    const void* args_loop_gain_to_gpu[] = {
        &loop_gain,
        sdp_mem_gpu_buffer(loop_gain_mem, status),
    };

    sdp_launch_cuda_kernel(kernel_name_var_copy,
            num_blocks_copy_gpu,
            num_threads_copy_gpu,
            0,
            0,
            args_loop_gain_to_gpu,
            status
    );

    sdp_mem_check_location(loop_gain_mem, SDP_MEM_GPU, status);

    const void* args_threshold_to_gpu[] = {
        &threshold,
        sdp_mem_gpu_buffer(threshold_mem, status),
    };

    sdp_launch_cuda_kernel(kernel_name_var_copy,
            num_blocks_copy_gpu,
            num_threads_copy_gpu,
            0,
            0,
            args_threshold_to_gpu,
            status
    );

    sdp_mem_check_location(threshold_mem, SDP_MEM_GPU, status);

    // assign basic thread and block sizes
    uint64_t num_threads[] = {256, 1, 1};
    uint64_t num_blocks[] = {
        ((dirty_img_size + num_threads[0] - 1) / num_threads[0]), 1, 1
    };

    // number of elements each thread processes in the find_maximum_value
    int elements_per_thread = PIXELS_PER_THREAD;

    // get size for number of reductions to perform
    int64_t reduction_size = (int64_t)dirty_img_size;

    // set number of threads and block for the reduction
    uint64_t num_threads_reduce[] = {256, 1, 1};
    uint64_t num_blocks_reduce[] = {
        ((reduction_size + num_threads_reduce[0] - 1) / num_threads_reduce[0]),
        1, 1
    };

    // size and shape for array of maximums used in the reduction
    int64_t max_size = (int64_t)num_blocks_reduce[0];
    int64_t max_shape[] = {max_size};

    // to store image maximum value and index
    sdp_Mem* max_val_mem = sdp_mem_create(data_type,
            SDP_MEM_GPU,
            1,
            max_shape,
            status
    );
    sdp_Mem* max_idx_mem = sdp_mem_create(SDP_MEM_INT,
            SDP_MEM_GPU,
            1,
            max_shape,
            status
    );

    // set up some loop variables
    int cur_cycle = 0;
    bool stop = 0;

    // CLEAN loop executes while the stop conditions (threshold and cycle limit) are not met
    while (cur_cycle < cycle_limit)
    {
        // reset maximum values for new loop
        sdp_mem_clear_contents(max_val_mem, status);
        sdp_mem_clear_contents(max_idx_mem, status);

        bool init_idx = false;

        const void* args[] = {
            sdp_mem_gpu_buffer_const(residual, status),
            sdp_mem_gpu_buffer(max_idx_mem, status),
            sdp_mem_gpu_buffer(max_val_mem, status),
            sdp_mem_gpu_buffer(max_idx_mem, status),
            &dirty_img_size,
            &init_idx
        };

        // find the maximum value in the residual
        if (data_type == SDP_MEM_DOUBLE)
        {
            kernel_name = "find_maximum_value<double, int>";
        }
        else if (data_type == SDP_MEM_FLOAT)
        {
            kernel_name = "find_maximum_value<float, int>";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type");
        }

        // launch multiple kernels to perform reduction
        // scale the number of blocks according to the size of the reduction
        while (num_blocks_reduce[0] > 1)
        {
            sdp_launch_cuda_kernel(kernel_name,
                    num_blocks_reduce, num_threads_reduce, 0, 0, args, status
            );

            args[4] = &max_size;

            num_blocks_reduce[0] =
                    ((num_blocks_reduce[0] + num_threads_reduce[0] - 1) /
                    num_threads_reduce[0]);

            args[0] = sdp_mem_gpu_buffer_const(max_val_mem, status);

            init_idx = true;
        }

        // perform final reduction with 1 block for final answer
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks_reduce, num_threads_reduce, 0, 0, args, status
        );

        // check if threshold has been reached once every 100 iterations
        if (cur_cycle > 0 && cur_cycle % 100 == 0)
        {
            sdp_mem_copy_contents(thresh_reached_mem,
                    thresh_reached_gpu_mem,
                    0,
                    0,
                    variable_size,
                    status
            );

            SDP_LOG_INFO("thresh reached: %d", *thresh_reached_ptr);
            SDP_LOG_INFO("iteration number: %d", cur_cycle);

            if (*thresh_reached_ptr > 0)
            {
                break;
            }
        }

        // reset number of blocks after reducing them in the reduction loop
        num_blocks_reduce[0] =
                ((reduction_size + num_threads_reduce[0] - 1) /
                num_threads_reduce[0]);

        // add clean components here
        uint64_t num_threads_clean_comp[] = {1, 1, 1};
        uint64_t num_blocks_clean_comp[] = {1, 1, 1};

        if (data_type == SDP_MEM_DOUBLE)
        {
            kernel_name = "add_clean_comp<double, double>";
        }
        else if (data_type == SDP_MEM_FLOAT)
        {
            kernel_name = "add_clean_comp<float, float>";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type");
        }

        const void* args_clean_comp[] = {
            sdp_mem_gpu_buffer(clean_model, status),
            sdp_mem_gpu_buffer(max_idx_mem, status),
            sdp_mem_gpu_buffer(loop_gain_mem, status),
            sdp_mem_gpu_buffer(max_val_mem, status),
            sdp_mem_gpu_buffer(threshold_mem, status),
            sdp_mem_gpu_buffer(thresh_reached_gpu_mem, status)
        };

        sdp_launch_cuda_kernel(kernel_name,
                num_blocks_clean_comp,
                num_threads_clean_comp,
                0,
                0,
                args_clean_comp,
                status
        );

        // subtract psf
        if (data_type == SDP_MEM_DOUBLE)
        {
            kernel_name = "subtract_psf<double, double>";
        }
        else if (data_type == SDP_MEM_FLOAT)
        {
            kernel_name = "subtract_psf<float, float>";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type");
        }

        const void* args_subtract_psf[] = {
            &dirty_img_dim,
            &psf_dim,
            sdp_mem_gpu_buffer(loop_gain_mem, status),
            sdp_mem_gpu_buffer(max_idx_mem, status),
            sdp_mem_gpu_buffer(max_val_mem, status),
            &elements_per_thread,
            sdp_mem_gpu_buffer(working_psf_mem, status),
            sdp_mem_gpu_buffer(residual, status),
            sdp_mem_gpu_buffer(threshold_mem, status),
        };

        // set number of threads and block for the psf subtraction
        uint64_t num_threads_subtract_psf[] = {256, 1, 1};
        uint64_t num_blocks_subtract_psf[] = {
            ((dirty_img_size + num_threads_subtract_psf[0] - 1) /
            num_threads_subtract_psf[0]), 1, 1
        };

        sdp_launch_cuda_kernel(kernel_name,
                num_blocks_subtract_psf,
                num_threads_subtract_psf,
                0,
                0,
                args_subtract_psf,
                status
        );

        cur_cycle += 1;
    }

    // release memory for psf slicing in clean loop
    sdp_mem_ref_dec(working_psf_mem);

    // convert to complex representaion for convolution with cbeam
    if (data_type == SDP_MEM_DOUBLE)
    {
        kernel_name = "create_copy_complex<double, cuDoubleComplex>";
    }
    else if (data_type == SDP_MEM_FLOAT)
    {
        kernel_name = "create_copy_complex<float, cuFloatComplex>";
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data type");
    }

    const void* args_create_complex[] = {
        sdp_mem_gpu_buffer_const(clean_model, status),
        &dirty_img_size,
        sdp_mem_gpu_buffer(clean_comp_complex_mem, status)
    };

    sdp_launch_cuda_kernel(kernel_name,
            num_blocks, num_threads, 0, 0, args_create_complex, status
    );

    // create cbeam
    // SDP_LOG_DEBUG("cbeam dim %d", cbeam_dim);

    uint64_t num_threads_create_cbeam[] = {256, 1, 1};
    uint64_t num_blocks_create_cbeam[] = {
        ((cbeam_size + num_threads_create_cbeam[0] - 1) /
        num_threads_create_cbeam[0]), 1, 1
    };

    if (data_type == SDP_MEM_DOUBLE)
    {
        kernel_name = "create_cbeam<double, cuDoubleComplex>";
    }
    else if (data_type == SDP_MEM_FLOAT)
    {
        kernel_name = "create_cbeam<float, cuFloatComplex>";
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data type");
    }

    const void* args4[] = {
        &cbeam_details_ptr[0],
        &cbeam_details_ptr[1],
        &cbeam_details_ptr[2],
        &cbeam_dim,
        sdp_mem_gpu_buffer(cbeam_complex_mem, status)
    };

    sdp_launch_cuda_kernel(kernel_name,
            num_blocks_create_cbeam,
            num_threads_create_cbeam,
            0,
            0,
            args4,
            status
    );

    // for result of convolution of CLEAN beam and with CLEAN components
    sdp_Mem* convolution_result_mem = sdp_mem_create(complex_data_type,
            SDP_MEM_GPU,
            2,
            dirty_img_shape,
            status
    );
    sdp_mem_clear_contents(convolution_result_mem, status);

    // convolve clean components with clean beam
    sdp_fft_convolution(clean_comp_complex_mem,
            cbeam_complex_mem,
            convolution_result_mem,
            status
    );

    // convert back to real number only representation
    if (data_type == SDP_MEM_DOUBLE)
    {
        kernel_name = "create_copy_real<cuDoubleComplex, double>";
    }
    else if (data_type == SDP_MEM_FLOAT)
    {
        kernel_name = "create_copy_real<cuFloatComplex, float>";
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data type");
    }

    const void* args_create_real[] = {
        sdp_mem_gpu_buffer_const(convolution_result_mem, status),
        &dirty_img_size,
        sdp_mem_gpu_buffer(skymodel, status)
    };

    sdp_launch_cuda_kernel(kernel_name,
            num_blocks, num_threads, 0, 0, args_create_real, status
    );

    // add remaining residual
    if (data_type == SDP_MEM_DOUBLE)
    {
        kernel_name = "add_residual<double>";
    }
    else if (data_type == SDP_MEM_FLOAT)
    {
        kernel_name = "add_residual<float>";
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data type");
    }

    const void* args_add_residual[] = {
        sdp_mem_gpu_buffer(residual, status),
        &dirty_img_size,
        sdp_mem_gpu_buffer(skymodel, status)
    };

    sdp_launch_cuda_kernel(kernel_name,
            num_blocks, num_threads, 0, 0, args_add_residual, status
    );

    // clear memory
    sdp_mem_ref_dec(loop_gain_mem);
    sdp_mem_ref_dec(threshold_mem);
    sdp_mem_ref_dec(thresh_reached_gpu_mem);
    sdp_mem_ref_dec(thresh_reached_mem);
    sdp_mem_ref_dec(cbeam_complex_mem);
    sdp_mem_ref_dec(clean_comp_complex_mem);
    sdp_mem_ref_dec(convolution_result_mem);
    sdp_mem_ref_dec(max_val_mem);
    sdp_mem_ref_dec(max_idx_mem);
}


void sdp_hogbom_clean(
        const sdp_Mem* dirty_img,
        const sdp_Mem* psf,
        const sdp_Mem* cbeam_details,
        const double loop_gain,
        const double threshold,
        const int cycle_limit,
        sdp_Mem* clean_model,
        sdp_Mem* residual,
        sdp_Mem* skymodel,
        sdp_Error* status
)
{
    if (*status) {return;}

    const int64_t dirty_img_dim = sdp_mem_shape_dim(dirty_img, 0);
    const int64_t psf_dim = sdp_mem_shape_dim(psf, 0);
    const int64_t clean_model_dim = sdp_mem_shape_dim(clean_model, 0);
    const int64_t residual_dim = sdp_mem_shape_dim(residual, 0);
    const int64_t skymodel_dim = sdp_mem_shape_dim(skymodel, 0);
    const int64_t cbeam_details_dim = sdp_mem_shape_dim(cbeam_details, 0);

    const sdp_MemLocation location = sdp_mem_location(dirty_img);
    const sdp_MemType data_type = sdp_mem_type(dirty_img);

    if (sdp_mem_is_read_only(skymodel))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Output is not writable");
        return;
    }

    if (sdp_mem_location(dirty_img) != sdp_mem_location(skymodel))
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Memory location mismatch");
        return;
    }
    if (sdp_mem_type(dirty_img) != sdp_mem_type(psf))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("The Dirty image and PSF must be of the same data type");
        return;
    }

    if (sdp_mem_type(dirty_img) != sdp_mem_type(skymodel))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("The input and output must be of the same data type");
        return;
    }

    if (clean_model_dim != dirty_img_dim)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR(
                "The CLEAN model and the dirty image must be the same size"
        );
        return;
    }

    if (residual_dim != dirty_img_dim)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR(
                "The residual image and the dirty image must be the same size"
        );
        return;
    }

    if (skymodel_dim != dirty_img_dim)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR(
                "The skymodel image and the dirty image must be the same size"
        );
        return;
    }

    if (cbeam_details_dim != 4)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR(
                "The array describing the CLEAN beam must include BMAJ, BMIN, THETA and SIZE"
        );
        return;
    }

    if (location == SDP_MEM_CPU)
    {
        // copy dirty image to starting residual
        sdp_mem_copy_contents(residual, dirty_img, 0, 0,
                (dirty_img_dim * dirty_img_dim), status
        );

        if (data_type == SDP_MEM_DOUBLE)
        {
            hogbom_clean<double>(
                    (const double*)sdp_mem_data_const(psf),
                    (const double*)sdp_mem_data_const(cbeam_details),
                    loop_gain,
                    threshold,
                    cycle_limit,
                    dirty_img_dim,
                    psf_dim,
                    data_type,
                    (double*)sdp_mem_data(clean_model),
                    (double*)sdp_mem_data(residual),
                    (double*)sdp_mem_data(skymodel),
                    status
            );
        }
        else if (data_type == SDP_MEM_FLOAT)
        {
            hogbom_clean<float>(
                    (const float*)sdp_mem_data_const(psf),
                    (const float*)sdp_mem_data_const(cbeam_details),
                    (float)loop_gain,
                    (float)threshold,
                    cycle_limit,
                    dirty_img_dim,
                    psf_dim,
                    data_type,
                    (float*)sdp_mem_data(clean_model),
                    (float*)sdp_mem_data(residual),
                    (float*)sdp_mem_data(skymodel),
                    status
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type");
        }
    }
    else if (location == SDP_MEM_GPU)
    {
        if (data_type == SDP_MEM_DOUBLE)
        {
            sdp_hogbom_clean_gpu<double>(
                    dirty_img,
                    psf,
                    cbeam_details,
                    loop_gain,
                    threshold,
                    cycle_limit,
                    dirty_img_dim,
                    psf_dim,
                    data_type,
                    clean_model,
                    residual,
                    skymodel,
                    status
            );
        }
        else if (data_type == SDP_MEM_FLOAT)
        {
            sdp_hogbom_clean_gpu<float>(
                    dirty_img,
                    psf,
                    cbeam_details,
                    (float)loop_gain,
                    (float)threshold,
                    cycle_limit,
                    dirty_img_dim,
                    psf_dim,
                    data_type,
                    clean_model,
                    residual,
                    skymodel,
                    status
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type");
        }
    }
}
