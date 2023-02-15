/* See the LICENSE file at the top-level directory of this distribution. 

CLEAN algorithm taken from:
A.R. Thompson, J.M. Moran, and G.W. Swenson Jr., Interferometry and Synthesis
in Radio Astronomy, Astronomy and Astrophysics Library, 2017, page 552
DOI 10.1007/978-3-319-44431-4_11

and

Deconvolution Tutorial, December 1996, T. Cornwell and A.H. Bridle, page 6
https://www.researchgate.net/publication/2336887_Deconvolution_Tutorial

*/

#include "ska-sdp-func/clean/sdp_hogbom_clean.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/fourier_transforms/sdp_fft.h"
#include "ska-sdp-func/utility/sdp_mem.h"

#include <string>
#include <cmath>
#include <complex>
#include <iostream>

using std::complex;

#define INDEX_2D(N2, N1, I2, I1)    (N1 * I2 + I1)

inline void create_cbeam(
        const double* cbeam_details,
        int16_t psf_dim,
        complex<double>* cbeam
){
    // fit a guassian to the main lobe of the psf

    double A = 1;
    double x0 = psf_dim/2;
    double y0 = psf_dim/2;
    double sigma_X = cbeam_details[0];
    double sigma_Y = cbeam_details[1];
    double theta = cbeam_details[2];

    double a = pow(cos(theta),2) / (2 * pow(sigma_X,2)) + pow(sin(theta),2) / (2 * pow(sigma_Y,2));
    double b = sin(2 * theta) / (4 * pow(sigma_X,2)) - sin(2 * theta) / (4 * pow(sigma_Y,2));
    double c = pow(sin(theta),2) / (2 * pow(sigma_X,2)) + pow(cos(theta),2) / (2 * pow(sigma_Y,2));

    for(int x = 0; x < psf_dim; x ++) {
        for(int y = 0; y < psf_dim; y ++) {

            const unsigned int i_cbeam = INDEX_2D(psf_dim,psf_dim,x,y);
            double component = A * exp(-(a * pow(x - x0,2) + 2 * b * (x - x0) * (y - y0) + c * pow(y - y0,2)));
            cbeam[i_cbeam] = complex<double>(component, 0.0);

            }
        }
}

inline void create_copy_complex(
        const double* in,
        int64_t size,
        complex<double>* out
){
        // creates a copy of a double array to a complex double array
        for(int i = 0; i < size; i++ ){
            out[i] = complex<double>(in[i], 0.0);
        }
}

static void hogbom_clean(
        const double* dirty_img,
        const double* psf,
        const double* cbeam_details,
        const double loop_gain,
        const double threshold,
        const double cycle_limit,
        const int64_t dirty_img_dim,
        const int64_t psf_dim,
        double* skymodel,
        sdp_Error* status
){
        // calculate useful shapes and sizes
        int64_t dirty_img_size = dirty_img_dim * dirty_img_dim;
        int64_t psf_size = psf_dim * psf_dim;

        int64_t dirty_img_shape[] = {dirty_img_dim, dirty_img_dim};
        int64_t psf_shape[] = {psf_dim, psf_dim};

        // Create intermediate data arrays
        sdp_Mem* cbeam_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, psf_shape, status);
        complex<double>* cbeam_ptr = (complex<double>*)sdp_mem_data(cbeam_mem);
        sdp_Mem* clean_comp_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, dirty_img_shape, status);
        complex<double>* clean_comp_ptr = (complex<double>*)sdp_mem_data(clean_comp_mem);
        sdp_Mem* residual_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, dirty_img_shape, status);
        complex<double>* residual_ptr = (complex<double>*)sdp_mem_data(residual_mem);

        create_copy_complex(dirty_img, dirty_img_size, residual_ptr);
        
        // set up some loop variables
        int cur_cycle = 0;
        bool stop = 0;

        // create CLEAN Beam
        create_cbeam(cbeam_details, psf_dim, cbeam_ptr);

        // CLEAN loop executes while the stop conditions (threashold and cycle limit) are not met
        while (cur_cycle < cycle_limit && stop == 0) {

            // Find index and value of the maximum value in residual
            double highest_value = std::real(residual_ptr[0]);
            int max_idx_flat = 0;

            for (int i = 0; i < dirty_img_size; i++) {
                if (std::real(residual_ptr[i]) > highest_value) {
                    highest_value = std::real(residual_ptr[i]);
                    max_idx_flat = i;
                }
            }
            
            // check maximum value against threshold
            if (std::real(residual_ptr[max_idx_flat]) < threshold) {
                stop = 1;
                break;
            }

            // unravel x and y from flat index
            int max_idx_x;
            int max_idx_y;

            max_idx_x = max_idx_flat / dirty_img_dim;
            max_idx_y = max_idx_flat % dirty_img_dim;

            // add fraction of maximum to clean components list
            clean_comp_ptr[max_idx_flat] += complex<double>(loop_gain * highest_value, 0);

            // identify psf window to subtract from residual
            int64_t psf_x_start, psf_x_end, psf_y_start, psf_y_end;

            psf_x_start = dirty_img_dim - max_idx_x;
            psf_x_end = psf_x_start + dirty_img_dim;
            psf_y_start = dirty_img_dim - max_idx_y;
            psf_y_end = dirty_img_dim - max_idx_y + dirty_img_dim;

            for (int x = psf_x_start, i = 0; x < psf_x_end; x++, i++){
                for (int y = psf_y_start, j = 0; y < psf_y_end; y++, j++){

                    const unsigned int i_psf = INDEX_2D(psf_dim,psf_dim,x,y);
                    const unsigned int i_res = INDEX_2D(dirty_img_dim,dirty_img_dim,i,j);
                    residual_ptr[i_res] -= complex<double>((loop_gain * highest_value * psf[i_psf]), 0.0);
                }
            }

             cur_cycle += 1;
        }


        // convolve clean components with clean beam
        /* 
        use the convolution theorem with the SDP FFT function to complete this      
        */

        // pad images
        int64_t pad_dim = dirty_img_dim + psf_dim - 1;

        // make sure padded image is a power of 2
        while (ceil(log2(pad_dim)) != floor(log2(pad_dim))){

            pad_dim += 1;
        }

        int64_t pad_shape[] = {pad_dim, pad_dim};
        int64_t pad_size = pad_dim * pad_dim;
        
        sdp_Mem* cbeam_pad_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, pad_shape, status);
        complex<double>* cbeam_pad_ptr = (complex<double>*)sdp_mem_data(cbeam_pad_mem);

        sdp_mem_clear_contents(cbeam_pad_mem, status);

        int64_t extra_cbeam = (pad_dim - psf_dim)/2;

        for (int x = extra_cbeam, i = 0; x < (extra_cbeam + psf_dim); x++, i ++){
            for (int y = extra_cbeam, j = 0; y < (extra_cbeam + psf_dim); y++, j++){

                const unsigned int i_padded = INDEX_2D(pad_dim,pad_dim,x,y);
                const unsigned int i_cbeam = INDEX_2D(psf_dim,psf_dim,i,j);

                cbeam_pad_ptr[i_padded] = cbeam_ptr[i_cbeam];
            }
        }

        sdp_Mem* clean_comp_pad_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, pad_shape, status);
        complex<double>* clean_comp_pad_ptr = (complex<double>*)sdp_mem_data(clean_comp_pad_mem);

        sdp_mem_clear_contents(clean_comp_pad_mem, status);

        int64_t extra_clean_comp = (pad_dim - dirty_img_dim)/2;

        for (int x = extra_clean_comp, i = 0; x < (extra_clean_comp + dirty_img_dim); x++, i ++){
            for (int y = extra_clean_comp, j = 0; y < (extra_clean_comp + dirty_img_dim); y++, j++){

                const unsigned int i_padded = INDEX_2D(pad_dim,pad_dim,x,y);
                const unsigned int i_clean_comp = INDEX_2D(dirty_img_dim,dirty_img_dim,i,j);

                clean_comp_pad_ptr[i_padded] = clean_comp_ptr[i_clean_comp];
            }
        }

        sdp_Mem* cbeam_fft_result_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, pad_shape, status);
        complex<double>* cbeam_fft_result_ptr = (complex<double>*)sdp_mem_data(cbeam_fft_result_mem);
        sdp_Mem* clean_comp_fft_result_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, pad_shape, status);
        complex<double>* clean_comp_fft_result_ptr = (complex<double>*)sdp_mem_data(clean_comp_fft_result_mem);

        // get FFT of padded clean beam
        sdp_Fft *cbeam_fft_plan = sdp_fft_create(cbeam_pad_mem, cbeam_fft_result_mem, 2, 1, status);
        sdp_fft_exec(cbeam_fft_plan, cbeam_pad_mem, cbeam_fft_result_mem, status);
        sdp_fft_free(cbeam_fft_plan);
        
        // get FFT of padded clean components
        sdp_Fft *clean_comp_fft_plan = sdp_fft_create(clean_comp_pad_mem, clean_comp_fft_result_mem, 2, 1, status);
        sdp_fft_exec(clean_comp_fft_plan, clean_comp_pad_mem, clean_comp_fft_result_mem, status);
        sdp_fft_free(clean_comp_fft_plan);

        sdp_Mem* multiply_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, pad_shape, status);
        complex<double>* multiply_ptr = (complex<double>*)sdp_mem_data(multiply_mem);

        sdp_mem_clear_contents(multiply_mem, status);

        // multiply FFTs together
        for (int i = 0; i < pad_size; i++){
            
            multiply_ptr[i] = clean_comp_fft_result_ptr[i] * cbeam_fft_result_ptr[i];

            // double a_real = std::real(clean_comp_fft_result_ptr[i]);
            // double b_real = std::real(cbeam_fft_result_ptr[i]);
            // double a_img = std::imag(clean_comp_fft_result_ptr[i]);
            // double b_img = std::imag(cbeam_fft_result_ptr[i]);
                

            // multiply_ptr[i] = complex<double>(a_real * b_real - a_img * b_img, a_real * b_img + a_img * b_real);

        }

        // inverse FFT of result
        sdp_Mem* multiply_fft_result_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, pad_shape, status);
        complex<double>* multiply_fft_result_ptr = (complex<double>*)sdp_mem_data(multiply_fft_result_mem);
        
        sdp_Fft *conv_fft_plan = sdp_fft_create(multiply_mem, multiply_fft_result_mem,2,0,status);
        sdp_fft_exec(conv_fft_plan,multiply_mem,multiply_fft_result_mem,status);
        sdp_fft_free(conv_fft_plan);

        // shift edges to centre (FFTshift) and remove padding
        for (int x = (pad_dim - dirty_img_dim/2), i = 0; x < (pad_dim); x++, i ++){
            for (int y = (pad_dim - dirty_img_dim/2), j = 0; y < (pad_dim); y++, j++){

                const unsigned int i_quarter = INDEX_2D(pad_dim,pad_dim,x,y);
                const unsigned int i_skymodel = INDEX_2D(dirty_img_dim,dirty_img_dim,i,j);

                skymodel[i_skymodel] = std::real(multiply_fft_result_ptr[i_quarter]);

            }
        }

        for (int x = (0), i = (dirty_img_dim/2); x < (dirty_img_dim/2); x++, i ++){
            for (int y = (pad_dim - dirty_img_dim/2), j = 0; y < (pad_dim); y++, j++){

                const unsigned int i_quarter = INDEX_2D(pad_dim,pad_dim,x,y);
                const unsigned int i_skymodel = INDEX_2D(dirty_img_dim,dirty_img_dim,i,j);

                skymodel[i_skymodel] = std::real(multiply_fft_result_ptr[i_quarter]);

            }
        }

        for (int x = (pad_dim - dirty_img_dim/2), i = 0; x < (pad_dim); x++, i ++){
            for (int y = (0), j = (dirty_img_dim/2); y < (dirty_img_dim/2); y++, j++){

                const unsigned int i_quarter = INDEX_2D(pad_dim,pad_dim,x,y);
                const unsigned int i_skymodel = INDEX_2D(dirty_img_dim,dirty_img_dim,i,j);

                skymodel[i_skymodel] = std::real(multiply_fft_result_ptr[i_quarter]);

            }
        }

        for (int x = (0), i = (dirty_img_dim/2); x < (dirty_img_dim/2); x++, i ++){
            for (int y = (0), j = (dirty_img_dim/2); y < (dirty_img_dim/2); y++, j++){

                const unsigned int i_quarter = INDEX_2D(pad_dim,pad_dim,x,y);
                const unsigned int i_skymodel = INDEX_2D(dirty_img_dim,dirty_img_dim,i,j);

                skymodel[i_skymodel] = std::real(multiply_fft_result_ptr[i_quarter]);

            }
        }

        sdp_mem_ref_dec(cbeam_mem);
        sdp_mem_ref_dec(clean_comp_mem);
        sdp_mem_ref_dec(residual_mem);
        sdp_mem_ref_dec(cbeam_pad_mem);
        sdp_mem_ref_dec(clean_comp_pad_mem);
        sdp_mem_ref_dec(clean_comp_fft_result_mem);
        sdp_mem_ref_dec(cbeam_fft_result_mem);
        sdp_mem_ref_dec(multiply_mem);
        sdp_mem_ref_dec(multiply_fft_result_mem);

}

void sdp_hogbom_clean(
        const sdp_Mem* dirty_img,
        const sdp_Mem* psf,
        const sdp_Mem* cbeam_details,
        const double loop_gain,
        const double threshold,
        const double cycle_limit,
        sdp_Mem* skymodel,
        sdp_Error* status
){
    if (*status) return;

    const int64_t dirty_img_dim = sdp_mem_shape_dim(dirty_img, 0);
    const int64_t psf_dim = sdp_mem_shape_dim(psf, 0);

    if (sdp_mem_is_read_only(skymodel)){
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Output is not writable");
        return;
    }

    if (sdp_mem_location(dirty_img) != sdp_mem_location(psf)){
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Memory location mismatch");
        return;
    }

    if (sdp_mem_type(dirty_img) != sdp_mem_type(psf)){
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("The Dirty image and PSF must be of the same data type");
        return;
    }

    if (sdp_mem_location(dirty_img) == SDP_MEM_CPU)
    {
        hogbom_clean(
            (const double*)sdp_mem_data_const(dirty_img),
            (const double*)sdp_mem_data_const(psf),
            (const double*)sdp_mem_data_const(cbeam_details),
            loop_gain,
            threshold,
            cycle_limit,
            dirty_img_dim,
            psf_dim,
            (double*)sdp_mem_data(skymodel),
            status
        );
    }
    
}