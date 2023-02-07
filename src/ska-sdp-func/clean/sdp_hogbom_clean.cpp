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

#include <string.h>
#include <math.h>

#define INDEX_2D(N2, N1, I2, I1)    (N1 * I2 + I1)

inline void create_cbeam(
        const double* cbeam_details,
        int16_t psf_dim,
        double* cbeam
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

    for(double x = 0; x <= psf_dim; x += 1) {
        for(double y = 0; y <= psf_dim; y += 1) {

            const unsigned int i_cbeam = INDEX_2D(psf_dim,psf_dim,x,y);
            cbeam[i_cbeam] = A * exp(-(a * pow(x - x0,2) + 2 * b * (x - x0) * (y - y0) + c * pow(y - y0,2)));

            }
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
        int64_t dirty_img_size = dirty_img_dim * dirty_img_dim;

        // Create intermediate data arrays
        double* maximum = (double*) calloc(cycle_limit * 3, sizeof(double));
        double* clean_comp = (double*) calloc(dirty_img_size, sizeof(double));
        double* residual = (double*) malloc(dirty_img_size * sizeof(double));
        memcpy(residual, dirty_img, dirty_img_size * sizeof(double));
        double* multiply = (double*) malloc(dirty_img_size * sizeof(double));
        double* result_of_convolution = (double*) malloc(dirty_img_size * sizeof(double));
        double* cbeam = (double*) calloc(psf_dim, sizeof(double));

        // set up some loop variables
        int cur_cycle = 0;
        bool stop = false;

        // create CLEAN Beam
        create_cbeam(cbeam_details, psf_dim, cbeam);

        // CLEAN loop executes while the stop conditions (threashold and cycle limit) are not met
        while (cur_cycle < cycle_limit && stop == false) {

            printf("Current Cycle: %d of %d Cycles Limit\r", cur_cycle, cycle_limit); // for debugging, to be removed when finished

            // Find index and value of the maximum value in residual
            double highest_value = residual[0];
            int max_idx_flat = 0;

            for (int i = 0; i < dirty_img_size; i++) {
                if (residual[i] > highest_value) {
                    highest_value = residual[i];
                    max_idx_flat = i;
                }
            }
            
            // check maximum value against threshold
            if (residual[max_idx_flat] < threshold) {
                stop = 1;
                break;
            }

            // unravel x and y from flat index
            int max_idx_x;
            int max_idx_y;

            max_idx_x = max_idx_flat / dirty_img_size;
            max_idx_y = max_idx_flat % dirty_img_size;

            // Save position and peak value
            unsigned int i_maximum = INDEX_2D(cycle_limit,3,cur_cycle,0);
            maximum[i_maximum] = highest_value;

            i_maximum = INDEX_2D(cycle_limit,3,cur_cycle,1);
            maximum[cur_cycle + 1] = max_idx_x;

            i_maximum = INDEX_2D(cycle_limit,3,cur_cycle,2);
            maximum[cur_cycle + 2] = max_idx_y;

            // add fraction of maximum to clean components list
            clean_comp[max_idx_flat] += loop_gain * highest_value;

            // identify psf window to subtract from residual
            int psf_x_start, psf_x_end, psf_y_start, psf_y_end;

            psf_x_start = dirty_img_size - max_idx_x;
            psf_x_end = dirty_img_size - max_idx_x + dirty_img_size;
            psf_y_start = dirty_img_size - max_idx_y;
            psf_y_end = dirty_img_size - max_idx_y + dirty_img_size;

            for (int x = psf_x_start, i = 0; x < psf_x_end; x++, i++){
                for (int y = psf_y_start, j = 0; y < psf_y_end; y++, j++){

                    const unsigned int i_psf = INDEX_2D(psf_dim,psf_dim,x,y);
                    const unsigned int i_res = INDEX_2D(dirty_img_dim,dirty_img_dim,i,j);
                    residual[i_res] -= (loop_gain * highest_value * psf[i_psf]);
                }
            }

             cur_cycle += 1;
        }

        // convolve clean components with clean beam
        /* 
        use the convolution theorem with the SDP FFT function to complete this
        save the result to result_of_convolution for the next step.        
        */

        // wrap cbeam in sdp_mem, so it can be used with FFT code
        int64_t cbeam_shape[] = {psf_dim, psf_dim};
        sdp_Mem* cbeam_mem = sdp_mem_create_wrapper(cbeam, SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, cbeam_shape, NULL, status);

        // wrap clean compentents in sdp_mem, so it can be used with FFT code
        int64_t dirty_img_shape[] = {dirty_img_dim, dirty_img_dim};
        sdp_Mem* clean_com_mem = sdp_mem_create_wrapper(clean_comp, SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, dirty_img_shape, NULL, status);

        // get FFT of clean components
        sdp_Mem* clean_comp_fft_result = sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, dirty_img_shape, status);
        
        sdp_Fft *clean_comp_fft_plan = sdp_fft_create(clean_com_mem, clean_comp_fft_result, 2, 1, status);
        sdp_fft_exec(clean_comp_fft_plan, clean_com_mem, clean_comp_fft_result, status);
        sdp_fft_free(clean_comp_fft_plan);

        // get FFT of clean beam
        sdp_Mem* cbeam_fft_result = sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, cbeam_shape, status);
        
        sdp_Fft *cbeam_fft_plan = sdp_fft_create(cbeam_mem, cbeam_fft_result,2,1,status);
        sdp_fft_exec(cbeam_fft_plan, cbeam_mem, cbeam_fft_result, status);
        sdp_fft_free(cbeam_fft_plan);

        // multiply FFTs together
        double* clean_comp_fft = (double*)sdp_mem_data(clean_comp_fft_result);
        double* cbeam_fft = (double*)sdp_mem_data(cbeam_fft_result);

        for (int i = 0; i < dirty_img_size; i++){
            
            multiply[i] = clean_comp_fft[i] * cbeam_fft[i];
        }

        // wrap multiply in sdp_mem, so it can be used with FFT code
        sdp_Mem* multiply_mem = sdp_mem_create_wrapper(multiply, SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, cbeam_shape, NULL, status);

        // inverse FFT of result
        sdp_Mem* multip_result = sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, dirty_img_shape, status);
        
        sdp_Fft *conv_fft_plan = sdp_fft_create(multiply_mem, multip_result,2,0,status);
        sdp_fft_exec(conv_fft_plan,multiply_mem,multip_result,status);
        sdp_fft_free(conv_fft_plan);

        double* result_of_convolution = (double*)sdp_mem_data(multip_result);

        // add residual to the results of the convolution
        for (int i = 0; i < dirty_img_size; i++){
            
            skymodel[i] = result_of_convolution[i] + residual[i];
        }

        sdp_mem_ref_dec(clean_comp_fft_result);
        sdp_mem_ref_dec(cbeam_fft_result);
        sdp_mem_ref_dec(multip_result);
        free(maximum);
        free(clean_comp);
        free(residual);
        free(multiply);
        free(result_of_convolution);
        free(cbeam);
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

    if (sdp_mem_location(dirty_img) != sdp_mem_location(psf) != sdp_mem_location(cbeam_details)){
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