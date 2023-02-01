/* See the LICENSE file at the top-level directory of this distribution. 

MSCLEAN algorithm taken from:
Offringa, A. R., and O. Smirnov. "An optimized algorithm for multiscale wideband deconvolution of radio
astronomical images." Monthly Notices of the Royal Astronomical Society, 471.1 (2017): 301-316.
https://arxiv.org/pdf/1706.06786.pdf

*/

#include "ska-sdp-func/clean/sdp_ms_clean.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/fourier_transforms/sdp_fft.h"

#include <string.h>
#include <math.h>

#define INDEX_2D(N2, N1, I2, I1)            (N1 * I2 + I1)
#define INDEX_3D(N3, N2, N1, I3, I2, I1)    (N1 * (N2 * I3 + I2) + I1)

inline void create_cbeam(
        const double* psf,
        double* cbeam,
        int16_t psf_dim
){
    // fit a guassian to the main lobe of the psf

    double A = 1;
    double x0 = psf_dim/2;
    double y0 = psf_dim/2;
    double sigma_X = 10;
    double sigma_Y = 20;
    double theta = 0;

    double a = pow(cos(theta),2) / (2 * pow(sigma_X,2)) + pow(sin(theta),2) / (2 * pow(sigma_Y,2));
    double b = sin(2 * theta) / (4 * pow(sigma_X,2)) - sin(2 * theta) / (4 * pow(sigma_Y,2));
    double c = pow(sin(theta),2) / (2 * pow(sigma_X,2)) + pow(cos(theta),2) / (2 * pow(sigma_Y,2));

    for(int x = 0; x <= psf_dim; x += 1) {
        for(int y = 0; y <= psf_dim; y += 1) {

            const unsigned int i_cbeam = INDEX_2D(psf_dim,psf_dim,x,y);
            cbeam[i_cbeam] = A * exp(-(a * pow(x - x0,2) + 2 * b * (x - x0) * (y - y0) + c * pow(y - y0,2)));

            }
        }
}

inline void sub_minor_loop(
        double *residual,
        double *psf,
        int scale_to_clean,
        double loop_gain,
        double threshold,
        int cycle_limit,
        int64_t dirty_img_size,
        int64_t dirty_img_dim,
        int64_t psf_dim,
        int64_t scale_dim,
        double* clean_comp

){

        // set up some loop variables
        int cur_cycle = 0;
        bool stop = false;

         // CLEAN loop executes while the stop conditions (threashold and cycle limit) are not met
        while (cur_cycle < cycle_limit && stop == false) {

            printf("Current Cycle: %d of %d Cycles Limit\r", cur_cycle, cycle_limit); // for debugging, to be removed when finished

            // Find index and value of the maximum value in residual
            double highest_value = 0;
            int max_idx_flat = 0;

            for(int x = 0; x < dirty_img_dim; x++){
                for(int y = 0; x < dirty_img_dim; y++){

                    const unsigned int i_residual = INDEX_3D(scale_dim,dirty_img_dim,dirty_img_dim,scale_to_clean,x,y);

                    if (residual[i_residual] > highest_value) {
                        highest_value = residual[i_residual];
                        max_idx_flat = i_residual;
                    }
                }
            }
            
            // check maximum value against threshold
            if (residual[max_idx_flat] < threshold) {
                stop = 1;
                break;
            }

            // unravel x and y for scale to be cleaned from flat index
            int max_idx_x;
            int max_idx_y;
            
            max_idx_x = (max_idx_flat - scale_to_clean * dirty_img_dim * dirty_img_dim)/dirty_img_dim;
            max_idx_y = max_idx_flat - scale_to_clean * dirty_img_dim * dirty_img_dim - max_idx_x * dirty_img_dim;

            // add fraction of maximum to clean components list
            const unsigned int i_clean_comp = INDEX_2D(dirty_img_dim,dirty_img_dim,max_idx_x,max_idx_y);
            clean_comp[i_clean_comp] += loop_gain * highest_value;

            // identify psf window to subtract from residual
            int psf_x_start, psf_x_end, psf_y_start, psf_y_end;

            psf_x_start = dirty_img_size - max_idx_x;
            psf_x_end = dirty_img_size - max_idx_x + dirty_img_size;
            psf_y_start = dirty_img_size - max_idx_y;
            psf_y_end = dirty_img_size - max_idx_y + dirty_img_size;

            for (int x = psf_x_start, i = 0; x < psf_x_end; x++, i++){
                for (int y = psf_y_start, j = 0; y < psf_y_end; y++, j++){

                    const unsigned int i_psf = INDEX_3D(scale_dim, psf_dim,psf_dim,scale_to_clean,x,y);
                    const unsigned int i_res = INDEX_3D(scale_dim, dirty_img_dim,dirty_img_dim,scale_to_clean,i,j);
                    residual[i_res] -= (loop_gain * highest_value * psf[i_psf]);
                }
            }

             cur_cycle += 1;
        }

}

inline void fft_convolve(
        double* in1,
        int64_t in1_dim,
        double* in2,
        int64_t in2_dim,
        double* out,
        sdp_Error* status
){
        // TODO: zero padding for different sizes, will only work for same size at the moment

        // wrap in1 in sdp_mem, so it can be used with FFT code
        int64_t in1_shape[] = {in1_dim, in1_dim};
        sdp_Mem* in1_mem = sdp_mem_create_wrapper(in1, SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, in1_shape, NULL, status);

        // wrap in2 in sdp_mem, so it can be used with FFT code
        int64_t in2_shape[] = {in2_dim, in2_dim};
        sdp_Mem* in2_mem = sdp_mem_create_wrapper(in2, SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, in2_shape, NULL, status);

        // sdp_Mem object to store the result of in1 fft
        sdp_Mem* in1_fft_result = sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, in1_shape, status);
        
        // get fft of in1
        sdp_Fft *in1_fft_plan = sdp_fft_create(in1_mem, in1_fft_result, 2, 1, status);
        sdp_fft_exec(in1_fft_plan, in1_mem, in1_fft_result, status);
        sdp_fft_free(in1_fft_plan);

        // sdp_Mem object to store the result of in2 fft
        sdp_Mem* in2_fft_result = sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, in2_shape, status);
        
        // get fft of in2
        sdp_Fft *in2_fft_plan = sdp_fft_create(in2_mem, in2_fft_result, 2, 1, status);
        sdp_fft_exec(in2_fft_plan, in2_mem, in2_fft_result, status);
        sdp_fft_free(in2_fft_plan);

        //create memory to hold the results of multiplying ffts together
        double* multiply = (double*) malloc(in1_dim * in1_dim * sizeof(double));

        // multiply FFTs together
        double* in1_fft = (double*)sdp_mem_data(in1_fft_result);
        double* in2_fft = (double*)sdp_mem_data(in2_fft_result);

        for(int x = 0; x <= in1_dim; x += 1) {
            for(int y = 0; y <= in1_dim; y += 1) {

            const unsigned int i_multiply = INDEX_2D(in1_dim,in1_dim,x,y);
            
                multiply[i_multiply] = in1_fft[i_multiply] * in2_fft[i_multiply];
            }
        }

        // wrap multiply_result in sdp_mem, so it can be used with FFT code
        int64_t in1_shape[] = {in1_dim, in1_dim};
        sdp_Mem* multiply_mem = sdp_mem_create_wrapper(multiply_mem, SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, in1_shape, NULL, status);
        
        // sdp_Mem object to store result of multiply iFFT
        sdp_Mem* multiply_fft_result = sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, in1_shape, status);

        //get iFFT of multiply
        sdp_Fft *multiply_fft_plan = sdp_fft_create(multiply_mem, multiply_fft_result,2,0,status);
        sdp_fft_exec(multiply_fft_plan,multiply_mem,multiply_fft_result,status);
        sdp_fft_free(multiply_fft_plan);

        out = (double*)sdp_mem_data(multiply_fft_result);

        free(multiply);
        sdp_mem_ref_dec(in1_fft_result);
        sdp_mem_ref_dec(in2_fft_result);
        sdp_mem_ref_dec(multiply_fft_result);

}

inline void create_scale_bias(
        const double *scale_list,
        const int64_t scale_dim,
        double *scale_bias
) {
    // create the scale bias terrms
    double first_scale = 0;

    for (int i = 0; i < scale_dim; i++) {
        double scale = scale_list[i];

        if (i == 0 && scale != 0) {
        first_scale = scale;
        }

        if (i == 1 && scale != 0 && first_scale == 0) {
        first_scale = scale;
        }

        if (scale == 0) {
        scale_bias[i] = 1;
        } 
        else {
        scale_bias[i] = pow(0.6, -1 - log2(scale/first_scale));
        }
    }
}

inline void create_scale_kernels(
        const double* scale_list,
        const int64_t scale_dim,
        const double dirty_img_dim,
        double* scale_kernels
){
    
    // create a scale kernel for each scale specified in scale_list

    // find sigma from list of scales
    double* sigma_list = (double*) malloc(scale_dim * sizeof(double));

    for (int i = 0; i < scale_dim; i++) {
        double scale = scale_list[i];

        if (scale == 0) {
        sigma_list[i] = 0;
        }
        else {
        sigma_list[i] = (3.0 / 16.0) * scale;
        }
    }

    for (int i = 0; i < scale_dim; i++) {
        double* gaus = (double*) malloc(dirty_img_dim * sizeof(double));
        int mu = dirty_img_dim / 2;
        double twoSigmaSquared = 2.0 * pow(sigma_list[i],2);
        double sum = 0;

        for (int y = 0; y < dirty_img_dim; y++) {
            double vI = (double)y - mu;
            gaus[y] = exp(-vI * vI / twoSigmaSquared);
        }

        for (int y = 0; y < dirty_img_dim; y++) {
            for (int x = 0; x < dirty_img_dim; x++) {
                const unsigned int i_kernel = INDEX_3D(scale_dim,dirty_img_dim,dirty_img_dim,i,x,y);
                scale_kernels[i_kernel] = gaus[x] * gaus[y];
                sum += scale_kernels[i_kernel];
            }
        }

        double normFactor = 1.0 / sum;

        for (int y = 0; y < dirty_img_dim; y++) {
            for (int x = 0; x < dirty_img_dim; x++) {
                const unsigned int i_kernel = INDEX_3D(scale_dim,dirty_img_dim,dirty_img_dim,i,x,y);
                scale_kernels[i_kernel] *= normFactor;
            }
        }
        free(gaus);

    }

    free(sigma_list);    
}


inline void create_scaled_residual(
        const int64_t scale_dim,
        const int64_t dirty_img_dim,
        double* scale_kernels,
        const double* residulal,
        double* scale_residual,
        sdp_Error* status
){
        // convole scale kernels with the residulas to create scaled residual images for each scale in scale_list
        // read each kernel in to a new variable so it can be 
        double* curr_kernel = (double*) malloc(dirty_img_dim * dirty_img_dim * sizeof(double));
        // variable to save result
        double* result = (double*) malloc(dirty_img_dim * dirty_img_dim * sizeof(double));
        
        for(int s = 0; s < scale_dim; s++){
            for(int x = 0; x < dirty_img_dim; x++){
                for(int y = 0; x < dirty_img_dim; y++){

                    const unsigned int i_scale_kernel = INDEX_3D(scale_dim,dirty_img_dim,dirty_img_dim,s,x,y);
                    const unsigned int i_curr_kernel = INDEX_2D(dirty_img_dim,dirty_img_dim,x,y);

                    curr_kernel[i_curr_kernel] = scale_kernels[i_scale_kernel];
                }
            }

            fft_convolve((double*)residulal, dirty_img_dim, curr_kernel, dirty_img_dim, result, status);
        
            // read result in to list of scaled residuals
            for(int x = 0; x < dirty_img_dim; x++){
                for(int y = 0; x < dirty_img_dim; y++){

                    const unsigned int i_scale_residual = INDEX_3D(scale_dim,dirty_img_dim,dirty_img_dim,s,x,y);
                    const unsigned int i_curr_result = INDEX_2D(dirty_img_dim,dirty_img_dim,x,y);

                    scale_residual[i_scale_residual] = result[i_curr_result];
                    
                }
            }
        }

        free(curr_kernel);
        free(result);

}


static void ms_clean(
        const double* dirty_img,
        const double* psf,
        const double* scale_list,
        const double loop_gain,
        const double threshold,
        const int cycle_limit,
        const int sub_minor_cycle_limit,
        const int64_t dirty_img_dim,
        const int64_t psf_dim,
        const int64_t scale_dim,
        const double ms_gain,
        double* skymodel,
        sdp_Error* status
){
        int64_t dirty_img_size = dirty_img_dim * dirty_img_dim;
        int64_t psf_size = psf_dim * psf_dim;

        // Create intermediate data arrays
        double* clean_comp = (double*) calloc(dirty_img_size, sizeof(double)); // holds the clean components returned from sub minor loop [x][y]
        double* model = (double*) calloc(dirty_img_size, sizeof(double)); // holds the model image [x][y]
        double* residual = (double*) malloc(dirty_img_size * sizeof(double)); // hold the residuals [x][y]
        memcpy(residual, dirty_img, dirty_img_size * sizeof(double)); // copy the dirty image to the residual to initialise [x][y]
        double* cbeam = (double*) calloc(psf_dim, sizeof(double)); // holds the clean beam
        double* scale_kernels = (double*) malloc(dirty_img_size * scale_dim * sizeof(double)); // holds the list of generated scale kernels [scale][x][y]
        double* scale_psf = (double*) malloc(psf_size * scale_dim * sizeof(double)); // holds the list of generated scaled PSFs [scale][x][y]
        double* scale_residual = (double*) malloc(dirty_img_size * scale_dim * sizeof(double)); // holds the list of generated scaled residuals [scale][x][y]
        double* highest_value_by_scale = (double*) calloc(scale_dim, sizeof(double)); // hold the list of peak values detected in each scaled residual [scale]
        double* max_idx_flat = (double*) calloc(scale_dim, sizeof(unsigned int)); // hold the list of peak co-ordinates detected in each scaled residual [scale]
        double* scale_bias = (double*) calloc(scale_dim, sizeof(double)); // holds the list of generated scale biases [scale]
        double* curr_kernel = (double*) malloc(dirty_img_size * sizeof(double)); // hold the current kernel being worked with [x][y]
        double* result = (double*) malloc(psf_size * sizeof(double)); // holds the results of an FFT convolution [x][y]


        // set up some loop variables
        int cur_cycle = 0;
        bool stop = false;

        // create CLEAN Beam
        create_cbeam(psf, cbeam, psf_dim);

        // create list of scale bias
        create_scale_bias(scale_list,scale_dim,scale_bias);

        // create scale kernels
        create_scale_kernels(scale_list, scale_dim, dirty_img_dim, scale_kernels);

        // create scaled PSFs
        // read each kernel in to a new variable to pass to fft_convolve
        // create new variables for results

        for(int s = 0; s < scale_dim; s++){
            for(int x = 0; x < dirty_img_dim; x++){
                for(int y = 0; x < dirty_img_dim; y++){

                    const unsigned int i_scale_kernel = INDEX_3D(scale_dim,dirty_img_dim,dirty_img_dim,s,x,y);
                    const unsigned int i_curr_kernel = INDEX_2D(dirty_img_dim,dirty_img_dim,x,y);

                    curr_kernel[i_curr_kernel] = scale_kernels[i_scale_kernel];
                    
                }
            }

            // convolve psf with scale kernel twice
            fft_convolve((double*)psf, psf_dim, curr_kernel, dirty_img_dim, result, status);
            fft_convolve(result, psf_dim, curr_kernel, dirty_img_dim, result, status);

            // read result in to list of scaled psfs
            for(int x = 0; x < psf_dim; x++){
                for(int y = 0; x < psf_dim; y++){

                    const unsigned int i_scale_psf = INDEX_3D(scale_dim,psf_dim,psf_dim,s,x,y);
                    const unsigned int i_curr_result = INDEX_2D(psf_dim,psf_dim,x,y);

                    scale_psf[i_scale_psf] = result[i_curr_result];
                    
                }
            }
        }

        // CLEAN loop executes while the stop conditions (threashold and cycle limit) are not met
        while (cur_cycle < cycle_limit && stop == false) {

            // create scaled dirty images / residulas
            create_scaled_residual(scale_dim, dirty_img_dim, scale_kernels, residual, scale_residual, status);

            // find max value for each scale
            for(int s = 0; s < scale_dim; s++){
                for(int x = 0; x < dirty_img_dim; x++){
                    for(int y = 0; x < dirty_img_dim; y++){

                        const unsigned int i_scale_residual = INDEX_3D(scale_dim,dirty_img_dim,dirty_img_dim,s,x,y);

                        if (residual[i_scale_residual] > highest_value_by_scale[s]) {
                            highest_value_by_scale[s] = residual[i_scale_residual];
                            max_idx_flat[s] = i_scale_residual;
                        }
                    }
                }
            }

            // multiply each max by its corresponding scale bias and find the highest biased max
            int highest_scale = 0;
            double highest_biased_value = 0;
            for(int s = 0; s < scale_dim; s++){
                highest_value_by_scale[s] *= scale_bias[s];

                if(highest_value_by_scale[s] > highest_biased_value){
                    highest_biased_value = highest_value_by_scale[s];
                    highest_scale = s;
                }
            }

            // target value to CLEAN the highest peak down to
            double stop_val = highest_value_by_scale[highest_scale] * ms_gain;

            // check the value to be CLEANed is not below the threashold
            if(stop_val < threshold){
                stop = true;
                break;
            }

            sub_minor_loop(scale_residual, scale_psf, highest_scale, loop_gain, stop_val, sub_minor_cycle_limit, dirty_img_size, dirty_img_dim, psf_dim, scale_dim, clean_comp);

            // read kernel for selected scale
            for(int x = 0; x < dirty_img_dim; x++){
                for(int y = 0; x < dirty_img_dim; y++){

                    const unsigned int i_scale_kernel = INDEX_3D(scale_dim,dirty_img_dim,dirty_img_dim,highest_scale,x,y);
                    const unsigned int i_curr_kernel = INDEX_2D(dirty_img_dim,dirty_img_dim,x,y);

                    curr_kernel[i_curr_kernel] = scale_kernels[i_scale_kernel];
                    
                }
            }

            // convolve returned clean components with current kernel and add to model
            fft_convolve(clean_comp, dirty_img_dim, curr_kernel, dirty_img_dim, result, status);

            for (int i = 0; i < dirty_img_size; i++){
                model[i] += result[i];
            }

            // convolve the convolved clean components by the psf and subtract from residual
            fft_convolve(result, dirty_img_dim, (double*)psf, psf_dim, result, status);

            for (int i = 0; i < dirty_img_size; i++){
                residual[i] -= result[i];
            }

            cur_cycle += 1;
        }

        // convolve the model with the clean beam and add residuals
        fft_convolve(model, dirty_img_dim, cbeam, psf_dim, result, status);

        for (int i = 0; i < dirty_img_size; i++){
                skymodel[i] = result[i] + residual[i];
            }

        // free memory
        free(clean_comp);
        free(model);
        free(residual);
        free(cbeam);
        free(scale_kernels);
        free(scale_psf);
        free(scale_residual);
        free(highest_value_by_scale);
        free(max_idx_flat);
        free(scale_bias);
        free(curr_kernel);
        free(result);


}


void sdp_ms_clean(
        const sdp_Mem* dirty_img,
        const sdp_Mem* psf,
        const sdp_Mem* scale_list,
        const double loop_gain,
        const double threshold,
        const int cycle_limit,
        const int sub_minor_cycle_limit,
        const double ms_gain,
        sdp_Mem* skymodel,
        sdp_Error* status
){
    if (*status) return;

    const int64_t dirty_img_dim = sdp_mem_shape_dim(dirty_img, 0);
    const int64_t psf_dim = sdp_mem_shape_dim(psf, 0);
    const int64_t scale_dim = sdp_mem_shape_dim(scale_list, 0);

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
        ms_clean(
            (const double*)sdp_mem_data_const(dirty_img),
            (const double*)sdp_mem_data_const(psf),
            (const double*)sdp_mem_data_const(scale_list),
            loop_gain,
            threshold,
            cycle_limit,
            sub_minor_cycle_limit,
            dirty_img_dim,
            psf_dim,
            scale_dim,
            ms_gain,
            (double*)sdp_mem_data(skymodel),
            status
        );
    }
    
}