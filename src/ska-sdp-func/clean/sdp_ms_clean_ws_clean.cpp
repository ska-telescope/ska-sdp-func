/* See the LICENSE file at the top-level directory of this distribution. 

MSCLEAN algorithm taken from:
Offringa, A. R., and O. Smirnov. "An optimized algorithm for multiscale wideband deconvolution of radio
astronomical images." Monthly Notices of the Royal Astronomical Society, 471.1 (2017): 301-316.
https://arxiv.org/pdf/1706.06786.pdf

*/

#include "ska-sdp-func/clean/sdp_ms_clean_ws_clean.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/fourier_transforms/sdp_fft.h"
#include "ska-sdp-func/numeric_functions/sdp_fft_convolution.h"

#include <string.h>
#include <math.h>
#include <complex>

using std::complex;

#define INDEX_2D(N2, N1, I2, I1)            (N1 * I2 + I1)
#define INDEX_3D(N3, N2, N1, I3, I2, I1)    (N1 * (N2 * I3 + I2) + I1)

inline void create_cbeam(
        const double* cbeam_details,
        int16_t psf_dim,
        complex<double>* cbeam
){
    // fit a guassian to the main lobe of the psf

    double A = 1;
    double x0 = (psf_dim/2);
    double y0 = (psf_dim/2);
    double sigma_X = cbeam_details[0];
    double sigma_Y = cbeam_details[1];
    double theta = (M_PI/180) * cbeam_details[2];

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

inline void create_copy_real(
        const complex<double>* in,
        int64_t size,
        double* out
){
        // creates a copy of a complex double array to a double array
        for(int i = 0; i < size; i++ ){
            out[i] = std::real(in[i]);
        }
}

inline void sub_minor_loop(
        complex<double>* residual,
        complex<double>* psf,
        int scale_to_clean,
        double loop_gain,
        double threshold,
        int cycle_limit,
        int64_t dirty_img_dim,
        int64_t psf_dim,
        complex<double>* clean_comp

){
        // SDP_LOG_DEBUG("Sub-minor Settings"); // for debugging, to be removed when finished
        // SDP_LOG_DEBUG("loop gain: %f", loop_gain); // for debugging, to be removed when finished
        // SDP_LOG_DEBUG("threshold: %f", threshold); // for debugging, to be removed when finished
        // SDP_LOG_DEBUG("Scale: %d", scale_to_clean); // for debugging, to be removed when finished

        // set up some loop variables
        int cur_cycle = 0;
        bool stop = false;

         // CLEAN loop executes while the stop conditions (threashold and cycle limit) are not met
        while (cur_cycle < cycle_limit && stop == false) {

            // SDP_LOG_DEBUG("Current Cycle: %d of %d Cycles Limit", cur_cycle, cycle_limit); // for debugging, to be removed when finished

            // Find index and value of the maximum value in residual
            double highest_value = 0;
            int max_idx_flat = 0;

            for(int x = 0; x < dirty_img_dim; x++){
                for(int y = 0; y < dirty_img_dim; y++){

                    const unsigned int i_residual = INDEX_3D(scale_dim,dirty_img_dim,dirty_img_dim,scale_to_clean,x,y);

                    if (std::real(residual[i_residual]) > highest_value) {
                        // SDP_LOG_DEBUG("Current Value: %f", std::real(residual[i_residual])); // for debugging, to be removed when finished
                        highest_value = std::real(residual[i_residual]);
                        max_idx_flat = i_residual;
                    }
                }
            }
            
            // check maximum value against threshold
            if (highest_value < threshold) {
                stop = 1;
                break;
            }

            // unravel x and y for scale to be cleaned from flat index
            int max_idx_x;
            int max_idx_y;

            max_idx_x = (max_idx_flat / dirty_img_dim) % dirty_img_dim;
            max_idx_y = max_idx_flat % dirty_img_dim;

            // add fraction of maximum to clean components list
            // SDP_LOG_DEBUG("Adding CLEAN components");
            const unsigned int i_clean_comp = INDEX_2D(dirty_img_dim,dirty_img_dim,max_idx_x,max_idx_y);
            clean_comp[i_clean_comp] += complex<double>(loop_gain * highest_value, 0.0);

            // identify psf window to subtract from residual
            int psf_x_start, psf_x_end, psf_y_start, psf_y_end;

            psf_x_start = dirty_img_dim - max_idx_x;
            psf_x_end = psf_x_start + dirty_img_dim;
            psf_y_start = dirty_img_dim - max_idx_y;
            psf_y_end = psf_y_start + dirty_img_dim;

            for (int x = psf_x_start, i = 0; x < psf_x_end; x++, i++){
                for (int y = psf_y_start, j = 0; y < psf_y_end; y++, j++){

                    const unsigned int i_psf = INDEX_3D(scale_dim, psf_dim,psf_dim,scale_to_clean,x,y);
                    const unsigned int i_res = INDEX_3D(scale_dim, dirty_img_dim,dirty_img_dim,scale_to_clean,i,j);
                    residual[i_res] -= complex<double>(loop_gain * highest_value, 0.0) * psf[i_psf];
                }
            }

             cur_cycle += 1;
        }

}

inline void create_scale_bias(
        const double *scale_list,
        const int64_t scale_dim,
        double *scale_bias
) {
    // create the scale bias terrms
    double first_scale = 0;
    double beta = 0.6;

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
        scale_bias[i] = pow(beta, -1 - log2(scale/first_scale));
        }
    }
}

inline void create_scale_kern(
        complex<double>* scale_kern_list,
        const double* scales,
        int64_t length
) {
    //created a set of 2D gaussians at a specifed scales

    int num_scales = 5;
    double sigma = 0;
    int center_x = length / 2;
    int center_y = length / 2;
    double two_sigma_square = 0;

    for (int scale_idx = 0; scale_idx < num_scales; scale_idx++){
        if (scales[scale_idx] == 0){
            const unsigned int i_list = INDEX_3D(num_scales,length,length,scale_idx,length/2,length/2);
            scale_kern_list[i_list] = complex<double>(1, 0.0);
        }
        else{
            sigma = (3.0 / 16.0) * scales[scale_idx];
            two_sigma_square = 2.0 * sigma * sigma;

            for (int x = 0; x < length; x++) {
                for (int y = 0; y < length; y++) {
                    double distance = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y);

                    const unsigned int i_list = INDEX_3D(num_scales,length,length,scale_idx,x,y);
                    scale_kern_list[i_list] = complex<double>(exp((-distance / two_sigma_square) / (M_PI * two_sigma_square)), 0.0);
                }
            }

        }
    }
}


inline void create_scaled_residual(
        const int64_t scale_dim,
        const int64_t dirty_img_dim,
        complex<double>* scale_kernels,
        complex<double>* curr_kern_ptr,
        sdp_Mem* curr_kern_mem,
        sdp_Mem* residulal,
        sdp_Mem* result,
        complex<double>* result_ptr,
        complex<double>* scaled_residual,
        sdp_Error* status
){
        // convole scale kernels with the residulas to create scaled residual images for each scale in scale_list
        // read each kernel in to a new variable so it can be used       
        for(int s = 0; s < scale_dim; s++){
            SDP_LOG_DEBUG("Loading Kernel %d", s);
            for(int x = 0; x < dirty_img_dim; x++){
                for(int y = 0; y < dirty_img_dim; y++){

                    const unsigned int i_scale_kernel = INDEX_3D(scale_dim,dirty_img_dim,dirty_img_dim,s,x,y);
                    const unsigned int i_curr_kernel = INDEX_2D(dirty_img_dim,dirty_img_dim,x,y);

                    curr_kern_ptr[i_curr_kernel] = scale_kernels[i_scale_kernel];
                }
            }
            
            SDP_LOG_DEBUG("Scaling dirty image %d", s);
            sdp_fft_convolution(residulal, curr_kern_mem, result, status);
        
            // read result in to list of scaled residuals
            for(int x = 0; x < dirty_img_dim; x++){
                for(int y = 0; y < dirty_img_dim; y++){

                    const unsigned int i_scale_residual = INDEX_3D(scale_dim,dirty_img_dim,dirty_img_dim,s,x,y);
                    const unsigned int i_curr_result = INDEX_2D(dirty_img_dim,dirty_img_dim,x,y);

                    scaled_residual[i_scale_residual] = result_ptr[i_curr_result];
                }
            }
        }
}


static void ms_clean_ws_clean(
        const double* dirty_img,
        const double* psf,
        const double* cbeam_details,
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

        int64_t dirty_img_shape[] = {dirty_img_dim, dirty_img_dim};
        int64_t psf_shape[] = {psf_dim, psf_dim};
        int64_t scaled_residuals_shape[] = {scale_dim, dirty_img_dim, dirty_img_dim};
        int64_t scale_bias_shape[] = {scale_dim};
        int64_t scaled_psf_shape[] = {scale_dim, psf_dim, psf_dim};
        int64_t scale_kern_list_shape[] = {scale_dim, dirty_img_dim, dirty_img_dim};
        int64_t scale_list_shape[] = {scale_dim};

        // Create intermediate data arrays
        sdp_Mem* psf_complex_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, psf_shape, status);
        complex<double>* psf_complex_ptr = (complex<double>*)sdp_mem_data(psf_complex_mem);
        sdp_mem_clear_contents(psf_complex_mem, status);
        sdp_Mem* cbeam_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, dirty_img_shape, status);
        complex<double>* cbeam_ptr = (complex<double>*)sdp_mem_data(cbeam_mem);
        sdp_mem_clear_contents(cbeam_mem, status);
        sdp_Mem* scale_bias_mem = sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 1, scale_bias_shape, status);
        double* scale_bias_ptr = (double*)sdp_mem_data(scale_bias_mem);
        sdp_mem_clear_contents(scale_bias_mem, status);
        sdp_Mem* scale_kern_list_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 3, scale_kern_list_shape, status);
        complex<double>* scale_kern_list_ptr = (complex<double>*)sdp_mem_data(scale_kern_list_mem);
        sdp_mem_clear_contents(scale_kern_list_mem, status);
        sdp_Mem* cur_scaled_psf_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, psf_shape, status);
        complex<double>* cur_scaled_psf_ptr = (complex<double>*)sdp_mem_data(cur_scaled_psf_mem);
        sdp_mem_clear_contents(cur_scaled_psf_mem, status);
        sdp_Mem* scaled_psf_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 3, scaled_psf_shape, status);
        complex<double>* scaled_psf_ptr = (complex<double>*)sdp_mem_data(scaled_psf_mem);
        sdp_mem_clear_contents(scaled_psf_mem, status);
        sdp_Mem* curr_kern_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, dirty_img_shape, status);
        complex<double>* curr_kern_ptr = (complex<double>*)sdp_mem_data(curr_kern_mem);
        sdp_mem_clear_contents(curr_kern_mem, status);
        sdp_Mem* result_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, dirty_img_shape, status);
        complex<double>* result_ptr = (complex<double>*)sdp_mem_data(result_mem);
        sdp_mem_clear_contents(result_mem, status);
        sdp_Mem* residual_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, dirty_img_shape, status);
        complex<double>* residual_ptr = (complex<double>*)sdp_mem_data(residual_mem);
        sdp_mem_clear_contents(residual_mem, status);
        sdp_Mem* scaled_residuals_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 3, scaled_residuals_shape, status);
        complex<double>* scaled_residuals_ptr = (complex<double>*)sdp_mem_data(scaled_residuals_mem);
        sdp_mem_clear_contents(scaled_residuals_mem, status);
        sdp_Mem* peak_per_scale_mem = sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 1, scale_list_shape, status);
        double* peak_per_scale_ptr = (double*)sdp_mem_data(peak_per_scale_mem);
        sdp_mem_clear_contents(peak_per_scale_mem, status);
        sdp_Mem* model_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, dirty_img_shape, status);
        complex<double>* model_ptr = (complex<double>*)sdp_mem_data(model_mem);
        sdp_mem_clear_contents(model_mem, status);
        sdp_Mem* clean_comp_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, dirty_img_shape, status);
        complex<double>* clean_comp_ptr = (complex<double>*)sdp_mem_data(clean_comp_mem);
        sdp_mem_clear_contents(clean_comp_mem, status);
        sdp_Mem* psf_result_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, psf_shape, status);
        complex<double>* psf_result_ptr = (complex<double>*)sdp_mem_data(psf_result_mem);
        sdp_mem_clear_contents(psf_result_mem, status);
        // sdp_Mem* residual_real_mem = sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, dirty_img_shape, status);
        // double* residual_real_ptr = (double*)sdp_mem_data(residual_real_mem);
        // sdp_mem_clear_contents(residual_real_mem, status);

        // Convolution code only works with complex input, so make residual and psf complex
        create_copy_complex(dirty_img, dirty_img_size, residual_ptr);
        create_copy_complex(psf, psf_size, psf_complex_ptr);

        // create CLEAN Beam
        SDP_LOG_DEBUG("Creating CLEAN beam");
        create_cbeam(cbeam_details, dirty_img_dim, cbeam_ptr);

        // create list of scale bias
        SDP_LOG_DEBUG("Creating Scale Biases");
        create_scale_bias(scale_list,scale_dim,scale_bias_ptr);

        // create scale kernels
        SDP_LOG_DEBUG("Creating Scale Kernels");
        create_scale_kern(scale_kern_list_ptr, scale_list, dirty_img_dim);

        // create scaled PSFs
        // read each kernel in to a new variable to pass to fft_convolve
        // create new variables for results

        for(int s = 0; s < scale_dim; s++){
            SDP_LOG_DEBUG("Loading Kernel %d", s);
            // copy first kernel for current operation
            for(int x = 0; x < dirty_img_dim; x++){
                for(int y = 0; y < dirty_img_dim; y++){

                    const unsigned int i_scale_kernel = INDEX_3D(scale_dim,dirty_img_dim,dirty_img_dim,s,x,y);
                    const unsigned int i_curr_kernel = INDEX_2D(dirty_img_dim,dirty_img_dim,x,y);

                    curr_kern_ptr[i_curr_kernel] = scale_kern_list_ptr[i_scale_kernel];
                    
                }
            }

            // convolve psf with scale kernel twice
            SDP_LOG_DEBUG("Scaling PSF %d", s);
            sdp_fft_convolution(psf_complex_mem, curr_kern_mem, psf_result_mem, status);
            sdp_fft_convolution(psf_result_mem, curr_kern_mem, cur_scaled_psf_mem, status);

            // read result in to list of scaled psfs
            for(int x = 0; x < psf_dim; x++){
                for(int y = 0; y < psf_dim; y++){

                    const unsigned int i_scale_psf = INDEX_3D(scale_dim,psf_dim,psf_dim,s,x,y);
                    const unsigned int i_curr_result = INDEX_2D(psf_dim,psf_dim,x,y);

                    scaled_psf_ptr[i_scale_psf] = cur_scaled_psf_ptr[i_curr_result];
                }
            }
        }

        // // set up some loop variables
        int cur_cycle = 0;
        bool stop = 0;
        double max_scaled_biased = 0;
        int max_scale = 0;
        double stop_val = 0;

        // CLEAN loop executes while the stop conditions (threashold and cycle limit) are not met
        while (cur_cycle < cycle_limit && stop == false) {

            // create scaled dirty images / residulas
            create_scaled_residual(
                scale_dim, 
                dirty_img_dim, 
                scale_kern_list_ptr, 
                curr_kern_ptr, 
                curr_kern_mem, 
                residual_mem, 
                result_mem, 
                result_ptr, 
                scaled_residuals_ptr, 
                status);

            SDP_LOG_DEBUG("Current Cycle %d of %d", cur_cycle, cycle_limit);

            sdp_mem_clear_contents(peak_per_scale_mem, status);
            sdp_mem_clear_contents(clean_comp_mem, status);

            // find the peak at each scale
            for (int i = 0; i < scale_dim; i++) {
                for (int x = 0; x < dirty_img_dim; x++){
                    for (int y = 0; y < dirty_img_dim; y++){
                
                        const unsigned int i_list = INDEX_3D(scale_dim,dirty_img_dim,dirty_img_dim,i,x,y);
                        if (std::real(scaled_residuals_ptr[i_list]) > peak_per_scale_ptr[i]) {
                            peak_per_scale_ptr[i] = std::real(scaled_residuals_ptr[i_list]);
                        }
                    }
                }
            }

            // multiply each max by its corresponding scale bias and find the highest biased max
            max_scale = 0;
            max_scaled_biased = 0;
            for(int s = 0; s < scale_dim; s++){
                peak_per_scale_ptr[s] *= scale_bias_ptr[s];

                if(peak_per_scale_ptr[s] > max_scaled_biased){
                    max_scaled_biased = peak_per_scale_ptr[s];
                    max_scale = s;
                }
            }

            // target value to CLEAN the highest peak down to
            stop_val = (max_scaled_biased/scale_bias_ptr[max_scale]) * ms_gain;

            // SDP_LOG_DEBUG("Maximum %f", scaled_residuals_ptr[index_per_scale_ptr[max_scale]]);
            // SDP_LOG_DEBUG("Stop value %f", stop_val);
            // SDP_LOG_DEBUG("Threshold %f", threshold);
            // SDP_LOG_DEBUG("biased peak %f", max_scaled_biased);
            // SDP_LOG_DEBUG("Selected scale %d", max_scale);


            // check the value to be CLEANed is not below the threashold
            if(stop_val < threshold){
                stop = true;
                break;
            }

            sub_minor_loop(
                scaled_residuals_ptr, 
                scaled_psf_ptr, 
                max_scale, 
                loop_gain, 
                stop_val, 
                sub_minor_cycle_limit, 
                dirty_img_dim, 
                psf_dim, 
                clean_comp_ptr);

            // read kernel for selected scale
            for(int x = 0; x < dirty_img_dim; x++){
                for(int y = 0; y < dirty_img_dim; y++){

                    const unsigned int i_scale_kernel = INDEX_3D(scale_dim,dirty_img_dim,dirty_img_dim,max_scale,x,y);
                    const unsigned int i_curr_kernel = INDEX_2D(dirty_img_dim,dirty_img_dim,x,y);

                    curr_kern_ptr[i_curr_kernel] = scale_kern_list_ptr[i_scale_kernel];
                    
                }
            }

            // convolve returned clean components with current kernel and add to model
            sdp_fft_convolution(clean_comp_mem, curr_kern_mem, result_mem, status);

            for (int i = 0; i < dirty_img_size; i++){
                model_ptr[i] += result_ptr[i];
            }

            // convolve the convolved clean components by the psf and subtract from residual
            sdp_fft_convolution(result_mem, psf_complex_mem, psf_result_mem, status);

            for (int i = 0; i < dirty_img_size; i++){
                residual_ptr[i] -= psf_result_ptr[i];
            }

            cur_cycle += 1;
        }

        // convolve the model with the clean beam and add residuals
        sdp_fft_convolution(model_mem, cbeam_mem, result_mem, status);

        create_copy_real(result_ptr, dirty_img_size, skymodel);
        // create_copy_real(residual_ptr, dirty_img_size, residual_real_ptr);

        // for (int i = 0; i < dirty_img_size; i++){
        //         skymodel[i] += residual_real_ptr[i];
        //     }

        // free memory
        sdp_mem_ref_dec(psf_complex_mem);
        sdp_mem_ref_dec(cbeam_mem);
        sdp_mem_ref_dec(scale_bias_mem);
        sdp_mem_ref_dec(scale_kern_list_mem);
        sdp_mem_ref_dec(cur_scaled_psf_mem);
        sdp_mem_ref_dec(scaled_psf_mem);
        sdp_mem_ref_dec(curr_kern_mem);
        sdp_mem_ref_dec(result_mem);
        sdp_mem_ref_dec(residual_mem);
        sdp_mem_ref_dec(scaled_residuals_mem);
        sdp_mem_ref_dec(peak_per_scale_mem);
        sdp_mem_ref_dec(model_mem);
        sdp_mem_ref_dec(clean_comp_mem);
        sdp_mem_ref_dec(psf_result_mem);
        // sdp_mem_ref_dec(residual_real_mem);
}


void sdp_ms_clean_ws_clean(
        const sdp_Mem* dirty_img,
        const sdp_Mem* psf,
        const sdp_Mem* cbeam_details,
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

    if (sdp_mem_location(dirty_img) != sdp_mem_location(skymodel)){
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Memory location mismatch");
        return;
    }

    if (sdp_mem_type(dirty_img) != sdp_mem_type(psf)){
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("The Dirty image and PSF must be of the same data type");
        return;
    }

    if (sdp_mem_type(dirty_img) != sdp_mem_type(skymodel)){
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("The input and output must be of the same data type");
        return;
    }


    if (sdp_mem_location(dirty_img) == SDP_MEM_CPU)
    {
        ms_clean_ws_clean(
            (const double*)sdp_mem_data_const(dirty_img),
            (const double*)sdp_mem_data_const(psf),
            (const double*)sdp_mem_data_const(cbeam_details),
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