/* See the LICENSE file at the top-level directory of this distribution. 

MSCLEAN cornwell algorithm taken from:
T. J. Cornwell, "Multiscale CLEAN Deconvolution of Radio Synthesis Images," 
in IEEE Journal of Selected Topics in Signal Processing, vol. 2, no. 5,
pp. 793-801, Oct. 2008, doi: 10.1109/JSTSP.2008.2006388.
https://ieeexplore.ieee.org/document/4703304

and

RASCIL implementation of the algorithm.

*/

#include "ska-sdp-func/clean/sdp_ms_clean_cornwell.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/numeric_functions/sdp_fft_convolution.h"

#include <math.h>
#include <complex>

using std::complex;

#define INDEX_2D(N2, N1, I2, I1)            (N1 * I2 + I1)
#define INDEX_3D(N3, N2, N1, I3, I2, I1)    (N1 * (N2 * I3 + I2) + I1)
#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)

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

inline void create_scale_kern(
        complex<double>* scale_kern_list,
        const double* scales,
        int64_t length
) {

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
                    scale_kern_list[i_list] = complex<double>(exp(-distance / two_sigma_square) / (M_PI * two_sigma_square), 0.0);
                }
            }

        }
    }
}

// static void ms_clean_cornwell(
//         const double* dirty_img,
//         const double* psf,
//         const double* cbeam_details,
//         const double* scale_list,
//         const double loop_gain,
//         const double threshold,
//         const double cycle_limit,
//         const int64_t dirty_img_dim,
//         const int64_t psf_dim,
//         const int64_t scale_dim,
//         double* skymodel,
//         sdp_Error* status
// ){

static void ms_clean_cornwell(
        const sdp_Mem* dirty_img,
        const sdp_Mem* psf,
        const sdp_Mem* cbeam_details,
        const sdp_Mem* scale_list,
        const double loop_gain,
        const double threshold,
        const double cycle_limit,
        const int64_t dirty_img_dim,
        const int64_t psf_dim,
        const int64_t scale_dim,
        sdp_Mem* skymodel,
        sdp_Error* status
){
        // pointers for input data
        const double* dirty_img_ptr = (const double*)sdp_mem_data_const(dirty_img);
        const double* psf_ptr = (const double*)sdp_mem_data_const(psf);
        const double* cbeam_details_ptr = (const double*)sdp_mem_data_const(cbeam_details);
        const double* scale_list_ptr = (const double*)sdp_mem_data_const(scale_list);
        double* skymodel_ptr = (double*)sdp_mem_data(skymodel);

        // calculate useful shapes and sizes
        int64_t dirty_img_size = dirty_img_dim * dirty_img_dim;
        int64_t psf_size = psf_dim * psf_dim;

        int64_t dirty_img_shape[] = {dirty_img_dim, dirty_img_dim};
        int64_t psf_shape[] = {psf_dim, psf_dim};
        int64_t scaled_residuals_shape[] = {scale_dim, dirty_img_dim, dirty_img_dim};
        int64_t coupling_matrix_shape[] = {scale_dim, scale_dim};
        int64_t scaled_psf_shape[] = {scale_dim, scale_dim, psf_dim, psf_dim};
        int64_t scale_kern_list_shape[] = {scale_dim, dirty_img_dim, dirty_img_dim};
        int64_t scale_list_shape[] = {scale_dim};


        // Create intermediate data arrays
        sdp_Mem* psf_complex_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, psf_shape, status);
        complex<double>* psf_complex_ptr = (complex<double>*)sdp_mem_data(psf_complex_mem);
        sdp_mem_clear_contents(psf_complex_mem, status);
        sdp_Mem* cbeam_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, psf_shape, status);
        complex<double>* cbeam_ptr = (complex<double>*)sdp_mem_data(cbeam_mem);
        sdp_mem_clear_contents(cbeam_mem, status);
        sdp_Mem* clean_comp_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, dirty_img_shape, status);
        complex<double>* clean_comp_ptr = (complex<double>*)sdp_mem_data(clean_comp_mem);
        sdp_mem_clear_contents(clean_comp_mem, status);
        sdp_Mem* residual_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, dirty_img_shape, status);
        complex<double>* residual_ptr = (complex<double>*)sdp_mem_data(residual_mem);
        sdp_mem_clear_contents(residual_mem, status);
        sdp_Mem* cur_scaled_residual_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, dirty_img_shape, status);
        complex<double>* cur_scaled_residual_ptr = (complex<double>*)sdp_mem_data(cur_scaled_residual_mem);
        sdp_mem_clear_contents(cur_scaled_residual_mem, status);
        sdp_Mem* scaled_residuals_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 3, scaled_residuals_shape, status);
        complex<double>* scaled_residuals_ptr = (complex<double>*)sdp_mem_data(scaled_residuals_mem);
        sdp_mem_clear_contents(scaled_residuals_mem, status);
        sdp_Mem* scaled_psf_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 4, scaled_psf_shape, status);
        complex<double>* scaled_psf_ptr = (complex<double>*)sdp_mem_data(scaled_psf_mem);
        sdp_mem_clear_contents(scaled_psf_mem, status);
        sdp_Mem* cur_scaled_psf_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, psf_shape, status);
        complex<double>* cur_scaled_psf_ptr = (complex<double>*)sdp_mem_data(cur_scaled_psf_mem);
        sdp_mem_clear_contents(cur_scaled_psf_mem, status);
        sdp_Mem* cur_scaled_psf_pre_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, psf_shape, status);
        complex<double>* cur_scaled_psf_pre_ptr = (complex<double>*)sdp_mem_data(cur_scaled_psf_pre_mem);
        sdp_mem_clear_contents(cur_scaled_psf_pre_mem, status);
        // sdp_Mem* coupling_matrix_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, scaled_residuals_shape, status);
        // complex<double>* coupling_matrix_ptr = (complex<double>*)sdp_mem_data(coupling_matrix_mem);
        // sdp_mem_clear_contents(coupling_matrix_mem, status);
        sdp_Mem* scale_kern_list_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 3, scale_kern_list_shape, status);
        complex<double>* scale_kern_list_ptr = (complex<double>*)sdp_mem_data(scale_kern_list_mem);
        sdp_mem_clear_contents(scale_kern_list_mem, status);
        sdp_Mem* cur_1_scale_kern_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, dirty_img_shape, status);
        complex<double>* cur_1_scale_kern_ptr = (complex<double>*)sdp_mem_data(cur_1_scale_kern_mem);
        sdp_mem_clear_contents(cur_1_scale_kern_mem, status);
        sdp_Mem* cur_2_scale_kern_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, dirty_img_shape, status);
        complex<double>* cur_2_scale_kern_ptr = (complex<double>*)sdp_mem_data(cur_2_scale_kern_mem);
        sdp_mem_clear_contents(cur_2_scale_kern_mem, status);
        sdp_Mem* coupling_matrix_mem = sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, coupling_matrix_shape, status);
        double* coupling_matrix_ptr = (double*)sdp_mem_data(coupling_matrix_mem);
        sdp_mem_clear_contents(coupling_matrix_mem, status);
        sdp_Mem* peak_per_scale_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 1, scale_list_shape, status);
        complex<double>* peak_per_scale_ptr = (complex<double>*)sdp_mem_data(peak_per_scale_mem);
        sdp_mem_clear_contents(peak_per_scale_mem, status);
        sdp_Mem* index_per_scale_mem = sdp_mem_create(SDP_MEM_INT, SDP_MEM_CPU, 1, scale_list_shape, status);
        int* index_per_scale_ptr = (int*)sdp_mem_data(index_per_scale_mem);
        sdp_mem_clear_contents(index_per_scale_mem, status);
        sdp_Mem* x_per_scale_mem = sdp_mem_create(SDP_MEM_INT, SDP_MEM_CPU, 1, scale_list_shape, status);
        int* x_per_scale_ptr = (int*)sdp_mem_data(x_per_scale_mem);
        sdp_mem_clear_contents(x_per_scale_mem, status);
        sdp_Mem* y_per_scale_mem = sdp_mem_create(SDP_MEM_INT, SDP_MEM_CPU, 1, scale_list_shape, status);
        int* y_per_scale_ptr = (int*)sdp_mem_data(y_per_scale_mem);
        sdp_mem_clear_contents(y_per_scale_mem, status);

        // Convolution code only works with complex input, so make residual and psf complex
        create_copy_complex(dirty_img_ptr, dirty_img_size, residual_ptr);
        create_copy_complex(psf_ptr, psf_size, psf_complex_ptr);
        
        // create CLEAN beam
        SDP_LOG_DEBUG("Creating CLEAN beam");
        create_cbeam(cbeam_details_ptr, psf_dim, cbeam_ptr);

        // create scale kernels
        SDP_LOG_DEBUG("Creating Scale Kernels");
        create_scale_kern(scale_kern_list_ptr, scale_list_ptr, dirty_img_dim);

        // scale psf and dirty image with scale kernel
        for (int s = 0; s < scale_dim; s++) {
            SDP_LOG_DEBUG("Loading Kernel %d", s);
            // copy first kernel for current operation
            for (int x = 0; x < dirty_img_dim; x++){
                for (int y = 0; y < dirty_img_dim; y++){

                    const unsigned int i_list_s = INDEX_3D(scale_dim,dirty_img_dim,dirty_img_dim,s,x,y);
                    const unsigned int i_cur = INDEX_2D(dirty_img_dim,dirty_img_dim,x,y);
                    cur_1_scale_kern_ptr[i_cur] = scale_kern_list_ptr[i_list_s];
                    }
                }

            for (int p = 0; p < scale_dim; p++) {

                // copy second kernel for current operation
                SDP_LOG_DEBUG("Loading Kernel %d", p);
                for (int x = 0; x < dirty_img_dim; x++){
                    for (int y = 0; y < dirty_img_dim; y++){

                    const unsigned int i_list_p = INDEX_3D(scale_dim,dirty_img_dim,dirty_img_dim,p,x,y);
                    const unsigned int i_cur = INDEX_2D(dirty_img_dim,dirty_img_dim,x,y);
                    cur_2_scale_kern_ptr[i_cur] = scale_kern_list_ptr[i_list_p];

                    }
                }

                // scale psf twice
                SDP_LOG_DEBUG("Scaling PSF %d, %d", s,p);
                sdp_fft_convolution(psf_complex_mem, cur_1_scale_kern_mem, cur_scaled_psf_pre_mem, status);
                sdp_fft_convolution(cur_scaled_psf_pre_mem, cur_2_scale_kern_mem, cur_scaled_psf_mem, status);

                // load scaled psf to scaled psf list
                for (int x = 0; x < psf_dim; x++){
                    for (int y = 0; y < psf_dim; y++){
                    
                        const unsigned int i_scaled_psf_list = INDEX_4D(scale_dim,scale_dim,psf_dim,psf_dim,s,p,x,y);
                        const unsigned int i_cur = INDEX_2D(psf_dim,psf_dim,x,y);
                        scaled_psf_ptr[i_scaled_psf_list] = cur_scaled_psf_ptr[i_cur];
                    }
                }

            }

            // scale dirty image once
            SDP_LOG_DEBUG("Scaling dirty image %d", s);
            sdp_fft_convolution(residual_mem, cur_1_scale_kern_mem, cur_scaled_residual_mem, status);

            // load scaled residual to scaled residual list
            for (int x = 0; x < dirty_img_dim; x++){
                for (int y = 0; y < dirty_img_dim; y++){
                
                    const unsigned int i_list = INDEX_3D(scale_dim,dirty_img_dim,dirty_img_dim,s,x,y);
                    const unsigned int i_cur = INDEX_2D(dirty_img_dim,dirty_img_dim,x,y);
                    scaled_residuals_ptr[i_list] = cur_scaled_residual_ptr[i_cur];
                    // skymodel_ptr[i_cur] = std::real(scaled_residuals_ptr[i_cur]);
                }
            }

        
        }

        // evaluate the coupling matrix
        SDP_LOG_DEBUG("Evaluate coupling matrix");
        double max_scaled_psf = 0;
        for (int s = 0; s < scale_dim; s++) {
            for (int p = 0; p < scale_dim; p++) {
                for (int x = 0; x < psf_dim; x++){
                    for (int y = 0; y < psf_dim; y++){

                        const unsigned int i_scaled_psf_list = INDEX_4D(scale_dim,scale_dim,psf_dim,psf_dim,s,p,x,y);
                        if (std::real(scaled_psf_ptr[i_scaled_psf_list]) > max_scaled_psf) {
                           max_scaled_psf = std::real(scaled_psf_ptr[i_scaled_psf_list]);
                        }
                    }
                }
                const unsigned int i_cur = INDEX_2D(scale_dim,scale_dim,s,p);
                coupling_matrix_ptr[i_cur] = max_scaled_psf;
                max_scaled_psf = 0;
            }
        }

        // // print to skymodel
        // int s = 4;
        // int p = 4;
        // for (int x = 0; x < dirty_img_dim; x++){
        //     for (int y = 0; y < dirty_img_dim; y++){
                    
        //         const unsigned int i_list = INDEX_3D(scale_dim,dirty_img_dim,dirty_img_dim,s,x,y);
        //         const unsigned int i_cur = INDEX_2D(dirty_img_dim,dirty_img_dim,x,y);
        //         const unsigned int i_scaled_psf_list = INDEX_4D(scale_dim,scale_dim,psf_dim,psf_dim,s,p,x,y);
                
        //         skymodel_ptr[i_cur] = std::real(scaled_residuals_ptr[i_list]);
                
        //         // const unsigned int i_cur = INDEX_2D(dirty_img_dim,dirty_img_dim,x,y);
        //         // const unsigned int i_cur_coup = INDEX_2D(scale_dim,scale_dim,x,y);


        //         // skymodel_ptr[i_cur] = std::real(scaled_psf_ptr[i_scaled_psf_list]);
        //     }
        // }


        // // print to skymodel
        // int s = 4;
        // int p = 4;
        // for (int x = 0; x < scale_dim; x++){
        //     for (int y = 0; y < scale_dim; y++){
                    
        //         const unsigned int i_cur = INDEX_2D(dirty_img_dim,dirty_img_dim,x,y);
        //         const unsigned int i_cur_coup = INDEX_2D(scale_dim,scale_dim,x,y);

        //         skymodel_ptr[i_cur] = std::real(coupling_matrix_ptr[i_cur_coup]);
        //     }
        // }



        // set up some loop variables
        int cur_cycle = 0;
        bool stop = 0;
        double max_scaled_biased = 0;
        int max_scale = 0;

        // CLEAN loop executes while the stop conditions (threashold and cycle limit) are not met
        while (cur_cycle < cycle_limit && stop == 0) {
            
            SDP_LOG_DEBUG("Current Cycle %d of %f", cur_cycle, cycle_limit);

            sdp_mem_clear_contents(peak_per_scale_mem, status);

            // find the peak at each scale
            for (int i = 0; i < scale_dim; i++) {
                for (int x = 0; x < dirty_img_dim; x++){
                    for (int y = 0; y < dirty_img_dim; y++){
                
                        const unsigned int i_list = INDEX_3D(scale_dim,dirty_img_dim,dirty_img_dim,i,x,y);
                        if (std::real(scaled_residuals_ptr[i_list]) > std::real(peak_per_scale_ptr[i])) {
                            peak_per_scale_ptr[i] = scaled_residuals_ptr[i_list];
                            index_per_scale_ptr[i] = i_list;
                            x_per_scale_ptr[i] = x;
                            y_per_scale_ptr[i] = y;
                        }
                    }
                }
            }

            SDP_LOG_DEBUG("Current max pre bias %f" , std::real(peak_per_scale_ptr[1]));
            
            // bias with coupling matrix to find overall peak for all scales
            for (int i = 0; i < scale_dim; i++) {
                const unsigned int i_cur = INDEX_2D(scale_dim,scale_dim,i,i);
                peak_per_scale_ptr[i] /= coupling_matrix_ptr[i_cur];
            }

            // find overall peak
            max_scaled_biased = 0;
            for (int i = 0; i < scale_dim; i++) {
                if (std::real(peak_per_scale_ptr[i]) > max_scaled_biased){
                    max_scaled_biased = std::real(peak_per_scale_ptr[i]);
                    max_scale = i;
                }
            }
            
            SDP_LOG_DEBUG("Current max %f" , max_scaled_biased);
            SDP_LOG_DEBUG("Current max location %d, %d" , x_per_scale_ptr[max_scale], y_per_scale_ptr[max_scale]);
            SDP_LOG_DEBUG("Current max scale %d" , max_scale);

            
            // check maximum value against threshold
            if (max_scaled_biased < threshold) {
                stop = 1;
                break;
            }

            // add fraction of maximum to clean components list
            clean_comp_ptr[index_per_scale_ptr[max_scale]] += complex<double>(loop_gain * max_scaled_biased, 0);

            // cross subtract psf from other scales
            // identify psf window to subtract from residual
            int64_t psf_x_start, psf_x_end, psf_y_start, psf_y_end;

            psf_x_start = dirty_img_dim - x_per_scale_ptr[max_scale];
            psf_x_end = psf_x_start + dirty_img_dim;
            psf_y_start = dirty_img_dim - y_per_scale_ptr[max_scale];
            psf_y_end = psf_y_start + dirty_img_dim;

            for (int s = 0; s < scale_dim; s++){
                for (int x = psf_x_start, i = 0; x < psf_x_end; x++, i++){
                    for (int y = psf_y_start, j = 0; y < psf_y_end; y++, j++){
                        const unsigned int i_psf = INDEX_4D(scale_dim,scale_dim,psf_dim,psf_dim,s,max_scale,x,y);
                        const unsigned int i_res = INDEX_3D(scale_dim,dirty_img_dim,dirty_img_dim,s,i,j);

                        scaled_residuals_ptr[i_res] -= (complex<double>((loop_gain * max_scaled_biased),0) * scaled_psf_ptr[i_psf]);

                        // const unsigned int i_cur = INDEX_2D(dirty_img_dim,dirty_img_dim,i,j);
                        // skymodel_ptr[i_cur] = std::real(scaled_residuals_ptr[i_res]);

                    }
                }
            }

            cur_cycle += 1;
        }

        // convolve clean components with clean beam
        sdp_Mem* convolution_result_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, dirty_img_shape, status);
        complex<double>* convolution_result_ptr = (complex<double>*)sdp_mem_data(convolution_result_mem);

        sdp_fft_convolution(clean_comp_mem, cbeam_mem, convolution_result_mem, status);

        // complex result to real for the output
        create_copy_real(convolution_result_ptr, dirty_img_size, skymodel_ptr);

        // release memory
        sdp_mem_ref_dec(psf_complex_mem);
        sdp_mem_ref_dec(cbeam_mem);
        sdp_mem_ref_dec(clean_comp_mem);
        sdp_mem_ref_dec(residual_mem);
        sdp_mem_ref_dec(scaled_residuals_mem);
        sdp_mem_ref_dec(scaled_psf_mem);
        sdp_mem_ref_dec(coupling_matrix_mem);
        sdp_mem_ref_dec(scale_kern_list_mem);
        sdp_mem_ref_dec(cur_scaled_residual_mem);
        sdp_mem_ref_dec(cur_scaled_psf_mem);
        sdp_mem_ref_dec(cur_scaled_psf_pre_mem);
        sdp_mem_ref_dec(cur_1_scale_kern_mem);
        sdp_mem_ref_dec(cur_2_scale_kern_mem);
        sdp_mem_ref_dec(peak_per_scale_mem);
        sdp_mem_ref_dec(x_per_scale_mem);
        sdp_mem_ref_dec(y_per_scale_mem);
        sdp_mem_ref_dec(convolution_result_mem);
}










void sdp_ms_clean_cornwell(
        const sdp_Mem* dirty_img,
        const sdp_Mem* psf,
        const sdp_Mem* cbeam_details,
        const sdp_Mem* scale_list,
        const double loop_gain,
        const double threshold,
        const double cycle_limit,
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
        // ms_clean_cornwell(
        //     (const double*)sdp_mem_data_const(dirty_img),
        //     (const double*)sdp_mem_data_const(psf),
        //     (const double*)sdp_mem_data_const(cbeam_details),
        //     (const double*)sdp_mem_data_const(scale_list),
        //     loop_gain,
        //     threshold,
        //     cycle_limit,
        //     dirty_img_dim,
        //     psf_dim,
        //     scale_dim,
        //     (double*)sdp_mem_data(skymodel),
        //     status
        // );
        ms_clean_cornwell(
            dirty_img,
            psf,
            cbeam_details,
            scale_list,
            loop_gain,
            threshold,
            cycle_limit,
            dirty_img_dim,
            psf_dim,
            scale_dim,
            skymodel,
            status
        );
    }
    
}