/* See the LICENSE file at the top-level directory of this distribution. */

#include "ska-sdp-func/clean/sdp_ms_clean.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/fourier_transforms/sdp_fft.h"

#include <string.h>
#include <math.h>

#define INDEX_2D(N2, N1, I2, I1)    (N1 * I2 + I1)

static void create_cbeam(
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

    for(double x = 0; x <= psf_dim; x += 1) {
        for(double y = 0; y <= psf_dim; y += 1) {

            const unsigned int i_cbeam = INDEX_2D(psf_dim,psf_dim,x,y);
            cbeam[i_cbeam] = A * exp(-(a * pow(x - x0,2) + 2 * b * (x - x0) * (y - y0) + c * pow(y - y0,2)));

            }
        }
}

inline void create_scale_kernels(
        const double* scale_list,
        const int64_t scale_dim,
        double* scale_kernels
){
    // create a scale kernel for each scale specified in scale_list


}


inline void create_scaled_psf(
        const int64_t scale_dim,
        double* scale_kernels,
        const double* psf,
        double* scale_psf
){
    // convole scale kernels with the PSF to create scaled PSFs for each scale in scale_list

}


inline void create_scaled_dirty_imgs(
        const int64_t scale_dim,
        double* scale_kernels,
        const double* dirty_img,
        double* scale_dirty_img
){
    // convole scale kernels with the dirt image to create scaled dirty imagess for each scale in scale_list

}


static void ms_clean(
        const double* dirty_img,
        const double* psf,
        const double* scale_list,
        const double loop_gain,
        const double threshold,
        const double cycle_limit,
        const int64_t dirty_img_dim,
        const int64_t psf_dim,
        const int64_t scale_dim,
        double* skymodel,
        sdp_Error* status
){
        int64_t dirty_img_size = dirty_img_dim * dirty_img_dim;
        int64_t psf_size = psf_dim * psf_dim;

        // Create intermediate data arrays
        double* maximum = (double*) calloc(cycle_limit * 3, sizeof(double));
        double* clean_comp = (double*) calloc(dirty_img_size, sizeof(double));
        double* residual = (double*) malloc(dirty_img_size * sizeof(double));
        memcpy(residual, dirty_img, dirty_img_size * sizeof(double));
        double* multiply = (double*) malloc(dirty_img_size * sizeof(double));
        double* result_of_convolution = (double*) malloc(dirty_img_size * sizeof(double));
        double* cbeam = (double*) calloc(psf_dim, sizeof(double));
        double* scale_kernels = (double*) malloc(dirty_img_size * scale_dim * sizeof(double));
        double* scale_psf = (double*) malloc(psf_size * scale_dim * sizeof(double));
        double* scale_dirty_img = (double*) malloc(dirty_img_size * scale_dim * sizeof(double));

        // set up some loop variables
        int cur_cycle;
        bool stop;

        // create CLEAN Beam
        create_cbeam(psf, cbeam, psf_dim);

        // create scale kernels
        create_scale_kernels(scale_list, scale_dim, scale_kernels);

        // create scaled PSFs
        create_scaled_psf(scale_dim, scale_kernels, psf, scale_psf);

        // create scaled dirty images
        create_scaled_dirty_imgs(scale_dim, scale_kernels, dirty_img, scale_dirty_img);


        // free memory
        free(maximum);
        free(clean_comp);
        free(residual);
        free(multiply);
        free(result_of_convolution);
        free(cbeam);
        free(scale_kernels);
        free(scale_psf);
        free(scale_dirty_img);

}


void sdp_ms_clean(
        const sdp_Mem* dirty_img,
        const sdp_Mem* psf,
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
        ms_clean(
            (const double*)sdp_mem_data_const(dirty_img),
            (const double*)sdp_mem_data_const(psf),
            (const double*)sdp_mem_data_const(scale_list),
            loop_gain,
            threshold,
            cycle_limit,
            dirty_img_dim,
            psf_dim,
            scale_dim,
            (double*)sdp_mem_data(skymodel),
            status
        );
    }
    
}