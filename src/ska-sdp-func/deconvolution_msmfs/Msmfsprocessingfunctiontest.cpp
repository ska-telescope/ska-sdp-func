/*
 * Msmfsprocessingfunctiontest.cpp
 * Andrew Ensor
 * C++ program for testing the SDP processing function interface steps for the MSMFS cleaning algorithm
 * Note apart from convenience calls of allocate_simple_dirty_image, allocate_simple_psf_image,
 * calculate_simple_dirty_image, calculate_simple_psf_image, free_simple_psf_image, free_simple_dirty_image
 * this code would also be compilable as C
*/

#include "Msmfsprocessingfunctiontest.h"
#include "Msmfssimpletest.h" // C++ interface used for convenience during testing

#define MSMFS_PRECISION_SINGLE 1
#define MAX_TAYLOR_MOMENTS 6 /* compile-time upper limit on the number of possible taylor moments */


/**********************************************************************
 * Main method to execute
 **********************************************************************/
int interface_test()
{
    printf("Msmfs processing function interface test starting");
    #ifdef MSMFS_PRECISION_SINGLE
        printf(" using single precision\n");
        #define PRECISION float
        #define SDP_MEM_PRECISION SDP_MEM_FLOAT
    #else
        printf(" using double precision\n");
        #define PRECISION double
        #define SDP_MEM_PRECISION SDP_MEM_DOUBLE
    #endif

    // specify msmsf key configuration parameters
    const unsigned int dirty_moment_size = 8192; // one dimensional size of image, assumed square
    const unsigned int num_scales = 6; // number of scales to use in msmfs cleaning
    unsigned int num_taylor = 3; // number of taylor moments to use in msmfs cleaning
    if (num_taylor > MAX_TAYLOR_MOMENTS)
    {
        printf("Number of Taylor moments was set at %u but will be capped at %u, change MAX_TAYLOR_MOMENTS to adjust",
            num_taylor, MAX_TAYLOR_MOMENTS);
        num_taylor = MAX_TAYLOR_MOMENTS;
    }
    const unsigned int psf_moment_size = dirty_moment_size/4; // one dimensional size of psf, assumed square
    const unsigned int image_border = 0; // border around dirty moment images and psfs to clip when using convolved images or convolved psfs
    const double convolution_accuracy = 1.2E-3; // fraction of peak accuracy used to determine supports for convolution kernels
    const double clean_loop_gain = 0.35; // loop gain fraction of peak point to clean from the peak each minor cycle
    const unsigned int max_gaussian_sources_host = 200; // maximum number of gaussian sources to find during cleaning (so bounds number clean minor cycles)
    const double scale_bias_factor = 0.6; // 0.6 is typical bias multiplicative factor to favour cleaning with smaller scales
    const double clean_threshold = 0.001; // fractional threshold at which to stop cleaning (or non-positive to disable threshold check)

    // create a simple test input image
    PRECISION *dirty_moment_images_device = allocate_simple_dirty_image<PRECISION>(dirty_moment_size, num_taylor);
    calculate_simple_dirty_image<PRECISION>(dirty_moment_images_device, num_taylor, dirty_moment_size);

    // create a simple test input psf
    PRECISION *psf_moment_images_device = allocate_simple_psf_image<PRECISION>(psf_moment_size, 2*num_taylor-1);
    calculate_simple_psf_image<PRECISION>(psf_moment_images_device, 2*num_taylor-1, psf_moment_size);

    // wrap the simple test input images and psfs as sdp_Mem
    sdp_Error *status = NULL;
    const int64_t dirty_moment_shape[] = {num_taylor, dirty_moment_size, dirty_moment_size};
    const int64_t psf_moment_shape[] = {2*num_taylor-1, psf_moment_size, psf_moment_size};
    sdp_Mem *dirty_moment_images = sdp_mem_create_wrapper(dirty_moment_images_device, SDP_MEM_PRECISION, SDP_MEM_GPU, 3, dirty_moment_shape, 0, status);
    sdp_Mem *psf_moment_images = sdp_mem_create_wrapper(psf_moment_images_device, SDP_MEM_PRECISION, SDP_MEM_GPU, 3, psf_moment_shape, 0, status);

    perform_msmfs(
        dirty_moment_images, psf_moment_images,
        dirty_moment_size, num_scales, num_taylor, psf_moment_size, image_border,
        (PRECISION)convolution_accuracy, (PRECISION)clean_loop_gain, max_gaussian_sources_host,
        (PRECISION)scale_bias_factor, (PRECISION)clean_threshold);

    // clean up simple test input image and simple test input psf
    free_simple_psf_image<PRECISION>(psf_moment_images_device);
    free_simple_dirty_image<PRECISION>(dirty_moment_images_device); // note this free could be once scale_moment_residuals_device created

    checkCudaStatus();

    printf("Msmfs processing function interface test ending\n");
    return 0;
}