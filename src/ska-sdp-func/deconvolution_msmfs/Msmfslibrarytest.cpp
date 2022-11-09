/*
 * Msmfslibrarytest.cpp
 * Andrew Ensor
 * C with C++ templates/CUDA program for testing steps of the MSMFS cleaning algorithm
*/

#include "Msmfslibrarytest.h"

#define MSMFS_PRECISION_SINGLE 1

/**********************************************************************
 * Main method to execute
 **********************************************************************/
int library_test()
{
    printf("Msmfs library test starting");
    #ifdef MSMFS_PRECISION_SINGLE
        printf(" using single precision\n");
        #define PRECISION float
    #else
        printf(" using double precision\n");
        #define PRECISION double
    #endif

    setLogLevel(LOG_INFO);

    // specify msmsf key configuration parameters
    const unsigned int dirty_moment_size = 8192; // one dimensional size of image, assumed square
    const unsigned int num_scales = 6; // number of scales to use in msmfs cleaning
    unsigned int num_taylor = 3; // number of taylor moments to use in msmfs cleaning
    if (num_taylor > MAX_TAYLOR_MOMENTS)
    {
        logger(LOG_WARNING,
            "Number of Taylor moments was set at %u but will be capped at %u, change MAX_TAYLOR_MOMENTS to adjust",
            num_taylor, MAX_TAYLOR_MOMENTS);
        num_taylor = MAX_TAYLOR_MOMENTS;
    }
    const unsigned int psf_moment_size = dirty_moment_size/4; // one dimensional size of psf, assumed square
    const unsigned int image_border = 0; // border around dirty moment images and psfs to clip when using convolved images or convolved psfs
    const PRECISION convolution_accuracy = (PRECISION)1.2E-3; // fraction of peak accuracy used to determine supports for convolution kernels
    const PRECISION clean_loop_gain = 0.35; // loop gain fraction of peak point to clean from the peak each minor cycle
    const unsigned int max_gaussian_sources_host = 200; // maximum number of gaussian sources to find during cleaning (so bounds number clean minor cycles)
    const PRECISION scale_bias_factor = (PRECISION)0.6; // 0.6 is typical bias multiplicative factor to favour cleaning with smaller scales
    const PRECISION clean_threshold = (PRECISION)0.001; // fractional threshold at which to stop cleaning (or non-positive to disable threshold check)

    // calculate msmfs derived parameters
    const unsigned int num_psf = 2*num_taylor - 1; // determined by the number of Taylor terms
    const unsigned int scale_moment_size = dirty_moment_size-2*image_border; // one dimensional size of scale moment residuals, assumed square
    const unsigned int psf_convolved_size = psf_moment_size-2*image_border; // one dimensional size of convolved psfs, assumed square
    logger(LOG_NOTICE,
        "Msmfs performed on %ux%u image with %u scales, %u Taylor terms, %u border pixels"
        ", and with %u PSF each of size %ux%u",
        dirty_moment_size, dirty_moment_size, num_scales, num_taylor, image_border,
        num_psf, psf_moment_size, psf_moment_size);

    // create a simple test input image
    PRECISION *dirty_moment_images_device = allocate_dirty_image<PRECISION>(dirty_moment_size, num_taylor);
    calculate_simple_dirty_image<PRECISION>(dirty_moment_images_device, dirty_moment_size, num_taylor);

    // create a simple test input psf
    PRECISION *psf_moment_images_device = allocate_psf_image<PRECISION>(psf_moment_size, num_psf);
    calculate_simple_psf_image<PRECISION>(psf_moment_images_device, psf_moment_size, num_psf);

    // calculate suitable cuda block size in1D and 2D and number of available cuda threads
    int cuda_block_size;
    dim3 cuda_block_size_2D;
    int cuda_num_threads;
    calculate_cuda_configs(&cuda_block_size, &cuda_block_size_2D, &cuda_num_threads);
    
    // set up gaussian shape scales with specified or default variances all L1-normalised
    // note that variance equal to 0 is treated as a special case delta function with peak 1
    // note suitable support will be calculated for each depending on convolution_accuracy but clipped to fit within image border
    // note if variances_host parameter is NULL then suitable default shape variances are calculated and used
    Gaussian_shape_configurations<PRECISION> shape_configs = allocate_shape_configurations<PRECISION>(NULL, num_scales, convolution_accuracy, scale_bias_factor);

    checkCudaStatus();

    // allocate device memory for the scale moment residuals that get calculated from the dirty images
    PRECISION *scale_moment_residuals_device = allocate_scale_moment_residuals<PRECISION>(scale_moment_size, num_scales, num_taylor);

    // calculate the scale moment residuals on the GPU
    calculate_scale_moment_residuals<PRECISION>
        (dirty_moment_images_device, dirty_moment_size, num_taylor,
        scale_moment_residuals_device, scale_moment_size, num_scales,
        shape_configs.variances_device, shape_configs.convolution_support_device, cuda_block_size_2D);

    checkCudaStatus();

    // ***** temporary code to display the initial scale moment residuals *****
//    display_scale_moment_residuals<PRECISION>(scale_moment_residuals_device, num_scales, num_taylor, scale_moment_size);
//    checkCudaStatus();    
    // ***** end of temporary code *****

    // allocate device memory for the inverse hessian matrices
    PRECISION *inverse_hessian_matrices_device = allocate_inverse_hessian_matrices<PRECISION>(num_scales, num_taylor);

    // calculate the inverse of each scale-dependent moment hessian matrix on the device
    calculate_inverse_hessian_matrices<PRECISION>
        (psf_moment_images_device, psf_moment_size, num_psf,
        inverse_hessian_matrices_device, num_scales, num_taylor,
        shape_configs.variances_device, shape_configs.double_convolution_support_device, true, cuda_block_size);

    checkCudaStatus();    

    // ***** temporary code to display entries calculated for inverse hessian matrices *****
    display_inverse_hessian_matrices<PRECISION>(inverse_hessian_matrices_device, num_scales, num_taylor);
    // ***** end of temporary code *****

    // create an initially empty list of sources to hold those gaussian source found in the minor cycle loops (for single major cycle)
    Gaussian_source_list<PRECISION> gaussian_source_list = allocate_gaussian_source_list<PRECISION>(max_gaussian_sources_host);

    // determine maximum number of clean minor cycles
    // note this is presumed to be not more than max_gaussian_sources_host divided by the number (1) of major cycles
    // so that gaussian_sources_device has sufficent space to hold all the sources discovered during cleaning
    const unsigned int max_clean_cycles = max_gaussian_sources_host; // upper bound on the number of minor cycle iterations

    // preallocate device data structures that will be used during cleaning minor cycles
    // but which are unlikely to be of interest afterwards
    bool larger_psf_convolved_buffer = true; 
    Cleaning_device_data_structures<PRECISION> working_data = allocate_device_data_structures<PRECISION>(num_taylor,
        scale_moment_size, larger_psf_convolved_buffer, psf_moment_size, psf_convolved_size, num_psf);

    // perform clean minor cycles
    const unsigned int min_clean_cycles = 0;

    perform_major_cycles<PRECISION>(max_clean_cycles, min_clean_cycles, clean_threshold, clean_loop_gain,
        scale_moment_residuals_device, num_scales, num_taylor, scale_moment_size,
        psf_moment_images_device, num_psf, psf_moment_size, psf_convolved_size, 
        larger_psf_convolved_buffer,
        inverse_hessian_matrices_device,
        shape_configs.variances_device, shape_configs.scale_bias_device, shape_configs.double_convolution_support_device,
        working_data.peak_point_smpsol_device, working_data.peak_point_scale_device,
        working_data.peak_point_index_device,
        working_data.smpsol_max_device, working_data.smpsol_scale_device,
        working_data.psf_convolved_images_device, working_data.horiz_convolved_device,
        working_data.is_existing_source_device,
        gaussian_source_list.gaussian_sources_device, gaussian_source_list.num_gaussian_sources_device,
        cuda_block_size, cuda_block_size_2D, cuda_num_threads 
        );

    // deallocate device data structures that were used during cleaning minor cycles
    free_device_data_structures<PRECISION>(working_data);

    // ***** temporary code to display the current scale moment residuals for one scale and moment *****
//    display_scale_moment_residuals<PRECISION>(scale_moment_residuals_device, num_scales, num_taylor, scale_moment_size);
    // ***** end of temporary code *****

    // ***** temporary code to display the gaussian sources found in the minor cycle loops *****
    display_gaussian_source_list<PRECISION>(gaussian_source_list.gaussian_sources_device, gaussian_source_list.num_gaussian_sources_device,
        max_gaussian_sources_host, dirty_moment_size, image_border, num_taylor);
    // ***** end of temporary code *****

    // clean up the source model
    free_gaussian_source_list<PRECISION>(gaussian_source_list);

    // clean up allocated memory for scale moment residuals and inverse hessian matrices
    free_inverse_hessian_matrices<PRECISION>(inverse_hessian_matrices_device);
    free_scale_moment_residuals<PRECISION>(scale_moment_residuals_device);

    // deallocate shape configurations
    free_shape_configurations<PRECISION>(shape_configs);

    // clean up simple test input image and simple test input psf
    free_psf_image<PRECISION>(psf_moment_images_device);
    free_dirty_image<PRECISION>(dirty_moment_images_device); // note this free could be once scale_moment_residuals_device created

    checkCudaStatus();

    printf("Msmfs lib test ending\n");
    return 0;
}