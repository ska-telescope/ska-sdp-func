/*
 * Msmfsprocessingfunctioninterface.cpp
 * Andrew Ensor
 * C with C++ templates/CUDA program providing an SDP processing function interface steps for the MSMFS cleaning algorithm
*/

#include "Msmfsprocessingfunctioninterface.h"
#include "Msmfscommon.h"
#include "Msmfslogger.h"
#include "Msmfsfunctionshost.h"
#include "Msmfssimpletest.h"

/*****************************************************************************
 * Templated (non-C interface) version of the function which performs the entire msmfs deconvolution
 *****************************************************************************/
template<typename PRECISION>
void sdp_perform_msmfs
    (
    PRECISION *dirty_moment_images_device,
    PRECISION *psf_moment_images_device,
    const unsigned int dirty_moment_size,
    const unsigned int num_scales,
    const unsigned int num_taylor,
    const unsigned int psf_moment_size,
    const unsigned int image_border,
    const PRECISION convolution_accuracy,
    const PRECISION clean_loop_gain,
    const unsigned int max_gaussian_sources_host,
    const PRECISION scale_bias_factor,
    const PRECISION clean_threshold,
    Gaussian_source_list<PRECISION> gaussian_source_list
    )
{
    // calculate msmfs derived parameters
    const unsigned int num_psf = 2*num_taylor - 1; // determined by the number of Taylor terms
    const unsigned int scale_moment_size = dirty_moment_size-2*image_border; // one dimensional size of scale moment residuals, assumed square
    const unsigned int psf_convolved_size = psf_moment_size-2*image_border; // one dimensional size of convolved psfs, assumed square
    sdp_logger(LOG_NOTICE,
        "Msmfs performed on %ux%u image with %u scales, %u Taylor terms, %u border pixels"
        ", and with %u PSF each of size %ux%u",
        dirty_moment_size, dirty_moment_size, num_scales, num_taylor, image_border,
        num_psf, psf_moment_size, psf_moment_size);

    // calculate suitable cuda block size in1D and 2D and number of available cuda threads
    int cuda_block_size = 0;
    dim3 cuda_block_size_2D = 0;
    int cuda_num_threads = 0;
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
//    display_inverse_hessian_matrices<PRECISION>(inverse_hessian_matrices_device, num_scales, num_taylor);
    // ***** end of temporary code *****

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

    // clean up allocated memory for scale moment residuals and inverse hessian matrices
    free_inverse_hessian_matrices<PRECISION>(inverse_hessian_matrices_device);
    free_scale_moment_residuals<PRECISION>(scale_moment_residuals_device);

    // deallocate shape configurations
    free_shape_configurations<PRECISION>(shape_configs);
}


/*****************************************************************************
 * C (untemplated) version of the function which allocates and clears the data structure that will
 * hold all the dirty moment images on the device.
 *****************************************************************************/
sdp_Mem *sdp_msmfs_allocate_dirty_image
    (const unsigned int dirty_moment_size, unsigned int num_taylor, sdp_MemType mem_type)
{
    sdp_Mem *dirty_moment_images = NULL;
    if (mem_type == SDP_MEM_FLOAT)
    {
        // wrap the test input images as sdp_Mem
        sdp_Error *status = NULL;
        float *dirty_moment_images_device = allocate_dirty_image<float>(dirty_moment_size, num_taylor);
        const int64_t dirty_moment_shape[] = {num_taylor, dirty_moment_size, dirty_moment_size};
        dirty_moment_images = sdp_mem_create_wrapper(dirty_moment_images_device, SDP_MEM_FLOAT, SDP_MEM_GPU, 3, dirty_moment_shape, 0, status);
    }
    else if (mem_type == SDP_MEM_DOUBLE)
    {
        // wrap the test input images as sdp_Mem
        sdp_Error *status = NULL;
        double *dirty_moment_images_device = allocate_dirty_image<double>(dirty_moment_size, num_taylor);
        const int64_t dirty_moment_shape[] = {num_taylor, dirty_moment_size, dirty_moment_size};
        dirty_moment_images = sdp_mem_create_wrapper(dirty_moment_images_device, SDP_MEM_DOUBLE, SDP_MEM_GPU, 3, dirty_moment_shape, 0, status);
    }
    else
    {
        sdp_logger(LOG_CRIT, "Dirty moment images must be either SDP_MEM_FLOAT or else SDP_MEM_DOUBLE");
    }
    return dirty_moment_images;
}


/*****************************************************************************
 * C (untemplated) version of the function which adds some sources to dirty_moment_images_device for testing
 *****************************************************************************/
void sdp_msmfs_calculate_simple_dirty_image
    (
    sdp_Mem *dirty_moment_images, unsigned int dirty_moment_size, unsigned int num_taylor
    )
{
    if (sdp_mem_type(dirty_moment_images) == SDP_MEM_FLOAT)
    {
        calculate_simple_dirty_image<float>((float *)sdp_mem_data(dirty_moment_images), dirty_moment_size, num_taylor);
    }
    else if (sdp_mem_type(dirty_moment_images) == SDP_MEM_DOUBLE)
    {
        calculate_simple_dirty_image<double>((double *)sdp_mem_data(dirty_moment_images), dirty_moment_size, num_taylor);
    }
    else
    {
        sdp_logger(LOG_CRIT, "Dirty moment images must be either SDP_MEM_FLOAT or else SDP_MEM_DOUBLE");
    }
}


/*****************************************************************************
 * C (untemplated) version of the function which deallocates device data structure that was used to
 * hold all the dirty moment images on the device
 *****************************************************************************/
void sdp_msmfs_free_dirty_image(sdp_Mem *dirty_moment_images)
{
    if (sdp_mem_type(dirty_moment_images) == SDP_MEM_FLOAT)
    {
        free_dirty_image<float>((float *)sdp_mem_data(dirty_moment_images));
    }
    else if (sdp_mem_type(dirty_moment_images) == SDP_MEM_DOUBLE)
    {
        free_dirty_image<double>((double *)sdp_mem_data(dirty_moment_images));
    }
    else
    {
        sdp_logger(LOG_CRIT, "Dirty moment images must be either SDP_MEM_FLOAT or else SDP_MEM_DOUBLE");
    }
}


/*****************************************************************************
 * C (untemplated) version of the function which allocates and clears the data structure that will
 * hold all the psf moment images on the device
 *****************************************************************************/
sdp_Mem *sdp_msmfs_allocate_psf_image
    (const unsigned int psf_moment_size, unsigned int num_psf, sdp_MemType mem_type)
{
    sdp_Mem *psf_moment_images = NULL;
    if (mem_type == SDP_MEM_FLOAT)
    {
        // wrap the test input images as sdp_Mem
        sdp_Error *status = NULL;
        float *psf_moment_images_device = allocate_psf_image<float>(psf_moment_size, num_psf);
        const int64_t psf_moment_shape[] = {num_psf, psf_moment_size, psf_moment_size};
        psf_moment_images = sdp_mem_create_wrapper(psf_moment_images_device, SDP_MEM_FLOAT, SDP_MEM_GPU, 3, psf_moment_shape, 0, status);
    }
    else if (mem_type == SDP_MEM_DOUBLE)
    {
        // wrap the test input images as sdp_Mem
        sdp_Error *status = NULL;
        double *psf_moment_images_device = allocate_psf_image<double>(psf_moment_size, num_psf);
        const int64_t psf_moment_shape[] = {num_psf, psf_moment_size, psf_moment_size};
        psf_moment_images = sdp_mem_create_wrapper(psf_moment_images_device, SDP_MEM_DOUBLE, SDP_MEM_GPU, 3, psf_moment_shape, 0, status);
    }
    else
    {
        sdp_logger(LOG_CRIT, "PSF moment images must be either SDP_MEM_FLOAT or else SDP_MEM_DOUBLE");
    }
    return psf_moment_images;
}


/*****************************************************************************
 * C (untemplated) version of the function which create a simple test input paraboloid psf
 * with specified radius and dropoff amplitude between successive taylor terms.
 *****************************************************************************/
void sdp_msmfs_calculate_simple_psf_image
    (
    sdp_Mem *psf_moment_images, unsigned int psf_moment_size, unsigned int num_psf
    )
{
    if (sdp_mem_type(psf_moment_images) == SDP_MEM_FLOAT)
    {
        calculate_simple_psf_image<float>((float *)sdp_mem_data(psf_moment_images), psf_moment_size, num_psf);
    }
    else if (sdp_mem_type(psf_moment_images) == SDP_MEM_DOUBLE)
    {
        calculate_simple_psf_image<double>((double *)sdp_mem_data(psf_moment_images), psf_moment_size, num_psf);
    }
    else
    {
        sdp_logger(LOG_CRIT, "PSF moment images must be either SDP_MEM_FLOAT or else SDP_MEM_DOUBLE");
    }
}


/*****************************************************************************
 * C (untemplated) version of the function which deallocates device data structure that was used to
 * hold all the psf moment images on the device
 *****************************************************************************/
void sdp_msmfs_free_psf_image(sdp_Mem *psf_moment_images)
{
    if (sdp_mem_type(psf_moment_images) == SDP_MEM_FLOAT)
    {
        free_psf_image<float>((float *)sdp_mem_data(psf_moment_images));
    }
    else if (sdp_mem_type(psf_moment_images) == SDP_MEM_DOUBLE)
    {
        free_psf_image<double>((double *)sdp_mem_data(psf_moment_images));
    }
    else
    {
        sdp_logger(LOG_CRIT, "PSF moment images must be either SDP_MEM_FLOAT or else SDP_MEM_DOUBLE");
    }
}


/*****************************************************************************
 * C (untemplated) version of the function which performs the entire msmfs deconvolution using sdp_Mem handles
 *****************************************************************************/
void sdp_msmfs_perform
    (
    sdp_Mem *dirty_moment_images,
    sdp_Mem *psf_moment_images,
    const unsigned int dirty_moment_size,
    const unsigned int num_scales,
    const unsigned int num_taylor,
    const unsigned int psf_moment_size,
    const unsigned int image_border,
    const double convolution_accuracy,
    const double clean_loop_gain,
    const unsigned int max_gaussian_sources_host,
    const double scale_bias_factor,
    const double clean_threshold,
    unsigned int *num_gaussian_sources_host,
    sdp_Mem *gaussian_source_position,
    sdp_Mem *gaussian_source_variance,
    sdp_Mem *gaussian_source_taylor_intensities
    )
{
    if (sdp_mem_type(dirty_moment_images)==SDP_MEM_FLOAT && sdp_mem_type(psf_moment_images)==SDP_MEM_FLOAT)
    {
        // create an initially empty list of sources to hold those gaussian source found in the minor cycle loops (for single major cycle)
        Gaussian_source_list<float> gaussian_source_list = allocate_gaussian_source_list<float>(max_gaussian_sources_host);
        
        sdp_perform_msmfs<float>(
            (float *)sdp_mem_data(dirty_moment_images), (float *)sdp_mem_data(psf_moment_images),
            dirty_moment_size, num_scales, num_taylor, psf_moment_size, image_border,
            (float)convolution_accuracy, (float)clean_loop_gain, max_gaussian_sources_host,
            (float)scale_bias_factor, (float)clean_threshold,
            gaussian_source_list);

        // copy the resulting gaussian_source_list back to the host
        Gaussian_source<float> *gaussian_sources_host = nullptr; // sources that have distinct scales/positions (duplicates get merged)
        gaussian_sources_host = (Gaussian_source<float>*)malloc(max_gaussian_sources_host*sizeof(Gaussian_source<float>));
        copy_gaussian_source_list_to_host<float>
            (gaussian_source_list.gaussian_sources_device, gaussian_source_list.num_gaussian_sources_device,
            max_gaussian_sources_host, num_gaussian_sources_host, gaussian_sources_host);

// NOTE printf INCLUDED TEMPORARILY UNTIL THIS CODE IS UNIT TESTED
//        printf("In total %u distinct sources were discovered during cleaning\n", *num_gaussian_sources_host);
        // copy each distinct source that has been found to the sdp_Mem data structures
        for (unsigned int source_index=0; source_index<*num_gaussian_sources_host; source_index++)
        {
            Gaussian_source<float> source = (Gaussian_source<float>)gaussian_sources_host[source_index];
            unsigned int x_pos = (source.index % (dirty_moment_size-2*image_border)) + image_border;
            unsigned int y_pos = (source.index / (dirty_moment_size-2*image_border)) + image_border; 
            ((uint2 *)sdp_mem_data(gaussian_source_position))[source_index].x = x_pos;
            ((uint2 *)sdp_mem_data(gaussian_source_position))[source_index].y = y_pos;
            ((float *)sdp_mem_data(gaussian_source_variance))[source_index] = source.variance;
//            printf("Source %3u has scale variance %8.2lf and position (%5u,%5u) with taylor term intensities: ",
//                source_index, source.variance, x_pos, y_pos);
            for (unsigned int taylor_index=0; taylor_index<num_taylor; taylor_index++)
            {
                ((float *)sdp_mem_data(gaussian_source_taylor_intensities))[source_index*num_taylor + taylor_index] = source.intensities[taylor_index];
//                printf("%+12lf ", source.intensities[taylor_index]);
            }
//            printf("\n");
        }
        free(gaussian_sources_host);

        // ***** temporary code to display the gaussian sources found in the minor cycle loops *****
//        display_gaussian_source_list<float>(gaussian_source_list.gaussian_sources_device, gaussian_source_list.num_gaussian_sources_device,
//            max_gaussian_sources_host, dirty_moment_size, image_border, num_taylor);
        // ***** end of temporary code *****

        // clean up the source model
        free_gaussian_source_list<float>(gaussian_source_list);
    }
    else if (sdp_mem_type(dirty_moment_images)==SDP_MEM_DOUBLE && sdp_mem_type(psf_moment_images)==SDP_MEM_DOUBLE)
    {
        // create an initially empty list of sources to hold those gaussian source found in the minor cycle loops (for single major cycle)
        Gaussian_source_list<double> gaussian_source_list = allocate_gaussian_source_list<double>(max_gaussian_sources_host);
        
        sdp_perform_msmfs<double>(
            (double *)sdp_mem_data(dirty_moment_images), (double *)sdp_mem_data(psf_moment_images),
            dirty_moment_size, num_scales, num_taylor, psf_moment_size, image_border,
            convolution_accuracy, clean_loop_gain, max_gaussian_sources_host,
            scale_bias_factor, clean_threshold,
            gaussian_source_list);

        // copy the resulting gaussian_source_list back to the host
        Gaussian_source<double> *gaussian_sources_host = nullptr; // sources that have distinct scales/positions (duplicates get merged)
        gaussian_sources_host = (Gaussian_source<double>*)malloc(max_gaussian_sources_host*sizeof(Gaussian_source<double>));
        copy_gaussian_source_list_to_host<double>
            (gaussian_source_list.gaussian_sources_device, gaussian_source_list.num_gaussian_sources_device,
            max_gaussian_sources_host, num_gaussian_sources_host, gaussian_sources_host);

// NOTE printf INCLUDED TEMPORARILY UNTIL THIS CODE IS UNIT TESTED
        printf("In total %u distinct sources were discovered during cleaning\n", *num_gaussian_sources_host);
        // copy each distinct source that has been found to the sdp_Mem data structures
        for (unsigned int source_index=0; source_index<*num_gaussian_sources_host; source_index++)
        {
            Gaussian_source<double> source = (Gaussian_source<double>)gaussian_sources_host[source_index];
            unsigned int x_pos = (source.index % (dirty_moment_size-2*image_border)) + image_border;
            unsigned int y_pos = (source.index / (dirty_moment_size-2*image_border)) + image_border; 
            ((uint2 *)sdp_mem_data(gaussian_source_position))[source_index].x = x_pos;
            ((uint2 *)sdp_mem_data(gaussian_source_position))[source_index].y = y_pos;
            ((double *)sdp_mem_data(gaussian_source_variance))[source_index] = source.variance;
//            printf("Source %3u has scale variance %8.2lf and position (%5u,%5u) with taylor term intensities: ",
//                source_index, source.variance, x_pos, y_pos);
            for (unsigned int taylor_index=0; taylor_index<num_taylor; taylor_index++)
            {
                ((double *)sdp_mem_data(gaussian_source_taylor_intensities))[source_index*num_taylor + taylor_index] = source.intensities[taylor_index];
//                printf("%+12lf ", source.intensities[taylor_index]);
            }
//            printf("\n");
        }
        free(gaussian_sources_host);

        // ***** temporary code to display the gaussian sources found in the minor cycle loops *****
        display_gaussian_source_list<double>(gaussian_source_list.gaussian_sources_device, gaussian_source_list.num_gaussian_sources_device,
            max_gaussian_sources_host, dirty_moment_size, image_border, num_taylor);
        // ***** end of temporary code *****

        // clean up the source model
        free_gaussian_source_list<double>(gaussian_source_list);
    }
    else
    {
        sdp_logger(LOG_CRIT, "Dirty moment images and PSF moment images must both be either SDP_MEM_FLOAT or else both SDP_MEM_DOUBLE");
    }
}
