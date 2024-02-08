// ============================================================================
// Name        : sdp_grid_simulator.cpp
// Author      : Vlad Stolyarov
// Version     : v0.01
// Copyright   :
// Description : An example top-level program that uses
//             : sdp_grid_simulator_VLA and sdp_cuFFTxT
//             : to invert a simulated grid
// ============================================================================

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cufftXt.h>
#include <iostream>
#include <vector>
#include "fitsio.h"

#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"

#include "sdp_cuFFTxT.h"
#include "sdp_grid_simulator_VLA.h"

using namespace std;

#ifndef M_PI
#define M_PI 3.1415926535897931
#endif


// Output the absolute value of the 2D array into the FITS file, compressing if required


int fits_write(
        sdp_Double2* image,
        long imsize,
        long imsize_default,
        const char* filename
)
{
    int status;
    long ii, jj, kk, ll, idx, idx2;
    long kkstart, llstart;
    long fpixel = 1, naxis = 2, nelements, exposure;
    long naxes[2];  /* image is imsize pixels wide by imsize rows */
    float* array;// [200][300];
    fitsfile* fptr;       /* pointer to the FITS file, defined in fitsio.h */
    status = 0; /* initialize status before calling fitsio routines */

    if (imsize < imsize_default)
        imsize_default = imsize;

    naxes[0] = imsize_default;
    naxes[1] = imsize_default;

    fits_create_file(&fptr, filename, &status); /* create new file */

    /* Create the primary array image (16-bit short integer pixels */
    fits_create_img(fptr, FLOAT_IMG, naxis, naxes, &status);
    /* Write a keyword; must pass the ADDRESS of the value - to be replaced*/
    exposure = 1500.;
    fits_update_key(fptr,
            TLONG,
            "EXPOSURE",
            &exposure,
            "Total Exposure Time",
            &status
    );

    nelements = naxes[0] * naxes[1]; /* number of pixels to write */
    array = (float*)malloc(nelements * sizeof(float));
    if (array == NULL)
    {
        printf("fits_write: memory not allocated.\n");
        exit(-1);
    }
    /*Fill the array to write into the fits file */
    if (imsize == imsize_default)
    {
        for (jj = 0; jj < naxes[1]; jj++)
            for (ii = 0; ii < naxes[0]; ii++)
            {
                idx = jj * naxes[1] + ii;
                array[idx] =
                        (float)(sqrt(image[idx].x * image[idx].x +
                        image[idx].y * image[idx].y
                        ));
            }
    }
    else
    {
        int ratio = imsize / imsize_default;
        SDP_LOG_INFO("Compressing output image in %d times, %d -> %d",
                ratio,
                imsize,
                imsize_default
        );
        for (jj = 0; jj < naxes[0]; jj++)
            for (ii = 0; ii < naxes[0]; ii++)
            {
                idx = jj * naxes[0] + ii;
                llstart = ratio * jj;
                kkstart = ratio * ii;
                for (ll = llstart; ll < llstart + ratio; ll++)
                    for (kk = kkstart; kk < kkstart + ratio; kk++)
                    {
                        idx2 = ll * imsize + kk;
                        array[idx] +=
                                (float)(sqrt(image[idx2].x * image[idx2].x +
                                image[idx2].y * image[idx2].y
                                ));
                    }
                array[idx] /= (float)(ratio * ratio);
            }
    }
    /* Write the array of integers to the image */
    fits_write_img(fptr, TFLOAT, fpixel, nelements, array, &status);
    fits_close_file(fptr, &status); /* close the file */
    fits_report_error(stderr, status); /* print out any error messages */

    return status;
}


int main(int argc, char** argv)
{
    int num_sources;
    double ha_start;
    double ha_step;
    double uvw_scale;
    int ha_num;
    double dec_rad;
    int64_t grid_size;
    int64_t grid_size_default;
    sdp_MemType sources_type;
    sdp_MemType grid_sim_type;
    sdp_MemType image_out_type;
    sdp_Error status = SDP_SUCCESS;

    SDP_LOG_INFO("SDP Grid Simulator (VLA) v.0.01\n");

    grid_size = 8192;
    grid_size_default = 4096;
    printf("Program Name Is: %s\n", argv[0]);
    if (argc == 2)
    {
        grid_size = (int64_t) atoi(argv[1]);
    }
    else if (argc == 3)
    {
        grid_size = (int64_t) atoi(argv[1]);
        grid_size_default = (int64_t) atoi(argv[2]);
    }

    SDP_LOG_INFO("Grid size is %ld x %ld\n", grid_size, grid_size);
    SDP_LOG_INFO("Grid output size is %ld x %ld\n",
            grid_size_default,
            grid_size_default
    );

    /*
     * Define a list of sources,
     * Amplitude, l, m
     */

    num_sources = 4;
    const double sources_arr[][3] =
    {{1.0, 0.001,  0.001},
        {5.0, 0.0025, 0.0025},
        {10.0, 0.025, 0.025},
        {5.0, 0.025, 0.05}};

    int64_t sources_shape[] = {num_sources, 3};
    sources_type = SDP_MEM_DOUBLE;
    sdp_Mem* sources =
            sdp_mem_create(sources_type, SDP_MEM_CPU, 2, sources_shape,
            &status
            );

    SDP_LOG_INFO("Preparing test source list");
    sdp_mem_clear_contents(sources, &status);
    void* sources_1 = (void*)sdp_mem_data(sources);
    double* temp = (double*)sources_1;
    for (size_t i = 0; i < num_sources; i++)
    {
        temp[3 * i] = sources_arr[i][0];
        temp[3 * i + 1] = sources_arr[i][1];
        temp[3 * i + 2] = sources_arr[i][2];
    }

    double* est_sources = (double*) sdp_mem_data(sources);
    for (size_t i = 0; i < num_sources; i++)
        SDP_LOG_INFO("%f %f %f",
                est_sources[3 * i],
                est_sources[3 * i + 1],
                est_sources[3 * i + 2]
        );

    // Scale factor for uvw values
    uvw_scale = 10.;

    // Hour angle start, step, number of points
    ha_start = 0.0;
    ha_step = 0.04;
    ha_num = 79;

    // Declination of the phase centre
    dec_rad = M_PI / 4.0;

    // Grid and image size
    int64_t grid_sim_shape[] = {grid_size, grid_size};
    grid_sim_type = SDP_MEM_COMPLEX_DOUBLE;
    sdp_Mem* grid_sim =
            sdp_mem_create(grid_sim_type,
            SDP_MEM_CPU,
            2,
            grid_sim_shape,
            &status
            );

    sdp_mem_clear_contents(grid_sim, &status);

    sdp_grid_simulator_VLA(
            sources,
            ha_start,
            ha_step,
            ha_num,
            dec_rad,
            uvw_scale,
            grid_sim,
            &status
    );

    // Output into the FITS file
    const char filename_vis[] = "!grid_sim.fits";
    SDP_LOG_INFO("Writing grid into %s", filename_vis);
    int status_fits;

    sdp_Double2* grid_out = (sdp_Double2*) sdp_mem_data(grid_sim);
    status_fits = fits_write(grid_out,
            grid_size,
            grid_size_default,
            filename_vis
    );
    SDP_LOG_INFO("FITSIO status %d\n", status_fits);

    // Allocate memory for the image
    int64_t image_out_shape[] = {grid_size, grid_size};
    image_out_type = SDP_MEM_COMPLEX_DOUBLE;
    sdp_Mem* image_out =
            sdp_mem_create(image_out_type,
            SDP_MEM_CPU,
            2,
            image_out_shape,
            &status
            );

    sdp_mem_clear_contents(image_out, &status);

    // Run cuFFTxT to invert the simulated grid
    sdp_cuFFTxT(
            grid_sim,
            image_out,
            &status
    );

    // Output into the FITS file
    const char filename_img[] = "!image_sim.fits";

    sdp_Double2* image_fits = (sdp_Double2*) sdp_mem_data(image_out);
    status_fits = fits_write(image_fits,
            grid_size,
            grid_size_default,
            filename_img
    );
    SDP_LOG_INFO("FITSIO status %d\n", status_fits);

    sdp_mem_free(sources);
    sdp_mem_free(grid_sim);

    return 0;
}
