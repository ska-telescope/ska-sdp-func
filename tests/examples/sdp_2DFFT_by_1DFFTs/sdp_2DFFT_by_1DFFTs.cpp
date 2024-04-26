// ============================================================================
// Name        : sdp_2DFFT_by_1DFFTs.cpp
// Author      : Vlad Stolyarov
// Version     : v0.01
// Copyright   :
// Description : An example top-level program that uses
//             : sdp_grid_simulator_VLA and cuFFT functions
//             : to invert a simulated grid
// ============================================================================

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cufftXt.h>
#include <cufft.h>
#include <iostream>
#include <vector>
#include "fitsio.h"

#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"

#include "sdp_grid_simulator_VLA.h"

using namespace std;

#ifndef M_PI
#define M_PI 3.1415926535897931
#endif




//fftshift - https://www.dsprelated.com/showthread/comp.dsp/20790-1.php

void fftshift(sdp_Double2* x,
		int m, int n){
//	int m, n;      // FFT row and column dimensions might be different
	int m2, n2;
	int i, k;
	int idx,idx1, idx2, idx3;
	sdp_Double2 tmp13, tmp24;

	m2 = m / 2;    // half of row dimension
	n2 = n / 2;    // half of column dimension

	// interchange entries in 4 quadrants, 1 <--> 3 and 2 <--> 4

	for (i = 0; i < m2; i++)
	{
	     for (k = 0; k < n2; k++)
	     {
	          idx			= i*n + k;
	          tmp13			= x[idx];

	          idx1          = (i+m2)*n + (k+n2);
	          x[idx]        = x[idx1];
	          x[idx1]       = tmp13;

	          idx2          = (i+m2)*n + k;
	          tmp24         = x[idx2];

	          idx3          = i*n + (k+n2);
	          x[idx2]       = x[idx3];
	          x[idx3]       = tmp24;
	     }
	}
}


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
    int gpu_start, idx, idx2;
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

    // 1D FFT to 2D FFT

    cudaError_t	cudaStatus;
    cufftResult cufftStatus;
    cufftHandle plan;
    cufftDoubleComplex *odata, *idata;
    cufftDoubleComplex *h_odata, *h_idata;
    sdp_Double2* image_fits = (sdp_Double2*) sdp_mem_data(image_out);

    // Create 1D FFT setup
    cufftDoubleComplex *odata_1d, *idata_1d;
    cufftDoubleComplex *h_odata_1d, *h_idata_1d;
    cufftHandle plan_1d;

    int batch_size = 1024; //2,4,8,16 etc
    cudaStatus = cudaMalloc((void**)&odata_1d, sizeof(cufftDoubleComplex)*grid_size*batch_size);
    cudaStatus = cudaMalloc((void**)&idata_1d, sizeof(cufftDoubleComplex)*grid_size*batch_size);
    cudaStatus = cudaMallocHost((void**)&h_odata_1d, sizeof(cufftDoubleComplex)*grid_size*batch_size);
    cudaStatus = cudaMallocHost((void**)&h_idata_1d, sizeof(cufftDoubleComplex)*grid_size*batch_size);
    if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMalloc failed! Can't allocate GPU memory for d_original_signal\n");
		exit(EXIT_FAILURE);
	}
    SDP_LOG_INFO("cudaMalloc\n", cudaStatus);


    int rank = 1;                           // --- 1D FFTs
    int n[] = { (int)grid_size };           // --- Size of the Fourier transform
    int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
    int idist = grid_size, odist = grid_size; // --- Distance between batches
    int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
    int batch = batch_size;                 // --- Number of batched executions
    cufftStatus = cufftPlanMany(&plan_1d, rank, n,
                  inembed, istride, idist,
                  onembed, ostride, odist, CUFFT_Z2Z, batch);

	if (cufftStatus != CUFFT_SUCCESS){
		fprintf(stderr, "cufftPlan1d failed! Can't create a plan! %s\n", cufftStatus);
		exit(EXIT_FAILURE);
	}
	SDP_LOG_INFO("cufftPlanMany\n", cufftStatus);

    // Working through columns
	size_t i,j,k;
	for(i=0;i<grid_size; i+=batch_size){
	    for(k=0;k<batch_size;k++)
		// Copy the column
		  for(j=0;j<grid_size; j++){
			idx = (int)(i + k + j*grid_size);
			h_idata_1d[j+k*grid_size].x = grid_out[idx].x;
			h_idata_1d[j+k*grid_size].y = grid_out[idx].y;
		}
		// H2D copy the column
	    cudaStatus = cudaMemcpy(idata_1d, h_idata_1d, sizeof(cufftDoubleComplex)*grid_size*batch_size, cudaMemcpyHostToDevice);
	    if (cudaStatus != cudaSuccess){
			fprintf(stderr, "cudaMemcpy failed! Can't copy to GPU memory\n");
			exit(EXIT_FAILURE);
		}

	    // Make inverse 1D FFT
		cufftStatus = cufftExecZ2Z(plan_1d, idata_1d, odata_1d, CUFFT_INVERSE);
		if (cufftStatus != CUFFT_SUCCESS){
			fprintf(stderr, "cufftExecZ2Z failed! Can't make Z2Z transform!\n");
			exit(EXIT_FAILURE);
		}

		// D2H copy the column
	    cudaStatus = cudaMemcpy(h_odata_1d, odata_1d, sizeof(cufftDoubleComplex)*grid_size*batch_size, cudaMemcpyDeviceToHost);
	    if (cudaStatus != cudaSuccess){
			fprintf(stderr, "cudaMemcpy failed! Can't copy to GPU memory\n");
			exit(EXIT_FAILURE);
		}
	    // Update the column in image_out array
	    for(k=0;k<batch_size;k++)
		  for(j=0;j<grid_size; j++){
			idx = (int)(i + k + j*grid_size);
			image_fits[idx].x =  h_odata_1d[j+k*grid_size].x;
			image_fits[idx].y =  h_odata_1d[j+k*grid_size].y;
		}
	}




    // Working through rows
	for(j=0;j<grid_size; j+=batch_size){
        for(k=0;k<batch_size;k++)
		// Copy the row
		  for(i=0;i<grid_size; i++){
			idx = (int)(i + (j+k)*grid_size);
			h_idata_1d[i+k*grid_size].x = image_fits[idx].x;
			h_idata_1d[i+k*grid_size].y = image_fits[idx].y;
		}
		// H2D copy the column
	    cudaStatus = cudaMemcpy(idata_1d, h_idata_1d, sizeof(cufftDoubleComplex)*grid_size*batch_size, cudaMemcpyHostToDevice);
	    if (cudaStatus != cudaSuccess){
			fprintf(stderr, "cudaMemcpy failed! Can't copy to GPU memory\n");
			exit(EXIT_FAILURE);
		}

	    // Make inverse 1D FFT
		cufftStatus = cufftExecZ2Z(plan_1d, idata_1d, odata_1d, CUFFT_INVERSE);
		if (cufftStatus != CUFFT_SUCCESS){
			fprintf(stderr, "cufftExecZ2Z failed! Can't make Z2Z transform!\n");
			exit(EXIT_FAILURE);
		}

		// D2H copy the column
	    cudaStatus = cudaMemcpy(h_odata_1d, odata_1d, sizeof(cufftDoubleComplex)*grid_size*batch_size, cudaMemcpyDeviceToHost);
	    if (cudaStatus != cudaSuccess){
			fprintf(stderr, "cudaMemcpy failed! Can't copy to GPU memory\n");
			exit(EXIT_FAILURE);
		}
	    // Update the row in image_out array
        for(k=0;k<batch_size;k++)
          for(i=0;i<grid_size; i++){
			idx = (int)(i + (j+k)*grid_size);
			image_fits[idx].x =  h_odata_1d[i+k*grid_size].x/(grid_size * grid_size);
			image_fits[idx].y =  h_odata_1d[i+k*grid_size].y/(grid_size * grid_size);
		}
	}

	SDP_LOG_INFO("cufftExecZ2Z finished%s\n", cufftStatus);
    	fftshift(image_fits, (int)grid_size, (int)grid_size);
	SDP_LOG_INFO("fftshift finished%s\n", cufftStatus);


    // Output into the FITS file
    const char filename_img1D[] = "!image_sim_1d2d.fits";

    status_fits = fits_write(image_fits,
            grid_size,
            grid_size_default,
            filename_img1D
    );
    SDP_LOG_INFO("FITSIO status %d\n", status_fits);

    sdp_mem_free(sources);
    sdp_mem_free(grid_sim);
    sdp_mem_free(image_out);
    
    cudaFree(odata_1d);
    cudaFree(idata_1d);
    cudaFree(h_odata_1d);
    cudaFree(h_idata_1d);

    return 0;
}
