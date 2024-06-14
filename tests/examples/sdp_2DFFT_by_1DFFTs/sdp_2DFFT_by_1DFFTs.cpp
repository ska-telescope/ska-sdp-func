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
#include <cufft.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <time.h>
#include <fftw3.h>
#include "fitsio.h"

#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"

//#include "sdp_cuFFTxT.h"
#include "sdp_grid_simulator_VLA.h"

using namespace std;

#ifndef M_PI
#define M_PI 3.1415926535897931
#endif

#define NUM_STREAMS 4

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}



//fftshift - https://www.dsprelated.com/showthread/comp.dsp/20790-1.php

void fftshift(sdp_Double2* x,
		int64_t m, int64_t n){
//	int m, n;      // FFT row and column dimensions might be different
	int64_t m2, n2;
	int64_t i, k;
	int64_t idx,idx1, idx2, idx3;
	sdp_Double2 tmp13, tmp24;
	double dnorm;

	m2 = m / 2;    // half of row dimension
	n2 = n / 2;    // half of column dimension
	dnorm = (double)(m*n);
	// interchange entries in 4 quadrants, 1 <--> 3 and 2 <--> 4

	for (i = 0; i < m2; i++)
	{
	     for (k = 0; k < n2; k++)
	     {
	          idx			= i*n + k;
	          tmp13.x			= x[idx].x/dnorm;
	          tmp13.y			= x[idx].y/dnorm;

	          idx1          = (i+m2)*n + (k+n2);
	          x[idx].x        = x[idx1].x/dnorm;
	          x[idx].y        = x[idx1].y/dnorm;

	          x[idx1]       = tmp13;

	          idx2          = (i+m2)*n + k;
	          tmp24.x         = x[idx2].x/dnorm;
	          tmp24.y         = x[idx2].y/dnorm;

	          idx3          = i*n + (k+n2);
	          x[idx2].x       = x[idx3].x/dnorm;
	          x[idx2].y       = x[idx3].y/dnorm;

	          x[idx3]       = tmp24;
	     }
	}
}


// naively transpose a square matrix
void dgsit(sdp_Double2 *A, int64_t n){
	int64_t i, j;
    sdp_Double2 temp;
    for (i = 0; i < n; i++){
        for (j = i + 1; j < n; j++){
            temp = A[i * n + j];
            A[i * n + j] = A[j * n + i];
            A[j * n + i] = temp;
        }
    }
}

inline void __transpose_block(sdp_Double2 *A, int64_t n, int64_t istart, int64_t jstart, int64_t block){
	int64_t i, j;
    sdp_Double2 temp;
    for (i = istart; i < istart+block; i++){
        for (j = jstart; j < jstart+block; j++){
            temp = A[i * n + j];
            A[i * n + j] = A[j * n + i];
            A[j * n + i] = temp;
        }
    }
}


inline void __transpose_block_diag(sdp_Double2 *A, int64_t n, int64_t istart, int64_t jstart, int64_t block){
	int64_t i, j;
    sdp_Double2 temp;
    for (i = istart; i < istart+block; i++){
        for (j = i + 1; j < jstart+block; j++){
            temp = A[i * n + j];
            A[i * n + j] = A[j * n + i];
            A[j * n + i] = temp;
        }
    }
}

void transpose_inplace_block(sdp_Double2 *A, int64_t n, int64_t block){
	int64_t i, j;
	int64_t r = n % block, m = n - r;
    printf("%d %d %d %d\n", block, n, r, m);
    sdp_Double2 temp;
    // if dimension of matrix is less than block size just do naive
    if (n < block){
        return dgsit(A, n);
    }
    // transpose square blocks
    for (i = 0; i < m; i += block){
        for (j = i; j < m; j += block){
            (i == j) ? __transpose_block_diag(A, n, i, j, block) : __transpose_block(A, n, i, j, block);

        }
    }
    // take care of the remaining swaps naively

    if (r){
    	printf("Corner processing\n");
        // transpose rectangular sub-matrix
        for (j = m; j < n; j++){
            for (i = 0; i < m; i++){
                temp = A[i * n + j];
                A[i * n + j] = A[j * n + i];
                A[j * n + i] = temp;
            }
        }
        // transpose square sub matrix in "bottom right"
        for (i = m; i < n; i++){
            for (j = i + 1; j < n; j++){
                temp = A[i * n + j];
                A[i * n + j] = A[j * n + i];
                A[j * n + i] = temp;
            }
        }
    }

}

void transpose_inplace(sdp_Double2* data, int64_t m)
{
  const int64_t size1 = m;
  const int64_t size2 = m;
  int64_t i, j, k;

  for (i = 0; i < size1; i++)
    {
      for (j = i + 1 ; j < size2 ; j++)
        {
    	    int64_t e1 = (i *  m + j);
    	    int64_t e2 = (j *  m + i);
            sdp_Double2 tmp = data[e1] ;
            data[e1] = data[e2] ;
            data[e2] = tmp ;
        }
    }

}

void cufft_stream_1stage(
		cufftDoubleComplex *idata_1d_all,
		cufftDoubleComplex *odata_1d_all,
		sdp_Double2* grid_out,
		sdp_Double2* image_fits,
		int64_t grid_size,
		int64_t batch_size,
		cufftHandle* plan_1d,
		cudaStream_t* streams,
		int stream_number,
		size_t j
		){
	cudaError_t	cudaStatus;
	cufftResult cufftStatus;
	int64_t idx_d, idx_h;

	idx_d = stream_number*batch_size*grid_size;
	idx_h = (int64_t)(((int64_t)j+(int64_t)stream_number*batch_size)*grid_size);
	//SDP_LOG_INFO("Stream %d, J = %ld, idx_h = %ld, idx_d = %ld", stream_number, (int)j,idx_h,idx_d);

	cudaStatus = cudaMemcpyAsync((idata_1d_all+idx_d), (cufftDoubleComplex*)(grid_out + idx_h), sizeof(cufftDoubleComplex)*grid_size*batch_size, cudaMemcpyHostToDevice, streams[stream_number]);
    if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMemcpy failed! Can't copy to GPU memory\n");
		exit(EXIT_FAILURE);
	}

	cufftStatus = cufftExecZ2Z(plan_1d[stream_number], (idata_1d_all+idx_d), (odata_1d_all+idx_d), CUFFT_INVERSE);
	if (cufftStatus != CUFFT_SUCCESS){
		fprintf(stderr, "cufftExecZ2Z failed! Can't make Z2Z transform!\n");
		exit(EXIT_FAILURE);
	}
	cudaStatus = cudaMemcpyAsync((cufftDoubleComplex*)(image_fits + idx_h), (odata_1d_all+idx_d), sizeof(cufftDoubleComplex)*grid_size*batch_size, cudaMemcpyDeviceToHost, streams[stream_number]);

    if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMemcpy failed! Can't copy from GPU memory\n");
		exit(EXIT_FAILURE);
	}

}

void cufft_stream_2stage(
		cufftDoubleComplex *idata_1d_all,
		cufftDoubleComplex *odata_1d_all,
		sdp_Double2* image_fits,
		int64_t grid_size,
		int64_t batch_size,
		cufftHandle* plan_1d,
		cudaStream_t* streams,
		int stream_number,
		size_t j
		){
	cudaError_t	cudaStatus;
	cufftResult cufftStatus;
	int64_t idx_d, idx_h;

	idx_d = stream_number*batch_size*grid_size;
	idx_h = (int64_t)(((int64_t)j+(int64_t)stream_number*batch_size)*grid_size);
	//SDP_LOG_INFO("Stream %d, J = %ld, idx_h = %ld, idx_d = %ld", stream_number, (int)j,idx_h,idx_d);

	cudaStatus = cudaMemcpyAsync((idata_1d_all+idx_d), (cufftDoubleComplex*)(image_fits + idx_h), sizeof(cufftDoubleComplex)*grid_size*batch_size, cudaMemcpyHostToDevice, streams[stream_number]);
    if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMemcpy failed! Can't copy to GPU memory\n");
		exit(EXIT_FAILURE);
	}

	cufftStatus = cufftExecZ2Z(plan_1d[stream_number], (idata_1d_all+idx_d), (odata_1d_all+idx_d), CUFFT_INVERSE);
	if (cufftStatus != CUFFT_SUCCESS){
		fprintf(stderr, "cufftExecZ2Z failed! Can't make Z2Z transform!\n");
		exit(EXIT_FAILURE);
	}
	cudaStatus = cudaMemcpyAsync((cufftDoubleComplex*)(image_fits + idx_h), (odata_1d_all+idx_d), sizeof(cufftDoubleComplex)*grid_size*batch_size, cudaMemcpyDeviceToHost, streams[stream_number]);

    if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMemcpy failed! Can't copy from GPU memory\n");
		exit(EXIT_FAILURE);
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
    sdp_Error* status_sdp;
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
    	// Allocate memory using sdp_Mem.
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
    int gpu_start, idx, idx2, idx3, idx4;
    size_t block_size;
    int batch_size;
    int num_streams;
    int do_cuFFT2D, do_FFTW3;
    int nthreads;

    clock_t start, end;
    clock_t start_1d2d, end_1d2d;
    double cpu_time_used;

    int64_t grid_size;
    int64_t grid_size_default;
    sdp_MemType sources_type;
    sdp_MemType grid_sim_type;
    sdp_MemType image_out_type;
    sdp_Error status = SDP_SUCCESS;

    SDP_LOG_INFO("SDP Grid Simulator (VLA) v.0.01\n");

    grid_size = 8192;
    grid_size_default = 4096;
    block_size = 8;
    batch_size =1024;
    num_streams = NUM_STREAMS;
    nthreads=4;
    do_cuFFT2D = 1;
    do_FFTW3 = 0;

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
    else if (argc == 4)
    {
        grid_size = (int64_t) atoi(argv[1]);
        grid_size_default = (int64_t) atoi(argv[2]);
        batch_size = atoi(argv[3]);
    }
    else if (argc == 5)
    {
        grid_size = (int64_t) atoi(argv[1]);
        grid_size_default = (int64_t) atoi(argv[2]);
        batch_size = atoi(argv[3]);
        num_streams = atoi(argv[4]);
    }
    else if (argc == 6)
    {
        grid_size = (int64_t) atoi(argv[1]);
        grid_size_default = (int64_t) atoi(argv[2]);
        batch_size = atoi(argv[3]);
        num_streams = atoi(argv[4]);
        block_size = (size_t) atoi(argv[5]);
    }
    else if (argc == 7)
    {
        grid_size = (int64_t) atoi(argv[1]);
        grid_size_default = (int64_t) atoi(argv[2]);
        batch_size = atoi(argv[3]);
        num_streams = atoi(argv[4]);
        block_size = (size_t) atoi(argv[5]);
        nthreads = atoi(argv[6]);
    }
    else
    {
    	printf("Usage: 2DFFT_by_1DFFTs <grid_size=8192> <grid_size_default for FITS output = 4096> <batch_size=1024> <num_streams=NUM_STREAMS> <block_size=8> <nthreads=4\n");
    }


    SDP_LOG_INFO("Grid size is %ld x %ld", grid_size, grid_size);
    SDP_LOG_INFO("Grid output size is %ld x %ld",
            grid_size_default,
            grid_size_default
    );
    SDP_LOG_INFO("Block size is %ld", block_size);
    SDP_LOG_INFO("Batch size is %d", batch_size);
    SDP_LOG_INFO("Number of streams is %d", num_streams);
    SDP_LOG_INFO("Number of posix threads is %d", nthreads);

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
    SDP_LOG_INFO("FITSIO status %d", status_fits);

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

    // 2D cuFFTinv
    cudaError_t	cudaStatus;
    cufftResult cufftStatus;
    cufftHandle plan;
    cufftDoubleComplex *odata, *idata;
    cufftDoubleComplex *h_odata, *h_idata;
    sdp_Double2* image_fits = (sdp_Double2*) sdp_mem_data(image_out);

    // 1D FFT to 2D FFT

    // Clear image_out array
    sdp_mem_clear_contents(image_out, &status);
     // Create 1D FFT setup
    cufftDoubleComplex *idata_1d_all, *odata_1d_all;

    cudaStream_t* streams = (cudaStream_t*) malloc(sizeof(cudaStream_t)*num_streams);
    for (int i = 0; i < num_streams; i++) gpuErrchk(cudaStreamCreate(&streams[i]));

    cudaStatus = cudaMalloc((void**)&odata_1d_all, sizeof(cufftDoubleComplex)*grid_size*batch_size*num_streams);
    cudaStatus = cudaMalloc((void**)&idata_1d_all, sizeof(cufftDoubleComplex)*grid_size*batch_size*num_streams);

    if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMalloc failed! Can't allocate GPU memory for d_original_signal\n");
		exit(EXIT_FAILURE);
	}
    SDP_LOG_INFO("cudaMalloc %s", cudaStatus);


    int rank = 1;                           // --- 1D FFTs
    int n[] = { (int)grid_size };           // --- Size of the Fourier transform
    int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
    int idist = grid_size, odist = grid_size; // --- Distance between batches
    int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
    int batch = batch_size;                 // --- Number of batched executions

    // --- Creates cuFFT plans and sets them in streams
    cufftHandle* plan_1d = (cufftHandle*) malloc(sizeof(cufftHandle)*num_streams);
    for (int i = 0; i < num_streams; i++) {
        //cufftPlan1d(&plan_1d[i], N, CUFFT_C2C, 1);
        cufftStatus = cufftPlanMany(&plan_1d[i], rank, n,
                      inembed, istride, idist,
                      onembed, ostride, odist, CUFFT_Z2Z, batch);
    	if (cufftStatus != CUFFT_SUCCESS){
    		fprintf(stderr, "cufftPlan1d failed! Can't create a plan! %s\n", cufftStatus);
    		exit(EXIT_FAILURE);
    	}
    	SDP_LOG_INFO("cufftPlanMany %d %s", i, cufftStatus);
        cufftSetStream(plan_1d[i], streams[i]);
    }

    gpuErrchk(cudaHostRegister(grid_out, sizeof(cufftDoubleComplex)*grid_size*grid_size, cudaHostRegisterPortable));
    gpuErrchk(cudaHostRegister(image_fits, sizeof(cufftDoubleComplex)*grid_size*grid_size, cudaHostRegisterPortable));

	size_t i,j,k;
	start = clock();
    start_1d2d = clock();
   // Working through columns
	for(j=0;j<grid_size; j+=num_streams*batch_size){

        for (k = 0; k < num_streams; ++k)
        	cufft_stream_1stage(
        		idata_1d_all,
        		odata_1d_all,
        		grid_out,
        		image_fits,
        		grid_size,
        		batch_size,
        		plan_1d,
        		streams,
        		k,
        		j
        		);

	    for(i = 0; i < num_streams; i++) gpuErrchk(cudaStreamSynchronize(streams[i]));
	}

	cudaDeviceSynchronize();
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000;
	SDP_LOG_INFO("Rows 1D FFT took %f ms", cpu_time_used);

	start = clock();
	// Corner rotate (transpose)
	transpose_inplace_block(image_fits, grid_size, (int64_t) block_size);
	//transpose_inplace(image_fits, grid_size);
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000;
	SDP_LOG_INFO("Corner rotate took %f ms", cpu_time_used);


	start = clock();
    // Working through rows
	for(j=0;j<grid_size; j+=num_streams*batch_size){

        for (k = 0; k < num_streams; ++k)
        	cufft_stream_2stage(
        		idata_1d_all,
        		odata_1d_all,
        		image_fits,
        		grid_size,
        		batch_size,
        		plan_1d,
        		streams,
        		k,
        		j
        		);

	    for(i = 0; i < num_streams; i++) gpuErrchk(cudaStreamSynchronize(streams[i]));
	}
    cudaDeviceSynchronize();
	end = clock();
    end_1d2d = clock();

	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000;
	SDP_LOG_INFO("Columns 1D FFT took %f ms", cpu_time_used);

    gpuErrchk(cudaHostUnregister(grid_out));
    gpuErrchk(cudaHostUnregister(image_fits));

	cpu_time_used = ((double) (end_1d2d - start_1d2d)) / CLOCKS_PER_SEC * 1000;
	SDP_LOG_INFO("1D-2D FFT took in total %f ms", cpu_time_used);


	SDP_LOG_INFO("cufftExecZ2Z finished %s", cufftStatus);
    fftshift(image_fits, (int)grid_size, (int)grid_size);
	SDP_LOG_INFO("fftshift finished %s", cufftStatus);


    // Output into the FITS file
    const char filename_img1D[] = "!image_sim_1d2d.fits";

    status_fits = fits_write(image_fits,
            grid_size,
            grid_size_default,
            filename_img1D
    );
    SDP_LOG_INFO("FITSIO status %d", status_fits);

    for(int i = 0; i < num_streams; i++) {
    	gpuErrchk(cudaStreamDestroy(streams[i]));
    	cufftDestroy(plan_1d[i]);
    }

    cudaFree(idata_1d_all);
    cudaFree(odata_1d_all);


    // 2d cuFFT
    if(do_cuFFT2D == 0) {
    sdp_mem_clear_contents(image_out, &status);

    cudaStatus = cudaMalloc((void**)&idata, sizeof(cufftDoubleComplex)*grid_size*grid_size);

    if (cudaStatus != cudaSuccess){
 		fprintf(stderr, "cudaMalloc failed! Can't allocate GPU memory\n");
 		exit(EXIT_FAILURE);
 	}
     SDP_LOG_INFO("cudaMalloc %s", cudaStatus);

     start = clock();

     cudaStatus = cudaMemcpy(idata, (cufftDoubleComplex*)grid_out, sizeof(cufftDoubleComplex)*grid_size * grid_size, cudaMemcpyHostToDevice);
     if (cudaStatus != cudaSuccess){
 		fprintf(stderr, "cudaMemcpy failed! Can't copy to GPU memory\n");
 		exit(EXIT_FAILURE);
 	}

     // Create a 2D FFT plan.
      cufftStatus = cufftPlan2d(&plan, grid_size, grid_size, CUFFT_Z2Z);
  	if (cufftStatus != CUFFT_SUCCESS){
  		fprintf(stderr, "cufftPlan2d failed! Can't create a plan! %s\n", cufftStatus);
  		exit(EXIT_FAILURE);
  	}
  	SDP_LOG_INFO("cufftPlan2d %s", cufftStatus);

      // Inverse transform the grid
  	cufftStatus = cufftExecZ2Z(plan, idata, idata, CUFFT_INVERSE);

	if (cufftStatus != CUFFT_SUCCESS){
		fprintf(stderr, "cufftExecZ2Z failed! Can't make Z2Z transform!\n");
		exit(EXIT_FAILURE);
	}
	SDP_LOG_INFO("cufftExecZ2Z %s ", cufftStatus);

    cudaStatus = cudaMemcpy((cufftDoubleComplex*)(image_fits), idata, sizeof(cufftDoubleComplex)*grid_size * grid_size, cudaMemcpyDeviceToHost);

    if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMemcpy failed! Can't copy from GPU memory\n");
		exit(EXIT_FAILURE);
	}

    // Destroy the CUFFT plan.
    cufftDestroy(plan);

    cudaDeviceSynchronize();
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000;
	SDP_LOG_INFO("2D FFT took %f ms", cpu_time_used);

    fftshift(image_fits, (int)grid_size, (int)grid_size);

	start = clock();
	// Corner rotate (transpose)
	//transpose_inplace_block(image_fits, grid_size, block_size);
	transpose_inplace(image_fits, grid_size);
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000;
	SDP_LOG_INFO("Corner rotate 2D FFT result took %f ms", cpu_time_used);


    // Output into the FITS file
    const char filename_img[] = "!image_sim_cuFFT2D.fits";

    status_fits = fits_write(image_fits,
            grid_size,
            grid_size_default,
            filename_img
    );
    SDP_LOG_INFO("FITSIO status %d", status_fits);

    cudaFree(idata);
    }

    // FFTW3 test
    if(do_FFTW3 == 0) {
    sdp_mem_clear_contents(image_out, &status);

    if(fftw_init_threads() == 0)
    {
    	printf("Error in threds initialisation, exiting...\n");
    	exit(EXIT_FAILURE);
    }
    fftw_plan_with_nthreads(nthreads);

    nthreads = fftw_planner_nthreads();
    SDP_LOG_INFO("Using %d threads", nthreads);

    struct timespec startm, finishm;
    clock_gettime(CLOCK_REALTIME, &startm);
    start = clock();
    int dir = FFTW_BACKWARD;
    fftw_plan p = fftw_plan_dft_2d(grid_size, grid_size, reinterpret_cast<fftw_complex*>(grid_out), reinterpret_cast<fftw_complex*>(image_fits), dir, FFTW_ESTIMATE);
    fftw_execute(p);
    clock_gettime(CLOCK_REALTIME, &finishm);
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000 /nthreads;
	SDP_LOG_INFO("2D FFTW3 took %f ms", cpu_time_used);
	cpu_time_used = (double)(finishm.tv_nsec - startm.tv_nsec)/1000000.0;
	//SDP_LOG_INFO("2D FFTW3 took %f ms", cpu_time_used);

    fftshift(image_fits, (int)grid_size, (int)grid_size);
    SDP_LOG_INFO("fftshift finished");

	start = clock();
	// Corner rotate (transpose)
	//transpose_inplace_block(image_fits, grid_size, block_size);
	transpose_inplace(image_fits, grid_size);

	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000;
	SDP_LOG_INFO("Corner rotate 2D FFTW3 result took %f ms", cpu_time_used);

    // Output into the FITS file
    const char filename_img_FFTW3[] = "!image_sim_FFTW3.fits";

    status_fits = fits_write(image_fits,
            grid_size,
            grid_size_default,
            filename_img_FFTW3
    );
    SDP_LOG_INFO("FITSIO status %d", status_fits);

    fftw_destroy_plan(p);
    fftw_cleanup_threads();
    }

    sdp_mem_free(sources);
    sdp_mem_free(grid_sim);
    sdp_mem_free(image_out);

    cudaDeviceReset();

    return 0;
}

