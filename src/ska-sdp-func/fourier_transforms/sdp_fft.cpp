/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>

#include "ska-sdp-func/fourier_transforms/sdp_fft.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/private_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"

#ifdef SDP_HAVE_CUDA
#include <cufft.h>
#include <cuda_runtime.h>
#endif

using std::complex;

struct sdp_Float2
{
    float x;
    float y;
};

struct sdp_Double2
{
    double x;
    double y;
};


static int is_power_of_two(int n, int* bits)
{
    int cnt = 0;
    (*bits) = -1;
    while (n > 0)
    {
        if ((n & 1) == 1) cnt++;
        n = n >> 1;
        (*bits)++;
    }

    if (cnt == 1)
    {
        return 1;
    }
    return 0;
}


template<typename T>
static void get_twiddle_value(
        int64_t n,
        int64_t m,
        int inverse,
        T* twid
)
{
    double division = ((double) m) / ((double) n);
    if (inverse)
    {
        twid->x = cos( 2.0 * M_PI * division );
        twid->y = sin( 2.0 * M_PI * division );
    }
    else
    {
        twid->x = cos( -2.0 * M_PI * division );
        twid->y = sin( -2.0 * M_PI * division );
    }
}


// This uses Stockham autosort FFT algorithm. More at Computational
// Frameworks for the FFT by Charles van Loan
template<typename T>
static void perform_cpu_fft_power_of_two_inplace(
        T* input,
        T* temp,
        int64_t n,
        int bits,
        int inverse
)
{
    T DFT_value_even, DFT_value_odd, ftemp2, ftemp;
    T twid; // Twiddle factor for use in Fourier transform
    int r = 0, j = 0, k = 0, PoT = 0, PoTm1 = 0;
    int A_index = 0, B_index = 0;
    int Nhalf = n >> 1;

    PoT = 1;
    PoTm1 = 0;
    for (r = 1; r <= bits; r++)
    {
        PoTm1 = PoT;
        PoT = PoT << 1;
        for (int64_t s = 0; s < Nhalf; s++)
        {
            j = s >> (r - 1);
            k = s & (PoTm1 - 1);
            get_twiddle_value(PoT, k, inverse, &twid);
            A_index = j * PoTm1 + k;
            B_index = j * PoTm1 + k + Nhalf;

            ftemp2 = input[B_index];
            ftemp = input[A_index];

            DFT_value_even.x = ftemp.x + twid.x * ftemp2.x - twid.y * ftemp2.y;
            DFT_value_even.y = ftemp.y + twid.x * ftemp2.y + twid.y * ftemp2.x;

            DFT_value_odd.x = ftemp.x - twid.x * ftemp2.x + twid.y * ftemp2.y;
            DFT_value_odd.y = ftemp.y - twid.x * ftemp2.y - twid.y * ftemp2.x;

            temp[j * PoT + k        ] = DFT_value_even;
            temp[j * PoT + k + PoTm1] = DFT_value_odd;
        }
        memcpy(input, temp, n * sizeof(T));
    }
}


template<typename T>
static void perform_cpu_dft_general(
        T* input,
        T* temp,
        int64_t N,
        int inverse
)
{
    for (int64_t i = 0; i < N; i++)
    {
        T sum, DFT_value;
        T twid; // Twiddle factor for use in Fourier Transform
        sum.x = 0; sum.y = 0;
        for (int64_t s = 0; s < N; s++)
        {
            get_twiddle_value(N, i * s, inverse, &twid);
            DFT_value.x = twid.x * input[s].x - twid.y * input[s].y;
            DFT_value.y = twid.x * input[s].y + twid.y * input[s].x;
            sum.x = sum.x + DFT_value.x;
            sum.y = sum.y + DFT_value.y;
        }
        temp[i] = sum;
    }
}


template<typename T>
static void sdp_1d_fft_inplace(
        T* input,
        T* temp,
        int64_t num_x,
        int64_t batch_size,
        int do_inverse
)
{
    int bits = 0;
    int ispoweroftwo = is_power_of_two(num_x, &bits);

    #pragma omp parallel for
    for (int64_t f = 0; f < batch_size; f++)
    {
        if (ispoweroftwo)
        {
            perform_cpu_fft_power_of_two_inplace(&input[f * num_x],
                    &temp[f * num_x], num_x, bits, do_inverse
            );
        }
        else
        {
            perform_cpu_dft_general(&input[f * num_x],
                    &temp[f * num_x], num_x, do_inverse
            );
            memcpy(&input[f * num_x], &temp[f * num_x], num_x * sizeof(T));
        }
    }
}


template<typename T>
static void sdp_1d_fft(
        T* output,
        T* input,
        T* temp,
        int64_t num_x,
        int64_t batch_size,
        int do_inverse
)
{
    if (input != output)
    { // out-of-place transform
        memcpy(output, input, num_x * batch_size * sizeof(T));
    }
    sdp_1d_fft_inplace(output, temp, num_x, batch_size, do_inverse);
}


// This could be a processing function...
template<typename T>
static void sdp_transpose_simple(
        T* out,
        T* in,
        int64_t num_x,
        int64_t num_y
)
{
    for (int64_t i = 0; i < num_y; i++)
    {
        for (int64_t j = 0; j < num_x; j++)
        {
            out[j * num_y + i] = in[i * num_x + j];
        }
    }
}


template<typename T>
static void sdp_2d_fft_inplace(
        T* input,
        T* temp,
        int64_t num_x,
        int64_t num_y,
        int64_t batch_size,
        int do_inverse
)
{
    for (int64_t f = 0; f < batch_size; f++)
    {
        int64_t pos = num_x * num_y * f;
        // Apply FFT on columns
        sdp_transpose_simple(&temp[pos], &input[pos], num_y, num_x);
        sdp_1d_fft_inplace(&temp[pos], &input[pos], num_x, num_y, do_inverse);

        // Apply FFT on Rows
        sdp_transpose_simple(&input[pos], &temp[pos], num_x, num_y);
        sdp_1d_fft_inplace(&input[pos], &temp[pos], num_y, num_x, do_inverse);
    }
}


// Performs 2D FFT with num_x fastest moving dimension and num_y slow moving dimension
template<typename T>
static void sdp_2d_fft(
        T* output,
        T* input,
        T* temp,
        int64_t num_x,
        int64_t num_y,
        int64_t batch_size,
        int do_inverse
)
{
    if (input != output)
    { // out-of-place transform
        memcpy(output, input, num_x * num_y * batch_size * sizeof(T));
    }
    sdp_2d_fft_inplace(output, temp, num_x, num_y, batch_size, do_inverse);
}


struct sdp_Fft
{
    sdp_Mem* input;
    sdp_Mem* output;
    sdp_Mem* temp;
    int num_dims;
    int num_x;
    int num_y;
    int batch_size;
    int is_forward;
    int cufft_plan;
};

struct sdp_Fft_extended
{
    sdp_Mem* input;
    sdp_Mem* output;
    sdp_Mem* temp;
    int num_dims;
    int num_x;
    int num_y;
    int batch_size;
    int num_streams;
    int is_forward;
    int* cufft_plan;
    sdp_CudaStream* cufft_stream;
};

static void check_params(
        const sdp_Mem* input,
        const sdp_Mem* output,
        int32_t num_dims_fft,
        sdp_Error* status
)
{
    if (*status) return;
    if (sdp_mem_is_read_only(output))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Output array is read-only");
        return;
    }
    if (sdp_mem_location(input) != sdp_mem_location(output))
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Input and output arrays must be in the same location");
        return;
    }
    if (sdp_mem_num_dims(input) != sdp_mem_num_dims(output))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Input and output arrays must have the same "
                "number of dimensions"
        );
        return;
    }
    if (!sdp_mem_is_c_contiguous(input) || !sdp_mem_is_c_contiguous(output))
    {
        // TODO: Remove this restriction.
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("All arrays must be C-contiguous");
        return;
    }
    if (sdp_mem_is_complex(input) && sdp_mem_is_complex(output))
    {
        for (int32_t i = 0; i < sdp_mem_num_dims(input); ++i)
        {
            if (sdp_mem_shape_dim(input, i) != sdp_mem_shape_dim(output, i))
            {
                *status = SDP_ERR_RUNTIME;
                SDP_LOG_ERROR("Inconsistent array dimension sizes");
                return;
            }
        }
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data types");
    }
    if (num_dims_fft != sdp_mem_num_dims(input) &&
            num_dims_fft != sdp_mem_num_dims(input) - 1)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Number of FFT dimensions must be equal to "
                "or one smaller than the number of array dimensions"
        );
        return;
    }
}


sdp_Fft* sdp_fft_create(
        const sdp_Mem* input,
        const sdp_Mem* output,
        int32_t num_dims_fft,
        int32_t is_forward,
        sdp_Error* status
)
{
    sdp_Fft* fft = 0;
    sdp_Mem* temp = NULL;
    int cufft_plan = 0;
    int batch_size = 1;
    int num_x = 0;
    int num_y = 0;
    check_params(input, output, num_dims_fft, status);
    if (*status) return fft;

    const int32_t num_dims = sdp_mem_num_dims(input);
    const int32_t last_dim = num_dims - 1;

    if (sdp_mem_location(input) == SDP_MEM_GPU)
    {
#ifdef SDP_HAVE_CUDA
        int idist = 0, istride = 0, odist = 0, ostride = 0;
        cufftType cufft_type = CUFFT_C2C;
        if (sdp_mem_type(input) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(output) == SDP_MEM_COMPLEX_DOUBLE)
        {
            cufft_type = CUFFT_Z2Z;
        }
        else if (sdp_mem_type(input) == SDP_MEM_COMPLEX_FLOAT &&
                sdp_mem_type(output) == SDP_MEM_COMPLEX_FLOAT)
        {
            cufft_type = CUFFT_C2C;
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data types");
            return fft;
        }
        int* dim_size = (int*) calloc(num_dims_fft, sizeof(int));
        int* inembed = (int*) calloc(num_dims_fft, sizeof(int));
        int* onembed = (int*) calloc(num_dims_fft, sizeof(int));
        for (int i = 0; i < num_dims_fft; ++i)
        {
            dim_size[i] = sdp_mem_shape_dim(input, i + num_dims - num_dims_fft);

            // Set default values for inembed and onembed to dimension sizes.
            // TODO: This will need to be changed for non-standard strides.
            inembed[i] = dim_size[i];
            onembed[i] = dim_size[i];
        }
        if (num_dims != num_dims_fft)
        {
            batch_size = sdp_mem_shape_dim(input, 0);
        }
        istride = sdp_mem_stride_elements_dim(input, last_dim);
        ostride = sdp_mem_stride_elements_dim(output, last_dim);
        idist = sdp_mem_stride_elements_dim(input, 0);
        odist = sdp_mem_stride_elements_dim(output, 0);
        const cufftResult error = cufftPlanMany(
                &cufft_plan, num_dims_fft, dim_size,
                inembed, istride, idist, onembed, ostride, odist,
                cufft_type, batch_size
        );
        free(onembed);
        free(inembed);
        free(dim_size);
        if (error != CUFFT_SUCCESS)
        {
            *status = SDP_ERR_RUNTIME;
            SDP_LOG_ERROR("cufftPlanMany error (code %d)", error);
            return fft;
        }
#else
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The processing function library was compiled "
                "without CUDA support"
        );
        return fft;
#endif
    }
    else if (sdp_mem_location(input) == SDP_MEM_CPU)
    {
        int64_t num_x_stride = 0, num_y_stride = 0, batch_stride = 0;
        if (num_dims_fft == 1)
        {
            num_x = sdp_mem_shape_dim(input, last_dim);
            num_x_stride = sdp_mem_stride_elements_dim(input, last_dim);
            num_y = 1;
            num_y_stride = num_x;
        }
        if (num_dims_fft == 2)
        {
            num_x = sdp_mem_shape_dim(input, last_dim - 1);
            num_y = sdp_mem_shape_dim(input, last_dim);
            num_x_stride = sdp_mem_stride_elements_dim(input, last_dim);
            num_y_stride = sdp_mem_stride_elements_dim(input, last_dim - 1);
        }
        if (num_dims_fft > 2)
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported FFT dimension");
        }
        batch_stride = num_x * num_y;
        if (num_dims != num_dims_fft)
        {
            batch_size = sdp_mem_shape_dim(input, 0);
            batch_stride = sdp_mem_stride_elements_dim(input, 0);
        }

        if (num_x_stride != 1
                && num_y_stride != num_x
                && batch_stride != num_x * num_y)
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data strides");
        }

        if (!*status)
        {
            int64_t* shape = (int64_t*) calloc(num_dims, sizeof(int64_t));
            for (int f = 0; f < num_dims; f++)
            {
                shape[f] = sdp_mem_shape_dim(input, f);
            }
            temp = sdp_mem_create(
                    sdp_mem_type(input), SDP_MEM_CPU, num_dims, shape, status
            );
            free(shape);
        }
    }
    else
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Unsupported FFT location");
        return fft;
    }
    if (!*status)
    {
        fft = (sdp_Fft*) calloc(1, sizeof(sdp_Fft));
        fft->input = sdp_mem_create_alias(input);
        fft->output = sdp_mem_create_alias(output);
        fft->temp = temp;
        fft->num_dims = num_dims_fft;
        fft->num_x = num_x;
        fft->num_y = num_y;
        fft->batch_size = batch_size;
        fft->is_forward = is_forward;
        fft->cufft_plan = cufft_plan;
    }
    return fft;
}


void sdp_fft_exec(
        sdp_Fft* fft,
        sdp_Mem* input,
        sdp_Mem* output,
        sdp_Error* status
)
{
    if (*status || !fft || !input || !output) return;
    check_params(input, output, fft->num_dims, status);
    if (*status) return;
    if (!sdp_mem_is_matching(fft->input, input, 1) ||
            !sdp_mem_is_matching(fft->output, output, 1))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Arrays do not match those used for FFT plan creation");
        return;
    }
    if (sdp_mem_location(input) == SDP_MEM_GPU)
    {
#ifdef SDP_HAVE_CUDA
        cufftResult error = CUFFT_SUCCESS;
        const int fft_dir = fft->is_forward ? CUFFT_FORWARD : CUFFT_INVERSE;
        if (sdp_mem_type(input) == SDP_MEM_COMPLEX_FLOAT &&
                sdp_mem_type(output) == SDP_MEM_COMPLEX_FLOAT)
        {
            error = cufftExecC2C(fft->cufft_plan,
                    (cufftComplex*)sdp_mem_data(input),
                    (cufftComplex*)sdp_mem_data(output), fft_dir
            );
        }
        else if (sdp_mem_type(input) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(output) == SDP_MEM_COMPLEX_DOUBLE)
        {
            error = cufftExecZ2Z(fft->cufft_plan,
                    (cufftDoubleComplex*)sdp_mem_data(input),
                    (cufftDoubleComplex*)sdp_mem_data(output), fft_dir
            );
        }
        if (error != CUFFT_SUCCESS)
        {
            *status = SDP_ERR_RUNTIME;
            SDP_LOG_ERROR("cufftExec error (code %d)", error);
        }
#else
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The processing function library was compiled "
                "without CUDA support"
        );
#endif
    }
    else if (sdp_mem_location(input) == SDP_MEM_CPU)
    {
        int do_inverse = (fft->is_forward == 1 ? 0 : 1);

        if (fft->num_dims == 1)
        {
            if (sdp_mem_type(input) == SDP_MEM_COMPLEX_FLOAT
                    && sdp_mem_type(output) == SDP_MEM_COMPLEX_FLOAT)
            {
                sdp_1d_fft(
                        (sdp_Float2*) sdp_mem_data(output),
                        (sdp_Float2*) sdp_mem_data(input),
                        (sdp_Float2*) sdp_mem_data(fft->temp),
                        fft->num_x,
                        fft->batch_size,
                        do_inverse
                );
            }
            if (sdp_mem_type(input) == SDP_MEM_COMPLEX_DOUBLE
                    && sdp_mem_type(output) == SDP_MEM_COMPLEX_DOUBLE)
            {
                sdp_1d_fft(
                        (sdp_Double2*) sdp_mem_data(output),
                        (sdp_Double2*) sdp_mem_data(input),
                        (sdp_Double2*) sdp_mem_data(fft->temp),
                        fft->num_x,
                        fft->batch_size,
                        do_inverse
                );
            }
        }
        if (fft->num_dims == 2)
        {
            if (sdp_mem_type(input) == SDP_MEM_COMPLEX_FLOAT
                    && sdp_mem_type(output) == SDP_MEM_COMPLEX_FLOAT)
            {
                sdp_2d_fft(
                        (sdp_Float2*) sdp_mem_data(output),
                        (sdp_Float2*) sdp_mem_data(input),
                        (sdp_Float2*) sdp_mem_data(fft->temp),
                        fft->num_x,
                        fft->num_y,
                        fft->batch_size,
                        do_inverse
                );
            }
            if (sdp_mem_type(input) == SDP_MEM_COMPLEX_DOUBLE
                    && sdp_mem_type(output) == SDP_MEM_COMPLEX_DOUBLE)
            {
                sdp_2d_fft(
                        (sdp_Double2*) sdp_mem_data(output),
                        (sdp_Double2*) sdp_mem_data(input),
                        (sdp_Double2*) sdp_mem_data(fft->temp),
                        fft->num_x,
                        fft->num_y,
                        fft->batch_size,
                        do_inverse
                );
            }
        }
        if (fft->num_dims > 2)
        {
            *status = SDP_ERR_INVALID_ARGUMENT;
            SDP_LOG_ERROR("3D FFT is not supported on host memory.");
            return;
        }
    }
}


void sdp_fft_free(sdp_Fft* fft)
{
    if (!fft) return;
#ifdef SDP_HAVE_CUDA
    if (sdp_mem_location(fft->input) == SDP_MEM_GPU)
    {
        cufftDestroy(fft->cufft_plan);
    }
#endif
    sdp_mem_ref_dec(fft->input);
    sdp_mem_ref_dec(fft->output);
    if (fft->temp != NULL)
    {
        sdp_mem_free(fft->temp);
    }
    free(fft);
}

sdp_Fft_extended* sdp_fft_extended_create(
        const sdp_Mem* input,  // idata_1d_all, created outside
        const sdp_Mem* output, // odata_1d_all, created outside
        int32_t num_dims_fft,  // 1
        int32_t is_forward,    // 0 or 1
		int32_t num_streams,   // number of streams
		int32_t batch_size,    // batch size (default should be 1024)
        sdp_Error* status
)
{
    sdp_Fft_extended* fft = 0;
    sdp_Mem* temp = NULL;
    sdp_CudaStream* streams = NULL;
    int* plan_1d = NULL;

    int num_x = 0;
    int num_y = 0;
    check_params(input, output, num_dims_fft, status);
    if (*status) return fft;

    const int32_t num_dims = sdp_mem_num_dims(input);
    const int32_t last_dim = num_dims - 1;

    if (sdp_mem_location(input) == SDP_MEM_GPU)
    {
#ifdef SDP_HAVE_CUDA
    	// Create CUDA streams
        cudaError_t	cudaStatus;
        cufftResult cufftStatus;
        //size_t i,j,k;

        streams = (sdp_CudaStream*) malloc(sizeof(sdp_CudaStream)*num_streams);

        for (int i = 0; i < num_streams; i++) {
        	cudaStatus = cudaStreamCreate(&streams[i].stream);
            if (cudaStatus != cudaSuccess)
            {
            	SDP_LOG_ERROR("cudaStreamCreate failed! Can't create a stream %d", i);
            	*status=SDP_ERR_MEM_ALLOC_FAILURE;
            	return fft;
            }
        }

        cufftType cufft_type = CUFFT_C2C;
        if (sdp_mem_type(input) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(output) == SDP_MEM_COMPLEX_DOUBLE)
        {
            cufft_type = CUFFT_Z2Z;
        }
        else if (sdp_mem_type(input) == SDP_MEM_COMPLEX_FLOAT &&
                sdp_mem_type(output) == SDP_MEM_COMPLEX_FLOAT)
        {
            cufft_type = CUFFT_C2C;
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data types");
            return fft;
        }

        // Find out grid_size
        int grid_size = sdp_mem_num_elements(input);
        grid_size /= (batch_size*num_streams);

        int rank = 1;                           // --- 1D FFTs
        int n[] = { (int)grid_size };           // --- Size of the Fourier transform
        int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
        int idist = grid_size, odist = grid_size; // --- Distance between batches
        int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
        int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
        int batch = batch_size;                 // --- Number of batched executions

        // Allocate cuFFT plans
        plan_1d = (int*) malloc(sizeof(int)*num_streams);

        // --- Creates cuFFT plans and sets them in streams
         for (int i = 0; i < num_streams; i++) {
            cufftStatus = cufftPlanMany(&plan_1d[i], rank, n,
                          inembed, istride, idist,
                          onembed, ostride, odist, cufft_type, batch);
        	if (cufftStatus != CUFFT_SUCCESS){
        		SDP_LOG_ERROR("cufftPlanMany error (code %d) in plan %d", cufftStatus, i);
        		*status = SDP_ERR_RUNTIME;
        		return fft;
        	}
        	SDP_LOG_INFO("cufftPlanMany %d %s", i, cufftStatus);
        	cufftStatus = cufftSetStream(plan_1d[i], streams[i].stream);
        	if (cufftStatus != CUFFT_SUCCESS){
        		SDP_LOG_ERROR("cufftSetStream error (code %d) in stream %d", cufftStatus, i);
        		*status = SDP_ERR_RUNTIME;
        		return fft;
        	}
        }
#else
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The processing function library was compiled "
                "without CUDA support"
        );
        return fft;
#endif
    }
    else if (sdp_mem_location(input) == SDP_MEM_CPU)
    {
        int64_t num_x_stride = 0, num_y_stride = 0, batch_stride = 0;
        if (num_dims_fft == 1)
        {
            num_x = sdp_mem_shape_dim(input, last_dim);
            num_x_stride = sdp_mem_stride_elements_dim(input, last_dim);
            num_y = 1;
            num_y_stride = num_x;
        }
        if (num_dims_fft == 2)
        {
            num_x = sdp_mem_shape_dim(input, last_dim - 1);
            num_y = sdp_mem_shape_dim(input, last_dim);
            num_x_stride = sdp_mem_stride_elements_dim(input, last_dim);
            num_y_stride = sdp_mem_stride_elements_dim(input, last_dim - 1);
        }
        if (num_dims_fft > 2)
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported FFT dimension");
        }
        batch_stride = num_x * num_y;
        if (num_dims != num_dims_fft)
        {
            batch_size = sdp_mem_shape_dim(input, 0);
            batch_stride = sdp_mem_stride_elements_dim(input, 0);
        }

        if (num_x_stride != 1
                && num_y_stride != num_x
                && batch_stride != num_x * num_y)
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data strides");
        }

        if (!*status)
        {
            int64_t* shape = (int64_t*) calloc(num_dims, sizeof(int64_t));
            for (int f = 0; f < num_dims; f++)
            {
                shape[f] = sdp_mem_shape_dim(input, f);
            }
            temp = sdp_mem_create(
                    sdp_mem_type(input), SDP_MEM_CPU, num_dims, shape, status
            );
            free(shape);
        }
    }
    else
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Unsupported FFT location");
        return fft;
    }
    if (!*status)
    {
        fft = (sdp_Fft_extended*) calloc(1, sizeof(sdp_Fft_extended));
        fft->input = sdp_mem_create_alias(input);
        fft->output = sdp_mem_create_alias(output);
        fft->temp = temp;
        fft->num_dims = num_dims_fft;
        fft->num_x = num_x;
        fft->num_y = num_y;
        fft->batch_size = batch_size;
        fft->num_streams = num_streams;
        fft->is_forward = is_forward;
        fft->cufft_plan = plan_1d;
        fft->cufft_stream = streams;
    }
    return fft;
}

void sdp_fft_extended_free(
		sdp_Fft_extended* fft,
		sdp_Error* status)
{
    if (!fft) return;
#ifdef SDP_HAVE_CUDA
    cudaError_t	cudaStatus;
    cufftResult cufftStatus;
    if (sdp_mem_location(fft->input) == SDP_MEM_GPU)
    {
        for(int i = 0; i < fft->num_streams; i++)
        {
        	cudaStatus = cudaStreamDestroy(fft->cufft_stream[i].stream);
            if (cudaStatus != cudaSuccess)
            {
            	SDP_LOG_ERROR("cudaStreamDestroy failed! Can't can't destroy stream %d", i);
            	*status=SDP_ERR_MEM_ALLOC_FAILURE;
            	return;
            }

            cufftStatus= cufftDestroy(fft->cufft_plan[i]);
            if (cufftStatus != CUFFT_SUCCESS)
            {
            	SDP_LOG_ERROR("cufftDestroy failed! Can't destroy a plan %d", fft->cufft_plan[i]);
            	*status = SDP_ERR_RUNTIME;
            	return;
            }
        }
    }
#endif
    sdp_mem_ref_dec(fft->input);
    sdp_mem_ref_dec(fft->output);
    if (fft->temp != NULL)
    {
        sdp_mem_free(fft->temp);
    }
    free(fft);
}

template<typename T>
void fft_phase(int num_x, int num_y, complex<T>* data)
{
    #pragma omp parallel for collapse(2)
    for (int iy = 0; iy < num_y; ++iy)
        for (int ix = 0; ix < num_x; ++ix)
            data[iy * num_x + ix] *= (T) (1 - (((ix + iy) & 1) << 1));
}


void sdp_fft_phase(sdp_Mem* data, sdp_Error* status)
{
    if (*status || !data) return;
    const int num_y = (int) sdp_mem_shape_dim(data, 0);
    const int num_x = (int) sdp_mem_num_dims(data) > 1 ?
                sdp_mem_shape_dim(data, 1) : 1;
    if (sdp_mem_location(data) == SDP_MEM_CPU)
    {
        if (sdp_mem_type(data) == SDP_MEM_COMPLEX_DOUBLE)
        {
            fft_phase(num_x, num_y, (complex<double>*) sdp_mem_data(data));
        }
        else if (sdp_mem_type(data) == SDP_MEM_COMPLEX_FLOAT)
        {
            fft_phase(num_x, num_y, (complex<float>*) sdp_mem_data(data));
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
        }
    }
    else if (sdp_mem_location(data) == SDP_MEM_GPU)
    {
        size_t num_threads[] = {32, 8, 1}, num_blocks[] = {1, 1, 1};
        const char* kernel_name = 0;
        if (sdp_mem_type(data) == SDP_MEM_COMPLEX_FLOAT)
        {
            kernel_name = "fft_phase<float>";
        }
        else if (sdp_mem_type(data) == SDP_MEM_COMPLEX_DOUBLE)
        {
            kernel_name = "fft_phase<double>";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            return;
        }
        if (num_x == 1)
        {
            num_threads[0] = 1;
            num_threads[1] = 256;
        }
        if (num_y == 1)
        {
            num_threads[0] = 256;
            num_threads[1] = 1;
        }
        const void* arg[] = {&num_x, &num_y, sdp_mem_gpu_buffer(data, status)};
        num_blocks[0] = (num_x + num_threads[0] - 1) / num_threads[0];
        num_blocks[1] = (num_y + num_threads[1] - 1) / num_threads[1];
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, 0, 0, arg, status
        );
    }
    else
    {
        *status = SDP_ERR_MEM_LOCATION;
    }
}

// FFT phase shift implementation
template<typename T>
void sdp_fftshift(T* x,
			int64_t m,
			int64_t n
)
{
//	int m, n;      // FFT row and column dimensions might be different
	int64_t m2, n2;
	int64_t i, k;
	int64_t idx,idx1, idx2, idx3;
	T tmp13, tmp24;

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

// Naively transpose a square matrix
template<typename T>
void sdp_transpose_inplace_simple(
		T *inout,
		int64_t n
)
{
	int64_t i, j;
    T temp;
    for (i = 0; i < n; i++){
        for (j = i + 1; j < n; j++){
            temp = inout[i * n + j];
            inout[i * n + j] = inout[j * n + i];
            inout[j * n + i] = temp;
        }
    }
}


template<typename T>
void sdp_transpose_block(
		T *inout,
		int64_t n,
		int64_t istart,
		int64_t jstart,
		int64_t block
)
{
	int64_t i, j;
    T temp;
    for (i = istart; i < istart+block; i++){
        for (j = jstart; j < jstart+block; j++){
            temp = inout[i * n + j];
            inout[i * n + j] = inout[j * n + i];
            inout[j * n + i] = temp;
        }
    }
}

template<typename T>
void sdp_transpose_block_diag(
		T* inout,
		int64_t n,
		int64_t istart,
		int64_t jstart,
		int64_t block
)
{
	int64_t i, j;
    T temp;
    for (i = istart; i < istart+block; i++){
        for (j = i + 1; j < jstart+block; j++){
            temp = inout[i * n + j];
            inout[i * n + j] = inout[j * n + i];
            inout[j * n + i] = temp;
        }
    }
}

// Block transpose inplace accelerated function
template<typename T>
void sdp_transpose_inplace_accelerated(
    T* inout,
    int64_t n,
	int64_t block
)
{
	int64_t i, j;
	int64_t r = n % block, m = n - r;
    T temp;
    // if dimension of matrix is less than block size just do naive
    if (n < block){
        return sdp_transpose_inplace_simple(inout, n);
    }
    // transpose square blocks
#   pragma omp parallel for collapse(1)
    for (i = 0; i < m; i += block){
        for (j = i; j < m; j += block){
            (i == j) ? sdp_transpose_block_diag(inout, n, i, j, block) : sdp_transpose_block(inout, n, i, j, block);

        }
    }

    // take care of the remaining swaps naively
    if (r){
        // transpose rectangular sub-matrix
        for (j = m; j < n; j++){
            for (i = 0; i < m; i++){
                temp = inout[i * n + j];
                inout[i * n + j] = inout[j * n + i];
                inout[j * n + i] = temp;
            }
        }
        // transpose square sub matrix in "bottom right"
        for (i = m; i < n; i++){
            for (j = i + 1; j < n; j++){
                temp = inout[i * n + j];
                inout[i * n + j] = inout[j * n + i];
                inout[j * n + i] = temp;
            }
        }
    }
}

#ifdef SDP_HAVE_CUDA
// 1D batched cuFFT initialisation with streams
void sdp_1d_cufft_init(
		cufftHandle* plan_1d,
		cudaStream_t* streams,
		cufftType cufft_type,
		int num_streams,
		int grid_size,
		int batch_size,
		sdp_Error* status
)
{
    cudaError_t	cudaStatus;
    cufftResult cufftStatus;

    for (int i = 0; i < num_streams; i++) {
    	cudaStatus = cudaStreamCreate(&streams[i]);
        if (cudaStatus != cudaSuccess)
        {
        	SDP_LOG_ERROR("cudaStreamCreate failed! Can't create a stream %d", i);
        	*status=SDP_ERR_MEM_ALLOC_FAILURE;
        	return;
        }
    }

    int rank = 1;                           // --- 1D FFTs
    int n[] = { (int)grid_size };           // --- Size of the Fourier transform
    int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
    int idist = grid_size, odist = grid_size; // --- Distance between batches
    int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
    int batch = batch_size;                 // --- Number of batched executions

    // --- Creates cuFFT plans and sets them in streams
     for (int i = 0; i < num_streams; i++) {
        cufftStatus = cufftPlanMany(&plan_1d[i], rank, n,
                      inembed, istride, idist,
                      onembed, ostride, odist, cufft_type, batch);
    	if (cufftStatus != CUFFT_SUCCESS){
    		SDP_LOG_ERROR("cufftPlan1d failed! Can't create a plan! %s", cufftStatus);
    		*status = SDP_ERR_RUNTIME;
    		return;
    	}
    	SDP_LOG_INFO("cufftPlanMany %d %s", i, cufftStatus);
    	cufftStatus = cufftSetStream(plan_1d[i], streams[i]);
    	if (cufftStatus != CUFFT_SUCCESS){
    		SDP_LOG_ERROR("cufftSetStream failed! Can't set a stream! %s", cufftStatus);
    		*status = SDP_ERR_RUNTIME;
    		return;
    	}
    }
    SDP_LOG_INFO("sdp_1d_cufft_init completed");
}

// 1D cuFFT with streams
template <typename cudaT, typename T>
void sdp_1d_cufft_accelerated(
		cudaT *idata_1d_all,
		cudaT *odata_1d_all,
		T* image_fits,
		int64_t grid_size,
		int64_t batch_size,
		cufftHandle* plan_1d,
		cudaStream_t* streams,
		int stream_number,
		size_t j,
		int do_inverse,
		sdp_Error* status
)
{
	cudaError_t	cudaStatus;
	cufftResult cufftStatus;
	int64_t idx_d, idx_h;

	idx_d = stream_number*batch_size*grid_size;
	idx_h = (int64_t)(((int64_t)j+(int64_t)stream_number*batch_size)*grid_size);
	cudaStatus = cudaMemcpyAsync((idata_1d_all+idx_d), (cudaT*)(image_fits + idx_h), sizeof(cudaT)*grid_size*batch_size, cudaMemcpyHostToDevice, streams[stream_number]);
    if (cudaStatus != cudaSuccess){
    	SDP_LOG_ERROR("cudaMemcpy failed! Can't copy to GPU memory");
		*status = SDP_ERR_MEM_COPY_FAILURE;
		return;
	}

    if(sizeof(cudaT) == 16){
    	SDP_LOG_INFO("Stream %d, plan %d, cufftExecZ2Z", stream_number, plan_1d[stream_number]);
    	cufftStatus = cufftExecZ2Z(plan_1d[stream_number], (cufftDoubleComplex*)(idata_1d_all+idx_d), (cufftDoubleComplex*)(odata_1d_all+idx_d), do_inverse);
    }
    else if(sizeof(cudaT) == 8){
     	SDP_LOG_INFO("Stream %d, plan %d, cufftExecC2C", stream_number, plan_1d[stream_number]);
   	cufftStatus = cufftExecC2C(plan_1d[stream_number], (cufftComplex*)(idata_1d_all+idx_d), (cufftComplex*)(odata_1d_all+idx_d), do_inverse);
    }
    else {
    	SDP_LOG_ERROR("Wrong size of the grid array elements, should be 8 (complex) or 16 (double complex), having %d", sizeof(cudaT));
    	*status = SDP_ERR_INVALID_ARGUMENT;
    	return;
    }
	if (cufftStatus != CUFFT_SUCCESS){
		SDP_LOG_ERROR("cufftExecZ2Z/C2C failed! Can't make Z2Z/C2C transform!");
		*status = SDP_ERR_RUNTIME;
		return;
	}
	cudaStatus = cudaMemcpyAsync((cudaT*)(image_fits + idx_h), (odata_1d_all+idx_d), sizeof(cudaT)*grid_size*batch_size, cudaMemcpyDeviceToHost, streams[stream_number]);

    if (cudaStatus != cudaSuccess){
		SDP_LOG_ERROR("cudaMemcpy failed! Can't copy from GPU memory");
		*status = SDP_ERR_MEM_COPY_FAILURE;
		return;
	}

}

/*
 * 2D inplace FFT arranged as a series of 1D FFTs
 *
 * Scratch GPU arrays idata_1d_all and odata_1d_all should be allocated externally
 *
 *   cufftDoubleComplex *idata_1d_all, *odata_1d_all;
 *   cudaStatus = cudaMalloc((void**)&odata_1d_all, sizeof(cufftDoubleComplex)*grid_size*batch_size*num_streams);
 *   cudaStatus = cudaMalloc((void**)&idata_1d_all, sizeof(cufftDoubleComplex)*grid_size*batch_size*num_streams);
 *
 */
template <typename cudaT, typename T>
void sdp_2d_cufft_accelerated(
		T* inout,
		cudaT* idata_1d_all,
		cudaT* odata_1d_all,
		cufftType cufft_type,
		int64_t grid_size,
		int num_streams,
		int batch_size,
		int64_t block,
		int do_inverse,
		sdp_Error* status
		)
{
    cudaError_t	cudaStatus;
    cufftResult cufftStatus;
    cudaStream_t* streams;
    cufftHandle* plan_1d;

    size_t i,j,k;

	cudaStatus = cudaHostRegister(inout, sizeof(cudaT)*grid_size*grid_size, cudaHostRegisterPortable);
    if (cudaStatus != cudaSuccess)
    {
    	SDP_LOG_ERROR("cudaHostRegister failed! Can't pin a memory! %s", cudaStatus);
    	*status=SDP_ERR_MEM_ALLOC_FAILURE;
    	return;
    }

    streams = (cudaStream_t*) malloc(sizeof(cudaStream_t)*num_streams);
    plan_1d = (cufftHandle*) malloc(sizeof(cufftHandle)*num_streams);
    sdp_1d_cufft_init(
    		plan_1d,
    		streams,
			cufft_type,
    		num_streams,
    		grid_size,
    		batch_size,
			status);

	// Corner rotate (transpose)
	//sdp_transpose_inplace_accelerated(inout, grid_size, (int64_t) block);

    // Working through columns
    for(j=0;j<grid_size; j+=num_streams*batch_size)
	{
        for (k = 0; k < num_streams; ++k)
        	sdp_1d_cufft_accelerated(
        		idata_1d_all,
        		odata_1d_all,
        		inout,
        		grid_size,
        		batch_size,
        		plan_1d,
        		streams,
        		k,
        		j,
				do_inverse,
				status
        		);
	    for(i = 0; i < num_streams; i++)
	    {
	    	cudaStatus = cudaStreamSynchronize(streams[i]);
	        if (cudaStatus != cudaSuccess)
	        {
	        	SDP_LOG_ERROR("cudaStreamSynchronize in columns failed! Can't synchronize stream %d", i);
	        	*status = SDP_ERR_RUNTIME;
	        	return;
	        }
	    }
	}
	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
    	SDP_LOG_ERROR("cudaDeviceSynchronize failed! Can't synchronize the device");
    	*status = SDP_ERR_RUNTIME;
    	return;
    }

	// Corner rotate (transpose)
	sdp_transpose_inplace_accelerated(inout, grid_size, (int64_t) block);

    // Working through rows
	for(j=0;j<grid_size; j+=num_streams*batch_size)
	{
        for (k = 0; k < num_streams; ++k)
        	sdp_1d_cufft_accelerated(
        		idata_1d_all,
        		odata_1d_all,
        		inout,
        		grid_size,
        		batch_size,
        		plan_1d,
        		streams,
        		k,
        		j,
				do_inverse,
				status
        		);

        for(i = 0; i < num_streams; i++)
	    {
	    	cudaStatus = cudaStreamSynchronize(streams[i]);
	        if (cudaStatus != cudaSuccess)
	        {
	        	SDP_LOG_ERROR("cudaStreamSynchronize in rows failed! Can't synchronize stream %d", i);
	        	*status = SDP_ERR_RUNTIME;
	        	return;
	        }
	    }
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
	    SDP_LOG_ERROR("cudaDeviceSynchronize failed! Can't synchronize the device %d", i);
	    *status = SDP_ERR_RUNTIME;
	    return;
	}
	SDP_LOG_INFO("cufftExec finished");

    for(int i = 0; i < num_streams; i++)
    {
    	cudaStatus = cudaStreamDestroy(streams[i]);
        if (cudaStatus != cudaSuccess)
        {
        	SDP_LOG_ERROR("cudaStreamDestroy failed! Can't can't destroy stream %d", i);
        	*status=SDP_ERR_MEM_ALLOC_FAILURE;
        	return;
        }

        cufftStatus= cufftDestroy(plan_1d[i]);
        if (cufftStatus != CUFFT_SUCCESS)
        {
        	SDP_LOG_ERROR("cufftDestroy failed! Can't destroy a plan %d", i);
        	*status = SDP_ERR_RUNTIME;
        	return;
        }
    }

    cudaStatus = cudaHostUnregister(inout);
    if (cudaStatus != cudaSuccess)
    {
    	SDP_LOG_ERROR("cudaHostUnregister failed! Can't unregister a memory! %s", cudaStatus);
    	*status=SDP_ERR_MEM_ALLOC_FAILURE;
    	return;
    }

}
#endif

void sdp_fft_extended_exec(
        sdp_Fft_extended* fft,
        sdp_Mem* input,
        sdp_Mem* output,
        sdp_Error* status
)
{
    if (*status || !fft || !input || !output) return;
    //check_params(input, output, fft->num_dims, status);
    //if (*status) return;
/*
    if (!sdp_mem_is_matching(fft->input, input, 1) ||
            !sdp_mem_is_matching(fft->output, output, 1))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Arrays do not match those used for FFT plan creation");
        return;
    }
*/
    if (sdp_mem_location(fft->input) == SDP_MEM_GPU)
    {
#ifdef SDP_HAVE_CUDA
        cufftResult error = CUFFT_SUCCESS;
        cudaError_t	cudaStatus;
        //cufftResult cufftStatus;

        int i,j,k;
        int64_t block_size = 8;
        int do_inverse = (fft->is_forward == 1 ? 0 : 1);

        // Find out grid_size
        int64_t grid_size = sdp_mem_shape_dim(input, 0);

        // Copy content from input to output
        int64_t num_elements = sdp_mem_num_elements(input);
        sdp_mem_copy_contents(
                output,
                input,
                0,
                0,
                num_elements,
                status
        );

        //void* inout = sdp_mem_data(output);

        if (sdp_mem_type(input) == SDP_MEM_COMPLEX_FLOAT &&
                sdp_mem_type(output) == SDP_MEM_COMPLEX_FLOAT)
        {
          	SDP_LOG_INFO("Doing FFT for SDP_MEM_COMPLEX_FLOAT");
          	cudaStatus = cudaHostRegister((sdp_Float2*) sdp_mem_data(output), sizeof(sdp_Float2)*grid_size*grid_size, cudaHostRegisterPortable);
              if (cudaStatus != cudaSuccess)
              {
                  	SDP_LOG_ERROR("cudaHostRegister failed! Can't pin a memory! %d", cudaStatus);
                   	*status=SDP_ERR_MEM_ALLOC_FAILURE;
                   	return;
              }
        	// Corner rotate (transpose)
        	sdp_transpose_inplace_accelerated((sdp_Float2*) sdp_mem_data(output), grid_size, block_size);
            // Working through columns
            for(j=0;j<grid_size; j+=fft->num_streams*fft->batch_size)
        	{
                for (k = 0; k < fft->num_streams; ++k)
                	sdp_1d_cufft_accelerated(
                    	(cufftComplex*) sdp_mem_data(fft->input),
    					(cufftComplex*) sdp_mem_data(fft->output),
						(sdp_Float2*) sdp_mem_data(output),
                		grid_size,
                		fft->batch_size,
                		fft->cufft_plan,
						(cudaStream_t*)(fft->cufft_stream),
                		k,
                		j,
        				do_inverse,
        				status
                		);

        	    for(i = 0; i < fft->num_streams; i++)
        	    {
        	    	cudaStatus = cudaStreamSynchronize(fft->cufft_stream[i].stream);
        	        if (cudaStatus != cudaSuccess)
        	        {
        	        	SDP_LOG_ERROR("cudaStreamSynchronize in columns failed! Can't synchronize stream %d", i);
        	        	*status = SDP_ERR_RUNTIME;
        	        	return;
        	        }
        	    }
        	}
        	cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess)
            {
            	SDP_LOG_ERROR("cudaDeviceSynchronize failed! Can't synchronize the device");
            	*status = SDP_ERR_RUNTIME;
            	return;
            }

        	// Corner rotate (transpose)
        	sdp_transpose_inplace_accelerated((sdp_Float2*) sdp_mem_data(output), grid_size, block_size);

            // Working through rows
        	for(j=0;j<grid_size; j+=fft->num_streams*fft->batch_size)
        	{
                for (k = 0; k < fft->num_streams; ++k)
                	sdp_1d_cufft_accelerated(
                        	(cufftComplex*) sdp_mem_data(fft->input),
        					(cufftComplex*) sdp_mem_data(fft->output),
							(sdp_Float2*)sdp_mem_data(output),
                    		grid_size,
                    		fft->batch_size,
                    		fft->cufft_plan,
							(cudaStream_t*)fft->cufft_stream,
                    		k,
                    		j,
            				do_inverse,
            				status
                		);

        	}
            cudaStatus = cudaHostUnregister((sdp_Float2*) sdp_mem_data(output));
            if (cudaStatus != cudaSuccess)
            {
            	SDP_LOG_ERROR("cudaHostUnregister failed! Can't unregister a memory! %d", cudaStatus);
            	*status=SDP_ERR_MEM_ALLOC_FAILURE;
            	return;
            }

        }
        else if (sdp_mem_type(input) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(output) == SDP_MEM_COMPLEX_DOUBLE)
        {
        	SDP_LOG_INFO("Doing FFT for SDP_MEM_COMPLEX_DOUBLE");
        	cudaStatus = cudaHostRegister((sdp_Double2*) sdp_mem_data(output), sizeof(sdp_Double2)*grid_size*grid_size, cudaHostRegisterPortable);
                  if (cudaStatus != cudaSuccess)
                  {
                      	SDP_LOG_ERROR("cudaHostRegister failed! Can't pin a memory! %s", cudaStatus);
                       	*status=SDP_ERR_MEM_ALLOC_FAILURE;
                       	return;
                  }
        	// Corner rotate (transpose)
        	sdp_transpose_inplace_accelerated((sdp_Double2*) sdp_mem_data(output), grid_size, block_size);
            // Working through columns
            for(j=0;j<grid_size; j+=fft->num_streams*fft->batch_size)
        	{
                for (k = 0; k < fft->num_streams; ++k)
                	sdp_1d_cufft_accelerated(
                		(cufftDoubleComplex*) sdp_mem_data(fft->input),
						(cufftDoubleComplex*) sdp_mem_data(fft->output),
						(sdp_Double2*) sdp_mem_data(output),
                		grid_size,
                		fft->batch_size,
                		fft->cufft_plan,
						(cudaStream_t*)fft->cufft_stream,
                		k,
                		j,
        				do_inverse,
        				status
                		);

        	    for(i = 0; i < fft->num_streams; i++)
        	    {
        	    	cudaStatus = cudaStreamSynchronize(fft->cufft_stream[i].stream);
        	        if (cudaStatus != cudaSuccess)
        	        {
        	        	SDP_LOG_ERROR("cudaStreamSynchronize in columns failed! Can't synchronize stream %d", i);
        	        	*status = SDP_ERR_RUNTIME;
        	        	return;
        	        }
        	    }
        	}
        	cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess)
            {
            	SDP_LOG_ERROR("cudaDeviceSynchronize failed! Can't synchronize the device");
            	*status = SDP_ERR_RUNTIME;
            	return;
            }

        	// Corner rotate (transpose)
        	sdp_transpose_inplace_accelerated((sdp_Double2*) sdp_mem_data(output), grid_size, block_size);

            // Working through rows
        	for(j=0;j<grid_size; j+=fft->num_streams*fft->batch_size)
        	{
                for (k = 0; k < fft->num_streams; ++k)
                	sdp_1d_cufft_accelerated(
                    		(cufftDoubleComplex*) sdp_mem_data(fft->input),
    						(cufftDoubleComplex*) sdp_mem_data(fft->output),
							(sdp_Double2*)sdp_mem_data(output),
                    		grid_size,
                    		fft->batch_size,
                    		fft->cufft_plan,
							(cudaStream_t*)fft->cufft_stream,
                    		k,
                    		j,
            				do_inverse,
            				status
                		);

        	}
            cudaStatus = cudaHostUnregister((sdp_Double2*) sdp_mem_data(output));
            if (cudaStatus != cudaSuccess)
            {
            	SDP_LOG_ERROR("cudaHostUnregister failed! Can't unregister a memory! %d", cudaStatus);
            	*status=SDP_ERR_MEM_ALLOC_FAILURE;
            	return;
            }

        }
        else
        {
            *status = SDP_ERR_RUNTIME;
            SDP_LOG_ERROR("Inconsistent input/output data type", error);
        }


    	cudaStatus = cudaDeviceSynchronize();
    	if (cudaStatus != cudaSuccess)
    	{
    	    SDP_LOG_ERROR("cudaDeviceSynchronize failed! Can't synchronize the device %d", 0);
    	    *status = SDP_ERR_RUNTIME;
    	    return;
    	}
    	SDP_LOG_INFO("cufftExec finished");

#else
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The processing function library was compiled "
                "without CUDA support"
        );
#endif
    }
    else if (sdp_mem_location(input) == SDP_MEM_CPU)
    {
        int do_inverse = (fft->is_forward == 1 ? 0 : 1);


        if (fft->num_dims == 1)
        {
            if (sdp_mem_type(input) == SDP_MEM_COMPLEX_FLOAT
                    && sdp_mem_type(output) == SDP_MEM_COMPLEX_FLOAT)
            {
            	sdp_1d_fft(
                        (sdp_Float2*) sdp_mem_data(output),
                        (sdp_Float2*) sdp_mem_data(input),
                        (sdp_Float2*) sdp_mem_data(fft->temp),
                        fft->num_x,
                        fft->batch_size,
                        do_inverse
                );
            }
            if (sdp_mem_type(input) == SDP_MEM_COMPLEX_DOUBLE
                    && sdp_mem_type(output) == SDP_MEM_COMPLEX_DOUBLE)
            {
                sdp_1d_fft(
                        (sdp_Double2*) sdp_mem_data(output),
                        (sdp_Double2*) sdp_mem_data(input),
                        (sdp_Double2*) sdp_mem_data(fft->temp),
                        fft->num_x,
                        fft->batch_size,
                        do_inverse
                );
            }
        }
        if (fft->num_dims == 2)
        {
            if (sdp_mem_type(input) == SDP_MEM_COMPLEX_FLOAT
                    && sdp_mem_type(output) == SDP_MEM_COMPLEX_FLOAT)
            {
                sdp_2d_fft(
                        (sdp_Float2*) sdp_mem_data(output),
                        (sdp_Float2*) sdp_mem_data(input),
                        (sdp_Float2*) sdp_mem_data(fft->temp),
                        fft->num_x,
                        fft->num_y,
                        fft->batch_size,
                        do_inverse
                );
            }
            if (sdp_mem_type(input) == SDP_MEM_COMPLEX_DOUBLE
                    && sdp_mem_type(output) == SDP_MEM_COMPLEX_DOUBLE)
            {
                sdp_2d_fft(
                        (sdp_Double2*) sdp_mem_data(output),
                        (sdp_Double2*) sdp_mem_data(input),
                        (sdp_Double2*) sdp_mem_data(fft->temp),
                        fft->num_x,
                        fft->num_y,
                        fft->batch_size,
                        do_inverse
                );
            }
        }

        if (fft->num_dims > 2)
        {
            *status = SDP_ERR_INVALID_ARGUMENT;
            SDP_LOG_ERROR("3D FFT is not supported on host memory.");
            return;
        }
    }
}

