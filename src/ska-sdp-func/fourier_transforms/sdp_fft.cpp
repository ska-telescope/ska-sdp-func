/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "ska-sdp-func/fourier_transforms/sdp_fft.h"
#include "ska-sdp-func/utility/sdp_logging.h"

#ifdef SDP_HAVE_CUDA
#include <cufft.h>
#endif

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
