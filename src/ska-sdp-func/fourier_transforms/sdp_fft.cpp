/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>

#include "ska-sdp-func/fourier_transforms/sdp_fft.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"

#ifdef SDP_HAVE_CUDA
#include <cufft.h>
#endif

#ifdef SDP_HAVE_MKL
#include "mkl.h"
#ifdef SDP_USE_POCKETFFT
#undef SDP_USE_POCKETFFT
#endif
#endif

#ifdef SDP_USE_POCKETFFT
#include "pocketfft_hdronly.h"
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
#ifdef SDP_HAVE_CUDA
    int cufft_plan;
#endif
#ifdef SDP_HAVE_MKL
    DFTI_DESCRIPTOR_HANDLE mkl_plan;
#endif
#ifdef SDP_USE_POCKETFFT
    pocketfft::shape_t shape;
    pocketfft::stride_t stride_in;
    pocketfft::stride_t stride_out;
    pocketfft::shape_t axes;
#endif
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

#ifdef SDP_HAVE_CUDA


static
void sdp_fft_create_cuda(
        sdp_Fft* fft,
        const sdp_Mem* input,
        const sdp_Mem* output,
        int32_t num_dims_fft,
        int32_t /* is_forward */,
        sdp_Error* status
)
{
    const int32_t num_dims = sdp_mem_num_dims(input);
    const int32_t last_dim = num_dims - 1;

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
        return;
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
    istride = sdp_mem_stride_elements_dim(input, last_dim);
    ostride = sdp_mem_stride_elements_dim(output, last_dim);
    idist = sdp_mem_stride_elements_dim(input, 0);
    odist = sdp_mem_stride_elements_dim(output, 0);
    const cufftResult error = cufftPlanMany(
            &fft->cufft_plan, num_dims_fft, dim_size,
            inembed, istride, idist, onembed, ostride, odist,
            cufft_type, fft->batch_size
    );
    free(onembed);
    free(inembed);
    free(dim_size);
    if (error != CUFFT_SUCCESS)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("cufftPlanMany error (code %d)", error);
    }
}


static
void sdp_fft_exec_cuda(
        sdp_Fft* fft,
        sdp_Mem* input,
        sdp_Mem* output,
        sdp_Error* status
)
{
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
}
#endif // SDP_HAVE_CUDA

#ifdef SDP_HAVE_MKL


static
void sdp_fft_create_mkl(
        sdp_Fft* fft,
        const sdp_Mem* input,
        const sdp_Mem* output,
        int32_t num_dims_fft,
        int32_t is_forward,
        sdp_Error* status
)
{
    const int32_t num_dims = sdp_mem_num_dims(input);
    const int32_t last_dim = num_dims - 1;

    MKL_LONG mkl_status = 0;
    DFTI_CONFIG_VALUE precision = DFTI_DOUBLE;
    if (sdp_mem_type(input) == SDP_MEM_COMPLEX_DOUBLE &&
            sdp_mem_type(output) == SDP_MEM_COMPLEX_DOUBLE)
    {
        precision = DFTI_DOUBLE;
    }
    else if (sdp_mem_type(input) == SDP_MEM_COMPLEX_FLOAT &&
            sdp_mem_type(output) == SDP_MEM_COMPLEX_FLOAT)
    {
        precision = DFTI_SINGLE;
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data types");
        return fft;
    }

    // Create descriptor.
    MKL_LONG* dim_size = (MKL_LONG*) calloc(
            num_dims_fft, sizeof(MKL_LONG)
    );
    MKL_LONG num_elem_fft = 1;
    for (int i = 0; i < num_dims_fft; ++i)
    {
        dim_size[i] = sdp_mem_shape_dim(
                input, i + num_dims - num_dims_fft
        );
        num_elem_fft *= dim_size[i];
    }
    if (num_dims_fft == 1)
    {
        mkl_status = DftiCreateDescriptor(
                &fft->mkl_plan, precision, DFTI_COMPLEX,
                1, (MKL_LONG) sdp_mem_shape_dim(input, last_dim)
        );
    }
    else
    {
        mkl_status = DftiCreateDescriptor(
                &fft->mkl_plan, precision, DFTI_COMPLEX,
                num_dims_fft, dim_size
        );
    }
    free(dim_size);

    // Check for any errors from MKL.
    if (mkl_status && !DftiErrorClass(mkl_status, DFTI_NO_ERROR))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Error in DftiCreateDescriptor (code %lld): %s",
                mkl_status, DftiErrorMessage(mkl_status)
        );
        return fft;
    }

    // Set descriptor parameters.
    DftiSetValue(fft->mkl_plan, DFTI_NUMBER_OF_TRANSFORMS,
            (MKL_LONG) fft->batch_size
    );
    DftiSetValue(fft->mkl_plan, DFTI_INPUT_DISTANCE, num_elem_fft);
    DftiSetValue(fft->mkl_plan, DFTI_OUTPUT_DISTANCE, num_elem_fft);
}


static
void sdp_fft_exec_mkl(
        sdp_Fft* fft,
        sdp_Mem* input,
        sdp_Mem* output,
        sdp_Error* status
)
{
    MKL_LONG mkl_status = 0;

    // Check if transform should be in-place or out-of-place.
    DFTI_CONFIG_VALUE placement = DFTI_NOT_INPLACE;
    if (sdp_mem_data(output) == sdp_mem_data(input))
    {
        placement = DFTI_INPLACE;
    }
    DftiSetValue(fft->mkl_plan, DFTI_PLACEMENT, placement);

    // Commit descriptor.
    mkl_status = DftiCommitDescriptor(fft->mkl_plan);
    if (mkl_status && !DftiErrorClass(mkl_status, DFTI_NO_ERROR))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Error in DftiCommitDescriptor (code %lld): %s",
                mkl_status, DftiErrorMessage(mkl_status)
        );
    }

    // Compute FFT.
    if (fft->is_forward)
    {
        mkl_status = DftiComputeForward(
                fft->mkl_plan, sdp_mem_data(input), sdp_mem_data(output)
        );
        if (mkl_status && !DftiErrorClass(mkl_status, DFTI_NO_ERROR))
        {
            *status = SDP_ERR_RUNTIME;
            SDP_LOG_ERROR("Error in DftiComputeForward (code %lld): %s",
                    mkl_status, DftiErrorMessage(mkl_status)
            );
        }
    }
    else
    {
        mkl_status = DftiComputeBackward(
                fft->mkl_plan, sdp_mem_data(input), sdp_mem_data(output)
        );
        if (mkl_status && !DftiErrorClass(mkl_status, DFTI_NO_ERROR))
        {
            *status = SDP_ERR_RUNTIME;
            SDP_LOG_ERROR("Error in DftiComputeBackward (code %lld): %s",
                    mkl_status, DftiErrorMessage(mkl_status)
            );
        }
    }
}
#endif // SDP_HAVE_MKL


#ifdef SDP_USE_POCKETFFT


static
void sdp_fft_create_pocketfft(
        sdp_Fft* fft,
        const sdp_Mem* input,
        const sdp_Mem* output,
        int32_t num_dims_fft,
        int32_t /* is_forward */,
        sdp_Error* /* status */
)
{
    const int32_t num_dims = sdp_mem_num_dims(input);

    // Allocate information about data layout.
    // (Note that this is a std::vector, but was originally allocated
    //  using calloc. There's a chance this might cause problems with
    //  certain standard library implementations!)
    for (int i = 0; i < num_dims; i++)
    {
        fft->shape.push_back(sdp_mem_shape_dim(input, i));
        fft->stride_in.push_back(sdp_mem_stride_bytes_dim(input, i));
        fft->stride_out.push_back(sdp_mem_stride_bytes_dim(output, i));
    }
    for (int i = 0; i < num_dims_fft; i++)
    {
        fft->axes.push_back(num_dims - num_dims_fft + i);
    }
}


static
void sdp_fft_exec_pocketfft(
        sdp_Fft* fft,
        sdp_Mem* input,
        sdp_Mem* output,
        sdp_Error* status
)
{
    if (sdp_mem_type(input) == SDP_MEM_COMPLEX_FLOAT
            && sdp_mem_type(output) == SDP_MEM_COMPLEX_FLOAT)
    {
        pocketfft::c2c<float>(
                fft->shape,
                fft->stride_in,
                fft->stride_out,
                fft->axes,
                fft->is_forward ? pocketfft::FORWARD : pocketfft::BACKWARD,
                (std::complex<float>*) sdp_mem_data(input),
                (std::complex<float>*) sdp_mem_data(output),
                1.0
        );
    }
    else if (sdp_mem_type(input) == SDP_MEM_COMPLEX_DOUBLE
            && sdp_mem_type(output) == SDP_MEM_COMPLEX_DOUBLE)
    {
        pocketfft::c2c<double>(
                fft->shape,
                fft->stride_in,
                fft->stride_out,
                fft->axes,
                fft->is_forward ? pocketfft::FORWARD : pocketfft::BACKWARD,
                (std::complex<double>*) sdp_mem_data(input),
                (std::complex<double>*) sdp_mem_data(output),
                1.0
        );
    }
    else
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Inconsistent input and output buffer types.");
    }
}


#else


static
void sdp_fft_create_native(
        sdp_Fft* fft,
        const sdp_Mem* input,
        const sdp_Mem* output,
        int32_t num_dims_fft,
        int32_t is_forward,
        sdp_Error* status
)
{
    // No MKL available.
    const int32_t num_dims = sdp_mem_num_dims(input);
    const int32_t last_dim = num_dims - 1;
    int64_t num_x_stride = 0, num_y_stride = 0, batch_stride = 0;
    if (num_dims_fft == 1)
    {
        fft->num_x = sdp_mem_shape_dim(input, last_dim);
        num_x_stride = sdp_mem_stride_elements_dim(input, last_dim);
        fft->num_y = 1;
        num_y_stride = fft->num_x;
    }
    if (num_dims_fft == 2)
    {
        fft->num_x = sdp_mem_shape_dim(input, last_dim - 1);
        fft->num_y = sdp_mem_shape_dim(input, last_dim);
        num_x_stride = sdp_mem_stride_elements_dim(input, last_dim);
        num_y_stride = sdp_mem_stride_elements_dim(input, last_dim - 1);
    }
    if (num_dims_fft > 2)
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported FFT dimension");
    }
    batch_stride = fft->num_x * fft->num_y;
    if (num_dims != num_dims_fft)
    {
        batch_stride = sdp_mem_stride_elements_dim(input, 0);
    }

    if (num_x_stride != 1
            && num_y_stride != fft->num_x
            && batch_stride != fft->num_x * fft->num_y)
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
        fft->temp = sdp_mem_create(
                sdp_mem_type(input), SDP_MEM_CPU, num_dims, shape, status
        );
        free(shape);
    }
}


static
void sdp_fft_exec_native(
        sdp_Fft* fft,
        sdp_Mem* input,
        sdp_Mem* output,
        sdp_Error* status
)
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
    }
}

#endif


sdp_Fft* sdp_fft_create(
        const sdp_Mem* input,
        const sdp_Mem* output,
        int32_t num_dims_fft,
        int32_t is_forward,
        sdp_Error* status
)
{
    check_params(input, output, num_dims_fft, status);
    if (*status) return NULL;

    // Allocate space for a plan.
    sdp_Fft* fft = (sdp_Fft*) calloc(1, sizeof(sdp_Fft));
    fft->input = sdp_mem_create_alias(input);
    fft->output = sdp_mem_create_alias(output);
    fft->num_dims = num_dims_fft;
    fft->batch_size = 1;
    fft->is_forward = is_forward;

    // Set up the batch size.
    const int32_t num_dims = sdp_mem_num_dims(input);
    if (num_dims != num_dims_fft)
    {
        fft->batch_size = sdp_mem_shape_dim(input, 0);
    }

    if (sdp_mem_location(input) == SDP_MEM_GPU)
    {
#ifdef SDP_HAVE_CUDA
        sdp_fft_create_cuda(fft, input, output, num_dims_fft, is_forward,
                status
        );
#else
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The processing function library was compiled "
                "without CUDA support"
        );
#endif
    }
    else if (sdp_mem_location(input) == SDP_MEM_CPU)
    {
#ifdef SDP_HAVE_MKL
        sdp_fft_create_mkl(fft, input, output, num_dims_fft, is_forward,
                status
        );

#elif defined(SDP_USE_POCKETFFT)
        sdp_fft_create_pocketfft(fft, input, output, num_dims_fft, is_forward,
                status
        );

#else
        sdp_fft_create_native(fft, input, output, num_dims_fft, is_forward,
                status
        );

#endif
    }
    else
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Unsupported FFT location");
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
        sdp_fft_exec_cuda(fft, input, output, status);
#else
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The processing function library was compiled "
                "without CUDA support"
        );
#endif
    }
    else if (sdp_mem_location(input) == SDP_MEM_CPU)
    {
#ifdef SDP_HAVE_MKL
        sdp_fft_exec_mkl(fft, input, output, status);
#elif defined(SDP_USE_POCKETFFT)
        sdp_fft_exec_pocketfft(fft, input, output, status);
#else
        sdp_fft_exec_native(fft, input, output, status);
#endif
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
#ifdef SDP_HAVE_MKL
    (void) DftiFreeDescriptor(&fft->mkl_plan);
#endif
#ifdef SDP_USE_POCKETFFT
    fft->shape.clear();
    fft->stride_in.clear();
    fft->stride_out.clear();
    fft->axes.clear();
#endif
    sdp_mem_ref_dec(fft->input);
    sdp_mem_ref_dec(fft->output);
    sdp_mem_free(fft->temp);
    free(fft);
}


template<typename DATA_TYPE>
static void fft_norm(sdp_Mem* data, sdp_Error* status)
{
    if (*status) return;
    sdp_MemViewCpu<DATA_TYPE, 2> data_;
    sdp_mem_check_and_view(data, &data_, status);
    const int num_x = (int) data_.shape[0];
    const int num_y = (int) data_.shape[1];
    const double factor = 1.0 / (num_x * num_y);
    #pragma omp parallel for
    for (int ix = 0; ix < num_x; ++ix)
    {
        for (int iy = 0; iy < num_y; ++iy)
        {
            data_(ix, iy) *= factor;
        }
    }
}


void sdp_fft_norm(sdp_Mem* data, sdp_Error* status)
{
    if (*status || !data) return;
    if (sdp_mem_location(data) == SDP_MEM_CPU)
    {
        if (sdp_mem_type(data) == SDP_MEM_COMPLEX_DOUBLE)
        {
            fft_norm<complex<double> >(data, status);
        }
        else if (sdp_mem_type(data) == SDP_MEM_COMPLEX_FLOAT)
        {
            fft_norm<complex<float> >(data, status);
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
        }
    }
    else if (sdp_mem_location(data) == SDP_MEM_GPU)
    {
        uint64_t num_threads[] = {32, 8, 1}, num_blocks[] = {1, 1, 1};
        const int num_x = sdp_mem_shape_dim(data, 0);
        const int num_y = sdp_mem_shape_dim(data, 1);
        num_blocks[0] = (num_x + num_threads[0] - 1) / num_threads[0];
        num_blocks[1] = (num_y + num_threads[1] - 1) / num_threads[1];
        const double factor = 1.0 / (num_x * num_y);
        sdp_MemViewGpu<complex<double>, 2> data_dbl;
        sdp_MemViewGpu<complex<float>, 2> data_flt;
        const char* kernel_name = 0;
        int is_dbl = 0;
        if (sdp_mem_type(data) == SDP_MEM_COMPLEX_FLOAT)
        {
            is_dbl = 0;
            sdp_mem_check_and_view(data, &data_flt, status);
            kernel_name = "sdp_fft_norm<complex<float> >";
        }
        else if (sdp_mem_type(data) == SDP_MEM_COMPLEX_DOUBLE)
        {
            is_dbl = 1;
            sdp_mem_check_and_view(data, &data_dbl, status);
            kernel_name = "sdp_fft_norm<complex<double> >";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            return;
        }
        const void* arg[] = {
            is_dbl ? (const void*)&data_dbl : (const void*)&data_flt,
            (const void*)&factor
        };
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, 0, 0, arg, status
        );
    }
    else
    {
        *status = SDP_ERR_MEM_LOCATION;
    }
}


template<typename DATA_TYPE>
static void fft_phase(sdp_Mem* data, sdp_Error* status)
{
    if (*status) return;
    sdp_MemViewCpu<DATA_TYPE, 2> data_;
    sdp_mem_check_and_view(data, &data_, status);
    const int num_x = (int) data_.shape[0];
    const int num_y = (int) data_.shape[1];
    #pragma omp parallel for
    for (int ix = 0; ix < num_x; ++ix)
    {
        for (int iy = 0; iy < num_y; ++iy)
        {
            data_(ix, iy) *= (1 - (((ix + iy) & 1) << 1));
        }
    }
}


#if 0


template<typename T>
static inline void swap_values(T& a, T& b)
{
    T temp = a;
    a = b;
    b = temp;
}


// The fft_phase function, above, is faster than this by quite a long way.
template<typename DATA_TYPE>
static void fft_shift(sdp_Mem* data, sdp_Error* status)
{
    if (*status) return;
    sdp_MemViewCpu<DATA_TYPE, 2> data_;
    sdp_mem_check_and_view(data, &data_, status);
    const int rows = (int) data_.shape[0];
    const int cols = (int) data_.shape[1];
    const int half_rows = rows / 2;
    const int half_cols = cols / 2;

    // Swap quadrants: 1 with 3, and 2 with 4 for even-sized arrays.
    #pragma omp parallel for
    for (int i = 0; i < half_rows; ++i)
    {
        for (int j = 0; j < half_cols; ++j)
        {
            // Swap (i, j) with (i + half_rows, j + half_cols)
            swap_values(data_(i, j), data_(i + half_rows, j + half_cols));

            // Swap (i, j + half_cols) with (i + half_rows, j)
            swap_values(data_(i, j + half_cols), data_(i + half_rows, j));
        }
    }

    // Handle odd-sized arrays.
    if (rows % 2 != 0)
    {
        for (int j = 0; j < half_cols; ++j)
        {
            // Swap middle row's left and right parts.
            swap_values(data_(half_rows, j), data_(half_rows, j + half_cols));
        }
    }
    if (cols % 2 != 0)
    {
        for (int i = 0; i < half_rows; ++i)
        {
            // Swap middle column's top and bottom parts.
            swap_values(data_(i, half_cols), data_(i + half_rows, half_cols));
        }
    }
}
#endif


void sdp_fft_phase(sdp_Mem* data, sdp_Error* status)
{
    if (*status || !data) return;
    if (sdp_mem_location(data) == SDP_MEM_CPU)
    {
        if (sdp_mem_type(data) == SDP_MEM_COMPLEX_DOUBLE)
        {
            fft_phase<complex<double> >(data, status);
        }
        else if (sdp_mem_type(data) == SDP_MEM_COMPLEX_FLOAT)
        {
            fft_phase<complex<float> >(data, status);
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
        }
    }
    else if (sdp_mem_location(data) == SDP_MEM_GPU)
    {
        uint64_t num_threads[] = {32, 8, 1}, num_blocks[] = {1, 1, 1};
        sdp_MemViewGpu<complex<double>, 2> data_dbl;
        sdp_MemViewGpu<complex<float>, 2> data_flt;
        const char* kernel_name = 0;
        int is_dbl = 0;
        if (sdp_mem_type(data) == SDP_MEM_COMPLEX_FLOAT)
        {
            is_dbl = 0;
            sdp_mem_check_and_view(data, &data_flt, status);
            kernel_name = "sdp_fft_phase<complex<float> >";
        }
        else if (sdp_mem_type(data) == SDP_MEM_COMPLEX_DOUBLE)
        {
            is_dbl = 1;
            sdp_mem_check_and_view(data, &data_dbl, status);
            kernel_name = "sdp_fft_phase<complex<double> >";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
        }
        if (*status) return;
        const int num_x = sdp_mem_shape_dim(data, 0);
        const int num_y = sdp_mem_shape_dim(data, 1);
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
        const void* arg[] = {
            is_dbl ? (const void*)&data_dbl : (const void*)&data_flt,
        };
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
