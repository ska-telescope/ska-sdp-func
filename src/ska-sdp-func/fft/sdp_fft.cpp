/* See the LICENSE file at the top-level directory of this distribution. */

#include <cstdlib>
#include <cstring>

#include "ska-sdp-func/fft/sdp_fft.h"
#include "ska-sdp-func/utility/sdp_logging.h"

#ifdef SDP_HAVE_CUDA
#include <cufft.h>
#endif

struct sdp_Fft
{
    sdp_MemType precision;
    sdp_MemLocation location;
    sdp_FftType fft_type;
    int num_dims;
    int batch_size;
    int is_forward;
    int cufft_plan;
};


sdp_Fft* sdp_fft_create(
        sdp_MemType precision,
        sdp_MemLocation location,
        sdp_FftType fft_type,
        int num_dims,
        const int64_t* dim_size,
        int batch_size,
        int is_forward,
        sdp_Error* status)
{
    sdp_Fft* fft = 0;
    int cufft_plan = 0;
    if (location == SDP_MEM_GPU)
    {
#ifdef SDP_HAVE_CUDA
        cufftType cufft_type = CUFFT_C2C;
        if (fft_type == SDP_FFT_C2C)
        {
            if (precision == SDP_MEM_DOUBLE)
            {
                cufft_type = CUFFT_Z2Z;
            }
            else if (precision == SDP_MEM_FLOAT)
            {
                cufft_type = CUFFT_C2C;
            }
            else
            {
                *status = SDP_ERR_DATA_TYPE;
                SDP_LOG_ERROR("Unsupported precision for SDP_FFT_C2C");
                return fft;
            }
        }
        else
        {
            *status = SDP_ERR_RUNTIME;
            SDP_LOG_ERROR("Unsupported FFT type");
            return fft;
        }
        int* dim_size_copy = (int*) calloc(num_dims, sizeof(int));
        for (int i = 0; i < num_dims; ++i) dim_size_copy[i] = dim_size[i];
        int total_elements = 1;
        for (int i = 0; i < num_dims; ++i) total_elements *= (int) dim_size[i];
        const cufftResult error = cufftPlanMany(
                &cufft_plan, num_dims, dim_size_copy,
                0, 1, total_elements, 0, 1, total_elements,
                cufft_type, batch_size);
        free(dim_size_copy);
        if (error != CUFFT_SUCCESS)
        {
            *status = SDP_ERR_RUNTIME;
            SDP_LOG_ERROR("cufftPlanMany error (code %d)", error);
            return fft;
        }
#else
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The processing function library was compiled "
                "without CUDA support");
        return fft;
#endif
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
        fft->precision = precision;
        fft->location = location;
        fft->fft_type = fft_type;
        fft->num_dims = num_dims;
        fft->batch_size = batch_size;
        fft->is_forward = is_forward;
        fft->cufft_plan = cufft_plan;
    }
    return fft;
}


void sdp_fft_exec(sdp_Fft* fft, sdp_Mem* input, sdp_Mem* output,
        sdp_Error* status)
{
    if (*status || !fft || !input || !output) return;
    if (sdp_mem_is_read_only(output))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Output array is read-only");
        return;
    }
    if (sdp_mem_location(input) != fft->location ||
            sdp_mem_location(output) != fft->location)
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("FFT plan and arrays must be in the same location");
        return;
    }
    if (fft->location == SDP_MEM_GPU)
    {
#ifdef SDP_HAVE_CUDA
        cufftResult error = CUFFT_SUCCESS;
        const int fft_dir = fft->is_forward ? CUFFT_FORWARD : CUFFT_INVERSE;
        if (fft->fft_type == SDP_FFT_C2C)
        {
            if (fft->precision == SDP_MEM_FLOAT &&
                    sdp_mem_type(input) == SDP_MEM_COMPLEX_FLOAT &&
                    sdp_mem_type(output) == SDP_MEM_COMPLEX_FLOAT)
            {
                error = cufftExecC2C(fft->cufft_plan,
                        (cufftComplex*)sdp_mem_data(input),
                        (cufftComplex*)sdp_mem_data(output), fft_dir);
            }
            else if (fft->precision == SDP_MEM_DOUBLE &&
                    sdp_mem_type(input) == SDP_MEM_COMPLEX_DOUBLE &&
                    sdp_mem_type(output) == SDP_MEM_COMPLEX_DOUBLE)
            {
                error = cufftExecZ2Z(fft->cufft_plan,
                        (cufftDoubleComplex*)sdp_mem_data(input),
                        (cufftDoubleComplex*)sdp_mem_data(output), fft_dir);
            }
            else
            {
                *status = SDP_ERR_DATA_TYPE;
                SDP_LOG_ERROR("Inconsistent data type(s) for SDP_FFT_C2C.");
            }
        }
        if (error != CUFFT_SUCCESS)
        {
            *status = SDP_ERR_RUNTIME;
            SDP_LOG_ERROR("cufftExec error (code %d)", error);
        }
#else
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("The processing function library was compiled "
                "without CUDA support");
#endif
    }
    else
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Unsupported FFT location");
    }
}


void sdp_fft_free(sdp_Fft* fft)
{
    if (!fft) return;
#ifdef SDP_HAVE_CUDA
    if (fft->location == SDP_MEM_GPU)
    {
        cufftDestroy(fft->cufft_plan);
    }
#endif
    free(fft);
}
