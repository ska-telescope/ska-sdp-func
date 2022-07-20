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
    sdp_Mem* input;
    sdp_Mem* output;
    int num_dims;
    int batch_size;
    int is_forward;
    int cufft_plan;
};

static void check_params(
        const sdp_Mem* input,
        const sdp_Mem* output,
        int32_t num_dims_fft,
        sdp_Error* status)
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
                "number of dimensions");
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
    if (num_dims_fft != sdp_mem_num_dims(input) &&
            num_dims_fft != sdp_mem_num_dims(input) - 1)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Number of FFT dimensions must be equal to "
                "or one smaller than the number of array dimensions");
        return;
    }
}


sdp_Fft* sdp_fft_create(
        const sdp_Mem* input,
        const sdp_Mem* output,
        int32_t num_dims_fft,
        int32_t is_forward,
        sdp_Error* status)
{
    sdp_Fft* fft = 0;
    int cufft_plan = 0;
    int batch_size = 1;
    check_params(input, output, num_dims_fft, status);
    if (*status) return fft;
    if (sdp_mem_location(input) == SDP_MEM_GPU)
    {
#ifdef SDP_HAVE_CUDA
        int idist = 0, istride = 0, odist = 0, ostride = 0;
        const int32_t num_dims = sdp_mem_num_dims(input);
        const int32_t last_dim = num_dims - 1;
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
                cufft_type, batch_size);
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
        fft->input = sdp_mem_create_alias(input);
        fft->output = sdp_mem_create_alias(output);
        fft->num_dims = num_dims_fft;
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
        sdp_Error* status)
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
                    (cufftComplex*)sdp_mem_data(output), fft_dir);
        }
        else if (sdp_mem_type(input) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(output) == SDP_MEM_COMPLEX_DOUBLE)
        {
            error = cufftExecZ2Z(fft->cufft_plan,
                    (cufftDoubleComplex*)sdp_mem_data(input),
                    (cufftDoubleComplex*)sdp_mem_data(output), fft_dir);
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
    free(fft);
}
