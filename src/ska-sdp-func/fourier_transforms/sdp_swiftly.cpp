
#include "ska-sdp-func/utility/sdp_mem.h"
#include "ska-sdp-func/utility/sdp_logging.h"

#include "sdp_swiftly.h"
#include "sdp_pswf.h"
#include "pocketfft_hdronly.h"
#include "sdp_mem_utils.h"

struct sdp_SwiFTly
{
    // Parameters
    int64_t image_size;
    int64_t yN_size; // internal facet size
    int64_t xM_size; // internal subgrid size
    double W; // PSWF parameter
    // Preparation data
    sdp_Mem* Fb; // window correction function
    sdp_Mem* Fn; // window function
    // ... FFT plan?
};

sdp_SwiFTly* sdp_swiftly_create(
    int64_t image_size,
    int64_t yN_size,
    int64_t xM_size,
    double W,
    sdp_Error* status)
{
    if (*status) return NULL;

    // Sanity-check sizes
    if (image_size <= 0 || xM_size <= 0 || yN_size <= 0) {
        SDP_LOG_ERROR("sdp_swiftly_create: Negative size passed.");
        *status = SDP_ERR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (image_size % xM_size != 0) {
        SDP_LOG_ERROR("sdp_swiftly_create: Image size not divisible by subgrid size.");
        *status = SDP_ERR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (image_size % yN_size != 0) {
        SDP_LOG_ERROR("sdp_swiftly_create: Image size not divisible by facet size.");
        *status = SDP_ERR_INVALID_ARGUMENT;
        return nullptr;
    }
    if ((xM_size * yN_size) % image_size != 0) {
        SDP_LOG_ERROR("sdp_swiftly_create: Contribution size not integer.");
        *status = SDP_ERR_INVALID_ARGUMENT;
        return nullptr;
    }

    // Simplifying assumptions (could be lifted, but likely not worth it)
    if (xM_size % 2 != 0) {
        SDP_LOG_ERROR("sdp_swiftly_create: Subgrid size not even.");
        *status = SDP_ERR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (yN_size % 2 != 0) {
        SDP_LOG_ERROR("sdp_swiftly_create: Facet size not even.");
        *status = SDP_ERR_INVALID_ARGUMENT;
        return nullptr;
    }
    
    // Generate PSWF
    sdp_Mem *Fb_mem = sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 1, &yN_size, status);
    if (*status) return NULL;
    sdp_generate_pswf(0, W * (M_PI / 2), Fb_mem, status);
    if (*status) { sdp_mem_free(Fb_mem); return NULL; }

    // Allocate Fn
    const int64_t xM_yN_size = (xM_size * yN_size) / image_size;
    sdp_Mem *Fn_mem = sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 1, &xM_yN_size, status);
    if (*status) { sdp_mem_free(Fb_mem); return NULL; }

    // Generate Fn
    const double *pswf = static_cast<double*>(sdp_mem_data(Fb_mem));
    double *Fn = static_cast<double*>(sdp_mem_data(Fn_mem));
    const int xM_step = image_size / xM_size;
    const int Fn_offset = (yN_size / 2) % xM_step;
    for (int i = 0; i < xM_yN_size; i++) {
        Fn[i] = pswf[Fn_offset + i * xM_step];
    }

    // Generate Fb (overwriting PSWF inplace!)
    double *Fb = static_cast<double*>(sdp_mem_data(Fb_mem));
    for (int i = 1; i < yN_size; i++) {
        Fb[i] = 1 / Fb[i];
    }

    // Done
    sdp_SwiFTly* swiftly = static_cast<struct sdp_SwiFTly*>(
        malloc(sizeof(sdp_SwiFTly)));
    swiftly->image_size = image_size;
    swiftly->xM_size = xM_size;
    swiftly->yN_size = yN_size;
    swiftly->W = W;
    swiftly->image_size = image_size;
    swiftly->Fb = Fb_mem;
    swiftly->Fn = Fn_mem;
    return swiftly;
}

void sdp_swiftly_free(sdp_SwiFTly* swiftly)
{
    if (swiftly) {
        sdp_mem_free(swiftly->Fb);
        sdp_mem_free(swiftly->Fn);
    }
}

void sdp_swiftly_prepare_facet(
    sdp_SwiFTly* swiftly,
    sdp_Mem *facet,
    sdp_Mem *prep_facet_out,
    int64_t facet_offset,
    sdp_Error* status)
{
    if (*status) return;

    // Parameter checks
    const int64_t yN_size = swiftly->yN_size;
    if (sdp_mem_shape_dim(facet, 0) > yN_size) {
        SDP_LOG_ERROR("Facet data too large (%d>%d)!", sdp_mem_shape_dim(facet, 0), yN_size);
        *status = SDP_ERR_INVALID_ARGUMENT;
    }
    sdp_CpuVec<const std::complex<double>, 1> fct;
    sdp_mem_checked_get_vec(facet, &fct, status);
    sdp_CpuVec<std::complex<double>, 1> out;
    sdp_mem_checked_get_vec(prep_facet_out, &out, status);
    sdp_CpuVec<double, 1> Fb;
    sdp_mem_checked_get_vec(swiftly->Fb, &Fb, status);
    if (*status) return;

    // We shift the facet centre to its correct global position modulo
    // yN_size (non-centered for FFT, i.e. apply ifftshift). Determine
    // start & end index accordingly.
    const int64_t start_ = (facet_offset - fct.shape[0] / 2) % yN_size;
    const int64_t start = (start_ < 0 ? start_ + yN_size : start_);
    const int64_t end = (start + fct.shape[0]) % yN_size;
    const int64_t Fb_off = (yN_size / 2 - fct.shape[0] / 2) - start;

    // Does the facet data alias around the edges?
    int64_t i;
    if (start < end) {
        for (i = 0; i < start; i++) {
            out[i] = 0;
        }
        for (i = start; i < end; i++) {
            out[i] = fct[i - start] * Fb[i + Fb_off];
        }
        for (i = end; i < yN_size; i++) {
            out[i] = 0;
        }
    } else {
        for (i = 0; i < end; i++) {
            out[i] = fct[i + yN_size - start] * Fb[i + yN_size + Fb_off];
        }
        for (i = end; i < start; i++) {
            out[i] = 0;
        }
        for (i = start; i < yN_size; i++) {
            out[i] = fct[i - start] * Fb[i + Fb_off];
        }
    }

    // Perform FFT
    pocketfft::shape_t shape = { static_cast<size_t>(out.shape[0]) };
    pocketfft::stride_t stride = { out.stride[0] * sizeof(double) * 2 };
    pocketfft::c2c(shape, stride, stride, { 0 }, pocketfft::BACKWARD,
    out.ptr, out.ptr, 1. / yN_size);

    // Shift FFT result. Pretty sure this will eventually cancel out
    // with a shift on the receiver end, but leave it for the moment
    // to keep compatibility with the Python version.
    for (i = 0; i < yN_size/2; i++) {
        std::complex<double> tmp = out[i];
        out[i] = out[i+yN_size/2];
        out[i+yN_size/2] = tmp;
    }
}
