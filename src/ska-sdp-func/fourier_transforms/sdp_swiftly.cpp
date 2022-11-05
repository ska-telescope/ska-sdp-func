
#include "ska-sdp-func/utility/sdp_mem.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"
#include "ska-sdp-func/utility/sdp_logging.h"

#include "sdp_swiftly.h"
#include "sdp_pswf.h"
#include "pocketfft_hdronly.h"

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

inline int64_t mod_p(int64_t a, int64_t b) {
    // Modulo that always gives positive result (i.e. remainder of
    // floor(a/b))
    a %= b;
    if (a >= 0)
        return a;
    else
        return a + b;
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
    if (sdp_mem_num_dims(facet) > 0 && sdp_mem_shape_dim(facet, 0) > yN_size) {
        SDP_LOG_ERROR("Facet data too large (%d>%d)!", sdp_mem_shape_dim(facet, 0), yN_size);
        *status = SDP_ERR_INVALID_ARGUMENT;
    }
    sdp_MemViewCpu<const std::complex<double>, 1> fct;
    sdp_mem_check_and_view(facet, &fct, status);
    sdp_MemViewCpu<std::complex<double>, 1> out;
    sdp_mem_check_and_view(prep_facet_out, &out, status);
    sdp_mem_check_shape(prep_facet_out, 0, yN_size, status);
    sdp_MemViewCpu<double, 1> Fb;
    sdp_mem_check_and_view(swiftly->Fb, &Fb, status);
    if (*status) return;

    // We shift the facet centre to its correct global position modulo
    // yN_size (non-centered for FFT, i.e. apply ifftshift). Determine
    // start & end index accordingly.
    const int64_t start = mod_p(facet_offset - fct.shape[0] / 2,  yN_size);
    const int64_t end = (start + fct.shape[0]) % yN_size;
    const int64_t Fb_off = (yN_size / 2 - fct.shape[0] / 2) - start;

    // Does the facet data alias around the edges?
    int64_t i;
    if (start < end) {
        // Establish something along the lines of "00<data>00000"
        for (i = 0; i < start; i++) {
            out(i) = 0;
        }
        for (i = start; i < end; i++) {
            out(i) = fct(i - start) * Fb(i + Fb_off);
        }
        for (i = end; i < yN_size; i++) {
            out(i) = 0;
        }
    } else {
        // ... or alternatively "ta>00000<da"
        for (i = 0; i < end; i++) {
            out(i) = fct(i + yN_size - start) * Fb(i + yN_size + Fb_off);
        }
        for (i = end; i < start; i++) {
            out(i) = 0;
        }
        for (i = start; i < yN_size; i++) {
            out(i) = fct(i - start) * Fb(i + Fb_off);
        }
    }

    // Perform FFT
    const pocketfft::shape_t shape = { static_cast<size_t>(yN_size) };
    const ptrdiff_t cpx_size = sizeof(std::complex<double>);
    const pocketfft::stride_t stride = { out.stride[0] * cpx_size };
    pocketfft::c2c(shape, stride, stride, { 0 }, pocketfft::BACKWARD,
                   out.ptr, out.ptr, 1. / yN_size);

    // missing shift will be corrected for in sdp_swiftly_extract_from_facet

}

void sdp_swiftly_extract_from_facet(
    sdp_SwiFTly* swiftly,
    sdp_Mem *prep_facet,
    sdp_Mem *contribution_out,
    int64_t subgrid_offset,
    sdp_Error* status)
{
    if (*status) return;

    // Parameter checks
    const int64_t yN_size = swiftly->yN_size;
    const int64_t image_size = swiftly->image_size;
    const int64_t xM_yN_size = (swiftly->xM_size * yN_size) / image_size;
    sdp_MemViewCpu<const std::complex<double>, 1> fct;
    sdp_mem_check_and_view(prep_facet, &fct, status);
    sdp_mem_check_shape(prep_facet, 0, yN_size, status);
    sdp_MemViewCpu<std::complex<double>, 1> out;
    sdp_mem_check_and_view(contribution_out, &out, status);
    sdp_mem_check_shape(contribution_out, 0, xM_yN_size, status);
    if (*status) return;

    // Calculate grid offsets (still in yN_size image
    // resolution). This is taking into account the missing fftshift
    // from sdp_swiftly_prepare_facet.
    const int64_t sg_offs = subgrid_offset / (image_size / yN_size);
    const int64_t aliased_sg_offs = mod_p(sg_offs - xM_yN_size/2, xM_yN_size);
    const int64_t offs = sg_offs - aliased_sg_offs - xM_yN_size/2;
    const int64_t offs1 = mod_p(offs + xM_yN_size, yN_size);
    const int64_t offs2 = mod_p(offs, yN_size);

    int64_t i = 0;
    int64_t stop1 = yN_size - offs1;
    if (stop1 > aliased_sg_offs) stop1 = aliased_sg_offs;
    for ( ; i < stop1; i++) {
        out(i) = fct(i + offs1);
    }
    for ( ; i < aliased_sg_offs; i++) {
        out(i) = fct(i + offs1 - yN_size);
    }

    int64_t stop2 = yN_size - offs2;
    if (stop2 > xM_yN_size) stop2 = xM_yN_size;
    for ( ; i < stop2; i++) {
        out(i) = fct(i + offs2);
    }
    for ( ; i < xM_yN_size; i++) {
        out(i) = fct(i + offs2 - yN_size);
    }

    // Perform FFT
    const pocketfft::shape_t shape = { static_cast<size_t>(xM_yN_size) };
    const ptrdiff_t cpx_size = sizeof(std::complex<double>);
    const pocketfft::stride_t stride = { out.stride[0] * cpx_size };
    pocketfft::c2c(shape, stride, stride, { 0 }, pocketfft::FORWARD,
                   out.ptr, out.ptr, 1.);

}

void sdp_swiftly_add_to_subgrid(
    sdp_SwiFTly* swiftly,
    sdp_Mem *contribution,
    sdp_Mem *subgrid_image_inout,
    int64_t facet_offset,
    sdp_Error* status)
{
    if (*status) return;

    // Parameter checks
    const int64_t xM_size = swiftly->xM_size;
    const int64_t image_size = swiftly->image_size;
    const int64_t xM_yN_size = (xM_size * swiftly->yN_size) / image_size;
    sdp_MemViewCpu<const std::complex<double>, 1> contrib;
    sdp_mem_check_and_view(contribution, &contrib, status);
    sdp_mem_check_shape(contribution, 0, xM_yN_size, status);
    sdp_MemViewCpu<std::complex<double>, 1> out;
    sdp_mem_check_and_view(subgrid_image_inout, &out, status);
    sdp_mem_check_shape(subgrid_image_inout, 0, xM_size, status);
    sdp_MemViewCpu<double, 1> Fn;
    sdp_mem_check_and_view(swiftly->Fn, &Fn, status);
    if (*status) return;

    // Calculate facet offsets (in xM_size resolution).
    const int64_t fct_offs = facet_offset / (image_size / xM_size);
    const int64_t offs = mod_p(-xM_yN_size / 2 + xM_size/2 + fct_offs, xM_size);
    int64_t i = 0;
    int64_t stop1 = xM_size - offs;
    if (stop1 > xM_yN_size) stop1 = xM_yN_size;
    for (i = 0; i < stop1; i++) {
        out(i + offs) = Fn(i) * contrib((i + fct_offs + xM_yN_size / 2) % xM_yN_size);
    }
    for (; i < xM_yN_size; i++) {
        out(i + offs - xM_size) = Fn(i) * contrib((i + fct_offs + xM_yN_size / 2) % xM_yN_size);
    }

}

void sdp_swiftly_finish_subgrid_inplace(
    sdp_SwiFTly* swiftly,
    sdp_Mem *subgrid_inout,
    int64_t subgrid_offset,
    sdp_Error* status)
{
    if (*status) return;

    // Parameter checks
    const int64_t xM_size = swiftly->xM_size;
    sdp_MemViewCpu<std::complex<double>, 1> sg;
    sdp_mem_check_and_view(subgrid_inout, &sg, status);
    sdp_mem_check_shape(subgrid_inout, 0, xM_size, status);
    if (*status) return;

    // Perform shift for input and output. We use a phase ramp to do
    // all of this in-place - not sure whether that's truly the most
    // efficient option here.
    double phase_step = 2 * M_PI * (subgrid_offset + xM_size / 2) / xM_size;
    double phase_step2 = M_PI * (subgrid_offset + xM_size / 2);
    std::complex<double> phasor(1.0);
    std::complex<double> step(cos(phase_step), sin(phase_step));
    std::complex<double> step2(cos(phase_step2), sin(phase_step2));
    int i;
    for (i = 0; i < xM_size / 2; i++, phasor *= step) {
        std::complex<double> tmp = sg(i);
        sg(i) = phasor * sg(i + xM_size / 2);
        sg(i + xM_size / 2) = phasor * step2 * tmp;
    }

    // Perform FFT
    const pocketfft::shape_t shape = { static_cast<size_t>(xM_size) };
    const ptrdiff_t cpx_size = sizeof(std::complex<double>);
    const pocketfft::stride_t stride = { sg.stride[0] * cpx_size };
    pocketfft::c2c(shape, stride, stride, { 0 }, pocketfft::BACKWARD,
                   sg.ptr, sg.ptr, 1. / xM_size);

}
