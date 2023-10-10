#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"

#include "pocketfft_hdronly.h"
#include "sdp_pswf.h"
#include "sdp_swiftly.h"

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
        sdp_Error* status
)
{
    if (*status) return NULL;

    // Sanity-check sizes
    if (image_size <= 0 || xM_size <= 0 || yN_size <= 0)
    {
        SDP_LOG_ERROR("sdp_swiftly_create: Negative size passed.");
        *status = SDP_ERR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (image_size % xM_size != 0)
    {
        SDP_LOG_ERROR(
                "sdp_swiftly_create: Image size not divisible by subgrid size."
        );
        *status = SDP_ERR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (image_size % yN_size != 0)
    {
        SDP_LOG_ERROR(
                "sdp_swiftly_create: Image size not divisible by facet size."
        );
        *status = SDP_ERR_INVALID_ARGUMENT;
        return nullptr;
    }
    if ((xM_size * yN_size) % image_size != 0)
    {
        SDP_LOG_ERROR("sdp_swiftly_create: Contribution size not integer.");
        *status = SDP_ERR_INVALID_ARGUMENT;
        return nullptr;
    }

    // Simplifying assumptions (could be lifted, but likely not worth it)
    if (xM_size % 2 != 0)
    {
        SDP_LOG_ERROR("sdp_swiftly_create: Subgrid size not even.");
        *status = SDP_ERR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (yN_size % 2 != 0)
    {
        SDP_LOG_ERROR("sdp_swiftly_create: Facet size not even.");
        *status = SDP_ERR_INVALID_ARGUMENT;
        return nullptr;
    }

    // Generate PSWF
    sdp_Mem* Fb_mem = sdp_mem_create(SDP_MEM_DOUBLE,
            SDP_MEM_CPU,
            1,
            &yN_size,
            status
    );
    if (*status) return NULL;
    sdp_generate_pswf(0, W * (M_PI / 2), Fb_mem, status);
    if (*status) { sdp_mem_free(Fb_mem); return NULL; }

    // Allocate Fn
    const int64_t xM_yN_size = (xM_size * yN_size) / image_size;
    sdp_Mem* Fn_mem = sdp_mem_create(SDP_MEM_DOUBLE,
            SDP_MEM_CPU,
            1,
            &xM_yN_size,
            status
    );
    if (*status) { sdp_mem_free(Fb_mem); return NULL; }

    // Generate Fn
    const double* pswf = static_cast<double*>(sdp_mem_data(Fb_mem));
    double* Fn = static_cast<double*>(sdp_mem_data(Fn_mem));
    const int xM_step = image_size / xM_size;
    const int Fn_offset = (yN_size / 2) % xM_step;
    for (int i = 0; i < xM_yN_size; i++)
    {
        Fn[i] = pswf[Fn_offset + i * xM_step];
    }

    // Generate Fb (overwriting PSWF inplace!)
    double* Fb = static_cast<double*>(sdp_mem_data(Fb_mem));
    for (int i = 1; i < yN_size; i++)
    {
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


int64_t sdp_swiftly_get_image_size(sdp_SwiFTly* swiftly)
{
    return swiftly->image_size;
}


int64_t sdp_swiftly_get_facet_size(sdp_SwiFTly* swiftly)
{
    return swiftly->yN_size;
}


int64_t sdp_swiftly_get_subgrid_size(sdp_SwiFTly* swiftly)
{
    return swiftly->xM_size;
}


void sdp_swiftly_free(sdp_SwiFTly* swiftly)
{
    if (swiftly)
    {
        sdp_mem_free(swiftly->Fb);
        sdp_mem_free(swiftly->Fn);
    }
}


inline int64_t mod_p(int64_t a, int64_t b)
{
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
        sdp_Mem* facet,
        sdp_Mem* prep_facet_out,
        int64_t facet_offset,
        sdp_Error* status
)
{
    if (*status) return;

    // Parameter checks
    const int64_t yN_size = swiftly->yN_size;
    if (sdp_mem_num_dims(facet) > 1 && sdp_mem_shape_dim(facet, 1) > yN_size)
    {
        SDP_LOG_ERROR("Facet data too large (%d>%d)!",
                sdp_mem_shape_dim(facet, 1), yN_size
        );
        *status = SDP_ERR_INVALID_ARGUMENT;
    }
    sdp_MemViewCpu<const std::complex<double>, 2> fct;
    sdp_mem_check_and_view(facet, &fct, status);
    sdp_MemViewCpu<std::complex<double>, 2> out;
    sdp_mem_check_and_view(prep_facet_out, &out, status);
    sdp_mem_check_shape_dim(prep_facet_out, 1, yN_size, status);
    sdp_mem_check_same_shape(facet, 0, prep_facet_out, 0, status);
    sdp_MemViewCpu<double, 1> Fb;
    sdp_mem_check_and_view(swiftly->Fb, &Fb, status);
    if (*status) return;

    // We shift the facet centre to its correct global position modulo
    // yN_size (non-centered for FFT, i.e. apply ifftshift). Determine
    // start & end index accordingly.
    const int64_t start = mod_p(facet_offset - fct.shape[1] / 2,  yN_size);
    const int64_t end = (start + fct.shape[1]) % yN_size;
    const int64_t Fb_off = (yN_size / 2 - fct.shape[1] / 2) - start;

    // Broadcast along first axis
    const int64_t bc0_size = out.shape[0]; int64_t i0;
    for (i0 = 0; i0 < bc0_size; i0++)
    {
        // Does the facet data alias around the edges?
        int64_t i;
        if (start < end)
        {
            // Establish something along the lines of "00<data>00000"
            for (i = 0; i < start; i++)
            {
                out(i0, i) = 0;
            }
            for (i = start; i < end; i++)
            {
                out(i0, i) = fct(i0, i - start) * Fb(i + Fb_off);
            }
            for (i = end; i < yN_size; i++)
            {
                out(i0, i) = 0;
            }
        }
        else
        {
            // ... or alternatively "ta>00000<da"
            for (i = 0; i < end; i++)
            {
                out(i0, i) = fct(i0, i + yN_size - start) * Fb(
                        i + yN_size + Fb_off
                );
            }
            for (i = end; i < start; i++)
            {
                out(i0, i) = 0;
            }
            for (i = start; i < yN_size; i++)
            {
                out(i0, i) = fct(i0, i - start) * Fb(i + Fb_off);
            }
        }
    }

    // Perform FFT(s)
    const pocketfft::shape_t shape = {
        static_cast<size_t>(bc0_size),
        static_cast<size_t>(yN_size)
    };
    const ptrdiff_t cpx_size = sizeof(std::complex<double>);
    const pocketfft::stride_t stride = {
        out.stride[0] * cpx_size,
        out.stride[1] * cpx_size
    };
    pocketfft::c2c(shape, stride, stride, { 1 }, pocketfft::BACKWARD,
            out.ptr, out.ptr, 1. / yN_size
    );

    // missing shift will be corrected for in sdp_swiftly_extract_from_facet
}


void sdp_swiftly_extract_from_facet(
        sdp_SwiFTly* swiftly,
        sdp_Mem* prep_facet,
        sdp_Mem* contribution_out,
        int64_t subgrid_offset,
        sdp_Error* status
)
{
    if (*status) return;

    // Parameter checks
    const int64_t yN_size = swiftly->yN_size;
    const int64_t image_size = swiftly->image_size;
    const int64_t xM_yN_size = (swiftly->xM_size * yN_size) / image_size;
    sdp_MemViewCpu<const std::complex<double>, 2> fct;
    sdp_mem_check_and_view(prep_facet, &fct, status);
    sdp_mem_check_shape_dim(prep_facet, 1, yN_size, status);
    sdp_MemViewCpu<std::complex<double>, 2> out;
    sdp_mem_check_and_view(contribution_out, &out, status);
    sdp_mem_check_shape_dim(contribution_out, 1, xM_yN_size, status);
    sdp_mem_check_same_shape(prep_facet, 0, contribution_out, 0, status);
    if (*status) return;

    // Calculate grid offsets (still in yN_size image
    // resolution). This is taking into account the missing fftshift
    // from sdp_swiftly_prepare_facet.
    const int64_t sg_offs = subgrid_offset / (image_size / yN_size);
    const int64_t aliased_sg_offs = mod_p(sg_offs - xM_yN_size / 2, xM_yN_size);
    const int64_t offs = sg_offs - aliased_sg_offs - xM_yN_size / 2;
    const int64_t offs1 = mod_p(offs + xM_yN_size, yN_size);
    const int64_t offs2 = mod_p(offs, yN_size);

    // Broadcast along first axis
    const int64_t bc0_size = out.shape[0]; int64_t i0;
    for (i0 = 0; i0 < bc0_size; i0++)
    {
        int64_t i = 0;
        int64_t stop1 = yN_size - offs1;
        if (stop1 > aliased_sg_offs) stop1 = aliased_sg_offs;
        for ( ; i < stop1; i++)
        {
            out(i0, i) = fct(i0, i + offs1);
        }
        for ( ; i < aliased_sg_offs; i++)
        {
            out(i0, i) = fct(i0, i + offs1 - yN_size);
        }

        int64_t stop2 = yN_size - offs2;
        if (stop2 > xM_yN_size) stop2 = xM_yN_size;
        for ( ; i < stop2; i++)
        {
            out(i0, i) = fct(i0, i + offs2);
        }
        for ( ; i < xM_yN_size; i++)
        {
            out(i0, i) = fct(i0, i + offs2 - yN_size);
        }
    }

    // Perform FFT(s)
    const pocketfft::shape_t shape = {
        static_cast<size_t>(bc0_size),
        static_cast<size_t>(xM_yN_size)
    };
    const ptrdiff_t cpx_size = sizeof(std::complex<double>);
    const pocketfft::stride_t stride = {
        out.stride[0] * cpx_size,
        out.stride[1] * cpx_size
    };
    pocketfft::c2c(shape, stride, stride, { 1 }, pocketfft::FORWARD,
            out.ptr, out.ptr, 1.
    );
}


void sdp_swiftly_add_to_subgrid(
        sdp_SwiFTly* swiftly,
        sdp_Mem* contribution,
        sdp_Mem* subgrid_image_inout,
        int64_t facet_offset,
        sdp_Error* status
)
{
    if (*status) return;

    // Parameter checks
    const int64_t xM_size = swiftly->xM_size;
    const int64_t image_size = swiftly->image_size;
    const int64_t xM_yN_size = (xM_size * swiftly->yN_size) / image_size;
    sdp_MemViewCpu<const std::complex<double>, 2> contrib;
    sdp_mem_check_and_view(contribution, &contrib, status);
    sdp_mem_check_shape_dim(contribution, 1, xM_yN_size, status);
    sdp_MemViewCpu<std::complex<double>, 2> out;
    sdp_mem_check_and_view(subgrid_image_inout, &out, status);
    sdp_mem_check_shape_dim(subgrid_image_inout, 1, xM_size, status);
    sdp_mem_check_same_shape(contribution, 0, subgrid_image_inout, 0, status);
    sdp_MemViewCpu<double, 1> Fn;
    sdp_mem_check_and_view(swiftly->Fn, &Fn, status);
    if (*status) return;

    // Calculate facet offsets (in xM_size resolution).
    const int64_t fct_offs =
            mod_p(facet_offset, image_size) / (image_size / xM_size);
    const int64_t offs = mod_p(-xM_yN_size / 2 + xM_size / 2 + fct_offs,
            xM_size
    );

    // Broadcast along first axis
    const int64_t bc0_size = out.shape[0]; int64_t i0;
    for (i0 = 0; i0 < bc0_size; i0++)
    {
        int64_t i = 0;
        int64_t stop1 = xM_size - offs;
        if (stop1 > xM_yN_size) stop1 = xM_yN_size;
        for (i = 0; i < stop1; i++)
        {
            out(i0, i + offs) = Fn(i) *
                    contrib(i0, (i + fct_offs + xM_yN_size / 2) % xM_yN_size);
        }
        for (; i < xM_yN_size; i++)
        {
            out(i0, i + offs - xM_size) = Fn(i) *
                    contrib(i0, (i + fct_offs + xM_yN_size / 2) % xM_yN_size);
        }
    }
}


void sdp_swiftly_add_to_subgrid_2d(
        sdp_SwiFTly* swiftly,
        sdp_Mem* contribution,
        sdp_Mem* subgrid_image_inout,
        int64_t facet_offset0,
        int64_t facet_offset1,
        sdp_Error* status
)
{
    if (*status) return;

    // Parameter checks
    const int64_t xM_size = swiftly->xM_size;
    const int64_t image_size = swiftly->image_size;
    const int64_t xM_yN_size = (xM_size * swiftly->yN_size) / image_size;
    sdp_MemViewCpu<const std::complex<double>, 2> contrib;
    sdp_mem_check_and_view(contribution, &contrib, status);
    sdp_mem_check_shape_dim(contribution, 1, xM_yN_size, status);
    sdp_MemViewCpu<std::complex<double>, 2> out;
    sdp_mem_check_and_view(subgrid_image_inout, &out, status);
    sdp_mem_check_shape_dim(subgrid_image_inout, 1, xM_size, status);
    sdp_MemViewCpu<double, 1> Fn;
    sdp_mem_check_and_view(swiftly->Fn, &Fn, status);
    if (*status) return;

    // Calculate facet offsets (in xM_size resolution).
    const int64_t fct_offs0 = facet_offset0 / (image_size / xM_size);
    const int64_t offs0 = mod_p(-xM_yN_size / 2 + xM_size / 2 + fct_offs0,
            xM_size
    );
    const int64_t fct_offs1 = facet_offset1 / (image_size / xM_size);
    const int64_t offs1 = mod_p(-xM_yN_size / 2 + xM_size / 2 + fct_offs1,
            xM_size
    );

    int64_t stop0 = xM_size - offs0;
    if (stop0 > xM_yN_size) stop0 = xM_yN_size;
    int64_t stop1 = xM_size - offs1;
    if (stop1 > xM_yN_size) stop1 = xM_yN_size;

    int64_t i0, i1;
    for (i0 = 0; i0 < stop0; i0++)
    {
        double fn0 = Fn(i0);
        for (i1 = 0; i1 < stop0; i1++)
        {
            out(i0 + offs0, i1 + offs1) = fn0 * Fn(i1) *
                    contrib((i0 + fct_offs0 + xM_yN_size / 2) % xM_yN_size,
                    (i1 + fct_offs1 + xM_yN_size / 2) % xM_yN_size
                    );
        }
        for (; i1 < xM_yN_size; i1++)
        {
            out(i0 + offs0, i1 + offs1 - xM_size) = fn0 * Fn(i1) *
                    contrib((i0 + fct_offs0 + xM_yN_size / 2) % xM_yN_size,
                    (i1 + fct_offs1 + xM_yN_size / 2) % xM_yN_size
                    );
        }
    }
    for (; i0 < xM_yN_size; i0++)
    {
        double fn0 = Fn(i0);
        for (i1 = 0; i1 < stop0; i1++)
        {
            out(i0 + offs0 - xM_size, i1 + offs1) = fn0 * Fn(i1) *
                    contrib((i0 + fct_offs0 + xM_yN_size / 2) % xM_yN_size,
                    (i1 + fct_offs1 + xM_yN_size / 2) % xM_yN_size
                    );
        }
        for (; i1 < xM_yN_size; i1++)
        {
            out(i0 + offs0 - xM_size, i1 + offs1 - xM_size) = fn0 * Fn(i1) *
                    contrib((i0 + fct_offs0 + xM_yN_size / 2) % xM_yN_size,
                    (i1 + fct_offs1 + xM_yN_size / 2) % xM_yN_size
                    );
        }
    }
}


void sdp_swiftly_finish_subgrid_inplace(
        sdp_SwiFTly* swiftly,
        sdp_Mem* subgrid_inout,
        int64_t subgrid_offset,
        sdp_Error* status
)
{
    if (*status) return;

    // Parameter checks
    const int64_t xM_size = swiftly->xM_size;
    sdp_MemViewCpu<std::complex<double>, 2> sg;
    sdp_mem_check_and_view(subgrid_inout, &sg, status);
    sdp_mem_check_shape_dim(subgrid_inout, 1, xM_size, status);
    if (*status) return;
    const int64_t bc0_size = sg.shape[0];

    // Perform FFT shift, broadcasting along first axis
    int i, i0;
    for (i0 = 0; i0 < bc0_size; i0++)
    {
        for (i = 0; i < xM_size / 2; i++)
        {
            std::complex<double> tmp = sg(i0, i);
            sg(i0, i) = sg(i0, i + xM_size / 2);
            sg(i0, i + xM_size / 2) = tmp;
        }
    }

    // Perform FFT to temporary memory
    std::complex<double>* tmp = static_cast<std::complex<double>*>(
        alloca(bc0_size * xM_size * sizeof(std::complex<double>)));
    const pocketfft::shape_t shape = {
        static_cast<size_t>(bc0_size),
        static_cast<size_t>(xM_size)
    };
    const ptrdiff_t cpx_size = sizeof(std::complex<double>);
    const pocketfft::stride_t stride = {
        sg.stride[0] * cpx_size,
        sg.stride[1] * cpx_size
    };
    const pocketfft::stride_t stride2 = {
        xM_size* cpx_size, cpx_size
    };
    pocketfft::c2c(shape, stride, stride2, { 1 }, pocketfft::BACKWARD,
            sg.ptr, tmp, 1. / xM_size
    );

    // Move back, applying the subgrid offset
    for (i0 = 0; i0 < bc0_size; i0++)
    {
        for (i = 0; i < xM_size; i++)
        {
            sg(i0,
                    i
            ) =
                    tmp[i0 * xM_size +
                            mod_p(i + subgrid_offset + xM_size / 2, xM_size)];
        }
    }
}


void sdp_swiftly_finish_subgrid(
        sdp_SwiFTly* swiftly,
        sdp_Mem* subgrid_image,
        sdp_Mem* subgrid_out,
        int64_t subgrid_offset,
        sdp_Error* status
)
{
    if (*status) return;

    // Parameter checks
    const int64_t xM_size = swiftly->xM_size;
    sdp_MemViewCpu<const std::complex<double>, 2> sg_img;
    sdp_mem_check_and_view(subgrid_image, &sg_img, status);
    sdp_mem_check_shape_dim(subgrid_image, 1, xM_size, status);
    sdp_MemViewCpu<std::complex<double>, 2> sg;
    sdp_mem_check_and_view(subgrid_out, &sg, status);
    if (sdp_mem_num_dims(subgrid_out) > 1 &&
            sdp_mem_shape_dim(subgrid_out, 1) > xM_size)
    {
        SDP_LOG_ERROR("Subgrid data too large (%d>%d)!",
                sdp_mem_shape_dim(subgrid_out, 1), xM_size
        );
        *status = SDP_ERR_INVALID_ARGUMENT;
    }
    sdp_mem_check_same_shape(subgrid_image, 0, subgrid_out, 0, status);
    if (*status) return;

    // Allocate temporary memory
    sdp_Mem* buf_mem = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU,
            2, sg_img.shape, status
    );
    if (*status) return;
    sdp_MemViewCpu<std::complex<double>, 2> buf;
    sdp_mem_check_and_view(buf_mem, &buf, status);
    assert(*status);

    // Perform FFT shift to temporary memory
    const int64_t bc0_size = sg.shape[0];
    int i, i0;
    for (i0 = 0; i0 < bc0_size; i0++)
    {
        for (i = 0; i < xM_size / 2; i++)
        {
            buf(i0, i) = sg_img(i0, i + xM_size / 2);
            buf(i0, i + xM_size / 2) = sg_img(i0, i);
        }
    }

    // Perform in-place FFT
    const pocketfft::shape_t shape = {
        static_cast<size_t>(bc0_size),
        static_cast<size_t>(xM_size)
    };
    const ptrdiff_t cpx_size = sizeof(std::complex<double>);
    const pocketfft::stride_t stride = {
        buf.stride[0] * cpx_size,
        buf.stride[1] * cpx_size
    };
    pocketfft::c2c(shape, stride, stride, { 1 }, pocketfft::BACKWARD,
            buf.ptr, buf.ptr, 1. / xM_size
    );

    // Move back portion we are interested in, applying the subgrid offset
    const int64_t xA_size = sdp_mem_shape_dim(subgrid_out, 1);
    for (i0 = 0; i0 < bc0_size; i0++)
    {
        for (i = 0; i < xA_size; i++)
        {
            int64_t j = mod_p(i - xA_size / 2 + subgrid_offset + xM_size,
                    xM_size
            );
            sg(i0, i) = buf(i0, j);
        }
    }

    sdp_mem_free(buf_mem);
}


void sdp_swiftly_finish_subgrid_inplace_2d(
        sdp_SwiFTly* swiftly,
        sdp_Mem* subgrid_inout,
        int64_t subgrid_offset0,
        int64_t subgrid_offset1,
        sdp_Error* status
)
{
    if (*status) return;

    // Parameter checks
    const int64_t xM_size = swiftly->xM_size;
    sdp_MemViewCpu<std::complex<double>, 2> sg;
    sdp_mem_check_and_view(subgrid_inout, &sg, status);
    sdp_mem_check_shape_dim(subgrid_inout, 0, xM_size, status);
    sdp_mem_check_shape_dim(subgrid_inout, 1, xM_size, status);
    if (*status) return;

    // Perform FFT shift on input
    int64_t i0, i1;
    if (sg.shape[1] != 1)
    {
        for (i0 = 0; i0 < xM_size / 2; i0++)
        {
#pragma GCC ivdep
            for (i1 = 0; i1 < xM_size / 2; i1++)
            {
                std::complex<double> tmp = sg(i0, i1);
                sg(i0, i1) = sg(i0 + xM_size / 2, i1 + xM_size / 2);
                sg(i0 + xM_size / 2, i1 + xM_size / 2) = tmp;
                std::complex<double> tmp2 = sg(i0 + xM_size / 2, i1);
                sg(i0 + xM_size / 2, i1) = sg(i0, i1 + xM_size / 2);
                sg(i0, i1 + xM_size / 2) = tmp2;
            }
        }
    }
    else
    {
        for (i0 = 0; i0 < xM_size / 2; i0++)
        {
#pragma GCC ivdep
#pragma GCC unroll 2
            for (i1 = 0; i1 < xM_size / 2; i1++)
            {
                std::complex<double> tmp = sg(i0, i1);
                sg(i0, i1) = sg(i0 + xM_size / 2, i1 + xM_size / 2);
                sg(i0 + xM_size / 2, i1 + xM_size / 2) = tmp;
                std::complex<double> tmp2 = sg(i0 + xM_size / 2, i1);
                sg(i0 + xM_size / 2, i1) = sg(i0, i1 + xM_size / 2);
                sg(i0, i1 + xM_size / 2) = tmp2;
            }
        }
    }

    // Perform FFT to temporary memory
    std::complex<double>* tmp = static_cast<std::complex<double>*>(
        alloca(xM_size * xM_size * sizeof(std::complex<double>)));
    const pocketfft::shape_t shape = {
        static_cast<size_t>(xM_size),
        static_cast<size_t>(xM_size)
    };
    const ptrdiff_t cpx_size = sizeof(std::complex<double>);
    const pocketfft::stride_t stride = {
        sg.stride[0] * cpx_size,
        sg.stride[1] * cpx_size
    };
    const pocketfft::stride_t stride_tmp = {
        xM_size* cpx_size, cpx_size
    };
    pocketfft::c2c(shape, stride, stride, { 0, 1 }, pocketfft::BACKWARD,
            sg.ptr, tmp, 1. / xM_size / xM_size
    );

    // Add subgrid offset on output
    int64_t off = mod_p(subgrid_offset1 + xM_size / 2, xM_size);
    for (i0 = 0; i0 < xM_size; i0++)
    {
        int64_t _i0 =
                mod_p(i0 + subgrid_offset0 + xM_size / 2, xM_size) * xM_size;
        for (i1 = 0; i1 < xM_size - off; i1++)
        {
            sg(i0, i1) = tmp[_i0 + i1 + off];
        }
        for (; i1 < xM_size; i1++)
        {
            sg(i0, i1) = tmp[_i0 + i1 + off - xM_size];
        }
    }
}


// Precisely the inverse of sdp_swiftly_finish_subgrid_inplace
void sdp_swiftly_prepare_subgrid_inplace(
        sdp_SwiFTly* swiftly,
        sdp_Mem* subgrid_inout,
        int64_t subgrid_offset,
        sdp_Error* status
)
{
    if (*status) return;

    // Parameter checks
    const int64_t xM_size = swiftly->xM_size;
    sdp_MemViewCpu<std::complex<double>, 2> sg;
    sdp_mem_check_and_view(subgrid_inout, &sg, status);
    sdp_mem_check_shape_dim(subgrid_inout, 1, xM_size, status);
    if (*status) return;
    const int64_t bc0_size = sg.shape[0];

    // Perform FFT
    const pocketfft::shape_t shape = {
        static_cast<size_t>(bc0_size),
        static_cast<size_t>(xM_size)
    };
    const ptrdiff_t cpx_size = sizeof(std::complex<double>);
    const pocketfft::stride_t stride = {
        sg.stride[0] * cpx_size,
        sg.stride[1] * cpx_size
    };
    pocketfft::c2c(shape, stride, stride, { 1 }, pocketfft::FORWARD,
            sg.ptr, sg.ptr, 1.
    );

    // Perform shift for input and output. We use a phase ramp to do
    // all of this in-place - not sure whether that's truly the most
    // efficient option here.
    double phase_step = -2 * M_PI * (subgrid_offset + xM_size / 2) / xM_size;
    double phase_step2 = -M_PI * (subgrid_offset + xM_size / 2);
    std::complex<double> phasor(1.0);
    std::complex<double> step(cos(phase_step), sin(phase_step));
    std::complex<double> step2(cos(phase_step2), sin(phase_step2));
    int i;
    for (i = 0; i < xM_size / 2; i++, phasor *= step)
    {
        // Broadcast along first axis
        int64_t i0;
        for (i0 = 0; i0 < bc0_size; i0++)
        {
            std::complex<double> tmp = sg(i0, i);
            sg(i0, i) = phasor * sg(i0, i + xM_size / 2);
            sg(i0, i + xM_size / 2) = phasor * step2 * tmp;
        }
    }
}


// Precisely the inverse of sdp_swiftly_finish_subgrid_inplace_2d
void sdp_swiftly_prepare_subgrid_inplace_2d(
        sdp_SwiFTly* swiftly,
        sdp_Mem* subgrid_inout,
        int64_t subgrid_offset0,
        int64_t subgrid_offset1,
        sdp_Error* status
)
{
    if (*status) return;

    // Parameter checks
    const int64_t xM_size = swiftly->xM_size;
    sdp_MemViewCpu<std::complex<double>, 2> sg;
    sdp_mem_check_and_view(subgrid_inout, &sg, status);
    sdp_mem_check_shape_dim(subgrid_inout, 0, xM_size, status);
    sdp_mem_check_shape_dim(subgrid_inout, 1, xM_size, status);
    if (*status) return;

    // Perform FFT
    const pocketfft::shape_t shape = {
        static_cast<size_t>(xM_size),
        static_cast<size_t>(xM_size)
    };
    const ptrdiff_t cpx_size = sizeof(std::complex<double>);
    const pocketfft::stride_t stride = {
        sg.stride[0] * cpx_size,
        sg.stride[1] * cpx_size
    };
    pocketfft::c2c(shape, stride, stride, { 0, 1 }, pocketfft::FORWARD,
            sg.ptr, sg.ptr, 1.
    );

    // Perform shift for input and output. We use a phase ramp to do
    // all of this in-place - not sure whether that's truly the most
    // efficient option here.
    double phase_step0 = -2 * M_PI * (subgrid_offset0 + xM_size / 2) / xM_size;
    double phase_step0_2 = -M_PI * (subgrid_offset0 + xM_size / 2);
    double phase_step1 = -2 * M_PI * (subgrid_offset1 + xM_size / 2) / xM_size;
    double phase_step1_2 = -M_PI * (subgrid_offset1 + xM_size / 2);
    std::complex<double> phasor0(1.0);
    std::complex<double> step0(cos(phase_step0), sin(phase_step0));
    std::complex<double> step0_2(cos(phase_step0_2), sin(phase_step0_2));
    int64_t i0;
    for (i0 = 0; i0 < xM_size / 2; i0++, phasor0 *= step0)
    {
        std::complex<double> phasor1(phasor0);
        std::complex<double> step1(cos(phase_step1), sin(phase_step1));
        std::complex<double> step1_2(cos(phase_step1_2), sin(phase_step1_2));
        int64_t i1;
        for (i1 = 0; i1 < xM_size / 2; i1++, phasor1 *= step1)
        {
            std::complex<double> tmp = sg(i0, i1);
            sg(i0, i1) =
                    phasor1 * sg(i0 + xM_size / 2, i1 + xM_size / 2);
            sg(i0 + xM_size / 2, i1 + xM_size / 2) =
                    phasor1 * step0_2 * step1_2 * tmp;

            std::complex<double> tmp2 = sg(i0 + xM_size / 2, i1);
            sg(i0 + xM_size / 2, i1) =
                    phasor1 * step0_2 * sg(i0, i1 + xM_size / 2);
            sg(i0, i1 + xM_size / 2) =
                    phasor1 * step1_2 * tmp2;
        }
    }
}


// Effectively the inverse of sdp_swiftly_add_to_subgrid except that
// we multiply by Fn and do the FFT here instead of in
// sdp_swiftly_add_to_facet
void sdp_swiftly_extract_from_subgrid(
        sdp_SwiFTly* swiftly,
        sdp_Mem* subgrid_image,
        sdp_Mem* contribution_out,
        int64_t facet_offset,
        sdp_Error* status
)
{
    if (*status) return;

    // Parameter checks
    const int64_t xM_size = swiftly->xM_size;
    const int64_t image_size = swiftly->image_size;
    const int64_t xM_yN_size = (xM_size * swiftly->yN_size) / image_size;
    sdp_MemViewCpu<const std::complex<double>, 2> sg_img;
    sdp_mem_check_and_view(subgrid_image, &sg_img, status);
    sdp_mem_check_shape_dim(subgrid_image, 1, xM_size, status);
    sdp_MemViewCpu<std::complex<double>, 2> contrib;
    sdp_mem_check_and_view(contribution_out, &contrib, status);
    sdp_mem_check_shape_dim(contribution_out, 1, xM_yN_size, status);
    sdp_mem_check_same_shape(subgrid_image, 0, contribution_out, 0, status);
    sdp_MemViewCpu<double, 1> Fn;
    sdp_mem_check_and_view(swiftly->Fn, &Fn, status);
    if (*status) return;

    // Broadcast along first axis
    const int64_t bc0_size = sg_img.shape[0]; int64_t i0;
    for (i0 = 0; i0 < bc0_size; i0++)
    {
        // Calculate facet offsets (in xM_size resolution).
        const int64_t fct_offs = facet_offset / (image_size / xM_size);
        const int64_t offs = mod_p(-xM_yN_size / 2 + xM_size / 2 + fct_offs,
                xM_size
        );
        int64_t i = 0;
        int64_t stop1 = xM_size - offs;
        if (stop1 > xM_yN_size) stop1 = xM_yN_size;
        for (i = 0; i < stop1; i++)
        {
            contrib(i0, (i + fct_offs + xM_yN_size / 2) % xM_yN_size) =
                    sg_img(i0, i + offs) * Fn(i);
        }
        for (; i < xM_yN_size; i++)
        {
            contrib(i0, (i + fct_offs + xM_yN_size / 2) % xM_yN_size) =
                    sg_img(i0, i + offs - xM_size) * Fn(i);
        }
    }

    // Perform FFT(s)
    const pocketfft::shape_t shape = {
        static_cast<size_t>(bc0_size),
        static_cast<size_t>(xM_yN_size)
    };
    const ptrdiff_t cpx_size = sizeof(std::complex<double>);
    const pocketfft::stride_t stride = {
        contrib.stride[0] * cpx_size,
        contrib.stride[1] * cpx_size
    };
    pocketfft::c2c(shape, stride, stride, { 1 }, pocketfft::BACKWARD,
            contrib.ptr, contrib.ptr, 1. / xM_yN_size
    );
}


// Effectively the inverse of sdp_swiftly_add_to_subgrid_2d except
// that we multiply by Fn and do the FFT here instead of in
// sdp_swiftly_add_to_facet
void sdp_swiftly_extract_from_subgrid_2d(
        sdp_SwiFTly* swiftly,
        sdp_Mem* subgrid_image,
        sdp_Mem* contribution_out,
        int64_t facet_offset0,
        int64_t facet_offset1,
        sdp_Error* status
)
{
    if (*status) return;

    // Parameter checks
    const int64_t xM_size = swiftly->xM_size;
    const int64_t image_size = swiftly->image_size;
    const int64_t xM_yN_size = (xM_size * swiftly->yN_size) / image_size;
    sdp_MemViewCpu<const std::complex<double>, 2> sg_img;
    sdp_mem_check_and_view(subgrid_image, &sg_img, status);
    sdp_mem_check_shape_dim(subgrid_image, 1, xM_size, status);
    sdp_MemViewCpu<std::complex<double>, 2> contrib;
    sdp_mem_check_and_view(contribution_out, &contrib, status);
    sdp_mem_check_shape_dim(contribution_out, 1, xM_yN_size, status);
    sdp_MemViewCpu<double, 1> Fn;
    sdp_mem_check_and_view(swiftly->Fn, &Fn, status);
    if (*status) return;

    // Calculate facet offsets (in xM_size resolution).
    const int64_t fct_offs0 = facet_offset0 / (image_size / xM_size);
    const int64_t offs0 = mod_p(-xM_yN_size / 2 + xM_size / 2 + fct_offs0,
            xM_size
    );
    const int64_t fct_offs1 = facet_offset1 / (image_size / xM_size);
    const int64_t offs1 = mod_p(-xM_yN_size / 2 + xM_size / 2 + fct_offs1,
            xM_size
    );

    int64_t stop0 = xM_size - offs0;
    if (stop0 > xM_yN_size) stop0 = xM_yN_size;
    int64_t stop1 = xM_size - offs1;
    if (stop1 > xM_yN_size) stop1 = xM_yN_size;

    int64_t i0, i1;
    for (i0 = 0; i0 < stop0; i0++)
    {
        double fn0 = Fn(i0);
        for (i1 = 0; i1 < stop0; i1++)
        {
            contrib((i0 + fct_offs0 + xM_yN_size / 2) % xM_yN_size,
                    (i1 + fct_offs1 + xM_yN_size / 2) % xM_yN_size
            ) =
                    fn0 * Fn(i1) * sg_img(i0 + offs0, i1 + offs1);
        }
        for (; i1 < xM_yN_size; i1++)
        {
            contrib((i0 + fct_offs0 + xM_yN_size / 2) % xM_yN_size,
                    (i1 + fct_offs1 + xM_yN_size / 2) % xM_yN_size
            ) =
                    fn0 * Fn(i1) * sg_img(i0 + offs0, i1 + offs1 - xM_size);
        }
    }
    for (; i0 < xM_yN_size; i0++)
    {
        double fn0 = Fn(i0);
        for (i1 = 0; i1 < stop0; i1++)
        {
            contrib((i0 + fct_offs0 + xM_yN_size / 2) % xM_yN_size,
                    (i1 + fct_offs1 + xM_yN_size / 2) % xM_yN_size
            ) =
                    fn0 * Fn(i1) * sg_img(i0 + offs0 - xM_size, i1 + offs1);
        }
        for (; i1 < xM_yN_size; i1++)
        {
            contrib((i0 + fct_offs0 + xM_yN_size / 2) % xM_yN_size,
                    (i1 + fct_offs1 + xM_yN_size / 2) % xM_yN_size
            ) =
                    fn0 * Fn(i1) * sg_img(i0 + offs0 - xM_size,
                    i1 + offs1 - xM_size
                    );
        }
    }

    // Perform FFT(s)
    const pocketfft::shape_t shape = {
        static_cast<size_t>(xM_yN_size),
        static_cast<size_t>(xM_yN_size)
    };
    const ptrdiff_t cpx_size = sizeof(std::complex<double>);
    const pocketfft::stride_t stride = {
        contrib.stride[0] * cpx_size,
        contrib.stride[1] * cpx_size
    };
    pocketfft::c2c(shape, stride, stride, { 0, 1 }, pocketfft::BACKWARD,
            contrib.ptr, contrib.ptr, 1. / xM_yN_size / xM_yN_size
    );
}


// Effectively the inverse of sdp_swiftly_extract_from_facet except
// that the FFT has already been done in
// sdp_swiftly_extract_from_subgrid[_2d]
void sdp_swiftly_add_to_facet(
        sdp_SwiFTly* swiftly,
        sdp_Mem* contribution,
        sdp_Mem* prep_facet_inout,
        int64_t subgrid_offset,
        sdp_Error* status
)
{
    if (*status) return;

    // Parameter checks
    const int64_t yN_size = swiftly->yN_size;
    const int64_t image_size = swiftly->image_size;
    const int64_t xM_yN_size = (swiftly->xM_size * yN_size) / image_size;
    sdp_MemViewCpu<const std::complex<double>, 2> contrib;
    sdp_mem_check_and_view(contribution, &contrib, status);
    sdp_mem_check_shape_dim(contribution, 1, xM_yN_size, status);
    sdp_MemViewCpu<std::complex<double>, 2> fct;
    sdp_mem_check_and_view(prep_facet_inout, &fct, status);
    sdp_mem_check_shape_dim(prep_facet_inout, 1, yN_size, status);
    sdp_mem_check_same_shape(contribution, 0, prep_facet_inout, 0, status);
    if (*status) return;

    // Calculate grid offsets (still in yN_size image
    // resolution). This is taking into account the missing fftshift
    // from sdp_swiftly_finish_facet.
    const int64_t sg_offs = subgrid_offset / (image_size / yN_size);
    const int64_t aliased_sg_offs = mod_p(sg_offs - xM_yN_size / 2, xM_yN_size);
    const int64_t offs = sg_offs - aliased_sg_offs - xM_yN_size / 2;
    const int64_t offs1 = mod_p(offs + xM_yN_size, yN_size);
    const int64_t offs2 = mod_p(offs, yN_size);

    // Broadcast along first axis
    const int64_t bc0_size = contrib.shape[0]; int64_t i0;
    for (i0 = 0; i0 < bc0_size; i0++)
    {
        int64_t i = 0;
        int64_t stop1 = yN_size - offs1;
        if (stop1 > aliased_sg_offs) stop1 = aliased_sg_offs;
        for ( ; i < stop1; i++)
        {
            fct(i0, i + offs1) = contrib(i0, i);
        }
        for ( ; i < aliased_sg_offs; i++)
        {
            fct(i0, i + offs1 - yN_size) = contrib(i0, i);
        }

        int64_t stop2 = yN_size - offs2;
        if (stop2 > xM_yN_size) stop2 = xM_yN_size;
        for ( ; i < stop2; i++)
        {
            fct(i0, i + offs2) = contrib(i0, i);
        }
        for ( ; i < xM_yN_size; i++)
        {
            fct(i0, i + offs2 - yN_size) = contrib(i0, i);
        }
    }
}


void sdp_swiftly_finish_facet(
        sdp_SwiFTly* swiftly,
        sdp_Mem* prep_facet_inout,
        sdp_Mem* facet_out,
        int64_t facet_offset,
        sdp_Error* status
)
{
    if (*status) return;

    // Parameter checks
    const int64_t yN_size = swiftly->yN_size;
    if (sdp_mem_num_dims(facet_out) > 1 &&
            sdp_mem_shape_dim(facet_out, 1) > yN_size)
    {
        SDP_LOG_ERROR("Facet data too large (%d>%d)!",
                sdp_mem_shape_dim(facet_out, 1), yN_size
        );
        *status = SDP_ERR_INVALID_ARGUMENT;
    }
    sdp_MemViewCpu<std::complex<double>, 2> pfct;
    sdp_mem_check_and_view(prep_facet_inout, &pfct, status);
    sdp_mem_check_shape_dim(prep_facet_inout, 1, yN_size, status);
    sdp_MemViewCpu<std::complex<double>, 2> fct;
    sdp_mem_check_and_view(facet_out, &fct, status);
    sdp_mem_check_same_shape(facet_out, 0, prep_facet_inout, 0, status);
    sdp_MemViewCpu<double, 1> Fb;
    sdp_mem_check_and_view(swiftly->Fb, &Fb, status);
    if (*status) return;

    // Perform FFT(s)
    const int64_t bc0_size = pfct.shape[0];
    const pocketfft::shape_t shape = {
        static_cast<size_t>(bc0_size),
        static_cast<size_t>(yN_size)
    };
    const ptrdiff_t cpx_size = sizeof(std::complex<double>);
    const pocketfft::stride_t stride = {
        pfct.stride[0] * cpx_size,
        pfct.stride[1] * cpx_size
    };
    pocketfft::c2c(shape, stride, stride, { 1 }, pocketfft::FORWARD,
            pfct.ptr, pfct.ptr, 1.
    );

    // We shift the facet centre to its correct global position modulo
    // yN_size (non-centered for FFT, i.e. apply ifftshift). Determine
    // start & end index accordingly.
    const int64_t start = mod_p(facet_offset - fct.shape[1] / 2,  yN_size);
    const int64_t end = (start + fct.shape[1]) % yN_size;
    const int64_t Fb_off = (yN_size / 2 - fct.shape[1] / 2) - start;

    // Broadcast along first axis
    int64_t i0;
    for (i0 = 0; i0 < bc0_size; i0++)
    {
        // Does the facet data alias around the edges?
        int64_t i;
        if (start < end)
        {
            for (i = start; i < end; i++)
            {
                fct(i0, i - start) = pfct(i0, i) * Fb(i + Fb_off);
            }
        }
        else
        {
            for (i = 0; i < end; i++)
            {
                fct(i0, i + yN_size - start) = pfct(i0, i) * Fb(
                        i + yN_size + Fb_off
                );
            }
            for (i = start; i < yN_size; i++)
            {
                fct(i0, i - start) = pfct(i0, i) * Fb(i + Fb_off);
            }
        }
    }
}
