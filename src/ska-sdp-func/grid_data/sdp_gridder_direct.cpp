/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>
#include <cstdlib>

#include "ska-sdp-func/fourier_transforms/sdp_pswf.h"
#include "ska-sdp-func/grid_data/sdp_gridder_direct.h"
#include "ska-sdp-func/grid_data/sdp_gridder_grid_correct.h"
#include "ska-sdp-func/grid_data/sdp_gridder_utils.h"
#include "ska-sdp-func/math/sdp_math_macros.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"

using std::complex;

struct sdp_GridderDirect
{
    int image_size;
    int subgrid_size;
    double theta;
    double w_step;
    double shear_u;
    double shear_v;
    int support;
    sdp_Mem* pswf_sg;
};

// Begin anonymous namespace for file-local functions.
namespace {

// Local function to generate subimage by direct FT.
template<
        typename DIR_TYPE,
        typename FLUX_TYPE,
        typename UVW_TYPE,
        typename VIS_TYPE
>
void idft(
        sdp_GridderDirect* plan,
        const sdp_Mem* uvw,
        const sdp_Mem* vis,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        const sdp_Mem* lmn,
        int subgrid_offset_u,
        int subgrid_offset_v,
        int subgrid_offset_w,
        double freq0_hz,
        double dfreq_hz,
        sdp_Mem* image,
        sdp_Error* status
)
{
    if (*status) return;

    // Get views to data.
    const int64_t num_uvw = sdp_mem_shape_dim(uvw, 0);
    const int64_t image_size = sdp_mem_shape_dim(image, 0);
    const int num_chan = (int) sdp_mem_shape_dim(vis, 1);
    sdp_MemViewCpu<const int, 1> start_chs_, end_chs_;
    sdp_MemViewCpu<const UVW_TYPE, 2> uvw_;
    sdp_MemViewCpu<const complex<VIS_TYPE>, 2> vis_;
    sdp_MemViewCpu<const DIR_TYPE, 2> lmn_;
    sdp_MemViewCpu<FLUX_TYPE, 2> image_;
    sdp_mem_check_and_view(start_chs, &start_chs_, status);
    sdp_mem_check_and_view(end_chs, &end_chs_, status);
    sdp_mem_check_and_view(uvw, &uvw_, status);
    sdp_mem_check_and_view(vis, &vis_, status);
    sdp_mem_check_and_view(lmn, &lmn_, status);
    sdp_mem_check_and_view(image, &image_, status);
    const double* pswf = (const double*) sdp_mem_data_const(plan->pswf_sg);
    if (*status) return;

    // Scale subgrid offset values.
    const double du = (double) subgrid_offset_u / plan->theta;
    const double dv = (double) subgrid_offset_v / plan->theta;
    const double dw = (double) subgrid_offset_w * plan->w_step;

    // Loop over pixels.
    #pragma omp parallel for collapse(2)
    for (int64_t il = 0; il < image_size; ++il)
    {
        for (int64_t im = 0; im < image_size; ++im)
        {
            FLUX_TYPE local_pix = 0;
            const int64_t s = il * image_size + im; // Linearised pixel index.

            // Loop over uvw values.
            for (int64_t i = 0; i < num_uvw; ++i)
            {
                // Skip if there's no visibility to grid.
                if (start_chs_(i) >= end_chs_(i)) continue;

                // Loop over channels.
                for (int c = 0; c < num_chan; ++c)
                {
                    const double inv_wave = (freq0_hz + dfreq_hz * c) / C_0;

                    // Scale and shift uvws.
                    const double u = uvw_(i, 0) * inv_wave - du;
                    const double v = uvw_(i, 1) * inv_wave - dv;
                    const double w = uvw_(i, 2) * inv_wave - dw;

                    const double phase = 2.0 * M_PI *
                            (lmn_(s, 0) * u + lmn_(s, 1) * v + lmn_(s, 2) * w);
                    const complex<VIS_TYPE> phasor(cos(phase), sin(phase));
                    local_pix += vis_(i, c) * phasor;
                }
            }

            // Store local pixel value, appropriately tapered by the PSWF.
            // (We can't taper the whole image, as the input may be nonzero.)
            image_(il, im) += local_pix * (FLUX_TYPE) (pswf[il] * pswf[im]);
        }
    }
}

} // End anonymous namespace for file-local functions.


sdp_GridderDirect* sdp_gridder_direct_create(
        int image_size,
        int subgrid_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        int support,
        sdp_Error* status
)
{
    sdp_GridderDirect* plan = (sdp_GridderDirect*)calloc(
            1, sizeof(sdp_GridderDirect)
    );
    plan->image_size = image_size;
    plan->subgrid_size = subgrid_size;
    plan->theta = theta;
    plan->w_step = w_step;
    plan->shear_u = shear_u;
    plan->shear_v = shear_v;
    plan->support = support;
    const int64_t pswf_sg_shape[] = {subgrid_size};
    plan->pswf_sg = sdp_mem_create(
            SDP_MEM_DOUBLE, SDP_MEM_CPU, 1, pswf_sg_shape, status
    );
    sdp_generate_pswf(0, support * (M_PI / 2), plan->pswf_sg, status);
    ((double*)sdp_mem_data(plan->pswf_sg))[0] = 1e-15;
    return plan;
}


void sdp_gridder_direct_degrid(
        sdp_GridderDirect* plan,
        const sdp_Mem* subgrid_image,
        int subgrid_offset_u,
        int subgrid_offset_v,
        int subgrid_offset_w,
        double freq0_hz,
        double dfreq_hz,
        const sdp_Mem* uvws,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        sdp_Mem* vis,
        sdp_Error* status
)
{
    if (*status) return;

    // Count nonzero pixels in image.
    int64_t num_src = sdp_gridder_count_nonzero_pixels(subgrid_image, status);
    const sdp_MemType dir_type = sdp_mem_type(uvws);
    const sdp_MemLocation loc = sdp_mem_location(vis);
    const int64_t lmn_shape[] = {num_src, 3};
    sdp_Mem* flux = sdp_mem_create(SDP_MEM_DOUBLE, loc, 1, &num_src, status);
    sdp_Mem* lmn = sdp_mem_create(dir_type, loc, 2, lmn_shape, status);

    // Convert image into positions and flux values for all nonzero pixels,
    // and use DFT for degridding.
    sdp_gridder_image_to_flmn(
            subgrid_image, plan->theta, plan->shear_u, plan->shear_v,
            plan->pswf_sg, flux, lmn, status
    );
    sdp_gridder_dft(uvws, start_chs, end_chs, flux, lmn,
            subgrid_offset_u, subgrid_offset_v, subgrid_offset_w,
            plan->theta, plan->w_step, freq0_hz, dfreq_hz, vis, status
    );

    // Free scratch memory.
    sdp_mem_free(flux);
    sdp_mem_free(lmn);
}


void sdp_gridder_direct_degrid_correct(
        sdp_GridderDirect* plan,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        int w_offset,
        sdp_Error* status
)
{
    sdp_gridder_grid_correct_pswf(plan->image_size, plan->theta, plan->w_step,
            plan->shear_u, plan->shear_v, plan->support, 0,
            facet, facet_offset_l, facet_offset_m, status
    );
    if (sdp_mem_is_complex(facet))
    {
        sdp_gridder_grid_correct_w_stack(plan->image_size, plan->theta,
                plan->w_step, plan->shear_u, plan->shear_v,
                facet, facet_offset_l, facet_offset_m, w_offset, false, status
        );
    }
}


void sdp_gridder_direct_grid(
        sdp_GridderDirect* plan,
        const sdp_Mem* vis,
        const sdp_Mem* uvws,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        double freq0_hz,
        double dfreq_hz,
        sdp_Mem* subgrid_image,
        int subgrid_offset_u,
        int subgrid_offset_v,
        int subgrid_offset_w,
        sdp_Error* status
)
{
    if (*status) return;

    // Allocate space for coordinates of all pixels in the image.
    int64_t num_src = sdp_mem_num_elements(subgrid_image);
    const sdp_MemType img_type = sdp_mem_type(subgrid_image);
    const sdp_MemType dir_type = sdp_mem_type(uvws);
    const sdp_MemLocation loc = sdp_mem_location(vis);
    const int64_t lmn_shape[] = {num_src, 3};
    sdp_Mem* lmn = sdp_mem_create(dir_type, loc, 2, lmn_shape, status);

    // Convert image into positions, and use DFT for gridding.
    sdp_gridder_image_to_flmn(
            subgrid_image, plan->theta, plan->shear_u, plan->shear_v,
            NULL, NULL, lmn, status
    );
    if (img_type == SDP_MEM_COMPLEX_DOUBLE && dir_type == SDP_MEM_DOUBLE &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE)
    {
        idft<double, complex<double>, double, double>(
                plan, uvws, vis, start_chs, end_chs, lmn,
                subgrid_offset_u, subgrid_offset_v, subgrid_offset_w,
                freq0_hz, dfreq_hz, subgrid_image, status
        );
    }
    else if (img_type == SDP_MEM_COMPLEX_FLOAT && dir_type == SDP_MEM_FLOAT &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT)
    {
        idft<float, complex<float>, float, float>(
                plan, uvws, vis, start_chs, end_chs, lmn,
                subgrid_offset_u, subgrid_offset_v, subgrid_offset_w,
                freq0_hz, dfreq_hz, subgrid_image, status
        );
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
    }

    // Free scratch memory.
    sdp_mem_free(lmn);
}


void sdp_gridder_direct_grid_correct(
        sdp_GridderDirect* plan,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        int w_offset,
        sdp_Error* status
)
{
    sdp_gridder_grid_correct_pswf(plan->image_size, plan->theta, plan->w_step,
            plan->shear_u, plan->shear_v, plan->support, 0,
            facet, facet_offset_l, facet_offset_m, status
    );
    if (sdp_mem_is_complex(facet))
    {
        sdp_gridder_grid_correct_w_stack(plan->image_size, plan->theta,
                plan->w_step, plan->shear_u, plan->shear_v,
                facet, facet_offset_l, facet_offset_m, w_offset, true, status
        );
    }
}


void sdp_gridder_direct_free(sdp_GridderDirect* plan)
{
    if (!plan) return;
    sdp_mem_free(plan->pswf_sg);
    free(plan);
}


int sdp_gridder_direct_image_size(const sdp_GridderDirect* plan)
{
    return plan->image_size;
}


double sdp_gridder_direct_shear_u(const sdp_GridderDirect* plan)
{
    return plan->shear_u;
}


double sdp_gridder_direct_shear_v(const sdp_GridderDirect* plan)
{
    return plan->shear_v;
}


int sdp_gridder_direct_subgrid_size(const sdp_GridderDirect* plan)
{
    return plan->subgrid_size;
}


int sdp_gridder_direct_support(const sdp_GridderDirect* plan)
{
    return plan->support;
}


double sdp_gridder_direct_theta(const sdp_GridderDirect* plan)
{
    return plan->theta;
}


double sdp_gridder_direct_w_step(const sdp_GridderDirect* plan)
{
    return plan->w_step;
}
